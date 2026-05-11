"""
API Routes for Hybrid Implicit ALS + SBERT Recommender
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.api.schemas import *
from src.models.hybrid_recommender import HybridRecommender
from src.data.db_loader import DatabaseLoader
from src.utils.config import get_settings
from src.utils.logging_config import logger
from src.utils.callback import notify_retrain_complete_sync, notify_incremental_update_sync
from pathlib import Path
from typing import Optional

router = APIRouter()

# Global model instance (loaded at startup)
recommender: Optional[HybridRecommender] = None
is_retraining: bool = False  # Track retraining status

def get_recommender():
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return recommender

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if models are loaded"""
    try:
        rec = get_recommender()
        return HealthResponse(
            status="ok",
            models_loaded=True
        )
    except:
        return HealthResponse(
            status="error",
            models_loaded=False
        )

@router.get("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(user_id: str, limit: int = 10):
    """
    Get hybrid recommendations (ALS + SBERT)
    
    Combines:
    - Implicit ALS collaborative filtering
    - SBERT semantic content-based filtering
    - Popularity fallback for cold start
    """
    rec = get_recommender()
    
    # Handle 'undefined' from frontend
    if user_id == 'undefined' or not user_id:
        parsed_user_id = 0
    else:
        try:
            parsed_user_id = int(user_id)
        except ValueError:
            parsed_user_id = 0
            
    logger.info(f"Getting recommendations for user {parsed_user_id}, limit: {limit}")
    try:
        results = rec.recommend(parsed_user_id, limit=limit)
        
        items = [
            RecommendationItem(
                book_id=r['book_id'],
                score=r['score'],
                reasons=r['reasons']
            )
            for r in results
        ]
        
        return RecommendationsResponse(
            user_id=parsed_user_id if parsed_user_id != 0 else None,
            limit=limit,
            items=items
        )
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/similar", response_model=SimilarResponse)
async def get_similar(book_id: int, limit: int = 10):
    """
    Get semantically similar books using SBERT embeddings
    
    Uses cosine similarity between SBERT embeddings
    """
    rec = get_recommender()
    
    try:
        if not rec.content_model:
            raise HTTPException(status_code=503, detail="SBERT model not loaded")
        
        similar = rec.content_model.get_similar_items(book_id, top_k=limit)
        
        items = [
            SimilarItem(book_id=int(bid), score=float(score))
            for bid, score in similar
        ]
        
        return SimilarResponse(book_id=book_id, items=items)
    except Exception as e:
        logger.error(f"Error getting similar books: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/diversity", response_model=DiversityResponse)
async def get_diversity(book_id: int, limit: int = 5):
    """Get diverse book recommendations for a reference book"""
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100"
        )

    rec = get_recommender()

    if rec.diversity_model is None:
        raise HTTPException(status_code=503, detail="Diversity model not loaded")

    try:
        result = rec.diversity_recommendations(
            book_id=book_id,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to compute diversity recommendations: {exc}")
        raise HTTPException(status_code=500, detail="Failed to compute diversity recommendations")

    return DiversityResponse(**result)

@router.post("/feedback")
async def record_feedback(request: FeedbackRequest):
    """
   Record user feedback and trigger online learning
    
    Event types and their strengths:
    - rating: rating_value (1-5, explicit rating from user)
    - history: 1.0 (implicit signal: user read/opened the book)
    - favorite: 5.0 if rating_value is None or > 0, 0.0 if rating_value = 0 (remove favorite)
    
    Examples:
    - Add favorite: {"event": "favorite"} → strength = 5.0
    - Remove favorite: {"event": "favorite", "rating_value": 0} → strength = 0.0
    
    Note: Online learning only updates SBERT user profiles.
    ALS model requires full retrain.
    """
    rec = get_recommender()
    
    # Map event to strength
    if request.event == 'rating':
        if not request.rating_value or request.rating_value < 1:
            raise HTTPException(
                status_code=400, 
                detail="rating_value (1-5) is required for 'rating' event"
            )
        strength = float(request.rating_value)
    elif request.event == 'favorite':
        # Support unfavorite by passing rating_value=0
        if request.rating_value == 0:
            strength = 0.0  # Remove favorite
        else:
            strength = 5.0  # Add favorite
    elif request.event == 'history':
        progress = request.progress or 0.0
        strength = max(0.5, (progress / 100.0) * 5.0)  # Align with db_loader logic
    else:
        strength = 1.0
    
    logger.info(f"Feedback: user={request.user_id}, book={request.book_id}, "
               f"event={request.event}, strength={strength}")
    
    # Add to online learning buffer (if enabled)
    if rec.online_learning:
        # Get buffer before to track which users will be updated if buffer triggers
        buffer_before = [i['user_id'] for i in rec.interaction_buffer]
        
        buffer_triggered = rec.add_interaction(
            user_id=request.user_id,
            book_id=request.book_id,
            strength=strength,
            interaction_type=request.event
        )
        
        buffer_status = rec.get_buffer_status()
        
        # If buffer triggered update, notify backend for cache invalidation
        if buffer_triggered:
            # Include current user in the list of updated users
            updated_user_ids = list(set(buffer_before + [request.user_id]))
            logger.info(f"📤 Buffer triggered, notifying backend about {len(updated_user_ids)} updated users...")
            notify_incremental_update_sync(updated_user_ids)
        
        return {
            "status": "recorded",
            "online_learning": True,
            "buffer_triggered_update": buffer_triggered,
            "buffer_status": buffer_status,
            "note": "Only SBERT profiles updated. ALS requires full retrain."
        }
    else:
        # Fallback: just log (online learning disabled)
        return {
            "status": "recorded",
            "online_learning": False,
            "message": "Feedback logged for future batch retraining"
        }

@router.get("/model/info")
async def get_model_info():
    """Get information about current loaded model"""
    rec = get_recommender()
    
    info = {
        "status": "loaded",
        "model_type": "HybridRecommender",
        "alpha": rec.alpha,
        "online_learning": rec.get_buffer_status() if rec.online_learning else {"enabled": False},
        "cf_model": {
            "num_users": len(rec.als_model.user_id_map) if rec.als_model and rec.als_model.user_id_map else 0,
            "num_items": len(rec.als_model.item_id_map) if rec.als_model and rec.als_model.item_id_map else 0,
            "matrix_nnz": rec.als_model.user_item_matrix.nnz if rec.als_model and rec.als_model.user_item_matrix is not None else 0,
            "model_type": "Implicit ALS",
            "factors": rec.als_factors,
            "iterations": rec.als_iterations,
            "regularization": rec.als_regularization
        } if rec.als_model else None,
        "content_model": {
            "num_books": len(rec.content_model.book_ids) if rec.content_model and rec.content_model.book_ids is not None else 0,
            "num_user_profiles": len(rec.content_model.user_profiles) if rec.content_model and rec.content_model.user_profiles else 0,
            "embedding_dim": rec.content_model.embeddings.shape[1] if rec.content_model and rec.content_model.embeddings is not None else 0,
            "model_type": "SBERT",
            "model_name": rec.content_model.model_name if rec.content_model else None
        } if rec.content_model else None,
        "is_retraining": is_retraining
    }
    
    return info

@router.post("/online-learning/update")
async def trigger_incremental_update(force: bool = False):
    """
    Trigger incremental model update with buffered interactions
    
    Note: Only SBERT user profiles will be updated. ALS requires full retrain.
    
    Args:
        force: Force update even if buffer is not full
    """
    rec = get_recommender()
    
    if not rec.online_learning:
        raise HTTPException(
            status_code=400,
            detail="Online learning is disabled. Enable it first with POST /online-learning/enable"
        )
    
    buffer_status_before = rec.get_buffer_status()
    
    updated_user_ids = rec.incremental_update(force=force)
    
    buffer_status_after = rec.get_buffer_status()
    processed = max(
        0,
        int(buffer_status_before.get("buffer_size") or 0)
        - int(buffer_status_after.get("buffer_size") or 0)
    )
    status = "updated" if processed > 0 else "skipped"
    
    # Notify Java backend about updated users for cache invalidation
    if status == "updated" and updated_user_ids:
        logger.info(f"📤 Notifying backend about incremental update for {len(updated_user_ids)} users...")
        notify_incremental_update_sync(updated_user_ids)
    
    response = {
        "status": status,
        "before": buffer_status_before,
        "after": buffer_status_after,
        "interactions_processed": processed,
        "updated_user_ids": updated_user_ids if updated_user_ids else [],
        "note": "Only SBERT profiles updated. ALS requires full retrain."
    }
    
    if status == "skipped":
        response["message"] = "Buffer chưa đạt ngưỡng, chưa có cập nhật nào được thực hiện."
    
    return response

@router.post("/online-learning/enable")
async def enable_online_learning(buffer_size: int = 100):
    """Enable online learning (SBERT only)"""
    if buffer_size < 10 or buffer_size > 1000:
        raise HTTPException(
            status_code=400,
            detail="buffer_size must be between 10 and 1000"
        )
    
    rec = get_recommender()
    rec.enable_online_learning(buffer_size=buffer_size)
    
    return {
        "status": "enabled",
        "buffer_size": buffer_size,
        "note": "Only SBERT profiles will be updated incrementally. ALS requires full retrain."
    }

@router.post("/online-learning/disable")
async def disable_online_learning():
    """Disable online learning"""
    rec = get_recommender()
    rec.disable_online_learning()
    
    return {"status": "disabled"}

@router.get("/online-learning/status")
async def get_online_learning_status():
    """Get online learning buffer status"""
    rec = get_recommender()
    return rec.get_buffer_status()

@router.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Trigger background retraining
    
    Reloads data from database and retrains both ALS and SBERT models
    """
    global is_retraining
    
    if is_retraining:
        raise HTTPException(status_code=409, detail="Retraining already in progress")
    
    background_tasks.add_task(retrain_models)
    
    return {
        "status": "retraining",
        "message": "Models are being retrained in background. Check /model/info for status."
    }

@router.get("/user/profile/{user_id}")
async def get_user_profile(user_id: int, top_n: int = 20):
    """
    Get user's semantic profile from SBERT
    
    Returns top keywords/concepts from user's reading history
    """
    rec = get_recommender()
    
    if not rec.content_model:
        raise HTTPException(status_code=503, detail="SBERT model not loaded")
    
    if user_id not in rec.content_model.user_profiles:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in SBERT profiles")
    
    try:
        # Get user's embedding profile
        profile_embedding = rec.content_model.user_profiles[user_id]
        
        # Get user's interaction history
        user_books = rec.content_model.user_interactions.get(user_id, {})
        
        return {
            "user_id": user_id,
            "num_interactions": len(user_books),
            "profile_dimension": len(profile_embedding),
            "top_books": sorted(user_books.items(), key=lambda x: x[1], reverse=True)[:10],
            "message": "SBERT uses dense embeddings (no sparse keywords like TF-IDF)"
        }
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/interactions/{user_id}")
async def get_user_interactions(user_id: int):
    """Get user's interaction history"""
    rec = get_recommender()
    
    # Try to get from SBERT model first
    if rec.content_model and user_id in rec.content_model.user_interactions:
        interactions = rec.content_model.user_interactions.get(user_id, {})
    elif rec.als_model and user_id in rec.als_model.user_id_map:
        # Fallback to ALS model (no strength info, just presence)
        interactions = {}
        logger.info(f"User {user_id} found in ALS but interaction strengths not available")
    else:
        raise HTTPException(status_code=404, detail=f"No interactions found for user {user_id}")
    
    if not interactions:
        raise HTTPException(status_code=404, detail=f"No interactions found for user {user_id}")
    
    return {
        "user_id": user_id,
        "num_interactions": len(interactions),
        "books": [
            {"book_id": book_id, "strength": strength}
            for book_id, strength in sorted(interactions.items(), key=lambda x: x[1], reverse=True)
        ]
    }

async def retrain_models():
    """Background task to retrain models"""
    global is_retraining, recommender
    
    try:
        is_retraining = True
        logger.info("🔄 Starting background retraining...")
        
        # Load fresh data
        settings = get_settings()
        loader = DatabaseLoader(settings.db_uri, settings.db_schema)
        books_df, interactions_df = loader.load_all()
        
        # Instantiate new model for retraining to avoid race conditions
        new_recommender = HybridRecommender(
            alpha=recommender.alpha if recommender else settings.alpha,
            als_factors=recommender.als_factors if recommender else settings.cf_factors,
            als_iterations=recommender.als_iterations if recommender else settings.cf_iterations,
            als_regularization=recommender.als_regularization if recommender else settings.cf_regularization,
            sbert_model=recommender.sbert_model_name if recommender else 'keepitreal/vietnamese-sbert'
        )
        
        # Retrain new instance
        new_recommender.train(books_df, interactions_df)
        
        # Save updated models
        artifacts_dir = Path("./artifacts")
        new_recommender.save(artifacts_dir)
        
        # Atomic swap
        recommender = new_recommender
        
        logger.info("✅ Background retraining completed!")
        
        # Notify Java backend to invalidate cache
        logger.info("📤 Notifying backend about retrain completion...")
        notify_retrain_complete_sync(model_key="implicit")
        
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
    finally:
        is_retraining = False
