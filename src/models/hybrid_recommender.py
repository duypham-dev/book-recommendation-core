"""
Hybrid Recommender: Implicit ALS + SBERT

Combines:
1. Implicit ALS - Fast collaborative filtering for implicit feedback
2. SBERT - Semantic content-based filtering with Vietnamese embeddings
3. Weighted fusion + popularity fallback
"""
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from src.models.collaborative import CollaborativeModel
from src.features.sbert_features import SBERTContentModel
from src.models.diversity import DiversityRecommender
from src.utils.logging_config import logger


class HybridRecommender:
    """
    Hybrid Recommendation: Implicit ALS + SBERT
    
    Architecture:
    1. Implicit ALS: Fast collaborative filtering
    2. SBERT: Semantic content-based filtering
    3. Fusion: Weighted average
    4. Fallback: Popularity-based for cold start
    """
    
    def __init__(self, alpha: float = 0.4,
                 # ALS params
                 als_factors: int = 64, als_iterations: int = 30, 
                 als_regularization: float = 0.01,
                 # SBERT params
                 sbert_model: str = 'keepitreal/vietnamese-sbert',
                 device: str = None):
        """
        Args:
            alpha: Weight for ALS (0-1), SBERT weight = 1 - alpha
            als_factors: ALS latent factors
            als_iterations: ALS iterations
            als_regularization: ALS regularization parameter
            sbert_model: SBERT model name
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.alpha = alpha
        
        # ALS parameters
        self.als_factors = als_factors
        self.als_iterations = als_iterations
        self.als_regularization = als_regularization
        
        # SBERT parameters
        self.sbert_model_name = sbert_model
        self.device = device
        
        # Models
        self.als_model: Optional[CollaborativeModel] = None
        self.content_model: Optional[SBERTContentModel] = None
        self.diversity_model: Optional[DiversityRecommender] = None
        
        # Popularity fallback
        self.popularity: Optional[Dict[int, int]] = None
        
        # Diversity support
        self.books_df: Optional[pd.DataFrame] = None
        self._diversity_rating_map: Dict[int, float] = {}
        
        # Online learning support (incremental updates for SBERT only)
        self.online_learning = False
        self.interaction_buffer = []
        self.buffer_size = 100
    
    def train(self, books_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """
        Train both ALS and SBERT models
        
        Args:
            books_df: DataFrame with book metadata
            interactions_df: DataFrame with [user_id, book_id, strength]
        """
        logger.info("="*70)
        logger.info("Training Hybrid Implicit ALS + SBERT Recommender")
        logger.info("="*70)
        
        # Prepare data for diversity model
        self.books_df = self._prepare_books_for_diversity(books_df)
        self._diversity_rating_map = self._compute_diversity_rating_map(interactions_df)
        
        # 1. Train Implicit ALS
        logger.info("\n1️⃣ Training Implicit ALS model...")
        self.als_model = CollaborativeModel(
            factors=self.als_factors,
            iterations=self.als_iterations,
            regularization=self.als_regularization
        )
        self.als_model.fit(interactions_df)
        logger.info("✅ ALS model trained")
        
        # 2. Train SBERT content model
        logger.info("\n2️⃣ Training SBERT content model...")
        self.content_model = SBERTContentModel(
            model_name=self.sbert_model_name,
            device=self.device
        )
        self.content_model.fit(books_df)
        self.content_model.build_user_profiles(interactions_df)
        logger.info("✅ SBERT model trained")
        
        # 3. Compute popularity
        logger.info("\n3️⃣ Computing popularity scores...")
        self._compute_popularity(interactions_df)
        logger.info("✅ Popularity scores computed")
        
        # 4. Build diversity model
        logger.info("\n4️⃣ Building diversity model...")
        self._build_diversity_model()
        logger.info("✅ Diversity model built")
        
        logger.info("\n" + "="*70)
        logger.info("✅ Hybrid Implicit ALS + SBERT training completed!")
        logger.info("="*70)
    
    def _compute_popularity(self, interactions_df: pd.DataFrame):
        """Compute item popularity from interactions"""
        pop_counts = interactions_df['book_id'].value_counts().to_dict()
        self.popularity = dict(sorted(pop_counts.items(), 
                                     key=lambda x: x[1], reverse=True))
    
    def _prepare_books_for_diversity(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant columns for diversity recommendations"""
        if books_df is None or books_df.empty:
            return pd.DataFrame()
        
        baseline_columns = ["book_id"]
        candidate_columns = [
            "genres_text",
            "tags",
            "tag_name",
            "categories",
            "category",
            "authors",
            "title",
            "description",
            "summary",
        ]
        
        columns = baseline_columns + [col for col in candidate_columns if col in books_df.columns]
        return books_df[columns].copy()
    
    def _compute_diversity_rating_map(self, interactions_df: pd.DataFrame) -> Dict[int, float]:
        """Compute average rating per book for diversity ranking"""
        if interactions_df is None or interactions_df.empty:
            return {}
        
        rating_column = None
        for candidate in ("rating_value", "rating", "strength"):
            if candidate in interactions_df.columns:
                rating_column = candidate
                break
        
        if rating_column is None:
            return {}
        
        ratings_df = interactions_df.dropna(subset=["book_id", rating_column])
        grouped = ratings_df.groupby("book_id")[rating_column].mean()
        return {int(book_id): float(value) for book_id, value in grouped.items()}
    
    def recommend(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        Get hybrid recommendations
        
        Args:
            user_id: User ID
            limit: Number of recommendations
        
        Returns:
            List of recommendation dicts with scores and sources
        """
        # Get user's interaction history for filtering
        interacted_items = set()
        if self.content_model and user_id in self.content_model.user_interactions:
            interacted_items = set(self.content_model.user_interactions[user_id].keys())
        
        # 1. Get ALS recommendations
        als_results = []
        if self.als_model and user_id in self.als_model.user_id_map:
            als_results = self.als_model.recommend(
                user_id, 
                top_k=limit * 3,  # Get more for fusion
                filter_items=interacted_items
            )
        
        # 2. Get SBERT recommendations
        content_results = []
        if self.content_model and user_id in self.content_model.user_profiles:
            content_results = self.content_model.recommend_for_user(
                user_id,
                top_k=limit * 3,
                filter_items=interacted_items
            )
        
        # 3. Combine scores
        if als_results and content_results:
            combined = self._combine_scores(als_results, content_results)
            results = [
                {
                    'book_id': int(book_id),
                    'score': float(score),
                    'reasons': reasons
                }
                for book_id, score, reasons in combined[:limit]
            ]
        elif als_results:
            # ALS only
            results = [
                {
                    'book_id': int(book_id),
                    'score': float(score),
                    'reasons': {'als': float(score), 'sbert': 0.0, 'pop': 0.0}
                }
                for book_id, score in als_results[:limit]
            ]
        elif content_results:
            # SBERT only
            results = [
                {
                    'book_id': int(book_id),
                    'score': float(score),
                    'reasons': {'als': 0.0, 'sbert': float(score), 'pop': 0.0}
                }
                for book_id, score in content_results[:limit]
            ]
        else:
            # Fallback to popularity
            logger.debug(f"User {user_id} - using popularity fallback")
            results = self._get_popularity_recommendations(limit, interacted_items)
        
        return results
    
    def _combine_scores(self, als_results: List[Tuple[int, float]],
                       content_results: List[Tuple[int, float]]) -> List[Tuple[int, float, dict]]:
        """
        Combine ALS and SBERT scores with weighted average
        
        final_score = alpha * ALS_score + (1 - alpha) * SBERT_score
        
        Args:
            als_results: List of (book_id, als_score)
            content_results: List of (book_id, sbert_score)
        
        Returns:
            Combined list of (book_id, final_score, reasons_dict)
        """
        # Normalize scores to [0, 1]
        als_dict = self._normalize_scores(als_results)
        content_dict = self._normalize_scores(content_results)
        
        # Combine
        all_books = set(als_dict.keys()) | set(content_dict.keys())
        
        combined = []
        for book_id in all_books:
            als_score = als_dict.get(book_id, 0.0)
            sbert_score = content_dict.get(book_id, 0.0)
            
            # Weighted average
            final_score = self.alpha * als_score + (1 - self.alpha) * sbert_score
            
            # Build reasons breakdown
            reasons = {
                'als': float(als_score),
                'sbert': float(sbert_score),
                'pop': 0.0
            }
            
            combined.append((book_id, final_score, reasons))
        
        # Sort by score
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined
    
    def _normalize_scores(self, results: List[Tuple[int, float]]) -> Dict[int, float]:
        """Normalize scores to [0, 1] using min-max scaling"""
        if not results:
            return {}
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return {book_id: 1.0 for book_id, _ in results}
        
        normalized = {}
        for book_id, score in results:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized[book_id] = norm_score
        
        return normalized
    
    def _get_popularity_recommendations(self, limit: int, exclude: set) -> List[Dict]:
        """Fallback to popularity-based recommendations"""
        if self.popularity is None:
            return []
        
        results = []
        for book_id, count in self.popularity.items():
            if book_id not in exclude:
                results.append({
                    'book_id': int(book_id),
                    'score': float(count),
                    'reasons': {'als': 0.0, 'sbert': 0.0, 'pop': float(count)}
                })
                if len(results) >= limit:
                    break
        
        return results
    
    def similar_books(self, book_id: int, limit: int = 10) -> List[Dict]:
        """Get similar books using SBERT embeddings"""
        if not self.content_model:
            return []
        
        similar = self.content_model.get_similar_items(book_id, top_k=limit)
        
        return [
            {
                'book_id': bid,
                'score': score,
                'source': 'sbert_similarity'
            }
            for bid, score in similar
        ]
    
    def get_user_profile_keywords(self, user_id: int, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get user's profile keywords (top interacted books for SBERT)"""
        if not self.content_model:
            return []
        return self.content_model.get_profile_keywords(user_id, top_n)
    
    def diversity_recommendations(
        self,
        book_id: int,
        limit: int = 5,
    ) -> Dict[str, List[Dict]]:
        """
        Get diverse recommendations for a book
        
        Args:
            book_id: Reference book ID
            limit: Number of recommendations per category
            
        Returns:
            Dict with diversity categories and recommendations
        """
        if self.diversity_model is None:
            raise RuntimeError("Diversity model not initialized")
        
        results = self.diversity_model.recommend(
            book_id=book_id,
            limit=limit,
        )
        
        return {
            'book_id': int(book_id),
            'items': [
                {
                    'book_id': int(item.book_id),
                    'rating': float(item.rating),
                    'score': float(item.score),
                    'metadata': {k: float(v) for k, v in item.metadata.items()} if item.metadata else {},
                }
                for item in results
            ],
        }
    
    def _build_diversity_model(self):
        """Build diversity recommender using SBERT embeddings"""
        if self.books_df is None or self.books_df.empty:
            logger.warning("Cannot build diversity model: books_df not available")
            self.diversity_model = None
            return
        
        # Prepare SBERT embeddings for diversity model
        embeddings = None
        if (
            self.content_model
            and getattr(self.content_model, "embeddings", None) is not None
            and getattr(self.content_model, "book_ids", None) is not None
        ):
            try:
                id_to_idx = {int(bid): idx for idx, bid in enumerate(self.content_model.book_ids)}
                vectors = []
                missing = []
                for bid in self.books_df["book_id"]:
                    idx = id_to_idx.get(int(bid))
                    if idx is None:
                        missing.append(int(bid))
                        continue
                    vectors.append(self.content_model.embeddings[idx])

                if missing:
                    logger.warning(
                        "Diversity embeddings missing for %d books; falling back to TF-IDF for those entries.",
                        len(missing),
                    )

                if vectors and len(vectors) == len(self.books_df):
                    embeddings = np.vstack(vectors).astype(np.float32)
                else:
                    embeddings = None
            except Exception as exc:
                logger.warning(f"Failed to prepare SBERT embeddings for diversity: {exc}")
        
        # Determine tag columns for diversity
        tag_candidates = [
            column
            for column in (
                "genres_text",
                "tags",
                "tag_name",
                "categories",
                "category",
                "authors",
                "title",
                "description",
                "summary",
            )
            if column in self.books_df.columns
        ]
        
        try:
            self.diversity_model = DiversityRecommender(
                books_df=self.books_df,
                interactions_df=None,
                tag_columns=tag_candidates or None,
                embeddings=embeddings,
            )
            
            # Set rating map for diversity ranking
            if self._diversity_rating_map:
                self.diversity_model.rating_map = {
                    int(book_id): float(rating)
                    for book_id, rating in self._diversity_rating_map.items()
                }
                ratings = list(self.diversity_model.rating_map.values())
                self.diversity_model.global_rating = (
                    float(np.nanmean(ratings)) if ratings else 0.0
                )
            
            logger.info(f"Built diversity model with {len(self.books_df)} books")
        except Exception as e:
            logger.error(f"Failed to build diversity model: {e}")
            self.diversity_model = None
    
    # ==================== Online Learning Methods ====================
    
    def enable_online_learning(self, buffer_size: int = 100):
        """Enable online learning for SBERT model only"""
        self.online_learning = True
        self.buffer_size = buffer_size
        self.interaction_buffer = []
        logger.info(f"Online learning enabled (buffer_size={buffer_size})")
        logger.info("Note: Only SBERT profiles will be updated incrementally. ALS requires full retrain.")
    
    def disable_online_learning(self):
        """Disable online learning"""
        self.online_learning = False
        self.interaction_buffer = []
        logger.info("Online learning disabled")
    
    def add_interaction(self, user_id: int, book_id: int, strength: float = 1.0, 
                       interaction_type: str = 'view') -> bool:
        """
        Add new interaction to buffer for online learning
        
        Args:
            user_id: User ID
            book_id: Book ID
            strength: Interaction strength
            interaction_type: Type of interaction
            
        Returns:
            True if buffer triggered update, False otherwise
        """
        if not self.online_learning:
            logger.warning("Online learning is disabled. Enable it first.")
            return False
        
        # Add to buffer
        self.interaction_buffer.append({
            'user_id': user_id,
            'book_id': book_id,
            'strength': strength,
            'type': interaction_type
        })
        
        logger.debug(f"Added interaction to buffer: user={user_id}, book={book_id}, "
                    f"strength={strength}, buffer_size={len(self.interaction_buffer)}/{self.buffer_size}")
        
        # Trigger update if buffer is full
        if len(self.interaction_buffer) >= self.buffer_size:
            logger.info(f"Buffer full ({len(self.interaction_buffer)}/{self.buffer_size}), triggering incremental update...")
            self.incremental_update(force=True)
            return True
        
        return False
    
    def incremental_update(self, force: bool = False) -> list:
        """
        Incrementally update SBERT user profiles with buffered interactions
        
        Note: ALS model is NOT updated (requires full retrain)
        
        Args:
            force: Force update even if buffer is not full
            
        Returns:
            List of user IDs that were updated
        """
        if not self.online_learning:
            logger.warning("Online learning is disabled")
            return []
        
        if len(self.interaction_buffer) == 0:
            logger.info("Buffer is empty, nothing to update")
            return []
        
        if not force and len(self.interaction_buffer) < self.buffer_size:
            logger.info(f"Buffer not full ({len(self.interaction_buffer)}/{self.buffer_size}), "
                       f"skipping update (use force=True to override)")
            return []
        
        logger.info(f"Starting incremental update with {len(self.interaction_buffer)} interactions...")
        
        # Collect unique user IDs that will be updated
        updated_user_ids = list(set(interaction['user_id'] for interaction in self.interaction_buffer))
        
        # Update SBERT user profiles only
        if self.content_model:
            for interaction in self.interaction_buffer:
                self.content_model.update_user_profile(
                    user_id=interaction['user_id'],
                    book_id=interaction['book_id'],
                    strength=interaction['strength'],
                    interaction_type=interaction['type']
                )
            logger.info(f"✅ Updated {len(self.interaction_buffer)} SBERT user profiles")
        
        # Update popularity
        for interaction in self.interaction_buffer:
            book_id = interaction['book_id']
            if self.popularity and book_id in self.popularity:
                self.popularity[book_id] += 1
            elif self.popularity:
                self.popularity[book_id] = 1
        
        # Clear buffer
        buffer_count = len(self.interaction_buffer)
        self.interaction_buffer = []
        
        logger.info(f"✅ Incremental update completed, cleared {buffer_count} interactions from buffer")
        logger.info(f"📤 Updated user IDs: {updated_user_ids}")
        logger.info("⚠️  Note: ALS model NOT updated (requires full retrain)")
        
        return updated_user_ids
    
    def get_buffer_status(self) -> Dict:
        """Get online learning buffer status"""
        return {
            "enabled": self.online_learning,
            "buffer_size": len(self.interaction_buffer),
            "buffer_capacity": self.buffer_size,
            "buffer_full": len(self.interaction_buffer) >= self.buffer_size if self.online_learning else False,
            "note": "Only SBERT profiles updated incrementally. ALS requires full retrain."
        }
    
    def save(self, artifacts_dir: Path):
        """Save all models"""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ALS model
        if self.als_model:
            self.als_model.save(artifacts_dir / 'als_model.pkl')
        
        # Save SBERT model
        if self.content_model:
            self.content_model.save(artifacts_dir / 'sbert_model.pkl')
        
        # Save metadata
        with open(artifacts_dir / 'hybrid_recommender_metadata.pkl', 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'als_factors': self.als_factors,
                'als_iterations': self.als_iterations,
                'als_regularization': self.als_regularization,
                'sbert_model_name': self.sbert_model_name,
                'popularity': self.popularity,
                'online_learning': self.online_learning,
                'buffer_size': self.buffer_size,
                'diversity_rating_map': self._diversity_rating_map
            }, f)
        
        # Save books_df for diversity model
        if self.books_df is not None:
            self.books_df.to_pickle(artifacts_dir / 'books.pkl')
        
        logger.info(f"Saved Hybrid Implicit ALS + SBERT model to {artifacts_dir}")
    
    @classmethod
    def load(cls, artifacts_dir: Path, alpha: float = None, device: str = None):
        """Load saved models"""
        metadata = {}
        # Load metadata
        metadata_path = artifacts_dir / 'hybrid_recommender_metadata.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            alpha = alpha or metadata.get('alpha', 0.6)
            als_factors = metadata.get('als_factors', 64)
            als_iterations = metadata.get('als_iterations', 30)
            als_regularization = metadata.get('als_regularization', 0.01)
            sbert_model_name = metadata.get('sbert_model_name', 'keepitreal/vietnamese-sbert')
            popularity = metadata.get('popularity')
        else:
            alpha = alpha or 0.6
            als_factors = 64
            als_iterations = 30
            als_regularization = 0.01
            sbert_model_name = 'keepitreal/vietnamese-sbert'
            popularity = None
        
        model = cls(
            alpha=alpha,
            als_factors=als_factors,
            als_iterations=als_iterations,
            als_regularization=als_regularization,
            sbert_model=sbert_model_name,
            device=device
        )
        
        # Load ALS model
        als_path = artifacts_dir / 'als_model.pkl'
        if als_path.exists():
            model.als_model = CollaborativeModel.load(als_path)
        
        # Load SBERT model
        sbert_path = artifacts_dir / 'sbert_model.pkl'
        if sbert_path.exists():
            model.content_model = SBERTContentModel.load(sbert_path, device=device)
        
        model.popularity = popularity
        model._diversity_rating_map = metadata.get('diversity_rating_map', {})
        
        # Load books_df
        books_path = artifacts_dir / 'books.pkl'
        if books_path.exists():
            model.books_df = pd.read_pickle(books_path)
        else:
            logger.warning("Books metadata not found in artifacts; diversity recommendations disabled until retrain.")
        
        # Build diversity model
        model._build_diversity_model()
        
        logger.info(f"Loaded Hybrid Implicit ALS + SBERT model from {artifacts_dir}")
        return model
