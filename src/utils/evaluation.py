"""
Evaluation utilities for recommendation systems
Shared by both train.py and train_neural.py
"""
from typing import Dict, Set, List, Any
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from src.utils.logging_config import logger


class RecommenderEvaluator:
    """Unified evaluator for all recommender types"""
    
    @staticmethod
    def compute_metrics(recommender: Any, 
                       test_df: pd.DataFrame, 
                       train_df: pd.DataFrame = None,
                       k: int = 10) -> Dict[str, float]:
        """
        Compute HR@K and NDCG@K on test set
        
        Args:
            recommender: Trained recommender (HybridRecommender or HybridNeuralRecommender)
            test_df: Test interactions (holdout set, NOT in training)
            train_df: Training interactions (optional, for filtering)
            k: Number of recommendations
            
        Returns:
            Dict with HR@K, NDCG@K, users_evaluated, users_skipped
        """
        logger.info(f"Computing metrics @{k}...")
        
        # Group test data by user (ground truth)
        test_grouped = test_df.groupby('user_id')['book_id'].apply(set).to_dict()
        
        hits = []
        ndcg_scores = []
        skipped = 0
        
        for user_id, true_items in test_grouped.items():
            if len(true_items) == 0:
                continue
            
            # Skip cold-start users (not in training)
            if not RecommenderEvaluator._user_in_model(recommender, user_id):
                skipped += 1
                logger.debug(f"Skipped user {user_id}: cold-start (not in training)")
                continue
            
            # Get recommendations
            try:
                recs = recommender.recommend(user_id, limit=k)
                
                if not recs:
                    skipped += 1
                    logger.debug(f"Skipped user {user_id}: no recommendations")
                    continue
                
                pred_items = [r['book_id'] for r in recs]
                
            except Exception as e:
                logger.warning(f"Error getting recommendations for user {user_id}: {e}")
                skipped += 1
                continue
            
            # Compute Hit Rate
            matched = set(pred_items) & true_items
            hit = len(matched) > 0
            hits.append(1 if hit else 0)
            
            # Debug logging
            logger.debug(f"User {user_id}: predicted {len(pred_items)}, "
                        f"ground truth {len(true_items)}, matched {len(matched)}")
            if matched:
                logger.debug(f"  Matched items: {matched}")
            
            # Compute NDCG
            relevance = [1 if item in true_items else 0 for item in pred_items]
            ideal_relevance = sorted(relevance, reverse=True)
            
            if sum(relevance) > 0:
                ndcg = ndcg_score([ideal_relevance], [relevance])
                ndcg_scores.append(ndcg)
            else:
                ndcg_scores.append(0.0)
        
        # Aggregate metrics
        hr = np.mean(hits) if hits else 0.0
        ndcg_avg = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        logger.info(f"Evaluated {len(hits)} users (skipped {skipped} cold-start users)")
        
        return {
            'HR@K': hr,
            'NDCG@K': ndcg_avg,
            'users_evaluated': len(hits),
            'users_skipped': skipped
        }
    
    @staticmethod
    def _user_in_model(recommender: Any, user_id: int) -> bool:
        """Check if user exists in trained model"""
        # For HybridRecommender: check als_model (primary CF component)
        if hasattr(recommender, 'als_model') and recommender.als_model is not None:
            if hasattr(recommender.als_model, 'user_id_map'):
                if user_id in recommender.als_model.user_id_map:
                    return True
        
        # For HybridNeuralRecommender (NCF) — future compatibility
        if hasattr(recommender, 'ncf_model') and recommender.ncf_model is not None:
            if hasattr(recommender.ncf_model, 'user_id_map'):
                if user_id in recommender.ncf_model.user_id_map:
                    return True
        
        # For content-based fallback: check SBERT user profiles
        if hasattr(recommender, 'content_model') and recommender.content_model is not None:
            if hasattr(recommender.content_model, 'user_profiles'):
                if user_id in recommender.content_model.user_profiles:
                    return True
        
        return False
    
    @staticmethod
    def compute_coverage(recommender: Any, 
                        interactions_df: pd.DataFrame, 
                        num_samples: int = 100) -> float:
        """
        Compute catalog coverage (diversity metric)
        
        Args:
            recommender: Trained recommender
            interactions_df: All interactions (to get catalog)
            num_samples: Number of users to sample
            
        Returns:
            Coverage ratio (0-1)
        """
        all_items = set(interactions_df['book_id'].unique())
        recommended_items = set()
        
        user_ids = interactions_df['user_id'].unique()[:num_samples]
        
        for user_id in user_ids:
            try:
                recs = recommender.recommend(user_id, limit=10)
                recommended_items.update([r['book_id'] for r in recs])
            except:
                continue
        
        coverage = len(recommended_items) / len(all_items) if all_items else 0.0
        return coverage
    
    @staticmethod
    def print_sample_recommendations(recommender: Any, 
                                    books_df: pd.DataFrame,
                                    user_id: int,
                                    limit: int = 5):
        """Print sample recommendations with details"""
        logger.info(f"\nTop {limit} recommendations for user {user_id}:")
        
        try:
            recs = recommender.recommend(user_id, limit=limit)
            
            for i, rec in enumerate(recs, 1):
                book_id = rec['book_id']
                score = rec['score']
                reasons = rec.get('reasons', {})
                
                # Get book info
                book_info = books_df[books_df['book_id'] == book_id]
                title = book_info['title'].values[0] if len(book_info) > 0 else "Unknown"
                
                # Print safely
                logger.info(f"  {i}. Book ID={book_id} (score={score:.4f})")
                
                # Print reasons if available
                if reasons:
                    reason_str = ", ".join([f"{k}={v:.3f}" for k, v in reasons.items()])
                    logger.info(f"      {reason_str}")
                
                # Print title (handle Unicode safely)
                try:
                    logger.info(f"      Title: {title[:50]}")
                except:
                    logger.info(f"      Title: <encoding error>")
                    
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")


class DataSplitter:
    """Utility for train/test splitting"""
    
    @staticmethod
    def temporal_split(interactions_df: pd.DataFrame, 
                      test_ratio: float = 0.2,
                      timestamp_col: str = 'ts') -> tuple:
        """
        Split data by timestamp (temporal split)
        
        Args:
            interactions_df: Interaction data
            test_ratio: Ratio for test set (0-1)
            timestamp_col: Name of timestamp column
            
        Returns:
            (train_df, test_df)
        """
        logger.info(f"Temporal split: {(1-test_ratio)*100:.0f}% train, {test_ratio*100:.0f}% test")
        
        # Sort by timestamp if available
        if timestamp_col in interactions_df.columns:
            interactions_sorted = interactions_df.sort_values(timestamp_col)
        else:
            logger.warning(f"Column '{timestamp_col}' not found, using original order")
            interactions_sorted = interactions_df
        
        split_idx = int(len(interactions_sorted) * (1 - test_ratio))
        train_df = interactions_sorted.iloc[:split_idx].copy()
        test_df = interactions_sorted.iloc[split_idx:].copy()
        
        logger.info(f"Train: {len(train_df)} interactions, Test: {len(test_df)} interactions")
        logger.info(f"Train users: {train_df['user_id'].nunique()}, Test users: {test_df['user_id'].nunique()}")
        
        return train_df, test_df
    
    @staticmethod
    def random_split(interactions_df: pd.DataFrame,
                    test_ratio: float = 0.2,
                    random_state: int = 42) -> tuple:
        """
        Random split (shuffle before splitting)
        
        Args:
            interactions_df: Interaction data
            test_ratio: Ratio for test set
            random_state: Random seed
            
        Returns:
            (train_df, test_df)
        """
        logger.info(f"Random split: {(1-test_ratio)*100:.0f}% train, {test_ratio*100:.0f}% test")
        
        interactions_shuffled = interactions_df.sample(frac=1, random_state=random_state)
        
        split_idx = int(len(interactions_shuffled) * (1 - test_ratio))
        train_df = interactions_shuffled.iloc[:split_idx].copy()
        test_df = interactions_shuffled.iloc[split_idx:].copy()
        
        logger.info(f"Train: {len(train_df)} interactions, Test: {len(test_df)} interactions")
        
        return train_df, test_df
