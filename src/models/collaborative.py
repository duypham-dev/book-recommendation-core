import implicit
import scipy.sparse as sp
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from src.utils.logging_config import logger

class CollaborativeModel:
    def __init__(self, factors: int = 64, iterations: int = 30, regularization: float = 0.01):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=42
        )
        self.user_item_matrix = None
        self.item_user_matrix = None  # Needed for implicit.recommend()
        self.user_ids = None
        self.item_ids = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.item_id_reverse = {}
    
    def fit(self, interactions_df: pd.DataFrame):
        """Build user-item matrix and train ALS"""
        logger.info("Training collaborative filtering model...")
        
        # Aggregate interactions (sum strengths per user-book)
        agg = interactions_df.groupby(['user_id', 'book_id'])['strength'].sum().reset_index()
        
        # Cap maximum strength to avoid dominance
        MAX_STRENGTH = 10.0
        agg['strength'] = agg['strength'].clip(upper=MAX_STRENGTH)
        
        # Map IDs to matrix indices
        self.user_ids = sorted(agg['user_id'].unique())
        self.item_ids = sorted(agg['book_id'].unique())
        
        logger.info(f"Training CF with {len(self.user_ids)} users and {len(self.item_ids)} items")
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.item_id_reverse = {idx: iid for iid, idx in self.item_id_map.items()}

        # Build sparse matrix (users × items)
        rows = agg['user_id'].map(self.user_id_map).values
        cols = agg['book_id'].map(self.item_id_map).values
        data = agg['strength'].values
        
        self.user_item_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        )

        # Create item-user matrix for training
        self.item_user_matrix = self.user_item_matrix.T.tocsr()

        # Train with item-user matrix (items × users)
        self.model.fit(self.user_item_matrix)
        
        logger.info(f"CF matrix: {self.user_item_matrix.shape}, nnz={self.user_item_matrix.nnz}")
    
    def recommend(self, user_id: int, top_k: int = 10, filter_items: set = None) -> List[Tuple[int, float]]:
        """Get recommendations for a user"""
        if user_id not in self.user_id_map:
            logger.debug(f"User {user_id} not in training data")
            return []
        
        user_idx = self.user_id_map[user_id]
        
        # Calculate safe N value
        user_items_extracted = self.user_item_matrix[user_idx].nonzero()[1]
        num_interacted = len(user_items_extracted)
        num_available = len(self.item_ids) - num_interacted
    
        # Can't recommend more than available items
        safe_n = min(top_k, num_available)
        
        if safe_n <= 0:
            logger.debug(f"No items available to recommend for user {user_id}")
            return []
        
        try:
            # implicit.recommend() expects user_items in USERS×ITEMS format (CSR)
            # Pass the FULL user_item_matrix, not item_user_matrix!
            ids, scores = self.model.recommend(
                userid=user_idx,
                user_items=self.user_item_matrix[user_idx],  # Users × Items matrix (14×175)
                N=safe_n,
                filter_already_liked_items=True
            )
        except (IndexError, ValueError) as e:
            logger.warning(f"Error in recommend for user {user_id}: {e}")
            return []
        
        # Map back to book IDs
        results = []
        for idx, score in zip(ids, scores):
            if idx >= len(self.item_id_reverse):
                logger.warning(f"Item index {idx} out of bounds")
                continue
                
            book_id = self.item_id_reverse[idx]
            if filter_items and book_id in filter_items:
                continue
            results.append((book_id, float(score)))
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, path: Path):
        """Save model"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'user_item_matrix': self.user_item_matrix,
                'item_user_matrix': self.item_user_matrix,
                'user_ids': self.user_ids,
                'item_ids': self.item_ids,
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'item_id_reverse': self.item_id_reverse
            }, f)
        logger.info(f"Saved CF model to {path}")
    
    @classmethod
    def load(cls, path: Path):
        """Load saved model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.model = data['model']
        model.user_item_matrix = data['user_item_matrix']
        model.item_user_matrix = data.get('item_user_matrix', data['user_item_matrix'].T.tocsr())
        model.user_ids = data['user_ids']
        model.item_ids = data['item_ids']
        model.user_id_map = data['user_id_map']
        model.item_id_map = data['item_id_map']
        model.item_id_reverse = data['item_id_reverse']
        return model

# import implicit
# import scipy.sparse as sp
# import numpy as np
# import pandas as pd
# import pickle
# from pathlib import Path
# from typing import Dict, List, Tuple
# from src.utils.logging_config import logger

# class CollaborativeModel:
#     def __init__(self, factors: int = 64, iterations: int = 30, regularization: float = 0.01):
#         self.model = implicit.als.AlternatingLeastSquares(
#             factors=factors,
#             iterations=iterations,
#             regularization=regularization,
#             random_state=42
#         )
#         self.user_item_matrix = None
#         self.user_ids = None
#         self.item_ids = None
#         self.user_id_map = {}
#         self.item_id_map = {}
#         self.item_id_reverse = {}
#     def fit(self, interactions_df: pd.DataFrame):
#         """Build user-item matrix and train ALS"""
#         logger.info("Training collaborative filtering model...")
        
#         # DEBUG: Print raw data
#         logger.info(f"Raw interactions shape: {interactions_df.shape}")
#         logger.info(f"Raw unique users: {sorted(interactions_df['user_id'].unique())}")
#         logger.info(f"Raw unique books: {sorted(interactions_df['book_id'].unique())}")
        
#         # Aggregate interactions (sum strengths per user-book)
#         agg = interactions_df.groupby(['user_id', 'book_id'])['strength'].sum().reset_index()
        
#         # DEBUG: Print after aggregation
#         logger.info(f"After aggregation: {agg.shape}")
#         logger.info(f"Aggregated unique users: {sorted(agg['user_id'].unique())}")
        
#         # Map IDs to matrix indices
#         self.user_ids = sorted(agg['user_id'].unique())
#         self.item_ids = sorted(agg['book_id'].unique())
        
#         logger.info(f"Final user_ids for training: {self.user_ids}")
#         logger.info(f"Final item_ids count: {len(self.item_ids)}")
        
#         self.user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
#         self.item_id_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
#         self.item_id_reverse = {idx: iid for iid, idx in self.item_id_map.items()}
        
#         # Build sparse matrix
#         rows = agg['user_id'].map(self.user_id_map).values
#         cols = agg['book_id'].map(self.item_id_map).values
#         data = agg['strength'].values
        
#         self.user_item_matrix = sp.csr_matrix(
#             (data, (rows, cols)),
#             shape=(len(self.user_ids), len(self.item_ids))
#         )
        
#         # Train (implicit expects item-user format)
#         self.model.fit(self.user_item_matrix.T.tocsr())
        
#         logger.info(f"CF matrix: {self.user_item_matrix.shape}, nnz={self.user_item_matrix.nnz}")
#     # def fit(self, interactions_df: pd.DataFrame):
#     #     """Build user-item matrix and train ALS"""
#     #     logger.info("Training collaborative filtering model...")
        
#     #     # Aggregate interactions (sum strengths per user-book)
#     #     agg = interactions_df.groupby(['user_id', 'book_id'])['strength'].sum().reset_index()
        
#     #     # Map IDs to matrix indices
#     #     self.user_ids = sorted(agg['user_id'].unique())
#     #     self.item_ids = sorted(agg['book_id'].unique())
#     #     self.user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
#     #     self.item_id_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
#     #     self.item_id_reverse = {idx: iid for iid, idx in self.item_id_map.items()}
        
#     #     # Build sparse matrix
#     #     rows = agg['user_id'].map(self.user_id_map).values
#     #     cols = agg['book_id'].map(self.item_id_map).values
#     #     data = agg['strength'].values
        
#     #     self.user_item_matrix = sp.csr_matrix(
#     #         (data, (rows, cols)),
#     #         shape=(len(self.user_ids), len(self.item_ids))
#     #     )
        
#     #     # Train (implicit expects item-user format)
#     #     self.model.fit(self.user_item_matrix.T.tocsr())
        
#     #     logger.info(f"CF matrix: {self.user_item_matrix.shape}, nnz={self.user_item_matrix.nnz}")
    
#     def recommend(self, user_id: int, top_k: int = 10, filter_items: set = None) -> List[Tuple[int, float]]:
#         """Get recommendations for a user"""
#         if user_id not in self.user_id_map:
#             return []
        
#         user_idx = self.user_id_map[user_id]
        
#         # Calculate safe N value
#         # Get number of items user has already interacted with
#         user_items = self.user_item_matrix[user_idx].nonzero()[1]
#         num_interacted = len(user_items)
#         num_available = len(self.item_ids) - num_interacted
        
#         # Can't recommend more than available items
#         safe_n = min(top_k, num_available)
        
#         if safe_n <= 0:
#             return []
        
#         try:
#             # Get recommendations
#             ids, scores = self.model.recommend(
#                 user_idx,
#                 self.user_item_matrix[user_idx],
#                 N=safe_n,
#                 filter_already_liked_items=True
#             )
#         except IndexError as e:
#             # If still IndexError, return empty (model needs retraining)
#             logger.warning(f"IndexError in recommend for user {user_id}: {e}. Model may need retraining.")
#             return []
        
#         # Map back to book IDs
#         results = []
#         for idx, score in zip(ids, scores):
#             book_id = self.item_id_reverse[idx]
#             if filter_items and book_id in filter_items:
#                 continue
#             results.append((book_id, float(score)))
#             if len(results) >= top_k:
#                 break
        
#         return results
    
#     def save(self, path: Path):
#         """Save model"""
#         path.parent.mkdir(parents=True, exist_ok=True)
#         with open(path, 'wb') as f:
#             pickle.dump({
#                 'model': self.model,
#                 'user_item_matrix': self.user_item_matrix,
#                 'user_ids': self.user_ids,
#                 'item_ids': self.item_ids,
#                 'user_id_map': self.user_id_map,
#                 'item_id_map': self.item_id_map,
#                 'item_id_reverse': self.item_id_reverse
#             }, f)
#         logger.info(f"Saved CF model to {path}")
    
#     @classmethod
#     def load(cls, path: Path):
#         """Load saved model"""
#         with open(path, 'rb') as f:
#             data = pickle.load(f)
        
#         model = cls()
#         model.model = data['model']
#         model.user_item_matrix = data['user_item_matrix']
#         model.user_ids = data['user_ids']
#         model.item_ids = data['item_ids']
#         model.user_id_map = data['user_id_map']
#         model.item_id_map = data['item_id_map']
#         model.item_id_reverse = data['item_id_reverse']
#         return model