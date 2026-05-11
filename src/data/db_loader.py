# ==================== src/data/db_loader.py ====================
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, Tuple
import numpy as np
from src.utils.config import get_settings
from src.utils.logging_config import logger

class DatabaseLoader:
    def __init__(self, db_uri: str, schema: str):
        self.engine = create_engine(db_uri)
        self.schema = schema
    
    def load_books(self) -> pd.DataFrame:
        """Load books with authors and genres for content-based filtering"""
        query = f"""
        SELECT
            b.book_id,
            b.title,
            b.description,
            b.publisher,
            b.publication_year,
            COALESCE(string_agg(DISTINCT a.author_name, ' '), '') AS authors_text,
            COALESCE(string_agg(DISTINCT g.genre_name, ' '), '') AS genres_text
        FROM {self.schema}.books b
        LEFT JOIN {self.schema}.book_authors ba ON ba.book_id = b.book_id
        LEFT JOIN {self.schema}.authors a ON a.author_id = ba.author_id
        LEFT JOIN {self.schema}.book_genres bg ON bg.book_id = b.book_id
        LEFT JOIN {self.schema}.genres g ON g.genre_id = bg.genre_id
        WHERE b.is_deleted = FALSE
        GROUP BY b.book_id, b.title, b.description, b.publisher, b.publication_year
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        logger.info(f"Loaded {len(df)} books")
        return df
    
    def load_interactions(self) -> pd.DataFrame:
        """Load all user-item interactions with normalized weights and time decay"""
        query = f"""
        SELECT user_id, book_id, created_at AS ts, 
            rating_value::float AS strength, 
            'rating' AS type
        FROM {self.schema}.ratings
        WHERE created_at >= NOW() - INTERVAL '1 year'
        
        UNION ALL
        SELECT user_id, book_id, last_read_at AS ts, 
            CASE
                WHEN progress IS NULL THEN 0.5
                ELSE GREATEST(0.5, (progress / 100.0) * 5.0)
            END::float AS strength,
            'history' AS type
        FROM {self.schema}.reading_history
        WHERE last_read_at >= NOW() - INTERVAL '1 year'
        
        UNION ALL
        SELECT user_id, book_id, added_at AS ts, 
            5.0 AS strength,  -- Favorites = max rating
            'favorite' AS type
        FROM {self.schema}.favorites
        WHERE added_at >= NOW() - INTERVAL '1 year'
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        
        # Log strength distribution
        logger.info(f"Strength stats: min={df['strength'].min():.2f}, "
                    f"max={df['strength'].max():.2f}, "
                    f"mean={df['strength'].mean():.2f}, "
                    f"median={df['strength'].median():.2f}")
        
        logger.info(f"Loaded {len(df)} interactions")
        return df
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both books and interactions"""
        return self.load_books(), self.load_interactions()
