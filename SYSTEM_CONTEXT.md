# Book Recommendation System (RS) - Core Context

This document provides a comprehensive overview of the source code architecture and logic of the Recommendation System (RS) module. This context is designed to help AI assistants understand the codebase structure, algorithms, and data flows to facilitate future feature integrations or debugging.

## 1. System Overview

The Recommendation System is built using **FastAPI** and implements a **Hybrid Recommendation Engine** combining Collaborative Filtering and Content-Based Filtering. The system is designed to provide book recommendations, find similar books, enforce diversity in recommendations, and support online learning to incrementally update user profiles.

### Key Technologies
- **Web Framework:** FastAPI, Uvicorn
- **Collaborative Filtering:** `implicit` (Alternating Least Squares - ALS)
- **Content-Based Filtering:** `sentence-transformers` (SBERT - specifically `keepitreal/vietnamese-sbert`)
- **Data Processing:** Pandas, NumPy
- **Database:** PostgreSQL (via `sqlalchemy` and `psycopg2-binary`)

## 2. Directory Structure (`src/`)

The core logic is structured under the `src/` directory with the following modules:

### `src/api/` - API Layer
Contains the FastAPI routes and request/response schemas.
- `routes.py`: Defines all HTTP endpoints.
  - `GET /recommendations`: Returns hybrid recommendations (ALS + SBERT).
  - `GET /similar`: Returns semantically similar books using SBERT embeddings.
  - `GET /diversity`: Returns diverse recommendations to avoid the "filter bubble" effect.
  - `POST /feedback`: Records user interactions (rating, read, favorite) and triggers online learning.
  - `POST /retrain`: Triggers a full background retrain of the ALS and SBERT models.
  - `POST /online-learning/update`: Forces an incremental update of the user's SBERT profile from buffered interactions.
- `schemas.py`: Pydantic models for API request validation and response formatting.

### `src/models/` - Recommendation Engines
Contains the core algorithmic implementations.
- `hybrid_recommender.py`: The main orchestrator. It manages the `CollaborativeModel`, `SBERTContentModel`, and `DiversityRecommender`. It computes the final hybrid score using a weighted sum: `final_score = alpha * ALS_score + (1 - alpha) * SBERT_score`. It also handles "popularity fallback" for cold-start users and buffers interactions for online learning.
- `collaborative.py`: Wraps the `implicit.als.AlternatingLeastSquares` model. It handles building the sparse user-item interaction matrix and training the ALS factors.
- `diversity.py`: Implements a diversity re-ranker. It likely uses Maximal Marginal Relevance (MMR) or a similar algorithm, utilizing SBERT embeddings and book metadata to ensure recommended books are not too similar to each other.

### `src/features/` - Feature Engineering
Handles text processing and embeddings.
- `sbert_features.py`: Defines `SBERTContentModel`. It takes book metadata (title, description, genres, authors), encodes them into dense vectors using the Vietnamese SBERT model, and computes item-item cosine similarity. It also builds user profiles by averaging the embeddings of the books a user has interacted with (weighted by interaction strength).
- `text_processor.py`: Contains text cleaning utilities (lowercasing, removing special characters) to prepare book descriptions before feeding them into SBERT.

### `src/data/` - Data Ingestion
- `db_loader.py`: Connects to PostgreSQL.
  - `load_books()`: Fetches book metadata (title, description, authors, genres).
  - `load_interactions()`: Fetches user-book interactions. It normalizes different types of interactions into a generic `strength` score:
    - Explicit Ratings (1-5 stars) -> `strength = rating_value`
    - Reading History -> `strength = max(0.5, (progress/100) * 5.0)`
    - Favorites -> `strength = 5.0`

### `src/utils/` - Utilities
- `config.py`: Loads environment variables (`.env`) like database credentials and model hyperparameters (`ALPHA`, `ALS_FACTORS`, etc.).
- `evaluation.py`: Contains metrics (Hit Rate, NDCG, Coverage) to evaluate the recommender's performance.
- `logging_config.py`: Centralized logging configuration.
- `callback.py`: Contains webhook logic (`notify_retrain_complete_sync`, `notify_incremental_update_sync`) to notify the main Java/Node backend when models are retrained or user caches need invalidation.

## 3. Data Flow & Execution Logic

### 3.1. Training Pipeline (`train.py`)
1. Data Loader fetches all books and interactions from the database.
2. `HybridRecommender.train()` is called.
3. ALS model builds the user-item sparse matrix and factorizes it.
4. SBERT model encodes all books and builds a semantic profile for each user based on their history.
5. Diversity Model prepares embeddings and tags for reranking.
6. The models are saved to the local `./artifacts` directory (`als_model.pkl`, `sbert_model.pkl`, `hybrid_recommender_metadata.pkl`).

### 3.2. Inference Pipeline (`server.py` & `routes.py`)
1. On startup, the FastAPI server loads the pre-trained artifacts into memory.
2. When a `GET /recommendations` request arrives:
   - The engine gets the top `K * 3` items from ALS.
   - The engine gets the top `K * 3` items from SBERT.
   - The scores are Min-Max normalized.
   - A final weighted score is calculated. The list is sorted, and the top `K` items are returned.
   - If a user has no history (Cold Start), the system falls back to returning the most popular books globally.

### 3.3. Online Learning (Incremental Updates)
- When a user interacts with a book, the Java Backend sends a request to `POST /feedback`.
- The interaction is stored in an `interaction_buffer` in memory.
- When the buffer reaches its capacity (or is manually triggered via `/online-learning/update`), the system incrementally updates the user's **SBERT semantic profile** by shifting their user vector closer to the vector of the newly interacted book.
- **Note:** The ALS model cannot be updated incrementally in real-time. It requires a full batch retrain via `POST /retrain`.

## 4. How to Extend (For AI Assistants)

When adding new features, follow these guidelines:
- **Adding new Data Sources:** Modify SQL queries in `src/data/db_loader.py`. Ensure you map new interactions to a `strength` metric to maintain compatibility.
- **Tuning Recommendations:** The balance between Collaborative and Content-based filtering is controlled by the `ALPHA` parameter (default ~ 0.4 to 0.6).
- **Adding new Endpoints:** Add routes in `src/api/routes.py` and corresponding Request/Response formats in `src/api/schemas.py`.
- **Modifying Embeddings:** Adjust the SBERT model loading or text concatenation logic inside `src/features/sbert_features.py`.
- **Model Versioning:** The system currently saves artifacts to a local folder. For cloud deployments, consider altering `hybrid_recommender.py` `save()` and `load()` methods to support S3/GCS.
