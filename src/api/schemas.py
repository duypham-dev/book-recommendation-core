# ==================== src/api/schemas.py ====================
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class RecommendationItem(BaseModel):
    book_id: int
    score: float
    reasons: Dict[str, float]

class RecommendationsResponse(BaseModel):
    user_id: Optional[int] = None
    limit: int
    items: List[RecommendationItem]

class SimilarItem(BaseModel):
    book_id: int
    score: float

class SimilarResponse(BaseModel):
    book_id: int
    items: List[SimilarItem]

class DiversityItem(BaseModel):
    book_id: int
    rating: float
    score: float
    metadata: Dict[str, float]

class DiversityResponse(BaseModel):
    book_id: int
    items: List[DiversityItem]

class FeedbackRequest(BaseModel):
    user_id: int
    book_id: int
    event: str = Field(..., pattern="^(rating|favorite|history)$")
    rating_value: Optional[int] = Field(None, ge=0, le=5, description="Rating: 1-5 for 'rating', 0 to remove 'favorite'")
    progress: Optional[float] = Field(None, ge=0.0, le=100.0, description="Reading progress: 0-100 for 'history'")

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
