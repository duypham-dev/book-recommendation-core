# ==================== src/utils/config.py ====================
from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import computed_field

class Settings(BaseSettings):
    # Database
    db_host: str
    db_port: int = 5432
    db_name: str = "book_recommendation_db"
    db_user: str
    db_password: str
    db_schema: str = "book_recommendation_system"
    
    @computed_field
    @property
    def db_uri(self) -> str:
        """Dynamically construct database URI from components"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    # Model parameters
    cf_factors: int = 64
    cf_iterations: int = 30
    cf_regularization: float = 0.01
    alpha: float = 0.6  # hybrid blend weight
    
    # API
    api_title: str = "Book Recommendation API"
    api_version: str = "1.0.0"
    
    # Paths
    artifacts_dir: str = "./artifacts"
    
    # Java Backend callback URL for cache invalidation
    java_backend_url: str = "http://localhost:8080/api/v1"
    callback_enabled: bool = True
    callback_timeout: int = 5  # seconds
    
    # RabbitMQ
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672"
    
    # Python environment
    pythonunbuffered: int = 1

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()