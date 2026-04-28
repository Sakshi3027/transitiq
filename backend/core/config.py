from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    # App
    app_name: str = "TransitIQ"
    app_version: str = "1.0.0"
    debug: bool = False

    # Database
    postgres_user: str = "transitiq"
    postgres_password: str = "transitiq_secret"
    postgres_db: str = "transitiq"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl_seconds: int = 300

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:29092"
    kafka_topic_transit_events: str = "transit.events"
    kafka_topic_predictions: str = "transit.predictions"
    kafka_topic_anomalies: str = "transit.anomalies"
    kafka_group_id: str = "transitiq-consumer-group"

    # ML
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "transit-delay-prediction"
    model_registry_name: str = "transit-delay-model"
    drift_threshold: float = 0.05
    retrain_accuracy_threshold: float = 0.85

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change-me-in-production"
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    # Weather
    weather_api_key: str = "your_openweathermap_api_key"
    weather_city: str = "Chicago"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()