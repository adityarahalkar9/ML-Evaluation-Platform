from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ML Model Evaluation & Inference Platform"
    api_version: str = "1.0.0"

    storage_dir: Path = Path("storage")

    default_seed: int = 42
    device: str = "auto"          # auto | cpu | cuda
    deterministic: bool = False

    max_train_workers: int = 1    # serialize training jobs by default

    upload_max_mb: float = 200.0
    dataset_max_rows: int = 5_000_000
    max_classes: int = 50

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="MLEP_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()