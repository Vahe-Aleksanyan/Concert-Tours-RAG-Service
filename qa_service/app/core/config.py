from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    CHROMA_COLLECTION: str = "concert_tours"
    CHROMA_HOST: str = "chroma"
    CHROMA_PORT: int = 8000

    MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 512
    RETRIEVAL_K: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> "Settings":
    return Settings()

settings = get_settings()