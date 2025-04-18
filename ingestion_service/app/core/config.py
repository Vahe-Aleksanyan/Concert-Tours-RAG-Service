from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import AnyUrl, Field, root_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    CHROMA_COLLECTION: str = "concert_tours"
    CHROMA_HOST: str = "chroma"
    CHROMA_PORT: int = 8000
    ALLOW_ORIGINS: List[AnyUrl] = ["http://localhost", "http://127.0.0.1"]

    # Prompt tuning
    MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"  # Add this line
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 512
    RETRIEVAL_K: int = 10

    CHROMA_PERSIST_DIR: Path = Path("/data/chroma/persist")

    # Where to store local Chroma if using `persist_directory`
    DATA_DIR: Path = Path("/data/chroma")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @root_validator(pre=True)
    def parse_allow_origins(cls, values):
        allow_origins = values.get("ALLOW_ORIGINS")
        if isinstance(allow_origins, str):
            # Split the string and convert to list of AnyUrl
            values["ALLOW_ORIGINS"] = [AnyUrl.parse(url.strip()) for url in allow_origins.split(",")]
        return values


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
