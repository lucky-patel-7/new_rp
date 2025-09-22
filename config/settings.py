"""
Configuration settings for the Resume Parser application.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv


load_dotenv()


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI configuration."""

    api_key: Optional[str] = Field(default=None)
    endpoint: Optional[str] = Field(default=None)
    api_version: str = Field(default="2024-02-01")
    chat_deployment: Optional[str] = Field(default=None)
    embedding_deployment: Optional[str] = Field(default=None)

    class Config:
        env_prefix = "AZURE_OPENAI_"


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""

    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    collection_name: str = Field(default="resume_embeddings")
    vector_size: int = Field(default=3072)

    class Config:
        env_prefix = "QDRANT_"


class PostgreSQLSettings(BaseSettings):
    """PostgreSQL database configuration."""

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="resume_db")
    username: Optional[str] = Field(default=None)
    password: Optional[str] = Field(default=None)

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    class Config:
        env_prefix = "POSTGRES_"


class ApplicationSettings(BaseSettings):
    """General application configuration."""

    app_name: str = Field(default="Resume Parser API")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    upload_dir: str = Field(default="uploads")
    max_file_size_mb: int = Field(default=10)
    allowed_file_types: list[str] = Field(default=["pdf", "doc", "docx", "txt"])

    class Config:
        env_prefix = "APP_"


class Settings:
    """Main settings class that combines all configurations."""

    def __init__(self):
        self.azure_openai = AzureOpenAISettings()
        self.qdrant = QdrantSettings()
        self.postgres = PostgreSQLSettings()
        self.app = ApplicationSettings()

    @property
    def is_development(self) -> bool:
        return self.app.debug

    @property
    def max_file_size_bytes(self) -> int:
        return self.app.max_file_size_mb * 1024 * 1024


# Global settings instance
settings = Settings()