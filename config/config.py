"""
Central configuration management.
All settings are loaded from config/settings.yaml and overridden by environment variables.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    chat_model: str = "llama3"
    embed_model: str = "nomic-embed-text"
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 2.0


class ChromaConfig(BaseModel):
    persist_directory: str = "./data/chroma"
    collection_name: str = "documents"
    distance_function: str = "cosine"


class SQLiteConfig(BaseModel):
    database_path: str = "./data/db/localrag.db"


class IngestionConfig(BaseModel):
    watch_folder: str = "./data/watch_folder"
    uploads_folder: str = "./data/uploads"
    chunk_size: int = 512
    chunk_overlap: int = 64
    supported_extensions: List[str] = [".pdf", ".docx", ".md", ".txt"]
    poll_interval: int = 5


class RetrievalConfig(BaseModel):
    top_k: int = 5
    score_threshold: float = 0.3
    rerank: bool = False


class ChatConfig(BaseModel):
    max_history_turns: int = 20
    max_context_tokens: int = 3000
    system_prompt: str = (
        "You are a helpful assistant that answers questions based on the provided documents. "
        "Always cite your sources using the format [Source: filename, chunk N]. "
        "If the answer is not found in the documents, say so clearly. "
        "Be concise and accurate."
    )


class SecurityConfig(BaseModel):
    cors_origins: List[str] = ["http://localhost:3000"]
    api_key_enabled: bool = False
    api_key_header: str = "X-API-Key"
    rate_limit_requests: int = 100
    rate_limit_window: int = 60


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file: str = "./data/logs/app.log"
    rotation: str = "10 MB"
    retention: str = "7 days"


class MetricsConfig(BaseModel):
    enabled: bool = True
    port: int = 9090
    path: str = "/metrics"


class MCPConfig(BaseModel):
    transport: str = "stdio"
    server_name: str = "localrag-mcp"
    server_version: str = "1.0.0"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1


# ---------------------------------------------------------------------------
# Main settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Application settings loaded from YAML + env overrides.
    Environment variables take precedence over YAML values.
    """

    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_key: str = Field(default="change-me-in-production", alias="API_KEY")
    api_key_enabled: bool = Field(default=False, alias="API_KEY_ENABLED")

    # Sub-configs (populated from YAML)
    server: ServerConfig = ServerConfig()
    ollama: OllamaConfig = OllamaConfig()
    chroma: ChromaConfig = ChromaConfig()
    sqlite: SQLiteConfig = SQLiteConfig()
    ingestion: IngestionConfig = IngestionConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    chat: ChatConfig = ChatConfig()
    security: SecurityConfig = SecurityConfig()
    logging: LoggingConfig = LoggingConfig()
    metrics: MetricsConfig = MetricsConfig()
    mcp: MCPConfig = MCPConfig()

    model_config = {"env_file": str(ROOT_DIR / ".env"), "extra": "ignore"}

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path relative to project root."""
        p = Path(relative)
        if p.is_absolute():
            return p
        return (ROOT_DIR / relative).resolve()

    @property
    def chroma_path(self) -> Path:
        return self.resolve_path(self.chroma.persist_directory)

    @property
    def sqlite_path(self) -> Path:
        return self.resolve_path(self.sqlite.database_path)

    @property
    def watch_folder_path(self) -> Path:
        return self.resolve_path(self.ingestion.watch_folder)

    @property
    def uploads_folder_path(self) -> Path:
        return self.resolve_path(self.ingestion.uploads_folder)

    @property
    def log_file_path(self) -> Path:
        return self.resolve_path(self.logging.file)


def _load_yaml_settings() -> dict:
    yaml_path = ROOT_DIR / "config" / "settings.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _build_settings() -> Settings:
    yaml_data = _load_yaml_settings()
    overrides: dict = {}

    # Map YAML sections to sub-config models
    section_map = {
        "server": ServerConfig,
        "ollama": OllamaConfig,
        "chroma": ChromaConfig,
        "sqlite": SQLiteConfig,
        "ingestion": IngestionConfig,
        "retrieval": RetrievalConfig,
        "chat": ChatConfig,
        "security": SecurityConfig,
        "logging": LoggingConfig,
        "metrics": MetricsConfig,
        "mcp": MCPConfig,
    }

    for key, model_cls in section_map.items():
        if key in yaml_data:
            overrides[key] = model_cls(**yaml_data[key])

    return Settings(**overrides)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _build_settings()


def ensure_directories(settings: Settings) -> None:
    """Create all required data directories."""
    dirs = [
        settings.chroma_path,
        settings.sqlite_path.parent,
        settings.watch_folder_path,
        settings.uploads_folder_path,
        settings.log_file_path.parent,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
