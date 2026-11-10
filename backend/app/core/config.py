"""Платформа баптаулары (.env арқылы өзгертіледі)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# windcast-agent/ түбірі (backend/app/core/config.py -> 3 деңгей жоғары)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    data_dir: Path = PROJECT_ROOT / "data"
    reports_dir: Path = PROJECT_ROOT / "reports"
    database_url: str = ""  # бос болса -> data/windcast.db (SQLite)

    # LLM (OpenAI GPT)
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    openai_base_url: str = ""  # қажет болса прокси/Azure үшін
    agent_max_retries: int = 2
    agent_max_tool_rounds: int = 6

    # Open-Meteo
    openmeteo_timeout: int = 60
    openmeteo_model: str = "best_match"

    # Жоспарлаушы
    scheduler_enabled: bool = True
    scheduler_interval_minutes: int = 60

    default_lang: str = "kk"

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(self.data_dir / 'windcast.db').as_posix()}"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def exports_dir(self) -> Path:
        return self.reports_dir / "exports"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.data_dir, s.raw_dir, s.processed_dir, s.cache_dir, s.models_dir, s.reports_dir, s.exports_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s
