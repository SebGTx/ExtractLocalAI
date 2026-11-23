# core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ollama_model: str
    ollama_host: str
    ollama_force_cpu: bool
    max_document_length: int
    log_level: str
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

settings = Settings()
