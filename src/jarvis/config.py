import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """All configurable values for JARVIS, with sensible defaults."""

    llm_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"


def load_settings() -> Settings:
    """Read settings from environment variables, falling back to defaults."""
    defaults = Settings()
    return Settings(
        llm_model=os.environ.get("JARVIS_LLM_MODEL", defaults.llm_model),
        ollama_url=os.environ.get("JARVIS_OLLAMA_URL", defaults.ollama_url),
    )