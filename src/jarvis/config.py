import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """All configurable values for JARVIS, with sensible defaults."""

    llm_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"
    voice_model: str = "models/en_GB-alan-medium.onnx"
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"


def load_settings() -> Settings:
    """Read settings from environment variables, falling back to defaults."""
    defaults = Settings()
    return Settings(
        llm_model=os.environ.get("JARVIS_LLM_MODEL", defaults.llm_model),
        ollama_url=os.environ.get("JARVIS_OLLAMA_URL", defaults.ollama_url),
        voice_model=os.environ.get("JARVIS_VOICE_MODEL", defaults.voice_model),
        whisper_model=os.environ.get("JARVIS_WHISPER_MODEL", defaults.whisper_model),
        whisper_device=os.environ.get("JARVIS_WHISPER_DEVICE", defaults.whisper_device),
    )