from abc import ABC, abstractmethod


class SpeechError(Exception):
    """Raised when JARVIS can't load a voice or produce speech."""


class Speaker(ABC):
    """Anything that can say text out loud."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Say the text. Returns once speaking is finished."""


class SilentSpeaker(Speaker):
    """A speaker that does nothing, used in text-only mode."""

    def speak(self, text: str) -> None:
        pass