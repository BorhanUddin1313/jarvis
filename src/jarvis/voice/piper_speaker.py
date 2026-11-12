import io
import wave
from pathlib import Path
from typing import Callable

from jarvis.voice.base import Speaker, SpeechError


def play_wav_windows(data: bytes) -> None:
    """Play WAV audio from memory using Windows' built-in sound module."""
    import winsound

    winsound.PlaySound(data, winsound.SND_MEMORY)


class PiperSpeaker(Speaker):
    """Speaks text with a local Piper voice model."""

    def __init__(self, voice, play: Callable[[bytes], None] = play_wav_windows) -> None:
        self._voice = voice
        self._play = play

    @classmethod
    def from_model(cls, model_path: str | Path, **kwargs) -> "PiperSpeaker":
        """Load a Piper voice from an .onnx file."""
        path = Path(model_path)
        if not path.exists():
            raise SpeechError(f"Voice model not found: {path}")

        from piper import PiperVoice

        return cls(PiperVoice.load(str(path)), **kwargs)

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        self._play(self.synthesize(text))

    def synthesize(self, text: str) -> bytes:
        """Turn text into WAV audio bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)
        return buffer.getvalue()