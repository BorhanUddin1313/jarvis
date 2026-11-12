import numpy as np

SAMPLE_RATE = 16_000
MIN_SECONDS = 0.3


class WhisperTranscriber:
    """Turns recorded audio into text with faster-whisper."""

    def __init__(self, model, sample_rate: int = SAMPLE_RATE) -> None:
        self._model = model
        self._sample_rate = sample_rate

    @classmethod
    def load(cls, model_name: str = "base.en", device: str = "cpu") -> "WhisperTranscriber":
        """Load a Whisper model. The first run downloads it."""
        from faster_whisper import WhisperModel

        compute_type = "int8" if device == "cpu" else "float16"
        return cls(WhisperModel(model_name, device=device, compute_type=compute_type))

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) < MIN_SECONDS * self._sample_rate:
            return ""
        segments, _info = self._model.transcribe(
            audio, language="en", beam_size=1, vad_filter=True
        )
        return " ".join(segment.text.strip() for segment in segments).strip()