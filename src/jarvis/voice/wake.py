import numpy as np

FRAME = 1280  # 80 ms of audio at 16 kHz, the chunk size openWakeWord expects


def to_int16(frame: np.ndarray) -> np.ndarray:
    """Convert float audio (-1.0 to 1.0) into 16-bit integers."""
    return (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)


class WakeWordDetector:
    """Checks short audio frames for the wake phrase."""

    def __init__(self, model, threshold: float = 0.5) -> None:
        self._model = model
        self._threshold = threshold

    @classmethod
    def load(cls, name: str = "hey_jarvis", threshold: float = 0.5) -> "WakeWordDetector":
        from openwakeword.model import Model

        return cls(Model(wakeword_models=[name], inference_framework="onnx"), threshold)

    def detected(self, frame: np.ndarray) -> bool:
        scores = self._model.predict(frame)
        return max(scores.values(), default=0.0) >= self._threshold

    def reset(self) -> None:
        """Clear the model's memory of recent audio, so it doesn't fire twice."""
        self._model.reset()

    def score(self, frame: np.ndarray) -> float:
        """How confident the model is that this frame ends the wake phrase (0 to 1)."""
        return max(self._model.predict(frame).values(), default=0.0)

    def detected(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self._threshold