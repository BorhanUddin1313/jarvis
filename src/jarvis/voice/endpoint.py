import numpy as np

FRAME_SECONDS = 0.08  # one 1280-sample frame at 16 kHz


class Endpointer:
    """Decides when the user has finished speaking, based on how loud each frame is."""

    def __init__(
        self,
        silence_threshold: float = 0.01,
        end_silence: float = 1.0,
        no_speech_timeout: float = 5.0,
        max_seconds: float = 15.0,
    ) -> None:
        self._threshold = silence_threshold
        self._end_frames = round(end_silence / FRAME_SECONDS)
        self._no_speech_frames = round(no_speech_timeout / FRAME_SECONDS)
        self._max_frames = round(max_seconds / FRAME_SECONDS)
        self._total = 0
        self._silent = 0
        self.heard_speech = False

    def feed(self, frame: np.ndarray) -> bool:
        """Add one frame. Returns True once recording should stop."""
        self._total += 1
        loudness = float(np.sqrt(np.mean(np.square(frame))))

        if loudness >= self._threshold:
            self.heard_speech = True
            self._silent = 0
        else:
            self._silent += 1

        if self._total >= self._max_frames:
            return True
        if not self.heard_speech and self._total >= self._no_speech_frames:
            return True
        return self.heard_speech and self._silent >= self._end_frames