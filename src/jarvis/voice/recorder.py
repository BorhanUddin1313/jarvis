import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000  # Whisper expects 16,000 samples per second


def record_until_enter(prompt: str = "Listening... press Enter to stop.") -> np.ndarray:
    """Record from the default microphone until the user presses Enter."""
    chunks: list[np.ndarray] = []

    def on_audio(indata, frames, time_info, status) -> None:
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=on_audio):
        input(prompt)

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)[:, 0]