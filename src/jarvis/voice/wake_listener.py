import queue
from typing import Callable

import numpy as np
import sounddevice as sd

from jarvis.voice.endpoint import Endpointer
from jarvis.voice.wake import FRAME, WakeWordDetector, to_int16

SAMPLE_RATE = 16_000


def _next_frame(frames: queue.Queue) -> np.ndarray | None:
    """Wait briefly for audio; the timeout keeps Ctrl+C responsive."""
    try:
        return frames.get(timeout=0.5)
    except queue.Empty:
        return None


def listen_for_command(
    detector: WakeWordDetector,
    make_endpointer: Callable[[], Endpointer],
    on_wake: Callable[[], None] | None = None,
    require_wake: bool = True,
) -> np.ndarray:
    """Optionally wait for the wake word, then record until the user stops talking."""
    frames: queue.Queue = queue.Queue()

    def on_audio(indata, frame_count, time_info, status) -> None:
        frames.put(indata[:, 0].copy())

    # In follow-up mode, beep before the mic opens so the beep isn't recorded.
    if not require_wake and on_wake:
        on_wake()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME, callback=on_audio
    ):
        if require_wake:
            while True:
                frame = _next_frame(frames)
                if frame is not None and detector.detected(to_int16(frame)):
                    break
            detector.reset()
            if on_wake:
                on_wake()

        endpointer = make_endpointer()
        recorded: list[np.ndarray] = []
        while True:
            frame = _next_frame(frames)
            if frame is None:
                continue
            recorded.append(frame)
            if endpointer.feed(frame):
                break

    if not endpointer.heard_speech or not recorded:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(recorded)