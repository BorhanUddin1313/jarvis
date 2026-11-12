"""Show live microphone loudness and wake word score, to tune JARVIS's thresholds."""
import numpy as np
import sounddevice as sd

from jarvis.voice.wake import FRAME, WakeWordDetector, to_int16

detector = WakeWordDetector.load()
print("Stay quiet for a few seconds, then talk, then say 'Hey Jarvis'. Ctrl+C to stop.")

with sd.InputStream(samplerate=16_000, channels=1, dtype="float32", blocksize=FRAME) as stream:
    while True:
        frame, _overflowed = stream.read(FRAME)
        frame = frame[:, 0]
        loudness = float(np.sqrt(np.mean(np.square(frame))))
        wake = detector.score(to_int16(frame))
        bar = "#" * min(int(loudness * 1000), 60)
        print(f"loudness {loudness:.4f}  wake {wake:.2f}  {bar}")