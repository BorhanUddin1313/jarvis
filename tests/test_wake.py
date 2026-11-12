import numpy as np

from jarvis.voice.endpoint import Endpointer
from jarvis.voice.wake import FRAME, WakeWordDetector, to_int16

LOUD = np.full(FRAME, 0.1, dtype=np.float32)
QUIET = np.zeros(FRAME, dtype=np.float32)
SILENT_INT16 = np.zeros(FRAME, dtype=np.int16)


class FakeWakeModel:
    def __init__(self, scores):
        self.scores = scores
        self.was_reset = False

    def predict(self, frame):
        return self.scores

    def reset(self):
        self.was_reset = True


def feed_all(endpointer, frames):
    return [endpointer.feed(frame) for frame in frames]


def test_wake_detected_above_threshold():
    assert WakeWordDetector(FakeWakeModel({"hey_jarvis": 0.9})).detected(SILENT_INT16)


def test_wake_not_detected_below_threshold():
    assert not WakeWordDetector(FakeWakeModel({"hey_jarvis": 0.2})).detected(SILENT_INT16)


def test_reset_clears_model():
    model = FakeWakeModel({})
    WakeWordDetector(model).reset()
    assert model.was_reset


def test_to_int16_scales_and_clips():
    frame = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    assert to_int16(frame).tolist() == [0, 32767, -32767, 32767]


def test_stops_after_silence_following_speech():
    endpointer = Endpointer(end_silence=0.4)
    assert feed_all(endpointer, [LOUD] * 3 + [QUIET] * 4) == [False] * 7
    assert endpointer.feed(QUIET) is True


def test_keeps_recording_while_speaking():
    endpointer = Endpointer(max_seconds=60)
    assert not any(feed_all(endpointer, [LOUD] * 50))


def test_gives_up_without_speech():
    endpointer = Endpointer(no_speech_timeout=0.4)
    assert feed_all(endpointer, [QUIET] * 5) == [False] * 4 + [True]
    assert endpointer.heard_speech is False


def test_stops_at_max_length():
    endpointer = Endpointer(max_seconds=0.8, end_silence=10)
    results = feed_all(endpointer, [LOUD] * 10)
    assert results[-1] is True
    assert not any(results[:-1])