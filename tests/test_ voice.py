import pytest

from jarvis.voice.base import SilentSpeaker, SpeechError
from jarvis.voice.piper_speaker import PiperSpeaker
import numpy as np
from jarvis.voice.whisper_listener import SAMPLE_RATE, WhisperTranscriber

class FakeVoice:
    """Pretends to be a Piper voice by writing a tiny silent WAV."""

    def synthesize_wav(self, text, wav_file):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 100)


def test_speak_plays_wav_audio():
    played = []
    PiperSpeaker(FakeVoice(), play=played.append).speak("hello")
    assert len(played) == 1
    assert played[0].startswith(b"RIFF")


def test_empty_text_is_not_spoken():
    played = []
    PiperSpeaker(FakeVoice(), play=played.append).speak("   ")
    assert played == []


def test_missing_model_raises_speech_error(tmp_path):
    with pytest.raises(SpeechError):
        PiperSpeaker.from_model(tmp_path / "missing.onnx")


def test_silent_speaker_does_nothing():
    SilentSpeaker().speak("hello")

class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeWhisper:
    """Pretends to be a faster-whisper model."""

    def __init__(self, texts):
        self.texts = texts
        self.called = False

    def transcribe(self, audio, **kwargs):
        self.called = True
        return (FakeSegment(t) for t in self.texts), None


def one_second():
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


def test_transcribe_joins_segments():
    model = FakeWhisper([" What time", " is it? "])
    assert WhisperTranscriber(model).transcribe(one_second()) == "What time is it?"


def test_too_short_audio_is_skipped():
    model = FakeWhisper(["should not appear"])
    short = np.zeros(100, dtype=np.float32)
    assert WhisperTranscriber(model).transcribe(short) == ""
    assert model.called is False


def test_silence_gives_empty_text():
    assert WhisperTranscriber(FakeWhisper([])).transcribe(one_second()) == ""