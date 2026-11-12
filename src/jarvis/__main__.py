import argparse
import logging

from jarvis.brain import OFFLINE_REPLY, Brain
from jarvis.config import Settings, load_settings
from jarvis.llm.ollama import OllamaLLM
from jarvis.logging_setup import setup_logging
from jarvis.router import Router
from jarvis.skills.clock import ClockSkill
from jarvis.skills.greeting import GreetingSkill
from jarvis.skills.joke import JokeSkill
from jarvis.voice.base import SilentSpeaker, Speaker, SpeechError
from jarvis.voice.piper_speaker import PiperSpeaker
from jarvis.voice.whisper_listener import WhisperTranscriber

logger = logging.getLogger(__name__)

EXIT_WORDS = {"exit", "quit", "bye"}


def build_assistant(settings: Settings) -> tuple[Brain, Router]:
    clock = ClockSkill()
    joke = JokeSkill()

    router = Router([joke, GreetingSkill(), clock], fallback=lambda text: OFFLINE_REPLY)
    llm = OllamaLLM(model=settings.llm_model, base_url=settings.ollama_url)

    # Small models overuse tools, so only offer ones whose keywords match.
    # Try shortlist=False once a larger model is available.
    brain = Brain(llm, tools=[clock, joke], offline=router.route, shortlist=True)
    return brain, router


def build_speaker(settings: Settings, enabled: bool) -> Speaker:
    if not enabled:
        return SilentSpeaker()
    try:
        speaker = PiperSpeaker.from_model(settings.voice_model)
    except SpeechError as exc:
        logger.warning("Voice disabled: %s", exc)
        return SilentSpeaker()
    logger.info("Voice enabled: %s", settings.voice_model)
    return speaker


def build_transcriber(settings: Settings, enabled: bool) -> WhisperTranscriber | None:
    if not enabled:
        return None
    logger.info("Loading speech model %s (the first run downloads it)", settings.whisper_model)
    return WhisperTranscriber.load(settings.whisper_model, settings.whisper_device)


def read_command(transcriber: WhisperTranscriber | None) -> str:
    """Get the next command, typed or spoken."""
    if transcriber is None:
        return input("You: ").strip()

    typed = input("You (type, or press Enter to talk): ").strip()
    if typed:
        return typed

    from jarvis.voice.recorder import record_until_enter

    audio = record_until_enter()
    heard = transcriber.transcribe(audio)
    print(f"You said: {heard or '(nothing heard)'}")
    return heard


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis")
    parser.add_argument("--debug", action="store_true", help="show debug logs")
    parser.add_argument("--voice", action="store_true", help="speak replies out loud")
    parser.add_argument("--listen", action="store_true", help="push-to-talk speech input")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    settings = load_settings()
    logger.info("JARVIS starting up with model %s", settings.llm_model)

    brain, router = build_assistant(settings)
    speaker = build_speaker(settings, args.voice)
    transcriber = build_transcriber(settings, args.listen)
    print("JARVIS ready. Type 'help' for commands, 'exit' to quit.")

    while True:
        try:
            text = read_command(transcriber)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text.lower().strip(" .!") in EXIT_WORDS:
            break

        reply = router.route(text) if text.lower() == "help" else brain.ask(text)
        print(f"JARVIS: {reply}")
        try:
            speaker.speak(reply)
        except Exception:
            logger.exception("Speaking failed")

    logger.info("JARVIS shutting down")


if __name__ == "__main__":
    main()