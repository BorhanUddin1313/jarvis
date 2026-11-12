import argparse
import logging

from jarvis.brain import Brain
from jarvis.config import Settings, load_settings
from jarvis.llm.ollama import OllamaLLM
from jarvis.logging_setup import setup_logging
from jarvis.router import Router
from jarvis.skills.clock import ClockSkill
from jarvis.skills.greeting import GreetingSkill
from jarvis.skills.joke import JokeSkill

logger = logging.getLogger(__name__)

EXIT_WORDS = {"exit", "quit", "bye"}


def build_router(settings: Settings) -> Router:
    llm = OllamaLLM(model=settings.llm_model, base_url=settings.ollama_url)
    brain = Brain(llm)
    skills = [JokeSkill(), GreetingSkill(), ClockSkill()]
    return Router(skills, fallback=brain.ask)


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis")
    parser.add_argument("--debug", action="store_true", help="show debug logs")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    settings = load_settings()
    logger.info("JARVIS starting up with model %s", settings.llm_model)

    router = build_router(settings)
    print("JARVIS ready. Type 'help' for commands, 'exit' to quit.")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            break

        print(f"JARVIS: {router.route(text)}")

    logger.info("JARVIS shutting down")


if __name__ == "__main__":
    main()