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


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis")
    parser.add_argument("--debug", action="store_true", help="show debug logs")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    settings = load_settings()
    logger.info("JARVIS starting up with model %s", settings.llm_model)

    brain, router = build_assistant(settings)
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

        reply = router.route(text) if text.lower() == "help" else brain.ask(text)
        print(f"JARVIS: {reply}")

    logger.info("JARVIS shutting down")


if __name__ == "__main__":
    main()