import argparse
import logging

from jarvis.logging_setup import setup_logging
from jarvis.router import Router
from jarvis.skills.clock import ClockSkill
from jarvis.skills.greeting import GreetingSkill

logger = logging.getLogger(__name__)

EXIT_WORDS = {"exit", "quit", "bye"}


def build_router() -> Router:
    return Router([GreetingSkill(), ClockSkill()])


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis")
    parser.add_argument("--debug", action="store_true", help="show debug logs")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    logger.info("JARVIS starting up")

    router = build_router()
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