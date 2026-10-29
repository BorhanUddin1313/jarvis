import logging

from jarvis.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    logger.info("JARVIS starting up")


if __name__ == "__main__":
    main()