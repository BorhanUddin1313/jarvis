import logging

from jarvis.logging_setup import setup_logging


def test_default_level_is_info():
    setup_logging()
    assert logging.getLogger().level == logging.INFO


def test_custom_level_is_applied():
    setup_logging(logging.DEBUG)
    assert logging.getLogger().level == logging.DEBUG