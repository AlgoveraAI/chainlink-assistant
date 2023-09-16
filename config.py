import os
import logging
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger with the given name and level.

    :param name: Name of the logger.
    :param level: Level of the logger.
    :return: Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = get_logger(__name__)

# Set the root directory
ROOT_DIR = Path(os.getenv("ROOT_DIR"))
DATA_DIR = ROOT_DIR / "data"

# Make sure the data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

if not os.environ.get("MAX_THREADS"):
    logger.warning("MAX_THREADS not set. Defaulting to 1.")
    MAX_THREADS = 4
else:
    MAX_THREADS = int(os.environ.get("MAX_THREADS"))

if not os.environ.get("WS_HOST"):
    logger.warning("WS_HOST not set. Defaulting to ws://localhost:8000")
    WS_HOST = "ws://localhost:8000"
else:
    WS_HOST = os.environ.get("WS_HOST")
