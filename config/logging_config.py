"""
Centralized Logging Configuration

Author:
    Mrityunjay Dubey, Hari Sharma

Description:
    Creates a single logger instance that is shared across the
    entire Retrieval-Augmented Generation (RAG) pipeline.

    Every module should import this logger instead of creating
    its own logger.
"""

import logging
import sys

from config.config import LOG_LEVEL


def get_logger() -> logging.Logger:
    """
    Configure and return the application logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger("RAGPipeline")

    # Prevent duplicate handlers when running inside Jupyter Notebook
    if logger.hasHandlers():
        return logger

    logger.setLevel(LOG_LEVEL)

    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S"
    )

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


logger = get_logger()