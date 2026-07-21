"""
It is a simple test script for verifying the logger configuration.

Author:
    Mrityunjay Dubey, Hari Sharma

Description:
    This script logs a sample message to confirm that the application's logger
    is configured and working correctly.
"""

from config.logging_config import logger

logger.info("Logger is working successfully.")