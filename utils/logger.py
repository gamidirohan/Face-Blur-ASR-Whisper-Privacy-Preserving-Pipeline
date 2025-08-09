from loguru import logger
import sys

# Configure Loguru logger
logger.remove()
logger.add(sys.stderr, level="INFO", backtrace=True, diagnose=True, enqueue=True)

__all__ = ["logger"]
