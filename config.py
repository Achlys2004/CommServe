import os
import logging
from pathlib import Path

# Default to project-relative path unless overridden
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings" / "chroma"
GENERATED_DIR = BASE_DIR / "generatedfiles"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
GENERATED_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Database Path
DB_PATH = BASE_DIR / "data" / "olist.db"

os.environ["MPLBACKEND"] = "Agg"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "commserve.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "backend": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "propagate": False,
        }
    },
    "root": {"level": "INFO", "handlers": ["console", "file"]},
}


def setup_logging():
    """Setup logging configuration."""
    from logging.config import dictConfig

    dictConfig(LOGGING_CONFIG)
