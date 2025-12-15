import pathlib
import logging
import os
import sys
from artifical_neural_network import ArtificialNeuralNetwork
from lstm_model import LSTMModel
from lstm_cnn_model import CNNLSTMModel
import traditional_models as tm

# Helper list to run evals in a loop
MODEL_LIST = [
    ArtificialNeuralNetwork,
    LSTMModel,
    CNNLSTMModel,
    tm.RandomForestModel,
    tm.XGBoostModel,
    tm.LightGBMModel,
    tm.CatBoostModel,
    tm.SVCModel,
]

# PATHS
"""
This module defines various file system paths used in the application.
"""

# Base directory of the application
BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Directory for datasets
DATASETS_DIR = BASE_DIR / "datasets"

# CICIDS2017 dataset path
CICIDS2017_PATH = DATASETS_DIR / "cicids2017"

# Ton_IoT dataset path
TON_IOT_PATH = DATASETS_DIR / "ton_iot"

# UNSW-NB15 dataset path
UNSW_NB15_PATH = DATASETS_DIR / "unsw_nb15"

# Log file path
LOG_PATH = BASE_DIR / "logs.log"

# Hyperparameters save directory
HYPERPARAMS_DIR = BASE_DIR / "hyperparams"

# Models save directory
RESULTS_DIR = BASE_DIR / "results"

_DIRS_TO_CREATE = [
    DATASETS_DIR,
    HYPERPARAMS_DIR,
    RESULTS_DIR,
]

# Create directories if they do not exist so we don't run into exceptions later
for directory in _DIRS_TO_CREATE:
    directory.mkdir(parents=True, exist_ok=True)


# LOGGER
# Configuration
ENABLE_CONSOLE_LOGGING: bool = True

# Log file path
if not LOG_PATH.exists():
    LOG_PATH.touch()

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}
level_name = os.getenv("LOG_LEVEL", "INFO").upper()
level = _LEVELS.get(level_name, logging.INFO)


def get_logger(name: str = "app") -> logging.Logger:
    # Singleton logger
    logger = logging.getLogger("app")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8", delay=True)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        if ENABLE_CONSOLE_LOGGING:
            # sys.stdout writes to the console
            ch: logging.StreamHandler = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(fmt)
            logger.addHandler(ch)

    return logger


# For quick access to the singleton logger
logger: logging.Logger = get_logger()


if __name__ == "__main__":
    ...
