import pathlib
import logging
import os
import sys
import pandas as pd
import glob
import time

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
MODELS_DIR = BASE_DIR / "models"

_DIRS_TO_CREATE = [
    DATASETS_DIR,
    HYPERPARAMS_DIR,
    MODELS_DIR,
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


@classmethod
def get_logger(cls, name: str = "app") -> logging.Logger:
    # Singleton logger
    logger = logging.getLogger("app")
    logger.setLevel(cls.level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8", delay=True)
        fh.setLevel(cls.level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        if cls.ENABLE_CONSOLE_LOGGING:
            # sys.stdout writes to the console
            ch: logging.StreamHandler = logging.StreamHandler(sys.stdout)
            ch.setLevel(cls.level)
            ch.setFormatter(fmt)
            logger.addHandler(ch)

    return logger


# DATASET FETCHER
def get_cicids2017() -> pd.DataFrame:
    """Load the CICIDS2017 dataset from CSV files into a single DataFrame. \\
    This function assumes that the dataset is somewhere under the directory: datasets/cicids2017 \\
    and looks for all CSV files in that directory. \\
    It reads each CSV file, concatenates them into a single DataFrame, and returns it.
    """
    data_path = str(CICIDS2017_PATH / "*.csv")
    logger.info(f"Loading CICIDS2017 dataset from {data_path}")
    all_files = glob.glob(data_path)
    logger.info(f"Found {len(all_files)} files in {CICIDS2017_PATH}")
    df_list = []
    start = time.time()
    for file in all_files:
        df = pd.read_csv(file, engine="pyarrow", dtype_backend="pyarrow")
        df.columns = df.columns.str.strip()
        df_list.append(df)
    end = time.time()
    logger.info(f"Loaded {len(all_files)} files in {end - start:.2f} seconds")
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def get_toniot() -> pd.DataFrame:
    """Load the TON_IoT dataset from a CSV file into a DataFrame. \\
    This function assumes that the dataset is somewhere under the directory: datasets/ton_iot \\
    and looks for the file named 'train_test_network.csv'.
    """
    file = next(TON_IOT_PATH.rglob("train_test_network.csv"))
    if not file:
        raise FileNotFoundError("train_test_network.csv not found.")
    logger.info(f"Found dataset in {file}")
    start = time.time()
    df = pd.read_csv(file, engine="pyarrow", dtype_backend="pyarrow")
    df.columns = df.columns.str.strip()
    end = time.time()
    logger.info(f"Loaded dataset in {end - start:.2f} seconds")
    return df


def get_unsw_nb15() -> pd.DataFrame:
    """Load the UNSW-NB15 dataset from a CSV file into a DataFrame. \\
    This function assumes that the dataset is somewhere under the directory: datasets/unsw_nb15 \\
    and looks for the files named 'UNSW_NB15_testing-set.csv' and 'UNSW_NB15_training-set.csv'. \\
    Since it is common practice to combine the training and testing sets for this dataset, this function \\
    loads both files and concatenates them into a single DataFrame.
    """
    train_file = next(UNSW_NB15_PATH.rglob("UNSW_NB15_training-set.csv"))
    test_file = next(UNSW_NB15_PATH.rglob("UNSW_NB15_testing-set.csv"))
    if not train_file or not test_file:
        raise FileNotFoundError("Training or testing set files not found.")
    logger.info(f"Found training dataset in {train_file}")
    logger.info(f"Found testing dataset in {test_file}")
    start = time.time()
    df_train = pd.read_csv(train_file, engine="pyarrow", dtype_backend="pyarrow")
    df_test = pd.read_csv(test_file, engine="pyarrow", dtype_backend="pyarrow")
    df_train.columns = df_train.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()
    combined_df = pd.concat([df_train, df_test], ignore_index=True)
    end = time.time()
    logger.info(f"Loaded datasets in {end - start:.2f} seconds")
    return combined_df


# For quick access to the singleton logger
logger = get_logger()


if __name__ == "__main__":
    df = get_cicids2017()
    logger.info(df.head())
