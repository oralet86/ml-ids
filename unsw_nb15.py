import time
import pandas as pd
from utils import UNSW_NB15_PATH, logger


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


def preprocess_unsw_nb15(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the UNSW-NB15 DataFrame."""
    logger.info("Starting preprocessing of UNSW-NB15 dataset...")
    start = time.time()

    pass

    end = time.time()
    logger.info(f"Completed preprocessing in {end - start:.2f} seconds")
    return df
