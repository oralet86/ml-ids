import time
import pandas as pd
from utils import CICIDS2017_PATH, logger
from glob import glob


def get_cicids2017() -> pd.DataFrame:
    """Load the CICIDS2017 dataset from CSV files into a single DataFrame. \\
    This function assumes that the dataset is somewhere under the directory: datasets/cicids2017 \\
    and looks for all CSV files in that directory. \\
    It reads each CSV file, concatenates them into a single DataFrame, and returns it.
    """
    data_path = str(CICIDS2017_PATH / "*.csv")
    logger.info(f"Loading CICIDS2017 dataset from {data_path}")
    all_files = glob(data_path)
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


def preprocess_cicids2017(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the CICIDS2017 DataFrame."""
    logger.info("Starting preprocessing of CICIDS2017 dataset...")
    start = time.time()

    pass

    end = time.time()
    logger.info(f"Completed preprocessing in {end - start:.2f} seconds")
    return df
