# Import standard libraries
import pandas as pd
import logging


logger = logging.getLogger(__name__)
RANDOM_STATE = 42

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Adding time features...")
    df["transaction_time"] = pd.to_datetime(df["transaction_time"])
    df["is_weekend"] = df["transaction_time"].dt.dayofweek >= 5
    df["year"] = df["transaction_time"].dt.year
    df["month"] = df["transaction_time"].dt.month
    df["day"] = df["transaction_time"].dt.day
    df["hour"] = df["transaction_time"].dt.hour
    df["minute"] = df["transaction_time"].dt.minute
    df.drop(columns=["transaction_time", "name_1", "name_2", "street"], inplace=True)
    return df


def load_and_preprocess(path_to_file: str) -> pd.DataFrame:
    logger.info("Loading data from %s", path_to_file)
    df = pd.read_csv(path_to_file)
    logger.info("Raw data shape: %s", df.shape)
    df = add_time_features(df)
    logger.info("Time features added. Shape now: %s", df.shape)
    return df