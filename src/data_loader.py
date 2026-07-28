"""
data_loader.py
---------------
Stage: Data Collection & Data Loading.
"""

import os
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Loads the raw flight price Excel file from disk into a DataFrame."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        """Load the raw Excel file. Raises a clear error if the file is missing."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Raw data file not found at '{self.file_path}'.\n"
                f"Place your Kaggle Flight Price Prediction file at that path "
                f"(check the filename/extension matches RAW_DATA_FILE in config.py)."
            )
        df = pd.read_excel(self.file_path)
        logger.info(f"Loaded dataset from {self.file_path} | shape={df.shape}")
        return df

    @staticmethod
    def describe_dataset(df: pd.DataFrame) -> None:
        """Stage: Data Understanding — quick structural summary."""
        logger.info("----- DATA UNDERSTANDING -----")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info("Data types:\n%s", df.dtypes)
        logger.info("Missing values per column:\n%s", df.isnull().sum())
        logger.info("Duplicate rows: %d", df.duplicated().sum())
        if "Price" in df.columns:
            logger.info("Price summary:\n%s", df["Price"].describe())