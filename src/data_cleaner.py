"""
data_cleaner.py
-----------------
Stage: Data Cleaning.
Handles missing values, duplicate rows, and invalid/outlier prices.
"""

import pandas as pd

from src.config import TARGET_COLUMN
from src.utils import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Cleans the raw flight price DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def drop_duplicates(self) -> "DataCleaner":
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info(f"Dropped {before - len(self.df)} duplicate rows.")
        return self

    def drop_missing_target(self) -> "DataCleaner":
        """Rows with no Price cannot be used for supervised training."""
        before = len(self.df)
        self.df = self.df.dropna(subset=[TARGET_COLUMN])
        logger.info(f"Dropped {before - len(self.df)} rows with missing target.")
        return self

    def fill_missing_categoricals(self) -> "DataCleaner":
        """Fill missing categorical values with the column mode (most frequent)."""
        categorical_cols = self.df.select_dtypes(include=["object", "str"]).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_value = self.df[col].mode().iloc[0]
                self.df[col] = self.df[col].fillna(mode_value)
                logger.info(f"Filled missing values in '{col}' with mode '{mode_value}'.")
        return self

    def remove_price_outliers(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> "DataCleaner":
        """Remove extreme price outliers using the 1st and 99th percentiles."""
        lower = self.df[TARGET_COLUMN].quantile(lower_quantile)
        upper = self.df[TARGET_COLUMN].quantile(upper_quantile)
        before = len(self.df)
        self.df = self.df[(self.df[TARGET_COLUMN] >= lower) & (self.df[TARGET_COLUMN] <= upper)]
        logger.info(
            f"Removed {before - len(self.df)} price outlier rows outside "
            f"[{lower:.0f}, {upper:.0f}]."
        )
        return self

    def get_data(self) -> pd.DataFrame:
        return self.df.reset_index(drop=True)