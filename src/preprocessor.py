"""
preprocessor.py
-----------------
Stage: Data Preprocessing & Feature Selection.
Builds a reusable scikit-learn ColumnTransformer that scales numeric
features and one-hot-encodes categorical features. Also provides a
simple correlation-based feature selection utility.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import TARGET_COLUMN
from src.utils import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """Builds the preprocessing ColumnTransformer for the flight dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_features = [
            col for col in df.select_dtypes(include="number").columns
            if col != TARGET_COLUMN
        ]
        self.categorical_features = list(df.select_dtypes(include="object").columns)
        logger.info(f"Numeric features ({len(self.numeric_features)}): {self.numeric_features}")
        logger.info(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")

    def build_transformer(self) -> ColumnTransformer:
        """Return a ColumnTransformer that scales numerics and one-hot-encodes categoricals."""
        transformer = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), self.numeric_features),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), self.categorical_features),
            ]
        )
        return transformer

    @staticmethod
    def select_correlated_features(df: pd.DataFrame, threshold: float = 0.02) -> list:
        """
        Stage: Feature Selection.
        Keep numeric features whose absolute correlation with the target
        exceeds a small threshold, filtering out near-zero-signal columns.
        """
        numeric_df = df.select_dtypes(include="number")
        correlations = numeric_df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).abs()
        selected = correlations[correlations >= threshold].sort_values(ascending=False)
        logger.info(f"Feature-target correlations:\n{selected}")
        return list(selected.index)