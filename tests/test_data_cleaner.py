"""
Unit tests for DataCleaner.
Run with: pytest tests/
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_cleaner import DataCleaner


def make_dirty_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Airline": ["IndiGo", "IndiGo", "Air India", None],
        "Total_Stops": ["non-stop", "non-stop", "1 stop", "1 stop"],
        "Price": [3000, 3000, np.nan, 500000],  # duplicate row, missing target, extreme outlier
    })


def test_drop_duplicates_removes_exact_duplicate_rows():
    df = make_dirty_df()
    result = DataCleaner(df).drop_duplicates().get_data()
    assert len(result) == 3


def test_drop_missing_target_removes_nan_price_rows():
    df = make_dirty_df()
    result = DataCleaner(df).drop_missing_target().get_data()
    assert result["Price"].isnull().sum() == 0


def test_fill_missing_categoricals_no_nulls_remain():
    df = make_dirty_df()
    result = DataCleaner(df).fill_missing_categoricals().get_data()
    assert result["Airline"].isnull().sum() == 0