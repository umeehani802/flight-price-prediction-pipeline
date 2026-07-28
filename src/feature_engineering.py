"""
feature_engineering.py
------------------------
Stage: Feature Engineering.
Converts raw text/date columns into model-ready numeric features.
"""

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Builds engineered features from the raw flight columns."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def parse_date_of_journey(self) -> "FeatureEngineer":
        """Extract day, month, and day-of-week from Date_of_Journey."""
        dt = pd.to_datetime(self.df["Date_of_Journey"], format="%d/%m/%Y", errors="coerce")
        self.df["Journey_Day"] = dt.dt.day
        self.df["Journey_Month"] = dt.dt.month
        self.df["Journey_DayOfWeek"] = dt.dt.dayofweek  # 0 = Monday
        self.df = self.df.drop(columns=["Date_of_Journey"])
        logger.info("Parsed Date_of_Journey into Journey_Day, Journey_Month, Journey_DayOfWeek.")
        return self

    def parse_time_column(self, column: str, prefix: str) -> "FeatureEngineer":
        """
        Extract hour and minute from a time column. Handles both plain
        'HH:MM' values and next-day arrival formats like '01:10 22 Mar'
        (where only the leading HH:MM portion is used).
        """
        time_only = self.df[column].astype(str).str.split().str[0]
        t = pd.to_datetime(time_only, format="%H:%M", errors="coerce")
        self.df[f"{prefix}_Hour"] = t.dt.hour
        self.df[f"{prefix}_Minute"] = t.dt.minute
        self.df = self.df.drop(columns=[column])
        logger.info(f"Parsed {column} into {prefix}_Hour, {prefix}_Minute.")
        return self

    def parse_duration(self) -> "FeatureEngineer":
        """Convert '2h 50m' style strings into total minutes."""
        def to_minutes(duration_str: str) -> int:
            hours, minutes = 0, 0
            parts = str(duration_str).strip().split()
            for part in parts:
                if "h" in part:
                    hours = int(part.replace("h", "") or 0)
                elif "m" in part:
                    minutes = int(part.replace("m", "") or 0)
            return hours * 60 + minutes

        self.df["Duration_Minutes"] = self.df["Duration"].apply(to_minutes)
        self.df = self.df.drop(columns=["Duration"])
        logger.info("Converted Duration into Duration_Minutes.")
        return self

    def encode_total_stops(self) -> "FeatureEngineer":
        """Map 'non-stop', '1 stop', etc. to an ordinal integer."""
        stop_mapping = {
            "non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4
        }
        self.df["Total_Stops"] = self.df["Total_Stops"].map(stop_mapping)
        self.df["Total_Stops"] = self.df["Total_Stops"].fillna(self.df["Total_Stops"].median())
        logger.info("Encoded Total_Stops as an ordinal integer.")
        return self

    def simplify_route(self, top_n: int = 10) -> "FeatureEngineer":
        """Group rare Route categories into 'Other' to avoid a sparse one-hot space."""
        top_routes = self.df["Route"].value_counts().nlargest(top_n).index
        self.df["Route"] = self.df["Route"].where(self.df["Route"].isin(top_routes), "Other")
        logger.info(f"Simplified Route to top {top_n} categories + 'Other'.")
        return self

    def simplify_additional_info(self, top_n: int = 4) -> "FeatureEngineer":
        """Group rare Additional_Info categories into 'Other'."""
        top_categories = self.df["Additional_Info"].value_counts().nlargest(top_n).index
        self.df["Additional_Info"] = self.df["Additional_Info"].where(
            self.df["Additional_Info"].isin(top_categories), "Other"
        )
        logger.info(f"Simplified Additional_Info to top {top_n} categories + 'Other'.")
        return self

    def get_data(self) -> pd.DataFrame:
        return self.df.reset_index(drop=True)