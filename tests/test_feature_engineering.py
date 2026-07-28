"""
Unit tests for FeatureEngineer.
Run with: pytest tests/
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import FeatureEngineer


def make_sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Airline": ["IndiGo", "Air India"],
        "Date_of_Journey": ["24/03/2019", "01/05/2019"],
        "Source": ["Delhi", "Mumbai"],
        "Destination": ["Cochin", "Delhi"],
        "Route": ["DEL → COK", "BOM → DEL"],
        "Dep_Time": ["22:20", "05:50"],
        "Arrival_Time": ["01:10 23 Mar", "13:15"],
        "Duration": ["2h 50m", "7h 25m"],
        "Total_Stops": ["non-stop", "1 stop"],
        "Additional_Info": ["No info", "No info"],
        "Price": [3897, 7662],
    })


def test_parse_duration_converts_to_minutes():
    df = make_sample_df()
    result = FeatureEngineer(df).parse_duration().get_data()
    assert "Duration_Minutes" in result.columns
    assert result.loc[0, "Duration_Minutes"] == 170  # 2h50m = 170 minutes
    assert "Duration" not in result.columns


def test_encode_total_stops_to_ordinal():
    df = make_sample_df()
    result = FeatureEngineer(df).encode_total_stops().get_data()
    assert result.loc[0, "Total_Stops"] == 0
    assert result.loc[1, "Total_Stops"] == 1


def test_parse_date_of_journey_extracts_components():
    df = make_sample_df()
    result = FeatureEngineer(df).parse_date_of_journey().get_data()
    assert result.loc[0, "Journey_Day"] == 24
    assert result.loc[0, "Journey_Month"] == 3
    assert "Date_of_Journey" not in result.columns


def test_parse_time_column_handles_next_day_arrival_format():
    """Arrival_Time sometimes includes a trailing date, e.g. '01:10 23 Mar'."""
    df = make_sample_df()
    result = FeatureEngineer(df).parse_time_column("Arrival_Time", "Arrival").get_data()
    assert result.loc[0, "Arrival_Hour"] == 1
    assert result.loc[0, "Arrival_Minute"] == 10