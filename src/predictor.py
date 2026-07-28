"""
predictor.py
------------
Stage: Prediction / Inference.
Loads the saved best model and serves predictions on new, unseen
flight records that follow the same raw schema as the training data.
"""

import joblib
import pandas as pd

from src.config import BEST_MODEL_FILE
from src.feature_engineering import FeatureEngineer
from src.utils import get_logger

logger = get_logger(__name__)


class FlightPricePredictor:
    """Wraps the trained pipeline for easy inference on raw new records."""

    def __init__(self, model_path: str = BEST_MODEL_FILE):
        self.model = joblib.load(model_path)
        logger.info(f"Loaded trained model from {model_path}")

    def predict(self, raw_df: pd.DataFrame) -> pd.Series:
        """
        Predict prices for new raw records.

        The input DataFrame must have the same raw columns as the
        original dataset (Airline, Date_of_Journey, Source, Destination,
        Route, Dep_Time, Arrival_Time, Duration, Total_Stops,
        Additional_Info), minus the Price column.
        """
        engineered = (
            FeatureEngineer(raw_df)
            .simplify_route()
            .simplify_additional_info()
            .encode_total_stops()
            .parse_date_of_journey()
            .parse_time_column("Dep_Time", "Dep")
            .parse_time_column("Arrival_Time", "Arrival")
            .parse_duration()
            .get_data()
        )
        predictions = self.model.predict(engineered)
        return pd.Series(predictions, name="Predicted_Price")