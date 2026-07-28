"""
model_evaluator.py
--------------------
Stages: Model Evaluation & Model Comparison.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates trained pipelines on a held-out test set and ranks them."""

    def __init__(self, trained_pipelines: dict):
        self.trained_pipelines = trained_pipelines
        self.results_df = None

    def evaluate_all(self, X_test, y_test) -> pd.DataFrame:
        """Compute RMSE, MAE, and R2 for every trained model."""
        records = []
        for name, pipeline in self.trained_pipelines.items():
            predictions = pipeline.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            mae = float(mean_absolute_error(y_test, predictions))
            r2 = float(r2_score(y_test, predictions))
            records.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2_Score": r2})
            logger.info(f"{name} -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")

        self.results_df = pd.DataFrame(records).sort_values("RMSE").reset_index(drop=True)
        return self.results_df

    def get_best_model_name(self) -> str:
        """Best model = lowest RMSE on the test set."""
        if self.results_df is None:
            raise RuntimeError("Call evaluate_all() before get_best_model_name().")
        return self.results_df.iloc[0]["Model"]