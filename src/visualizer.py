"""
visualizer.py
--------------
Stage: Result Visualization.
Plots comparing model performance and actual-vs-predicted prices.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import FIGURES_DIR
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


class ResultVisualizer:
    """Generates plots that summarize model results."""

    def __init__(self, output_dir: str = FIGURES_DIR):
        self.output_dir = output_dir
        ensure_dir(self.output_dir)

    def _save(self, fig, filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        logger.info(f"Saved figure: {path}")

    def plot_model_comparison(self, results_df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.barplot(data=results_df, x="Model", y="RMSE", hue="Model", ax=axes[0], palette="Blues_d", legend=False)
        axes[0].set_title("Model Comparison — RMSE (lower is better)")
        axes[0].tick_params(axis="x", rotation=20)

        sns.barplot(data=results_df, x="Model", y="R2_Score", hue="Model", ax=axes[1], palette="Greens_d", legend=False)
        axes[1].set_title("Model Comparison — R² Score (higher is better)")
        axes[1].tick_params(axis="x", rotation=20)
        self._save(fig, "model_comparison.png")

    def plot_actual_vs_predicted(self, y_test, predictions, model_name: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_test, predictions, alpha=0.4, color="teal")
        limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
        ax.plot(limits, limits, "r--", label="Perfect Prediction")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title(f"Actual vs. Predicted Price — {model_name}")
        ax.legend()
        self._save(fig, "actual_vs_predicted.png")

    def plot_residuals(self, y_test, predictions, model_name: str) -> None:
        residuals = np.array(y_test) - np.array(predictions)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(residuals, kde=True, ax=ax, color="salmon")
        ax.axvline(0, color="black", linestyle="--")
        ax.set_title(f"Residual Distribution — {model_name}")
        ax.set_xlabel("Residual (Actual - Predicted)")
        self._save(fig, "residual_distribution.png")