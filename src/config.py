"""
config.py
----------
Centralized configuration for the Flight Price Prediction pipeline.
All paths, constants, and model settings live here so the rest of the
codebase never hard-codes a path or a magic number.
"""

import os

# ---------------------------------------------------------------------------
# Base directory paths (resolved relative to this file so the project
# runs correctly regardless of the working directory it's launched from)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")

# Name of the raw CSV file. Replace this file with your real Kaggle
# "Flight Price Prediction" CSV — the schema must match the columns below.
RAW_DATA_FILE = os.path.join(DATA_RAW_DIR, "flight_price.xlsx")
PROCESSED_DATA_FILE = os.path.join(DATA_PROCESSED_DIR, "flight_price_processed.csv")
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "best_model.pkl")
METRICS_REPORT_FILE = os.path.join(REPORTS_DIR, "model_comparison_report.csv")
FINAL_SUMMARY_FILE = os.path.join(REPORTS_DIR, "final_results_summary.md")

# ---------------------------------------------------------------------------
# Dataset schema (as provided in the Kaggle Flight Price Prediction dataset)
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Price"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING_METRIC = "neg_root_mean_squared_error"

# ---------------------------------------------------------------------------
# Hyperparameter search grids used during tuning
# ---------------------------------------------------------------------------
MODEL_PARAM_GRIDS = {
    "LinearRegression": {},
    "RandomForestRegressor": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [8, 12, None],
        "model__min_samples_split": [2, 5],
    },
    "GradientBoostingRegressor": {
        "model__n_estimators": [200, 300],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3, 4],
    },
}