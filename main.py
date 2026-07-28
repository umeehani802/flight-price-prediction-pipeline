"""
main.py
-------
Entry point that orchestrates the complete end-to-end Data Science
Pipeline for the Flight Price Prediction task:

    Data Loading -> Data Understanding -> Data Cleaning ->
    Feature Engineering -> EDA -> Preprocessing -> Feature Selection ->
    Train/Test Split -> Model Training -> Hyperparameter Tuning ->
    Model Evaluation -> Model Comparison -> Model Saving ->
    Result Visualization -> Documentation of Results

Run with:  python main.py
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_FILE, PROCESSED_DATA_FILE, BEST_MODEL_FILE,
    METRICS_REPORT_FILE, FINAL_SUMMARY_FILE,
    TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
    DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR,
)
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.eda import ExploratoryDataAnalyzer
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.visualizer import ResultVisualizer
from src.utils import get_logger, ensure_dir

logger = get_logger(__name__)


def run_pipeline() -> None:
    ensure_dir(DATA_PROCESSED_DIR)
    ensure_dir(MODELS_DIR)
    ensure_dir(REPORTS_DIR)

    # 1. DATA LOADING ----------------------------------------------------
    loader = DataLoader(RAW_DATA_FILE)
    raw_df = loader.load()
    loader.describe_dataset(raw_df)

    # 2. DATA CLEANING ----------------------------------------------------
    cleaned_df = (
        DataCleaner(raw_df)
        .drop_duplicates()
        .drop_missing_target()
        .fill_missing_categoricals()
        .remove_price_outliers()
        .get_data()
    )

    # 3. FEATURE ENGINEERING -----------------------------------------------
    engineered_df = (
        FeatureEngineer(cleaned_df)
        .simplify_route()
        .simplify_additional_info()
        .encode_total_stops()
        .parse_date_of_journey()
        .parse_time_column("Dep_Time", "Dep")
        .parse_time_column("Arrival_Time", "Arrival")
        .parse_duration()
        .get_data()
    )
    engineered_df.to_csv(PROCESSED_DATA_FILE, index=False)
    logger.info(f"Processed dataset saved to {PROCESSED_DATA_FILE}")

    # 4. EXPLORATORY DATA ANALYSIS ------------------------------------------
    ExploratoryDataAnalyzer(engineered_df).run_all()

    # 5. FEATURE SELECTION --------------------------------------------------
    selected_features = Preprocessor.select_correlated_features(engineered_df, threshold=0.0)
    logger.info(f"Selected numeric features by correlation: {selected_features}")

    # 6. DATA SPLITTING -----------------------------------------------------
    X = engineered_df.drop(columns=[TARGET_COLUMN])
    y = engineered_df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # 7. PREPROCESSING PIPELINE ----------------------------------------------
    preprocessor = Preprocessor(X_train).build_transformer()

    # 8. MODEL SELECTION, TRAINING & HYPERPARAMETER TUNING ---------------------
    trainer = ModelTrainer(preprocessor)
    trained_pipelines = trainer.train_all(X_train, y_train)

    # 9. MODEL EVALUATION & COMPARISON -----------------------------------------
    evaluator = ModelEvaluator(trained_pipelines)
    results_df = evaluator.evaluate_all(X_test, y_test)
    results_df.to_csv(METRICS_REPORT_FILE, index=False)
    logger.info(f"Model comparison report saved to {METRICS_REPORT_FILE}")

    best_model_name = evaluator.get_best_model_name()
    best_pipeline = trained_pipelines[best_model_name]
    logger.info(f"BEST MODEL: {best_model_name}")

    # 10. MODEL SAVING -----------------------------------------------------------
    joblib.dump(best_pipeline, BEST_MODEL_FILE)
    logger.info(f"Best model ({best_model_name}) saved to {BEST_MODEL_FILE}")

    # 11. RESULT VISUALIZATION -----------------------------------------------------
    visualizer = ResultVisualizer()
    visualizer.plot_model_comparison(results_df)
    best_predictions = best_pipeline.predict(X_test)
    visualizer.plot_actual_vs_predicted(y_test, best_predictions, best_model_name)
    visualizer.plot_residuals(y_test, best_predictions, best_model_name)

    # 12. DOCUMENTATION OF RESULTS -------------------------------------------------
    write_results_summary(results_df, best_model_name, X_train.shape, X_test.shape)

    logger.info("PIPELINE COMPLETE.")


def write_results_summary(results_df: pd.DataFrame, best_model_name: str,
                           train_shape: tuple, test_shape: tuple) -> None:
    """Write a human-readable markdown summary of the final results."""
    best_row = results_df[results_df["Model"] == best_model_name].iloc[0]
    content = f"""# Flight Price Prediction — Results Summary

## Dataset
- Training samples: {train_shape[0]}
- Test samples: {test_shape[0]}
- Features used: {train_shape[1]}

## Model Comparison

{results_df.to_markdown(index=False)}

## Best Model: **{best_model_name}**
- RMSE: {best_row['RMSE']:.2f}
- MAE: {best_row['MAE']:.2f}
- R2 Score: {best_row['R2_Score']:.4f}

## Artifacts Produced
- Trained model: `models/best_model.pkl`
- Processed dataset: `data/processed/flight_price_processed.csv`
- Evaluation report: `outputs/reports/model_comparison_report.csv`
- Figures: `outputs/figures/` (EDA plots, model comparison, actual vs predicted, residuals)
"""
    with open(FINAL_SUMMARY_FILE, "w") as f:
        f.write(content)
    logger.info(f"Results summary written to {FINAL_SUMMARY_FILE}")


if __name__ == "__main__":
    run_pipeline()