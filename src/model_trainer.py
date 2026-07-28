"""
model_trainer.py
------------------
Stages: Model Selection, Model Training, Hyperparameter Tuning.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from src.config import MODEL_PARAM_GRIDS, CV_FOLDS, SCORING_METRIC, RANDOM_STATE
from src.utils import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Trains and tunes multiple candidate regression models."""

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.candidate_models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(random_state=RANDOM_STATE),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
        }
        self.trained_pipelines = {}

    def _build_pipeline(self, model) -> Pipeline:
        """Combine the preprocessing step and the model into one Pipeline."""
        return Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("model", model),
        ])

    def train_all(self, X_train, y_train) -> dict:
        """
        Train every candidate model using GridSearchCV for hyperparameter
        tuning (where a grid is defined). Returns a dict of fitted pipelines.
        """
        for name, model in self.candidate_models.items():
            logger.info(f"----- Training {name} -----")
            pipeline = self._build_pipeline(model)
            param_grid = MODEL_PARAM_GRIDS.get(name, {})

            if param_grid:
                search = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    cv=CV_FOLDS,
                    scoring=SCORING_METRIC,
                    n_jobs=-1,
                )
                search.fit(X_train, y_train)
                best_pipeline = search.best_estimator_
                logger.info(f"{name} best params: {search.best_params_}")
                logger.info(f"{name} best CV score ({SCORING_METRIC}): {search.best_score_:.4f}")
            else:
                pipeline.fit(X_train, y_train)
                best_pipeline = pipeline
                logger.info(f"{name} trained with default parameters (no grid defined).")

            self.trained_pipelines[name] = best_pipeline

        return self.trained_pipelines