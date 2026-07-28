# Flight Price Prediction — Results Summary

## Dataset
- Training samples: 8211
- Test samples: 2053
- Features used: 14

## Model Comparison

| Model                     |    RMSE |      MAE |   R2_Score |
|:--------------------------|--------:|---------:|-----------:|
| RandomForestRegressor     | 1109.49 |  584.28  |   0.921648 |
| GradientBoostingRegressor | 1218.5  |  818.419 |   0.905494 |
| LinearRegression          | 2088.8  | 1536.49  |   0.722286 |

## Best Model: **RandomForestRegressor**
- RMSE: 1109.49
- MAE: 584.28
- R2 Score: 0.9216

## Artifacts Produced
- Trained model: `models/best_model.pkl`
- Processed dataset: `data/processed/flight_price_processed.csv`
- Evaluation report: `outputs/reports/model_comparison_report.csv`
- Figures: `outputs/figures/` (EDA plots, model comparison, actual vs predicted, residuals)
