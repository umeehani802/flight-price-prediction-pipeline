markdown
# ✈️ Flight Price Prediction — End-to-End Data Science Pipeline

A complete, production-oriented, object-oriented Data Science pipeline built for
**Task 1 (Internship)**. It takes raw flight-booking data and produces a trained,
saved regression model that predicts ticket **Price**, along with full EDA,
evaluation, comparison, and visualization artifacts.

---

## 1. Project Overview

This project implements an end-to-end AI/ML workflow:

Data Loading -> Data Understanding -> Data Cleaning -> Feature Engineering ->
EDA -> Preprocessing -> Feature Selection -> Train/Test Split ->
Model Selection -> Model Training -> Hyperparameter Tuning ->
Model Evaluation -> Model Comparison -> Model Saving ->
Prediction/Inference -> Result Visualization -> Documentation


The code follows an **OOP architecture** — each pipeline stage is its own class
in `src/`, and `main.py` orchestrates them in order via a single `run_pipeline()`
function. This makes the code modular, testable, and easy to extend.

---

## 2. Dataset Information

**Source:** Kaggle — Flight Price Prediction (Indian domestic flights dataset)

| Column | Description |
|---|---|
| `Airline` | Name of the airline |
| `Date_of_Journey` | Date of the flight |
| `Source` | Departure city |
| `Destination` | Arrival city |
| `Route` | Flight route (e.g. `BLR → DEL`) |
| `Dep_Time` | Departure time |
| `Arrival_Time` | Arrival time |
| `Duration` | Total flight duration (e.g. `"2h 50m"`) |
| `Total_Stops` | Number of stops (e.g. `"non-stop"`, `"1 stop"`) |
| `Additional_Info` | Extra info (meal, baggage, layover, etc.) |
| `Price` | **Target variable** — ticket price |

- **Rows:** 10,683 (raw) → 10,264 after cleaning
- **Format:** `.xlsx`
- **Task type:** Regression (predicting a continuous price)

---

## 3. Project Structure

PythonProject/
├── data/
│ ├── raw/ # flight_price.xlsx (raw input)
│ └── processed/ # flight_price_processed.csv (cleaned + engineered)
├── models/
│ └── best_model.pkl # Saved best-performing trained pipeline
├── outputs/
│ ├── figures/ # All EDA & result visualizations (.png)
│ └── reports/ # Model comparison CSV + results summary (.md)
├── src/
│ ├── config.py # All paths, constants, hyperparameter grids
│ ├── utils.py # Logging & helper functions
│ ├── data_loader.py # Data Collection / Loading / Understanding
│ ├── data_cleaner.py # Data Cleaning
│ ├── feature_engineering.py # Feature Engineering
│ ├── eda.py # Exploratory Data Analysis
│ ├── preprocessor.py # Preprocessing + Feature Selection
│ ├── model_trainer.py # Model Selection, Training, Hyperparameter Tuning
│ ├── model_evaluator.py # Model Evaluation & Comparison
│ ├── visualizer.py # Result Visualization
│ └── predictor.py # Prediction / Inference on new data
├── tests/
│ ├── test_data_cleaner.py
│ └── test_feature_engineering.py
├── .github/workflows/ci.yml # CI pipeline (runs unit tests on every push)
├── main.py # Orchestrates the full pipeline
├── requirements.txt
├── .gitignore
└── README.md


---

## 4. Installation & Virtual Environment Setup

### Step 1 — Clone the repository
```bash
git clone <your-repo-url>
cd PythonProject
```

### Step 2 — Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Dependencies

pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, tabulate, pytest, openpyxl

All versions are listed in `requirements.txt`.

---

## 6. How to Run the Project

### Run the full pipeline
```bash
python main.py
```
This runs every stage — loading, cleaning, feature engineering, EDA,
preprocessing, training, tuning, evaluation, saving, visualization, and
report generation — with structured logging throughout.

> **Note:** Random Forest and Gradient Boosting hyperparameter tuning via
> `GridSearchCV` takes roughly 15–20 minutes on a standard laptop. This is
> expected — not a freeze.

### Run unit tests
```bash
pytest tests/ -v
```

### Run inference on new data
```python
import pandas as pd
from src.predictor import FlightPricePredictor

new_flights = pd.DataFrame({
    "Airline": ["IndiGo"],
    "Date_of_Journey": ["15/06/2019"],
    "Source": ["Delhi"],
    "Destination": ["Cochin"],
    "Route": ["DEL → COK"],
    "Dep_Time": ["09:00"],
    "Arrival_Time": ["13:15"],
    "Duration": ["4h 15m"],
    "Total_Stops": ["1 stop"],
    "Additional_Info": ["No info"],
})

predictor = FlightPricePredictor()
print(predictor.predict(new_flights))
```

---

## 7. CI/CD

`.github/workflows/ci.yml` runs automatically on every push/PR to `main`:
1. Installs dependencies
2. Runs `pytest tests/ -v`

Model training is intentionally excluded from CI due to its runtime
(~15–20 minutes for the full grid search), keeping CI fast and focused on
code correctness.

---

## 8. Results

### Data Cleaning Summary
- Raw rows: 10,683 → Cleaned rows: 10,264
- Removed 220 duplicate rows
- Filled 2 missing values (Route, Total_Stops) using mode imputation
- Removed 199 price outlier rows (kept the 1st–99th percentile range: ₹2,227–₹22,270)

### Feature Engineering
- `Duration` → parsed into `Duration_Minutes`
- `Date_of_Journey` → split into `Journey_Day`, `Journey_Month`, `Journey_DayOfWeek`
- `Dep_Time` / `Arrival_Time` → split into hour/minute (handles next-day arrival formats like `"01:10 23 Mar"`)
- `Total_Stops` → encoded as an ordinal integer (non-stop=0, 1 stop=1, ...)
- `Route` and `Additional_Info` → rare categories grouped into `"Other"`

### Feature Selection (correlation with Price)
| Feature | Correlation |
|---|---|
| Total_Stops | 0.660 |
| Duration_Minutes | 0.562 |
| Journey_Day | 0.123 |
| Arrival_Minute | 0.101 |
| Journey_Month | 0.059 |

### Model Comparison (Test Set)

| Model | RMSE | MAE | R2 Score |
|---|---|---|---|
| **Random Forest Regressor** | **1109.49** | **584.28** | **0.9216** |
| Gradient Boosting Regressor | 1218.50 | 818.42 | 0.9055 |
| Linear Regression | 2088.80 | 1536.49 | 0.7223 |

**Best model: Random Forest Regressor**, tuned via `GridSearchCV` (5-fold CV):
- `n_estimators=400`, `max_depth=None`, `min_samples_split=5`
- Saved to `models/best_model.pkl`

Tree-based models substantially outperform Linear Regression, confirming
non-linear relationships between price and features like stops, duration,
and route — consistent with the EDA findings below.

### Output Artifacts
- `outputs/figures/price_distribution.png` — target variable distribution
- `outputs/figures/flight_count_by_airline.png` — flight volume per airline
- `outputs/figures/average_price_by_airline.png` — price by airline (bar chart)
- `outputs/figures/average_price_by_stops.png` — price by number of stops (bar chart)
- `outputs/figures/flight_count_by_source.png` — flight volume by source city
- `outputs/figures/price_by_airline_boxplot.png` — price spread per airline
- `outputs/figures/price_by_route_boxplot.png` — price spread per route
- `outputs/figures/price_vs_duration.png` — price vs. flight duration scatter
- `outputs/figures/correlation_heatmap.png` — numeric feature correlations
- `outputs/figures/model_comparison.png` — RMSE & R² across models
- `outputs/figures/actual_vs_predicted.png` — best model fit quality
- `outputs/figures/residual_distribution.png` — prediction error distribution
- `outputs/reports/model_comparison_report.csv` — raw metrics table
- `outputs/reports/final_results_summary.md` — auto-generated results doc

### Key Insights from EDA
- Premium/legacy airlines (Jet Airways, Multiple carriers Premium economy)
  command the highest average prices; budget carriers (Trujet, SpiceJet)
  the lowest.
- Price increases consistently with number of stops — non-stop flights are
  cheapest, 2–3 stop flights are most expensive.
- Flight duration and total stops are the two strongest numeric predictors
  of price, aligning with the model's feature importance.

---

## 9. Key Engineering Decisions

- **Preprocessing** uses `StandardScaler` for numeric features and
  `OneHotEncoder` for categoricals, combined via a `ColumnTransformer`
  inside a single `sklearn.Pipeline` — preventing data leakage and keeping
  train/inference behavior consistent.
- **Model selection**: Linear Regression (baseline), Random Forest, Gradient
  Boosting — tuned via `GridSearchCV` with 5-fold cross-validation.
- **Best model** chosen by lowest RMSE on a held-out test set (20% split).
- **Inference module** (`predictor.py`) re-applies the exact same feature
  engineering steps used in training, ensuring consistent transformations
  on new/unseen data.



