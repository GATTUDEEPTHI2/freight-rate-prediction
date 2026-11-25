
# Freight Rate Prediction

## Project Overview
This project predicts freight load rates using machine learning. A CatBoost regression model is trained on historical freight data and used to generate predictions for the supplied validation dataset and December 2025 chart inputs.

## Dataset
- Training data: 48,000 rows (January–October 2025)
- Validation data: 12,000 rows (November–December 2025)
- December chart inputs: 31 rows

## Methodology
1. Inspected the dataset, column types, missing values, and unusual weight values.
2. Selected features available across the training, validation, and December datasets.
3. Extracted month, day, day of week, and day of year from the date.
4. Used a chronological split:
   - Training: January–August 2025
   - Internal validation: September–October 2025
5. Trained a CatBoost Regressor with categorical features for pickup, delivery, and equipment.
6. Evaluated the model using MAE, RMSE, and R-squared.
7. Retrained the model using all labeled training data and generated the required predictions.

## Internal Validation Results

| Metric | Result |
|---|---:|
| MAE | 122.86 |
| RMSE | 634.10 |
| R-squared | 0.8273 |

These metrics are from the internal September–October time-based split. They are not the official November–December assessment results.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- Matplotlib
- Seaborn

## Project Files
- `train_model.py` – Model training and prediction generation
- `score.py` – Official assessment validation script
- `requirements.txt` – Python dependencies
- `validation_predictions.csv` – Final validation predictions
- `data/december_chart_inputs.csv` – December predictions
- `scorer_results/candidate_december.png` – December prediction chart
- `freight_rate_model.cbm` – Saved trained model

## How to Run

### 1. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Train the model and generate predictions
```bash
python train_model.py
```

### 3. Run the assessment scorer
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Scorer Results
The official scorer successfully validated:
- 12,000 validation predictions
- 31 December predictions

It generated `scorer_results/candidate_december.png`.

## Author
Machine Learning Assessment Project
