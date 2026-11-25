
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load datasets
train = pd.read_csv("data/train_test.csv")
validation = pd.read_csv("data/validation.csv")
template = pd.read_csv("data/validation_predictions_template.csv")
december = pd.read_csv("data/december_chart_inputs.csv")

# 2. Features available in all datasets
FEATURES = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "month",
    "day",
    "dayofweek",
    "dayofyear"
]

CAT_FEATURES = ["pickup", "delivery", "equipment"]

def prepare_features(df):
    data = df.copy()

    dates = pd.to_datetime(data["date"], errors="coerce")

    if dates.isna().any():
        raise ValueError("Some dates could not be interpreted.")

    data["month"] = dates.dt.month
    data["day"] = dates.dt.day
    data["dayofweek"] = dates.dt.dayofweek
    data["dayofyear"] = dates.dt.dayofyear

    for col in CAT_FEATURES:
        data[col] = data[col].fillna("Unknown").astype(str)

    return data[FEATURES]


# 3. Prepare training features and target
X = prepare_features(train)
y = train["posted_rate"]

if y.isna().any():
    raise ValueError("The target column contains missing values.")

# 4. Time-based train/validation split
dates = pd.to_datetime(train["date"])

train_mask = dates < "2025-09-01"
valid_mask = dates >= "2025-09-01"

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_valid = X.loc[valid_mask]
y_valid = y.loc[valid_mask]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))

# 5. Train CatBoost model
model = CatBoostRegressor(
    iterations=1500,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    eval_metric="MAE",
    random_seed=42,
    verbose=200,
    allow_writing_files=False
)

model.fit(
    X_train,
    y_train,
    cat_features=CAT_FEATURES,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100,
    verbose=200
)

# 6. Evaluate the model
valid_predictions = model.predict(X_valid)

mae = mean_absolute_error(y_valid, valid_predictions)
rmse = np.sqrt(mean_squared_error(y_valid, valid_predictions))
r2 = r2_score(y_valid, valid_predictions)

print("\n========== MODEL EVALUATION ==========")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 7. Train final model using all available labeled data
best_iteration = model.get_best_iteration()
final_iterations = max(300, best_iteration + 1)

print("\nFinal model iterations:", final_iterations)

final_model = CatBoostRegressor(
    iterations=final_iterations,
    depth=8,
    learning_rate=0.05,
    loss_function="RMSE",
    random_seed=42,
    verbose=200,
    allow_writing_files=False
)

final_X = prepare_features(train)

final_model.fit(
    final_X,
    y,
    cat_features=CAT_FEATURES
)

final_model.save_model("freight_rate_model.cbm")

# 8. Predict the 12,000 validation rows
validation_X = prepare_features(validation)
validation_rates = final_model.predict(validation_X)

prediction_map = dict(
    zip(validation["load_id"], validation_rates)
)

output = template.copy()
output["predicted_rate"] = output["load_id"].map(prediction_map)

if output["predicted_rate"].isna().any():
    raise ValueError("Some validation IDs did not receive predictions.")

output.to_csv("validation_predictions.csv", index=False)

# 9. Predict all 31 December rows
december_X = prepare_features(december)
december["predicted_rate"] = final_model.predict(december_X)

december.to_csv("data/december_chart_inputs.csv", index=False)

print("\n========== PREDICTIONS COMPLETE ==========")
print("Validation predictions:", len(output))
print("December predictions:", len(december))
print("\nValidation file saved: validation_predictions.csv")
print("December file updated: data/december_chart_inputs.csv")
print("Model saved: freight_rate_model.cbm")
print("\nSample validation predictions:")
print(output.head())

print("\nSample December predictions:")
print(december.head())
