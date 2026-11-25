
import pandas as pd
import os

train = pd.read_csv("data/train_test.csv")
validation = pd.read_csv("data/validation.csv")
template = pd.read_csv("data/validation_predictions_template.csv")
december = pd.read_csv("data/december_chart_inputs.csv")

print("\n--- TRAINING DATA ---")
print("Shape:", train.shape)
print("Date range:", train["date"].min(), "to", train["date"].max())

print("\n--- VALIDATION DATA ---")
print("Shape:", validation.shape)
print("Columns:", validation.columns.tolist())
print("Date range:", validation["date"].min(), "to", validation["date"].max())

print("\n--- PREDICTION TEMPLATE ---")
print("Shape:", template.shape)
print("Columns:", template.columns.tolist())
print(template.head())

print("\n--- EQUIPMENT TYPES ---")
print(train["equipment"].value_counts())

print("\n--- NEGATIVE WEIGHTS ---")
print("Count:", (train["weight"] < 0).sum())

print("\n--- DECEMBER INPUT ---")
print("Shape:", december.shape)
print(december.head())
print("Date range:", december["date"].min(), "to", december["date"].max())

print("\n--- INSPECTION COMPLETE ---")
