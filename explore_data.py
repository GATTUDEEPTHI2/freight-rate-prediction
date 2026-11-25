
import pandas as pd
import os

# Check whether training file exists
file_path = "data/train_test.csv"

if not os.path.exists(file_path):
    print("Training file not found!")
    exit()

# Load dataset
df = pd.read_csv(file_path)

print("\n========== DATASET OVERVIEW ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head().to_string())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nNumerical Summary:")
print(df.describe().to_string())

print("\n========== INSPECTION COMPLETE ==========")
