
import pandas as pd

dates = pd.date_range(start="2025-12-01", end="2025-12-31", freq="D")

df = pd.DataFrame({
    "pickup": ["Lexington"] * 31,
    "delivery": ["Fort Wayne"] * 31,
    "distance": [360] * 31,
    "equipment": ["Dry Van"] * 31,
    "weight": [32000] * 31,
    "date": dates.strftime("%Y-%m-%d"),
    "predicted_rate": [None] * 31
})

df.to_csv("data/december_chart_inputs.csv", index=False)

print("December CSV created successfully!")
print(df.head())
print("\nTotal rows:", len(df))
