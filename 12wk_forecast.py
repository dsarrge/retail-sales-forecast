"""
Created on Mon Apr 21 12:05:56 2025
Title: Omni-Channel Retailer Sales Forecasting & Consumer Trends
Author: @dsarrge
"""

#%% === Imports ===
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta

#%% === Load Data and Model ===
df = pd.read_excel(r"cleaned_selltrhu.xlsx")
df = df.sort_values(["GEA SKU", "Customer", "Week End"])
df["Week End"] = pd.to_datetime(df["Week End"])

model = joblib.load(r"xgb_sellthru_model.pkl")

#%% === Feature List ===
features = [
    'week', 'month', 'quarter', 'year', 'weekofyear', 'weekday',
    'rolling_2','rolling_3', 'rolling_4','rolling_6', 'rolling_12',
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_6', 'lag_12',
    'Promo', 'Discount %', 'promo_lag_1', 'promo_lag_2', 'promo_lag_3'
]

#%% === Promo Feature Engineering ===
promo_df = pd.read_excel(r"promo_cal25.xlsx")

def promo_feature_engineering(df, promo_df):
    promo_df["Start Date"] = pd.to_datetime(promo_df["Start Date"])
    promo_df["End Date"] = pd.to_datetime(promo_df["End Date"])

    df["Promo"] = 0
    df["Discount %"] = 0.0

    for _, promo in promo_df.iterrows():
        mask = (
            (df["GEA SKU"] == promo["GEA SKU"]) &
            (df["Week End"] >= promo["Start Date"]) &
            (df["Week End"] <= promo["End Date"])
        )
        df.loc[mask, "Promo"] = 1
        df.loc[mask, "Discount %"] = (promo["MAP"] - promo["PMAP"]) / promo["MAP"]

    return df

df = promo_feature_engineering(df, promo_df)

#%% === Forecast Function ===
def forecast_future(df, model, features, forecast_weeks=12):
    forecast_df = df.copy()
    predictions = []

    for _ in range(forecast_weeks):
        next_date = forecast_df["Week End"].max() + timedelta(weeks=1)
        next_rows = []

        for (cust, sku), group in forecast_df.groupby(['Customer', 'GEA SKU']):
            latest_row = group[group["Week End"] == group["Week End"].max()].copy()
            if latest_row.empty:
                continue

            new_row = latest_row.copy()
            new_row["Week End"] = next_date
            new_row["week"] = next_date.isocalendar().week
            new_row["month"] = next_date.month
            new_row["quarter"] = (next_date.month - 1) // 3 + 1
            new_row["year"] = next_date.year
            new_row["weekofyear"] = next_date.isocalendar().week
            new_row["weekday"] = next_date.weekday()

            for lag in [1, 2, 3, 4, 6, 12]:
                lag_weeks = next_date - timedelta(weeks=lag)
                lag_value = group[group["Week End"] == lag_weeks]["WTD Net Sales U"]
                new_row[f"lag_{lag}"] = lag_value.values[0] if not lag_value.empty else 0

            for win in [2, 3, 4, 6, 12]:
                recent = group.sort_values("Week End").tail(win)["WTD Net Sales U"]
                new_row[f"rolling_{win}"] = recent.mean() if not recent.empty else 0

            for lag in [1, 2, 3]:
                lag_weeks = next_date - timedelta(weeks=lag)
                promo_val = group[group["Week End"] == lag_weeks]["Promo"]
                new_row[f"promo_lag_{lag}"] = promo_val.values[0] if not promo_val.empty else 0

            new_row["Promo"] = 0
            new_row["Discount %"] = 0.0

            next_rows.append(new_row[features + ["GEA SKU", "Customer", "Week End"]])

        if next_rows:
            next_df = pd.concat(next_rows, axis=0)
            X_future = next_df[features]
            y_pred_log = model.predict(X_future)
            y_pred = np.expm1(y_pred_log)
            next_df["WTD Net Sales U"] = y_pred
            forecast_df = pd.concat([forecast_df, next_df], axis=0)
            predictions.append(next_df[["GEA SKU", "Customer", "Week End", "WTD Net Sales U"]])

    forecast_result = pd.concat(predictions, axis=0).reset_index(drop=True)
    return forecast_result

#%% === Run Forecast ===
forecast_weeks = 12
forecast_result = forecast_future(df, model, features, forecast_weeks)

#%% === Format Excel ===

# Round forecast values
forecast_result["WTD Net Sales U"] = forecast_result["WTD Net Sales U"].round().astype(int)

# Calculate total forecast by Customer + SKU
totals = (
    forecast_result
    .groupby(["Customer", "GEA SKU"])["WTD Net Sales U"]
    .sum()
    .reset_index()
    .rename(columns={"WTD Net Sales U": "Total Forecast"})
)

# Pivot forecast weeks across columns
pivot_df = (
    forecast_result
    .pivot_table(index=["Customer", "GEA SKU"], columns="Week End", values="WTD Net Sales U")
    .reset_index()
)

# Merge totals
pivot_df = pivot_df.merge(totals, on=["Customer", "GEA SKU"])

# Clean column headers
pivot_df.columns.name = None
pivot_df.columns = [
    col.strftime("%m-%d-%y") if isinstance(col, pd.Timestamp) else col
    for col in pivot_df.columns
]

# Sort by Customer and descending SKU
pivot_df = pivot_df.sort_values(["Customer", "GEA SKU"], ascending=[True, False])

#%% === Save to Excel ===
output_path = r"C:\Users\dsarr\OneDrive\Desktop\Python\forecast_output_2025.xlsx"
pivot_df.to_excel(output_path, index=False)
print(f"Forecast saved to: {output_path}")
