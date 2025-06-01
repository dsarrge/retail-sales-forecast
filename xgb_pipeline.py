"""
Created on Mon Apr 21 08:07:20 2025
Title: Omni-Channel Retailer Sales Forecasting & Consumer Trends
Author: @dsarrge
"""
#%% === Load Packages ===

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#%%  === Load Cleaned Data ===

df = pd.read_excel(r"cleaned_selltrhu.xlsx")
df = df.sort_values(["GEA SKU", "Customer", "Week End"])

promo_df = pd.read_excel(r"promo_cal25.xlsx")
#%% === Promo Feature Engineering ===

def promo_feature_engineering(df, promo_df):
    promo_df["Start Date"] = pd.to_datetime(promo_df["Start Date"])
    promo_df["End Date"] = pd.to_datetime(promo_df["End Date"])

    df["Promo"] = 0
    df["Discount %"] = 0.0
#    df["Super Promo"] = 0

    for _, promo in promo_df.iterrows():
        mask = (
            (df["GEA SKU"] == promo["GEA SKU"]) &
            (df["Week End"] >= promo["Start Date"]) &
            (df["Week End"] <= promo["End Date"])
        )
        df.loc[mask, "Promo"] = 1
        df.loc[mask, "Discount %"] = (promo["MAP"] - promo["PMAP"]) / promo["MAP"]
#        df.loc[mask & (promo["Super PMAP"] < promo["PMAP"]), "Super Promo"] = 1
    return df

df = promo_feature_engineering(df, promo_df)

#%%  === Feature Engineering ===

def create_features(df):
    df = df.copy()
    df["week"] = df["Week End"].dt.isocalendar().week
    df["month"] = df["Week End"].dt.month
    df["quarter"] = df["Week End"].dt.quarter
    df["year"] = df["Week End"].dt.year
    df["weekofyear"] = df["Week End"].dt.isocalendar().week
    df["weekday"] = df["Week End"].dt.weekday

    for lag in [1, 2, 3, 4, 6, 12]:
        df[f"lag_{lag}"] = df.groupby(["GEA SKU", "Customer"])["WTD Net Sales U"].shift(lag)

    df["rolling_2"] = df.groupby(["GEA SKU", "Customer"])["WTD Net Sales U"].shift(1).rolling(2).mean()
    df["rolling_3"] = df.groupby(["GEA SKU", "Customer"])["WTD Net Sales U"].shift(1).rolling(3).mean()
    df["rolling_4"] = df.groupby(["GEA SKU", "Customer"])["WTD Net Sales U"].shift(1).rolling(4).mean()
    df["rolling_6"] = df.groupby(["GEA SKU", "Customer"])["WTD Net Sales U"].shift(1).rolling(6).mean()
    df["rolling_12"] = df.groupby(["GEA SKU", "Customer"])["WTD Net Sales U"].shift(1).rolling(12).mean()

    for lag in [1, 2, 3]:
        df[f"promo_lag_{lag}"] = df.groupby(["GEA SKU", "Customer"])["Promo"].shift(lag)

    return df

df = create_features(df)
df = df.dropna()

#%% === ML Modeling ===

features = [
    'week', 'month', 'quarter', 'year', 'weekofyear', 'weekday',
    'rolling_2','rolling_3', 'rolling_4','rolling_6', 'rolling_12',
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_6', 'lag_12',
    'Promo', 'Discount %'] + [col for col in df.columns if col.startswith("promo_lag")]

target = "WTD Net Sales U"

cutoff_date = df["Week End"].max() - pd.Timedelta(weeks=8)
train = df[df["Week End"] <= cutoff_date]
test = df[df["Week End"] > cutoff_date]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# Log Transformation
y_train_log = np.log1p(y_train)

#%% === Define XGBoost Model and Hyperparameter Grid ===
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Grid Search
grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train_log)

# Predict and Reverse Log
best_model = grid_search.best_estimator_
y_pred_log = best_model.predict(X_test)
y_pred = np.expm1(y_pred_log) # <-- reverse log1p

#%%  === Evaluation ===

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("XGBoost RMSE (log-transformed):", round(rmse, 2))

# === Residual Analysis ===
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='purple', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values (XGBoost, Log Transformed)')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Feature Analysis ===
feature_importance = best_model.feature_importances_
features = X_test.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for XGBoost Model')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% === Save Model ===

joblib.dump(best_model, "xgb_sellthru_model.pkl")
