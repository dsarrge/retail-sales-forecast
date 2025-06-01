# Omni-Channel Retailer Sales Forecasting & Consumer Trends

This project provides an end-to-end pipeline for forecasting retail SKU-level sales across multiple national customers using historical sell-thru data, promo calendars, and machine learning.

## 🔍 Overview

This repository covers:
- Data ingestion and cleansing across 4 major retailers
- Feature engineering with time-series lags, rolling windows, and promo influence
- Training an XGBoost regression model
- Generating 12-week ahead forecasts
- Outputting Excel-ready forecast reports

---

## 📁 Project Structure

- `masterfile_writing.py`: Cleans and merges raw Excel files from different retailers into a unified, enriched dataset (`cleaned_selltrhu.xlsx`).
- `xgb_pipeline.py`: Trains an XGBoost model on log-transformed sales using engineered time-series features and discount-based promo features.
- `12wk_forecast.py`: Applies the trained model to generate a 12-week rolling forecast by customer and SKU, and exports results as a pivoted Excel file.

---

## 📊 Features Engineered

- **Temporal:** `week`, `month`, `quarter`, `weekday`, `weekofyear`
- **Lag Features:** `lag_1`, `lag_2`, ..., `lag_12`
- **Rolling Means:** `rolling_2`, `rolling_3`, ..., `rolling_12`
- **Promo History:** `Promo`, `Discount %`, `promo_lag_1`, `promo_lag_2`, `promo_lag_3`

---

## 🔁 Workflow

1. **Clean & Merge**  
   `masterfile_writing.py` loads each retailer's weekly data, harmonizes sales columns, and exports a unified DataFrame.

2. **Model Training**  
   `xgb_pipeline.py` builds features, runs grid search with cross-validation on XGBoost, evaluates residuals, and saves the best model.

3. **Forecasting**  
   `12wk_forecast.py` loads the model, applies time-forward logic for rolling predictions, and formats a customer-SKU forecast Excel output.

---

## 📦 Requirements

- Python 3.8+
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib
- joblib
- openpyxl

Install all with:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib joblib openpyxl
```

---

## 📈 Output

- `cleaned_selltrhu.xlsx`: Unified dataset
- `xgb_sellthru_model.pkl`: Trained model file
- `forecast_output_2025.xlsx`: Forecast result with totals and week-by-week breakdown

---

## 💡 Future Enhancements

- **API Integration**: Connect to SPS Commerce Analytics for automated data ingestion and retail POS sync.
- **Inventory Analysis**: Extend forecast logic with inventory data to compute Weeks of Supply (WOS) and stock risk levels.
- **Interactive Dashboard**: Build a Plotly Dash or Streamlit dashboard for SKU-level drilldowns, trends, and forecast visualizations.
- **Model Expansion**: Explore LightGBM, Prophet, or hybrid models to benchmark accuracy vs. XGBoost.

---

## 🧠 Author

**Dylan Sarrge**  
MSBA Candidate | Retail Forecasting & Data Science
