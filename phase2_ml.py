"""
Phase 2: ML-Based Environmental Prediction Module
Project: Integrated Smart City Environment Monitoring & Visualization System

Objective:
- Predict numerical AQI using environmental sensor data
- Compare baseline, ensemble, and advanced ML models
- Export predictions for Power BI visualization
"""

# =========================================================
# 1. Import Required Libraries
# =========================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor


# =========================================================
# 2. File Paths & Directory Setup
# =========================================================

RAW_DATA_PATH = "data/raw/environment_data.csv"
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "predicted_aqi.csv")

# Create processed directory if not exists
os.makedirs(PROCESSED_DIR, exist_ok=True)


# =========================================================
# 3. Load Dataset
# =========================================================

df = pd.read_csv(RAW_DATA_PATH)

print("\n✅ Dataset Loaded Successfully")
print(df.head())
print("\nDataset Shape:", df.shape)


# =========================================================
# 4. Encode Categorical Variables
# =========================================================

# One-hot encode 'zone'
df = pd.get_dummies(df, columns=["zone"], drop_first=True)


# =========================================================
# 5. Target Variable Engineering
# =========================================================
# Convert AQI category to numerical AQI values

aqi_mapping = {
    "Good": 50,
    "Moderate": 100,
    "Bad": 200
}

df["AQI"] = df["air_quality_level"].map(aqi_mapping)

# Drop original label column
df.drop(columns=["air_quality_level"], inplace=True)


# =========================================================
# 6. Define Features (X) and Target (y)
# =========================================================

X = df.drop(columns=["AQI"])
y = df["AQI"]

print("\nFeatures:", list(X.columns))
print("Target: AQI")


# =========================================================
# 7. Feature Scaling
# =========================================================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# =========================================================
# 8. Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])


# =========================================================
# 9. Model Evaluation Function
# =========================================================

def evaluate_model(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 {model_name} Performance")
    print("-" * 35)
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.2f}")

    return rmse, r2


# =========================================================
# 10. Baseline Model — Linear Regression
# =========================================================

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
lr_rmse, lr_r2 = evaluate_model(
    "Linear Regression", y_test, lr_predictions
)


# =========================================================
# 11. Ensemble Model — Random Forest
# =========================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_rmse, rf_r2 = evaluate_model(
    "Random Forest Regressor", y_test, rf_predictions
)


# =========================================================
# 12. Advanced Model — XGBoost
# =========================================================

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective="reg:squarederror",
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)
xgb_rmse, xgb_r2 = evaluate_model(
    "XGBoost Regressor", y_test, xgb_predictions
)


# =========================================================
# 13. Model Comparison Summary
# =========================================================

comparison_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "RMSE": [lr_rmse, rf_rmse, xgb_rmse],
    "R2_Score": [lr_r2, rf_r2, xgb_r2]
})

print("\n📈 Model Comparison Summary")
print(comparison_df)


# =========================================================
# 14. Save Predictions for Power BI
# =========================================================

# Predict AQI for entire dataset using best model (XGBoost)
df_results = X.copy()
df_results["Actual_AQI"] = y.values
df_results["Predicted_AQI"] = xgb_model.predict(X_scaled)

df_results.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Predicted AQI saved successfully!")
print(f"📁 Output File: {OUTPUT_FILE}")


# =========================================================
# 15. Phase 2 Completed
# =========================================================

print("\n🎯 Phase 2 ML-Based AQI Prediction Completed Successfully")
