# ================================================
# Phase 3: AI-Based Anomaly Detection Module
# Project: Integrated Smart City Environment Monitoring System
# Method: Isolation Forest
# ================================================

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# -------------------------------
# 1. File Paths
# -------------------------------
INPUT_FILE = "data/processed/predicted_aqi.csv"
OUTPUT_FILE = "data/processed/anomaly_detection_results.csv"

# Ensure output directory exists
os.makedirs("data/processed", exist_ok=True)

# -------------------------------
# 2. Load Dataset
# -------------------------------
df = pd.read_csv(INPUT_FILE)

print("\n✅ Dataset Loaded Successfully")
print(df.head())
print("\nColumns:", list(df.columns))

# -------------------------------
# 3. Column Name Fix (CRITICAL)
# -------------------------------
# Standardize column naming for ML pipeline
df.rename(columns={
    "Predicted_AQI": "predicted_aqi",
    "Actual_AQI": "actual_aqi"
}, inplace=True)

# -------------------------------
# 4. Feature Selection
# -------------------------------
features = [
    "temperature",
    "humidity",
    "pm2_5",
    "pm10",
    "no2",
    "predicted_aqi"
]

X = df[features]

# -------------------------------
# 5. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 6. Isolation Forest Model
# -------------------------------
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.2,        # 20% anomaly assumption (research-safe)
    random_state=42
)

iso_forest.fit(X_scaled)

# -------------------------------
# 7. Anomaly Prediction
# -------------------------------
df["anomaly_label"] = iso_forest.predict(X_scaled)
df["anomaly_score"] = iso_forest.decision_function(X_scaled)

# Convert labels: -1 → Anomaly, 1 → Normal
df["anomaly_status"] = df["anomaly_label"].map({
    -1: "Anomalous",
     1: "Normal"
})

# -------------------------------
# 8. Risk Level Assignment
# -------------------------------
def risk_level(score):
    if score < -0.15:
        return "High"
    elif score < 0:
        return "Moderate"
    else:
        return "Low"

df["risk_level"] = df["anomaly_score"].apply(risk_level)

# -------------------------------
# 9. Save Results
# -------------------------------
df.to_csv(OUTPUT_FILE, index=False)

# -------------------------------
# 10. Summary Output
# -------------------------------
print("\n🚨 Anomaly Detection Summary")
print(df[["predicted_aqi", "anomaly_score", "anomaly_status", "risk_level"]])

print("\n📁 Output saved to:", OUTPUT_FILE)
print("\n🎯 Phase 3: AI-Based Anomaly Detection COMPLETED SUCCESSFULLY")
