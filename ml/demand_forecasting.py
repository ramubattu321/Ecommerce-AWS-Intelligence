"""
E-Commerce Intelligence Platform — Demand Forecasting ML Model
==============================================================
Models: Random Forest, Gradient Boosting, XGBoost
Target: Predict monthly order volume per state
Features: Temporal features, RFM signals, payment patterns

Run: python ml/demand_forecasting.py
"""

import pandas as pd
import numpy as np
import boto3, os, io, joblib, logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

AWS_BUCKET   = os.getenv("AWS_BUCKET", "ecommerce-intelligence-ramu")
AWS_REGION   = os.getenv("AWS_REGION", "us-west-2")
MODEL_PATH   = Path("ml/models")
MODEL_PATH.mkdir(exist_ok=True)


# ── LOAD DATA FROM S3 ─────────────────────────────────────────────────────────
def load_from_s3(key):
    s3  = boto3.client("s3", region_name=AWS_REGION,
                        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def build_features(master: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to monthly state-level demand and engineer features.
    """
    master["order_purchase_timestamp"] = pd.to_datetime(
        master["order_purchase_timestamp"], errors="coerce")
    master["year"]  = master["order_purchase_timestamp"].dt.year
    master["month"] = master["order_purchase_timestamp"].dt.month

    agg = master.groupby(["year","month","customer_state"]).agg(
        order_count    = ("order_id", "count"),
        total_revenue  = ("total_payment", "sum"),
        avg_order_val  = ("total_payment", "mean"),
        avg_review     = ("review_score", "mean"),
        late_rate      = ("is_late", "mean"),
        avg_delivery   = ("delivery_days", "mean"),
        num_customers  = ("customer_id", "nunique"),
    ).reset_index()

    # Lag features (previous month demand per state)
    agg = agg.sort_values(["customer_state","year","month"])
    agg["lag_1_orders"] = agg.groupby("customer_state")["order_count"].shift(1)
    agg["lag_2_orders"] = agg.groupby("customer_state")["order_count"].shift(2)
    agg["lag_3_orders"] = agg.groupby("customer_state")["order_count"].shift(3)
    agg["rolling_3m"]   = agg.groupby("customer_state")["order_count"]\
                             .transform(lambda x: x.shift(1).rolling(3).mean())

    # Season
    agg["quarter"]     = ((agg["month"] - 1) // 3 + 1)
    agg["is_q4"]       = (agg["quarter"] == 4).astype(int)
    agg["month_sin"]   = np.sin(2 * np.pi * agg["month"] / 12)
    agg["month_cos"]   = np.cos(2 * np.pi * agg["month"] / 12)

    # Encode state
    le = LabelEncoder()
    agg["state_enc"] = le.fit_transform(agg["customer_state"])
    joblib.dump(le, MODEL_PATH / "state_encoder.pkl")

    agg = agg.dropna()
    log.info(f"Feature matrix: {agg.shape}")
    return agg


# ── TRAIN & EVALUATE ──────────────────────────────────────────────────────────
def train_models(df: pd.DataFrame):
    """Train multiple models and compare performance."""
    FEATURES = [
        "year","month","quarter","is_q4","month_sin","month_cos",
        "state_enc","lag_1_orders","lag_2_orders","lag_3_orders","rolling_3m",
        "avg_order_val","avg_review","late_rate","avg_delivery","num_customers"
    ]
    TARGET = "order_count"

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression":    LinearRegression(),
        "Random Forest":        RandomForestRegressor(n_estimators=200, max_depth=10,
                                                      random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                          learning_rate=0.05, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        cv    = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
        results[name] = {"MAE":mae, "RMSE":rmse, "R2":r2, "CV_R2":cv,
                         "model":model, "preds":preds}
        log.info(f"{name}: MAE={mae:.1f}  RMSE={rmse:.1f}  R²={r2:.3f}  CV-R²={cv:.3f}")

    # Save best model
    best_name = max(results, key=lambda k: results[k]["R2"])
    best_model = results[best_name]["model"]
    joblib.dump(best_model, MODEL_PATH / "demand_forecast_model.pkl")
    joblib.dump(FEATURES,   MODEL_PATH / "feature_names.pkl")
    log.info(f"✅ Best model: {best_name} — saved to {MODEL_PATH}")

    # Feature importance plot (tree models)
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(8,6))
        fi.tail(12).plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title(f"Feature Importance — {best_name}")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("dashboard/feature_importance.png", dpi=150)
        log.info("Feature importance plot saved → dashboard/feature_importance.png")

    return results, X_test, y_test


# ── FORECAST NEXT 3 MONTHS ───────────────────────────────────────────────────
def forecast_next_quarter(df, model_path=MODEL_PATH/"demand_forecast_model.pkl"):
    """Use saved model to predict next 3 months for each state."""
    model    = joblib.load(model_path)
    features = joblib.load(MODEL_PATH / "feature_names.pkl")
    le       = joblib.load(MODEL_PATH / "state_encoder.pkl")

    states = df["customer_state"].unique()
    latest = df.groupby("customer_state").last().reset_index()

    forecasts = []
    for _, row in latest.iterrows():
        for ahead in range(1, 4):
            m = (row["month"] + ahead - 1) % 12 + 1
            y = row["year"] + (row["month"] + ahead - 1) // 12
            sample = {
                "year":row["year"], "month":m, "quarter":(m-1)//3+1,
                "is_q4": int((m-1)//3+1 == 4),
                "month_sin": np.sin(2*np.pi*m/12),
                "month_cos": np.cos(2*np.pi*m/12),
                "state_enc": le.transform([row["customer_state"]])[0],
                "lag_1_orders":row["order_count"],
                "lag_2_orders":row.get("lag_1_orders", row["order_count"]),
                "lag_3_orders":row.get("lag_2_orders", row["order_count"]),
                "rolling_3m": row.get("rolling_3m", row["order_count"]),
                "avg_order_val":row["avg_order_val"],
                "avg_review":row["avg_review"],
                "late_rate":row["late_rate"],
                "avg_delivery":row["avg_delivery"],
                "num_customers":row["num_customers"],
            }
            pred = model.predict(pd.DataFrame([sample])[features])[0]
            forecasts.append({"state":row["customer_state"],
                              "year":y, "month":m,
                              "predicted_orders":max(0, round(pred))})

    return pd.DataFrame(forecasts).sort_values(["state","year","month"])


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Loading master table from S3...")
    master = load_from_s3("processed/master.csv")

    log.info("Building features...")
    df = build_features(master)

    log.info("Training models...")
    results, X_test, y_test = train_models(df)

    log.info("Generating 3-month forecast...")
    forecast = forecast_next_quarter(df)
    forecast.to_csv("dashboard/demand_forecast.csv", index=False)
    log.info(f"Forecast saved:\n{forecast.head(10).to_string(index=False)}")
    log.info("🎉 Demand forecasting complete!")
