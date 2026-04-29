"""
E-Commerce Intelligence Platform — REST API
============================================
Serves demand forecast and customer segment predictions via Flask
Deploy on AWS EC2 or AWS Lambda (via Zappa/SAM)

Run locally: python api/app.py
Test:        curl -X POST http://localhost:5000/predict/demand \
               -H "Content-Type: application/json" \
               -d '{"state":"SP","month":3,"year":2025}'
"""

from flask import Flask, request, jsonify
from functools import wraps
import joblib, logging, os
import pandas as np
import numpy as np
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_PATH = Path("ml/models")
API_KEY    = os.getenv("API_KEY", "dev-key-ramu-2025")  # Set in production env

# ── LOAD MODELS AT STARTUP ─────────────────────────────────────────────────────
models = {}
try:
    models["demand"]  = joblib.load(MODEL_PATH / "demand_forecast_model.pkl")
    models["features"]= joblib.load(MODEL_PATH / "feature_names.pkl")
    models["encoder"] = joblib.load(MODEL_PATH / "state_encoder.pkl")
    models["kmeans"]  = joblib.load(MODEL_PATH / "kmeans_model.pkl")
    models["scaler"]  = joblib.load(MODEL_PATH / "rfm_scaler.pkl")
    log.info("✅ All models loaded")
except FileNotFoundError as e:
    log.warning(f"Model not found: {e}. Run ml scripts first.")


# ── AUTH DECORATOR ─────────────────────────────────────────────────────────────
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.utcnow().isoformat()
    })


# ── DEMAND FORECAST ENDPOINT ──────────────────────────────────────────────────
@app.route("/predict/demand", methods=["POST"])
@require_api_key
def predict_demand():
    """
    Predict order demand for a given state and month.
    Body: { "state": "SP", "month": 3, "year": 2025,
            "avg_order_val": 150.5, "avg_review": 4.1,
            "late_rate": 0.08, "avg_delivery": 12.3, "num_customers": 500 }
    """
    if "demand" not in models:
        return jsonify({"error": "Demand model not loaded"}), 503

    data = request.get_json()
    required = ["state","month","year"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    try:
        state_enc = models["encoder"].transform([data["state"]])[0]
    except ValueError:
        return jsonify({"error": f"Unknown state: {data['state']}"}), 400

    month   = int(data["month"])
    quarter = (month - 1) // 3 + 1
    sample  = {
        "year":           int(data["year"]),
        "month":          month,
        "quarter":        quarter,
        "is_q4":          int(quarter == 4),
        "month_sin":      np.sin(2 * np.pi * month / 12),
        "month_cos":      np.cos(2 * np.pi * month / 12),
        "state_enc":      state_enc,
        "lag_1_orders":   float(data.get("lag_1_orders", 500)),
        "lag_2_orders":   float(data.get("lag_2_orders", 480)),
        "lag_3_orders":   float(data.get("lag_3_orders", 460)),
        "rolling_3m":     float(data.get("rolling_3m", 480)),
        "avg_order_val":  float(data.get("avg_order_val", 150)),
        "avg_review":     float(data.get("avg_review", 4.0)),
        "late_rate":      float(data.get("late_rate", 0.1)),
        "avg_delivery":   float(data.get("avg_delivery", 12)),
        "num_customers":  int(data.get("num_customers", 400)),
    }

    import pandas as pd
    X = pd.DataFrame([sample])[models["features"]]
    prediction = max(0, round(models["demand"].predict(X)[0]))

    return jsonify({
        "state":             data["state"],
        "year":              int(data["year"]),
        "month":             month,
        "predicted_orders":  prediction,
        "confidence_note":   "Based on Random Forest / Gradient Boosting model"
    })


# ── CUSTOMER SEGMENT ENDPOINT ─────────────────────────────────────────────────
@app.route("/predict/segment", methods=["POST"])
@require_api_key
def predict_segment():
    """
    Predict customer segment based on RFM values.
    Body: { "recency": 30, "frequency": 5, "monetary": 850.0 }
    """
    if "kmeans" not in models:
        return jsonify({"error": "Segmentation model not loaded"}), 503

    data = request.get_json()
    required = ["recency","frequency","monetary"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    import pandas as pd
    rfm_vals = [[float(data["recency"]), float(data["frequency"]), float(data["monetary"])]]
    X_scaled = models["scaler"].transform(rfm_vals)
    cluster  = int(models["kmeans"].predict(X_scaled)[0])

    LABELS = {0:"Hibernating",1:"Promising",2:"Loyal",3:"Champions"}
    RECOMMENDATIONS = {
        "Champions":   "Reward with loyalty program. Early access to new products.",
        "Loyal":       "Upsell higher-value products. Offer membership benefits.",
        "Promising":   "Send personalized recommendations. Offer bundle deals.",
        "Hibernating": "Send aggressive re-engagement offer with discount.",
    }
    label = LABELS.get(cluster, f"Cluster {cluster}")
    return jsonify({
        "cluster":        cluster,
        "segment":        label,
        "recommendation": RECOMMENDATIONS.get(label, "Monitor and engage."),
        "input_rfm":      {"recency":data["recency"],"frequency":data["frequency"],
                          "monetary":data["monetary"]}
    })


# ── KPI SUMMARY ENDPOINT ──────────────────────────────────────────────────────
@app.route("/kpis", methods=["GET"])
@require_api_key
def get_kpis():
    """Return latest KPI summary from processed data."""
    try:
        import pandas as pd
        df = pd.read_csv("dashboard/ab_test_metrics.csv")
        return jsonify({
            "kpis": df.to_dict(orient="records"),
            "last_updated": datetime.utcnow().isoformat()
        })
    except FileNotFoundError:
        return jsonify({"error": "KPI data not found. Run ab_testing.py first."}), 404


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    log.info(f"Starting API on port {port} | debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
