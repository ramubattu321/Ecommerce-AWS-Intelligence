"""
E-Commerce Intelligence Platform — Customer Segmentation
=========================================================
Methods: RFM Analysis + K-Means Clustering
Output: Customer segments with business recommendations

Run: python ml/customer_segmentation.py
"""

import pandas as pd
import numpy as np
import boto3, os, io, joblib, logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

AWS_BUCKET = os.getenv("AWS_BUCKET", "ecommerce-intelligence-ramu")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL_PATH = Path("ml/models")
MODEL_PATH.mkdir(exist_ok=True)


def load_from_s3(key):
    s3 = boto3.client("s3", region_name=AWS_REGION,
                      aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


# ── RFM SCORING ───────────────────────────────────────────────────────────────
def compute_rfm(master: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary for each customer."""
    master["order_purchase_timestamp"] = pd.to_datetime(
        master["order_purchase_timestamp"], errors="coerce")
    snapshot = master["order_purchase_timestamp"].max()

    rfm = master.groupby("customer_id").agg(
        recency   = ("order_purchase_timestamp", lambda x: (snapshot - x.max()).days),
        frequency = ("order_id", "count"),
        monetary  = ("total_payment", "sum"),
        avg_review= ("review_score", "mean"),
        late_rate = ("is_late", "mean"),
    ).reset_index()

    # Score 1–4 (4 = best)
    rfm["R"] = pd.qcut(rfm["recency"],  4, labels=[4,3,2,1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"],  4, labels=[1,2,3,4]).astype(int)
    rfm["rfm_score"] = rfm["R"] + rfm["F"] + rfm["M"]

    # Rule-based segments
    conditions = [
        rfm["rfm_score"] >= 10,
        rfm["rfm_score"] >= 7,
        rfm["rfm_score"] >= 5,
        rfm["rfm_score"] >= 3,
    ]
    labels = ["Champions", "Loyal Customers", "At-Risk", "Lost"]
    rfm["rfm_segment"] = np.select(conditions, labels, default="Lost")

    log.info(f"RFM computed for {len(rfm):,} customers")
    log.info(rfm["rfm_segment"].value_counts().to_string())
    return rfm


# ── K-MEANS CLUSTERING ────────────────────────────────────────────────────────
def cluster_customers(rfm: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Apply K-Means clustering on RFM features."""
    features = ["recency", "frequency", "monetary"]
    X = rfm[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal k using elbow + silhouette
    inertias, silhouettes = [], []
    for n in range(2, 9):
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = silhouettes.index(max(silhouettes)) + 2
    log.info(f"Optimal clusters by silhouette: {best_k} (score={max(silhouettes):.3f})")

    # Final model
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["cluster"] = km_final.fit_predict(X_scaled)

    # Label clusters by mean monetary value
    cluster_means = rfm.groupby("cluster")["monetary"].mean().sort_values()
    cluster_labels = {c: l for c, l in zip(
        cluster_means.index,
        ["Hibernating","Promising","Loyal","Champions"][:best_k]
    )}
    rfm["cluster_label"] = rfm["cluster"].map(cluster_labels)

    # Save models
    joblib.dump(km_final, MODEL_PATH / "kmeans_model.pkl")
    joblib.dump(scaler,   MODEL_PATH / "rfm_scaler.pkl")

    # Elbow plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(2,9), inertias, "bo-")
    ax1.set_title("Elbow Curve"); ax1.set_xlabel("K"); ax1.set_ylabel("Inertia")
    ax2.plot(range(2,9), silhouettes, "ro-")
    ax2.set_title("Silhouette Score"); ax2.set_xlabel("K")
    plt.tight_layout()
    plt.savefig("dashboard/cluster_selection.png", dpi=150)
    log.info("Cluster selection plot saved → dashboard/cluster_selection.png")

    log.info("Cluster summary:")
    summary = rfm.groupby("cluster_label").agg(
        customers=("customer_id","count"),
        avg_recency=("recency","mean"),
        avg_frequency=("frequency","mean"),
        avg_monetary=("monetary","mean"),
    ).round(1)
    log.info(summary.to_string())

    return rfm, summary


# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Champions":        "Reward with loyalty program. Ask for reviews. Early access to new products.",
    "Loyal Customers":  "Upsell higher-value products. Offer membership benefits.",
    "Promising":        "Send personalized recommendations. Offer first-time bundle deals.",
    "At-Risk":          "Send win-back email campaign with discount. Survey for dissatisfaction.",
    "Hibernating":      "Send aggressive re-engagement offer. Consider removing from active list.",
    "Lost":             "Final re-engagement attempt. Otherwise suppress from campaigns.",
}


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Loading master table from S3...")
    master = load_from_s3("processed/master.csv")

    log.info("Computing RFM scores...")
    rfm = compute_rfm(master)

    log.info("Clustering customers...")
    rfm, summary = cluster_customers(rfm)

    # Add recommendations
    rfm["recommendation"] = rfm["rfm_segment"].map(RECOMMENDATIONS)

    # Save outputs
    rfm.to_csv("dashboard/customer_segments.csv", index=False)
    summary.to_csv("dashboard/cluster_summary.csv")

    log.info("Output saved → dashboard/customer_segments.csv")
    log.info("🎉 Customer segmentation complete!")
