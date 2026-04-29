"""
E-Commerce Intelligence Platform — ETL Pipeline
================================================
Extract: Load Olist CSV files from local (downloaded from Kaggle)
Transform: Clean, join, engineer features
Load: Upload processed data to AWS S3

Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Run: python etl/etl_pipeline.py
"""

import pandas as pd
import numpy as np
import boto3
import os
import io
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

AWS_BUCKET   = os.getenv("AWS_BUCKET", "ecommerce-intelligence-ramu")
AWS_REGION   = os.getenv("AWS_REGION", "us-west-2")
RAW_PREFIX   = "raw/"
CLEAN_PREFIX = "processed/"
LOCAL_DATA   = Path("data/raw")


def get_s3():
    return boto3.client(
        "s3", region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def upload_df(df, s3, key):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=buf.getvalue())
    log.info(f"✅ Uploaded s3://{AWS_BUCKET}/{key}  ({len(df):,} rows)")


# ── EXTRACT ────────────────────────────────────────────────────────────────────
def extract(s3):
    log.info("=== PHASE 1: EXTRACT ===")
    dfs = {}
    for f in LOCAL_DATA.glob("*.csv"):
        name = f.stem.replace("olist_", "").replace("_dataset", "")
        dfs[name] = pd.read_csv(f)
        log.info(f"Loaded {name}: {dfs[name].shape}")
        upload_df(dfs[name], s3, f"{RAW_PREFIX}{name}.csv")
    return dfs


# ── TRANSFORM ──────────────────────────────────────────────────────────────────
def transform(dfs):
    log.info("=== PHASE 2: TRANSFORM ===")

    # 1. Orders — parse dates, derive features
    orders = dfs["orders"].copy()
    for col in ["order_purchase_timestamp", "order_delivered_customer_date",
                "order_estimated_delivery_date", "order_approved_at"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    orders["year"]         = orders["order_purchase_timestamp"].dt.year
    orders["month"]        = orders["order_purchase_timestamp"].dt.month
    orders["day_of_week"]  = orders["order_purchase_timestamp"].dt.day_name()
    orders["hour"]         = orders["order_purchase_timestamp"].dt.hour
    orders["delivery_days"]= (orders["order_delivered_customer_date"] -
                               orders["order_purchase_timestamp"]).dt.days
    orders["is_late"]      = (orders["order_delivered_customer_date"] >
                               orders["order_estimated_delivery_date"]).astype(int)
    orders = orders[orders["order_status"] == "delivered"].dropna(
        subset=["order_delivered_customer_date"])
    log.info(f"Orders: {orders.shape}")

    # 2. Items — total value per item
    items = dfs["order_items"].copy()
    items["item_total"] = items["price"] + items["freight_value"]

    # 3. Payments — aggregate per order
    payments = dfs["order_payments"].groupby("order_id").agg(
        total_payment=("payment_value", "sum"),
        installments=("payment_installments", "max"),
        payment_type=("payment_type", lambda x: x.mode()[0])
    ).reset_index()

    # 4. Reviews — average score per order
    reviews = dfs["order_reviews"].groupby("order_id").agg(
        review_score=("review_score", "mean")
    ).reset_index()

    # 5. Master table — join everything
    master = (orders
        .merge(items.groupby("order_id").agg(
            num_items=("order_item_id","count"),
            order_value=("item_total","sum"),
            avg_price=("price","mean")
        ).reset_index(), on="order_id", how="left")
        .merge(payments, on="order_id", how="left")
        .merge(reviews, on="order_id", how="left")
        .merge(dfs["customers"][["customer_id","customer_state","customer_city"]],
               on="customer_id", how="left")
    )

    # 6. RFM features for customer segmentation
    snapshot = master["order_purchase_timestamp"].max()
    rfm = master.groupby("customer_id").agg(
        recency=("order_purchase_timestamp", lambda x: (snapshot - x.max()).days),
        frequency=("order_id", "count"),
        monetary=("total_payment", "sum")
    ).reset_index()

    # Score 1–4 for each dimension
    for col in ["recency","frequency","monetary"]:
        ascending = col == "recency"
        rfm[f"{col}_score"] = pd.qcut(rfm[col], 4,
            labels=[4,3,2,1] if ascending else [1,2,3,4]).astype(int)
    rfm["rfm_score"] = (rfm["recency_score"] + rfm["frequency_score"] +
                        rfm["monetary_score"])
    rfm["segment"] = pd.cut(rfm["rfm_score"],
        bins=[2,5,8,12], labels=["At-Risk","Potential","Champions"])

    log.info(f"Master table: {master.shape}")
    log.info(f"RFM table:    {rfm.shape}")

    return {"master": master, "rfm": rfm, "orders": orders,
            "items": items, "payments": payments}


# ── LOAD ────────────────────────────────────────────────────────────────────────
def load(processed, s3):
    log.info("=== PHASE 3: LOAD ===")
    for name, df in processed.items():
        upload_df(df, s3, f"{CLEAN_PREFIX}{name}.csv")
    log.info("All processed files uploaded to S3 ✅")


# ── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    s3 = get_s3()
    raw  = extract(s3)
    proc = transform(raw)
    load(proc, s3)
    log.info("🎉 ETL pipeline complete!")
