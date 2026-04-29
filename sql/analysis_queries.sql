-- ============================================================
-- E-Commerce Intelligence Platform — AWS Athena SQL Queries
-- Run these in AWS Athena after setting up the Glue crawler
-- on your S3 processed/ folder
-- ============================================================

-- ── 1. MONTHLY REVENUE TREND ──────────────────────────────────────────────────
SELECT
    year,
    month,
    COUNT(DISTINCT order_id)    AS total_orders,
    ROUND(SUM(total_payment),2) AS revenue,
    ROUND(AVG(total_payment),2) AS avg_order_value,
    ROUND(AVG(delivery_days),1) AS avg_delivery_days
FROM processed.master
GROUP BY year, month
ORDER BY year, month;


-- ── 2. TOP 10 STATES BY REVENUE ───────────────────────────────────────────────
SELECT
    customer_state,
    COUNT(DISTINCT order_id)    AS total_orders,
    ROUND(SUM(total_payment),2) AS revenue,
    ROUND(AVG(review_score),2)  AS avg_review_score,
    ROUND(AVG(delivery_days),1) AS avg_delivery_days
FROM processed.master
GROUP BY customer_state
ORDER BY revenue DESC
LIMIT 10;


-- ── 3. PAYMENT TYPE DISTRIBUTION ──────────────────────────────────────────────
SELECT
    payment_type,
    COUNT(*)                      AS total_orders,
    ROUND(SUM(total_payment),2)   AS total_revenue,
    ROUND(AVG(installments),1)    AS avg_installments,
    ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),1) AS pct_orders
FROM processed.master
GROUP BY payment_type
ORDER BY total_orders DESC;


-- ── 4. DELIVERY PERFORMANCE ANALYSIS ─────────────────────────────────────────
SELECT
    customer_state,
    COUNT(*)                                   AS total_orders,
    SUM(is_late)                               AS late_orders,
    ROUND(100.0*SUM(is_late)/COUNT(*),1)       AS late_rate_pct,
    ROUND(AVG(delivery_days),1)                AS avg_delivery_days,
    ROUND(AVG(CASE WHEN is_late=1 THEN delivery_days END),1) AS avg_late_days
FROM processed.master
GROUP BY customer_state
HAVING COUNT(*) > 100
ORDER BY late_rate_pct DESC
LIMIT 15;


-- ── 5. CUSTOMER REVIEW SCORE vs DELIVERY DAYS ─────────────────────────────────
SELECT
    review_score,
    COUNT(*)                     AS order_count,
    ROUND(AVG(delivery_days),1)  AS avg_delivery_days,
    ROUND(AVG(total_payment),2)  AS avg_order_value,
    SUM(is_late)                 AS late_orders
FROM processed.master
WHERE review_score IS NOT NULL
GROUP BY review_score
ORDER BY review_score;


-- ── 6. HOURLY ORDER PATTERN (PEAK HOURS) ──────────────────────────────────────
SELECT
    hour,
    COUNT(*) AS orders,
    ROUND(AVG(total_payment),2) AS avg_revenue
FROM processed.master
GROUP BY hour
ORDER BY hour;


-- ── 7. DAY OF WEEK SALES PATTERN ─────────────────────────────────────────────
SELECT
    day_of_week,
    COUNT(*) AS orders,
    ROUND(SUM(total_payment),2)  AS total_revenue,
    ROUND(AVG(total_payment),2)  AS avg_order_value
FROM processed.master
GROUP BY day_of_week
ORDER BY orders DESC;


-- ── 8. HIGH VALUE CUSTOMER IDENTIFICATION (TOP 5% by spend) ──────────────────
WITH customer_spend AS (
    SELECT
        customer_id,
        COUNT(DISTINCT order_id)    AS total_orders,
        ROUND(SUM(total_payment),2) AS total_spend,
        ROUND(AVG(total_payment),2) AS avg_order_value,
        ROUND(AVG(review_score),2)  AS avg_review,
        MIN(order_purchase_timestamp) AS first_order,
        MAX(order_purchase_timestamp) AS last_order
    FROM processed.master
    GROUP BY customer_id
),
percentiles AS (
    SELECT APPROX_PERCENTILE(total_spend, 0.95) AS p95 FROM customer_spend
)
SELECT c.*
FROM customer_spend c, percentiles p
WHERE c.total_spend >= p.p95
ORDER BY total_spend DESC;


-- ── 9. FUNNEL ANALYSIS ────────────────────────────────────────────────────────
-- (requires raw orders table with all statuses)
SELECT
    order_status,
    COUNT(*) AS orders,
    ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),2) AS pct
FROM processed.orders
GROUP BY order_status
ORDER BY orders DESC;


-- ── 10. MONTH-OVER-MONTH GROWTH (WINDOW FUNCTION) ────────────────────────────
WITH monthly AS (
    SELECT
        year, month,
        ROUND(SUM(total_payment),2) AS revenue
    FROM processed.master
    GROUP BY year, month
)
SELECT
    year, month, revenue,
    LAG(revenue) OVER (ORDER BY year, month) AS prev_month_revenue,
    ROUND(100.0*(revenue - LAG(revenue) OVER (ORDER BY year, month))
          / NULLIF(LAG(revenue) OVER (ORDER BY year, month),0), 1) AS mom_growth_pct
FROM monthly
ORDER BY year, month;


-- ── 11. RFM SEGMENT SUMMARY ───────────────────────────────────────────────────
SELECT
    segment,
    COUNT(*)                    AS customers,
    ROUND(AVG(recency),0)       AS avg_recency_days,
    ROUND(AVG(frequency),1)     AS avg_orders,
    ROUND(AVG(monetary),2)      AS avg_spend,
    ROUND(SUM(monetary),2)      AS total_revenue
FROM processed.rfm
GROUP BY segment
ORDER BY avg_spend DESC;


-- ── 12. A/B TEST METRIC QUERY (Campaign Control vs Test) ─────────────────────
-- Assumes you have an experiment_assignments table with user_id, variant
SELECT
    variant,
    COUNT(DISTINCT m.customer_id)  AS users,
    COUNT(DISTINCT m.order_id)     AS orders,
    ROUND(COUNT(DISTINCT m.order_id)*1.0/COUNT(DISTINCT m.customer_id),4) AS conversion_rate,
    ROUND(SUM(total_payment),2)    AS total_revenue,
    ROUND(AVG(total_payment),2)    AS avg_order_value,
    ROUND(SUM(total_payment)/COUNT(DISTINCT m.customer_id),2) AS revenue_per_user
FROM processed.master m
JOIN experiment_assignments e ON m.customer_id = e.customer_id
GROUP BY variant;
