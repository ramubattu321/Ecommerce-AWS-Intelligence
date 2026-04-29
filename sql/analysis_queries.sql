-- ============================================================
-- E-Commerce Intelligence Platform — AWS Athena SQL Queries
-- 12 Production Queries | Run in AWS Athena after Glue Crawler
-- Author: Ramu Battu — MS Data Analytics, CSU Fresno
-- ============================================================


-- ── QUERY 1: MONTHLY REVENUE TREND ───────────────────────────────────────────
SELECT
    year,
    month,
    COUNT(DISTINCT order_id)                AS total_orders,
    ROUND(SUM(total_payment), 2)            AS revenue,
    ROUND(AVG(total_payment), 2)            AS avg_order_value,
    ROUND(AVG(delivery_days), 1)            AS avg_delivery_days,
    SUM(is_late)                            AS late_orders,
    ROUND(100.0*SUM(is_late)/COUNT(*), 1)   AS late_rate_pct
FROM processed.master
GROUP BY year, month
ORDER BY year, month;


-- ── QUERY 2: TOP 10 STATES BY REVENUE ────────────────────────────────────────
SELECT
    customer_state,
    COUNT(DISTINCT order_id)                AS total_orders,
    ROUND(SUM(total_payment), 2)            AS revenue,
    ROUND(AVG(review_score), 2)             AS avg_review_score,
    ROUND(AVG(delivery_days), 1)            AS avg_delivery_days,
    ROUND(100.0*SUM(is_late)/COUNT(*), 1)   AS late_rate_pct
FROM processed.master
GROUP BY customer_state
ORDER BY revenue DESC
LIMIT 10;


-- ── QUERY 3: PAYMENT TYPE DISTRIBUTION ───────────────────────────────────────
SELECT
    payment_type,
    COUNT(*)                                                        AS total_orders,
    ROUND(SUM(total_payment), 2)                                    AS total_revenue,
    ROUND(AVG(installments), 1)                                     AS avg_installments,
    ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 1)                  AS pct_orders
FROM processed.master
GROUP BY payment_type
ORDER BY total_orders DESC;


-- ── QUERY 4: DELIVERY PERFORMANCE BY STATE ────────────────────────────────────
SELECT
    customer_state,
    COUNT(*)                                                        AS total_orders,
    SUM(is_late)                                                    AS late_orders,
    ROUND(100.0*SUM(is_late)/COUNT(*), 1)                          AS late_rate_pct,
    ROUND(AVG(delivery_days), 1)                                    AS avg_delivery_days,
    ROUND(AVG(CASE WHEN is_late=1 THEN delivery_days END), 1)      AS avg_late_days
FROM processed.master
GROUP BY customer_state
HAVING COUNT(*) > 100
ORDER BY late_rate_pct DESC
LIMIT 15;


-- ── QUERY 5: REVIEW SCORE vs DELIVERY PERFORMANCE ────────────────────────────
SELECT
    review_score,
    COUNT(*)                                AS order_count,
    ROUND(AVG(delivery_days), 1)            AS avg_delivery_days,
    ROUND(AVG(total_payment), 2)            AS avg_order_value,
    SUM(is_late)                            AS late_orders,
    ROUND(100.0*SUM(is_late)/COUNT(*), 1)   AS late_rate_pct
FROM processed.master
WHERE review_score IS NOT NULL
GROUP BY review_score
ORDER BY review_score;


-- ── QUERY 6: HOURLY ORDER PATTERN ────────────────────────────────────────────
SELECT
    hour,
    COUNT(*)                                AS orders,
    ROUND(AVG(total_payment), 2)            AS avg_revenue,
    ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(), 1) AS pct_of_orders
FROM processed.master
GROUP BY hour
ORDER BY hour;


-- ── QUERY 7: DAY OF WEEK SALES PATTERN ───────────────────────────────────────
SELECT
    day_of_week,
    COUNT(*)                                AS orders,
    ROUND(SUM(total_payment), 2)            AS total_revenue,
    ROUND(AVG(total_payment), 2)            AS avg_order_value,
    ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(), 1) AS pct_of_orders
FROM processed.master
GROUP BY day_of_week
ORDER BY orders DESC;


-- ── QUERY 8: HIGH VALUE CUSTOMER IDENTIFICATION (TOP 5%) ─────────────────────
WITH customer_spend AS (
    SELECT
        customer_id,
        COUNT(DISTINCT order_id)            AS total_orders,
        ROUND(SUM(total_payment), 2)        AS total_spend,
        ROUND(AVG(total_payment), 2)        AS avg_order_value,
        ROUND(AVG(review_score), 2)         AS avg_review,
        MIN(order_purchase_timestamp)       AS first_order,
        MAX(order_purchase_timestamp)       AS last_order
    FROM processed.master
    GROUP BY customer_id
),
percentiles AS (
    SELECT APPROX_PERCENTILE(total_spend, 0.95) AS p95
    FROM customer_spend
)
SELECT c.*
FROM customer_spend c, percentiles p
WHERE c.total_spend >= p.p95
ORDER BY total_spend DESC
LIMIT 100;


-- ── QUERY 9: ORDER STATUS FUNNEL ANALYSIS ────────────────────────────────────
SELECT
    order_status,
    COUNT(*)                                                        AS orders,
    ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (), 2)                 AS pct
FROM processed.orders
GROUP BY order_status
ORDER BY orders DESC;


-- ── QUERY 10: MONTH-OVER-MONTH REVENUE GROWTH (WINDOW FUNCTION) ───────────────
WITH monthly AS (
    SELECT
        year, month,
        ROUND(SUM(total_payment), 2)        AS revenue
    FROM processed.master
    GROUP BY year, month
)
SELECT
    year, month, revenue,
    LAG(revenue) OVER (ORDER BY year, month)                       AS prev_month_revenue,
    ROUND(100.0*(revenue - LAG(revenue) OVER (ORDER BY year, month))
          / NULLIF(LAG(revenue) OVER (ORDER BY year, month), 0), 1) AS mom_growth_pct
FROM monthly
ORDER BY year, month;


-- ── QUERY 11: RFM SEGMENT SUMMARY ────────────────────────────────────────────
SELECT
    segment,
    COUNT(*)                                AS customers,
    ROUND(AVG(recency), 0)                  AS avg_recency_days,
    ROUND(AVG(frequency), 1)                AS avg_orders,
    ROUND(AVG(monetary), 2)                 AS avg_spend,
    ROUND(SUM(monetary), 2)                 AS total_revenue,
    ROUND(100.0*SUM(monetary)/SUM(SUM(monetary)) OVER(), 1) AS revenue_share_pct
FROM processed.rfm
GROUP BY segment
ORDER BY avg_spend DESC;


-- ── QUERY 12: A/B TEST CAMPAIGN PERFORMANCE ──────────────────────────────────
-- Assumes experiment_assignments table with customer_id, variant
SELECT
    e.variant,
    COUNT(DISTINCT m.customer_id)           AS users,
    COUNT(DISTINCT m.order_id)              AS orders,
    ROUND(COUNT(DISTINCT m.order_id)*1.0
          /COUNT(DISTINCT m.customer_id), 4) AS conversion_rate,
    ROUND(SUM(total_payment), 2)            AS total_revenue,
    ROUND(AVG(total_payment), 2)            AS avg_order_value,
    ROUND(SUM(total_payment)
          /COUNT(DISTINCT m.customer_id), 2) AS revenue_per_user
FROM processed.master m
JOIN experiment_assignments e ON m.customer_id = e.customer_id
GROUP BY e.variant;
