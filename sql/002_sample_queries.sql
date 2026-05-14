-- Highest stockout categories during business hours.
SELECT *
FROM dw.v_stockout_rate_by_category
ORDER BY stockout_rate_6_22 DESC
LIMIT 20;

-- Warehouse data quality checks for the loaded staging data.
SELECT *
FROM dw.v_data_quality_checks
ORDER BY severity, check_name;

-- Rows available to the model. Train only where is_trainable_demand_observation is true.
SELECT
    source_split,
    is_trainable_demand_observation,
    COUNT(*) AS hourly_rows
FROM dw.v_model_training_features_hourly
GROUP BY source_split, is_trainable_demand_observation
ORDER BY source_split, is_trainable_demand_observation DESC;

-- Operational restock watchlist for a specific day.
SELECT *
FROM dw.v_daily_restock_monitor
WHERE full_date = DATE '2024-06-26'
  AND restock_priority IN ('critical', 'high')
ORDER BY stockout_hours_6_22 DESC, observed_daily_sales_amount DESC
LIMIT 50;

-- Daily sales and stockout trend.
SELECT
    d.full_date,
    SUM(f.observed_daily_sales_amount) AS observed_sales_amount,
    SUM(f.stockout_hours_6_22) AS stockout_hours_6_22,
    COUNT(*) FILTER (WHERE f.has_stockout) AS stockout_product_store_days
FROM dw.fact_sales_inventory_daily f
JOIN dw.dim_date d ON d.date_key = f.date_key
GROUP BY d.full_date
ORDER BY d.full_date;

-- DSS action distribution across the four criteria.
SELECT
    model_name,
    model_version,
    decision_action,
    COUNT(*) AS product_store_days,
    ROUND(AVG(stockout_rate_6_22)::NUMERIC, 4) AS avg_stockout_rate,
    ROUND(AVG(waste_risk_score)::NUMERIC, 4) AS avg_waste_risk,
    ROUND(AVG(demand_bias_rate)::NUMERIC, 4) AS avg_demand_bias,
    ROUND(AVG(restock_urgency_score)::NUMERIC, 4) AS avg_restock_urgency
FROM dw.v_dss_daily_decision_score
GROUP BY model_name, model_version, decision_action
ORDER BY avg_restock_urgency DESC;

-- Top recommended restocking actions.
SELECT
    full_date,
    store_id,
    product_id,
    estimate_source,
    stockout_hours_6_22,
    ROUND(estimated_lost_sales::NUMERIC, 4) AS estimated_lost_sales,
    ROUND(recommended_order_qty::NUMERIC, 4) AS recommended_order_qty,
    ROUND(restock_urgency_score::NUMERIC, 4) AS restock_urgency_score,
    decision_action,
    decision_reason
FROM dw.v_dss_daily_decision_score
ORDER BY restock_urgency_score DESC, estimated_lost_sales DESC
LIMIT 25;

-- Hourly drill-down for one store-product-date recommendation.
SELECT
    d.full_date,
    t.hour_of_day,
    s.store_id,
    p.product_id,
    h.observed_sales_amount,
    h.estimated_true_demand,
    h.estimated_lost_sales,
    h.stockout_flag,
    h.estimate_source
FROM dw.v_dss_hourly_demand_estimate h
JOIN dw.dim_date d ON d.date_key = h.date_key
JOIN dw.dim_time t ON t.time_key = h.time_key
JOIN dw.dim_store s ON s.store_key = h.store_key
JOIN dw.dim_product p ON p.product_key = h.product_key
WHERE d.full_date = DATE '2024-06-26'
  AND s.store_id = 1
  AND p.product_id = 1
ORDER BY t.hour_of_day;

-- Category-level DSS pressure summary.
SELECT
    first_category_id,
    COUNT(*) AS product_store_days,
    ROUND(SUM(observed_daily_sales_amount)::NUMERIC, 4) AS observed_sales_amount,
    ROUND(SUM(estimated_lost_sales)::NUMERIC, 4) AS estimated_lost_sales,
    ROUND(AVG(stockout_rate_6_22)::NUMERIC, 4) AS avg_stockout_rate,
    ROUND(AVG(waste_risk_score)::NUMERIC, 4) AS avg_waste_risk,
    ROUND(AVG(restock_urgency_score)::NUMERIC, 4) AS avg_restock_urgency
FROM dw.v_dss_daily_decision_score
GROUP BY first_category_id
ORDER BY avg_restock_urgency DESC, estimated_lost_sales DESC
LIMIT 20;
