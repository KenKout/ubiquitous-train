CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS dw;

CREATE TABLE IF NOT EXISTS staging.fresh_retail_observation_day (
    source_split TEXT NOT NULL CHECK (source_split IN ('train', 'eval')),
    city_id BIGINT NOT NULL,
    store_id BIGINT NOT NULL,
    management_group_id BIGINT NOT NULL,
    first_category_id BIGINT NOT NULL,
    second_category_id BIGINT NOT NULL,
    third_category_id BIGINT NOT NULL,
    product_id BIGINT NOT NULL,
    dt DATE NOT NULL,
    sale_amount DOUBLE PRECISION NOT NULL,
    hours_sale DOUBLE PRECISION[] NOT NULL,
    stock_hour6_22_cnt INTEGER NOT NULL,
    hours_stock_status SMALLINT[] NOT NULL,
    discount DOUBLE PRECISION NOT NULL,
    holiday_flag SMALLINT NOT NULL,
    activity_flag SMALLINT NOT NULL,
    precpt DOUBLE PRECISION NOT NULL,
    avg_temperature DOUBLE PRECISION NOT NULL,
    avg_humidity DOUBLE PRECISION NOT NULL,
    avg_wind_level DOUBLE PRECISION NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fresh_retail_observation_day_pk PRIMARY KEY (source_split, store_id, product_id, dt),
    CONSTRAINT hours_sale_24_chk CHECK (array_length(hours_sale, 1) = 24),
    CONSTRAINT hours_stock_status_24_chk CHECK (array_length(hours_stock_status, 1) = 24)
);

CREATE TABLE IF NOT EXISTS dw.dim_date (
    date_key INTEGER PRIMARY KEY,
    full_date DATE NOT NULL UNIQUE,
    day_of_week SMALLINT NOT NULL,
    day_name TEXT NOT NULL,
    week_of_year SMALLINT NOT NULL,
    month_number SMALLINT NOT NULL,
    month_name TEXT NOT NULL,
    quarter_number SMALLINT NOT NULL,
    year_number SMALLINT NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    holiday_flag SMALLINT NOT NULL
);

CREATE TABLE IF NOT EXISTS dw.dim_time (
    time_key SMALLINT PRIMARY KEY CHECK (time_key BETWEEN 0 AND 23),
    hour_of_day SMALLINT NOT NULL CHECK (hour_of_day BETWEEN 0 AND 23),
    is_business_hour_6_22 BOOLEAN NOT NULL,
    day_part TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dw.dim_city (
    city_key INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    city_id BIGINT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dw.dim_store (
    store_key INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    store_id BIGINT NOT NULL UNIQUE,
    city_key INTEGER NOT NULL REFERENCES dw.dim_city(city_key)
);

CREATE TABLE IF NOT EXISTS dw.dim_product (
    product_key INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product_id BIGINT NOT NULL UNIQUE,
    management_group_id BIGINT NOT NULL,
    first_category_id BIGINT NOT NULL,
    second_category_id BIGINT NOT NULL,
    third_category_id BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS dw.dim_model (
    model_key INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    training_start_date DATE,
    training_end_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT dim_model_name_version_uq UNIQUE (model_name, model_version)
);

CREATE TABLE IF NOT EXISTS dw.fact_sales_inventory_daily (
    date_key INTEGER NOT NULL REFERENCES dw.dim_date(date_key),
    store_key INTEGER NOT NULL REFERENCES dw.dim_store(store_key),
    product_key INTEGER NOT NULL REFERENCES dw.dim_product(product_key),
    source_split TEXT NOT NULL CHECK (source_split IN ('train', 'eval')),
    observed_daily_sales_amount DOUBLE PRECISION NOT NULL,
    stockout_hours_6_22 INTEGER NOT NULL,
    stockout_hours_total INTEGER NOT NULL,
    has_stockout BOOLEAN NOT NULL,
    discount_rate DOUBLE PRECISION NOT NULL,
    holiday_flag SMALLINT NOT NULL,
    activity_flag SMALLINT NOT NULL,
    precpt DOUBLE PRECISION NOT NULL,
    avg_temperature DOUBLE PRECISION NOT NULL,
    avg_humidity DOUBLE PRECISION NOT NULL,
    avg_wind_level DOUBLE PRECISION NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fact_sales_inventory_daily_pk PRIMARY KEY (date_key, store_key, product_key)
);

CREATE TABLE IF NOT EXISTS dw.fact_sales_inventory_hourly (
    date_key INTEGER NOT NULL REFERENCES dw.dim_date(date_key),
    time_key SMALLINT NOT NULL REFERENCES dw.dim_time(time_key),
    store_key INTEGER NOT NULL REFERENCES dw.dim_store(store_key),
    product_key INTEGER NOT NULL REFERENCES dw.dim_product(product_key),
    source_split TEXT NOT NULL CHECK (source_split IN ('train', 'eval')),
    observed_sales_amount DOUBLE PRECISION NOT NULL,
    stockout_flag BOOLEAN NOT NULL,
    is_censored_observation BOOLEAN NOT NULL,
    discount_rate DOUBLE PRECISION NOT NULL,
    activity_flag SMALLINT NOT NULL,
    precpt DOUBLE PRECISION NOT NULL,
    avg_temperature DOUBLE PRECISION NOT NULL,
    avg_humidity DOUBLE PRECISION NOT NULL,
    avg_wind_level DOUBLE PRECISION NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fact_sales_inventory_hourly_pk PRIMARY KEY (date_key, time_key, store_key, product_key)
);

CREATE TABLE IF NOT EXISTS dw.fact_demand_estimate_hourly (
    date_key INTEGER NOT NULL REFERENCES dw.dim_date(date_key),
    time_key SMALLINT NOT NULL REFERENCES dw.dim_time(time_key),
    store_key INTEGER NOT NULL REFERENCES dw.dim_store(store_key),
    product_key INTEGER NOT NULL REFERENCES dw.dim_product(product_key),
    model_key INTEGER NOT NULL REFERENCES dw.dim_model(model_key),
    observed_sales_amount DOUBLE PRECISION NOT NULL,
    estimated_true_demand DOUBLE PRECISION NOT NULL,
    estimated_lost_sales DOUBLE PRECISION NOT NULL,
    stockout_flag BOOLEAN NOT NULL,
    is_censored_observation BOOLEAN NOT NULL,
    prediction_lower_bound DOUBLE PRECISION,
    prediction_upper_bound DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fact_demand_estimate_hourly_pk PRIMARY KEY (date_key, time_key, store_key, product_key, model_key),
    CONSTRAINT estimated_lost_sales_nonnegative_chk CHECK (estimated_lost_sales >= 0)
);

CREATE TABLE IF NOT EXISTS dw.fact_replenishment_recommendation_daily (
    date_key INTEGER NOT NULL REFERENCES dw.dim_date(date_key),
    store_key INTEGER NOT NULL REFERENCES dw.dim_store(store_key),
    product_key INTEGER NOT NULL REFERENCES dw.dim_product(product_key),
    model_key INTEGER NOT NULL REFERENCES dw.dim_model(model_key),
    recommended_order_qty DOUBLE PRECISION NOT NULL,
    expected_demand DOUBLE PRECISION NOT NULL,
    expected_lost_sales DOUBLE PRECISION NOT NULL,
    stockout_risk_score DOUBLE PRECISION NOT NULL CHECK (stockout_risk_score BETWEEN 0 AND 1),
    expected_waste_qty DOUBLE PRECISION NOT NULL,
    service_level_target DOUBLE PRECISION NOT NULL CHECK (service_level_target BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fact_replenishment_recommendation_daily_pk PRIMARY KEY (date_key, store_key, product_key, model_key)
);

CREATE INDEX IF NOT EXISTS fresh_retail_observation_day_dt_idx
    ON staging.fresh_retail_observation_day(dt);

CREATE INDEX IF NOT EXISTS fresh_retail_observation_day_product_idx
    ON staging.fresh_retail_observation_day(product_id);

CREATE INDEX IF NOT EXISTS fact_sales_inventory_daily_product_date_idx
    ON dw.fact_sales_inventory_daily(product_key, date_key);

CREATE INDEX IF NOT EXISTS fact_sales_inventory_daily_store_date_idx
    ON dw.fact_sales_inventory_daily(store_key, date_key);

CREATE INDEX IF NOT EXISTS fact_sales_inventory_hourly_product_date_idx
    ON dw.fact_sales_inventory_hourly(product_key, date_key, time_key);

CREATE INDEX IF NOT EXISTS fact_sales_inventory_hourly_store_date_idx
    ON dw.fact_sales_inventory_hourly(store_key, date_key, time_key);

CREATE OR REPLACE VIEW dw.v_daily_restock_monitor AS
SELECT
    d.full_date,
    c.city_id,
    s.store_id,
    p.product_id,
    p.management_group_id,
    p.first_category_id,
    p.second_category_id,
    p.third_category_id,
    f.source_split,
    f.observed_daily_sales_amount,
    f.stockout_hours_6_22,
    f.stockout_hours_total,
    f.has_stockout,
    f.discount_rate,
    f.activity_flag,
    f.avg_temperature,
    f.precpt,
    CASE
        WHEN f.stockout_hours_6_22 >= 8 THEN 'critical'
        WHEN f.stockout_hours_6_22 >= 3 THEN 'high'
        WHEN f.stockout_hours_6_22 > 0 THEN 'watch'
        ELSE 'normal'
    END AS restock_priority
FROM dw.fact_sales_inventory_daily f
JOIN dw.dim_date d ON d.date_key = f.date_key
JOIN dw.dim_store s ON s.store_key = f.store_key
JOIN dw.dim_city c ON c.city_key = s.city_key
JOIN dw.dim_product p ON p.product_key = f.product_key;

CREATE OR REPLACE VIEW dw.v_stockout_rate_by_category AS
SELECT
    p.management_group_id,
    p.first_category_id,
    p.second_category_id,
    p.third_category_id,
    COUNT(*) AS product_store_days,
    SUM(f.stockout_hours_6_22) AS stockout_hours_6_22,
    ROUND(SUM(f.stockout_hours_6_22)::NUMERIC / NULLIF(COUNT(*) * 16, 0), 6) AS stockout_rate_6_22,
    SUM(f.observed_daily_sales_amount) AS observed_sales_amount
FROM dw.fact_sales_inventory_daily f
JOIN dw.dim_product p ON p.product_key = f.product_key
GROUP BY
    p.management_group_id,
    p.first_category_id,
    p.second_category_id,
    p.third_category_id;

DROP VIEW IF EXISTS dw.v_dss_kpi_by_category;
DROP VIEW IF EXISTS dw.v_dss_kpi_by_day;
DROP VIEW IF EXISTS dw.v_dss_daily_decision_score;
DROP VIEW IF EXISTS dw.v_dss_hourly_demand_estimate;
DROP VIEW IF EXISTS dw.v_latest_model;

CREATE OR REPLACE VIEW dw.v_latest_model AS
SELECT model_key, model_name, model_version, training_start_date, training_end_date, created_at
FROM dw.dim_model
ORDER BY created_at DESC, model_key DESC
LIMIT 1;

CREATE OR REPLACE VIEW dw.v_dss_hourly_demand_estimate AS
WITH latest_model AS (
    SELECT model_key, model_name, model_version
    FROM dw.v_latest_model
), hourly_baseline AS (
    SELECT
        h.*,
        AVG(h.observed_sales_amount) FILTER (WHERE NOT h.stockout_flag)
            OVER (PARTITION BY h.store_key, h.product_key, h.time_key) AS store_product_hour_avg,
        AVG(h.observed_sales_amount) FILTER (WHERE NOT h.stockout_flag)
            OVER (PARTITION BY h.product_key, h.time_key) AS product_hour_avg,
        AVG(h.observed_sales_amount) FILTER (WHERE NOT h.stockout_flag)
            OVER (PARTITION BY h.product_key) AS product_avg
    FROM dw.fact_sales_inventory_hourly h
), heuristic_estimate AS (
    SELECT
        date_key,
        time_key,
        store_key,
        product_key,
        source_split,
        observed_sales_amount,
        stockout_flag,
        is_censored_observation,
        discount_rate,
        activity_flag,
        precpt,
        avg_temperature,
        avg_humidity,
        avg_wind_level,
        CASE
            WHEN stockout_flag THEN GREATEST(
                COALESCE(store_product_hour_avg, product_hour_avg, product_avg, observed_sales_amount),
                observed_sales_amount
            )
            ELSE observed_sales_amount
        END AS heuristic_true_demand,
        CASE
            WHEN stockout_flag THEN GREATEST(
                COALESCE(store_product_hour_avg, product_hour_avg, product_avg, observed_sales_amount) - observed_sales_amount,
                0
            )
            ELSE 0
        END AS heuristic_lost_sales
    FROM hourly_baseline
)
SELECT
    h.date_key,
    h.time_key,
    h.store_key,
    h.product_key,
    h.source_split,
    h.observed_sales_amount,
    h.stockout_flag,
    h.is_censored_observation,
    h.discount_rate,
    h.activity_flag,
    h.precpt,
    h.avg_temperature,
    h.avg_humidity,
    h.avg_wind_level,
    COALESCE(m.estimated_true_demand, h.heuristic_true_demand) AS estimated_true_demand,
    COALESCE(m.estimated_lost_sales, h.heuristic_lost_sales) AS estimated_lost_sales,
    lm.model_name,
    lm.model_version,
    CASE
        WHEN m.model_key IS NOT NULL THEN 'Trained model estimate from latest model in dw.dim_model.'
        WHEN h.stockout_flag THEN 'Fallback heuristic: stockout hour demand estimated from non-stockout sales for the same store/product/hour, then product/hour fallback.'
        ELSE 'Fallback heuristic: non-stockout hour uses observed sales as demand.'
    END AS estimate_explanation
FROM heuristic_estimate h
LEFT JOIN latest_model lm ON TRUE
LEFT JOIN dw.fact_demand_estimate_hourly m
    ON m.model_key = lm.model_key
    AND m.date_key = h.date_key
    AND m.time_key = h.time_key
    AND m.store_key = h.store_key
    AND m.product_key = h.product_key;

CREATE OR REPLACE VIEW dw.v_dss_daily_decision_score AS
WITH hourly_daily AS (
    SELECT
        date_key,
        store_key,
        product_key,
        MAX(model_name) AS model_name,
        MAX(model_version) AS model_version,
        SUM(estimated_true_demand) AS estimated_true_demand,
        SUM(estimated_lost_sales) AS estimated_lost_sales,
        SUM(CASE WHEN stockout_flag THEN 1 ELSE 0 END) AS stockout_hours_total_from_hourly
    FROM dw.v_dss_hourly_demand_estimate
    GROUP BY date_key, store_key, product_key
), enriched AS (
    SELECT
        d.full_date,
        city.city_id,
        store.store_id,
        product.product_id,
        product.management_group_id,
        product.first_category_id,
        product.second_category_id,
        product.third_category_id,
        f.date_key,
        f.store_key,
        f.product_key,
        f.source_split,
        h.model_name,
        h.model_version,
        f.observed_daily_sales_amount,
        COALESCE(h.estimated_true_demand, f.observed_daily_sales_amount) AS estimated_true_demand,
        COALESCE(h.estimated_lost_sales, 0) AS estimated_lost_sales,
        f.stockout_hours_6_22,
        f.stockout_hours_total,
        f.has_stockout,
        f.discount_rate,
        f.holiday_flag,
        f.activity_flag,
        f.precpt,
        f.avg_temperature,
        f.avg_humidity,
        f.avg_wind_level,
        AVG(f.observed_daily_sales_amount) OVER (PARTITION BY f.product_key) AS product_avg_daily_sales
    FROM dw.fact_sales_inventory_daily f
    JOIN dw.dim_date d ON d.date_key = f.date_key
    JOIN dw.dim_store store ON store.store_key = f.store_key
    JOIN dw.dim_city city ON city.city_key = store.city_key
    JOIN dw.dim_product product ON product.product_key = f.product_key
    LEFT JOIN hourly_daily h
        ON h.date_key = f.date_key
        AND h.store_key = f.store_key
        AND h.product_key = f.product_key
), scored AS (
    SELECT
        *,
        stockout_hours_6_22 / 16.0 AS stockout_rate_6_22,
        CASE
            WHEN estimated_true_demand > 0 THEN estimated_lost_sales / estimated_true_demand
            ELSE 0
        END AS demand_bias_rate,
        CASE
            WHEN has_stockout THEN 0
            WHEN product_avg_daily_sales > 0 THEN LEAST(
                1,
                GREATEST(0, 1 - observed_daily_sales_amount / (product_avg_daily_sales * 1.25))
            )
            WHEN observed_daily_sales_amount = 0 THEN 1
            ELSE 0
        END AS waste_risk_score
    FROM enriched
), final_score AS (
    SELECT
        *,
        LEAST(
            1,
            stockout_rate_6_22 * 0.55
            + demand_bias_rate * 0.35
            + CASE WHEN activity_flag <> 0 THEN 0.10 ELSE 0 END
        ) AS restock_urgency_score
    FROM scored
)
SELECT
    *,
    CASE
        WHEN restock_urgency_score >= 0.65 THEN 'Restock immediately'
        WHEN stockout_rate_6_22 >= 0.20 THEN 'Increase next order'
        WHEN waste_risk_score >= 0.70 THEN 'Reduce order or markdown'
        WHEN demand_bias_rate >= 0.20 THEN 'Review censored demand'
        ELSE 'Maintain plan'
    END AS decision_action,
    CASE
        WHEN restock_urgency_score >= 0.65 THEN 'High stockout/lost-sales signal; prioritize replenishment.'
        WHEN stockout_rate_6_22 >= 0.20 THEN 'Repeated business-hour stockouts indicate understocking.'
        WHEN waste_risk_score >= 0.70 THEN 'Low sales with no stockout suggests overstock or spoilage risk.'
        WHEN demand_bias_rate >= 0.20 THEN 'Observed sales likely understate true customer demand.'
        ELSE 'No strong intervention signal from current criteria.'
    END AS decision_reason
FROM final_score;

CREATE OR REPLACE VIEW dw.v_dss_kpi_by_day AS
SELECT
    full_date,
    COUNT(*) AS product_store_days,
    SUM(observed_daily_sales_amount) AS observed_sales_amount,
    SUM(estimated_true_demand) AS estimated_true_demand,
    SUM(estimated_lost_sales) AS estimated_lost_sales,
    AVG(stockout_rate_6_22) AS avg_stockout_rate_6_22,
    AVG(waste_risk_score) AS avg_waste_risk_score,
    AVG(demand_bias_rate) AS avg_demand_bias_rate,
    AVG(restock_urgency_score) AS avg_restock_urgency_score,
    COUNT(*) FILTER (WHERE decision_action = 'Restock immediately') AS immediate_restock_count,
    COUNT(*) FILTER (WHERE decision_action = 'Reduce order or markdown') AS waste_action_count
FROM dw.v_dss_daily_decision_score
GROUP BY full_date;

CREATE OR REPLACE VIEW dw.v_dss_kpi_by_category AS
SELECT
    management_group_id,
    first_category_id,
    second_category_id,
    third_category_id,
    COUNT(*) AS product_store_days,
    SUM(observed_daily_sales_amount) AS observed_sales_amount,
    SUM(estimated_true_demand) AS estimated_true_demand,
    SUM(estimated_lost_sales) AS estimated_lost_sales,
    AVG(stockout_rate_6_22) AS avg_stockout_rate_6_22,
    AVG(waste_risk_score) AS avg_waste_risk_score,
    AVG(demand_bias_rate) AS avg_demand_bias_rate,
    AVG(restock_urgency_score) AS avg_restock_urgency_score,
    COUNT(*) FILTER (WHERE decision_action = 'Restock immediately') AS immediate_restock_count,
    COUNT(*) FILTER (WHERE decision_action = 'Reduce order or markdown') AS waste_action_count
FROM dw.v_dss_daily_decision_score
GROUP BY
    management_group_id,
    first_category_id,
    second_category_id,
    third_category_id;
