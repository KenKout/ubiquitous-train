#!/usr/bin/env python3
from __future__ import annotations

import os
from datetime import date, timedelta
from html import escape
from typing import Any

import altair as alt
import pandas as pd
import psycopg
import streamlit as st


DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5433")),
    "dbname": os.getenv("PGDATABASE", "fresh_retail_dw"),
    "user": os.getenv("PGUSER", "warehouse"),
    "password": os.getenv("PGPASSWORD", "warehouse"),
}


def run_query(sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc.name for desc in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


@st.cache_data(ttl=60)
def load_data_status() -> pd.DataFrame:
    return run_query(
        """
        SELECT
            (SELECT COUNT(*) FROM staging.fresh_retail_observation_day) AS staging_rows,
            (SELECT COUNT(*) FROM dw.fact_sales_inventory_daily) AS daily_fact_rows,
            (SELECT COUNT(*) FROM dw.fact_sales_inventory_hourly) AS hourly_fact_rows,
            (SELECT COUNT(*) FROM dw.fact_demand_estimate_hourly) AS demand_estimate_rows,
            (SELECT COUNT(*) FROM dw.fact_replenishment_recommendation_daily) AS recommendation_rows,
            (SELECT MIN(full_date) FROM dw.dim_date) AS min_date,
            (SELECT MAX(full_date) FROM dw.dim_date) AS max_date
        """
    )


@st.cache_data(ttl=60)
def load_model_status() -> pd.DataFrame:
    return run_query(
        """
        SELECT
            model_key,
            model_name,
            model_version,
            training_start_date,
            training_end_date,
            created_at,
            demand_estimate_rows,
            recommendation_rows
        FROM dw.v_latest_model_with_predictions
        """
    )


@st.cache_data(ttl=60)
def load_model_quality() -> pd.DataFrame:
    return run_query(
        """
        SELECT
            q.model_key,
            q.model_name,
            q.model_version,
            q.eval_rows,
            q.mae,
            q.rmse,
            q.wmape,
            q.bias,
            q.calibration_factor,
            (
                SELECT MAX(e.metric_value)
                FROM dw.fact_model_evaluation e
                WHERE e.model_key = q.model_key
                  AND e.evaluation_split = 'eval'
                  AND e.metric_name = 'raw_calibration_factor'
            ) AS raw_calibration_factor,
            q.prediction_rows,
            q.uncalibrated_wmape,
            q.uncalibrated_bias,
            q.metrics_loaded_at
        FROM dw.v_model_quality_summary q
        JOIN dw.v_latest_model_with_predictions lm ON lm.model_key = q.model_key
        """
    )


@st.cache_data(ttl=60)
def load_quality_checks() -> pd.DataFrame:
    return run_query(
        """
        SELECT check_name, failed_rows, severity, details
        FROM dw.v_data_quality_checks
        ORDER BY severity, check_name
        """
    )


@st.cache_data(ttl=60)
def load_filter_options() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    categories = run_query("SELECT DISTINCT first_category_id FROM dw.dim_product ORDER BY first_category_id")
    stores = run_query("SELECT store_id FROM dw.dim_store ORDER BY store_id")
    products = run_query("SELECT product_id FROM dw.dim_product ORDER BY product_id")
    return categories, stores, products


def build_filters(
    start_date: date,
    end_date: date,
    first_category_id: int | None,
    store_id: int | None,
    product_id: int | None,
    action: str,
) -> tuple[str, tuple[Any, ...]]:
    clauses = ["full_date BETWEEN %s AND %s"]
    params: list[Any] = [start_date, end_date]

    if first_category_id is not None:
        clauses.append("first_category_id = %s")
        params.append(first_category_id)
    if store_id is not None:
        clauses.append("store_id = %s")
        params.append(store_id)
    if product_id is not None:
        clauses.append("product_id = %s")
        params.append(product_id)
    if action != "All actions":
        clauses.append("decision_action = %s")
        params.append(action)

    return " AND ".join(clauses), tuple(params)


@st.cache_data(ttl=60)
def load_summary(where_sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
    return run_query(
        f"""
        SELECT
            COUNT(*) AS product_store_days,
            SUM(observed_daily_sales_amount) AS observed_sales_amount,
            SUM(estimated_true_demand) AS estimated_true_demand,
            SUM(estimated_lost_sales) AS estimated_lost_sales,
            AVG(stockout_rate_6_22) AS avg_stockout_rate_6_22,
            AVG(waste_risk_score) AS avg_waste_risk_score,
            AVG(demand_bias_rate) AS avg_demand_bias_rate,
            AVG(restock_urgency_score) AS avg_restock_urgency_score,
            SUM(recommended_order_qty) AS recommended_order_qty,
            COUNT(*) FILTER (WHERE decision_action = 'Restock immediately') AS immediate_restock_count,
            COUNT(*) FILTER (WHERE decision_action = 'Increase next order') AS increase_order_count,
            COUNT(*) FILTER (WHERE decision_action = 'Reduce order or markdown') AS reduce_order_count,
            COUNT(*) FILTER (WHERE decision_action = 'Review censored demand') AS review_bias_count
        FROM dw.v_dss_daily_decision_score
        WHERE {where_sql}
        """,
        params,
    )


@st.cache_data(ttl=60)
def load_trend(where_sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
    return run_query(
        f"""
        SELECT
            full_date,
            SUM(observed_daily_sales_amount) AS observed_sales_amount,
            SUM(estimated_lost_sales) AS estimated_lost_sales,
            AVG(stockout_rate_6_22) AS stockout_rate_6_22,
            AVG(waste_risk_score) AS waste_risk_score,
            AVG(demand_bias_rate) AS demand_bias_rate,
            AVG(restock_urgency_score) AS restock_urgency_score
        FROM dw.v_dss_daily_decision_score
        WHERE {where_sql}
        GROUP BY full_date
        ORDER BY full_date
        """,
        params,
    )


@st.cache_data(ttl=60)
def load_category_summary(where_sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
    return run_query(
        f"""
        SELECT
            first_category_id,
            COUNT(*) AS product_store_days,
            SUM(observed_daily_sales_amount) AS observed_sales_amount,
            SUM(estimated_lost_sales) AS estimated_lost_sales,
            AVG(stockout_rate_6_22) AS stockout_rate_6_22,
            AVG(waste_risk_score) AS waste_risk_score,
            AVG(restock_urgency_score) AS restock_urgency_score
        FROM dw.v_dss_daily_decision_score
        WHERE {where_sql}
        GROUP BY first_category_id
        ORDER BY restock_urgency_score DESC, estimated_lost_sales DESC
        LIMIT 20
        """,
        params,
    )


@st.cache_data(ttl=60)
def load_recommendations(where_sql: str, params: tuple[Any, ...], queue_view: str) -> pd.DataFrame:
    if queue_view == "Diverse action sample":
        return run_query(
            f"""
            WITH filtered AS (
                SELECT
                    full_date,
                    city_id,
                    store_id,
                    product_id,
                    first_category_id,
                    model_name,
                    model_version,
                    estimate_source,
                    observed_daily_sales_amount,
                    estimated_true_demand,
                    estimated_lost_sales,
                    recommended_order_qty,
                    stockout_hours_6_22,
                    stockout_rate_6_22,
                    demand_bias_rate,
                    waste_risk_score,
                    restock_urgency_score,
                    service_level_target,
                    expected_waste_qty,
                    activity_flag,
                    decision_action,
                    decision_reason,
                    inventory_proxy_note
                FROM dw.v_dss_daily_decision_score
                WHERE {where_sql}
            ), ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY decision_action
                        ORDER BY restock_urgency_score DESC, estimated_lost_sales DESC, stockout_hours_6_22 DESC
                    ) AS action_rank
                FROM filtered
            )
            SELECT *
            FROM ranked
            WHERE action_rank <= 40
            ORDER BY
                CASE decision_action
                    WHEN 'Restock immediately' THEN 1
                    WHEN 'Increase next order' THEN 2
                    WHEN 'Review censored demand' THEN 3
                    WHEN 'Reduce order or markdown' THEN 4
                    ELSE 5
                END,
                restock_urgency_score DESC,
                estimated_lost_sales DESC
            LIMIT 200
            """,
            params,
        ).drop(columns=["action_rank"], errors="ignore")

    extra_filter = ""
    if queue_view == "Exclude full-stockout rows":
        extra_filter = "AND NOT (stockout_hours_6_22 = 16 AND observed_daily_sales_amount = 0)"
    return run_query(
        f"""
        SELECT
            full_date,
            city_id,
            store_id,
            product_id,
            first_category_id,
            model_name,
            model_version,
            estimate_source,
            observed_daily_sales_amount,
            estimated_true_demand,
            estimated_lost_sales,
            recommended_order_qty,
            stockout_hours_6_22,
            stockout_rate_6_22,
            demand_bias_rate,
            waste_risk_score,
            restock_urgency_score,
            service_level_target,
            expected_waste_qty,
            activity_flag,
            decision_action,
            decision_reason,
            inventory_proxy_note
        FROM dw.v_dss_daily_decision_score
        WHERE {where_sql}
          {extra_filter}
        ORDER BY restock_urgency_score DESC, estimated_lost_sales DESC, stockout_hours_6_22 DESC
        LIMIT 200
        """,
        params,
    )


@st.cache_data(ttl=60)
def load_hourly_drilldown(full_date: date, store_id: int, product_id: int) -> pd.DataFrame:
    return run_query(
        """
        WITH latest_model AS (
            SELECT model_key, model_name, model_version
            FROM dw.v_latest_model_with_predictions
        )
        SELECT
            t.hour_of_day,
            f.observed_sales_amount,
            COALESCE(e.estimated_true_demand, f.observed_sales_amount) AS estimated_true_demand,
            COALESCE(e.estimated_lost_sales, 0) AS estimated_lost_sales,
            CASE WHEN f.stockout_flag THEN 1 ELSE 0 END AS stockout_flag,
            CASE WHEN e.model_key IS NOT NULL THEN 'model' ELSE 'observed' END AS estimate_source,
            CASE
                WHEN e.model_key IS NOT NULL THEN 'Latest model prediction fact joined directly for selected store-product-date.'
                WHEN f.stockout_flag THEN 'Hourly stockout observed, but no model prediction fact exists for this row.'
                ELSE 'Non-stockout hour uses observed sales as demand.'
            END AS estimate_explanation
        FROM dw.fact_sales_inventory_hourly f
        JOIN dw.dim_date d ON d.date_key = f.date_key
        JOIN dw.dim_time t ON t.time_key = f.time_key
        JOIN dw.dim_store s ON s.store_key = f.store_key
        JOIN dw.dim_product p ON p.product_key = f.product_key
        LEFT JOIN latest_model lm ON TRUE
        LEFT JOIN dw.fact_demand_estimate_hourly e
            ON e.model_key = lm.model_key
            AND e.date_key = f.date_key
            AND e.time_key = f.time_key
            AND e.store_key = f.store_key
            AND e.product_key = f.product_key
        WHERE d.full_date = %s
          AND s.store_id = %s
          AND p.product_id = %s
        ORDER BY t.hour_of_day
        """,
        (full_date, store_id, product_id),
    )


@st.cache_data(ttl=60)
def load_what_if(
    where_sql: str,
    params: tuple[Any, ...],
    urgency_threshold: float,
    service_level_target: float,
) -> pd.DataFrame:
    return run_query(
        f"""
        SELECT
            COUNT(*) AS product_store_days,
            COUNT(*) FILTER (WHERE restock_urgency_score >= %s) AS restock_count,
            COUNT(*) FILTER (
                WHERE restock_urgency_score < %s
                  AND stockout_rate_6_22 >= 0.20
            ) AS increase_order_count,
            COUNT(*) FILTER (
                WHERE restock_urgency_score < %s
                  AND stockout_rate_6_22 < 0.20
                  AND waste_risk_score >= 0.70
            ) AS markdown_count,
            COUNT(*) FILTER (
                WHERE restock_urgency_score < %s
                  AND stockout_rate_6_22 < 0.20
                  AND waste_risk_score < 0.70
                  AND demand_bias_rate >= 0.20
            ) AS review_bias_count,
            SUM(
                CASE
                    WHEN restock_urgency_score >= %s
                    THEN recommended_order_qty * (%s / NULLIF(service_level_target, 0))
                    ELSE 0
                END
            ) AS restock_order_qty_proxy
        FROM dw.v_dss_daily_decision_score
        WHERE {where_sql}
        """,
        (
            urgency_threshold,
            urgency_threshold,
            urgency_threshold,
            urgency_threshold,
            urgency_threshold,
            service_level_target,
            *params,
        ),
    )


def optional_int_select(label: str, values: pd.Series) -> int | None:
    options = ["All"] + [int(value) for value in values.dropna().tolist()]
    selected = st.sidebar.selectbox(label, options)
    return None if selected == "All" else int(selected)


def metric_value(value: Any, suffix: str = "") -> str:
    if pd.isna(value):
        return "0" + suffix
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{float(value):,.2f}{suffix}"


def metric_value_precise(value: Any, digits: int = 4, suffix: str = "") -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):,.{digits}f}{suffix}"


def metric_percent(value: Any, digits: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:,.{digits}f}%"


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    return result


def full_stockout_zero_sales_count(df: pd.DataFrame) -> int:
    if df.empty or "stockout_hours_6_22" not in df.columns or "observed_daily_sales_amount" not in df.columns:
        return 0
    stockout_hours = pd.to_numeric(df["stockout_hours_6_22"], errors="coerce").fillna(0)
    observed_sales = pd.to_numeric(df["observed_daily_sales_amount"], errors="coerce").fillna(0)
    return int(((stockout_hours >= 16) & (observed_sales <= 0)).sum())


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;600;700&family=Fira+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Fira Sans', sans-serif;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eff6ff 0%, #f8fafc 46%, #ffffff 100%);
            border-right: 1px solid #dbeafe;
        }

        .hero-panel {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.22), transparent 34%),
                linear-gradient(135deg, #0f172a 0%, #1e3a8a 58%, #1d4ed8 100%);
            color: #ffffff;
            border-radius: 24px;
            padding: 28px 30px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 24px 70px rgba(30, 64, 175, 0.26);
            margin-bottom: 18px;
        }

        .hero-kicker {
            color: #bfdbfe;
            font-family: 'Fira Code', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .hero-title {
            font-size: 2.2rem;
            line-height: 1.1;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .hero-copy {
            color: #dbeafe;
            font-size: 1rem;
            max-width: 980px;
            margin-bottom: 18px;
        }

        .status-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #dbeafe;
            background: rgba(255, 255, 255, 0.11);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 0.86rem;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 18px 18px 16px;
            min-height: 128px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.07);
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 5px;
            background: var(--accent);
        }

        .metric-label {
            color: #475569;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .metric-value {
            color: #0f172a;
            font-family: 'Fira Code', monospace;
            font-size: 1.55rem;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .metric-caption {
            color: #64748b;
            font-size: 0.82rem;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 12px 36px rgba(15, 23, 42, 0.06);
            margin-bottom: 16px;
        }

        .section-title {
            color: #0f172a;
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 2px;
        }

        .section-subtitle {
            color: #64748b;
            font-size: 0.88rem;
            margin-bottom: 14px;
        }

        .action-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
        }

        .action-card {
            border-radius: 16px;
            padding: 14px;
            border: 1px solid #e2e8f0;
            background: #f8fafc;
        }

        .action-label {
            color: #475569;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .action-value {
            color: #0f172a;
            font-family: 'Fira Code', monospace;
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 6px;
        }

        .insight-panel {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 54%, #334155 100%);
            color: #f8fafc;
            border-radius: 22px;
            padding: 22px;
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 22px 58px rgba(15, 23, 42, 0.18);
            margin-bottom: 16px;
        }

        .insight-kicker {
            color: #93c5fd;
            font-family: 'Fira Code', monospace;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }

        .insight-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .insight-copy {
            color: #cbd5e1;
            font-size: 0.95rem;
            max-width: 980px;
        }

        .lane-grid,
        .priority-grid,
        .category-card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin: 14px 0 20px;
        }

        .lane-card,
        .priority-card,
        .category-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
            padding: 16px;
            position: relative;
            overflow: hidden;
        }

        .lane-card::before,
        .priority-card::before,
        .category-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: var(--accent);
        }

        .card-eyebrow {
            color: #64748b;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }

        .card-title {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .card-subtitle {
            color: #64748b;
            font-size: 0.84rem;
            margin-bottom: 12px;
        }

        .card-stat-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 12px;
        }

        .mini-stat {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 10px;
        }

        .mini-label {
            color: #64748b;
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
        }

        .mini-value {
            color: #0f172a;
            font-family: 'Fira Code', monospace;
            font-weight: 700;
            font-size: 0.95rem;
            margin-top: 3px;
        }

        .signal-row {
            margin-top: 8px;
        }

        .signal-head {
            display: flex;
            justify-content: space-between;
            color: #475569;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .signal-track {
            height: 8px;
            border-radius: 999px;
            background: #e2e8f0;
            overflow: hidden;
        }

        .signal-fill {
            height: 100%;
            border-radius: 999px;
            background: var(--accent);
            width: var(--width);
        }

        .reason-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            color: #475569;
            font-size: 0.82rem;
            padding: 10px 12px;
            min-height: 58px;
        }

        .hour-strip {
            display: grid;
            grid-template-columns: repeat(24, minmax(0, 1fr));
            gap: 4px;
            margin: 12px 0 16px;
        }

        .hour-cell {
            min-height: 42px;
            border-radius: 9px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Fira Code', monospace;
            font-size: 0.72rem;
            font-weight: 700;
            color: #0f172a;
            border: 1px solid #e2e8f0;
            background: var(--cell-bg);
        }

        .soft-note {
            color: #64748b;
            font-size: 0.84rem;
            margin-top: -4px;
            margin-bottom: 12px;
        }

        .evidence-hero {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 14px;
            margin: 8px 0 16px;
        }

        .selection-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 12px 0 16px;
        }

        .selection-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            border-top: 4px solid var(--accent);
        }

        .selection-label {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 7px;
        }

        .selection-value {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }

        .evidence-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
        }

        .evidence-kicker {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .evidence-title {
            color: #0f172a;
            font-size: 1.18rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .evidence-copy {
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .evidence-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }

        .evidence-pill {
            border-radius: 999px;
            border: 1px solid #dbeafe;
            background: #eff6ff;
            color: #1e3a8a;
            padding: 6px 10px;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .legend-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 8px 0 14px;
        }

        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: #475569;
            font-size: 0.8rem;
            font-weight: 700;
        }

        .legend-swatch {
            width: 18px;
            height: 8px;
            border-radius: 999px;
            background: var(--swatch);
        }

        .evidence-table-note {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 8px;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stTabs"] button {
            font-weight: 700;
        }

        @media (max-width: 900px) {
            .hero-title { font-size: 1.65rem; }
            .action-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .evidence-hero { grid-template-columns: 1fr; }
            .selection-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .lane-grid,
            .priority-grid,
            .category-card-grid { grid-template-columns: 1fr; }
        }

        @media (max-width: 520px) {
            .action-grid { grid-template-columns: 1fr; }
            .hero-panel { padding: 22px; }
            .selection-grid { grid-template-columns: 1fr; }
            .hour-strip { grid-template-columns: repeat(12, minmax(0, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: Any, caption: str, accent: str = "#1E40AF") -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="--accent: {accent};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def render_action_grid(summary: pd.Series) -> None:
    items = [
        ("Restock now", summary["immediate_restock_count"], "#dc2626"),
        ("Increase order", summary["increase_order_count"], "#f59e0b"),
        ("Markdown", summary["reduce_order_count"], "#7c3aed"),
        ("Review demand", summary["review_bias_count"], "#2563eb"),
    ]
    cards = "".join(
        f'<div class="action-card" style="border-top: 4px solid {color};">'
        f'<div class="action-label">{label}</div>'
        f'<div class="action-value">{metric_value(value)}</div>'
        f'</div>'
        for label, value, color in items
    )
    st.markdown(
        f'<div class="section-card">'
        f'<div class="section-title">Action Mix</div>'
        f'<div class="section-subtitle">How the current filter set distributes into operational decisions.</div>'
        f'<div class="action-grid">{cards}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


ACTION_META = {
    "Restock immediately": {"accent": "#dc2626", "label": "Restock"},
    "Increase next order": {"accent": "#f59e0b", "label": "Increase"},
    "Reduce order or markdown": {"accent": "#7c3aed", "label": "Markdown"},
    "Review censored demand": {"accent": "#2563eb", "label": "Review"},
    "Maintain plan": {"accent": "#64748b", "label": "Maintain"},
}

ACTION_DISPLAY = {
    "All actions": "All actions",
    "Restock immediately": "Restock immediately",
    "Increase next order": "Increase next replenishment",
    "Reduce order or markdown": "Reduce order / markdown",
    "Review censored demand": "Review censored demand",
    "Maintain plan": "Maintain plan",
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp01(value: Any) -> float:
    return min(max(safe_float(value), 0.0), 1.0)


def percent_label(value: Any, digits: int = 0) -> str:
    return f"{clamp01(value) * 100:,.{digits}f}%"


def action_accent(action: Any) -> str:
    return ACTION_META.get(str(action), {"accent": "#475569"})["accent"]


def display_action_label(action: Any) -> str:
    return ACTION_DISPLAY.get(str(action), str(action))


def compact_html(html: str) -> str:
    return " ".join(line.strip() for line in html.splitlines() if line.strip())


def render_html(html: str) -> None:
    st.markdown(compact_html(html), unsafe_allow_html=True)


def signal_bar(label: str, value: Any, color: str) -> str:
    pct = percent_label(value)
    return (
        f'<div class="signal-row">'
        f'<div class="signal-head"><span>{escape(label)}</span><span>{pct}</span></div>'
        f'<div class="signal-track"><div class="signal-fill" style="--accent: {color}; --width: {pct};"></div></div>'
        f'</div>'
    )


def render_queue_insight(summary: pd.Series, recommendations: pd.DataFrame, queue_view: str) -> None:
    full_stockout_count = full_stockout_zero_sales_count(recommendations)
    total_visible = len(recommendations)
    estimated_lost_sales = safe_float(summary.get("estimated_lost_sales"))
    immediate_count = int(safe_float(summary.get("immediate_restock_count")))
    urgency = percent_label(summary.get("avg_restock_urgency_score"), 1)
    if total_visible and full_stockout_count == total_visible:
        title = "Your top queue is dominated by true full-stockout days"
        copy = "Every visible priority card has zero observed sales and stockout flags across all business hours. That is why the stockout and lost-demand share signals saturate at 100%."
    elif full_stockout_count:
        title = "Mixed queue with a full-stockout cluster"
        copy = f"{full_stockout_count:,} of {total_visible:,} visible items are full business-hour stockouts. Use the cards below to separate structural stockout risk from softer review/markdown actions."
    else:
        title = "Queue is no longer saturated by full-stockout rows"
        copy = "The visible recommendations contain partial-stockout or non-stockout cases, which should make the signal bars more varied and easier to compare."
    render_html(
        f"""
        <div class="insight-panel">
            <div class="insight-kicker">{escape(queue_view)} view</div>
            <div class="insight-title">{escape(title)}</div>
            <div class="insight-copy">
                {escape(copy)} Current filter window contains {immediate_count:,} immediate-restock candidates,
                {metric_value(estimated_lost_sales)} estimated lost sales, and {urgency} average priority.
            </div>
        </div>
        """
    )


def render_action_lanes(recommendations: pd.DataFrame) -> None:
    if recommendations.empty:
        return
    grouped = (
        recommendations.groupby("decision_action", dropna=False)
        .agg(
            rows=("decision_action", "size"),
            lost_sales=("estimated_lost_sales", "sum"),
            avg_priority=("restock_urgency_score", "mean"),
            avg_stockout=("stockout_rate_6_22", "mean"),
        )
        .reset_index()
    )
    action_order = {name: index for index, name in enumerate(ACTION_META)}
    grouped["sort_order"] = grouped["decision_action"].map(action_order).fillna(99)
    grouped = grouped.sort_values(["sort_order", "lost_sales"], ascending=[True, False])
    cards = []
    for _, row in grouped.iterrows():
        action_name = str(row["decision_action"])
        action_label = display_action_label(action_name)
        accent = action_accent(action_name)
        cards.append(
            f"""
            <div class="lane-card" style="--accent: {accent};">
                <div class="card-eyebrow">Action lane</div>
                <div class="card-title">{escape(action_label)}</div>
                <div class="card-subtitle">{int(row['rows']):,} visible recommendations</div>
                <div class="card-stat-row">
                    <div class="mini-stat"><div class="mini-label">Lost sales</div><div class="mini-value">{metric_value(row['lost_sales'])}</div></div>
                    <div class="mini-stat"><div class="mini-label">Priority</div><div class="mini-value">{percent_label(row['avg_priority'])}</div></div>
                    <div class="mini-stat"><div class="mini-label">Stockout</div><div class="mini-value">{percent_label(row['avg_stockout'])}</div></div>
                </div>
                {signal_bar('Lane priority', row['avg_priority'], accent)}
            </div>
            """
        )
    render_html(
        f'<div class="section-title">Action Lanes</div>'
        f'<div class="section-subtitle">A management view of what kind of intervention the queue is asking for.</div>'
        f'<div class="lane-grid">{"".join(cards)}</div>'
    )


def render_priority_cards(recommendations: pd.DataFrame, limit: int = 6) -> None:
    if recommendations.empty:
        return
    cards = []
    for index, (_, row) in enumerate(recommendations.head(limit).iterrows(), start=1):
        action_name = str(row["decision_action"])
        action_label = display_action_label(action_name)
        accent = action_accent(action_name)
        title = f"#{index} Store {int(row['store_id'])} / Product {int(row['product_id'])}"
        subtitle = f"{row['full_date']} · category {int(row['first_category_id'])} · {action_label}"
        
        # Tính breakdown urgency score
        stockout_rate = safe_float(row['stockout_rate_6_22'])
        bias_rate = safe_float(row['demand_bias_rate'])
        activity = safe_float(row['activity_flag'])
        
        stockout_contrib = stockout_rate * 0.55
        bias_contrib = bias_rate * 0.35
        activity_contrib = 0.10 if activity != 0 else 0.0
        total_urgency = min(1.0, stockout_contrib + bias_contrib + activity_contrib)
        
        urgency_breakdown = (
            f"Stockout rate ({stockout_rate:.1%}) × 0.55 = {stockout_contrib:.3f}  "
            f"|  Bias rate ({bias_rate:.1%}) × 0.35 = {bias_contrib:.3f}  "
            f"|  Activity ({int(activity)}) × 0.10 = {activity_contrib:.3f}  "
            f"|  Total = {total_urgency:.3f}"
        )
        
        cards.append(
            f"""
            <div class="priority-card" style="--accent: {accent};">
                <div class="card-eyebrow">Priority card</div>
                <div class="card-title">{escape(title)}</div>
                <div class="card-subtitle">{escape(subtitle)}</div>
                <div class="card-stat-row">
                    <div class="mini-stat"><div class="mini-label">Lost sales</div><div class="mini-value">{metric_value(row['estimated_lost_sales'])}</div></div>
                    <div class="mini-stat"><div class="mini-label">Order proxy</div><div class="mini-value">{metric_value(row['recommended_order_qty'])}</div></div>
                    <div class="mini-stat"><div class="mini-label">Stockout hrs</div><div class="mini-value">{int(safe_float(row['stockout_hours_6_22']))}/16</div></div>
                </div>
                {signal_bar('Business-hour stockout', row['stockout_rate_6_22'], '#dc2626')}
                {signal_bar('Lost demand share', row['demand_bias_rate'], '#ea580c')}
                {signal_bar('Priority score', row['restock_urgency_score'], accent)}
                <div class="urgency-breakdown" style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem; padding: 0.5rem; background: #f3f4f6; border-radius: 0.25rem;">
                    <strong>Urgency breakdown:</strong><br/>
                    {escape(urgency_breakdown)}
                </div>
                <div class="reason-box">{escape(str(row['decision_reason']))}</div>
            </div>
            """
        )
    render_html(
        f'<div class="section-title">Top Intervention Cards</div>'
        f'<div class="section-subtitle">The queue as operational cards instead of raw rows. Use these to pick what to inspect in hourly evidence.</div>'
        f'<div class="priority-grid">{"".join(cards)}</div>'
    )


def render_category_cards(category_summary: pd.DataFrame, limit: int = 6) -> None:
    if category_summary.empty:
        return
    cards = []
    for _, row in category_summary.head(limit).iterrows():
        urgency = clamp01(row["restock_urgency_score"])
        accent = "#dc2626" if urgency >= 0.65 else "#f59e0b" if urgency >= 0.35 else "#2563eb"
        cards.append(
            f"""
            <div class="category-card" style="--accent: {accent};">
                <div class="card-eyebrow">Category pressure</div>
                <div class="card-title">Category {int(row['first_category_id'])}</div>
                <div class="card-subtitle">{int(row['product_store_days']):,} product-store-days</div>
                <div class="card-stat-row">
                    <div class="mini-stat"><div class="mini-label">Lost sales</div><div class="mini-value">{metric_value(row['estimated_lost_sales'])}</div></div>
                    <div class="mini-stat"><div class="mini-label">Urgency</div><div class="mini-value">{percent_label(row['restock_urgency_score'])}</div></div>
                    <div class="mini-stat"><div class="mini-label">Stockout</div><div class="mini-value">{percent_label(row['stockout_rate_6_22'])}</div></div>
                </div>
                {signal_bar('Waste risk', row['waste_risk_score'], '#7c3aed')}
            </div>
            """
        )
    render_html(f'<div class="category-card-grid">{"".join(cards)}</div>')


def render_selection_overview(selected_row: pd.Series) -> None:
    action_label = display_action_label(selected_row["decision_action"])
    accent = action_accent(selected_row["decision_action"])
    
    # Tính breakdown urgency score
    stockout_rate = safe_float(selected_row['stockout_rate_6_22'])
    bias_rate = safe_float(selected_row['demand_bias_rate'])
    activity = safe_float(selected_row['activity_flag'])
    
    stockout_contrib = stockout_rate * 0.55
    bias_contrib = bias_rate * 0.35
    activity_contrib = 0.10 if activity != 0 else 0.0
    total_urgency = min(1.0, stockout_contrib + bias_contrib + activity_contrib)
    
    urgency_detail = (
        f"Stockout: {stockout_rate:.1%} × 0.55 = {stockout_contrib:.3f} | "
        f"Bias: {bias_rate:.1%} × 0.35 = {bias_contrib:.3f} | "
        f"Activity: {int(activity)} × 0.10 = {activity_contrib:.3f}"
    )
    
    cards = [
        ("Store", str(int(selected_row["store_id"])), "#1e40af"),
        ("Product", str(int(selected_row["product_id"])), "#0f766e"),
        ("Action", action_label, accent),
        ("Urgency", metric_value(safe_float(selected_row["restock_urgency_score"]) * 100, "%"), "#7c3aed"),
    ]
    html = "".join(
        f"""
        <div class="selection-card" style="--accent: {color};">
            <div class="selection-label">{escape(label)}</div>
            <div class="selection-value">{escape(value)}</div>
        </div>
        """
        for label, value, color in cards
    )
    render_html(
        f'<div class="selection-grid">{html}</div>'
        f'<div style="margin-top: 0.75rem; padding: 0.75rem; background: #f3f4f6; border-radius: 0.5rem; font-size: 0.8rem; color: #374151;">'
        f'<strong>Urgency Score Breakdown:</strong> {escape(urgency_detail)}'
        f'</div>'
    )


def format_hour_range(hours: list[int]) -> str:
    if not hours:
        return "none"
    hours = sorted(hours)
    ranges: list[str] = []
    start = previous = hours[0]
    for hour in hours[1:]:
        if hour == previous + 1:
            previous = hour
            continue
        ranges.append(f"{start}:00" if start == previous else f"{start}:00-{previous}:00")
        start = previous = hour
    ranges.append(f"{start}:00" if start == previous else f"{start}:00-{previous}:00")
    return ", ".join(ranges)


def hourly_status(row: pd.Series) -> str:
    stockout = safe_float(row.get("stockout_flag")) >= 1
    lost = safe_float(row.get("estimated_lost_sales"))
    observed = safe_float(row.get("observed_sales_amount"))
    if stockout and lost > 0:
        return "Stockout, demand recovered"
    if stockout:
        return "Stockout observed"
    if observed > 0:
        return "Observed sale"
    return "No sale observed"


def prepare_hourly_evidence_table(hourly_numeric: pd.DataFrame) -> pd.DataFrame:
    evidence = hourly_numeric.copy()
    evidence["hour"] = evidence["hour_of_day"].astype(int).map(lambda hour: f"{hour:02d}:00")
    evidence["status"] = evidence.apply(hourly_status, axis=1)
    evidence["recovered_gap"] = (evidence["estimated_true_demand"] - evidence["observed_sales_amount"]).clip(lower=0)
    evidence["stockout_flag"] = evidence["stockout_flag"].map(lambda value: safe_float(value) >= 1)
    return evidence[
        [
            "hour",
            "status",
            "observed_sales_amount",
            "estimated_true_demand",
            "estimated_lost_sales",
            "stockout_flag",
            "estimate_source",
        ]
    ]


def render_hourly_evidence_chart(hourly_numeric: pd.DataFrame) -> None:
    chart_df = hourly_numeric.copy()
    chart_df["hour_label"] = chart_df["hour_of_day"].astype(int).map(lambda hour: f"{hour:02d}:00")
    chart_df["stockout_status"] = chart_df["stockout_flag"].map(lambda value: "Stockout" if safe_float(value) >= 1 else "Available")
    line_df = chart_df.melt(
        id_vars=["hour_of_day", "hour_label", "stockout_status", "estimated_lost_sales"],
        value_vars=["observed_sales_amount", "estimated_true_demand"],
        var_name="series",
        value_name="amount",
    )
    line_df["series"] = line_df["series"].map(
        {
            "observed_sales_amount": "Observed sales",
            "estimated_true_demand": "Estimated demand",
        }
    )

    x_axis = alt.X("hour_label:N", title="Hour", sort=chart_df["hour_label"].tolist(), axis=alt.Axis(labelAngle=0))
    lost_bars = (
        alt.Chart(chart_df)
        .mark_bar(opacity=0.44, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=x_axis,
            y=alt.Y("estimated_lost_sales:Q", title="Sales / demand units"),
            color=alt.value("#f59e0b"),
            tooltip=[
                alt.Tooltip("hour_label:N", title="Hour"),
                alt.Tooltip("stockout_status:N", title="Source status"),
                alt.Tooltip("estimated_lost_sales:Q", title="Recovered lost sales", format=".4f"),
            ],
        )
    )
    demand_lines = (
        alt.Chart(line_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=x_axis,
            y=alt.Y("amount:Q", title="Sales / demand units"),
            color=alt.Color(
                "series:N",
                title="Demand signal",
                scale=alt.Scale(domain=["Observed sales", "Estimated demand"], range=["#475569", "#2563eb"]),
            ),
            strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=["Observed sales", "Estimated demand"], range=[[1, 0], [6, 3]]), legend=None),
            tooltip=[
                alt.Tooltip("hour_label:N", title="Hour"),
                alt.Tooltip("stockout_status:N", title="Source status"),
                alt.Tooltip("series:N", title="Signal"),
                alt.Tooltip("amount:Q", title="Value", format=".4f"),
            ],
        )
    )
    chart = (lost_bars + demand_lines).resolve_scale(y="shared").properties(height=330)
    st.altair_chart(chart, use_container_width=True)


def render_hourly_story(selected_row: pd.Series, hourly_numeric: pd.DataFrame) -> None:
    stockout_hours = int(pd.to_numeric(hourly_numeric["stockout_flag"], errors="coerce").fillna(0).sum())
    business_hours = hourly_numeric[(hourly_numeric["hour_of_day"] >= 6) & (hourly_numeric["hour_of_day"] <= 21)]
    business_stockout_hours = int(pd.to_numeric(business_hours["stockout_flag"], errors="coerce").fillna(0).sum()) if not business_hours.empty else 0
    lost_sales = safe_float(hourly_numeric["estimated_lost_sales"].sum())
    observed_sales = safe_float(hourly_numeric["observed_sales_amount"].sum())
    estimated_demand = safe_float(hourly_numeric["estimated_true_demand"].sum())
    peak_lost = hourly_numeric.loc[hourly_numeric["estimated_lost_sales"].idxmax()]
    peak_hour = int(peak_lost["hour_of_day"])
    stockout_hour_list = hourly_numeric.loc[hourly_numeric["stockout_flag"] >= 1, "hour_of_day"].astype(int).tolist()
    lost_hour_list = hourly_numeric.loc[hourly_numeric["estimated_lost_sales"] > 0, "hour_of_day"].astype(int).tolist()
    action_label = display_action_label(selected_row["decision_action"])
    accent = action_accent(selected_row["decision_action"])

    if stockout_hours == 0:
        readout = "No source stockout hours are present for this store-product-day. The recommendation is driven by non-stockout demand, waste, or queue context."
    elif lost_sales <= 0:
        readout = "Source stockout flags are present, but the model/fallback did not recover material lost sales for those hours. Treat this as a stockout evidence check."
    else:
        readout = (
            f"Stockout is observed during {format_hour_range(stockout_hour_list)}. "
            f"The model recovers lost demand during {format_hour_range(lost_hour_list)}, with the peak lost-sales signal at {peak_hour}:00."
        )

    explanation = (
        f"Observed sales total {metric_value(observed_sales)}, estimated demand totals {metric_value(estimated_demand)}, "
        f"and the recovered gap is {metric_value(lost_sales)}. This supports the action: {action_label}."
    )
    cells = []
    for _, row in hourly_numeric.iterrows():
        hour = int(row["hour_of_day"])
        stockout = safe_float(row["stockout_flag"]) >= 1
        lost = safe_float(row["estimated_lost_sales"])
        if stockout:
            bg = "#fecaca" if 6 <= hour <= 21 else "#fee2e2"
        elif lost > 0:
            bg = "#fde68a"
        else:
            bg = "#dbeafe"
        label = f"{hour:02d}"
        cells.append(f'<div class="hour-cell" style="--cell-bg: {bg};" title="hour {hour}, lost sales {lost:.3f}">{label}</div>')
    render_html(
        f"""
        <div class="evidence-hero">
            <div class="evidence-card" style="border-top: 4px solid {accent};">
                <div class="evidence-kicker">Why this action</div>
                <div class="evidence-title">{escape(action_label)} for store {int(selected_row['store_id'])} / product {int(selected_row['product_id'])}</div>
                <div class="evidence-copy">{escape(readout)} {escape(explanation)}</div>
                <div class="evidence-pill-row">
                    <span class="evidence-pill">Date {escape(str(selected_row['full_date']))}</span>
                    <span class="evidence-pill">Stockout {stockout_hours}/24h</span>
                    <span class="evidence-pill">Business stockout {business_stockout_hours}/16h</span>
                    <span class="evidence-pill">Peak {peak_hour:02d}:00</span>
                </div>
            </div>
            <div class="evidence-card">
                <div class="evidence-kicker">Daily totals for selected row</div>
                <div class="card-stat-row">
                    <div class="mini-stat"><div class="mini-label">Observed</div><div class="mini-value">{metric_value(observed_sales)}</div></div>
                    <div class="mini-stat"><div class="mini-label">Estimated</div><div class="mini-value">{metric_value(estimated_demand)}</div></div>
                    <div class="mini-stat"><div class="mini-label">Recovered gap</div><div class="mini-value">{metric_value(lost_sales)}</div></div>
                </div>
                <div class="evidence-copy">{escape(str(selected_row['decision_reason']))}</div>
            </div>
        </div>
        """
    )
    render_html(
        f"""
        <div class="section-card">
            <div class="section-title">24-Hour Source Evidence</div>
            <div class="section-subtitle">Each cell is one source hour. Red means the dataset says stockout; amber means recovered lost-sales signal; blue means normal observed demand evidence.</div>
            <div class="hour-strip">{"".join(cells)}</div>
            <div class="legend-row">
                <span class="legend-item"><span class="legend-swatch" style="--swatch:#fecaca;"></span>Source stockout flag</span>
                <span class="legend-item"><span class="legend-swatch" style="--swatch:#fde68a;"></span>Recovered lost-sales signal</span>
                <span class="legend-item"><span class="legend-swatch" style="--swatch:#dbeafe;"></span>Observed demand carries estimate</span>
            </div>
        </div>
        """
    )


def render_hero(status: pd.Series, model_status: pd.DataFrame, failed_check_count: int) -> None:
    if model_status.empty:
        model_label = "Heuristic fallback active"
        prediction_label = "No model predictions"
    else:
        model = model_status.iloc[0]
        model_label = f"{model['model_name']} / {model['model_version']}"
        prediction_label = f"{int(model['demand_estimate_rows']):,} hourly estimates"
    quality_label = "Quality checks passed" if failed_check_count == 0 else f"{failed_check_count} quality checks failed"
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-kicker">Fresh Retail Decision Intelligence</div>
            <div class="hero-title">Inventory actions under stockout-censored demand</div>
            <div class="hero-copy">
                A warehouse-backed DSS for prioritizing restock, order increase, markdown, and censored-demand review decisions across stores, products, categories, and hourly stockout signals.
            </div>
            <div class="status-row">
                <span class="status-pill">Model: {model_label}</span>
                <span class="status-pill">Predictions: {prediction_label}</span>
                <span class="status-pill">Daily facts: {int(status['daily_fact_rows']):,}</span>
                <span class="status-pill">Hourly facts: {int(status['hourly_fact_rows']):,}</span>
                <span class="status-pill">Data: {quality_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Fresh Retail DSS", layout="wide")
    inject_css()

    status = load_data_status().iloc[0]
    if int(status["daily_fact_rows"] or 0) == 0:
        st.error("No warehouse facts are loaded yet. Run `python3 etl/load_fresh_retail_dw.py --reset --load-hourly` first.")
        return

    model_status = load_model_status()
    model_quality = load_model_quality()
    quality_checks = load_quality_checks()
    failed_checks = quality_checks.loc[pd.to_numeric(quality_checks["failed_rows"], errors="coerce") > 0]

    render_hero(status, model_status, len(failed_checks))

    categories, stores, products = load_filter_options()

    st.sidebar.title("Decision Controls")
    st.sidebar.caption("Filters apply across all views and SQL-backed metrics.")
    min_date = status["min_date"]
    max_date = status["max_date"]
    default_start_date = max(min_date, max_date - timedelta(days=6))
    start_date = st.sidebar.date_input("Start date", value=default_start_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)
    first_category_id = optional_int_select("First category", categories["first_category_id"])
    store_id = optional_int_select("Store", stores["store_id"])
    product_id = optional_int_select("Product", products["product_id"])
    action = st.sidebar.selectbox(
        "Decision action",
        [
            "All actions",
            "Restock immediately",
            "Increase next order",
            "Reduce order or markdown",
            "Review censored demand",
            "Maintain plan",
        ],
        format_func=display_action_label,
    )
    queue_view = st.sidebar.selectbox(
        "Action queue view",
        ["Highest priority", "Diverse action sample", "Exclude full-stockout rows"],
        help="Highest priority can be dominated by full-stockout days. Use the other views to inspect non-saturated examples.",
    )
    st.sidebar.divider()
    st.sidebar.caption("Recommended operating mode: use model-backed decisions for ranking, then inspect hourly drill-down before acting.")

    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
        return

    where_sql, params = build_filters(start_date, end_date, first_category_id, store_id, product_id, action)
    summary = load_summary(where_sql, params).iloc[0]
    trend = load_trend(where_sql, params)
    category_summary = load_category_summary(where_sql, params)
    recommendations = load_recommendations(where_sql, params, queue_view)
    score_columns = [
        "stockout_rate_6_22",
        "demand_bias_rate",
        "waste_risk_score",
        "restock_urgency_score",
        "stockout_hours_6_22",
        "observed_daily_sales_amount",
    ]
    recommendations = coerce_numeric_columns(recommendations, score_columns)

    command_tab, actions_tab, trends_tab, drill_tab, quality_tab = st.tabs(
        ["Command Center", "Action Queue", "Trends & Categories", "Hourly Drill-down", "Model & Data"]
    )

    with command_tab:
        render_section_header("Executive KPIs", "Fast readout of stockout pressure, demand bias, lost sales, and operational queue size.")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            render_metric_card("Stockout rate", metric_value(safe_float(summary.get("avg_stockout_rate_6_22")) * 100, "%"), "Average business-hour stockout pressure", "#dc2626")
        with kpi2:
            render_metric_card("Estimated lost sales", metric_value(summary["estimated_lost_sales"]), "Recovered demand not captured by observed sales", "#f59e0b")
        with kpi3:
            render_metric_card("Immediate restocks", metric_value(summary["immediate_restock_count"]), "Rows crossing urgent action threshold", "#2563eb")
        with kpi4:
            render_metric_card("Avg urgency", metric_value(safe_float(summary.get("avg_restock_urgency_score")) * 100, "%"), "Composite DSS priority score", "#7c3aed")

        kpi5, kpi6, kpi7, kpi8 = st.columns(4)
        with kpi5:
            render_metric_card("Observed sales", metric_value(summary["observed_sales_amount"]), "Recorded demand signal", "#0f766e")
        with kpi6:
            render_metric_card("Estimated demand", metric_value(summary["estimated_true_demand"]), "Observed plus recovered demand", "#1e40af")
        with kpi7:
            render_metric_card("Lost demand share", metric_value(safe_float(summary.get("avg_demand_bias_rate")) * 100, "%"), "Estimated demand hidden by stockouts", "#ea580c")
        with kpi8:
            render_metric_card("Waste risk", metric_value(safe_float(summary.get("avg_waste_risk_score")) * 100, "%"), "Slow-moving proxy risk", "#4f46e5")

        render_action_grid(summary)

        render_section_header("What-if Policy Simulator", "Tune urgency and service-level assumptions without writing anything back to the warehouse.")
        what_if_col1, what_if_col2 = st.columns(2)
        urgency_threshold = what_if_col1.slider("Immediate restock urgency threshold", min_value=0.30, max_value=0.90, value=0.65, step=0.05)
        service_level_target = what_if_col2.slider("Service level target for order proxy", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
        what_if = load_what_if(where_sql, params, urgency_threshold, service_level_target).iloc[0]
        what1, what2, what3, what4 = st.columns(4)
        what1.metric("Restock Items", metric_value(what_if["restock_count"]))
        what2.metric("Increase Order Items", metric_value(what_if["increase_order_count"]))
        what3.metric("Markdown Items", metric_value(what_if["markdown_count"]))
        what4.metric("Restock Qty Proxy", metric_value(what_if["restock_order_qty_proxy"]))

    with actions_tab:
        render_section_header("Action Intelligence", "Operational lanes and priority cards for deciding where to intervene first.")
        if recommendations.empty:
            st.info("No recommendations match the current filters.")
        else:
            render_queue_insight(summary, recommendations, queue_view)
            render_action_lanes(recommendations)
            render_priority_cards(recommendations)
            with st.expander("Raw queue data and export"):
                st.download_button(
                    "Download filtered action queue",
                    recommendations.to_csv(index=False).encode("utf-8"),
                    file_name="fresh_retail_dss_actions.csv",
                    mime="text/csv",
                )
                st.dataframe(
                    recommendations,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "stockout_rate_6_22": st.column_config.ProgressColumn("Business-hour Stockout", min_value=0, max_value=1),
                        "demand_bias_rate": st.column_config.ProgressColumn("Lost Demand Share", min_value=0, max_value=1),
                        "waste_risk_score": st.column_config.ProgressColumn("Waste Risk", min_value=0, max_value=1),
                        "restock_urgency_score": st.column_config.ProgressColumn("Priority Score", min_value=0, max_value=1),
                        "observed_daily_sales_amount": st.column_config.NumberColumn("Observed Sales", format="%.3f"),
                        "estimated_true_demand": st.column_config.NumberColumn("Estimated Demand", format="%.3f"),
                        "estimated_lost_sales": st.column_config.NumberColumn("Lost Sales", format="%.3f"),
                        "recommended_order_qty": st.column_config.NumberColumn("Order Proxy", format="%.3f"),
                    },
                )

    with trends_tab:
        trend_left, trend_right = st.columns([1.35, 1])
        with trend_left:
            render_section_header("Decision Trend", "Daily movement in stockout, demand-bias, waste, and urgency signals.")
            if not trend.empty:
                trend_columns = ["stockout_rate_6_22", "waste_risk_score", "demand_bias_rate", "restock_urgency_score"]
                trend_chart = coerce_numeric_columns(trend, trend_columns)
                trend_chart["full_date"] = pd.to_datetime(trend_chart["full_date"])
                st.line_chart(trend_chart, x="full_date", y=trend_columns, height=360)
            else:
                st.info("No trend data for the selected filters.")
        with trend_right:
            render_section_header("Category Pressure", "Top categories by urgency and lost-sales risk.")
            if not category_summary.empty:
                category_columns = ["stockout_rate_6_22", "waste_risk_score", "restock_urgency_score"]
                chart_df = coerce_numeric_columns(category_summary, category_columns)
                chart_df["first_category_id"] = chart_df["first_category_id"].astype(str)
                st.bar_chart(chart_df, x="first_category_id", y=category_columns, height=360)
                render_category_cards(chart_df)
                with st.expander("Category table"):
                    st.dataframe(chart_df, width="stretch", hide_index=True)
            else:
                st.info("No category data for the selected filters.")

    with drill_tab:
        render_section_header("Hourly Evidence", "Inspect one recommendation hour-by-hour to see observed sales, recovered demand, lost sales, and stockout flags.")
        if recommendations.empty:
            st.info("No recommendations available for hourly drill-down.")
        else:
            recommendation_options = recommendations.reset_index(drop=True)
            selected_index = st.selectbox(
                "Select a recommendation",
                recommendation_options.index.tolist(),
                format_func=lambda idx: (
                    f"{recommendation_options.loc[idx, 'full_date']} | "
                    f"store {int(recommendation_options.loc[idx, 'store_id'])} | "
                    f"product {int(recommendation_options.loc[idx, 'product_id'])} | "
                    f"{display_action_label(recommendation_options.loc[idx, 'decision_action'])}"
                ),
            )
            selected_row = recommendation_options.loc[selected_index]
            render_selection_overview(selected_row)
            hourly = load_hourly_drilldown(selected_row["full_date"], int(selected_row["store_id"]), int(selected_row["product_id"]))
            if hourly.empty:
                st.info("Hourly fact rows are not loaded for this recommendation. Re-run ETL with `--load-hourly` to enable drill-down.")
            else:
                hourly_numeric = coerce_numeric_columns(hourly, ["observed_sales_amount", "estimated_true_demand", "estimated_lost_sales", "stockout_flag"])
                hourly_numeric["hour_of_day"] = hourly_numeric["hour_of_day"].astype(int)
                render_hourly_story(selected_row, hourly_numeric)
                business_hours = hourly_numeric[(hourly_numeric["hour_of_day"] >= 6) & (hourly_numeric["hour_of_day"] <= 21)]
                if not business_hours.empty and business_hours["stockout_flag"].eq(1).all():
                    st.info(
                        "The selected recommendation is a full business-hour stockout in the source data. "
                        "The stockout bars are therefore all 1 for hours 6-21; this is an observed stockout flag, not a model prediction."
                    )
                render_section_header("Demand Recovery Chart", "Orange bars show recovered lost sales. Blue dashed line is estimated demand. Gray line is observed sales.")
                render_hourly_evidence_chart(hourly_numeric)
                evidence_table = prepare_hourly_evidence_table(hourly_numeric)
                with st.expander("Hour-by-hour evidence table", expanded=True):
                    st.dataframe(
                        evidence_table,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "observed_sales_amount": st.column_config.NumberColumn("Observed sales", format="%.4f"),
                            "estimated_true_demand": st.column_config.NumberColumn("Estimated demand", format="%.4f"),
                            "estimated_lost_sales": st.column_config.NumberColumn("Recovered lost sales", format="%.4f"),
                            "stockout_flag": st.column_config.CheckboxColumn("Stockout flag"),
                            "estimate_source": st.column_config.TextColumn("Estimate source"),
                        },
                    )
                    st.markdown('<div class="evidence-table-note">Use this table to verify whether the recommendation comes from observed sales, source stockout flags, or recovered demand estimates.</div>', unsafe_allow_html=True)
                with st.expander("Raw hourly rows"):
                    st.dataframe(hourly, width="stretch", hide_index=True)

    with quality_tab:
        model_col, data_col = st.columns([1.15, 1])
        with model_col:
            render_section_header("Model Quality Guardrail", "Evaluation metrics imported from cloud training and used to constrain how decisions are interpreted.")
            if model_quality.empty:
                st.info("No model-quality metrics are loaded yet. Import cloud metrics JSON with `ml/import_cloud_predictions.py`.")
            else:
                quality = model_quality.iloc[0]
                mq1, mq2, mq3 = st.columns(3)
                mq1.metric("Eval Rows", metric_value(quality["eval_rows"]))
                mq2.metric("WMAPE", metric_value_precise(quality["wmape"], 4))
                mq3.metric("Bias", metric_percent(quality["bias"], 2))
                mq4, mq5, mq6 = st.columns(3)
                mq4.metric("Calibration", metric_value_precise(quality["calibration_factor"], 4))
                mq5.metric("MAE", metric_value_precise(quality["mae"], 4))
                mq6.metric("RMSE", metric_value_precise(quality["rmse"], 4))
                raw_calibration = quality.get("raw_calibration_factor")
                if pd.notna(raw_calibration) and pd.notna(quality["calibration_factor"]):
                    raw_factor = float(raw_calibration)
                    applied_factor = float(quality["calibration_factor"])
                    if abs(raw_factor - applied_factor) > 0.0001:
                        st.warning(
                            f"Calibration was capped: raw factor {raw_factor:.4f}, applied factor {applied_factor:.4f}. "
                            "The negative bias means the capped DSS estimate is underpredicting aggregate eval demand, but it avoids extreme order inflation."
                        )
                if pd.notna(quality["wmape"]) and float(quality["wmape"]) >= 0.8:
                    st.warning("Model error is high. Use recommendations as risk ranking and decision support, not exact order optimization.")
                if pd.notna(quality["calibration_factor"]) and float(quality["calibration_factor"]) >= 5:
                    st.warning("Calibration factor is very high. Lost-sales quantities may be inflated; prefer capped calibration or a larger/more representative training sample.")
                if pd.notna(quality["bias"]) and abs(float(quality["bias"])) <= 0.05:
                    st.success("Aggregate demand bias is within +/-5%, suitable for DSS-level lost-sales and action-priority reporting.")
                elif pd.notna(quality["bias"]):
                    st.error("Aggregate demand bias is outside +/-5%. Treat quantities as conservative risk proxies, not calibrated demand forecasts.")
                with st.expander("Model quality table"):
                    st.dataframe(model_quality, width="stretch", hide_index=True)
        with data_col:
            render_section_header("Warehouse Quality", "Data contract checks for arrays, grain, and stockout-count consistency.")
            if failed_checks.empty:
                st.success("All loaded warehouse quality checks passed.")
            else:
                st.error(f"{len(failed_checks)} quality check(s) failed.")
            with st.expander("Warehouse quality table"):
                st.dataframe(quality_checks, width="stretch", hide_index=True)

        with st.expander("How DSS scores are calculated"):
            st.markdown(
                """
                - Stockout reduction: business-hour stockout rate from `stockout_hours_6_22 / 16`.
                - Waste minimization: proxy score for low observed sales when no stockout occurred.
                - Faster restocking: urgency score combines stockout rate, estimated lost-sales bias, and promotion/activity signal.
                - Censored-demand bias: estimated lost sales during stockout hours using model output or fallback hourly baselines.
                - What-if: threshold changes recompute decision counts without modifying warehouse data.
                - Order and waste quantities are proxies because the source data has no stock-on-hand, expiry, lead time, or supplier constraints.
                """
            )


if __name__ == "__main__":
    main()
