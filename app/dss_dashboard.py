#!/usr/bin/env python3
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

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
def load_recommendations(where_sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
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
            decision_action,
            decision_reason,
            inventory_proxy_note
        FROM dw.v_dss_daily_decision_score
        WHERE {where_sql}
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
        }

        @media (max-width: 520px) {
            .action-grid { grid-template-columns: 1fr; }
            .hero-panel { padding: 22px; }
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
    recommendations = load_recommendations(where_sql, params)
    score_columns = ["stockout_rate_6_22", "demand_bias_rate", "waste_risk_score", "restock_urgency_score"]
    recommendations = coerce_numeric_columns(recommendations, score_columns)

    command_tab, actions_tab, trends_tab, drill_tab, quality_tab = st.tabs(
        ["Command Center", "Action Queue", "Trends & Categories", "Hourly Drill-down", "Model & Data"]
    )

    with command_tab:
        render_section_header("Executive KPIs", "Fast readout of stockout pressure, demand bias, lost sales, and operational queue size.")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            render_metric_card("Stockout rate", metric_value(summary["avg_stockout_rate_6_22"] * 100, "%"), "Average business-hour stockout pressure", "#dc2626")
        with kpi2:
            render_metric_card("Estimated lost sales", metric_value(summary["estimated_lost_sales"]), "Recovered demand not captured by observed sales", "#f59e0b")
        with kpi3:
            render_metric_card("Immediate restocks", metric_value(summary["immediate_restock_count"]), "Rows crossing urgent action threshold", "#2563eb")
        with kpi4:
            render_metric_card("Avg urgency", metric_value(summary["avg_restock_urgency_score"] * 100, "%"), "Composite DSS priority score", "#7c3aed")

        kpi5, kpi6, kpi7, kpi8 = st.columns(4)
        with kpi5:
            render_metric_card("Observed sales", metric_value(summary["observed_sales_amount"]), "Recorded demand signal", "#0f766e")
        with kpi6:
            render_metric_card("Estimated demand", metric_value(summary["estimated_true_demand"]), "Observed plus recovered demand", "#1e40af")
        with kpi7:
            render_metric_card("Demand bias", metric_value(summary["avg_demand_bias_rate"] * 100, "%"), "Sales understatement from censoring", "#ea580c")
        with kpi8:
            render_metric_card("Waste risk", metric_value(summary["avg_waste_risk_score"] * 100, "%"), "Slow-moving proxy risk", "#4f46e5")

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
        render_section_header("Prioritized Action Queue", "Ranked store-product-date recommendations with explainable reasons and risk scores.")
        if recommendations.empty:
            st.info("No recommendations match the current filters.")
        else:
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
                    "stockout_rate_6_22": st.column_config.ProgressColumn("Stockout 6-21", min_value=0, max_value=1),
                    "demand_bias_rate": st.column_config.ProgressColumn("Demand Bias", min_value=0, max_value=1),
                    "waste_risk_score": st.column_config.ProgressColumn("Waste Risk", min_value=0, max_value=1),
                    "restock_urgency_score": st.column_config.ProgressColumn("Restock Urgency", min_value=0, max_value=1),
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
                    f"{recommendation_options.loc[idx, 'decision_action']}"
                ),
            )
            selected_row = recommendation_options.loc[selected_index]
            detail1, detail2, detail3, detail4 = st.columns(4)
            detail1.metric("Store", int(selected_row["store_id"]))
            detail2.metric("Product", int(selected_row["product_id"]))
            detail3.metric("Action", selected_row["decision_action"])
            detail4.metric("Urgency", metric_value(float(selected_row["restock_urgency_score"]) * 100, "%"))
            st.caption(str(selected_row["decision_reason"]))
            hourly = load_hourly_drilldown(selected_row["full_date"], int(selected_row["store_id"]), int(selected_row["product_id"]))
            if hourly.empty:
                st.info("Hourly fact rows are not loaded for this recommendation. Re-run ETL with `--load-hourly` to enable drill-down.")
            else:
                hourly_numeric = coerce_numeric_columns(hourly, ["observed_sales_amount", "estimated_true_demand", "estimated_lost_sales", "stockout_flag"])
                hourly_numeric["hour_of_day"] = hourly_numeric["hour_of_day"].astype(int)
                line_col, stock_col = st.columns([1.4, 1])
                with line_col:
                    st.line_chart(hourly_numeric, x="hour_of_day", y=["observed_sales_amount", "estimated_true_demand", "estimated_lost_sales"], height=360)
                with stock_col:
                    st.bar_chart(hourly_numeric, x="hour_of_day", y="stockout_flag", height=360)
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
                if pd.notna(quality["wmape"]) and float(quality["wmape"]) >= 0.8:
                    st.warning("Model error is high. Use recommendations as risk ranking and decision support, not exact order optimization.")
                if pd.notna(quality["bias"]) and abs(float(quality["bias"])) <= 0.05:
                    st.success("Aggregate demand bias is within +/-5%, suitable for DSS-level lost-sales and action-priority reporting.")
                st.dataframe(model_quality, width="stretch", hide_index=True)
        with data_col:
            render_section_header("Warehouse Quality", "Data contract checks for arrays, grain, and stockout-count consistency.")
            if failed_checks.empty:
                st.success("All loaded warehouse quality checks passed.")
            else:
                st.error(f"{len(failed_checks)} quality check(s) failed.")
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
