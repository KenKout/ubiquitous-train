#!/usr/bin/env python3
from __future__ import annotations

import os
from datetime import date
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
            created_at
        FROM dw.v_latest_model
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
            observed_daily_sales_amount,
            estimated_true_demand,
            estimated_lost_sales,
            stockout_hours_6_22,
            stockout_rate_6_22,
            demand_bias_rate,
            waste_risk_score,
            restock_urgency_score,
            decision_action,
            decision_reason
        FROM dw.v_dss_daily_decision_score
        WHERE {where_sql}
        ORDER BY restock_urgency_score DESC, estimated_lost_sales DESC, stockout_hours_6_22 DESC
        LIMIT 200
        """,
        params,
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


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    return result


def main() -> None:
    st.set_page_config(page_title="Fresh Retail DSS", layout="wide")
    st.title("Fresh Retail Decision Support System")
    st.caption("Stockout reduction, waste minimization, faster restocking, and censored-demand bias control.")

    status = load_data_status().iloc[0]
    if int(status["daily_fact_rows"] or 0) == 0:
        st.error("No warehouse facts are loaded yet. Run `python3 etl/load_fresh_retail_dw.py --reset --load-hourly` first.")
        return

    model_status = load_model_status()
    if model_status.empty or int(status["demand_estimate_rows"] or 0) == 0:
        st.warning("No trained model predictions are loaded yet. The DSS will use fallback heuristic demand recovery until `ml/train_xgboost_demand_model.py --load-predictions` is run.")
    else:
        model = model_status.iloc[0]
        st.success(
            f"Using trained model `{model['model_name']}` version `{model['model_version']}` "
            f"with {int(status['demand_estimate_rows']):,} hourly predictions and "
            f"{int(status['recommendation_rows']):,} daily recommendations."
        )

    categories, stores, products = load_filter_options()

    st.sidebar.header("Decision Filters")
    min_date = status["min_date"]
    max_date = status["max_date"]
    start_date = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
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

    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
        return

    where_sql, params = build_filters(start_date, end_date, first_category_id, store_id, product_id, action)
    summary = load_summary(where_sql, params).iloc[0]

    st.subheader("Four DSS Criteria")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stockout Rate", metric_value(summary["avg_stockout_rate_6_22"] * 100, "%"))
    col2.metric("Waste Risk", metric_value(summary["avg_waste_risk_score"] * 100, "%"))
    col3.metric("Immediate Restocks", metric_value(summary["immediate_restock_count"]))
    col4.metric("Demand Bias", metric_value(summary["avg_demand_bias_rate"] * 100, "%"))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Observed Sales", metric_value(summary["observed_sales_amount"]))
    col6.metric("Estimated Demand", metric_value(summary["estimated_true_demand"]))
    col7.metric("Estimated Lost Sales", metric_value(summary["estimated_lost_sales"]))
    col8.metric("Avg Urgency", metric_value(summary["avg_restock_urgency_score"] * 100, "%"))

    trend = load_trend(where_sql, params)
    category_summary = load_category_summary(where_sql, params)
    recommendations = load_recommendations(where_sql, params)

    st.subheader("Decision Trend")
    if not trend.empty:
        trend_columns = [
            "stockout_rate_6_22",
            "waste_risk_score",
            "demand_bias_rate",
            "restock_urgency_score",
        ]
        trend_chart = coerce_numeric_columns(trend, trend_columns)
        trend_chart["full_date"] = pd.to_datetime(trend_chart["full_date"])
        st.line_chart(trend_chart, x="full_date", y=trend_columns)
    else:
        st.info("No trend data for the selected filters.")

    st.subheader("Top Category Pressure")
    if not category_summary.empty:
        category_columns = [
            "stockout_rate_6_22",
            "waste_risk_score",
            "restock_urgency_score",
        ]
        chart_df = coerce_numeric_columns(category_summary, category_columns)
        chart_df["first_category_id"] = chart_df["first_category_id"].astype(str)
        st.bar_chart(chart_df, x="first_category_id", y=category_columns)
    else:
        st.info("No category data for the selected filters.")

    st.subheader("Recommended Actions")
    score_columns = ["stockout_rate_6_22", "demand_bias_rate", "waste_risk_score", "restock_urgency_score"]
    recommendations = coerce_numeric_columns(recommendations, score_columns)
    st.dataframe(
        recommendations,
        use_container_width=True,
        hide_index=True,
        column_config={
            "stockout_rate_6_22": st.column_config.ProgressColumn("Stockout 6-21", min_value=0, max_value=1),
            "demand_bias_rate": st.column_config.ProgressColumn("Demand Bias", min_value=0, max_value=1),
            "waste_risk_score": st.column_config.ProgressColumn("Waste Risk", min_value=0, max_value=1),
            "restock_urgency_score": st.column_config.ProgressColumn("Restock Urgency", min_value=0, max_value=1),
        },
    )

    with st.expander("How the DSS scores are calculated"):
        st.markdown(
            """
            - Stockout reduction: business-hour stockout rate from `stockout_hours_6_22 / 16`.
            - Waste minimization: proxy score for low observed sales when no stockout occurred.
            - Faster restocking: urgency score combining stockout rate, estimated lost sales bias, and promotion/activity signal.
            - Censored-demand bias: estimated lost sales during stockout hours using non-stockout hourly baselines.
            - A row can show 100% stockout, 100% demand bias, and 100% urgency when the product was stocked out for all 16 business hours, observed sales were zero, and the trained model still estimated positive demand.
            """
        )


if __name__ == "__main__":
    main()
