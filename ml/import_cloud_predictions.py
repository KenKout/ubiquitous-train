#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psycopg


DEMAND_COLUMNS = [
    "date_key",
    "time_key",
    "store_key",
    "product_key",
    "model_key",
    "observed_sales_amount",
    "estimated_true_demand",
    "estimated_lost_sales",
    "stockout_flag",
    "is_censored_observation",
    "prediction_lower_bound",
    "prediction_upper_bound",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Kaggle/Colab prediction parquet into the DSS warehouse model facts.")
    parser.add_argument("--predictions", required=True, help="Prediction parquet written by ml/cloud_gpu_train.py.")
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5433")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE", "fresh_retail_dw"))
    parser.add_argument("--user", default=os.getenv("PGUSER", "warehouse"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", "warehouse"))
    parser.add_argument("--model-name", default="cloud_xgboost_gpu")
    parser.add_argument("--model-version", default=None, help="Default: UTC timestamp.")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--service-level-target", type=float, default=0.95)
    parser.add_argument("--replace-model-output", action="store_true", help="Delete existing fact rows for the same model before importing.")
    parser.add_argument("--skip-recommendations", action="store_true", help="Only import hourly predictions.")
    return parser.parse_args()


def connect(args: argparse.Namespace) -> psycopg.Connection:
    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )


def upsert_model_metadata(conn: psycopg.Connection, model_name: str, model_version: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO dw.dim_model (model_name, model_version, created_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (model_name, model_version) DO UPDATE SET
                created_at = NOW()
            RETURNING model_key
            """,
            (model_name, model_version),
        )
        model_key = cur.fetchone()[0]
    conn.commit()
    return int(model_key)


def replace_existing_output(conn: psycopg.Connection, model_key: int) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM dw.fact_replenishment_recommendation_daily WHERE model_key = %s", (model_key,))
        cur.execute("DELETE FROM dw.fact_demand_estimate_hourly WHERE model_key = %s", (model_key,))
    conn.commit()


def create_temp_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE tmp_cloud_demand_estimate_hourly (
                date_key INTEGER,
                time_key SMALLINT,
                store_key INTEGER,
                product_key INTEGER,
                model_key INTEGER,
                observed_sales_amount DOUBLE PRECISION,
                estimated_true_demand DOUBLE PRECISION,
                estimated_lost_sales DOUBLE PRECISION,
                stockout_flag BOOLEAN,
                is_censored_observation BOOLEAN,
                prediction_lower_bound DOUBLE PRECISION,
                prediction_upper_bound DOUBLE PRECISION
            )
            """
        )
    conn.commit()


def copy_dataframe(conn: psycopg.Connection, table: str, columns: list[str], df: pd.DataFrame) -> None:
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    buffer.seek(0)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT CSV)") as copy:
            copy.write(buffer.getvalue())
    conn.commit()


def normalize_prediction_chunk(df: pd.DataFrame, model_key: int) -> pd.DataFrame:
    required = {"date_key", "time_key", "store_key", "product_key", "observed_sales_amount", "stockout_flag"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Prediction file is missing required columns: {sorted(missing)}")

    result = pd.DataFrame()
    result["date_key"] = pd.to_numeric(df["date_key"], errors="coerce").fillna(0).astype(int)
    result["time_key"] = pd.to_numeric(df["time_key"], errors="coerce").fillna(0).astype(int)
    result["store_key"] = pd.to_numeric(df["store_key"], errors="coerce").fillna(0).astype(int)
    result["product_key"] = pd.to_numeric(df["product_key"], errors="coerce").fillna(0).astype(int)
    result["model_key"] = model_key
    observed = pd.to_numeric(df["observed_sales_amount"], errors="coerce").fillna(0).astype(float).to_numpy()
    stockout = df["stockout_flag"].astype(bool).to_numpy()

    if "estimated_true_demand" in df.columns:
        estimated_true_demand = pd.to_numeric(df["estimated_true_demand"], errors="coerce").fillna(0).astype(float).to_numpy()
    elif "predicted_demand" in df.columns:
        predicted = pd.to_numeric(df["predicted_demand"], errors="coerce").fillna(0).astype(float).to_numpy()
        estimated_true_demand = np.where(stockout, np.maximum(predicted, observed), observed)
    else:
        raise RuntimeError("Prediction file must contain either estimated_true_demand or predicted_demand.")

    if "estimated_lost_sales" in df.columns:
        estimated_lost_sales = pd.to_numeric(df["estimated_lost_sales"], errors="coerce").fillna(0).astype(float).to_numpy()
    else:
        estimated_lost_sales = np.where(stockout, np.maximum(estimated_true_demand - observed, 0), 0)

    if "prediction_lower_bound" in df.columns:
        prediction_lower_bound = pd.to_numeric(df["prediction_lower_bound"], errors="coerce").fillna(0).astype(float).to_numpy()
    else:
        prediction_lower_bound = np.where(stockout, np.maximum(estimated_true_demand * 0.85, observed), observed)

    if "prediction_upper_bound" in df.columns:
        prediction_upper_bound = pd.to_numeric(df["prediction_upper_bound"], errors="coerce").fillna(0).astype(float).to_numpy()
    else:
        prediction_upper_bound = np.where(stockout, np.maximum(estimated_true_demand * 1.15, observed), observed)

    final_estimated_true_demand = np.where(stockout, np.maximum(estimated_true_demand, observed), observed)
    final_estimated_lost_sales = np.where(stockout, np.maximum(final_estimated_true_demand - observed, 0), 0)

    result["observed_sales_amount"] = observed
    result["estimated_true_demand"] = final_estimated_true_demand
    result["estimated_lost_sales"] = np.maximum(final_estimated_lost_sales, np.where(stockout, estimated_lost_sales, 0))
    result["stockout_flag"] = stockout
    result["is_censored_observation"] = stockout
    result["prediction_lower_bound"] = prediction_lower_bound
    result["prediction_upper_bound"] = prediction_upper_bound
    return result[DEMAND_COLUMNS]


def insert_prediction_chunk(conn: psycopg.Connection, prediction_df: pd.DataFrame) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE tmp_cloud_demand_estimate_hourly")
    conn.commit()

    copy_dataframe(conn, "pg_temp.tmp_cloud_demand_estimate_hourly", DEMAND_COLUMNS, prediction_df)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO dw.fact_demand_estimate_hourly (
                date_key,
                time_key,
                store_key,
                product_key,
                model_key,
                observed_sales_amount,
                estimated_true_demand,
                estimated_lost_sales,
                stockout_flag,
                is_censored_observation,
                prediction_lower_bound,
                prediction_upper_bound
            )
            SELECT
                date_key,
                time_key,
                store_key,
                product_key,
                model_key,
                observed_sales_amount,
                estimated_true_demand,
                estimated_lost_sales,
                stockout_flag,
                is_censored_observation,
                prediction_lower_bound,
                prediction_upper_bound
            FROM tmp_cloud_demand_estimate_hourly
            ON CONFLICT (date_key, time_key, store_key, product_key, model_key) DO UPDATE SET
                observed_sales_amount = EXCLUDED.observed_sales_amount,
                estimated_true_demand = EXCLUDED.estimated_true_demand,
                estimated_lost_sales = EXCLUDED.estimated_lost_sales,
                stockout_flag = EXCLUDED.stockout_flag,
                is_censored_observation = EXCLUDED.is_censored_observation,
                prediction_lower_bound = EXCLUDED.prediction_lower_bound,
                prediction_upper_bound = EXCLUDED.prediction_upper_bound,
                created_at = NOW()
            """
        )
    conn.commit()


def import_predictions(conn: psycopg.Connection, path: Path, model_key: int, chunk_size: int) -> int:
    parquet_file = pq.ParquetFile(path)
    total_rows = 0
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        prediction_df = normalize_prediction_chunk(df, model_key)
        insert_prediction_chunk(conn, prediction_df)
        total_rows += len(prediction_df)
        print(f"imported {total_rows:,} hourly predictions")
    return total_rows


def refresh_model_training_dates(conn: psycopg.Connection, model_key: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE dw.dim_model m
            SET
                training_start_date = dates.training_start_date,
                training_end_date = dates.training_end_date
            FROM (
                SELECT MIN(d.full_date) AS training_start_date, MAX(d.full_date) AS training_end_date
                FROM dw.fact_demand_estimate_hourly e
                JOIN dw.dim_date d ON d.date_key = e.date_key
                WHERE e.model_key = %s
            ) dates
            WHERE m.model_key = %s
            """,
            (model_key, model_key),
        )
    conn.commit()


def load_recommendations(conn: psycopg.Connection, model_key: int, service_level_target: float) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT date_key FROM dw.fact_demand_estimate_hourly WHERE model_key = %s ORDER BY date_key", (model_key,))
        date_keys = [row[0] for row in cur.fetchall()]

    total_rows = 0
    for index, date_key in enumerate(date_keys, start=1):
        rows = load_recommendations_for_date(conn, model_key, date_key, service_level_target)
        total_rows += rows
        print(f"loaded recommendations for date_key {date_key} ({index}/{len(date_keys)}): {rows:,}; total {total_rows:,}")
    return total_rows


def load_recommendations_for_date(conn: psycopg.Connection, model_key: int, date_key: int, service_level_target: float) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO dw.fact_replenishment_recommendation_daily (
                date_key,
                store_key,
                product_key,
                model_key,
                recommended_order_qty,
                expected_demand,
                expected_lost_sales,
                stockout_risk_score,
                expected_waste_qty,
                service_level_target
            )
            WITH daily_model AS (
                SELECT
                    date_key,
                    store_key,
                    product_key,
                    model_key,
                    SUM(estimated_true_demand) AS expected_demand,
                    SUM(estimated_lost_sales) AS expected_lost_sales
                FROM dw.fact_demand_estimate_hourly
                WHERE model_key = %s
                  AND date_key = %s
                GROUP BY date_key, store_key, product_key, model_key
            )
            SELECT
                m.date_key,
                m.store_key,
                m.product_key,
                m.model_key,
                GREATEST(m.expected_demand * %s / 0.95, 0) AS recommended_order_qty,
                m.expected_demand,
                m.expected_lost_sales,
                LEAST(
                    1,
                    (f.stockout_hours_6_22 / 16.0) * 0.55
                    + CASE WHEN m.expected_demand > 0 THEN (m.expected_lost_sales / m.expected_demand) * 0.35 ELSE 0 END
                    + CASE WHEN f.activity_flag <> 0 THEN 0.10 ELSE 0 END
                ) AS stockout_risk_score,
                GREATEST(f.observed_daily_sales_amount - m.expected_demand, 0) AS expected_waste_qty,
                %s AS service_level_target
            FROM daily_model m
            JOIN dw.fact_sales_inventory_daily f
                ON f.date_key = m.date_key
                AND f.store_key = m.store_key
                AND f.product_key = m.product_key
            ON CONFLICT (date_key, store_key, product_key, model_key) DO UPDATE SET
                recommended_order_qty = EXCLUDED.recommended_order_qty,
                expected_demand = EXCLUDED.expected_demand,
                expected_lost_sales = EXCLUDED.expected_lost_sales,
                stockout_risk_score = EXCLUDED.stockout_risk_score,
                expected_waste_qty = EXCLUDED.expected_waste_qty,
                service_level_target = EXCLUDED.service_level_target,
                created_at = NOW()
            """,
            (model_key, date_key, service_level_target, service_level_target),
        )
        rows = cur.rowcount
    conn.commit()
    return rows


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.predictions)
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)

    model_version = args.model_version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    with connect(args) as conn:
        model_key = upsert_model_metadata(conn, args.model_name, model_version)
        print(f"model_key: {model_key}")
        if args.replace_model_output:
            print("replacing existing output for this model")
            replace_existing_output(conn, model_key)

        create_temp_table(conn)
        prediction_rows = import_predictions(conn, prediction_path, model_key, args.chunk_size)
        refresh_model_training_dates(conn, model_key)

        recommendation_rows = 0
        if not args.skip_recommendations:
            recommendation_rows = load_recommendations(conn, model_key, args.service_level_target)

        print(f"dw.fact_demand_estimate_hourly rows imported: {prediction_rows:,}")
        print(f"dw.fact_replenishment_recommendation_daily rows loaded: {recommendation_rows:,}")


if __name__ == "__main__":
    main()
