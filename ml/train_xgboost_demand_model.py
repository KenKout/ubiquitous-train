#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


FEATURE_COLUMNS = [
    "time_key",
    "city_id",
    "store_id",
    "product_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "discount_rate",
    "activity_flag",
    "holiday_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "day_of_week",
    "is_weekend",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost latent-demand model from warehouse hourly facts.")
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5433")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE", "fresh_retail_dw"))
    parser.add_argument("--user", default=os.getenv("PGUSER", "warehouse"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", "warehouse"))
    parser.add_argument("--model-name", default="xgboost_latent_demand")
    parser.add_argument("--model-type", choices=["xgboost", "catboost"], default="xgboost")
    parser.add_argument("--model-version", default=None, help="Default: UTC timestamp.")
    parser.add_argument("--artifact-dir", default="models")
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument("--train-sample-rate", type=float, default=None, help="Randomly sample train rows while scanning the full hourly fact, e.g. 0.02.")
    parser.add_argument("--max-eval-rows", type=int, default=300_000)
    parser.add_argument("--eval-sample-rate", type=float, default=None, help="Randomly sample eval rows while scanning the hourly fact.")
    parser.add_argument("--max-predict-rows", type=int, default=None, help="Limit prediction loading for faster tests.")
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--load-predictions", action="store_true", help="Write predictions to warehouse fact tables.")
    return parser.parse_args()


def connect(args: argparse.Namespace) -> psycopg.Connection:
    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )


def read_sql(conn: psycopg.Connection, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def hourly_feature_sql(limit_clause: str = "") -> str:
    return f"""
        SELECT
            h.date_key,
            d.full_date,
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
            d.day_of_week,
            d.is_weekend,
            d.holiday_flag,
            s.store_id,
            c.city_id,
            p.product_id,
            p.management_group_id,
            p.first_category_id,
            p.second_category_id,
            p.third_category_id
        FROM dw.fact_sales_inventory_hourly h
        JOIN dw.dim_date d ON d.date_key = h.date_key
        JOIN dw.dim_store s ON s.store_key = h.store_key
        JOIN dw.dim_city c ON c.city_key = s.city_key
        JOIN dw.dim_product p ON p.product_key = h.product_key
        {limit_clause}
    """


def load_training_data(
    conn: psycopg.Connection,
    max_train_rows: int | None,
    train_sample_rate: float | None,
) -> pd.DataFrame:
    limit_clause = ""
    params: tuple[Any, ...] = ()
    if train_sample_rate is not None and max_train_rows:
        limit_clause = "WHERE h.source_split = 'train' AND random() < %s LIMIT %s"
        params = (train_sample_rate, max_train_rows)
    elif train_sample_rate is not None:
        limit_clause = "WHERE h.source_split = 'train' AND random() < %s"
        params = (train_sample_rate,)
    elif max_train_rows:
        limit_clause = "WHERE h.source_split = 'train' ORDER BY h.date_key, h.store_key, h.product_key, h.time_key LIMIT %s"
        params = (max_train_rows,)
    else:
        limit_clause = "WHERE h.source_split = 'train'"

    df = read_sql(conn, hourly_feature_sql(limit_clause), params)
    if df.empty:
        raise RuntimeError("No hourly training rows found. Run `python3 etl/load_fresh_retail_dw.py --reset --load-hourly` first.")
    return df


def add_latent_target(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    non_stockout = data.loc[~data["stockout_flag"].astype(bool)]

    store_product_hour = (
        non_stockout.groupby(["store_key", "product_key", "time_key"])["observed_sales_amount"]
        .mean()
        .rename("store_product_hour_baseline")
    )
    product_hour = (
        non_stockout.groupby(["product_key", "time_key"])["observed_sales_amount"]
        .mean()
        .rename("product_hour_baseline")
    )
    product_avg = non_stockout.groupby("product_key")["observed_sales_amount"].mean().rename("product_baseline")
    global_avg = float(non_stockout["observed_sales_amount"].mean()) if not non_stockout.empty else 0.0

    data = data.join(store_product_hour, on=["store_key", "product_key", "time_key"])
    data = data.join(product_hour, on=["product_key", "time_key"])
    data = data.join(product_avg, on="product_key")
    baseline = (
        data["store_product_hour_baseline"]
        .fillna(data["product_hour_baseline"])
        .fillna(data["product_baseline"])
        .fillna(global_avg)
    )

    data["latent_demand_target"] = data["observed_sales_amount"].astype(float)
    stockout_mask = data["stockout_flag"].astype(bool)
    data.loc[stockout_mask, "latent_demand_target"] = np.maximum(
        data.loc[stockout_mask, "observed_sales_amount"].astype(float),
        baseline.loc[stockout_mask].astype(float),
    )
    data["latent_demand_target"] = data["latent_demand_target"].clip(lower=0)
    return data


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURE_COLUMNS].copy()
    x["is_weekend"] = x["is_weekend"].astype(int)
    for column in FEATURE_COLUMNS:
        x[column] = pd.to_numeric(x[column], errors="coerce").fillna(0)
    return x


def train_model(args: argparse.Namespace, train_df: pd.DataFrame):
    x_train = feature_matrix(train_df)
    y_train = train_df["latent_demand_target"].astype(float)

    if args.model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            objective="reg:squarederror",
            tree_method="hist",
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=4,
        )
    elif args.model_type == "catboost":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed")
        model = CatBoostRegressor(
            iterations=args.n_estimators,
            depth=args.max_depth,
            learning_rate=args.learning_rate,
            loss_function="Tweedie:variance_power=1.5",
            random_seed=42,
            verbose=False,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model.fit(x_train, y_train)
    return model


def evaluate_model(conn: psycopg.Connection, model: XGBRegressor, train_df: pd.DataFrame, args: argparse.Namespace) -> dict[str, float]:
    if args.eval_sample_rate is not None and args.max_eval_rows:
        eval_df = read_sql(conn, hourly_feature_sql("WHERE h.source_split = 'eval' AND random() < %s LIMIT %s"), (args.eval_sample_rate, args.max_eval_rows))
    elif args.eval_sample_rate is not None:
        eval_df = read_sql(conn, hourly_feature_sql("WHERE h.source_split = 'eval' AND random() < %s"), (args.eval_sample_rate,))
    elif args.max_eval_rows:
        eval_df = read_sql(conn, hourly_feature_sql("WHERE h.source_split = 'eval' ORDER BY h.date_key, h.store_key, h.product_key, h.time_key LIMIT %s"), (args.max_eval_rows,))
    else:
        eval_df = read_sql(conn, hourly_feature_sql("WHERE h.source_split = 'eval'"))
    if eval_df.empty:
        eval_df = train_df.sample(min(len(train_df), 50_000), random_state=42).copy()

    eval_df = add_latent_target(eval_df)
    x_eval = feature_matrix(eval_df)
    y_eval = eval_df["latent_demand_target"].astype(float).to_numpy()
    y_pred = np.clip(model.predict(x_eval), 0, None)
    denominator = np.sum(np.abs(y_eval))

    return {
        "rows": float(len(eval_df)),
        "mae": float(mean_absolute_error(y_eval, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_eval, y_pred))),
        "wape": float(np.sum(np.abs(y_eval - y_pred)) / denominator) if denominator > 0 else 0.0,
        "bias": float(np.sum(y_pred - y_eval) / denominator) if denominator > 0 else 0.0,
    }


def save_artifacts(
    args: argparse.Namespace,
    model: XGBRegressor,
    metrics: dict[str, float],
    model_version: str,
) -> tuple[Path, Path]:
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{args.model_name}_{model_version}.pkl"
    metrics_path = artifact_dir / f"{args.model_name}_{model_version}_metrics.json"

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "model_name": args.model_name,
        "model_type": args.model_type,
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return model_path, metrics_path


def upsert_model_metadata(
    conn: psycopg.Connection,
    model_name: str,
    model_version: str,
    train_df: pd.DataFrame,
) -> int:
    training_start = train_df["full_date"].min()
    training_end = train_df["full_date"].max()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO dw.dim_model (model_name, model_version, training_start_date, training_end_date)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (model_name, model_version) DO UPDATE SET
                training_start_date = EXCLUDED.training_start_date,
                training_end_date = EXCLUDED.training_end_date,
                created_at = NOW()
            RETURNING model_key
            """,
            (model_name, model_version, training_start, training_end),
        )
        model_key = cur.fetchone()[0]
    conn.commit()
    return int(model_key)


def copy_dataframe(conn: psycopg.Connection, table: str, columns: list[str], df: pd.DataFrame) -> None:
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    buffer.seek(0)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT CSV)") as copy:
            copy.write(buffer.getvalue())
    conn.commit()


def create_prediction_temp_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TEMP TABLE tmp_demand_estimate_hourly (
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


def insert_prediction_chunk(conn: psycopg.Connection, prediction_df: pd.DataFrame) -> None:
    temp_table = "pg_temp.tmp_demand_estimate_hourly"
    with conn.cursor() as cur:
        cur.execute("TRUNCATE tmp_demand_estimate_hourly")
    conn.commit()

    columns = list(prediction_df.columns)
    copy_dataframe(conn, temp_table, columns, prediction_df)
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
            FROM tmp_demand_estimate_hourly
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


def load_predictions(conn: psycopg.Connection, model: XGBRegressor, model_key: int, max_predict_rows: int | None) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT date_key FROM dw.fact_sales_inventory_hourly ORDER BY date_key")
        date_keys = [row[0] for row in cur.fetchall()]

    if not date_keys:
        raise RuntimeError("No hourly rows found for prediction loading.")

    create_prediction_temp_table(conn)
    loaded_rows = 0

    for index, date_key in enumerate(date_keys, start=1):
        remaining = None if max_predict_rows is None else max_predict_rows - loaded_rows
        if remaining is not None and remaining <= 0:
            break

        limit_clause = "WHERE h.date_key = %s ORDER BY h.store_key, h.product_key, h.time_key"
        params: tuple[Any, ...] = (date_key,)
        if remaining is not None:
            limit_clause += " LIMIT %s"
            params = (date_key, remaining)

        df = read_sql(conn, hourly_feature_sql(limit_clause), params)
        if df.empty:
            continue

        predictions = np.clip(model.predict(feature_matrix(df)), 0, None)
        observed = df["observed_sales_amount"].astype(float).to_numpy()
        stockout = df["stockout_flag"].astype(bool).to_numpy()
        estimated_lost_sales = np.where(stockout, np.maximum(predictions - observed, 0), 0)

        prediction_df = pd.DataFrame(
            {
                "date_key": df["date_key"].astype(int),
                "time_key": df["time_key"].astype(int),
                "store_key": df["store_key"].astype(int),
                "product_key": df["product_key"].astype(int),
                "model_key": model_key,
                "observed_sales_amount": observed,
                "estimated_true_demand": predictions,
                "estimated_lost_sales": estimated_lost_sales,
                "stockout_flag": stockout,
                "is_censored_observation": stockout,
                "prediction_lower_bound": np.maximum(predictions * 0.85, 0),
                "prediction_upper_bound": predictions * 1.15,
            }
        )

        insert_prediction_chunk(conn, prediction_df)
        loaded_rows += len(prediction_df)
        print(f"loaded predictions for date_key {date_key} ({index}/{len(date_keys)}): {len(prediction_df):,} rows; total {loaded_rows:,}")


def load_recommendations(conn: psycopg.Connection, model_key: int) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT date_key FROM dw.fact_demand_estimate_hourly WHERE model_key = %s ORDER BY date_key", (model_key,))
        date_keys = [row[0] for row in cur.fetchall()]

    total_rows = 0
    for index, date_key in enumerate(date_keys, start=1):
        rows = load_recommendations_for_date(conn, model_key, date_key)
        total_rows += rows
        print(f"loaded recommendations for date_key {date_key} ({index}/{len(date_keys)}): {rows:,} rows; total {total_rows:,}")


def load_recommendations_for_date(conn: psycopg.Connection, model_key: int, date_key: int) -> int:
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
                    SUM(estimated_lost_sales) AS expected_lost_sales,
                    SUM(CASE WHEN stockout_flag THEN 1 ELSE 0 END) AS stockout_hours_total
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
                GREATEST(m.expected_demand + m.expected_lost_sales, 0) AS recommended_order_qty,
                m.expected_demand,
                m.expected_lost_sales,
                LEAST(
                    1,
                    (m.stockout_hours_total / 24.0) * 0.70
                    + CASE WHEN m.expected_demand > 0 THEN (m.expected_lost_sales / m.expected_demand) * 0.30 ELSE 0 END
                ) AS stockout_risk_score,
                GREATEST(f.observed_daily_sales_amount - m.expected_demand, 0) AS expected_waste_qty,
                0.95 AS service_level_target
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
            (model_key, date_key),
        )
        rows = cur.rowcount
    conn.commit()
    return rows


def print_counts(conn: psycopg.Connection, model_key: int) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM dw.fact_demand_estimate_hourly WHERE model_key = %s", (model_key,))
        demand_rows = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM dw.fact_replenishment_recommendation_daily WHERE model_key = %s", (model_key,))
        recommendation_rows = cur.fetchone()[0]
    print(f"dw.fact_demand_estimate_hourly rows for model: {demand_rows:,}")
    print(f"dw.fact_replenishment_recommendation_daily rows for model: {recommendation_rows:,}")


def main() -> None:
    args = parse_args()
    model_version = args.model_version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    with connect(args) as conn:
        print("loading hourly training data")
        train_df = load_training_data(conn, args.max_train_rows, args.train_sample_rate)
        train_df = add_latent_target(train_df)
        print(f"training rows: {len(train_df):,}")

        print(f"training {args.model_type} latent-demand model")
        model = train_model(args, train_df)

        print("evaluating model")
        metrics = evaluate_model(conn, model, train_df, args)
        for name, value in metrics.items():
            print(f"{name}: {value:.6f}")

        model_path, metrics_path = save_artifacts(args, model, metrics, model_version)
        print(f"saved model: {model_path}")
        print(f"saved metrics: {metrics_path}")

        model_key = upsert_model_metadata(conn, args.model_name, model_version, train_df)
        print(f"model_key: {model_key}")

        if args.load_predictions:
            print("loading model predictions into warehouse")
            load_predictions(conn, model, model_key, args.max_predict_rows)
            load_recommendations(conn, model_key)
            print_counts(conn, model_key)
        else:
            print("skipping warehouse prediction load; pass --load-predictions to populate DSS model facts")


if __name__ == "__main__":
    main()
