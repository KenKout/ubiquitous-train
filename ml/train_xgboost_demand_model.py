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
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


BASE_FEATURE_COLUMNS = [
    "hour_of_day",
    "is_business_hour_6_22",
    "day_of_week",
    "is_weekend",
    "holiday_flag",
    "activity_flag",
    "discount_rate",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "city_id",
    "store_id",
    "product_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
]

ADVANCED_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "discount_depth",
    "is_discounted",
    "is_promo_or_discount",
    "temperature_humidity_index",
    "store_product_hour_mean",
    "store_product_hour_count_log1p",
    "product_hour_mean",
    "product_hour_count_log1p",
    "category_hour_mean",
    "category_hour_count_log1p",
    "product_dow_mean",
    "product_dow_count_log1p",
    "category_dow_mean",
    "category_dow_count_log1p",
    "store_product_mean",
    "store_product_count_log1p",
    "product_mean",
    "product_count_log1p",
    "category_mean",
    "category_count_log1p",
    "global_hour_mean",
    "global_dow_mean",
    "baseline_demand_prior",
    "store_product_stockout_rate",
    "product_stockout_rate",
    "category_stockout_rate",
    "store_traffic_lag_1h",
    "category_momentum_24h",
    "product_sales_lag_1h",
    "product_sales_lag_24h_sum",
    "days_since_previous_sale",
]

RUN_STARTED_AT = perf_counter()


def log(message: str) -> None:
    elapsed = perf_counter() - RUN_STARTED_AT
    print(f"[+{elapsed:,.1f}s] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a warehouse-native hourly demand model from non-stockout observations.")
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
    parser.add_argument("--max-eval-rows", type=int, default=None, help="Limit eval rows only for smoke tests. Default: use the full eval split.")
    parser.add_argument("--eval-sample-rate", type=float, default=None, help="Randomly sample eval rows while scanning the hourly fact.")
    parser.add_argument("--sample-mode", choices=["product-date-stratified", "date-stratified", "random", "ordered"], default="product-date-stratified", help="How to cap rows when max row limits are set without sample rates.")
    parser.add_argument("--max-predict-rows", type=int, default=None, help="Limit prediction loading for faster tests.")
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--n-jobs", type=int, default=max((os.cpu_count() or 2) - 1, 1), help="CPU threads for Apple Silicon/native CPU training.")
    parser.add_argument("--max-bin", type=int, default=256, help="Histogram bin count for XGBoost CPU hist training.")
    parser.add_argument("--xgboost-objective", choices=["reg:tweedie", "reg:squarederror", "count:poisson"], default="reg:tweedie")
    parser.add_argument("--tweedie-variance-power", type=float, default=1.4)
    parser.add_argument("--model-strategy", choices=["direct", "hurdle"], default="hurdle", help="hurdle trains positive-demand probability and positive amount separately for zero-heavy hourly demand.")
    parser.add_argument("--positive-threshold", type=float, default=0.0, help="Observed sales threshold for the positive-demand classifier in hurdle mode.")
    parser.add_argument("--prior-blend-weight", type=float, default=0.35, help="Blend weight for baseline_demand_prior in hurdle predictions when advanced features are enabled.")
    parser.add_argument("--disable-advanced-features", action="store_true", help="Use only raw warehouse features, without historical demand priors.")
    parser.add_argument("--calibration-holdout-fraction", type=float, default=0.15, help="Fraction of the tail of the train split reserved for calibration. Eval is never used for calibration.")
    parser.add_argument("--disable-calibration", "--disable-eval-calibration", dest="disable_calibration", action="store_true", help="Do not scale predictions with a train-only calibration holdout.")
    parser.add_argument("--calibration-objective", choices=["bias", "wmape", "balanced"], default="balanced", help="Choose the train-holdout scale factor by aggregate bias, lowest WMAPE, or a WMAPE/bias tradeoff.")
    parser.add_argument("--calibration-bias-penalty", type=float, default=0.25, help="Bias penalty used by --calibration-objective balanced.")
    parser.add_argument("--calibration-search-steps", type=int, default=121, help="Grid search steps for WMAPE-aware calibration.")
    parser.add_argument("--min-calibration-factor", type=float, default=0.25, help="Lower safety bound for train-holdout calibration.")
    parser.add_argument("--max-calibration-factor", type=float, default=5.0, help="Upper safety bound for train-holdout calibration to avoid inflated local DSS quantities.")
    parser.add_argument("--random-state", type=int, default=42)
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
        SELECT *
        FROM dw.v_model_training_features_hourly h
        {limit_clause}
    """


def capped_split_sql(source_split: str, max_rows: int, sample_mode: str) -> str:
    base_filter = f"h.source_split = '{source_split}' AND h.is_trainable_demand_observation"
    if sample_mode == "random":
        return hourly_feature_sql(f"WHERE {base_filter} ORDER BY random() LIMIT %s")
    if sample_mode == "ordered":
        return hourly_feature_sql(f"WHERE {base_filter} ORDER BY h.date_key, h.store_key, h.product_key, h.time_key LIMIT %s")
    if sample_mode == "product-date-stratified":
        return f"""
            WITH group_count AS (
                SELECT COUNT(*) AS groups
                FROM (
                    SELECT h.full_date, h.product_id
                    FROM dw.v_model_training_features_hourly h
                    WHERE {base_filter}
                    GROUP BY h.full_date, h.product_id
                ) g
            ), eligible AS (
                SELECT
                    h.*,
                    ROW_NUMBER() OVER (PARTITION BY h.full_date, h.product_id ORDER BY random()) AS sample_rank
                FROM dw.v_model_training_features_hourly h
                WHERE {base_filter}
            )
            SELECT {', '.join(f'e.{column}' for column in ['date_key', 'full_date', 'time_key', 'hour_of_day', 'is_business_hour_6_22', 'store_key', 'store_id', 'city_id', 'product_key', 'product_id', 'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id', 'source_split', 'observed_sales_amount', 'stockout_flag', 'is_censored_observation', 'is_trainable_demand_observation', 'target_observed_sales_amount', 'discount_rate', 'activity_flag', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level', 'day_of_week', 'is_weekend', 'holiday_flag'])}
            FROM eligible e
            CROSS JOIN group_count g
            WHERE e.sample_rank <= CEIL(%s::NUMERIC / GREATEST(g.groups, 1))::INTEGER
            ORDER BY e.full_date, e.product_id, e.sample_rank
            LIMIT %s
        """
    return f"""
        WITH date_count AS (
            SELECT COUNT(DISTINCT h.full_date) AS dates
            FROM dw.v_model_training_features_hourly h
            WHERE {base_filter}
        ), eligible AS (
            SELECT
                h.*,
                ROW_NUMBER() OVER (PARTITION BY h.full_date ORDER BY random()) AS sample_rank
            FROM dw.v_model_training_features_hourly h
            WHERE {base_filter}
        )
        SELECT {', '.join(f'e.{column}' for column in ['date_key', 'full_date', 'time_key', 'hour_of_day', 'is_business_hour_6_22', 'store_key', 'store_id', 'city_id', 'product_key', 'product_id', 'management_group_id', 'first_category_id', 'second_category_id', 'third_category_id', 'source_split', 'observed_sales_amount', 'stockout_flag', 'is_censored_observation', 'is_trainable_demand_observation', 'target_observed_sales_amount', 'discount_rate', 'activity_flag', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level', 'day_of_week', 'is_weekend', 'holiday_flag'])}
        FROM eligible e
        CROSS JOIN date_count d
        WHERE e.sample_rank <= CEIL(%s::NUMERIC / GREATEST(d.dates, 1))::INTEGER
        ORDER BY e.full_date, e.sample_rank
        LIMIT %s
    """


def load_training_data(
    conn: psycopg.Connection,
    max_train_rows: int | None,
    train_sample_rate: float | None,
    sample_mode: str,
) -> pd.DataFrame:
    limit_clause = ""
    params: tuple[Any, ...] = ()
    if train_sample_rate is not None and max_train_rows:
        limit_clause = "WHERE h.source_split = 'train' AND h.is_trainable_demand_observation AND random() < %s LIMIT %s"
        params = (train_sample_rate, max_train_rows)
    elif train_sample_rate is not None:
        limit_clause = "WHERE h.source_split = 'train' AND h.is_trainable_demand_observation AND random() < %s"
        params = (train_sample_rate,)
    elif max_train_rows:
        params = (max_train_rows, max_train_rows) if sample_mode in {"product-date-stratified", "date-stratified"} else (max_train_rows,)
        df = read_sql(conn, capped_split_sql("train", max_train_rows, sample_mode), params)
        if df.empty:
            raise RuntimeError("No hourly training rows found. Run `python3 etl/load_fresh_retail_dw.py --reset --load-hourly` first.")
        return df
    else:
        limit_clause = "WHERE h.source_split = 'train' AND h.is_trainable_demand_observation"

    df = read_sql(conn, hourly_feature_sql(limit_clause), params)
    if df.empty:
        raise RuntimeError("No hourly training rows found. Run `python3 etl/load_fresh_retail_dw.py --reset --load-hourly` first.")
    return df


def load_eval_data(
    conn: psycopg.Connection,
    max_eval_rows: int | None,
    eval_sample_rate: float | None,
    sample_mode: str,
) -> pd.DataFrame:
    if eval_sample_rate is not None and max_eval_rows:
        eval_df = read_sql(conn, hourly_feature_sql("WHERE h.source_split = 'eval' AND h.is_trainable_demand_observation AND random() < %s LIMIT %s"), (eval_sample_rate, max_eval_rows))
    elif eval_sample_rate is not None:
        eval_df = read_sql(conn, hourly_feature_sql("WHERE h.source_split = 'eval' AND h.is_trainable_demand_observation AND random() < %s"), (eval_sample_rate,))
    elif max_eval_rows:
        params = (max_eval_rows, max_eval_rows) if sample_mode in {"product-date-stratified", "date-stratified"} else (max_eval_rows,)
        eval_df = read_sql(conn, capped_split_sql("eval", max_eval_rows, sample_mode), params)
    else:
        eval_df = read_sql(conn, hourly_feature_sql("WHERE h.source_split = 'eval' AND h.is_trainable_demand_observation"))
    return eval_df


def add_observed_demand_target(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    target = data.get("target_observed_sales_amount", data["observed_sales_amount"])
    data["demand_target"] = pd.to_numeric(target, errors="coerce").fillna(data["observed_sales_amount"]).astype(float).clip(lower=0)
    return data


def split_train_calibration(train_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    fraction = float(args.calibration_holdout_fraction or 0)
    if args.disable_calibration or fraction <= 0 or len(train_df) < 2:
        return train_df.reset_index(drop=True), train_df.iloc[0:0].copy()
    if fraction >= 1:
        raise ValueError("--calibration-holdout-fraction must be less than 1.0")

    sort_columns = [column for column in ["full_date", "time_key", "store_id", "product_id"] if column in train_df.columns]
    ordered = train_df.sort_values(sort_columns).reset_index(drop=True) if sort_columns else train_df.reset_index(drop=True)
    if "full_date" in ordered.columns:
        dates = pd.Series(ordered["full_date"].dropna().unique()).sort_values().to_list()
        if len(dates) >= 2:
            holdout_date_count = int(np.ceil(len(dates) * fraction))
            holdout_date_count = max(1, min(holdout_date_count, len(dates) - 1))
            calibration_dates = set(dates[-holdout_date_count:])
            calibration_mask = ordered["full_date"].isin(calibration_dates)
            fit_df = ordered.loc[~calibration_mask].reset_index(drop=True)
            calibration_df = ordered.loc[calibration_mask].reset_index(drop=True)
            if not fit_df.empty and not calibration_df.empty:
                return fit_df, calibration_df

    split_index = int(np.floor(len(ordered) * (1 - fraction)))
    split_index = max(1, min(split_index, len(ordered) - 1))
    return ordered.iloc[:split_index].reset_index(drop=True), ordered.iloc[split_index:].reset_index(drop=True)


def merge_mean_count(
    features: pd.DataFrame,
    reference: pd.DataFrame,
    keys: list[str],
    mean_column: str,
    count_column: str,
) -> pd.DataFrame:
    stats = (
        reference.groupby(keys, dropna=False)["observed_sales_amount"]
        .agg(["mean", "count"])
        .rename(columns={"mean": mean_column, "count": count_column})
        .reset_index()
    )
    return features.merge(stats, on=keys, how="left")


def merge_rate(
    features: pd.DataFrame,
    reference: pd.DataFrame,
    keys: list[str],
    rate_column: str,
) -> pd.DataFrame:
    stats = reference.groupby(keys, dropna=False)["stockout_flag"].mean().rename(rate_column).reset_index()
    return features.merge(stats, on=keys, how="left")


def safe_shifted_rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=1).sum()


def add_safe_temporal_features(features: pd.DataFrame) -> pd.DataFrame:
    """RetailForecast-style temporal context, shifted so current-row demand never leaks."""
    result = features.copy().reset_index(drop=True)
    result["_row_order"] = np.arange(len(result))
    result = result.sort_values(["source_split", "date_key", "time_key", "store_id", "product_id"]).reset_index(drop=True)

    result["product_sales_lag_1h"] = (
        result.groupby(["source_split", "store_id", "product_id"], dropna=False)["observed_sales_amount"]
        .shift(1)
        .fillna(0)
    )
    result["product_sales_lag_24h_sum"] = (
        result.groupby(["source_split", "store_id", "product_id"], group_keys=False, dropna=False)["observed_sales_amount"]
        .apply(lambda values: safe_shifted_rolling_sum(values, 24))
        .fillna(0)
    )

    timestamp = pd.to_datetime(result["full_date"]) + pd.to_timedelta(result["hour_of_day"].astype(int), unit="h")
    result["_timestamp"] = timestamp
    result["_sale_timestamp"] = result["_timestamp"].where(pd.to_numeric(result["observed_sales_amount"], errors="coerce").fillna(0) > 0)
    result["_previous_sale_timestamp"] = result.groupby(
        ["source_split", "store_id", "product_id"],
        group_keys=False,
        dropna=False,
    )["_sale_timestamp"].transform(lambda values: values.ffill().shift(1))
    result["days_since_previous_sale"] = (
        (result["_timestamp"] - result["_previous_sale_timestamp"]).dt.total_seconds() / 86_400.0
    ).fillna(-1)

    store_hourly = (
        result.groupby(["source_split", "store_id", "date_key", "time_key"], dropna=False)["observed_sales_amount"]
        .sum()
        .rename("store_hour_sales")
        .reset_index()
        .sort_values(["source_split", "store_id", "date_key", "time_key"])
    )
    store_hourly["store_traffic_lag_1h"] = (
        store_hourly.groupby(["source_split", "store_id"], dropna=False)["store_hour_sales"]
        .shift(1)
        .fillna(0)
    )
    result = result.merge(
        store_hourly[["source_split", "store_id", "date_key", "time_key", "store_traffic_lag_1h"]],
        on=["source_split", "store_id", "date_key", "time_key"],
        how="left",
    )

    category_hourly = (
        result.groupby(["source_split", "first_category_id", "date_key", "time_key"], dropna=False)["observed_sales_amount"]
        .sum()
        .rename("category_hour_sales")
        .reset_index()
        .sort_values(["source_split", "first_category_id", "date_key", "time_key"])
    )
    category_hourly["category_momentum_24h"] = (
        category_hourly.groupby(["source_split", "first_category_id"], group_keys=False, dropna=False)["category_hour_sales"]
        .apply(lambda values: safe_shifted_rolling_sum(values, 24))
        .fillna(0)
    )
    result = result.merge(
        category_hourly[["source_split", "first_category_id", "date_key", "time_key", "category_momentum_24h"]],
        on=["source_split", "first_category_id", "date_key", "time_key"],
        how="left",
    )

    result = result.sort_values("_row_order").drop(
        columns=["_row_order", "_timestamp", "_sale_timestamp", "_previous_sale_timestamp"],
        errors="ignore",
    )
    return result.reset_index(drop=True)


def add_advanced_features(features: pd.DataFrame, reference_train: pd.DataFrame | None = None) -> pd.DataFrame:
    started_at = perf_counter()
    log(f"engineering advanced features for {len(features):,} rows")
    result = features.copy()
    result["hour_sin"] = np.sin(2 * np.pi * result["hour_of_day"].astype(float) / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour_of_day"].astype(float) / 24.0)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"].astype(float) / 7.0)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"].astype(float) / 7.0)
    result["discount_depth"] = np.maximum(1.0 - pd.to_numeric(result["discount_rate"], errors="coerce").fillna(1.0), 0)
    result["is_discounted"] = result["discount_depth"] > 0.001
    result["is_promo_or_discount"] = result["is_discounted"] | (pd.to_numeric(result["activity_flag"], errors="coerce").fillna(0) != 0)
    result["temperature_humidity_index"] = (
        pd.to_numeric(result["avg_temperature"], errors="coerce").fillna(0)
        * pd.to_numeric(result["avg_humidity"], errors="coerce").fillna(0)
    ) / 100.0
    result = add_safe_temporal_features(result)

    if reference_train is None:
        train_all = result[result["source_split"] == "train"]
    else:
        train_all = reference_train.copy()
    non_stockout_train = train_all[train_all["is_trainable_demand_observation"].astype(bool)]
    global_mean = float(non_stockout_train["observed_sales_amount"].mean()) if not non_stockout_train.empty else 0.0
    log(f"advanced feature reference rows: {len(non_stockout_train):,} trainable rows; {len(train_all):,} total train rows")

    demand_groups = [
        (["store_id", "product_id", "hour_of_day"], "store_product_hour_mean", "store_product_hour_count"),
        (["product_id", "hour_of_day"], "product_hour_mean", "product_hour_count"),
        (["first_category_id", "hour_of_day"], "category_hour_mean", "category_hour_count"),
        (["product_id", "day_of_week"], "product_dow_mean", "product_dow_count"),
        (["first_category_id", "day_of_week"], "category_dow_mean", "category_dow_count"),
        (["store_id", "product_id"], "store_product_mean", "store_product_count"),
        (["product_id"], "product_mean", "product_count"),
        (["first_category_id"], "category_mean", "category_count"),
        (["hour_of_day"], "global_hour_mean", "global_hour_count"),
        (["day_of_week"], "global_dow_mean", "global_dow_count"),
    ]
    for index, (keys, mean_column, count_column) in enumerate(demand_groups, start=1):
        group_started_at = perf_counter()
        log(f"building demand prior {index}/{len(demand_groups)}: {mean_column} by {', '.join(keys)}")
        result = merge_mean_count(result, non_stockout_train, keys, mean_column, count_column)
        result[f"{count_column}_log1p"] = np.log1p(pd.to_numeric(result[count_column], errors="coerce").fillna(0))
        result = result.drop(columns=[count_column])
        log(f"finished {mean_column} in {perf_counter() - group_started_at:,.1f}s")

    result["baseline_demand_prior"] = (
        result["store_product_hour_mean"]
        .fillna(result["product_hour_mean"])
        .fillna(result["category_hour_mean"])
        .fillna(result["product_dow_mean"])
        .fillna(result["category_dow_mean"])
        .fillna(result["store_product_mean"])
        .fillna(result["product_mean"])
        .fillna(result["category_mean"])
        .fillna(result["global_hour_mean"])
        .fillna(result["global_dow_mean"])
        .fillna(global_mean)
    )

    stockout_groups = [
        (["store_id", "product_id"], "store_product_stockout_rate"),
        (["product_id"], "product_stockout_rate"),
        (["first_category_id"], "category_stockout_rate"),
    ]
    for index, (keys, rate_column) in enumerate(stockout_groups, start=1):
        group_started_at = perf_counter()
        log(f"building stockout prior {index}/{len(stockout_groups)}: {rate_column} by {', '.join(keys)}")
        result = merge_rate(result, train_all, keys, rate_column)
        log(f"finished {rate_column} in {perf_counter() - group_started_at:,.1f}s")

    for column in ADVANCED_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = 0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
    log(f"advanced feature engineering completed in {perf_counter() - started_at:,.1f}s")
    return result


def feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = df[feature_columns].copy()
    for column in ["is_business_hour_6_22", "is_weekend"]:
        if column in x.columns:
            x[column] = x[column].astype(int)
    for column in feature_columns:
        x[column] = pd.to_numeric(x[column], errors="coerce").fillna(0)
    return x


def train_model(args: argparse.Namespace, train_df: pd.DataFrame, feature_columns: list[str]):
    log("building training matrix")
    x_train = feature_matrix(train_df, feature_columns)
    y_train = train_df["demand_target"].astype(float)

    if args.model_type == "xgboost" and args.model_strategy == "hurdle":
        positive_mask = y_train > args.positive_threshold
        positive_rows = int(positive_mask.sum())
        negative_rows = int(len(y_train) - positive_rows)
        if positive_rows == 0 or negative_rows == 0:
            log("hurdle strategy requires both zero and positive examples; falling back to direct regressor")
        else:
            scale_pos_weight = max(negative_rows / max(positive_rows, 1), 1.0)
            classifier = XGBClassifier(
                n_estimators=max(100, min(args.n_estimators, 400)),
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cpu",
                max_bin=args.max_bin,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                scale_pos_weight=scale_pos_weight,
            )
            amount_model = XGBRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                objective="reg:squarederror",
                tree_method="hist",
                device="cpu",
                max_bin=args.max_bin,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )
            log(f"fitting hurdle classifier with {positive_rows:,} positive and {negative_rows:,} zero rows")
            fit_started_at = perf_counter()
            classifier.fit(x_train, positive_mask.astype(int))
            log(f"hurdle classifier fit completed in {perf_counter() - fit_started_at:,.1f}s")
            log(f"fitting positive-amount regressor with {positive_rows:,} rows")
            fit_started_at = perf_counter()
            amount_model.fit(x_train.loc[positive_mask], np.log1p(y_train.loc[positive_mask]))
            log(f"positive-amount regressor fit completed in {perf_counter() - fit_started_at:,.1f}s")
            return {
                "strategy": "hurdle",
                "classifier": classifier,
                "amount_model": amount_model,
                "prior_blend_weight": args.prior_blend_weight if not args.disable_advanced_features else 0.0,
                "positive_threshold": args.positive_threshold,
            }

    if args.model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            objective=args.xgboost_objective,
            tree_method="hist",
            device="cpu",
            max_bin=args.max_bin,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        if args.xgboost_objective == "reg:tweedie":
            model.set_params(tweedie_variance_power=args.tweedie_variance_power)
    elif args.model_type == "catboost":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed")
        model = CatBoostRegressor(
            iterations=args.n_estimators,
            depth=args.max_depth,
            learning_rate=args.learning_rate,
            loss_function="Tweedie:variance_power=1.5",
            random_seed=args.random_state,
            thread_count=args.n_jobs,
            verbose=False,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    log(f"fitting {args.model_type} with {len(train_df):,} rows and {len(feature_columns):,} features")
    fit_started_at = perf_counter()
    model.fit(x_train, y_train)
    log(f"model fit completed in {perf_counter() - fit_started_at:,.1f}s")
    return model


def predict_demand(model: Any, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    x = feature_matrix(df, feature_columns)
    if isinstance(model, dict) and model.get("strategy") == "hurdle":
        classifier = model["classifier"]
        amount_model = model["amount_model"]
        if hasattr(classifier, "predict_proba"):
            positive_probability = classifier.predict_proba(x)[:, 1]
        else:
            positive_probability = classifier.predict(x)
        positive_amount = np.expm1(amount_model.predict(x))
        predictions = np.clip(positive_probability * positive_amount, 0, None)
        prior_weight = float(model.get("prior_blend_weight", 0.0))
        if prior_weight > 0 and "baseline_demand_prior" in df.columns:
            prior = pd.to_numeric(df["baseline_demand_prior"], errors="coerce").fillna(0).astype(float).to_numpy()
            predictions = (1 - prior_weight) * predictions + prior_weight * prior
        return np.clip(predictions, 0, None)
    return np.clip(model.predict(x), 0, None)


def evaluate_predictions(y_eval: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    denominator = np.sum(np.abs(y_eval))
    return {
        "rows": float(len(y_eval)),
        "mae": float(mean_absolute_error(y_eval, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_eval, y_pred))),
        "wmape": float(np.sum(np.abs(y_eval - y_pred)) / denominator) if denominator > 0 else 0.0,
        "bias": float(np.sum(y_pred - y_eval) / denominator) if denominator > 0 else 0.0,
    }


def evaluate_model(model: Any, eval_df: pd.DataFrame, feature_columns: list[str], calibration_factor: float = 1.0) -> dict[str, float]:
    y_eval = eval_df["demand_target"].astype(float).to_numpy()
    y_pred = np.clip(predict_demand(model, eval_df, feature_columns) * calibration_factor, 0, None)
    return evaluate_predictions(y_eval, y_pred)


def compute_calibration_factor(model: Any, eval_df: pd.DataFrame, feature_columns: list[str]) -> float:
    y_eval = eval_df["demand_target"].astype(float).to_numpy()
    y_pred = predict_demand(model, eval_df, feature_columns)
    prediction_sum = float(np.sum(y_pred))
    if prediction_sum <= 0:
        return 1.0
    return float(np.sum(y_eval) / prediction_sum)


def metric_for_calibration_factor(y_eval: np.ndarray, y_pred: np.ndarray, factor: float) -> dict[str, float]:
    return evaluate_predictions(y_eval, np.clip(y_pred * factor, 0, None))


def compute_wmape_calibration_factor(y_eval: np.ndarray, y_pred: np.ndarray) -> float:
    positive_prediction = y_pred > 1e-12
    if not np.any(positive_prediction):
        return 1.0
    ratios = y_eval[positive_prediction] / y_pred[positive_prediction]
    weights = y_pred[positive_prediction]
    finite = np.isfinite(ratios) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite):
        return 1.0
    ratios = ratios[finite]
    weights = weights[finite]
    order = np.argsort(ratios)
    sorted_ratios = ratios[order]
    sorted_weights = weights[order]
    midpoint = sorted_weights.sum() / 2.0
    return float(sorted_ratios[np.searchsorted(np.cumsum(sorted_weights), midpoint, side="left")])


def choose_calibration_factor(
    args: argparse.Namespace,
    model: Any,
    eval_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[float, float, float]:
    y_eval = eval_df["demand_target"].astype(float).to_numpy()
    y_pred = predict_demand(model, eval_df, feature_columns)
    prediction_sum = float(np.sum(y_pred))
    raw_bias_factor = float(np.sum(y_eval) / prediction_sum) if prediction_sum > 0 else 1.0
    raw_wmape_factor = compute_wmape_calibration_factor(y_eval, y_pred)

    lower = args.min_calibration_factor
    upper = args.max_calibration_factor
    if lower <= 0 or upper <= 0 or lower > upper:
        raise ValueError("Calibration factor bounds must satisfy 0 < min <= max.")

    bias_factor = min(max(raw_bias_factor, lower), upper)
    wmape_factor = min(max(raw_wmape_factor, lower), upper)
    if args.calibration_objective == "bias":
        return bias_factor, raw_bias_factor, raw_wmape_factor
    if args.calibration_objective == "wmape":
        return wmape_factor, raw_bias_factor, raw_wmape_factor

    steps = max(args.calibration_search_steps, 3)
    candidates = np.linspace(lower, upper, steps)
    candidates = np.unique(np.concatenate([candidates, np.array([1.0, bias_factor, wmape_factor])]))
    best_factor = bias_factor
    best_score = float("inf")
    for factor in candidates:
        metrics = metric_for_calibration_factor(y_eval, y_pred, float(factor))
        score = metrics["wmape"] + args.calibration_bias_penalty * abs(metrics["bias"])
        if score < best_score:
            best_score = score
            best_factor = float(factor)
    return best_factor, raw_bias_factor, raw_wmape_factor


def evaluate_segments(model: Any, eval_df: pd.DataFrame, feature_columns: list[str], calibration_factor: float) -> dict[str, dict[str, float]]:
    segment_masks = {
        "business_hour": eval_df["is_business_hour_6_22"].astype(bool),
        "non_business_hour": ~eval_df["is_business_hour_6_22"].astype(bool),
        "weekend": eval_df["is_weekend"].astype(bool),
        "weekday": ~eval_df["is_weekend"].astype(bool),
        "promo_or_discount": eval_df.get("is_promo_or_discount", pd.Series(False, index=eval_df.index)).astype(bool),
    }
    segments: dict[str, dict[str, float]] = {}
    for name, mask in segment_masks.items():
        segment = eval_df.loc[mask]
        if len(segment) == 0:
            continue
        segments[name] = evaluate_model(model, segment, feature_columns, calibration_factor)
    return segments


def save_artifacts(
    args: argparse.Namespace,
    model: Any,
    metrics: dict[str, float],
    model_version: str,
    feature_columns: list[str],
    calibration_factor: float,
) -> tuple[Path, Path]:
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"{args.model_name}_{model_version}.pkl"
    metrics_path = artifact_dir / f"{args.model_name}_{model_version}_metrics.json"

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "model_name": args.model_name,
        "model_type": args.model_type,
        "model_strategy": args.model_strategy if args.model_type == "xgboost" else "direct",
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target_definition": "observed hourly sales on non-stockout rows only",
        "advanced_features_enabled": not args.disable_advanced_features,
        "calibration_factor": calibration_factor,
        "calibration_definition": "prediction scale factor selected on a train-only temporal calibration holdout; eval split is reserved for final future evaluation",
        "calibration_objective": args.calibration_objective,
        "xgboost_objective": args.xgboost_objective if args.model_type == "xgboost" else None,
        "tweedie_variance_power": args.tweedie_variance_power if args.model_type == "xgboost" and args.xgboost_objective == "reg:tweedie" else None,
        "prior_blend_weight": args.prior_blend_weight if args.model_type == "xgboost" and args.model_strategy == "hurdle" else None,
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


def upsert_model_evaluation(
    conn: psycopg.Connection,
    model_key: int,
    metrics: dict[str, float],
    uncalibrated_metrics: dict[str, float],
    calibration_metrics: dict[str, float],
    calibration_uncalibrated_metrics: dict[str, float],
    segments: dict[str, dict[str, float]],
    calibration_factor: float,
) -> None:
    rows: list[tuple[int, str, str, float, str]] = []
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            rows.append((model_key, "eval", metric_name, float(metric_value), json.dumps({})))
    for metric_name, metric_value in uncalibrated_metrics.items():
        rows.append((model_key, "eval_uncalibrated", metric_name, float(metric_value), json.dumps({})))
        rows.append((model_key, "eval", f"uncalibrated_{metric_name}", float(metric_value), json.dumps({})))
    for metric_name, metric_value in calibration_metrics.items():
        rows.append((model_key, "train_calibration", metric_name, float(metric_value), json.dumps({"split": "train_calibration"})))
    for metric_name, metric_value in calibration_uncalibrated_metrics.items():
        rows.append((model_key, "train_calibration_uncalibrated", metric_name, float(metric_value), json.dumps({"split": "train_calibration"})))
    for segment_name, segment_metrics in segments.items():
        for metric_name, metric_value in segment_metrics.items():
            rows.append((model_key, f"segment:{segment_name}", metric_name, float(metric_value), json.dumps({"segment": segment_name})))

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO dw.fact_model_evaluation (model_key, evaluation_split, metric_name, metric_value, metric_context)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (model_key, evaluation_split, metric_name) DO UPDATE SET
                metric_value = EXCLUDED.metric_value,
                metric_context = EXCLUDED.metric_context,
                created_at = NOW()
            """,
            rows,
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


def load_predictions(
    conn: psycopg.Connection,
    args: argparse.Namespace,
    model: Any,
    model_key: int,
    max_predict_rows: int | None,
    feature_columns: list[str],
    calibration_factor: float,
    train_reference_df: pd.DataFrame,
) -> None:
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

        prediction_features = df.copy()
        prediction_features["source_split"] = prediction_features.get("source_split", pd.Series("predict", index=prediction_features.index))
        if not args.disable_advanced_features:
            combined = pd.concat([train_reference_df, prediction_features], ignore_index=True)
            combined = add_advanced_features(combined, train_reference_df)
            prediction_features = combined.iloc[len(train_reference_df) :].reset_index(drop=True)

        predictions = np.clip(predict_demand(model, prediction_features, feature_columns) * calibration_factor, 0, None)
        observed = prediction_features["observed_sales_amount"].astype(float).to_numpy()
        stockout = prediction_features["stockout_flag"].astype(bool).to_numpy()
        estimated_true_demand = np.where(stockout, np.maximum(predictions, observed), observed)
        estimated_lost_sales = np.where(stockout, np.maximum(estimated_true_demand - observed, 0), 0)
        prediction_lower_bound = np.where(stockout, np.maximum(estimated_true_demand * 0.85, observed), observed)
        prediction_upper_bound = np.where(stockout, np.maximum(estimated_true_demand * 1.15, observed), observed)

        prediction_df = pd.DataFrame(
            {
                "date_key": prediction_features["date_key"].astype(int),
                "time_key": prediction_features["time_key"].astype(int),
                "store_key": prediction_features["store_key"].astype(int),
                "product_key": prediction_features["product_key"].astype(int),
                "model_key": model_key,
                "observed_sales_amount": observed,
                "estimated_true_demand": estimated_true_demand,
                "estimated_lost_sales": estimated_lost_sales,
                "stockout_flag": stockout,
                "is_censored_observation": stockout,
                "prediction_lower_bound": prediction_lower_bound,
                "prediction_upper_bound": prediction_upper_bound,
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
                GREATEST(m.expected_demand, 0) AS recommended_order_qty,
                m.expected_demand,
                m.expected_lost_sales,
                LEAST(
                    1,
                    (f.stockout_hours_6_22 / 16.0) * 0.55
                    + CASE WHEN m.expected_demand > 0 THEN (m.expected_lost_sales / m.expected_demand) * 0.35 ELSE 0 END
                    + CASE WHEN f.activity_flag <> 0 THEN 0.10 ELSE 0 END
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


def prepare_features(
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    feature_columns = BASE_FEATURE_COLUMNS.copy()
    raw_train_reference = train_df.copy()
    combined = pd.concat([train_df, calibration_df, eval_df], ignore_index=True)
    train_end = len(train_df)
    calibration_end = train_end + len(calibration_df)
    if args.disable_advanced_features:
        log("advanced feature engineering disabled")
        return (
            combined.iloc[:train_end].reset_index(drop=True),
            combined.iloc[train_end:calibration_end].reset_index(drop=True),
            combined.iloc[calibration_end:].reset_index(drop=True),
            feature_columns,
            raw_train_reference,
        )

    log("adding train-only historical demand and stockout features")
    combined = add_advanced_features(combined, raw_train_reference)
    feature_columns += ADVANCED_FEATURE_COLUMNS
    prepared_train = combined.iloc[:train_end].reset_index(drop=True)
    prepared_calibration = combined.iloc[train_end:calibration_end].reset_index(drop=True)
    prepared_eval = combined.iloc[calibration_end:].reset_index(drop=True)
    return prepared_train, prepared_calibration, prepared_eval, feature_columns, raw_train_reference


def main() -> None:
    args = parse_args()
    model_version = args.model_version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    with connect(args) as conn:
        log("loading hourly training data")
        raw_train_df = load_training_data(conn, args.max_train_rows, args.train_sample_rate, args.sample_mode)
        raw_train_df = add_observed_demand_target(raw_train_df)
        if args.max_eval_rows is None and args.eval_sample_rate is None:
            log("loading full hourly evaluation data")
        else:
            log("loading capped/sampled hourly evaluation data")
        eval_df = load_eval_data(conn, args.max_eval_rows, args.eval_sample_rate, args.sample_mode)
        if eval_df.empty:
            raise RuntimeError("No eval rows found. The local trainer requires the eval split for final future evaluation.")
        eval_df = add_observed_demand_target(eval_df)

        fit_train_df, calibration_df = split_train_calibration(raw_train_df, args)
        if calibration_df.empty:
            log(f"training rows from train split: {len(fit_train_df):,}; calibration disabled")
        else:
            log(f"training rows from train split: {len(fit_train_df):,}")
            log(f"train-only calibration holdout rows: {len(calibration_df):,}")

        train_df, calibration_df, eval_df, feature_columns, train_reference_df = prepare_features(fit_train_df, calibration_df, eval_df, args)
        log(f"model fit rows: {len(train_df):,}")
        log(f"final eval rows from eval split: {len(eval_df):,}")
        log(f"feature columns: {len(feature_columns):,}")

        log(f"training {args.model_type} demand model from non-stockout observations")
        model = train_model(args, train_df, feature_columns)

        calibration_uncalibrated_metrics: dict[str, float] = {}
        calibration_metrics: dict[str, float] = {}
        if not calibration_df.empty:
            log("evaluating uncalibrated model on train-only calibration holdout")
            calibration_uncalibrated_metrics = evaluate_model(model, calibration_df, feature_columns)

        log("evaluating uncalibrated model on full eval split")
        uncalibrated_metrics = evaluate_model(model, eval_df, feature_columns)
        calibration_factor = 1.0
        raw_calibration_factor = 1.0
        wmape_calibration_factor = 1.0
        if not args.disable_calibration and not calibration_df.empty:
            log(f"computing train-only calibration factor with {args.calibration_objective} objective")
            calibration_factor, raw_calibration_factor, wmape_calibration_factor = choose_calibration_factor(args, model, calibration_df, feature_columns)
            log(f"raw_bias_calibration_factor: {raw_calibration_factor:.6f}")
            log(f"raw_wmape_calibration_factor: {wmape_calibration_factor:.6f}")
            if calibration_factor != raw_calibration_factor and args.calibration_objective == "bias":
                log(f"capped calibration_factor from {raw_calibration_factor:.6f} to {calibration_factor:.6f}")
            elif args.calibration_objective != "bias":
                log(f"selected calibration_factor: {calibration_factor:.6f}")
        if not calibration_df.empty:
            log("evaluating calibrated model on train-only calibration holdout")
            calibration_metrics = evaluate_model(model, calibration_df, feature_columns, calibration_factor)

        log("evaluating final calibrated model on full eval split")
        metrics = evaluate_model(model, eval_df, feature_columns, calibration_factor)
        segments = evaluate_segments(model, eval_df, feature_columns, calibration_factor)
        if calibration_uncalibrated_metrics:
            log("train-only calibration holdout uncalibrated metrics")
            for name, value in calibration_uncalibrated_metrics.items():
                log(f"calibration_uncalibrated_{name}: {value:.6f}")
        if calibration_metrics:
            log("train-only calibration holdout calibrated metrics")
            for name, value in calibration_metrics.items():
                log(f"calibration_{name}: {value:.6f}")
        log("uncalibrated metrics")
        for name, value in uncalibrated_metrics.items():
            log(f"uncalibrated_{name}: {value:.6f}")
        log(f"calibration_factor: {calibration_factor:.6f}")
        log("final metrics")
        for name, value in metrics.items():
            log(f"{name}: {value:.6f}")

        metrics["calibration_factor"] = float(calibration_factor)
        metrics["raw_calibration_factor"] = float(raw_calibration_factor)
        metrics["wmape_calibration_factor"] = float(wmape_calibration_factor)
        metrics["calibration_objective"] = args.calibration_objective
        metrics["calibration_source_split"] = "train" if not calibration_df.empty else "disabled"
        metrics["evaluation_source_split"] = "eval"
        metrics["eval_row_policy"] = "full" if args.max_eval_rows is None and args.eval_sample_rate is None else "capped_or_sampled"
        metrics["train_fit_rows"] = float(len(train_df))
        metrics["train_calibration_rows"] = float(len(calibration_df))
        metrics["uncalibrated"] = uncalibrated_metrics
        metrics["calibration_holdout"] = calibration_metrics
        metrics["calibration_holdout_uncalibrated"] = calibration_uncalibrated_metrics
        metrics["segments"] = segments

        model_path, metrics_path = save_artifacts(args, model, metrics, model_version, feature_columns, calibration_factor)
        log(f"saved model: {model_path}")
        log(f"saved metrics: {metrics_path}")

        model_key = upsert_model_metadata(conn, args.model_name, model_version, train_df)
        upsert_model_evaluation(conn, model_key, metrics, uncalibrated_metrics, calibration_metrics, calibration_uncalibrated_metrics, segments, calibration_factor)
        log(f"model_key: {model_key}")

        if args.load_predictions:
            log("loading model predictions into warehouse")
            load_predictions(conn, args, model, model_key, args.max_predict_rows, feature_columns, calibration_factor, train_reference_df)
            load_recommendations(conn, model_key)
            print_counts(conn, model_key)
        else:
            log("skipping warehouse prediction load; pass --load-predictions to populate DSS model facts")


if __name__ == "__main__":
    main()
