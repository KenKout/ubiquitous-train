#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psycopg
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional cloud dependency
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
]

PREDICTION_COLUMNS = [
    "date_key",
    "time_key",
    "store_key",
    "product_key",
    "source_split",
    "observed_sales_amount",
    "stockout_flag",
    "predicted_demand",
    "estimated_true_demand",
    "estimated_lost_sales",
    "prediction_lower_bound",
    "prediction_upper_bound",
]

WAREHOUSE_COLUMNS = [
    "date_key",
    "full_date",
    "time_key",
    "hour_of_day",
    "is_business_hour_6_22",
    "store_key",
    "store_id",
    "city_id",
    "product_key",
    "product_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "source_split",
    "observed_sales_amount",
    "stockout_flag",
    "is_censored_observation",
    "is_trainable_demand_observation",
    "target_observed_sales_amount",
    "discount_rate",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "day_of_week",
    "is_weekend",
    "holiday_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a FreshRetail DSS demand model on Kaggle or Colab and export predictions.")
    parser.add_argument("--features", default=None, help="Parquet exported by ml/export_model_features.py. If omitted, reads directly from PostgreSQL.")
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5433")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE", "fresh_retail_dw"))
    parser.add_argument("--user", default=os.getenv("PGUSER", "warehouse"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", "warehouse"))
    parser.add_argument("--warehouse-source-split", choices=["train", "eval"], default=None)
    parser.add_argument("--warehouse-start-date", default=None, help="Inclusive YYYY-MM-DD filter when reading directly from PostgreSQL.")
    parser.add_argument("--warehouse-end-date", default=None, help="Inclusive YYYY-MM-DD filter when reading directly from PostgreSQL.")
    parser.add_argument("--warehouse-first-category-id", type=int, default=None)
    parser.add_argument("--warehouse-store-id", type=int, default=None)
    parser.add_argument("--warehouse-product-id", type=int, default=None)
    parser.add_argument("--warehouse-sample-rate", type=float, default=None, help="Approximate random sample fraction when reading directly from PostgreSQL.")
    parser.add_argument("--warehouse-limit-rows", type=int, default=None)
    parser.add_argument("--warehouse-no-order", action="store_true", help="Skip ORDER BY for faster direct PostgreSQL reads.")
    parser.add_argument("--warehouse-chunk-size", type=int, default=250_000)
    parser.add_argument("--output-dir", default="cloud_outputs")
    parser.add_argument("--model-type", choices=["xgboost", "catboost"], default="xgboost")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model-name", default="cloud_xgboost_gpu")
    parser.add_argument("--model-version", default=None, help="Default: UTC timestamp.")
    parser.add_argument("--n-estimators", type=int, default=600, help="Boosting rounds/trees. Do not set this to 100k.")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--xgboost-objective", choices=["reg:tweedie", "reg:squarederror", "count:poisson"], default="reg:tweedie")
    parser.add_argument("--tweedie-variance-power", type=float, default=1.4)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional cap after filtering non-stockout train rows.")
    parser.add_argument("--max-eval-rows", type=int, default=None, help="Optional cap after filtering non-stockout eval rows.")
    parser.add_argument("--max-predict-rows", type=int, default=None, help="Optional cap for smoke tests only.")
    parser.add_argument("--prediction-batch-size", type=int, default=500_000)
    parser.add_argument("--disable-advanced-features", action="store_true", help="Use only raw warehouse features, without historical demand priors.")
    parser.add_argument("--disable-eval-calibration", action="store_true", help="Do not scale predictions to remove aggregate eval-set bias.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def connect(args: argparse.Namespace) -> psycopg.Connection:
    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )


def build_warehouse_query(args: argparse.Namespace) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    params: list[Any] = []
    if args.warehouse_source_split:
        clauses.append("source_split = %s")
        params.append(args.warehouse_source_split)
    if args.warehouse_start_date:
        clauses.append("full_date >= %s")
        params.append(args.warehouse_start_date)
    if args.warehouse_end_date:
        clauses.append("full_date <= %s")
        params.append(args.warehouse_end_date)
    if args.warehouse_first_category_id is not None:
        clauses.append("first_category_id = %s")
        params.append(args.warehouse_first_category_id)
    if args.warehouse_store_id is not None:
        clauses.append("store_id = %s")
        params.append(args.warehouse_store_id)
    if args.warehouse_product_id is not None:
        clauses.append("product_id = %s")
        params.append(args.warehouse_product_id)
    if args.warehouse_sample_rate is not None:
        if args.warehouse_sample_rate <= 0 or args.warehouse_sample_rate > 1:
            raise ValueError("--warehouse-sample-rate must be in the range (0, 1].")
        clauses.append("random() < %s")
        params.append(args.warehouse_sample_rate)

    where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
    order_sql = "" if args.warehouse_no_order else "ORDER BY date_key, source_split, store_key, product_key, time_key"
    limit_sql = ""
    if args.warehouse_limit_rows is not None:
        limit_sql = "LIMIT %s"
        params.append(args.warehouse_limit_rows)

    return (
        f"""
        SELECT {", ".join(WAREHOUSE_COLUMNS)}
        FROM dw.v_model_training_features_hourly
        {where_sql}
        {order_sql}
        {limit_sql}
        """,
        tuple(params),
    )


def load_features_from_warehouse(args: argparse.Namespace) -> pd.DataFrame:
    sql, params = build_warehouse_query(args)
    chunks: list[pd.DataFrame] = []
    total_rows = 0
    with connect(args) as conn:
        with conn.cursor(name="cloud_training_features") as cur:
            cur.itersize = args.warehouse_chunk_size
            cur.execute(sql, params)
            while True:
                rows = cur.fetchmany(args.warehouse_chunk_size)
                if not rows:
                    break
                columns = [desc.name for desc in cur.description]
                chunk = pd.DataFrame(rows, columns=columns)
                chunks.append(chunk)
                total_rows += len(chunk)
                print(f"loaded {total_rows:,} feature rows from warehouse")
    if not chunks:
        raise RuntimeError("No warehouse feature rows found. Check hourly facts and filters.")
    return pd.concat(chunks, ignore_index=True)


def load_features(args: argparse.Namespace) -> pd.DataFrame:
    if args.features:
        print(f"loading features from {args.features}")
        return pd.read_parquet(args.features)
    print("loading features directly from PostgreSQL warehouse")
    return load_features_from_warehouse(args)


def feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = df[feature_columns].copy()
    for column in ["is_business_hour_6_22", "is_weekend"]:
        if column in x.columns:
            x[column] = x[column].astype(int)
    for column in feature_columns:
        x[column] = pd.to_numeric(x[column], errors="coerce").fillna(0)
    return x


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


def add_advanced_features(features: pd.DataFrame) -> pd.DataFrame:
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

    non_stockout_train = result[(result["source_split"] == "train") & (result["is_trainable_demand_observation"].astype(bool))]
    train_all = result[result["source_split"] == "train"]
    global_mean = float(non_stockout_train["observed_sales_amount"].mean()) if not non_stockout_train.empty else 0.0

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
    for keys, mean_column, count_column in demand_groups:
        result = merge_mean_count(result, non_stockout_train, keys, mean_column, count_column)
        result[f"{count_column}_log1p"] = np.log1p(pd.to_numeric(result[count_column], errors="coerce").fillna(0))
        result = result.drop(columns=[count_column])

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
    for keys, rate_column in stockout_groups:
        result = merge_rate(result, train_all, keys, rate_column)

    for column in ADVANCED_FEATURE_COLUMNS:
        if column not in result.columns:
            result[column] = 0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
    return result


def sample_if_needed(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is not None and len(df) > max_rows:
        return df.sample(max_rows, random_state=random_state).reset_index(drop=True)
    return df.reset_index(drop=True)


def train_model(args: argparse.Namespace, train_df: pd.DataFrame, feature_columns: list[str]):
    x_train = feature_matrix(train_df, feature_columns)
    y_train = pd.to_numeric(train_df["target_observed_sales_amount"], errors="coerce").fillna(train_df["observed_sales_amount"]).astype(float)

    if args.model_type == "xgboost":
        xgboost_params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "objective": args.xgboost_objective,
            "tree_method": "hist",
            "device": args.device,
            "max_bin": args.max_bin,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": args.random_state,
            "n_jobs": -1,
        }
        if args.xgboost_objective == "reg:tweedie":
            xgboost_params["tweedie_variance_power"] = args.tweedie_variance_power
        model = XGBRegressor(**xgboost_params)
    elif args.model_type == "catboost":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed. On Kaggle/Colab, run `pip install catboost` first.")
        catboost_params = {
            "iterations": args.n_estimators,
            "depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "loss_function": "Tweedie:variance_power=1.5",
            "task_type": "GPU" if args.device == "cuda" else "CPU",
            "random_seed": args.random_state,
            "verbose": 100,
        }
        if args.device == "cuda":
            catboost_params["devices"] = "0"
        model = CatBoostRegressor(**catboost_params)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model.fit(x_train, y_train)
    return model


def demand_target(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["target_observed_sales_amount"], errors="coerce").fillna(df["observed_sales_amount"]).astype(float)


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
    x_eval = feature_matrix(eval_df, feature_columns)
    y_eval = demand_target(eval_df).to_numpy()
    y_pred = np.clip(model.predict(x_eval) * calibration_factor, 0, None)
    return evaluate_predictions(y_eval, y_pred)


def compute_calibration_factor(model: Any, eval_df: pd.DataFrame, feature_columns: list[str]) -> float:
    y_eval = demand_target(eval_df).to_numpy()
    y_pred = np.clip(model.predict(feature_matrix(eval_df, feature_columns)), 0, None)
    prediction_sum = float(np.sum(y_pred))
    if prediction_sum <= 0:
        return 1.0
    return float(np.sum(y_eval) / prediction_sum)


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


def prediction_frame(model: Any, df: pd.DataFrame, feature_columns: list[str], calibration_factor: float) -> pd.DataFrame:
    predictions = np.clip(model.predict(feature_matrix(df, feature_columns)) * calibration_factor, 0, None)
    observed = pd.to_numeric(df["observed_sales_amount"], errors="coerce").fillna(0).astype(float).to_numpy()
    stockout = df["stockout_flag"].astype(bool).to_numpy()
    estimated_true_demand = np.where(stockout, np.maximum(predictions, observed), observed)
    estimated_lost_sales = np.where(stockout, np.maximum(estimated_true_demand - observed, 0), 0)
    prediction_lower_bound = np.where(stockout, np.maximum(estimated_true_demand * 0.85, observed), observed)
    prediction_upper_bound = np.where(stockout, np.maximum(estimated_true_demand * 1.15, observed), observed)

    result = df[["date_key", "time_key", "store_key", "product_key", "source_split"]].copy()
    result["observed_sales_amount"] = observed
    result["stockout_flag"] = stockout
    result["predicted_demand"] = predictions
    result["estimated_true_demand"] = estimated_true_demand
    result["estimated_lost_sales"] = estimated_lost_sales
    result["prediction_lower_bound"] = prediction_lower_bound
    result["prediction_upper_bound"] = prediction_upper_bound
    return result[PREDICTION_COLUMNS]


def write_predictions(model: Any, features: pd.DataFrame, output_path: Path, batch_size: int, feature_columns: list[str], calibration_factor: float) -> int:
    if output_path.exists():
        output_path.unlink()
    total_rows = 0
    writer = None
    try:
        for start in range(0, len(features), batch_size):
            batch = features.iloc[start : start + batch_size]
            predictions = prediction_frame(model, batch, feature_columns, calibration_factor)
            arrow_table = pa.Table.from_pandas(predictions, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, arrow_table.schema, compression="zstd")
            writer.write_table(arrow_table)
            total_rows += len(predictions)
            print(f"predicted {total_rows:,} rows")
    finally:
        if writer is not None:
            writer.close()
    return total_rows


def main() -> None:
    args = parse_args()
    model_version = args.model_version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(args)
    if args.max_predict_rows is not None:
        features = features.sort_values(["date_key", "source_split", "store_key", "product_key", "time_key"]).head(args.max_predict_rows).reset_index(drop=True)

    feature_columns = BASE_FEATURE_COLUMNS.copy()
    if args.disable_advanced_features:
        print("advanced feature engineering disabled")
    else:
        print("adding train-only historical demand and stockout features")
        features = add_advanced_features(features)
        feature_columns += ADVANCED_FEATURE_COLUMNS

    train_df = features[(features["source_split"] == "train") & (features["is_trainable_demand_observation"].astype(bool))]
    eval_df = features[(features["source_split"] == "eval") & (features["is_trainable_demand_observation"].astype(bool))]
    train_df = sample_if_needed(train_df, args.max_train_rows, args.random_state)
    eval_df = sample_if_needed(eval_df, args.max_eval_rows, args.random_state)

    if train_df.empty:
        raise RuntimeError("No trainable rows found. Export full features, not `--trainable-only` eval data.")
    if eval_df.empty:
        print("no eval rows found; using a train sample for evaluation")
        eval_df = train_df.sample(min(len(train_df), 50_000), random_state=args.random_state).reset_index(drop=True)

    print(f"training rows: {len(train_df):,}")
    print(f"evaluation rows: {len(eval_df):,}")
    print(f"feature columns: {len(feature_columns):,}")
    print(f"training {args.model_type} on {args.device}")
    model = train_model(args, train_df, feature_columns)

    uncalibrated_metrics = evaluate_model(model, eval_df, feature_columns)
    calibration_factor = 1.0
    if not args.disable_eval_calibration:
        calibration_factor = compute_calibration_factor(model, eval_df, feature_columns)
    metrics = evaluate_model(model, eval_df, feature_columns, calibration_factor)
    segments = evaluate_segments(model, eval_df, feature_columns, calibration_factor)

    print("uncalibrated metrics")
    for key, value in uncalibrated_metrics.items():
        print(f"uncalibrated_{key}: {value:.6f}")
    print(f"calibration_factor: {calibration_factor:.6f}")
    print("final metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    prediction_path = output_dir / f"{args.model_name}_{model_version}_predictions.parquet"
    metrics_path = output_dir / f"{args.model_name}_{model_version}_metrics.json"
    metadata_path = output_dir / f"{args.model_name}_{model_version}_metadata.json"

    total_predictions = write_predictions(model, features, prediction_path, args.prediction_batch_size, feature_columns, calibration_factor)
    metrics["prediction_rows"] = float(total_predictions)
    metrics["calibration_factor"] = float(calibration_factor)
    metrics["uncalibrated"] = uncalibrated_metrics
    metrics["segments"] = segments

    metadata = {
        "model_name": args.model_name,
        "model_version": model_version,
        "model_type": args.model_type,
        "device": args.device,
        "xgboost_objective": args.xgboost_objective if args.model_type == "xgboost" else None,
        "tweedie_variance_power": args.tweedie_variance_power if args.model_type == "xgboost" and args.xgboost_objective == "reg:tweedie" else None,
        "feature_columns": feature_columns,
        "advanced_features_enabled": not args.disable_advanced_features,
        "target_definition": "observed hourly sales on non-stockout rows only",
        "calibration_factor": calibration_factor,
        "calibration_definition": "prediction scale factor = sum(eval observed demand) / sum(eval predicted demand) on non-stockout eval rows",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features_file": str(args.features) if args.features else "postgresql:dw.v_model_training_features_hourly",
        "predictions_file": str(prediction_path),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote predictions: {prediction_path}")
    print(f"wrote metrics: {metrics_path}")
    print(f"wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
