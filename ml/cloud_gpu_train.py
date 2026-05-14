#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional cloud dependency
    CatBoostRegressor = None


FEATURE_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a FreshRetail DSS demand model on Kaggle or Colab and export predictions.")
    parser.add_argument("--features", required=True, help="Parquet exported by ml/export_model_features.py.")
    parser.add_argument("--output-dir", default="cloud_outputs")
    parser.add_argument("--model-type", choices=["xgboost", "catboost"], default="xgboost")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model-name", default="cloud_xgboost_gpu")
    parser.add_argument("--model-version", default=None, help="Default: UTC timestamp.")
    parser.add_argument("--n-estimators", type=int, default=600, help="Boosting rounds/trees. Do not set this to 100k.")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional cap after filtering non-stockout train rows.")
    parser.add_argument("--max-eval-rows", type=int, default=None, help="Optional cap after filtering non-stockout eval rows.")
    parser.add_argument("--max-predict-rows", type=int, default=None, help="Optional cap for smoke tests only.")
    parser.add_argument("--prediction-batch-size", type=int, default=500_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURE_COLUMNS].copy()
    for column in ["is_business_hour_6_22", "is_weekend"]:
        x[column] = x[column].astype(int)
    for column in FEATURE_COLUMNS:
        x[column] = pd.to_numeric(x[column], errors="coerce").fillna(0)
    return x


def sample_if_needed(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is not None and len(df) > max_rows:
        return df.sample(max_rows, random_state=random_state).reset_index(drop=True)
    return df.reset_index(drop=True)


def train_model(args: argparse.Namespace, train_df: pd.DataFrame):
    x_train = feature_matrix(train_df)
    y_train = pd.to_numeric(train_df["target_observed_sales_amount"], errors="coerce").fillna(train_df["observed_sales_amount"]).astype(float)

    if args.model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            objective="reg:squarederror",
            tree_method="hist",
            device=args.device,
            max_bin=args.max_bin,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=args.random_state,
            n_jobs=-1,
        )
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


def evaluate_model(model: Any, eval_df: pd.DataFrame) -> dict[str, float]:
    x_eval = feature_matrix(eval_df)
    y_eval = pd.to_numeric(eval_df["target_observed_sales_amount"], errors="coerce").fillna(eval_df["observed_sales_amount"]).astype(float).to_numpy()
    y_pred = np.clip(model.predict(x_eval), 0, None)
    denominator = np.sum(np.abs(y_eval))
    return {
        "rows": float(len(eval_df)),
        "mae": float(mean_absolute_error(y_eval, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_eval, y_pred))),
        "wmape": float(np.sum(np.abs(y_eval - y_pred)) / denominator) if denominator > 0 else 0.0,
        "bias": float(np.sum(y_pred - y_eval) / denominator) if denominator > 0 else 0.0,
    }


def prediction_frame(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    predictions = np.clip(model.predict(feature_matrix(df)), 0, None)
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


def write_predictions(model: Any, features: pd.DataFrame, output_path: Path, batch_size: int) -> int:
    if output_path.exists():
        output_path.unlink()
    total_rows = 0
    writer = None
    try:
        for start in range(0, len(features), batch_size):
            batch = features.iloc[start : start + batch_size]
            predictions = prediction_frame(model, batch)
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

    print(f"loading features from {args.features}")
    features = pd.read_parquet(args.features)
    if args.max_predict_rows is not None:
        features = features.sort_values(["date_key", "source_split", "store_key", "product_key", "time_key"]).head(args.max_predict_rows).reset_index(drop=True)

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
    print(f"training {args.model_type} on {args.device}")
    model = train_model(args, train_df)

    metrics = evaluate_model(model, eval_df)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    prediction_path = output_dir / f"{args.model_name}_{model_version}_predictions.parquet"
    metrics_path = output_dir / f"{args.model_name}_{model_version}_metrics.json"
    metadata_path = output_dir / f"{args.model_name}_{model_version}_metadata.json"

    total_predictions = write_predictions(model, features, prediction_path, args.prediction_batch_size)
    metrics["prediction_rows"] = float(total_predictions)

    metadata = {
        "model_name": args.model_name,
        "model_version": model_version,
        "model_type": args.model_type,
        "device": args.device,
        "feature_columns": FEATURE_COLUMNS,
        "target_definition": "observed hourly sales on non-stockout rows only",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features_file": str(args.features),
        "predictions_file": str(prediction_path),
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote predictions: {prediction_path}")
    print(f"wrote metrics: {metrics_path}")
    print(f"wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
