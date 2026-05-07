#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None


NUM_COLS = [
    "discount",
    "stock_on_hand",
    "selling_price",
    "product_price",
    "store_traffic_lag_1h",
    "category_momentum",
    "days_since_last_sale",
]

CAT_COLS = [
    "product_id",
    "location",
    "store_id",
    "product_category",
    "product_subcategory",
    "holiday",
    "is_promotion",
    "day_of_week",
    "week_of_year",
    "month",
    "year",
    "hour",
    "stockout_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate the RetailForecast XGBoost training/evaluation flow on FreshRetailNet parquet data."
    )
    parser.add_argument("--train-path", default="FreshRetailNet-50K/data/train.parquet")
    parser.add_argument("--eval-path", default="FreshRetailNet-50K/data/eval.parquet")
    parser.add_argument("--train-daily-rows", type=int, default=100_000)
    parser.add_argument("--eval-daily-rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--target-mode", choices=["observed", "latent_baseline"], default="latent_baseline")
    parser.add_argument("--model-type", choices=["xgboost", "lightgbm", "catboost"], default="xgboost")
    parser.add_argument("--artifact-dir", default="models")
    parser.add_argument("--model-version", default=None)
    return parser.parse_args()


def load_daily_sample(path: Path, max_daily_rows: int | None, seed: int) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    if max_daily_rows is None:
        return parquet_file.read().to_pandas()

    rng = np.random.default_rng(seed)
    row_groups = np.arange(parquet_file.metadata.num_row_groups)
    rng.shuffle(row_groups)

    frames: list[pd.DataFrame] = []
    loaded_rows = 0
    for row_group in row_groups:
        table = parquet_file.read_row_group(int(row_group))
        df = table.to_pandas()
        frames.append(df)
        loaded_rows += len(df)
        if loaded_rows >= max_daily_rows:
            break

    sampled = pd.concat(frames, ignore_index=True)
    if len(sampled) > max_daily_rows:
        sampled = sampled.sample(n=max_daily_rows, random_state=seed).reset_index(drop=True)
    return sampled


def to_retailforecast_hourly(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    for column in data.columns:
        if data[column].dtype == "float64":
            data[column] = data[column].astype("float32")
        elif data[column].dtype == "int64":
            data[column] = data[column].astype("int32")

    data["hour"] = [list(range(24))] * len(data)
    data = data.explode(["hours_sale", "hours_stock_status", "hour"], ignore_index=True)

    data["hours_sale"] = pd.to_numeric(data["hours_sale"], errors="coerce").fillna(0).astype("float32")
    data["hours_stock_status"] = pd.to_numeric(data["hours_stock_status"], errors="coerce").fillna(0).astype("int8")
    data["hour"] = data["hour"].astype("int8")

    data = data.rename(
        columns={
            "hours_sale": "units_ordered",
            "hours_stock_status": "stockout_flag",
            "first_category_id": "product_category",
            "second_category_id": "product_subcategory",
            "activity_flag": "is_promotion",
            "city_id": "location",
            "holiday_flag": "holiday",
        }
    )

    data["date"] = pd.to_datetime(data["dt"]) + pd.to_timedelta(data["hour"].astype(int), unit="h")
    data["day_of_week"] = data["date"].dt.dayofweek.astype("int8")
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype("int16")
    data["month"] = data["date"].dt.month.astype("int8")
    data["year"] = data["date"].dt.year.astype("int16")

    # FreshRetailNet does not provide these RetailForecast schema fields.
    # Use zero-filled numeric placeholders so the pipeline shape remains stable.
    for column in ["stock_on_hand", "selling_price", "product_price"]:
        data[column] = 0.0
    for column in ["promotion_type", "is_bulk_order"]:
        data[column] = "missing"

    drop_cols = [
        "dt",
        "sale_amount",
        "stock_hour6_22_cnt",
        "management_group_id",
        "third_category_id",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
    ]
    data = data.drop(columns=[column for column in drop_cols if column in data.columns])
    return data


def add_days_since_last_sale(
    df: pd.DataFrame,
    date_col: str = "date",
    group_cols: list[str] | None = None,
    sales_col: str = "units_ordered",
) -> pd.DataFrame:
    group_cols = group_cols or ["store_id", "product_id"]
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(group_cols + [date_col])

    sale_mask = data[sales_col] > 0
    data.loc[sale_mask, "last_sale_date"] = data.loc[sale_mask, date_col]
    data["last_sale_date"] = data.groupby(group_cols)["last_sale_date"].ffill()
    data["days_since_last_sale"] = (data[date_col] - data["last_sale_date"]).dt.total_seconds() / 86400
    data["days_since_last_sale"] = data["days_since_last_sale"].fillna(-1)
    return data.drop(columns=["last_sale_date"])


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.sort_values(by=["date"]).reset_index(drop=True)

    hourly_store_sales = data.groupby(["store_id", "date"], observed=True)["units_ordered"].sum().reset_index()
    hourly_store_sales = hourly_store_sales.rename(columns={"units_ordered": "store_total_current"})
    hourly_store_sales["store_traffic_lag_1h"] = hourly_store_sales.groupby("store_id")["store_total_current"].shift(1)
    hourly_store_sales = hourly_store_sales.drop(columns=["store_total_current"])
    data = data.merge(hourly_store_sales, on=["store_id", "date"], how="left")

    hourly_cat_sales = data.groupby(["product_category", "date"], observed=True)["units_ordered"].sum().reset_index()
    hourly_cat_sales = hourly_cat_sales.rename(columns={"units_ordered": "cat_total_current"})
    hourly_cat_sales["category_momentum"] = (
        hourly_cat_sales.groupby("product_category")["cat_total_current"]
        .transform(lambda values: values.shift(1).rolling(window=24, min_periods=1).sum())
    )
    hourly_cat_sales = hourly_cat_sales.drop(columns=["cat_total_current"])
    data = data.merge(hourly_cat_sales, on=["product_category", "date"], how="left")

    data["store_traffic_lag_1h"] = data["store_traffic_lag_1h"].fillna(0)
    data["category_momentum"] = data["category_momentum"].fillna(0)
    return data


def baseline_tables(train_df: pd.DataFrame) -> dict[str, object]:
    available = train_df.loc[train_df["stockout_flag"] == 0].copy()
    if available.empty:
        return {
            "store_product_hour": pd.Series(dtype="float32"),
            "product_hour": pd.Series(dtype="float32"),
            "product": pd.Series(dtype="float32"),
            "global": 0.0,
        }

    return {
        "store_product_hour": available.groupby(["store_id", "product_id", "hour"], observed=True)["units_ordered"].mean(),
        "product_hour": available.groupby(["product_id", "hour"], observed=True)["units_ordered"].mean(),
        "product": available.groupby("product_id", observed=True)["units_ordered"].mean(),
        "global": float(available["units_ordered"].mean()),
    }


def add_latent_baseline_target(df: pd.DataFrame, baselines: dict[str, object]) -> pd.DataFrame:
    data = df.copy()
    data["true_demand"] = data["units_ordered"].astype("float32")

    store_product_hour = baselines["store_product_hour"]
    product_hour = baselines["product_hour"]
    product = baselines["product"]
    global_value = float(baselines["global"])

    data = data.join(store_product_hour.rename("store_product_hour_baseline"), on=["store_id", "product_id", "hour"])
    data = data.join(product_hour.rename("product_hour_baseline"), on=["product_id", "hour"])
    data = data.join(product.rename("product_baseline"), on="product_id")
    baseline = (
        data["store_product_hour_baseline"]
        .fillna(data["product_hour_baseline"])
        .fillna(data["product_baseline"])
        .fillna(global_value)
    )

    stockout_mask = data["stockout_flag"] == 1
    data.loc[stockout_mask, "true_demand"] = np.maximum(
        data.loc[stockout_mask, "units_ordered"].astype(float),
        baseline.loc[stockout_mask].astype(float),
    )
    data["true_demand"] = data["true_demand"].clip(lower=0)
    return data.drop(columns=["store_product_hour_baseline", "product_hour_baseline", "product_baseline"])


def prepare_features(train_df: pd.DataFrame, eval_df: pd.DataFrame, target_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    train = add_context_features(add_days_since_last_sale(train_df))
    eval_data = add_context_features(add_days_since_last_sale(eval_df))

    if target_mode == "latent_baseline":
        baselines = baseline_tables(train)
        train = add_latent_baseline_target(train, baselines)
        eval_data = add_latent_baseline_target(eval_data, baselines)
        target = "true_demand"
    else:
        target = "units_ordered"

    return train, eval_data, target


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[NUM_COLS + CAT_COLS].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    for column in NUM_COLS:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    for column in CAT_COLS:
        features[column] = features[column].astype(str).replace(["<NA>", "nan", "None"], "missing")
    return features


def build_model(args: argparse.Namespace):
    if args.model_type == "xgboost":
        return XGBRegressor(
            objective="count:poisson",
            eval_metric="poisson-nloglik",
            learning_rate=args.learning_rate,
            max_delta_step=1,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            tree_method="hist",
            n_jobs=4,
        )
    if args.model_type == "lightgbm":
        if LGBMRegressor is None:
            raise RuntimeError("lightgbm is not installed")
        return LGBMRegressor(
            objective="tweedie",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=4,
        )
    if args.model_type == "catboost":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed")
        return CatBoostRegressor(
            loss_function="Tweedie:variance_power=1.5",
            iterations=args.n_estimators,
            learning_rate=args.learning_rate,
            depth=args.max_depth,
            random_seed=42,
            verbose=False,
        )
    raise ValueError(f"Unsupported model type: {args.model_type}")


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-2)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ]
    )
    model = build_model(args)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0, None)
    epsilon = 1e-10
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    wmape = float(np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + epsilon))
    bias = float(((np.sum(y_pred) - np.sum(y_true)) / (np.sum(y_true) + epsilon)) * 100)
    baseline_preds = np.full_like(y_true, fill_value=float(np.mean(y_true)))
    baseline_mae = mean_absolute_error(y_true, baseline_preds)
    improvement = float(((baseline_mae - mae) / baseline_mae) * 100) if baseline_mae else 0.0
    return {
        "mae": float(mae),
        "rmse": rmse,
        "wmape": wmape,
        "wmape_percent": wmape * 100,
        "bias_percent": bias,
        "baseline_mae": float(baseline_mae),
        "baseline_mae_improvement_percent": improvement,
    }


def save_outputs(
    pipeline: Pipeline,
    metrics: dict[str, float],
    args: argparse.Namespace,
    train_rows: int,
    eval_rows: int,
    version: str,
) -> tuple[Path, Path]:
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / f"retailforecast_{args.model_type}_replication_{version}.pkl"
    metrics_path = artifact_dir / f"retailforecast_{args.model_type}_replication_{version}_metrics.json"

    artifact = {
        "pipeline": pipeline,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "target_mode": args.target_mode,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)

    payload = {
        "model": f"RetailForecast-style {args.model_type}",
        "target_mode": args.target_mode,
        "train_hourly_rows": train_rows,
        "eval_hourly_rows": eval_rows,
        "train_daily_rows_requested": args.train_daily_rows,
        "eval_daily_rows_requested": args.eval_daily_rows,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        **metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return model_path, metrics_path


def main() -> None:
    args = parse_args()
    version = args.model_version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    print("loading sampled daily parquet rows")
    train_daily = load_daily_sample(Path(args.train_path), args.train_daily_rows, args.seed)
    eval_daily = load_daily_sample(Path(args.eval_path), args.eval_daily_rows, args.seed + 1)
    print(f"train daily rows: {len(train_daily):,}")
    print(f"eval daily rows: {len(eval_daily):,}")

    print("converting to RetailForecast hourly schema")
    train_hourly = to_retailforecast_hourly(train_daily)
    eval_hourly = to_retailforecast_hourly(eval_daily)
    print(f"train hourly rows: {len(train_hourly):,}")
    print(f"eval hourly rows: {len(eval_hourly):,}")

    print(f"preparing features and target: {args.target_mode}")
    train_ready, eval_ready, target = prepare_features(train_hourly, eval_hourly, args.target_mode)

    x_train = clean_features(train_ready)
    y_train = train_ready[target].astype(float).clip(lower=0).to_numpy()
    x_eval = clean_features(eval_ready)
    y_eval = eval_ready[target].astype(float).clip(lower=0).to_numpy()

    print(f"training RetailForecast-style {args.model_type}")
    pipeline = build_pipeline(args)
    pipeline.fit(x_train, y_train)

    print("evaluating")
    predictions = pipeline.predict(x_eval)
    metrics = evaluate(y_eval, predictions)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    model_path, metrics_path = save_outputs(pipeline, metrics, args, len(train_ready), len(eval_ready), version)
    print(f"saved model: {model_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
