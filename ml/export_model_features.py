#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psycopg


FEATURE_SCHEMA = pa.schema(
    [
        pa.field("date_key", pa.int64()),
        pa.field("full_date", pa.date32()),
        pa.field("time_key", pa.int16()),
        pa.field("hour_of_day", pa.int16()),
        pa.field("is_business_hour_6_22", pa.bool_()),
        pa.field("store_key", pa.int64()),
        pa.field("store_id", pa.int64()),
        pa.field("city_id", pa.int64()),
        pa.field("product_key", pa.int64()),
        pa.field("product_id", pa.int64()),
        pa.field("management_group_id", pa.int64()),
        pa.field("first_category_id", pa.int64()),
        pa.field("second_category_id", pa.int64()),
        pa.field("third_category_id", pa.int64()),
        pa.field("source_split", pa.string()),
        pa.field("observed_sales_amount", pa.float64()),
        pa.field("stockout_flag", pa.bool_()),
        pa.field("is_censored_observation", pa.bool_()),
        pa.field("is_trainable_demand_observation", pa.bool_()),
        pa.field("target_observed_sales_amount", pa.float64()),
        pa.field("discount_rate", pa.float64()),
        pa.field("activity_flag", pa.int16()),
        pa.field("precpt", pa.float64()),
        pa.field("avg_temperature", pa.float64()),
        pa.field("avg_humidity", pa.float64()),
        pa.field("avg_wind_level", pa.float64()),
        pa.field("day_of_week", pa.int16()),
        pa.field("is_weekend", pa.bool_()),
        pa.field("holiday_flag", pa.int16()),
    ]
)

FEATURE_COLUMNS = [field.name for field in FEATURE_SCHEMA]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export warehouse model features to parquet for Kaggle or Colab training.")
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5433")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE", "fresh_retail_dw"))
    parser.add_argument("--user", default=os.getenv("PGUSER", "warehouse"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", "warehouse"))
    parser.add_argument("--output", default="exports/model_features.parquet")
    parser.add_argument("--source-split", choices=["train", "eval"], default=None)
    parser.add_argument("--start-date", default=None, help="Inclusive YYYY-MM-DD filter on full_date.")
    parser.add_argument("--end-date", default=None, help="Inclusive YYYY-MM-DD filter on full_date.")
    parser.add_argument("--first-category-id", type=int, default=None)
    parser.add_argument("--store-id", type=int, default=None)
    parser.add_argument("--product-id", type=int, default=None)
    parser.add_argument("--trainable-only", action="store_true", help="Export only non-stockout rows. Do not use when you need full prediction output.")
    parser.add_argument("--sample-rate", type=float, default=None, help="Randomly export an approximate fraction of rows, e.g. 0.25. Uses PostgreSQL random().")
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--no-order", action="store_true", help="Skip ORDER BY for faster large exports. Recommended with date filters or sample-rate.")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "none"])
    return parser.parse_args()


def connect(args: argparse.Namespace) -> psycopg.Connection:
    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )


def build_query(args: argparse.Namespace) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    params: list[Any] = []

    if args.source_split:
        clauses.append("source_split = %s")
        params.append(args.source_split)
    if args.start_date:
        clauses.append("full_date >= %s")
        params.append(args.start_date)
    if args.end_date:
        clauses.append("full_date <= %s")
        params.append(args.end_date)
    if args.first_category_id is not None:
        clauses.append("first_category_id = %s")
        params.append(args.first_category_id)
    if args.store_id is not None:
        clauses.append("store_id = %s")
        params.append(args.store_id)
    if args.product_id is not None:
        clauses.append("product_id = %s")
        params.append(args.product_id)
    if args.trainable_only:
        clauses.append("is_trainable_demand_observation")
    if args.sample_rate is not None:
        if args.sample_rate <= 0 or args.sample_rate > 1:
            raise ValueError("--sample-rate must be in the range (0, 1].")
        clauses.append("random() < %s")
        params.append(args.sample_rate)

    where_sql = ""
    if clauses:
        where_sql = "WHERE " + " AND ".join(clauses)

    limit_sql = ""
    if args.limit_rows is not None:
        limit_sql = "LIMIT %s"
        params.append(args.limit_rows)

    order_sql = ""
    if not args.no_order:
        order_sql = "ORDER BY date_key, source_split, store_key, product_key, time_key"

    column_sql = ", ".join(FEATURE_COLUMNS)
    sql = f"""
        SELECT {column_sql}
        FROM dw.v_model_training_features_hourly
        {where_sql}
        {order_sql}
        {limit_sql}
    """
    return sql, tuple(params)


def normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["full_date"] = pd.to_datetime(result["full_date"]).dt.date
    for field in FEATURE_SCHEMA:
        if pa.types.is_integer(field.type):
            result[field.name] = pd.to_numeric(result[field.name], errors="coerce").fillna(0).astype("int64")
        elif pa.types.is_floating(field.type):
            result[field.name] = pd.to_numeric(result[field.name], errors="coerce")
        elif pa.types.is_boolean(field.type):
            result[field.name] = result[field.name].astype(bool)
    return result[FEATURE_COLUMNS]


def export_features(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    compression = None if args.compression == "none" else args.compression
    sql, params = build_query(args)
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    try:
        with connect(args) as conn:
            with conn.cursor(name="model_feature_export") as cur:
                cur.itersize = args.chunk_size
                cur.execute(sql, params)
                while True:
                    rows = cur.fetchmany(args.chunk_size)
                    if not rows:
                        break
                    columns = [desc.name for desc in cur.description]
                    df = normalize_chunk(pd.DataFrame(rows, columns=columns))
                    table = pa.Table.from_pandas(df, schema=FEATURE_SCHEMA, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(output, FEATURE_SCHEMA, compression=compression)
                    writer.write_table(table)
                    total_rows += len(df)
                    print(f"exported {total_rows:,} rows")
    finally:
        if writer is not None:
            writer.close()

    print(f"wrote {total_rows:,} rows to {output}")
    if total_rows == 0:
        raise RuntimeError("No feature rows were exported. Check that hourly facts are loaded.")


def main() -> None:
    args = parse_args()
    export_features(args)


if __name__ == "__main__":
    main()
