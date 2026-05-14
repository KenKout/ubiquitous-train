#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psycopg


STAGING_COLUMNS = [
    "source_split",
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
    "dt",
    "sale_amount",
    "hours_sale",
    "stock_hour6_22_cnt",
    "hours_stock_status",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_data_dir() -> str:
    candidates = [
        PROJECT_ROOT / "FreshRetailNet-50K" / "data",
        PROJECT_ROOT.parent / "FreshRetailNet-50K" / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(PROJECT_ROOT.parent / "FreshRetailNet-50K" / "data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load FreshRetailNet-50K into the PostgreSQL warehouse.")
    parser.add_argument("--data-dir", default=default_data_dir(), help="Directory containing train.parquet and eval.parquet.")
    parser.add_argument("--schema-file", default="sql/001_schema.sql", help="Warehouse DDL file to apply before loading.")
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5433")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE", "fresh_retail_dw"))
    parser.add_argument("--user", default=os.getenv("PGUSER", "warehouse"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", "warehouse"))
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--limit-rows-per-split", type=int, default=None, help="Legacy shortcut: cap both train and eval splits to the same row count.")
    parser.add_argument("--train-limit-rows", type=int, default=None, help="Cap train.parquet daily rows. Omit to load the full train split.")
    parser.add_argument("--eval-limit-rows", type=int, default=None, help="Cap eval.parquet daily rows. Omit to load the full eval split.")
    parser.add_argument(
        "--staging-sample-mode",
        choices=["store-product-panel", "product-date-stratified", "date-stratified", "even", "head"],
        default="product-date-stratified",
        help="How to select rows when a split is capped. store-product-panel loads full histories for seeded store-product pairs and matched eval rows.",
    )
    parser.add_argument("--panel-seed", type=int, default=42, help="Random seed for store-product-panel sampling.")
    parser.add_argument("--reset", action="store_true", help="Truncate staging and warehouse tables before loading.")
    parser.add_argument("--schema-only", action="store_true", help="Apply schema and exit.")
    parser.add_argument("--load-hourly", action="store_true", help="Populate hourly fact table. Full data expands to about 116M rows.")
    parser.add_argument("--skip-staging", action="store_true", help="Reuse existing staging rows and only build dimensions/facts.")
    parser.add_argument("--hourly-workers", type=int, default=1, help="Parallel worker processes for hourly fact expansion by date.")
    parser.add_argument("--hourly-start-date", default=None, help="Inclusive YYYY-MM-DD date filter for hourly fact expansion.")
    parser.add_argument("--hourly-end-date", default=None, help="Inclusive YYYY-MM-DD date filter for hourly fact expansion.")
    return parser.parse_args()


@dataclass(frozen=True)
class DbConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str


HOURLY_FACT_SQL = """
INSERT INTO dw.fact_sales_inventory_hourly (
    date_key,
    time_key,
    store_key,
    product_key,
    source_split,
    observed_sales_amount,
    stockout_flag,
    is_censored_observation,
    discount_rate,
    activity_flag,
    precpt,
    avg_temperature,
    avg_humidity,
    avg_wind_level
)
SELECT
    d.date_key,
    (hour_idx - 1)::SMALLINT AS time_key,
    store.store_key,
    product.product_key,
    s.source_split,
    s.hours_sale[hour_idx] AS observed_sales_amount,
    s.hours_stock_status[hour_idx] = 1 AS stockout_flag,
    s.hours_stock_status[hour_idx] = 1 AS is_censored_observation,
    s.discount,
    s.activity_flag,
    s.precpt,
    s.avg_temperature,
    s.avg_humidity,
    s.avg_wind_level
FROM staging.fresh_retail_observation_day s
JOIN dw.dim_date d ON d.full_date = s.dt
JOIN dw.dim_store store ON store.store_id = s.store_id
JOIN dw.dim_product product ON product.product_id = s.product_id
CROSS JOIN LATERAL GENERATE_SUBSCRIPTS(s.hours_sale, 1) AS hour_idx
WHERE s.dt = %s
ON CONFLICT (date_key, time_key, store_key, product_key) DO UPDATE SET
    source_split = EXCLUDED.source_split,
    observed_sales_amount = EXCLUDED.observed_sales_amount,
    stockout_flag = EXCLUDED.stockout_flag,
    is_censored_observation = EXCLUDED.is_censored_observation,
    discount_rate = EXCLUDED.discount_rate,
    activity_flag = EXCLUDED.activity_flag,
    precpt = EXCLUDED.precpt,
    avg_temperature = EXCLUDED.avg_temperature,
    avg_humidity = EXCLUDED.avg_humidity,
    avg_wind_level = EXCLUDED.avg_wind_level;
"""


def connect(args: argparse.Namespace) -> psycopg.Connection:
    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )


def db_config_from_args(args: argparse.Namespace) -> DbConfig:
    return DbConfig(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )


def connect_config(config: DbConfig) -> psycopg.Connection:
    return psycopg.connect(
        host=config.host,
        port=config.port,
        dbname=config.dbname,
        user=config.user,
        password=config.password,
    )


def apply_schema(conn: psycopg.Connection, schema_file: Path) -> None:
    with schema_file.open("r", encoding="utf-8") as file:
        sql = file.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def reset_tables(conn: psycopg.Connection) -> None:
    tables = [
        "dw.fact_replenishment_recommendation_daily",
        "dw.fact_demand_estimate_hourly",
        "dw.fact_model_evaluation",
        "dw.fact_sales_inventory_hourly",
        "dw.fact_sales_inventory_daily",
        "dw.dim_model",
        "dw.dim_product",
        "dw.dim_store",
        "dw.dim_city",
        "dw.dim_time",
        "dw.dim_date",
        "staging.fresh_retail_observation_day",
    ]
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE")
    conn.commit()


def as_list(value: object) -> list:
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)  # type: ignore[arg-type]


def pg_float_array(value: object) -> str:
    return "{" + ",".join(format(float(item), ".15g") for item in as_list(value)) + "}"


def pg_int_array(value: object) -> str:
    return "{" + ",".join(str(int(item)) for item in as_list(value)) + "}"


def copy_dataframe(conn: psycopg.Connection, table: str, columns: list[str], df: pd.DataFrame) -> None:
    buffer = StringIO()
    df.to_csv(
        buffer,
        index=False,
        header=False,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    buffer.seek(0)

    quoted_columns = ", ".join(columns)
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({quoted_columns}) FROM STDIN WITH (FORMAT CSV)") as copy:
            copy.write(buffer.getvalue())
    conn.commit()


def resolve_split_limit(args: argparse.Namespace, source_split: str) -> int | None:
    specific_limit = args.train_limit_rows if source_split == "train" else args.eval_limit_rows
    if specific_limit is not None:
        return specific_limit
    return args.limit_rows_per_split


def validate_row_limit(name: str, limit_rows: int | None) -> None:
    if limit_rows is not None and limit_rows <= 0:
        raise ValueError(f"{name} must be greater than zero when provided")


def even_sample_indices(total_rows: int, limit_rows: int) -> list[int] | None:
    if limit_rows >= total_rows:
        return None
    return evenly_spaced_ranks(total_rows, limit_rows).tolist()


def evenly_spaced_ranks(count: int, quota: int) -> np.ndarray:
    if quota >= count:
        return np.arange(count, dtype=np.int64)
    return np.floor((np.arange(quota, dtype=np.float64) + 0.5) * count / quota).astype(np.int64)


def proportional_group_quotas(group_counts: dict[object, int], limit_rows: int) -> dict[object, int]:
    groups = sorted(group_counts)
    if not groups:
        return {}

    if limit_rows >= sum(group_counts.values()):
        return dict(group_counts)

    quotas = {group: 0 for group in groups}
    remaining = limit_rows
    capacities = group_counts.copy()

    if limit_rows >= len(groups):
        for group in groups:
            quotas[group] = 1
            capacities[group] = max(group_counts[group] - 1, 0)
        remaining -= len(groups)

    total_capacity = sum(capacities.values())
    if remaining <= 0 or total_capacity <= 0:
        return quotas

    fractional_parts: list[tuple[float, object]] = []
    for group in groups:
        raw_quota = remaining * capacities[group] / total_capacity
        extra = min(int(np.floor(raw_quota)), capacities[group])
        quotas[group] += extra
        fractional_parts.append((raw_quota - extra, group))

    assigned = sum(quotas.values())
    leftover = limit_rows - assigned
    for _, group in sorted(fractional_parts, key=lambda item: (-item[0], item[1])):
        if leftover <= 0:
            break
        if quotas[group] < group_counts[group]:
            quotas[group] += 1
            leftover -= 1

    if leftover > 0:
        for group in groups:
            if leftover <= 0:
                break
            available = group_counts[group] - quotas[group]
            take = min(available, leftover)
            quotas[group] += take
            leftover -= take

    return quotas


def batch_group_keys(batch: pa.RecordBatch, columns: list[str]) -> list[object]:
    df = pa.Table.from_batches([batch]).select(columns).to_pandas()
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"]).dt.date
    if len(columns) == 1:
        return df[columns[0]].tolist()
    return list(df.itertuples(index=False, name=None))


def count_groups(path: Path, batch_size: int, columns: list[str]) -> dict[object, int]:
    parquet_file = pq.ParquetFile(path)
    counts: dict[object, int] = {}
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        for group in batch_group_keys(batch, columns):
            counts[group] = counts.get(group, 0) + 1
    return counts


def stratified_sample_indices(path: Path, batch_size: int, limit_rows: int, columns: list[str]) -> list[int] | None:
    parquet_file = pq.ParquetFile(path)
    total_rows = parquet_file.metadata.num_rows
    if limit_rows >= total_rows:
        return None

    group_counts = count_groups(path, batch_size, columns)
    quotas = proportional_group_quotas(group_counts, limit_rows)
    selected_ranks = {
        group: set(evenly_spaced_ranks(count, quotas[group]).tolist())
        for group, count in group_counts.items()
        if quotas.get(group, 0) > 0
    }
    seen_by_group = {group: 0 for group in group_counts}
    selected_indices: list[int] = []
    row_offset = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        groups = batch_group_keys(batch, columns)
        for local_index, group in enumerate(groups):
            rank = seen_by_group[group]
            if rank in selected_ranks.get(group, set()):
                selected_indices.append(row_offset + local_index)
            seen_by_group[group] = rank + 1
        row_offset += len(groups)

    if len(selected_indices) != limit_rows:
        raise RuntimeError(f"Expected {limit_rows:,} sampled rows from {path.name}, got {len(selected_indices):,}")
    return selected_indices


def date_stratified_sample_indices(path: Path, batch_size: int, limit_rows: int) -> list[int] | None:
    return stratified_sample_indices(path, batch_size, limit_rows, ["dt"])


def product_date_stratified_sample_indices(path: Path, batch_size: int, limit_rows: int) -> list[int] | None:
    return stratified_sample_indices(path, batch_size, limit_rows, ["dt", "product_id"])


def count_store_product_pairs(path: Path, batch_size: int) -> dict[tuple[int, int], int]:
    parquet_file = pq.ParquetFile(path)
    counts: dict[tuple[int, int], int] = {}
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["store_id", "product_id"]):
        df = pa.Table.from_batches([batch]).to_pandas()
        for store_id, product_id in df[["store_id", "product_id"]].itertuples(index=False, name=None):
            pair = (int(store_id), int(product_id))
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def select_store_product_panel(path: Path, batch_size: int, target_rows: int | None, seed: int) -> tuple[frozenset[tuple[int, int]], int]:
    pair_counts = count_store_product_pairs(path, batch_size)
    pairs = list(pair_counts)
    if not pairs:
        return frozenset(), 0
    if target_rows is None:
        return frozenset(pairs), sum(pair_counts.values())

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(pairs))
    selected: list[tuple[int, int]] = []
    selected_rows = 0
    for pair_index in shuffled_indices:
        pair = pairs[int(pair_index)]
        next_rows = selected_rows + pair_counts[pair]
        if selected and abs(target_rows - selected_rows) <= abs(target_rows - next_rows):
            break
        selected.append(pair)
        selected_rows = next_rows
        if selected_rows >= target_rows:
            break

    return frozenset(selected), selected_rows


def sampled_row_indices(path: Path, batch_size: int, limit_rows: int | None, sample_mode: str) -> list[int] | None:
    if limit_rows is None or sample_mode in {"head", "store-product-panel"}:
        return None
    parquet_file = pq.ParquetFile(path)
    total_rows = parquet_file.metadata.num_rows
    if sample_mode == "even":
        return even_sample_indices(total_rows, limit_rows)
    if sample_mode == "date-stratified":
        return date_stratified_sample_indices(path, batch_size, limit_rows)
    if sample_mode == "product-date-stratified":
        return product_date_stratified_sample_indices(path, batch_size, limit_rows)
    raise ValueError(f"Unsupported staging sample mode: {sample_mode}")


def take_selected_rows(table: pa.Table, selected_indices: list[int], start_row: int, pointer: int) -> tuple[pa.Table | None, int]:
    end_row = start_row + table.num_rows
    relative_indices: list[int] = []
    while pointer < len(selected_indices) and selected_indices[pointer] < end_row:
        if selected_indices[pointer] >= start_row:
            relative_indices.append(selected_indices[pointer] - start_row)
        pointer += 1
    if not relative_indices:
        return None, pointer
    return table.take(pa.array(relative_indices, type=pa.int64())), pointer


def filter_store_product_panel(df: pd.DataFrame, panel_pairs: frozenset[tuple[int, int]] | None) -> pd.DataFrame:
    if panel_pairs is None:
        return df
    mask = [(int(store_id), int(product_id)) in panel_pairs for store_id, product_id in zip(df["store_id"], df["product_id"])]
    return df.loc[mask].copy()


def iter_parquet_batches(
    path: Path,
    source_split: str,
    batch_size: int,
    limit_rows: int | None,
    sample_mode: str,
    panel_pairs: frozenset[tuple[int, int]] | None = None,
):
    parquet_file = pq.ParquetFile(path)
    selected_indices = sampled_row_indices(path, batch_size, limit_rows, sample_mode)
    selected_pointer = 0
    row_offset = 0
    loaded = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        if limit_rows is not None and loaded >= limit_rows:
            break

        table = pa.Table.from_batches([batch])
        if selected_indices is not None:
            table, selected_pointer = take_selected_rows(table, selected_indices, row_offset, selected_pointer)
            row_offset += batch.num_rows
            if table is None:
                continue
        elif limit_rows is not None:
            remaining = limit_rows - loaded
            if table.num_rows > remaining:
                table = table.slice(0, remaining)
            row_offset += batch.num_rows
        else:
            row_offset += batch.num_rows

        df = filter_store_product_panel(table.to_pandas(), panel_pairs)
        if df.empty:
            continue
        df.insert(0, "source_split", source_split)
        df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
        df["hours_sale"] = df["hours_sale"].map(pg_float_array)
        df["hours_stock_status"] = df["hours_stock_status"].map(pg_int_array)
        df = df[STAGING_COLUMNS]

        loaded += len(df)
        yield df, loaded


def load_staging(conn: psycopg.Connection, data_dir: Path, batch_size: int, args: argparse.Namespace) -> None:
    panel_pairs: frozenset[tuple[int, int]] | None = None
    if args.staging_sample_mode == "store-product-panel":
        target_rows = resolve_split_limit(args, "train")
        panel_pairs, selected_train_rows = select_store_product_panel(data_dir / "train.parquet", batch_size, target_rows, args.panel_seed)
        if not panel_pairs:
            raise RuntimeError("No store-product pairs found for panel sampling.")
        if args.eval_limit_rows is not None:
            print("store-product-panel mode ignores --eval-limit-rows and loads eval rows for the selected train panel")
        target_text = "full train split" if target_rows is None else f"target {target_rows:,} train rows"
        print(
            f"selected store-product panel with seed {args.panel_seed}: "
            f"{len(panel_pairs):,} pairs, {selected_train_rows:,} train rows ({target_text})"
        )

    for source_split in ("train", "eval"):
        path = data_dir / f"{source_split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        if panel_pairs is not None:
            limit_rows = None
            print(f"loading staging {source_split}: selected store-product panel")
        else:
            limit_rows = resolve_split_limit(args, source_split)
            if limit_rows is None:
                print(f"loading staging {source_split}: full split")
            else:
                print(f"loading staging {source_split}: {limit_rows:,} row cap using {args.staging_sample_mode} sampling")

        for df, loaded in iter_parquet_batches(path, source_split, batch_size, limit_rows, args.staging_sample_mode, panel_pairs):
            copy_dataframe(conn, "staging.fresh_retail_observation_day", STAGING_COLUMNS, df)
            print(f"loaded staging {source_split}: {loaded:,} rows")


def populate_dimensions_and_daily_fact(conn: psycopg.Connection) -> None:
    sql = """
    INSERT INTO dw.dim_date (
        date_key, full_date, day_of_week, day_name, week_of_year,
        month_number, month_name, quarter_number, year_number, is_weekend, holiday_flag
    )
    SELECT
        TO_CHAR(dt, 'YYYYMMDD')::INTEGER AS date_key,
        dt AS full_date,
        EXTRACT(ISODOW FROM dt)::SMALLINT AS day_of_week,
        TO_CHAR(dt, 'FMDay') AS day_name,
        EXTRACT(WEEK FROM dt)::SMALLINT AS week_of_year,
        EXTRACT(MONTH FROM dt)::SMALLINT AS month_number,
        TO_CHAR(dt, 'FMMonth') AS month_name,
        EXTRACT(QUARTER FROM dt)::SMALLINT AS quarter_number,
        EXTRACT(YEAR FROM dt)::SMALLINT AS year_number,
        EXTRACT(ISODOW FROM dt) IN (6, 7) AS is_weekend,
        MAX(holiday_flag)::SMALLINT AS holiday_flag
    FROM staging.fresh_retail_observation_day
    GROUP BY dt
    ON CONFLICT (date_key) DO UPDATE SET
        holiday_flag = EXCLUDED.holiday_flag;

    INSERT INTO dw.dim_time (time_key, hour_of_day, is_business_hour_6_22, day_part)
    SELECT
        hour_value::SMALLINT AS time_key,
        hour_value::SMALLINT AS hour_of_day,
        hour_value BETWEEN 6 AND 21 AS is_business_hour_6_22,
        CASE
            WHEN hour_value BETWEEN 5 AND 10 THEN 'morning'
            WHEN hour_value BETWEEN 11 AND 16 THEN 'afternoon'
            WHEN hour_value BETWEEN 17 AND 21 THEN 'evening'
            ELSE 'night'
        END AS day_part
    FROM GENERATE_SERIES(0, 23) AS hour_value
    ON CONFLICT (time_key) DO NOTHING;

    INSERT INTO dw.dim_city (city_id)
    SELECT DISTINCT city_id
    FROM staging.fresh_retail_observation_day
    ON CONFLICT (city_id) DO NOTHING;

    INSERT INTO dw.dim_store (store_id, city_key)
    SELECT DISTINCT s.store_id, c.city_key
    FROM staging.fresh_retail_observation_day s
    JOIN dw.dim_city c ON c.city_id = s.city_id
    ON CONFLICT (store_id) DO UPDATE SET
        city_key = EXCLUDED.city_key;

    INSERT INTO dw.dim_product (
        product_id, management_group_id, first_category_id, second_category_id, third_category_id
    )
    SELECT DISTINCT
        product_id,
        management_group_id,
        first_category_id,
        second_category_id,
        third_category_id
    FROM staging.fresh_retail_observation_day
    ON CONFLICT (product_id) DO UPDATE SET
        management_group_id = EXCLUDED.management_group_id,
        first_category_id = EXCLUDED.first_category_id,
        second_category_id = EXCLUDED.second_category_id,
        third_category_id = EXCLUDED.third_category_id;

    INSERT INTO dw.fact_sales_inventory_daily (
        date_key,
        store_key,
        product_key,
        source_split,
        observed_daily_sales_amount,
        stockout_hours_6_22,
        stockout_hours_total,
        has_stockout,
        discount_rate,
        holiday_flag,
        activity_flag,
        precpt,
        avg_temperature,
        avg_humidity,
        avg_wind_level
    )
    SELECT
        d.date_key,
        store.store_key,
        product.product_key,
        s.source_split,
        s.sale_amount,
        s.stock_hour6_22_cnt,
        stock_status.stockout_hours_total,
        stock_status.stockout_hours_total > 0 AS has_stockout,
        s.discount,
        s.holiday_flag,
        s.activity_flag,
        s.precpt,
        s.avg_temperature,
        s.avg_humidity,
        s.avg_wind_level
    FROM staging.fresh_retail_observation_day s
    JOIN dw.dim_date d ON d.full_date = s.dt
    JOIN dw.dim_store store ON store.store_id = s.store_id
    JOIN dw.dim_product product ON product.product_id = s.product_id
    CROSS JOIN LATERAL (
        SELECT SUM(value)::INTEGER AS stockout_hours_total
        FROM UNNEST(s.hours_stock_status) AS value
    ) stock_status
    ON CONFLICT (date_key, store_key, product_key) DO UPDATE SET
        source_split = EXCLUDED.source_split,
        observed_daily_sales_amount = EXCLUDED.observed_daily_sales_amount,
        stockout_hours_6_22 = EXCLUDED.stockout_hours_6_22,
        stockout_hours_total = EXCLUDED.stockout_hours_total,
        has_stockout = EXCLUDED.has_stockout,
        discount_rate = EXCLUDED.discount_rate,
        holiday_flag = EXCLUDED.holiday_flag,
        activity_flag = EXCLUDED.activity_flag,
        precpt = EXCLUDED.precpt,
        avg_temperature = EXCLUDED.avg_temperature,
        avg_humidity = EXCLUDED.avg_humidity,
        avg_wind_level = EXCLUDED.avg_wind_level;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def parse_date_arg(value: str | None) -> date | None:
    if value is None:
        return None
    return pd.to_datetime(value).date()


def get_hourly_dates(conn: psycopg.Connection, start_date: date | None, end_date: date | None) -> list[date]:
    clauses: list[str] = []
    params: list[date] = []
    if start_date is not None:
        clauses.append("full_date >= %s")
        params.append(start_date)
    if end_date is not None:
        clauses.append("full_date <= %s")
        params.append(end_date)
    where_sql = ""
    if clauses:
        where_sql = "WHERE " + " AND ".join(clauses)
    with conn.cursor() as cur:
        cur.execute(f"SELECT full_date FROM dw.dim_date {where_sql} ORDER BY full_date", tuple(params))
        return [row[0] for row in cur.fetchall()]


def populate_hourly_date(conn: psycopg.Connection, full_date: date) -> int:
    with conn.cursor() as cur:
        cur.execute(HOURLY_FACT_SQL, (full_date,))
        inserted = cur.rowcount
    conn.commit()
    return int(inserted)


def populate_hourly_worker(config: DbConfig, worker_id: int, dates: list[date]) -> tuple[int, int]:
    total_rows = 0
    with connect_config(config) as conn:
        for index, full_date in enumerate(dates, start=1):
            rows = populate_hourly_date(conn, full_date)
            total_rows += rows
            print(f"worker {worker_id}: populated hourly fact for {full_date} ({index}/{len(dates)}): {rows:,} rows")
    return worker_id, total_rows


def split_dates(dates: list[date], worker_count: int) -> list[list[date]]:
    chunks = [[] for _ in range(worker_count)]
    for index, full_date in enumerate(dates):
        chunks[index % worker_count].append(full_date)
    return [chunk for chunk in chunks if chunk]


def populate_hourly_fact(
    conn: psycopg.Connection,
    config: DbConfig,
    worker_count: int,
    start_date: date | None,
    end_date: date | None,
) -> None:
    dates = get_hourly_dates(conn, start_date, end_date)
    if not dates:
        print("no dates matched hourly fact filters")
        return

    worker_count = max(1, min(worker_count, len(dates)))
    print(f"populating hourly fact for {len(dates):,} date(s) with {worker_count} worker(s)")

    if worker_count == 1:
        total_rows = 0
        for index, full_date in enumerate(dates, start=1):
            rows = populate_hourly_date(conn, full_date)
            total_rows += rows
            print(f"populated hourly fact for {full_date} ({index}/{len(dates)}): {rows:,} rows; total {total_rows:,}")
        return

    chunks = split_dates(dates, worker_count)
    total_rows = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(populate_hourly_worker, config, worker_id, chunk)
            for worker_id, chunk in enumerate(chunks, start=1)
        ]
        for future in as_completed(futures):
            worker_id, rows = future.result()
            total_rows += rows
            print(f"worker {worker_id} finished: {rows:,} rows; hourly total {total_rows:,}")


def populate_hourly_fact_legacy(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT full_date FROM dw.dim_date ORDER BY full_date")
        dates = [row[0] for row in cur.fetchall()]

    for index, full_date in enumerate(dates, start=1):
        inserted = populate_hourly_date(conn, full_date)
        print(f"populated hourly fact for {full_date} ({index}/{len(dates)}): {inserted:,} rows")


def print_counts(conn: psycopg.Connection) -> None:
    tables = [
        "staging.fresh_retail_observation_day",
        "dw.dim_date",
        "dw.dim_time",
        "dw.dim_city",
        "dw.dim_store",
        "dw.dim_product",
        "dw.fact_sales_inventory_daily",
        "dw.fact_sales_inventory_hourly",
    ]
    with conn.cursor() as cur:
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            print(f"{table}: {count:,}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    schema_file = Path(args.schema_file)
    config = db_config_from_args(args)
    validate_row_limit("--limit-rows-per-split", args.limit_rows_per_split)
    validate_row_limit("--train-limit-rows", args.train_limit_rows)
    validate_row_limit("--eval-limit-rows", args.eval_limit_rows)
    hourly_start_date = parse_date_arg(args.hourly_start_date)
    hourly_end_date = parse_date_arg(args.hourly_end_date)
    if hourly_start_date and hourly_end_date and hourly_start_date > hourly_end_date:
        raise ValueError("--hourly-start-date must be before or equal to --hourly-end-date")

    with connect(args) as conn:
        print("applying schema")
        apply_schema(conn, schema_file)

        if args.schema_only:
            print("schema applied")
            return

        if args.reset:
            print("resetting warehouse tables")
            reset_tables(conn)

        if args.skip_staging:
            print("skipping staging load; reusing existing staging table")
        else:
            print("loading merged staging table")
            load_staging(conn, data_dir, args.batch_size, args)

        print("populating dimensions and daily fact")
        populate_dimensions_and_daily_fact(conn)

        if args.load_hourly:
            print("populating hourly fact; this can be large on the full dataset")
            populate_hourly_fact(conn, config, args.hourly_workers, hourly_start_date, hourly_end_date)
        else:
            print("skipping hourly fact; pass --load-hourly to build it")

        print_counts(conn)


if __name__ == "__main__":
    main()
