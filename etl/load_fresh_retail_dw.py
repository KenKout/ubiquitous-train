#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from io import StringIO
from pathlib import Path

import pandas as pd
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
    parser.add_argument("--limit-rows-per-split", type=int, default=None, help="Useful for a fast demo load.")
    parser.add_argument("--reset", action="store_true", help="Truncate staging and warehouse tables before loading.")
    parser.add_argument("--schema-only", action="store_true", help="Apply schema and exit.")
    parser.add_argument("--load-hourly", action="store_true", help="Populate hourly fact table. Full data expands to about 116M rows.")
    parser.add_argument("--skip-staging", action="store_true", help="Reuse existing staging rows and only build dimensions/facts.")
    return parser.parse_args()


def connect(args: argparse.Namespace) -> psycopg.Connection:
    return psycopg.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
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


def iter_parquet_batches(path: Path, source_split: str, batch_size: int, limit_rows: int | None):
    parquet_file = pq.ParquetFile(path)
    loaded = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        if limit_rows is not None and loaded >= limit_rows:
            break

        table = pa.Table.from_batches([batch])
        if limit_rows is not None:
            remaining = limit_rows - loaded
            if table.num_rows > remaining:
                table = table.slice(0, remaining)

        df = table.to_pandas()
        df.insert(0, "source_split", source_split)
        df["dt"] = pd.to_datetime(df["dt"]).dt.strftime("%Y-%m-%d")
        df["hours_sale"] = df["hours_sale"].map(pg_float_array)
        df["hours_stock_status"] = df["hours_stock_status"].map(pg_int_array)
        df = df[STAGING_COLUMNS]

        loaded += len(df)
        yield df, loaded


def load_staging(conn: psycopg.Connection, data_dir: Path, batch_size: int, limit_rows: int | None) -> None:
    for source_split in ("train", "eval"):
        path = data_dir / f"{source_split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)

        for df, loaded in iter_parquet_batches(path, source_split, batch_size, limit_rows):
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


def populate_hourly_fact(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT full_date FROM dw.dim_date ORDER BY full_date")
        dates = [row[0] for row in cur.fetchall()]

    sql = """
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

    for index, full_date in enumerate(dates, start=1):
        with conn.cursor() as cur:
            cur.execute(sql, (full_date,))
            inserted = cur.rowcount
        conn.commit()
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
            load_staging(conn, data_dir, args.batch_size, args.limit_rows_per_split)

        print("populating dimensions and daily fact")
        populate_dimensions_and_daily_fact(conn)

        if args.load_hourly:
            print("populating hourly fact; this can be large on the full dataset")
            populate_hourly_fact(conn)
        else:
            print("skipping hourly fact; pass --load-hourly to build it")

        print_counts(conn)


if __name__ == "__main__":
    main()
