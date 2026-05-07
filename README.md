# Fresh Retail Warehouse DSS

This project builds a PostgreSQL data warehouse for FreshRetailNet-50K demand, stockout, promotion, and weather data.

## Start PostgreSQL

```bash
docker compose up -d postgres
```

Connection defaults:

```text
host: localhost
port: 5433
database: fresh_retail_dw
user: warehouse
password: warehouse
```

## Install Python Dependencies

```bash
uv pip install -r requirements.txt
```

## Prepare Dataset

The dataset files are intentionally not committed because they are large. Place the real parquet files at:

```text
FreshRetailNet-50K/data/train.parquet
FreshRetailNet-50K/data/eval.parquet
```

One option is to clone the Hugging Face dataset and hydrate/download the parquet files:

```bash
git clone https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
uv pip install huggingface_hub
python3 - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
repo_dir = Path('FreshRetailNet-50K')
for filename in ['data/train.parquet', 'data/eval.parquet']:
    hf_hub_download(
        repo_id='Dingdong-Inc/FreshRetailNet-50K',
        repo_type='dataset',
        filename=filename,
        local_dir=repo_dir,
        force_download=True,
    )
PY
```

## Load A Fast Demo Warehouse

This loads 1,000 rows from each split into one merged staging table, then builds dimensions, daily facts, and hourly facts. This is the recommended under-5-minute workflow for demos and development.

```bash
python3 etl/load_fresh_retail_dw.py --reset --limit-rows-per-split 1000 --load-hourly
```

For a larger but still practical sample, use 10,000 rows per split:

```bash
python3 etl/load_fresh_retail_dw.py --reset --limit-rows-per-split 10000 --load-hourly
```

## Load The Full Daily Warehouse

```bash
python3 etl/load_fresh_retail_dw.py --reset
```

## Load The Full Hourly Warehouse

The full hourly fact expands `4,850,000` daily rows into about `116,400,000` hourly rows. This is not recommended for quick iteration on a laptop.

```bash
python3 etl/load_fresh_retail_dw.py --reset --load-hourly
```

## Run The DSS Dashboard

The dashboard supports four decision criteria:

- Reduce stockout rate.
- Minimize product waste.
- Enable faster restocking decisions.
- Reduce censored-demand bias from stockout-period sales.

Install dependencies and start the app:

```bash
uv pip install -r requirements.txt
streamlit run app/dss_dashboard.py
```

## Train A Demand Model For Real Decisions

The basic DSS can fall back to heuristic demand recovery, but real decisions should use a trained model. The training script uses the loaded hourly warehouse facts, learns a tree-based latent-demand model, stores the model artifact, and writes predictions back to the warehouse.

For the fast demo warehouse, train CatBoost and load predictions:

```bash
python3 ml/train_xgboost_demand_model.py \
  --model-type catboost \
  --model-name catboost_latent_demand_fast \
  --model-version fast_sample \
  --n-estimators 80 \
  --max-depth 6 \
  --learning-rate 0.08 \
  --max-train-rows 24000 \
  --max-eval-rows 24000 \
  --load-predictions
```

To compare against the cloned `RetailForecast/` approach, run the replication script:

```bash
python3 ml/replicate_retailforecast_xgboost.py \
  --model-type catboost \
  --train-daily-rows 20000 \
  --eval-daily-rows 10000 \
  --n-estimators 200
```

Outputs:

- Model artifact: `models/<model_name>_<version>.pkl`
- Metrics: `models/<model_name>_<version>_metrics.json`
- Hourly predictions: `dw.fact_demand_estimate_hourly`
- Daily order decisions: `dw.fact_replenishment_recommendation_daily`

The dashboard automatically uses the latest trained model from `dw.dim_model` when predictions exist.

Evaluation note:

```text
The current quick model is for DSS demonstration, not production ordering. The first quick XGBoost model had WAPE around 100.6%, which is weak. RetailForecast-style CatBoost improved local sample WAPE to about 67-68%, while the original RetailForecast notebook reports about 52% WMAPE for CatBoost.
```

See detailed documentation:

- `docs/ARCHITECTURE.md`
- `docs/PROJECT_FLOW.md`
- `docs/EVALUATION.md`

## Data Model

Schemas:

- `staging`: raw merged source data.
- `dw`: dimensional warehouse tables, facts, and decision-support views.

Main merged source table:

- `staging.fresh_retail_observation_day`

Dimensions:

- `dw.dim_date`
- `dw.dim_time`
- `dw.dim_city`
- `dw.dim_store`
- `dw.dim_product`
- `dw.dim_model`

Facts:

- `dw.fact_sales_inventory_daily`
- `dw.fact_sales_inventory_hourly`
- `dw.fact_demand_estimate_hourly`
- `dw.fact_replenishment_recommendation_daily`

Useful views:

- `dw.v_daily_restock_monitor`
- `dw.v_stockout_rate_by_category`
- `dw.v_dss_hourly_demand_estimate`
- `dw.v_dss_daily_decision_score`
- `dw.v_dss_kpi_by_day`
- `dw.v_dss_kpi_by_category`

Sample decision-support queries are in `sql/002_sample_queries.sql`.
