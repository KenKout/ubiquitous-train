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

## Current Process

The current repository workflow is:

1. Start PostgreSQL with Docker Compose.
2. Install the Python dependencies.
3. Hydrate the real `FreshRetailNet-50K/data/train.parquet` and `FreshRetailNet-50K/data/eval.parquet` files.
4. Load the warehouse with a fast demo sample or a larger practical sample.
5. Train a demand model and optionally write predictions and recommendations back to PostgreSQL.
6. Run the Streamlit DSS dashboard against the latest model data when available.

For a precise record of what is currently implemented, see [docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md).

## Prepare Dataset

The dataset files are intentionally not committed because they are large. The loader auto-detects the sibling dataset path used for this project:

```text
../FreshRetailNet-50K/data/train.parquet
../FreshRetailNet-50K/data/eval.parquet
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

For Colab or larger local machines, use parallel hourly expansion. Start with 4 workers; increase only if PostgreSQL and disk writes remain stable:

```bash
uv run python etl/load_fresh_retail_dw.py \
  --reset \
  --limit-rows-per-split 1000000 \
  --load-hourly \
  --hourly-workers 4
```

You can also restrict hourly expansion to a date window while keeping loaded staging/daily data:

```bash
uv run python etl/load_fresh_retail_dw.py \
  --skip-staging \
  --load-hourly \
  --hourly-start-date 2024-06-01 \
  --hourly-end-date 2024-07-02 \
  --hourly-workers 4
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

On Colab, prefer:

```bash
uv run python etl/load_fresh_retail_dw.py --reset --load-hourly --hourly-workers 4
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

When `--load-predictions` is not used, the dashboard still works by reading the heuristic fallback views in PostgreSQL. When predictions are loaded, it automatically uses the latest trained model registered in `dw.dim_model`.

For the fast demo warehouse, train the M1-friendly XGBoost CPU histogram model and load predictions:

```bash
python3 ml/train_xgboost_demand_model.py \
  --model-type xgboost \
  --model-name xgboost_demand_fast \
  --model-version fast_sample \
  --n-estimators 120 \
  --max-depth 6 \
  --learning-rate 0.08 \
  --max-train-rows 24000 \
  --max-eval-rows 24000 \
  --load-predictions
```

Apple Silicon note:

```text
XGBoost with tree_method=hist and device=cpu runs natively on macOS arm64 wheels and is the safest fast option on M1/M2 machines.
The script also supports CatBoost when installed, but the core workflow does not require CUDA, MPS, Chronos, SAITS, LSTM, or GRU.
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

## Train On Kaggle Or Colab GPU

For larger samples, keep PostgreSQL local and send a parquet feature export to Kaggle/Colab:

```bash
uv run python etl/load_fresh_retail_dw.py --reset --limit-rows-per-split 100000 --load-hourly

uv run python ml/export_model_features.py \
  --output exports/freshretail_features_100k.parquet
```

When PostgreSQL and GPU training are on the same Colab runtime, you can skip this export and train directly from the warehouse:

```bash
uv run python ml/cloud_gpu_train.py \
  --warehouse-start-date 2024-06-01 \
  --warehouse-end-date 2024-07-02 \
  --warehouse-no-order \
  --output-dir cloud_outputs \
  --device cuda \
  --model-name colab_xgboost_gpu \
  --model-version v3_direct \
  --n-estimators 800
```

For faster parquet exports, use `--no-order`, `--sample-rate`, or date filters.

On Kaggle/Colab, train with:

```bash
python ml/cloud_gpu_train.py \
  --features /path/to/freshretail_features_100k.parquet \
  --output-dir cloud_outputs \
  --device cuda \
  --model-name kaggle_xgboost_gpu \
  --model-version v1 \
  --n-estimators 800 \
  --xgboost-objective reg:tweedie
```

Cloud training now enables advanced historical demand priors, stockout-rate features, cyclic time features, eval-set calibration, and segmented model metrics by default.

Then import the downloaded predictions locally:

```bash
uv run python ml/import_cloud_predictions.py \
  --predictions exports/kaggle_xgboost_gpu_v1_predictions.parquet \
  --model-name kaggle_xgboost_gpu \
  --model-version v1 \
  --replace-model-output
```

If matching metrics/metadata JSON files are next to the prediction parquet, they are imported into `dw.fact_model_evaluation` and shown in the dashboard model-quality panel.

Use `100000` as daily rows per split, not `100000` boosting rounds. Start with `600-1000` trees for GPU training. See `docs/CLOUD_TRAINING.md` for the full workflow.

Evaluation note:

```text
Model MAE/RMSE/WMAPE are evaluated on non-stockout rows, where observed sales are usable demand labels.
True demand during stockout is unobserved, so stockout-row evaluation remains a DSS approximation rather than ground truth.
```

See detailed documentation:

- `docs/ARCHITECTURE.md`
- `docs/PROJECT_FLOW.md`
- `docs/EVALUATION.md`
- `docs/CLOUD_TRAINING.md`

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
- `dw.fact_model_evaluation`

Useful views:

- `dw.v_daily_restock_monitor`
- `dw.v_stockout_rate_by_category`
- `dw.v_data_quality_checks`
- `dw.v_model_training_features_hourly`
- `dw.v_latest_model_with_predictions`
- `dw.v_model_quality_summary`
- `dw.v_dss_hourly_demand_estimate`
- `dw.v_dss_daily_decision_score`
- `dw.v_dss_kpi_by_day`
- `dw.v_dss_kpi_by_category`

Implementation notes:

- Bronze data is stored in `staging.fresh_retail_observation_day`.
- Daily and hourly warehouse facts are built in `dw.fact_sales_inventory_daily` and `dw.fact_sales_inventory_hourly`.
- Model outputs are stored in `dw.fact_demand_estimate_hourly` and `dw.fact_replenishment_recommendation_daily`.
- The dashboard queries `dw.v_dss_daily_decision_score`, which combines the latest model output if present, otherwise uses the SQL heuristic fallback.

Sample decision-support queries are in `sql/002_sample_queries.sql`.
