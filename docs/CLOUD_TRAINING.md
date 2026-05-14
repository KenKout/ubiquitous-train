# Cloud GPU Training Workflow

Use Kaggle or Colab for the expensive model step only. Keep PostgreSQL and the Streamlit DSS local.

## 1. Recommended Scale

Use `100,000` daily rows per split when runtime allows:

| Rows per split | Total daily rows | Hourly feature rows |
|---:|---:|---:|
| 10,000 | 20,000 | 480,000 |
| 50,000 | 100,000 | 2,400,000 |
| 100,000 | 200,000 | 4,800,000 |

Do not use `100k` boosting rounds. For XGBoost/CatBoost, start with `600` to `1000` trees on GPU.

## 2. Local: Load A Larger Warehouse Sample

```bash
uv run python etl/load_fresh_retail_dw.py \
  --reset \
  --limit-rows-per-split 100000 \
  --load-hourly
```

This creates about `4.8M` hourly rows. If that is too slow, use `50000` or `10000` first.

## 3. Local: Export Model Features

```bash
uv run python ml/export_model_features.py \
  --output exports/freshretail_features_100k.parquet
```

Optional smoke export:

```bash
uv run python ml/export_model_features.py \
  --output exports/freshretail_features_smoke.parquet \
  --limit-rows 500000
```

Upload the exported parquet to a Kaggle Dataset or Google Drive.

## 4. Kaggle: Train On GPU

Kaggle setup:

```bash
git clone https://github.com/KenKout/ubiquitous-train.git
cd ubiquitous-train
python -m pip install -U xgboost scikit-learn pyarrow pandas
```

Example Kaggle command:

```bash
python ml/cloud_gpu_train.py \
  --features /kaggle/input/freshretail-features-100k/freshretail_features_100k.parquet \
  --output-dir /kaggle/working/freshretail_cloud_outputs \
  --model-type xgboost \
  --device cuda \
  --model-name kaggle_xgboost_gpu \
  --model-version v1 \
  --n-estimators 800 \
  --max-depth 8 \
  --learning-rate 0.05
```

For a smoke run:

```bash
python ml/cloud_gpu_train.py \
  --features /kaggle/input/freshretail-features-smoke/freshretail_features_smoke.parquet \
  --output-dir /kaggle/working/freshretail_cloud_outputs \
  --device cuda \
  --model-version smoke \
  --n-estimators 100 \
  --max-predict-rows 200000
```

Output file to download:

```text
freshretail_cloud_outputs/kaggle_xgboost_gpu_v1_predictions.parquet
```

## 5. Colab Alternative

Colab command shape is the same. Mount Drive, then point `--features` to the Drive parquet path and `--output-dir` to a Drive output folder.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
git clone https://github.com/KenKout/ubiquitous-train.git
cd ubiquitous-train
python -m pip install -U xgboost scikit-learn pyarrow pandas
python ml/cloud_gpu_train.py \
  --features /content/drive/MyDrive/freshretail_features_100k.parquet \
  --output-dir /content/drive/MyDrive/freshretail_cloud_outputs \
  --device cuda \
  --model-name colab_xgboost_gpu \
  --model-version v1 \
  --n-estimators 800
```

## 6. Local: Import Predictions Back To PostgreSQL

After downloading the prediction parquet:

```bash
uv run python ml/import_cloud_predictions.py \
  --predictions exports/kaggle_xgboost_gpu_v1_predictions.parquet \
  --model-name kaggle_xgboost_gpu \
  --model-version v1 \
  --replace-model-output
```

Then run the dashboard:

```bash
uv run streamlit run app/dss_dashboard.py
```

The dashboard will use `dw.v_latest_model_with_predictions`, so the imported cloud model becomes the active DSS model automatically.

## 7. Why This Is Better Than Full PostgreSQL Training

PostgreSQL remains the DSS serving layer. Kaggle/Colab handles model compute. This avoids making the local database carry every full-scale training experiment while preserving the warehouse-to-decision architecture.
