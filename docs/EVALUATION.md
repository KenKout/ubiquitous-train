# DSS Model Evaluation

This document records the model evaluation status for the Fresh Retail DSS. It is intentionally explicit about what is currently strong, what is weak, and what should be improved before using the system for real inventory ordering.

## 1. Evaluation Summary

The DSS is functionally complete, but the forecasting model is still experimental.

Current conclusion:

```text
The current quick warehouse model is suitable for demonstrating the DSS flow, but not yet strong enough for real production ordering decisions.
```

Reason:

- The first quick XGBoost DSS model had `WAPE = 1.006`, or about `100.6%`.
- A WAPE around `100%` means the model's total absolute error is approximately equal to the total actual demand.
- In inventory planning terms, this is weak because the forecast error is about as large as the thing being forecast.
- RetailForecast-style replication improved the result, especially with CatBoost, but it still needs further tuning and/or larger representative training data.

## 2. Why WAPE Matters

WAPE means Weighted Absolute Percentage Error:

```text
WAPE = sum(abs(actual - forecast)) / sum(actual)
```

Interpretation:

| WAPE | Meaning for Inventory |
|---:|---|
| `< 30%` | Strong practical forecast for many retail use cases |
| `30% - 60%` | Potentially usable, depending on product volatility |
| `60% - 80%` | Weak but may still help rank high-risk products |
| `80% - 100%` | Poor for quantity decisions |
| `> 100%` | Very poor for ordering quantity decisions |

Therefore, the feedback that `WAPE = 1.006` is not good is correct.

## 3. Current Model Results

### Quick Warehouse XGBoost Model

Artifact:

```text
models/xgboost_latent_demand_20260507152251_metrics.json
```

Metrics:

| Metric | Value |
|---|---:|
| Evaluation rows | 240,000 |
| MAE | 0.0613 |
| RMSE | 0.1087 |
| WAPE | 1.0060 |
| WAPE percent | 100.60% |
| Bias | approximately 0% |

Interpretation:

- Bias is low, meaning the total forecast volume is close to the total target volume.
- WAPE is high, meaning individual hourly predictions are frequently wrong even if total volume balances out.
- This model should not be presented as the best final model.

### Fast Warehouse CatBoost Model

Artifact:

```text
models/catboost_latent_demand_fast_fast_sample_metrics.json
```

Metrics:

| Metric | Value |
|---|---:|
| Evaluation rows | 24,000 |
| MAE | 0.0598 |
| RMSE | 0.1103 |
| WAPE | 1.1070 |
| WAPE percent | 110.70% |
| Bias | -26.95% |

Interpretation:

- This was trained for speed on a very small sample.
- It proves that model predictions can be loaded into the DSS tables.
- It is not better than the earlier XGBoost model in evaluation quality.

## 4. RetailForecast Replication Results

The `RetailForecast/` repository was cloned and inspected. Its notebook uses:

- FreshRetailNet data transformed into an hourly forecasting schema.
- Contextual features such as store traffic lag and category momentum.
- Optional latent-demand recovery using imputation models such as SAITS/DLinear.
- Tree models including XGBoost, LightGBM, Random Forest, and CatBoost.
- Chronos for foundation-model time-series forecasting.

A standalone replication script was added:

```text
ml/replicate_retailforecast_xgboost.py
```

Despite the filename, it supports:

- `xgboost`
- `lightgbm`
- `catboost`

### Replication Results On Local Samples

| Run | Model | Target Mode | Train Daily Rows | Eval Daily Rows | Eval Hourly Rows | MAE | RMSE | WAPE | Bias |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Smoke | XGBoost | Latent baseline | 20,000 | 10,000 | 240,000 | 0.0450 | 0.1019 | 77.46% | +42.50% |
| Smoke | LightGBM | Latent baseline | 20,000 | 10,000 | 240,000 | 0.0468 | 0.1138 | 80.40% | +46.44% |
| Smoke | CatBoost | Latent baseline | 20,000 | 10,000 | 240,000 | 0.0394 | 0.0917 | 67.69% | +29.65% |
| Larger sample | CatBoost | Latent baseline | 100,000 | 50,000 | 1,200,000 | 0.0382 | 0.0995 | 68.31% | +38.35% |
| Larger sample | CatBoost | Observed sales | 100,000 | 50,000 | 1,200,000 | 0.0332 | 0.0945 | 70.48% | +41.26% |

Best local replication result so far:

```text
RetailForecast-style CatBoost, WAPE about 67.69%
```

This is meaningfully better than the quick warehouse model's `100.6%`, but still not good enough for accurate order quantity decisions.

## 5. RetailForecast Notebook Reference Results

The original `RetailForecast/pipeline_final.ipynb` includes stronger reported results on the author's environment and data preparation cache.

Observed notebook results include:

| Model | MAE | RMSE | WMAPE | Bias |
|---|---:|---:|---:|---:|
| Random Forest | 0.0320 | 0.1081 | 60.48% | -3.03% |
| LightGBM | 0.0324 | 0.0817 | 61.33% | -4.62% |
| CatBoost | 0.0275 | 0.1024 | 51.99% | -8.55% |
| Lasso | 0.0703 | 0.1420 | 133.00% | -14.20% |
| Chronos 0.1 quantile | 0.0607 | 0.1420 | 97.54% | -97.26% |
| Chronos 0.5 quantile | 0.0520 | 0.1145 | 83.46% | -67.43% |
| Chronos 0.9 quantile | 0.0952 | 0.1413 | 152.97% | +112.47% |

Important interpretation:

- The best reported notebook result is CatBoost at about `51.99%` WMAPE.
- Chronos was not automatically better in the notebook results.
- The notebook relies on expensive data preparation and latent-demand recovery steps that we have not fully reproduced inside the warehouse pipeline yet.

## 6. Why Our Local Replication Does Not Fully Match RetailForecast Yet

Differences from the original notebook:

1. We did not run full SAITS/DLinear imputation because it is heavy and requires PyTorch dependencies.
2. We used a simpler latent-demand baseline instead of the notebook's deep imputation target recovery.
3. We used samples instead of the full exploded hourly dataset.
4. FreshRetailNet does not include some RetailForecast schema fields such as `stock_on_hand`, `selling_price`, and `product_price`, so they are currently placeholder values.
5. Our goal is to keep the DSS demo runnable quickly, ideally within a few minutes.

## 7. macOS GPU Decision

The machine has Apple M1 GPU / Metal support, but the current models do not benefit from it:

| Model | Apple GPU / MPS Usefulness |
|---|---|
| XGBoost | Not useful; GPU path expects NVIDIA CUDA |
| CatBoost | Not useful; GPU path expects NVIDIA CUDA |
| LightGBM | Not useful in this installed setup |
| Chronos / SAITS / DLinear | Could use PyTorch MPS, but requires heavier dependencies and longer setup |

Decision:

```text
Use CPU-optimized tree models for the current DSS demo.
```

## 8. Recommended Demo Strategy

For the project demo, use a representative sample instead of the full dataset.

Recommended fast warehouse sample:

```text
1,000 to 10,000 daily rows per split
24,000 to 240,000 hourly training rows
48,000 to 480,000 hourly prediction rows
```

Why:

- Full hourly expansion is about `116.4M` rows.
- Full loading and prediction is too slow for interactive project work.
- A sample is enough to demonstrate the warehouse, model, DSS tables, and dashboard.
- The model evaluation should be reported honestly as experimental.

Recommended model for demonstration:

```text
RetailForecast-style CatBoost, because it had the best local replication WAPE.
```

Recommended wording for the report:

```text
The DSS currently demonstrates the complete decision-support workflow. Forecast accuracy is not yet production-grade; evaluation shows that CatBoost improves over the initial XGBoost baseline, but further work is required to reproduce the full RetailForecast latent-demand recovery pipeline and reduce WAPE before deployment.
```

## 9. How To Run Evaluation

Quick warehouse DSS model:

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

RetailForecast-style local comparison:

```bash
python3 ml/replicate_retailforecast_xgboost.py \
  --model-type catboost \
  --train-daily-rows 20000 \
  --eval-daily-rows 10000 \
  --n-estimators 200
```

Larger RetailForecast-style comparison:

```bash
python3 ml/replicate_retailforecast_xgboost.py \
  --model-type catboost \
  --train-daily-rows 100000 \
  --eval-daily-rows 50000 \
  --n-estimators 300
```

## 10. Final Evaluation Position

Current best statement:

```text
The DSS architecture and data pipeline are correct and runnable. The model layer is integrated, but model quality is still a limitation. The initial WAPE of about 100.6% is poor. RetailForecast-style CatBoost improves the result to about 67-68% WAPE on local samples, and the original RetailForecast notebook reports about 52% WMAPE for CatBoost. Further work should focus on reproducing the full latent-demand recovery pipeline and improving model quality before using the system for real ordering decisions.
```
