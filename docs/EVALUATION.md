# DSS Evaluation

This document records how the Fresh Retail DSS should be evaluated. The core evaluation is not only model error; it covers warehouse correctness, demand-estimation behavior, and whether the dashboard helps managers make replenishment decisions.

## 1. Evaluation Position

The project is now a warehouse-first DSS:

```text
Raw FreshRetailNet parquet
-> PostgreSQL staging and dimensional warehouse
-> hourly demand estimation
-> lost-sales estimation
-> daily risk scoring
-> recommended action
-> Streamlit dashboard
```

The model layer is useful for ranking and explaining stockout-censored demand, but it should not be described as production inventory optimization. FreshRetailNet does not include stock-on-hand, expiry dates, lead time, supplier limits, or true demand during stockout.

## 2. Warehouse Evaluation

Run:

```sql
SELECT *
FROM dw.v_data_quality_checks
ORDER BY severity, check_name;
```

Current checks:

| Check | Meaning |
|---|---|
| `hours_sale_length_24` | Every source row has 24 hourly sales values |
| `hours_stock_status_length_24` | Every source row has 24 hourly stockout flags |
| `daily_sales_equals_hourly_sum` | Daily sales equals sum of hourly sales |
| `business_hour_stockout_count_matches` | `stock_hour6_22_cnt` equals stockout flags from hours 6 through 21 |
| `duplicate_staging_grain` | Staging grain remains `source_split, store_id, product_id, dt` |

For local model evaluation, load a seeded store-product panel so selected pairs have continuous train history and matched future eval rows:

```bash
uv run python etl/load_fresh_retail_dw.py \
  --reset \
  --train-limit-rows 100000 \
  --staging-sample-mode store-product-panel \
  --panel-seed 42 \
  --load-hourly \
  --hourly-workers 4
```

## 3. Model Evaluation

The warehouse-native model trains from:

```text
dw.v_model_training_features_hourly
```

Training rule:

```text
Use only non-stockout rows.
Target = observed hourly sales amount.
```

Prediction rule:

```text
If stockout_flag = false:
    estimated_true_demand = observed_sales_amount

If stockout_flag = true:
    estimated_true_demand = max(observed_sales_amount, predicted_demand)
```

Metrics are evaluated on non-stockout eval rows because those are the rows where observed sales are usable labels. The local trainer reserves a temporal holdout from the train split for calibration; `eval.parquet` is not used to fit the model or choose the calibration factor.

Recommended local model run:

| Field | Value |
|---|---|
| Model | `xgboost_demand_m1_hurdle` |
| Version | `m1_panel100k_seed42_full_eval` |
| Train source | Selected store-product panel from `train.parquet` |
| Eval source | Matched future panel from `eval.parquet` |
| Calibration source | Train-only temporal holdout |

Interpretation:

```text
The run proves the M1-friendly training path, train-only calibration, future eval evaluation, and warehouse prediction load work end to end.
```

## 4. DSS Evaluation

The DSS should be graded by whether it supports the decision problem:

```text
Which store-product-date rows need restocking, order increase, markdown/reduction, or censored-demand review?
```

Recommended DSS checks:

| Check | Expected behavior |
|---|---|
| Action distribution | Rows are classified into restock, increase order, markdown, review, and maintain |
| Full business-hour stockout | Rows with `stockout_hours_6_22 = 16` rank high unless demand is truly near zero |
| Explainability | `decision_reason` explains every recommendation |
| Drill-down | Dashboard shows hourly observed sales, stockout flag, estimated demand, and lost sales |
| What-if | Threshold and service-level sliders change action counts without changing warehouse data |

Useful query:

```sql
SELECT decision_action, COUNT(*)
FROM dw.v_dss_daily_decision_score
GROUP BY decision_action
ORDER BY COUNT(*) DESC;
```

Current validated sample state:

```text
dw.fact_demand_estimate_hourly rows for latest M1 smoke model: 48,000
dw.fact_replenishment_recommendation_daily rows for latest M1 smoke model: 2,000
```

## 5. Apple Silicon Decision

Use XGBoost CPU histogram training for the default project demo:

```text
tree_method = hist
device = cpu
native macOS arm64 wheel via xgboost
```

This is the safest fast path on an M1 MacBook. CUDA-based GPU training is not applicable, and PyTorch MPS paths for Chronos/SAITS would make the project heavier without improving the DSS architecture.

## 6. Recommended Demo Command

```bash
uv run python ml/train_xgboost_demand_model.py \
  --model-type xgboost \
  --model-strategy hurdle \
  --model-name xgboost_demand_m1_hurdle \
  --model-version m1_panel100k_seed42_full_eval \
  --max-train-rows 2400000 \
  --n-estimators 300 \
  --max-depth 6 \
  --learning-rate 0.05 \
  --n-jobs 8 \
  --prior-blend-weight 0.5 \
  --calibration-objective balanced \
  --calibration-bias-penalty 0.25 \
  --max-calibration-factor 5 \
  --load-predictions
```

Do not pass `--max-eval-rows` for the final local evaluation. Use it only for smoke tests.

## 7. Report Wording

Use this wording in the final report:

```text
The DSS implements a complete warehouse-to-decision workflow for stockout-censored demand in fresh retail. Model accuracy is evaluated only on non-stockout observations because true stockout-period demand is unobserved. Recommended order quantity and waste risk are decision-support proxies, not true inventory optimization, because the dataset lacks stock-on-hand, expiry, lead time, and supplier constraints.
```
