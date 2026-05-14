# Tài Liệu Dữ Liệu và Dự Đoán - Fresh Retail DSS

> **Ngày cập nhật:** Tháng 5/2026
> **Phiên bản:** 1.0

---

## Mục Lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Ba Loại "Giá Trị" Trong Hệ Thống](#2-ba-loại-giá-trị-trong-hệ-thống)
3. [Nguồn Dữ Liệu Gốc - FreshRetailNet-50K](#3-nguồn-dữ-liệu-gốc---freshretailnet-50k)
4. [Schema Kho Dữ Liệu - PostgreSQL](#4-schema-kho-dữ-liệu---postgresql)
5. [Cách Tính Toán Các Chỉ Số Dự Đoán](#5-cách-tính-toán-các-chỉ-số-dự-đoán)
6. [Mô Hình Dự Báo Demand](#6-mô-hình-dự-báo-demand)
7. [Luồng Xử Lý Dữ Liệu](#7-luồng-xử-lý-dữ-liệu)
8. [Dashboard DSS - Hiển Thị Các Chỉ Số](#8-dashboard-dss---hiển-thị-các-chỉ-số)
9. [Giới Hạn Của Hệ Thống](#9-giới-hạn-của-hệ-thống)
10. [Từ Điển Dữ Liệu](#10-từ-điển-dữ-liệu)

---

## 1. Tổng Quan Hệ Thống

Fresh Retail DSS là hệ thống hỗ trợ quyết định cho chuỗi cửa hàng bán lẻ tươi sống (fresh retail). Hệ thống sử dụng dataset **FreshRetailNet-50K** từ Hugging Face để:

- **Ước tính demand thực sự** (true demand) khi xảy ra stockout
- **Phục hồi phần demand bị mất** (recovered lost sales)
- **Đề xuất replenishment** cho các cửa hàng

### Các Thành Phần Chính

```
┌─────────────────────────────────────────────────────────────┐
│                     FreshRetailNet-50K                      │
│              (Hugging Face: Dingdong-Inc/FreshRetailNet-50K)│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  ETL Pipeline (etl/load_fresh_retail_dw.py)                 │
│  - train.parquet (2024-03-28 → 2024-06-25)                  │
│  - eval.parquet  (2024-06-26 → 2024-07-02)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PostgreSQL Data Warehouse (fresh_retail_dw)                │
│  - Schema: staging, dw, eval                                │
│  - Bảng fact: demand estimates, sales, recommendations     │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│ ML Training Pipeline     │     │ Dashboard DSS               │
│ (XGBoost Hurdle Model)  │     │ (Streamlit app)             │
│ - Feature engineering    │     │ - KPI hiển thị             │
│ - Demand prediction      │     │ - Chart hourly drill-down   │
└─────────────────────────┘     └─────────────────────────────┘
```

---

## 2. Ba Loại "Giá Trị" Trong Hệ Thống

### 2.1 Giá Trị Dự Đoán (Predicted Values)

**Vị trí trong warehouse:**
```sql
dw.fact_demand_estimate_hourly.estimated_true_demand
dw.fact_demand_estimate_hourly.estimated_lost_sales
```

**Ý nghĩa:**
- **Estimated demand** = Giá trị demand được model hoặc fallback heuristic ước tính
- **Recovered gap / Lost sales** = Phần demand bị mất do stockout mà hệ thống phục hồi được

**Dashboard hiển thị:**
- `Estimated demand` = model demand estimate
- `Recovered gap / lost sales` = estimated demand - observed sales (trong stockout hours)

---

### 2.2 Giá Trị Thật Trong Eval (Observed Values)

**Vị trí trong dataset:**
```sql
eval.parquet.observed sales    -- sale_amount hoặc hours_sale
eval.parquet.stockout flag     -- hours_stock_status
```

**Đặc điểm:**
- Chỉ có **một phần** của true demand
- Trong eval.parquet **không có** trường `true customer demand` đầy đủ

**Quan hệ với stockout:**
```
Nếu KHÔNG stockout:
  observed sales ≈ true demand (đáng tin cậy)

Nếu stockout:
  observed sales < true demand (bị censored)
  true demand thật sự không được ghi lại
```

**Ví dụ:** 2024-06-26 | store 537 | product 559
```
Observed sales thật trong eval: 0.90
Estimated demand (model): 1.32
Recovered lost sales: 0.42
Stockout flag: 17:00-23:00
Current stock quantity: KHÔNG CÓ trong dataset
```

---

### 2.3 Giá Trị Stock Hiện Tại (Current Inventory)

**FreshRetailNet có:**
```sql
hours_stock_status          -- Giờ nào bị stockout (0/1)
stock_hour6_22_cnt          -- Số giờ stockout trong business hours (6-21)
```

**FreshRetailNet KHÔNG có:**
```
- stock_on_hand (số lượng tồn kho hiện tại)
- inventory_quantity (số lượng trong kho)
- current_stock_level (mức stock hiện tại)
- lead_time (thời gian đặt hàng)
- shelf_life / expiry (hạn sử dụng)
- supplier_capacity (công suất supplier)
- reorder_constraints (ràng buộc đặt hàng lại)
```

**Hệ thống chỉ có thể nói:**
- "Source data cho thấy stockout ở những giờ này"
- "Model ước tính có demand bị mất"
- "DSS khuyên restock / increase replenishment"

**KHÔNG thể nói:**
- "Hiện tại còn 5 units"
- "Nên order thêm 20 units"

---

### 2.4 Tóm Tắt So Sánh Ba Loại Giá Trị

| Loại Giá Trị | Nguồn | Độ Tin Cậy | Stockout? |
|-------------|-------|-----------|-----------|
| **Predicted (estimated)** | Model/fallback | Model-dependent | Có thể ước tính |
| **Observed (eval)** | eval.parquet | Cao (non-stockout), Thấp (stockout) | Chỉ ghi nhận được sales thực tế |
| **True Demand** | Không có | Không xác định được | Không ghi nhận |

---

## 3. Nguồn Dữ Liệu Gốc - FreshRetailNet-50K

### 3.1 Dataset Overview

```
Nguồn: Hugging Face - Dingdong-Inc/FreshRetailNet-50K
Số dòng train: 4,500,000 dòng (2024-03-28 → 2024-06-25)
Số dòng eval: 350,000 dòng (2024-06-26 → 2024-07-02)
Đơn vị: Mỗi dòng = một store-product-date
```

### 3.2 Schema Đầy Đủ

| Trường | Kiểu | Mô Tả |
|--------|------|--------|
| `city_id` | int64 | Mã thành phố (encoded) |
| `store_id` | int64 | Mã cửa hàng (encoded) |
| `management_group_id` | int64 | Nhóm quản lý sản phẩm |
| `first_category_id` | int64 | Danh mục cấp 1 |
| `second_category_id` | int64 | Danh mục cấp 2 |
| `third_category_id` | int64 | Danh mục cấp 3 |
| `product_id` | int64 | Mã sản phẩm (encoded) |
| `dt` | string | Ngày (YYYY-MM-DD) |
| `sale_amount` | double | Tổng sales ngày |
| `hours_sale` | list<double> | 24 giá trị sales theo giờ |
| `stock_hour6_22_cnt` | int32 | Số giờ stockout (6-21h), tối đa 16 |
| `hours_stock_status` | list<int64> | 24 flag stockout (0/1) |
| `discount` | double | Tỷ lệ giảm giá |
| `holiday_flag` | int32 | Flag ngày lễ |
| `activity_flag` | int32 | Flag khuyến mãi |
| `precpt` | double | Lượng mưa |
| `avg_temperature` | double | Nhiệt độ trung bình |
| `avg_humidity` | double | Độ ẩm trung bình |
| `avg_wind_level` | double | Cấp gió trung bình |

### 3.3 Các Trường Stockout

```
hours_stock_status[i] = 1  --> Giờ thứ i bị stockout (không có hàng bán)
hours_stock_status[i] = 0  --> Giờ thứ i bình thường

stock_hour6_22_cnt = SUM(hours_stock_status[6:22])
                    = Số giờ stockout trong khoảng 6:00-21:59
```

---

## 4. Schema Kho Dữ Liệu - PostgreSQL

### 4.1 Cấu Trúc Schema

```
fresh_retail_dw
├── staging
│   └── fresh_retail_observation_day    -- Raw data (train + eval merged)
├── dw                                   -- Data Warehouse
│   ├── dim_date                         -- Chiều ngày
│   ├── dim_time                         -- Chiều giờ
│   ├── dim_city                         -- Chiều thành phố
│   ├── dim_store                        -- Chiều cửa hàng
│   ├── dim_product                      -- Chiều sản phẩm
│   ├── dim_model                        -- Chiều model
│   ├── fact_sales_inventory_daily       -- Fact ngày
│   ├── fact_sales_inventory_hourly      -- Fact giờ
│   ├── fact_demand_estimate_hourly      -- Fact dự đoán demand
│   ├── fact_replenishment_recommendation_daily -- Fact đề xuất order
│   ├── fact_model_evaluation             -- Model metrics
│   └── v_*                              -- Views DSS
└── eval                                 -- Schema eval (parquet import)
```

### 4.2 Chi Tiết Các Bảng Fact Quan Trọng

#### `dw.fact_demand_estimate_hourly`

Đây là bảng chính lưu trữ các giá trị dự đoán.

| Trường | Kiểu | Mô Tả |
|--------|------|--------|
| `store_key` | int | FK → dim_store |
| `product_key` | int | FK → dim_product |
| `date_key` | int | FK → dim_date |
| `time_key` | int | FK → dim_time |
| `model_key` | int | FK → dim_model |
| `estimated_true_demand` | numeric | Demand ước tính (bao gồm phục hồi) |
| `estimated_lost_sales` | numeric | Sales bị mất do stockout |
| `prediction_lower_bound` | numeric | Cận dưới dự đoán |
| `prediction_upper_bound` | numeric | Cận trên dự đoán |
| `stockout_flag` | boolean | Flag stockout |
| `is_censored_observation` | boolean | Observation có thể bị bias |

#### `dw.fact_sales_inventory_hourly`

Lưu trữ sales thực tế theo giờ.

| Trường | Kiểu | Mô Tả |
|--------|------|--------|
| `store_key` | int | FK → dim_store |
| `product_key` | int | FK → dim_product |
| `date_key` | int | FK → dim_date |
| `time_key` | int | FK → dim_time |
| `observed_sales_amount` | numeric | Sales thực tế quan sát được |
| `stockout_flag` | boolean | Có stockout không |
| `is_censored_observation` | boolean | Có bị ảnh hưởng bởi stockout |

#### `dw.fact_replenishment_recommendation_daily`

Đề xuất order hàng ngày.

| Trường | Kiểu | Mô Tả |
|--------|------|--------|
| `store_key` | int | FK → dim_store |
| `product_key` | int | FK → dim_product |
| `date_key` | int | FK → dim_date |
| `recommended_order_qty` | numeric | Số lượng đề xuất order |
| `expected_demand` | numeric | Demand kỳ vọng |
| `expected_lost_sales` | numeric | Sales có thể bị mất |
| `stockout_risk_score` | numeric | Điểm rủi ro stockout (0-1) |
| `expected_waste_qty` | numeric | Lượng hàng có thể bị lãng phí |
| `service_level_target` | numeric | Mức service level target |

### 4.3 Các View DSS Quan Trọng

#### `dw.v_dss_daily_decision_score`

View chính cho DSS, bao gồm:

| Trường | Kiểu | Mô Tả |
|--------|------|--------|
| `estimated_true_demand` | numeric | Tổng demand ước tính trong ngày |
| `estimated_lost_sales` | numeric | Tổng sales bị mất |
| `demand_bias_rate` | numeric | `estimated_lost_sales / estimated_true_demand` |
| `stockout_rate_6_22` | numeric | `stockout_hours_6_22 / 16` |
| `waste_risk_score` | numeric | Điểm rủi ro overstock |
| `restock_urgency_score` | numeric | Điểm urgency restock (0-1) |
| `decision_action` | varchar | Hành động quyết định |
| `decision_reason` | text | Lý do quyết định |

#### Decision Actions (5 loại)

| Action | Điều Kiện | Ý Nghĩa |
|--------|-----------|---------|
| `Restock immediately` | `restock_urgency_score >= 0.65` | Cần restock ngay |
| `Increase next order` | `stockout_rate_6_22 >= 0.20` | Tăng đơn hàng tiếp theo |
| `Reduce order or markdown` | `waste_risk_score >= 0.70` | Giảm order hoặc markdown |
| `Review censored demand` | `demand_bias_rate >= 0.20` | Xem xét demand bị censored |
| `Maintain plan` | Else | Giữ nguyên kế hoạch |

---

## 5. Cách Tính Toán Các Chỉ Số Dự Đoán

### 5.1 Công Thức Cho Stockout Hours

```sql
-- Với giờ bị stockout (stockout_flag = true):
estimated_true_demand = MAX(observed_sales_amount, predicted_demand_or_baseline)
estimated_lost_sales  = MAX(estimated_true_demand - observed_sales_amount, 0)
```

**Giải thích:**
- Khi stockout, observed sales bị censored (thấp hơn thực tế)
- Model hoặc baseline cung cấp demand ước tính
- Phần chênh lệch = lost sales

### 5.2 Công Thức Cho Non-Stockout Hours

```sql
-- Với giờ không stockout (stockout_flag = false):
estimated_true_demand = observed_sales_amount
estimated_lost_sales  = 0
```

**Giải thích:**
- Khi không stockout, observed sales phản ánh đúng true demand
- Không có lost sales

### 5.3 Fallback Heuristic (Khi Không Có Model)

Khi không có predictions từ model, hệ thống sử dụng hierarchical fallback:

```
1. Store-product-hour average
2. Product-hour average
3. Category-hour average
4. Product average
5. Global hour average
```

### 5.4 Cơ Chế `is_censored_observation` (Dữ Liệu Bị Che - Che Giấu)

**Định nghĩa:**
```sql
is_censored_observation = stockout_flag
-- Khi stockout_flag = true, dữ liệu bị "che" (censored)
-- Observed sales không phản ánh đúng true demand
```

**Cơ chế hoạt động:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DỮ LIỆU BỊ CHE (CENSORED)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Khách hàng muốn mua: 10 cái bánh                                   │
│         │                                                           │
│         ▼                                                           │
│  Cửa hàng còn: 0 cái (stockout)                                    │
│         │                                                           │
│         ▼                                                           │
│  Observed sales ghi lại: 0 cái                                     │
│         │                                                           │
│         ▼                                                           │
│  is_censored_observation = true                                    │
│         │                                                           │
│         ▼                                                           │
│  ❌ KHÔNG dùng để train model                                     │
│  ❌ KHÔNG phải true demand                                         │
│  ⚠️  Model phải predict/extrapolate từ baseline                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    DỮ LIỆU KHÔNG BỊ CHE (NON-CENSORED)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Khách hàng muốn mua: 10 cái bánh                                   │
│         │                                                           │
│         ▼                                                           │
│  Cửa hàng còn: 15 cái (đủ hàng)                                    │
│         │                                                           │
│         ▼                                                           │
│  Observed sales ghi lại: 10 cái                                    │
│         │                                                           │
│         ▼                                                           │
│  is_censored_observation = false                                   │
│         │                                                           │
│         ▼                                                           │
│  ✅ Dùng để train model                                            │
│  ✅ True demand = Observed sales                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Sử dụng trong hệ thống:**

| Thành phần | Cách sử dụng `is_censored_observation` | Lý do |
|-----------|----------------------------------------|-------|
| **Training** | Chỉ dùng rows có `is_censored_observation = false` | Đảm bảo target là true demand đáng tin cậy |
| **Feature Engineering** | Tính historical averages từ non-censored rows | Baseline cần dữ liệu không bị che |
| **Heuristic Fallback** | Áp dụng cho stockout rows: `estimated = MAX(observed, baseline)` | Bù đắp phần demand bị mất |
| **Model Evaluation** | Không đánh giá trên censored rows | Không có ground truth |
| **DSS Dashboard** | Hiển thị warning: "Data may be censored" | Cảnh báo ngườ dùng |

**Công thức đầy đủ:**

```sql
-- Bước 1: Tạo flag từ dữ liệu gốc
is_censored_observation := hours_stock_status[hour] = 1

-- Bước 2: Đánh dấu dữ liệu trainable
is_trainable_demand_observation := NOT is_censored_observation

-- Bước 3: Target chỉ lấy từ non-censored
target_observed_sales_amount := 
    CASE WHEN NOT is_censored_observation 
    THEN observed_sales_amount 
    END

-- Bước 4: Baseline từ non-censored historical
store_product_hour_avg := 
    AVG(observed_sales_amount) FILTER (WHERE NOT is_censored_observation)
    OVER (PARTITION BY store, product, hour)

-- Bước 5: Extrapolate cho censored rows
estimated_true_demand := 
    CASE WHEN is_censored_observation 
    THEN GREATEST(observed_sales_amount, store_product_hour_avg, ...)
    ELSE observed_sales_amount 
    END
```

**Ví dụ thực tế:**

```
Store 537, Product 559, Giờ 15:00:

Ngày 2024-06-25 (không stockout):
  - is_censored_observation = false
  - observed_sales = 0.15
  - → Train model: "Giờ 15h, thứ 3, store 537 → demand ~0.15"

Ngày 2024-06-30 (stockout):
  - is_censored_observation = true
  - observed_sales = 0.0
  - → KHÔNG train model
  - → Model predict: 0.15 (từ historical)
  - → estimated_true_demand = 0.15
  - → estimated_lost_sales = 0.15 - 0 = 0.15
```

### 5.5 Công Thức Điểm DSS

```sql
-- Restock Urgency Score (Điểm ưu tiên restock):
restock_urgency_score = LEAST(1,
    stockout_rate_6_22 * 0.55
  + demand_bias_rate  * 0.35
  + CASE WHEN activity_flag <> 0 THEN 0.10 ELSE 0 END
)

-- Stockout Risk Score (Điểm rủi ro stockout):
stockout_risk_score = LEAST(1,
    (stockout_hours_6_22 / 16) * 0.55
  + (expected_lost_sales / NULLIF(expected_demand, 0)) * 0.35
  + CASE WHEN activity_flag <> 0 THEN 0.10 ELSE 0 END
)

-- Demand Bias Rate (Tỷ lệ demand bị mất):
demand_bias_rate = estimated_lost_sales / NULLIF(estimated_true_demand, 0)
```

**Phân tích trọng số Urgency Score:**

| Thành phần | Trọng số | Ý nghĩa | Khi nào cao? |
|-----------|----------|---------|--------------|
| `stockout_rate_6_22` | 0.55 (55%) | Tỷ lệ giờ stockout trong business hours | Cửa hàng hết hàng nhiều giờ |
| `demand_bias_rate` | 0.35 (35%) | Tỷ lệ demand bị mất | Demand bị che nhiều |
| `activity_flag` | 0.10 (10%) | Có khuyến mãi/hoạt động | Đang có promotion |

**Ví dụ tính Urgency:**

```
Ngày 2024-06-30:
  stockout_rate_6_22 = 0.625 (62.5%)
  demand_bias_rate   = 0.693 (69.3%)
  activity_flag      = 1 (có khuyến mãi)
  
  urgency = 0.625 * 0.55 + 0.693 * 0.35 + 1 * 0.10
         = 0.3438 + 0.2426 + 0.10
         = 0.6864
         
  0.6864 >= 0.65 → RESTOCK IMMEDIATELY ✓
```

---

## 6. Mô Hình Dự Báo Demand

### 6.1 Model Overview

```
Model Name: XGBoost Hurdle Model
Training Data: Chỉ non-stockout observations từ train.parquet
Target: observed_sales_amount (đáng tin cậy vì không stockout)
```

### 6.2 Two-Stage Hurdle Strategy

**Stage 1: Classification**
- XGBClassifier dự đoán P(demand > 0)
- Học xác suất có demand hay không

**Stage 2: Regression**
- XGBRegressor dự đoán log1p(demand | demand > 0)
- Học magnitude của demand (log-transformed)

**Final Prediction:**
```python
estimated_demand = P(demand > 0) × E[demand | demand > 0]
# Kết hợp probability và expected amount
```

### 6.3 Features Sử Dụng

**Base Features:**
- `hour_of_day`, `is_business_hour`, `day_of_week`, `is_weekend`
- `holiday_flag`, `discount_rate`
- Weather: `precpt`, `avg_temperature`, `avg_humidity`, `avg_wind_level`
- Store & Product IDs: `store_id`, `product_id`
- Category hierarchy: `management_group_id`, `first/second/third_category_id`

**Historical Prior Features (train only):**
- Store-product-hour mean
- Product-hour mean
- Category-hour mean
- Product/day-of-week means
- Stockout rate priors

**Temporal Features:**
- `product_sales_lag_1h`
- `days_since_previous_sale`
- Cyclic encoding: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`

### 6.4 Model Quality Metrics (Latest Model)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| WMAPE | 94.76% | High error - expected for fresh retail |
| Bias | -6.96% | Slight underprediction |
| Calibration Factor | 0.921 | Predictions scaled up ~8% |
| MAE | 0.062 | Per-hour absolute error |
| RMSE | 0.113 | Per-hour squared error |

**Segment Performance:**
| Segment | WMAPE | Bias |
|---------|-------|------|
| Business hours (6-22h) | 91.3% | -7.7% |
| Non-business hours | 170.7% | - |
| Weekend | 88.1% | -16.3% |

---

## 7. Luồng Xử Lý Dữ Liệu

### 7.1 ETL Pipeline

```bash
# Load demo data (1k rows per split)
python3 etl/load_fresh_retail_dw.py --reset --limit-rows-per-split 1000 --load-hourly

# Load recommended panel for model training
python3 etl/load_fresh_retail_dw.py --reset --train-limit-rows 100000 \
  --staging-sample-mode store-product-panel --panel-seed 42 --load-hourly
```

### 7.2 Training Pipeline

```bash
# Train XGBoost model and load predictions
python3 ml/train_xgboost_demand_model.py \
  --model-type xgboost \
  --model-strategy hurdle \
  --model-name xgboost_demand_m1_hurdle \
  --model-version m1_panel100k_seed42_full_eval \
  --load-predictions
```

### 7.3 Data Flow

```
train.parquet / eval.parquet
         │
         ▼
┌────────────────────────┐
│ load_fresh_retail_dw   │
│ - Parse hourly arrays │
│ - Expand to hourly rows│
│ - Populate staging.*  │
│ - Populate dw.*       │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│ train_xgboost_demand   │
│ - Filter non-stockout   │
│ - Feature engineering  │
│ - Train hurdle model   │
│ - Export model .pkl    │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│ import_cloud_predict   │
│ - Load model .pkl      │
│ - Predict on eval rows │
│ - Populate fact_demand_│
│   estimate_hourly      │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│ DSS Dashboard          │
│ - Query views          │
│ - Display KPIs         │
│ - Show charts          │
└────────────────────────┘
```

---

## 8. Dashboard DSS - Hiển Thị Các Chỉ Số

### 8.1 Executive KPIs (Command Center Tab)

| Chỉ Số | Nguồn | Mô Tả |
|--------|-------|--------|
| **Observed Sales** | `observed_daily_sales_amount` | Sales thực tế quan sát được |
| **Estimated Demand** | `SUM(estimated_true_demand)` | Tổng demand ước tính (model) |
| **Estimated Lost Sales** | `SUM(estimated_lost_sales)` | Tổng sales bị mất do stockout |
| **Lost Demand Share** | `AVG(demand_bias_rate)` | Tỷ lệ demand bị mất |

### 8.2 Hourly Drill-Down Tab

**Chart hiển thị:**
- Line: Observed Sales theo giờ
- Line: Estimated Demand theo giờ
- Bars: Recovered Lost Sales (màu cam)

**24-Hour Cell Strip:**
- Đỏ = stockout
- Cam = recovered lost sales
- Xanh = normal (không stockout)

### 8.3 Ghi Chú Quan Trọng Trên Dashboard

Dashboard nên hiển thị rõ:

```
Observed Sales       = What actually sold (thực tế bán được)
Estimated Demand     = Model-recovered demand (demand model phục hồi)
Recovered Gap        = Estimated lost sales during stockout
Stockout Flag        = Binary source signal, NOT inventory quantity
```

---

## 9. Giới Hạn Của Hệ Thống

### 9.1 Giới Hạn Về Dữ Liệu

```
1. KHÔNG CÓ true demand ground truth trong stockout periods
   - Dataset chỉ ghi nhận observed sales
   - Không thể validate lost sales estimates

2. KHÔNG CÓ current stock quantity
   - FreshRetailNet không có inventory levels
   - Không biết hiện tại còn bao nhiêu units

3. KHÔNG CÓ supply chain constraints
   - Lead time
   - Shelf life / Expiry dates
   - Supplier capacity
   - Reorder constraints
```

### 9.2 Giới Hạn Về Model

```
1. Model training CHỈ trên non-stockout observations
   - Không học được behavior khi stockout
   - Demand estimates trong stockout dựa trên extrapolation

2. High WMAPE (94.76%)
   - Fresh retail demand rất volatile
   - High error expected

3. Eval only trên non-stockout rows
   - Model quality cho stockout periods không được validate
```

### 9.3 DSS Chính Xác Là Gì?

```
DSS của mình đang làm: DEMAND RECOVERY
- Ước tính demand thực sự từ observed sales bị censored
- Phục hồi phần demand bị mất do stockout

DSS KHÔNG PHẢI là: INVENTORY OPTIMIZATION
- Không tối ưu số lượng order cụ thể
- Không có stock-on-hand data
- Không có reorder point calculations
```

### 9.4 Ví Dụ Minh Họa - 2024-06-26 | Store 537 | Product 559

```
Observed sales thật trong eval: 0.90
Estimated demand (model): 1.32
Recovered lost sales: 0.42
Stockout flag: 17:00-23:00 (6 giờ)
Current stock quantity: KHÔNG CÓ

Điều có thể nói:
✓ "Cửa hàng 537, sản phẩm 559 bị stockout 17:00-23:00"
✓ "Model ước tính demand thực tế là 1.32, cao hơn observed 0.90"
✓ "Có khoảng 0.42 demand bị mất do stockout"
✓ "Nên tăng replenishment cho sản phẩm này"

Điều KHÔNG thể nói:
✗ "Hiện tại còn 5 units"
✗ "Nên order thêm 20 units"
✗ "Stock sẽ hết sau 3 giờ"
```

---

## 10. Từ Điển Dữ Liệu

### 10.1 Trường Trong FreshRetailNet (Nguồn)

| Trường | Schema | Mô Tả Chi Tiết |
|--------|--------|----------------|
| `city_id` | raw | Mã thành phố đã được encode |
| `store_id` | raw | Mã cửa hàng đã được encode |
| `management_group_id` | raw | Nhóm quản lý sản phẩm (hierarchy) |
| `first_category_id` | raw | Danh mục cấp 1 (hierarchy) |
| `second_category_id` | raw | Danh mục cấp 2 (hierarchy) |
| `third_category_id` | raw | Danh mục cấp 3 (hierarchy) |
| `product_id` | raw | Mã sản phẩm đã được encode |
| `dt` | raw | Ngày theo format YYYY-MM-DD |
| `sale_amount` | raw | Tổng doanh số trong ngày |
| `hours_sale` | raw | List 24 giá trị sales theo từng giờ |
| `stock_hour6_22_cnt` | raw | Số giờ stockout trong 6:00-21:59 (0-16) |
| `hours_stock_status` | raw | List 24 flag stockout (0 = OK, 1 = stockout) |
| `discount` | raw | Tỷ lệ discount (0.0 - 1.0) |
| `holiday_flag` | raw | 1 = ngày lễ, 0 = ngày thường |
| `activity_flag` | raw | 1 = có promotion/activity, 0 = không |
| `precpt` | raw | Lượng mưa (precipitation) |
| `avg_temperature` | raw | Nhiệt độ trung bình trong ngày |
| `avg_humidity` | raw | Độ ẩm trung bình |
| `avg_wind_level` | raw | Cấp gió trung bình |

### 10.2 Trường Trong Kho Dữ Liệu DW

| Trường | Bảng/View | Mô Tả Chi Tiết |
|--------|-----------|----------------|
| `estimated_true_demand` | fact_demand_estimate_hourly | Demand ước tính bao gồm phần phục hồi |
| `estimated_lost_sales` | fact_demand_estimate_hourly | Sales bị mất do stockout |
| `prediction_lower_bound` | fact_demand_estimate_hourly | Cận dưới khoảng dự đoán |
| `prediction_upper_bound` | fact_demand_estimate_hourly | Cận trên khoảng dự đoán |
| `is_censored_observation` | fact_demand_estimate_hourly | Observation có thể bị bias bởi stockout |
| `observed_sales_amount` | fact_sales_inventory_hourly | Sales thực tế quan sát được |
| `stockout_flag` | fact_sales_inventory_hourly | Flag có stockout không |
| `recommended_order_qty` | fact_replenishment_recommendation_daily | Số lượng order đề xuất |
| `expected_demand` | fact_replenishment_recommendation_daily | Demand kỳ vọng trong ngày |
| `expected_lost_sales` | fact_replenishment_recommendation_daily | Lost sales kỳ vọng |
| `stockout_risk_score` | fact_replenishment_recommendation_daily | Điểm rủi ro stockout (0-1) |
| `expected_waste_qty` | fact_replenishment_recommendation_daily | Lượng hàng có thể bị waste |
| `service_level_target` | fact_replenishment_recommendation_daily | Target service level |
| `demand_bias_rate` | v_dss_daily_decision_score | Tỷ lệ demand bị bias = lost/estimated |
| `stockout_rate_6_22` | v_dss_daily_decision_score | Tỷ lệ giờ stockout trong business hours |
| `waste_risk_score` | v_dss_daily_decision_score | Điểm rủi ro overstock/waste |
| `restock_urgency_score` | v_dss_daily_decision_score | Điểm urgency restock (0-1) |
| `decision_action` | v_dss_daily_decision_score | Action: Restock/Increase/Reduce/Review/Maintain |
| `decision_reason` | v_dss_daily_decision_score | Lý do chi tiết cho quyết định |

### 10.3 Định Nghĩa Các Chỉ Số

| Chỉ Số | Công Thức | Ý Nghĩa |
|--------|-----------|---------|
| `estimated_true_demand` | max(observed, model_pred) | Demand thực tế ước tính |
| `estimated_lost_sales` | max(estimated_demand - observed, 0) | Phần demand bị mất |
| `demand_bias_rate` | lost_sales / estimated_demand | Tỷ lệ demand bị mất |
| `stockout_rate_6_22` | stockout_hours / 16 | Tỷ lệ giờ stockout |
| `restock_urgency_score` | 0.55×stockout_rate + 0.35×bias_rate + 0.10×activity | Điểm ưu tiên restock |
| `calibration_factor` | actual / predicted | Hệ số hiệu chỉnh model |

---

## Phụ Lục: Cấu Trúc File

```
ubiquitous-train/
├── app/
│   └── dss_dashboard.py              # Streamlit DSS Dashboard
├── docs/
│   ├── ARCHITECTURE.md               # System architecture
│   ├── CLOUD_TRAINING.md             # GPU training guide
│   ├── EVALUATION.md                 # Evaluation methodology
│   ├── IMPLEMENTATION_LOG.md         # Implementation history
│   └── PROJECT_FLOW.md               # Project flow
├── etl/
│   └── load_fresh_retail_dw.py      # ETL pipeline
├── ml/
│   ├── cloud_gpu_train.py            # Cloud GPU training
│   ├── export_model_features.py      # Export features
│   ├── import_cloud_predictions.py    # Import predictions
│   ├── replicate_retailforecast_xgboost.py
│   └── train_xgboost_demand_model.py # XGBoost training
├── models/                            # Trained models (.pkl)
├── exports/                           # Exported parquet files
├── sql/
│   ├── 001_schema.sql                # PostgreSQL schema
│   └── 002_sample_queries.sql        # Sample queries
├── docker-compose.yml                # PostgreSQL container
├── requirements.txt                  # Python dependencies
└── README.md                         # Main documentation
```

---

## Lịch Sử Phiên Bản

| Phiên Bản | Ngày | Thay Đổi |
|-----------|------|-----------|
| 1.0 | 2026-05-15 | Tạo tài liệu ban đầu |

---

**Tài liệu này mô tả hệ thống Fresh Retail DSS và các giá trị dự đoán. Điểm quan trọng cần nhớ:**

1. **Predicted value** có nhưng là ước tính từ model
2. **Observed eval value** chỉ đáng tin cậy ở non-stockout rows
3. **True demand** không tồn tại đầy đủ trong stockout periods
4. **Current stock quantity** không có trong dataset
5. DSS đang làm **demand recovery**, **không phải inventory optimization**
