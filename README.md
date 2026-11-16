# Store Sales – Time Series Forecasting

Forecasting 16 days of unit sales for 1,782 store × product-family series at Corporación Favorita (Ecuador), for the Kaggle competition [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). The metric is RMSLE.

Everything is in [`store_sales.ipynb`](store_sales.ipynb): EDA, feature engineering, model comparison and the final submission.

## Approach

**Validation that mirrors the test.** The test set covers the 16 days after Tuesday 2017-08-15. The model is trained on many earlier Tuesday "forecast origins": for each one it sees history up to that day and predicts each of the next 16 days directly, with no recursion. The hold-out window is the 16 days after Tuesday 2017-07-25. All models predict `log1p(sales)`, so RMSE in log space equals RMSLE.

**Features** (49 in total, computed only from data available at the origin, plus inputs known in advance):

- **Level and trend:** rolling means over 3–112 days, an exponentially weighted mean, and the last value.
- **Weekday profile:** the mean of the same weekday over the last 4, 8 and 16 weeks.
- **Yearly seasonality:** sales 364 days before the target date, and last year's change over the same stretch.
- **Promotions** (known in advance): on the target day, over the next 16 days, and compared with the recent norm.
- **Calendar and holidays:** national, regional and local days off, with transferred holidays handled; paydays.
- **Sparsity:** share of zero days, days since the last sale.
- **Other:** store and family pooled averages, oil price, and store metadata (type, cluster, city).

## Results (hold-out RMSLE)

| Model | RMSLE |
|---|---|
| Last value | 0.653 |
| Mean of last 14 days | 0.524 |
| 14-day mean + weekday adjustment | 0.500 |
| Ridge regression | 0.475 |
| XGBoost | 0.407 |
| LightGBM | 0.406 |
| **Blend 0.6 LightGBM + 0.4 XGBoost (submitted)** | **0.405** |

The submission refits both models on all data up to 2017-08-15.

## Findings

- **Weekly cycle:** weekends sell 20–27% above average; Thursday is the low point.
- **Month start:** days 1–5 run 8–19% higher, following the month-end payday.
- **Promotions:** promoted rows sell about 1.7× more within the same family.
- **2016 earthquake:** sales rose about 67% for four weeks, so training windows overlapping it are excluded.
- **Zero-forcing didn't help:** predicting 0 for series silent for 56 days made the score slightly worse, because some come back (notably school supplies).
- **Hardest family:** SCHOOL AND OFFICE SUPPLIES (RMSLE ≈ 0.9), driven by the back-to-school surge.

## Next steps

- Add school-calendar features by region.
- Validate on several hold-out windows instead of one.
- Train per-family models or add a neural model to the ensemble.
- Use a longer training history (this run was limited by available RAM).

## Running it

1. Download the competition data from Kaggle into this folder. The data files are not included in this repo.
2. Install pandas, numpy, scikit-learn, lightgbm, xgboost, matplotlib and jupyter.
3. Run the notebook top to bottom. It takes about 10 minutes on a 16-thread CPU and writes `submission.csv`.
