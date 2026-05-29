# Major Tech Stock Forecasting Using Machine Learning & Time Series

**Author:** Vaibhav Tukaram Chaudhari  
**Date:** 2026-05-27

---

## Overview

An end-to-end stock price forecasting pipeline for major technology companies — **AAPL, AMZN, GOOGL, MSFT, and TSLA** — covering 2019–2023. The project compares traditional statistical methods (ARIMA) against modern machine learning algorithms (Random Forest, XGBoost).

---

## Project Structure

```
├── major-tech-stock-2019-2024.csv   # Historical daily stock data
├── n3-stock-prediction.R            # Full R script (standalone)
├── n3-stock-price.Rmd               # R Markdown report source
├── plots/                           # Auto-generated visualization outputs
├── results/                         # Model prediction CSVs
└── diagnostics/                     # Residual diagnostic plots
```

---

## Models Compared

| Model               | RMSE    |
|---------------------|---------|
| Baseline (Naïve)    | 7.2865  |
| ARIMA               | 15.5482 |
| Random Forest       | 1.9500  |
| XGBoost (Validation)| 1.4553  |
| **Final Holdout**   | **1.7467** |

**XGBoost achieved the best overall performance** with a holdout RMSE of 1.75.

---

## Features Engineered

- Lag features (1, 2, 3, 7 days)
- Simple Moving Averages (5, 10, 20, 30 days)
- Daily Returns
- 5-Day Rolling Volatility

---

## Methodology

1. **Data Ingestion & Validation** — Load and validate 6,290 rows of OHLCV data
2. **Feature Engineering** — Compute technical indicators per ticker
3. **EDA & Visualization** — Trend plots, moving averages, correlation matrix
4. **Train/Test/Holdout Split** — 70% / 15% / 15% chronological split
5. **Model Training** — Baseline → ARIMA → Random Forest → XGBoost
6. **Holdout Evaluation** — RMSE, MAE, MAPE on unseen data
7. **Residual Diagnostics** — Sequence, histogram, Q-Q plot, ACF

---

## Requirements

Install R packages with:

```r
required_packages <- c(
  "data.table", "tidyverse", "lubridate", "forecast", "tseries",
  "randomForest", "xgboost", "caret", "Metrics", "zoo",
  "corrplot", "PerformanceAnalytics", "scales", "viridis",
  "patchwork", "reshape2"
)
install.packages(required_packages, dependencies = TRUE)
```

---

## How to Run

### Option 1 — R Script
```r
source("n3-stock-prediction.R")
```

### Option 2 — R Markdown Report
```r
rmarkdown::render("n3-stock-price.Rmd")
```

Make sure `major-tech-stock-2019-2024.csv` is in the same working directory.

---

## Key Findings

- ARIMA struggled with the non-stationarity inherent in stock price data (ADF p-value = 0.61)
- Random Forest reduced RMSE from 7.29 (baseline) to 1.95 — a **73% improvement**
- XGBoost further improved accuracy with a validation RMSE of **1.4553**
- Feature engineering (lags, MAs, volatility) was critical to ML model performance
- The model generalizes well — holdout RMSE (1.75) is close to validation RMSE (1.46)

---

## Future Scope

- LSTM / GRU / Transformer architectures
- Sentiment analysis from financial news
- Walk-forward cross-validation
- SHAP explainability
- Real-time forecasting API
- Interactive Shiny dashboard
- Cloud deployment pipeline
