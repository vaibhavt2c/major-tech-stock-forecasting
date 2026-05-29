# =============================================================================
# MAJOR TECH STOCK FORECASTING USING MACHINE LEARNING & TIME SERIES IN R
# FINAL FIXED + STABLE + OPTIMIZED VERSION
# =============================================================================

# =============================================================================
# FEATURES
# =============================================================================
# ✔ Stable Graphics Engine
# ✔ Automatic Plot Export
# ✔ Advanced Visualizations
# ✔ Optimized ML Pipeline
# ✔ Fixed XGBoost API
# ✔ Safe best_iteration Handling
# ✔ Fixed Graphics Device Errors
# ✔ Residual Diagnostics Export
# ✔ Portfolio-Ready Structure
# =============================================================================


# -----------------------------------------------------------------------------
# 1. PACKAGE INSTALLATION & LOADING
# -----------------------------------------------------------------------------

required_packages <- c(
  "data.table",
  "tidyverse",
  "lubridate",
  "forecast",
  "tseries",
  "randomForest",
  "xgboost",
  "caret",
  "Metrics",
  "zoo",
  "corrplot",
  "PerformanceAnalytics",
  "scales",
  "viridis",
  "patchwork",
  "reshape2"
)

missing_packages <- required_packages[
  !(required_packages %in% installed.packages()[, "Package"])
]

if(length(missing_packages) > 0){
  
  install.packages(
    missing_packages,
    dependencies = TRUE
  )
}

invisible(
  lapply(
    required_packages,
    library,
    character.only = TRUE
  )
)

set.seed(42)

cat("\nAll packages loaded successfully.\n")


# -----------------------------------------------------------------------------
# 2. RESET GRAPHICS DEVICES
# -----------------------------------------------------------------------------

try({
  
  while (!is.null(dev.list())) {
    dev.off()
  }
  
}, silent = TRUE)

graphics.off()

cat("\nGraphics devices reset successfully.\n")


# -----------------------------------------------------------------------------
# 3. GLOBAL THEME
# -----------------------------------------------------------------------------

theme_set(
  
  theme_minimal(base_size = 13) +
    
    theme(
      
      plot.title = element_text(
        size = 16,
        face = "bold",
        hjust = 0.5
      ),
      
      axis.title = element_text(
        face = "bold"
      ),
      
      legend.position = "bottom",
      
      panel.grid.minor = element_blank()
      
    )
)


# -----------------------------------------------------------------------------
# 4. CREATE OUTPUT FOLDERS
# -----------------------------------------------------------------------------

dirs <- c(
  "plots",
  "results",
  "diagnostics"
)

for(d in dirs){
  
  if(!dir.exists(d)){
    dir.create(d)
  }
}


# -----------------------------------------------------------------------------
# 5. LOAD DATASET
# -----------------------------------------------------------------------------

file_path <- "major-tech-stock-2019-2024.csv"

if(!file.exists(file_path)){
  
  stop(
    paste(
      "Dataset file not found:",
      file_path
    )
  )
}

stocks <- fread(file_path)

required_cols <- c(
  "Date",
  "Ticker",
  "Open",
  "High",
  "Low",
  "Close",
  "Volume"
)

missing_cols <- setdiff(
  required_cols,
  names(stocks)
)

if(length(missing_cols) > 0){
  
  stop(
    paste(
      "Missing columns:",
      paste(missing_cols, collapse = ", ")
    )
  )
}

if("Adj Close" %in% names(stocks)){
  
  stocks[, `Adj Close` := NULL]
}

stocks[, Date := as.Date(Date)]

stocks <- unique(stocks)

setorder(stocks, Ticker, Date)

cat("\nDataset Loaded Successfully.\n")


# -----------------------------------------------------------------------------
# 6. DATA SUMMARY
# -----------------------------------------------------------------------------

cat("\n====================================\n")
cat("DATA SUMMARY\n")
cat("====================================\n")

cat("\nRows:", nrow(stocks))
cat("\nColumns:", ncol(stocks))

cat("\n\nTickers:\n")

print(unique(stocks$Ticker))

cat("\nDate Range:\n")

print(range(stocks$Date))

cat("\nMissing Values:\n")

print(colSums(is.na(stocks)))


# -----------------------------------------------------------------------------
# 7. FEATURE ENGINEERING
# -----------------------------------------------------------------------------

feature_engineering <- function(dt){
  
  dt[, Daily_Return :=
       (Close - shift(Close)) / shift(Close),
     by = Ticker]
  
  lag_days <- c(1,2,3,7)
  
  for(i in lag_days){
    
    dt[, paste0("Lag_", i) :=
         shift(Close, i),
       by = Ticker]
  }
  
  ma_windows <- c(5,10,20,30)
  
  for(w in ma_windows){
    
    dt[, paste0("MA_", w) :=
         zoo::rollmean(
           Close,
           w,
           fill = NA,
           align = "right"
         ),
       by = Ticker]
  }
  
  dt[, Volatility_5 :=
       zoo::rollapply(
         Daily_Return,
         width = 5,
         FUN = sd,
         fill = NA,
         align = "right"
       ),
     by = Ticker]
  
  return(dt)
}

stocks <- feature_engineering(stocks)

stocks <- na.omit(stocks)

cat("\nFeature Engineering Completed.\n")


# -----------------------------------------------------------------------------
# 8. VISUALIZATIONS
# -----------------------------------------------------------------------------

cat("\nGenerating Visualizations...\n")


# -----------------------------------------------------------------------------
# STOCK TREND PLOT
# -----------------------------------------------------------------------------

p_trend <- ggplot(
  stocks,
  aes(Date, Close, color = Ticker)
) +
  
  geom_line(linewidth = 0.7) +
  
  scale_y_continuous(
    labels = scales::dollar
  ) +
  
  labs(
    title = "Major Tech Stock Price Trends",
    x = "Date",
    y = "Closing Price"
  )

ggsave(
  "plots/stock_price_trends.png",
  plot = p_trend,
  width = 12,
  height = 7,
  dpi = 300
)


# -----------------------------------------------------------------------------
# MOVING AVERAGE PLOT
# -----------------------------------------------------------------------------

p_ma <- ggplot(
  stocks[Ticker %in% unique(stocks$Ticker)[1:4]],
  aes(Date)
) +
  
  geom_line(
    aes(y = Close, color = "Actual"),
    alpha = 0.5
  ) +
  
  geom_line(
    aes(y = MA_30, color = "30-Day MA"),
    linewidth = 1
  ) +
  
  facet_wrap(
    ~Ticker,
    scales = "free_y"
  ) +
  
  labs(
    title = "Price vs 30-Day Moving Average",
    y = "Price",
    color = ""
  )

ggsave(
  "plots/moving_average_plot.png",
  plot = p_ma,
  width = 12,
  height = 7,
  dpi = 300
)


# -----------------------------------------------------------------------------
# VOLATILITY PLOT
# -----------------------------------------------------------------------------

p_volatility <- ggplot(
  stocks,
  aes(Date, Volatility_5, color = Ticker)
) +
  
  geom_line(alpha = 0.8) +
  
  labs(
    title = "Rolling 5-Day Volatility",
    y = "Volatility"
  )

ggsave(
  "plots/volatility_plot.png",
  plot = p_volatility,
  width = 12,
  height = 7,
  dpi = 300
)


# -----------------------------------------------------------------------------
# CORRELATION MATRIX
# -----------------------------------------------------------------------------

png(
  "plots/correlation_matrix.png",
  width = 1400,
  height = 1000,
  res = 150
)

numeric_cols <- stocks %>%
  select(where(is.numeric))

corr_matrix <- cor(
  numeric_cols,
  use = "complete.obs"
)

corrplot(
  corr_matrix,
  method = "color",
  type = "upper",
  addCoef.col = "black",
  tl.cex = 0.7,
  number.cex = 0.5,
  col = viridis::viridis(200)
)

try(dev.off(), silent = TRUE)


# -----------------------------------------------------------------------------
# TIME SERIES DECOMPOSITION
# -----------------------------------------------------------------------------

selected_stock <- stocks[
  Ticker == unique(Ticker)[1]
]

ts_data <- ts(
  selected_stock$Close,
  frequency = 252
)

png(
  "plots/time_series_decomposition.png",
  width = 1400,
  height = 1000,
  res = 150
)

decomp <- stl(
  ts_data,
  s.window = "periodic"
)

plot(
  decomp,
  main = paste(
    "Time Series Decomposition:",
    unique(selected_stock$Ticker)
  )
)

try(dev.off(), silent = TRUE)

cat("\nVisualizations Exported Successfully.\n")


# -----------------------------------------------------------------------------
# 9. TRAIN / TEST / HOLDOUT SPLIT
# -----------------------------------------------------------------------------

split_data <- function(dt){
  
  train_list <- list()
  test_list <- list()
  holdout_list <- list()
  
  tickers <- unique(dt$Ticker)
  
  for(t in tickers){
    
    temp <- dt[Ticker == t]
    
    n <- nrow(temp)
    
    train_end <- floor(0.70 * n)
    test_end <- floor(0.85 * n)
    
    train_list[[t]] <- temp[1:train_end]
    
    test_list[[t]] <- temp[
      (train_end + 1):test_end
    ]
    
    holdout_list[[t]] <- temp[
      (test_end + 1):n
    ]
  }
  
  train <- rbindlist(train_list)
  test <- rbindlist(test_list)
  holdout <- rbindlist(holdout_list)
  
  return(list(
    train = train,
    test = test,
    holdout = holdout
  ))
}

splits <- split_data(stocks)

train_data <- splits$train
test_data <- splits$test
holdout_data <- splits$holdout

cat("\nData Split Completed.\n")


# -----------------------------------------------------------------------------
# 10. BASELINE MODEL
# -----------------------------------------------------------------------------

baseline_pred <- shift(test_data$Close)

baseline_results <- na.omit(
  
  data.table(
    Actual = test_data$Close,
    Predicted = baseline_pred
  )
)

baseline_rmse <- rmse(
  baseline_results$Actual,
  baseline_results$Predicted
)

cat("\nBaseline RMSE:", baseline_rmse, "\n")


# -----------------------------------------------------------------------------
# 11. ARIMA MODEL
# -----------------------------------------------------------------------------

selected_ticker <- unique(train_data$Ticker)[1]

arima_train <- train_data[
  Ticker == selected_ticker
]

arima_test <- test_data[
  Ticker == selected_ticker
]

adf_result <- adf.test(
  arima_train$Close
)

cat("\nADF Test Results:\n")

print(adf_result)

arima_model <- auto.arima(
  arima_train$Close
)

forecast_values <- forecast(
  arima_model,
  h = nrow(arima_test)
)

arima_pred <- as.numeric(
  forecast_values$mean
)

arima_rmse <- rmse(
  arima_test$Close,
  arima_pred
)

cat("\nARIMA RMSE:", arima_rmse, "\n")


# -----------------------------------------------------------------------------
# RANDOM FOREST MODEL
# -----------------------------------------------------------------------------

rf_train <- copy(train_data)
rf_test <- copy(test_data)

rf_train[, Date := NULL]
rf_test[, Date := NULL]

rf_train[, Ticker := as.factor(Ticker)]

rf_test[, Ticker := factor(
  Ticker,
  levels = levels(rf_train$Ticker)
)]

rf_model <- randomForest(
  Close ~ .,
  data = rf_train,
  ntree = 300,
  importance = TRUE
)

rf_pred <- predict(
  rf_model,
  rf_test
)

rf_rmse <- rmse(
  rf_test$Close,
  rf_pred
)

cat("\nRandom Forest RMSE:", rf_rmse, "\n")


# -----------------------------------------------------------------------------
# XGBOOST MODEL
# -----------------------------------------------------------------------------

ticker_levels <- sort(
  unique(stocks$Ticker)
)

prepare_xgb_matrix <- function(dt){
  
  dt <- copy(dt)
  
  dt[, Date := NULL]
  
  dt[, Ticker :=
       factor(
         Ticker,
         levels = ticker_levels
       )]
  
  y <- dt$Close
  
  dt[, Close := NULL]
  
  x <- model.matrix(
    ~ . -1,
    data = dt
  )
  
  return(list(
    x = x,
    y = y
  ))
}

xgb_train <- prepare_xgb_matrix(train_data)
xgb_test <- prepare_xgb_matrix(test_data)

dtrain <- xgb.DMatrix(
  data = xgb_train$x,
  label = xgb_train$y
)

dtest <- xgb.DMatrix(
  data = xgb_test$x,
  label = xgb_test$y
)

params <- list(
  
  objective = "reg:squarederror",
  
  eval_metric = "rmse",
  
  eta = 0.05,
  
  max_depth = 6,
  
  subsample = 0.8,
  
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  
  params = params,
  
  data = dtrain,
  
  nrounds = 300,
  
  evals = list(
    train = dtrain,
    eval = dtest
  ),
  
  early_stopping_rounds = 20,
  
  maximize = FALSE,
  
  verbose = 1
)

xgb_pred <- predict(
  xgb_model,
  dtest
)

xgb_rmse <- rmse(
  xgb_test$y,
  xgb_pred
)

cat("\nXGBoost RMSE:", xgb_rmse, "\n")


# -----------------------------------------------------------------------------
# SAFE BEST ITERATION HANDLING
# -----------------------------------------------------------------------------

best_nrounds <- xgb_model$best_iteration

if(is.null(best_nrounds) || length(best_nrounds) == 0){
  
  best_nrounds <- 300
  
  cat("\nUsing fallback nrounds = 300\n")
}


# -----------------------------------------------------------------------------
# FINAL HOLDOUT MODEL
# -----------------------------------------------------------------------------

full_train <- rbind(
  train_data,
  test_data
)

xgb_full <- prepare_xgb_matrix(full_train)
xgb_hold <- prepare_xgb_matrix(holdout_data)

dtrain_full <- xgb.DMatrix(
  data = xgb_full$x,
  label = xgb_full$y
)

dhold <- xgb.DMatrix(
  data = xgb_hold$x,
  label = xgb_hold$y
)

final_model <- xgb.train(
  
  params = params,
  
  data = dtrain_full,
  
  nrounds = best_nrounds,
  
  verbose = 0
)

holdout_pred <- predict(
  final_model,
  dhold
)

final_rmse <- rmse(
  xgb_hold$y,
  holdout_pred
)

final_mae <- mae(
  xgb_hold$y,
  holdout_pred
)

final_mape <- mape(
  xgb_hold$y,
  holdout_pred
)

cat("\n====================================\n")
cat("FINAL HOLDOUT RESULTS\n")
cat("====================================\n")

cat("RMSE :", round(final_rmse,4), "\n")
cat("MAE  :", round(final_mae,4), "\n")
cat("MAPE :", round(final_mape,4), "\n")


# -----------------------------------------------------------------------------
# SAVE FINAL RESULTS
# -----------------------------------------------------------------------------

final_results <- data.frame(
  
  Actual = xgb_hold$y,
  
  Predicted = holdout_pred
)

write.csv(
  final_results,
  "results/final_holdout_predictions.csv",
  row.names = FALSE
)

cat("\nFinal Results Saved Successfully.\n")


# -----------------------------------------------------------------------------
# RESIDUAL DIAGNOSTICS
# -----------------------------------------------------------------------------

xgb_residuals <- xgb_test$y - xgb_pred

pdf(
  "diagnostics/xgb_residual_diagnostics.pdf",
  width = 10,
  height = 8
)

par(mfrow = c(2,2))

plot(
  xgb_residuals,
  type = "l",
  col = "steelblue",
  main = "Residual Sequence"
)

hist(
  xgb_residuals,
  col = "tomato",
  main = "Residual Histogram"
)

qqnorm(xgb_residuals)

qqline(
  xgb_residuals,
  col = "red"
)

acf(
  xgb_residuals,
  main = "Residual ACF"
)

par(mfrow = c(1,1))

try(dev.off(), silent = TRUE)


# -----------------------------------------------------------------------------
# PROJECT SUMMARY
# -----------------------------------------------------------------------------

summary_table <- data.frame(
  
  Model = c(
    "Baseline",
    "ARIMA",
    "Random Forest",
    "XGBoost",
    "Final Holdout"
  ),
  
  RMSE = c(
    baseline_rmse,
    arima_rmse,
    rf_rmse,
    xgb_rmse,
    final_rmse
  )
)

cat("\n====================================\n")
cat("PROJECT SUMMARY\n")
cat("====================================\n")

print(summary_table)

best_model <- summary_table[
  which.min(summary_table$RMSE),
]

cat("\nBest Model:\n")

print(best_model)

cat("\nProject Completed Successfully.\n")


# -----------------------------------------------------------------------------
# CLEAN MEMORY
# -----------------------------------------------------------------------------

rm(
  dtrain,
  dtest,
  dhold,
  dtrain_full
)

gc()

# =============================================================================
# END OF SCRIPT
# =============================================================================