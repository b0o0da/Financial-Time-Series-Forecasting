# 📈 Financial Time Series Forecasting — LSTM vs GRU

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

A deep learning pipeline that forecasts stock closing prices using **LSTM** and **GRU** recurrent networks, with rich technical-indicator feature engineering, per-window normalization, and a side-by-side model comparison — plus an interactive Streamlit app for backtesting and next-step forecasting.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Modeling Approach](#modeling-approach)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Notes](#notes)
- [License](#license)

---

## 🔍 Overview

This project predicts the next-day **Close price** of a stock from its recent price/volume history. It builds 25 engineered features from raw OHLCV data, trains two recurrent architectures (**LSTM** and **GRU**) on 60-day sliding windows, and compares their forecasting accuracy on a held-out test set.

---

## 📊 Dataset

- **Source:** [Netflix Stock Price Prediction dataset](https://www.kaggle.com/datasets/jainilcoder/netflix-stock-price-prediction) (Kaggle) — `NFLX.csv`
- **Raw columns:** `Date, Open, High, Low, Close, Volume` (`Adj Close` dropped)
- **Split:** chronological 85% train / 15% test (no shuffling, to preserve time order)

---

## 🧪 Feature Engineering

25 features are engineered per row from the raw OHLCV data:

| Category | Features |
|---|---|
| Returns | `Return`, `LogReturn` |
| Trend | `MA7`, `MA21`, `MA50`, `EMA10` |
| Volatility | `Volatility7`, `Volatility21` |
| Momentum | `RSI14`, `MACD`, `MACD_signal` |
| Bands | `BB_width` (Bollinger Band width) |
| Lag features | `Close_lag_1`, `Close_lag_2`, `Close_lag_3` |
| Spreads | `HL_spread`, `OC_spread` |
| Volume | `Volume_change` |
| Calendar | `dow_sin`, `dow_cos` (cyclical day-of-week) |

Rows with `NaN` (from rolling/lag windows) are dropped after feature generation.

---

## 🧩 Modeling Approach

- **Sequence length:** 60-day sliding windows
- **Normalization:** *per-window* Min-Max scaling — each window is scaled using its own min/max (not a global scaler), which avoids extrapolation issues when price levels drift outside the training range
- **Target:** `Close` price, scaled the same way as the input window

---

## 🏗️ Model Architectures

**LSTM**
- `LSTM(512, return_sequences=True)` → BatchNorm → Dropout(0.3)
- `LSTM(256)` → BatchNorm → Dropout(0.3)
- `Dense(128, relu)` → `Dense(1)`
- L2 regularization on kernel & recurrent weights

**GRU**
- `GRU(128, return_sequences=True)` → BatchNorm → Dropout(0.3)
- `GRU(64)` → BatchNorm → Dropout(0.3)
- `Dense(64, relu)` → `Dense(1)`
- L2 regularization on kernel & recurrent weights

**Training config (both models):** Adam optimizer (`lr=0.001`), MSE loss, MAE metric, 100 epochs, batch size 32, `EarlyStopping` (patience 7), `ReduceLROnPlateau`, `ModelCheckpoint` saving the best weights (`best_LSTM.keras`, `best_GRU.keras`).

---

## 📈 Results

Both models are evaluated on the test set using:

- **RMSE**, **MAE**, **MAPE (%)**, **R²**

Actual vs. predicted Close price and prediction-error-over-time are plotted for both models, plus a bar chart comparing RMSE/MAE side by side.

> Exact metric values depend on the training run — see the notebook output or the Streamlit app's backtest tab for current numbers.

---

## 🖥️ Streamlit App

An interactive app (`app.py`) reproduces the notebook's exact feature engineering and per-window normalization to let you:

- Upload any OHLCV CSV in the same format as `NFLX.csv`
- Backtest LSTM and/or GRU on a configurable held-out portion of the data
- View RMSE/MAE/MAPE/R² and an actual-vs-predicted chart
- Get a **next-step Close price forecast** from each model

---

## 🚀 How to Run

### Notebook (training)
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
jupyter notebook stock-price-prediction-lstm-vs-gru.ipynb
```

### Streamlit app (inference)
```bash
pip install -r requirements.txt
streamlit run app.py
```

> The app expects `best_LSTM.keras` and `best_GRU.keras` (produced by the notebook's `ModelCheckpoint` callbacks) in the same directory as `app.py`.

---

## 📁 Repository Structure

```
Financial-Time-Series-Forecasting/
├── stock-price-prediction-lstm-vs-gru.ipynb
├── app.py
├── requirements.txt
├── best_LSTM.keras
├── best_GRU.keras
└── README.md
```

---

## 📝 Notes

- Originally built to run on Kaggle (`/kaggle/input/...` paths) — update the dataset path if running locally.
- The 60-day window means the model needs at least 70+ rows of history (after feature-engineering NaNs are dropped) to produce a forecast.
- Per-window normalization makes the model more robust to long-term price drift, at the cost of losing absolute-scale information across windows.

---

## 📄 License

This project is open-source. Add your preferred license (e.g., MIT) here.
