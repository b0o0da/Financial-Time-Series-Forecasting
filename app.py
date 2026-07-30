"""
Streamlit app — Stock Price Forecasting: LSTM vs GRU

Reproduces the exact feature engineering + per-window normalization
pipeline from `stock-price-prediction-lstm-vs-gru.ipynb`, loads the
trained `best_LSTM.keras` / `best_GRU.keras` models, and lets the user:
  - Upload OHLCV data (same format as NFLX.csv)
  - Backtest both models on a held-out portion of the data
  - Get a next-step Close price forecast from each model

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Stock Forecasting - LSTM vs GRU", page_icon="📈", layout="wide")

WINDOW = 60
MODEL_PATHS = {"LSTM": "best_LSTM.keras", "GRU": "best_GRU.keras"}


# ---------------------------------------------------------------------
# Feature engineering (identical to the training notebook)
# ---------------------------------------------------------------------
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()

    d['Return'] = d['Close'].pct_change()
    d['LogReturn'] = np.log(d['Close'] / d['Close'].shift(1))

    d['MA7'] = d['Close'].rolling(window=7).mean()
    d['MA21'] = d['Close'].rolling(window=21).mean()
    d['MA50'] = d['Close'].rolling(window=50).mean()
    d['EMA10'] = d['Close'].ewm(span=10, adjust=False).mean()

    d['Volatility7'] = d['Return'].rolling(window=7).std()
    d['Volatility21'] = d['Return'].rolling(window=21).std()

    delta = d['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    d['RSI14'] = 100 - (100 / (1 + rs))

    ema12 = d['Close'].ewm(span=12, adjust=False).mean()
    ema26 = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD'] = ema12 - ema26
    d['MACD_signal'] = d['MACD'].ewm(span=9, adjust=False).mean()

    bb_mid = d['Close'].rolling(window=20).mean()
    bb_std = d['Close'].rolling(window=20).std()
    d['BB_width'] = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / bb_mid

    for lag in [1, 2, 3]:
        d[f'Close_lag_{lag}'] = d['Close'].shift(lag)

    d['HL_spread'] = d['High'] - d['Low']
    d['OC_spread'] = d['Close'] - d['Open']
    d['Volume_change'] = d['Volume'].pct_change()

    dow = d.index.dayofweek
    d['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    d['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    return d


def create_windowed_sequences(data: np.ndarray, target_idx: int, window: int = WINDOW):
    """Per-window Min-Max normalization (each window scaled by its own min/max)."""
    X, y, t_min, t_max = [], [], [], []
    for i in range(window, len(data) + 1):
        window_data = data[i - window:i]
        w_min = window_data.min(axis=0)
        w_max = window_data.max(axis=0)
        denom = np.where(w_max - w_min == 0, 1, w_max - w_min)
        X_scaled = (window_data - w_min) / denom
        X.append(X_scaled)
        t_min.append(w_min[target_idx])
        t_max.append(w_max[target_idx])
        if i < len(data):
            y.append((data[i, target_idx] - w_min[target_idx]) / denom[target_idx])
    return np.array(X), np.array(y), np.array(t_min), np.array(t_max)


@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = load_model(path)
        except Exception as e:
            st.warning(f"⚠️ Could not load {name} model from `{path}`: {e}")
    return models


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R2": r2}


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("📈 Stock Price Forecasting — LSTM vs GRU")
st.caption("Upload OHLCV data (Date, Open, High, Low, Close, Volume) — same format as NFLX.csv")

models = load_models()

with st.sidebar:
    st.header("⚙️ Settings")
    test_pct = st.slider("Backtest split (% held out for testing)", 5, 40, 15)
    selected_models = st.multiselect("Models to run", list(MODEL_PATHS.keys()), default=list(MODEL_PATHS.keys()))

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    raw = pd.read_csv(uploaded_file)
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values('Date').drop(columns=['Adj Close'], errors='ignore').set_index('Date')

    df_feat = add_features(raw).dropna().reset_index()

    if len(df_feat) < WINDOW + 10:
        st.error(f"Not enough rows after feature engineering ({len(df_feat)}). Need at least {WINDOW + 10}.")
        st.stop()

    feature_cols = [c for c in df_feat.columns if c != 'Date']
    target_idx = feature_cols.index('Close')

    st.subheader("📊 Data Preview")
    st.dataframe(df_feat[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10), use_container_width=True)

    split_idx = int(len(df_feat) * (1 - test_pct / 100))
    test_df = df_feat.iloc[split_idx:]
    test_raw = test_df[feature_cols].values

    X_test, y_test, t_min, t_max = create_windowed_sequences(test_raw, target_idx, WINDOW)
    y_test_real = y_test * (t_max[:len(y_test)] - t_min[:len(y_test)]) + t_min[:len(y_test)]

    results = {}
    preds = {}
    for name in selected_models:
        if name not in models:
            continue
        with st.spinner(f"Running {name} predictions..."):
            pred_scaled = models[name].predict(X_test[:len(y_test)], verbose=0).ravel()
            pred_real = pred_scaled * (t_max[:len(y_test)] - t_min[:len(y_test)]) + t_min[:len(y_test)]
            preds[name] = pred_real
            results[name] = evaluate(y_test_real, pred_real)

    if results:
        st.subheader("📋 Backtest Results")
        st.dataframe(pd.DataFrame(results).T.style.format("{:.4f}"), use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(test_df['Date'].iloc[WINDOW:WINDOW + len(y_test)], y_test_real, label="Actual", linewidth=2)
        for name, pred_real in preds.items():
            ax.plot(test_df['Date'].iloc[WINDOW:WINDOW + len(y_test)], pred_real, "--", label=f"{name} Pred")
        ax.set_title("Actual vs Predicted Close Price")
        ax.set_ylabel("Close Price ($)")
        ax.legend()
        st.pyplot(fig)

        st.subheader("🔮 Next-Step Forecast")
        last_window = df_feat[feature_cols].values[-WINDOW:]
        w_min, w_max = last_window.min(axis=0), last_window.max(axis=0)
        denom = np.where(w_max - w_min == 0, 1, w_max - w_min)
        X_next = ((last_window - w_min) / denom)[np.newaxis, ...]

        cols = st.columns(len(preds))
        for col, name in zip(cols, preds.keys()):
            pred_scaled = models[name].predict(X_next, verbose=0).ravel()[0]
            pred_real = pred_scaled * denom[target_idx] + w_min[target_idx]
            col.metric(f"{name} forecast", f"${pred_real:,.2f}")
else:
    st.info("👆 Upload a CSV file to run the forecast.")
