# World-best Streamlit Deep Forecasting Web App — Files

Below are the four files you asked for. Copy each section into its own file in your Git repo exactly as named.

---

## 1) `app.py`

```python
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
import io
import base64
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px

st.set_page_config(layout="wide", page_title="DeepForecast — Upload CSV, Forecast, Analyze")

st.title("DeepForecast — Upload CSV, Forecast & Trend Analysis (Streamlit Free Ready)")
st.markdown(
    "Upload a CSV with a datetime column and one or more numeric columns. The app offers Auto-ARIMA (fast & robust) and a compact LSTM (deep learning) option optimized for Streamlit free tier."
)

# ----------------------------- Helpers -----------------------------
@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def detect_datetime_columns(df):
    candidates = []
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c])
            # require at least 80% parsable without NaT
            if parsed.notna().mean() > 0.8:
                candidates.append(c)
        except Exception:
            continue
    return candidates


def prepare_series(df, date_col, value_col, freq=None):
    s = df[[date_col, value_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col])
    s = s.sort_values(date_col).dropna()
    s = s.set_index(date_col).asfreq(freq)
    return s[value_col]


def train_auto_arima(series, n_periods):
    # use pmdarima auto_arima (fast-ish) and predict
    model = pm.auto_arima(series, seasonal=True, m=detect_m(series.index), stepwise=True, error_action='ignore', suppress_warnings=True)
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    idx = pd.date_range(series.index[-1], periods=n_periods+1, closed='right', freq=series.index.freq)
    pred = pd.Series(fc, index=idx)
    lower = pd.Series(confint[:, 0], index=idx)
    upper = pd.Series(confint[:, 1], index=idx)
    return pred, lower, upper, model


def detect_m(index):
    # heuristic: daily data -> 7, monthly -> 12, hourly -> 24 etc
    if hasattr(index, 'freq') and index.freq is not None:
        freqstr = index.freqstr
        if 'D' in freqstr:
            return 7
        if 'M' in freqstr:
            return 12
        if 'H' in freqstr:
            return 24
    # fallback: infer via median difference
    diffs = np.diff(index.astype(np.int64) // 10**9)
    if len(diffs) == 0:
        return 1
    median = np.median(diffs)
    # seconds in a day ~86400
    if median <= 3600:
        return 24
    if median <= 86400:
        return 7
    return 12


def make_lstm_dataset(series, lookback=24):
    arr = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler


def train_lstm(series, n_periods, lookback=24, epochs=30, batch_size=16):
    # small LSTM that fits in Streamlit free
    X, y, scaler = make_lstm_dataset(series, lookback)
    if len(X) < 10:
        raise ValueError("Not enough data for LSTM. Choose a smaller lookback or use ARIMA.")
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    # forecast iteratively
    last_window = series.values[-lookback:].reshape(-1,1)
    scaled_last = scaler.transform(last_window)
    preds = []
    cur_window = scaled_last.copy()
    for _ in range(n_periods):
        x = cur_window.reshape((1, lookback, 1))
        p = model.predict(x, verbose=0)[0,0]
        preds.append(p)
        cur_window = np.vstack([cur_window[1:], [[p]]])
    preds = np.array(preds).reshape(-1,1)
    inv = scaler.inverse_transform(preds).ravel()
    idx = pd.date_range(series.index[-1], periods=n_periods+1, closed='right', freq=series.index.freq)
    return pd.Series(inv, index=idx), model


def seasonal_decompose_plot(series):
    try:
        res = seasonal_decompose(series.dropna(), model='additive', period=detect_m(series.index))
        return res
    except Exception:
        return None


def detect_anomalies(series):
    iso = IsolationForest(contamination=0.02, random_state=42)
    df = series.fillna(method='ffill').values.reshape(-1,1)
    iso.fit(df)
    is_anom = iso.predict(df) == -1
    return pd.Series(is_anom, index=series.index)


def df_to_download(df, filename="predictions.csv"):
    towrite = io.StringIO()
    df.to_csv(towrite, index=True)
    b64 = base64.b64encode(towrite.getvalue().encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# ----------------------------- UI -----------------------------

uploaded = st.file_uploader("Upload CSV", type=['csv'])

if uploaded is not None:
    df = read_csv(uploaded)
    st.sidebar.header("File preview & options")
    st.sidebar.dataframe(df.head())

    dt_cols = detect_datetime_columns(df)
    st.sidebar.markdown("**Detected datetime-like columns**")
    st.sidebar.write(dt_cols)

    date_col = st.sidebar.selectbox("Choose datetime column", options=([None] + dt_cols))
    if date_col is None:
        date_col = st.sidebar.selectbox("Or pick any column that has dates (raw)", options=df.columns)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    value_col = st.sidebar.selectbox("Choose value/target column to forecast", options=numeric_cols)

    freq = st.sidebar.selectbox("Choose frequency (leave None to infer)", options=[None,'D','H','M'])
    if freq == 'M':
        freq = 'M'

    n_periods = st.sidebar.number_input("Forecast periods (steps)", min_value=1, max_value=365, value=30)
    model_choice = st.sidebar.selectbox("Model", options=['Auto-ARIMA (fast)','LSTM (deep)'])
    lookback = st.sidebar.slider("LSTM lookback (timesteps)", min_value=3, max_value=120, value=24)

    # Prepare series
    try:
        series = prepare_series(df, date_col, value_col, freq=freq)
    except Exception as e:
        st.error(f"Failed to parse series: {e}")
        st.stop()

    st.subheader("Time series preview")
    st.line_chart(series)

    # Decompose
    st.subheader("Decomposition & Anomalies")
    dec = seasonal_decompose_plot(series)
    if dec is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(pd.DataFrame({'observed':dec.observed, 'trend':dec.trend}))
        with col2:
            st.line_chart(pd.DataFrame({'seasonal':dec.seasonal, 'resid':dec.resid}))
    else:
        st.info("Could not decompose series with chosen frequency; try another frequency or check data regularity.")

    anom = detect_anomalies(series)
    if anom.any():
        st.write("Detected anomalies (red) on the chart below")
        fig = px.line(series.reset_index(), x=series.index.name, y=value_col)
        fig.add_scatter(x=series.index[anom].to_list(), y=series[anom].to_list(), mode='markers')
        st.plotly_chart(fig)

    # Train & Forecast
    if st.button("Train & Forecast"):
        with st.spinner("Training / forecasting — this runs in your Streamlit session"):
            if model_choice == 'Auto-ARIMA (fast)':
                try:
                    pred, lower, upper, m = train_auto_arima(series, n_periods)
                    st.success("Auto-ARIMA finished")
                    df_pred = pd.DataFrame({'forecast':pred, 'lower':lower, 'upper':upper})
                    st.line_chart(pd.concat([series, df_pred['forecast']], axis=0))
                    st.write(df_pred.head())
                    st.markdown(df_to_download(df_pred, filename='auto_arima_forecast.csv'), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"ARIMA failed: {e}")
            else:
                try:
                    pred, model = train_lstm(series, n_periods, lookback=lookback, epochs=30)
                    st.success("LSTM finished")
                    df_pred = pd.DataFrame({'forecast':pred})
                    st.line_chart(pd.concat([series, df_pred['forecast']], axis=0))
                    st.write(df_pred.head())
                    st.markdown(df_to_download(df_pred, filename='lstm_forecast.csv'), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"LSTM failed: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Helpful tips**:\n- For monthly sales choose frequency `M`.\n- For hourly sensor data choose `H`.\n- Use Auto-ARIMA for short/lightweight forecasts.\n- Use LSTM for complex nonlinear patterns but keep lookback small on free-tier.")

else:
    st.info("Upload a CSV to get started. Demo CSVs are included in this repo: sales_monthly.csv, energy_hourly.csv, stock_daily.csv")

```

---

## 2) `requirements.txt`

```
streamlit==1.25.0
pandas>=1.3
numpy
scikit-learn
matplotlib
plotly
statsmodels
pmdarima
tensorflow==2.12.0
```

> Notes: versions chosen to increase compatibility on free Streamlit. If you run into install issues on the free platform, remove `tensorflow` and use only Auto-ARIMA path (very lightweight).

---

## 3) Demo data: `sales_monthly.csv`

```
date,sales
2019-01-01,120
2019-02-01,130
2019-03-01,150
2019-04-01,160
2019-05-01,155
2019-06-01,165
2019-07-01,170
2019-08-01,175
2019-09-01,180
2019-10-01,190
2019-11-01,200
2019-12-01,220
2020-01-01,125
2020-02-01,135
2020-03-01,140
2020-04-01,145
2020-05-01,150
2020-06-01,160
2020-07-01,170
2020-08-01,180
2020-09-01,185
2020-10-01,195
2020-11-01,205
2020-12-01,230
2021-01-01,130
2021-02-01,140
2021-03-01,155
2021-04-01,165
2021-05-01,158
2021-06-01,168
2021-07-01,175
2021-08-01,180
2021-09-01,185
2021-10-01,195
2021-11-01,210
2021-12-01,235
```

---

## 4) Demo data: `energy_hourly.csv`

```
datetime,consumption
2021-01-01 00:00,45
2021-01-01 01:00,43
2021-01-01 02:00,42
2021-01-01 03:00,41
2021-01-01 04:00,40
2021-01-01 05:00,42
2021-01-01 06:00,50
2021-01-01 07:00,65
2021-01-01 08:00,78
2021-01-01 09:00,85
2021-01-01 10:00,88
2021-01-01 11:00,90
2021-01-01 12:00,95
2021-01-01 13:00,92
2021-01-01 14:00,90
2021-01-01 15:00,88
2021-01-01 16:00,92
2021-01-01 17:00,100
2021-01-01 18:00,110
2021-01-01 19:00,105
2021-01-01 20:00,98
2021-01-01 21:00,90
2021-01-01 22:00,75
2021-01-01 23:00,60
2021-01-02 00:00,48
2021-01-02 01:00,46
2021-01-02 02:00,44
2021-01-02 03:00,42
2021-01-02 04:00,41
2021-01-02 05:00,43
2021-01-02 06:00,52
2021-01-02 07:00,68
2021-01-02 08:00,80
2021-01-02 09:00,87
2021-01-02 10:00,90
2021-01-02 11:00,92
2021-01-02 12:00,96
2021-01-02 13:00,93
2021-01-02 14:00,91
2021-01-02 15:00,89
2021-01-02 16:00,93
2021-01-02 17:00,101
2021-01-02 18:00,111
2021-01-02 19:00,106
2021-01-02 20:00,99
2021-01-02 21:00,91
2021-01-02 22:00,76
2021-01-02 23:00,62
```

---

## 5) Demo data: `stock_daily.csv`

```
date,close
2020-01-02,75.09
2020-01-03,74.36
2020-01-06,74.95
2020-01-07,75.8
2020-01-08,76.0
2020-01-09,77.41
2020-01-10,77.58
2020-01-13,79.24
2020-01-14,78.17
2020-01-15,77.83
2020-01-16,78.81
2020-01-17,79.68
2020-01-21,79.14
2020-01-22,79.43
2020-01-23,79.81
2020-01-24,79.58
2020-01-27,76.76
2020-01-28,80.06
2020-01-29,81.08
2020-01-30,80.97
2020-01-31,77.38
2020-02-03,77.17
2020-02-04,79.71
2020-02-05,80.36
2020-02-06,81.3
2020-02-07,80.88
2020-02-10,81.82
2020-02-11,81.88
2020-02-12,81.83
2020-02-13,81.65
```

---

### Quick deployment steps

1. Create a new Git repo. Add `app.py`, `requirements.txt`, and the three CSV files in the repo root.
2. On Streamlit Cloud (free), create a new app pointing at the Git repo and branch. Streamlit will install requirements and run `streamlit run app.py` automatically.
3. If deployment hits resource/timeout issues on Streamlit free: remove `tensorflow` from `requirements.txt` and use the Auto-ARIMA path only (still powerful for many problems).

---

If you want, I can now:

* produce a `README.md` with detailed instructions and badges,
* create a GitHub Actions workflow to run basic tests,
* or convert this into a single ZIP file with the files prepared for direct upload.

Tell me which of those you'd like next and I'll include it directly.
