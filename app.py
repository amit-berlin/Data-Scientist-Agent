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
import torch
import torch.nn as nn
import plotly.express as px

st.set_page_config(layout="wide", page_title="DeepForecast — Upload CSV, Forecast, Analyze")

st.title("DeepForecast — Upload CSV, Forecast & Trend Analysis (Streamlit Free Ready)")
st.markdown(
    "Upload a CSV with a datetime column and one or more numeric columns. "
    "The app offers Auto-ARIMA (fast & robust) and a compact PyTorch LSTM "
    "(deep learning) option optimized for Streamlit free tier." 
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

def detect_m(index):
    if hasattr(index, 'freq') and index.freq is not None:
        freqstr = index.freqstr
        if 'D' in freqstr: return 7
        if 'M' in freqstr: return 12
        if 'H' in freqstr: return 24
    diffs = np.diff(index.astype(np.int64) // 10**9)
    if len(diffs) == 0: return 1
    median = np.median(diffs)
    if median <= 3600: return 24
    if median <= 86400: return 7
    return 12

def train_auto_arima(series, n_periods):
    model = pm.auto_arima(series, seasonal=True, m=detect_m(series.index), stepwise=True,
                          error_action='ignore', suppress_warnings=True)
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    idx = pd.date_range(series.index[-1], periods=n_periods+1, closed='right', freq=series.index.freq)
    pred = pd.Series(fc, index=idx)
    lower = pd.Series(confint[:, 0], index=idx)
    upper = pd.Series(confint[:, 1], index=idx)
    return pred, lower, upper, model

# ----------------------------- PyTorch LSTM -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm(series, n_periods, lookback=24, epochs=50, lr=0.01):
    arr = series.values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X); y = np.array(y)

    if len(X) < 10:
        raise ValueError("Not enough data for LSTM. Choose a smaller lookback or use ARIMA.")

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    last_window = scaled[-lookback:].reshape(1, lookback, 1)
    preds = []
    for _ in range(n_periods):
        inp = torch.tensor(last_window, dtype=torch.float32)
        pred = model(inp).item()
        preds.append(pred)
        last_window = np.append(last_window[:,1:,:], [[[pred]]], axis=1)

    inv_preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).ravel()
    idx = pd.date_range(series.index[-1], periods=n_periods+1, freq=series.index.freq, closed='right')
    return pd.Series(inv_preds, index=idx), model

# ----------------------------- Analysis Helpers -----------------------------
def seasonal_decompose_plot(series):
    try:
        return seasonal_decompose(series.dropna(), model='additive', period=detect_m(series.index))
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
    href = f'<a href=\"data:file/csv;base64,{b64}\" download=\"{filename}\">Download CSV</a>'
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
    n_periods = st.sidebar.number_input("Forecast periods (steps)", min_value=1, max_value=365, value=30)
    model_choice = st.sidebar.selectbox("Model", options=['Auto-ARIMA (fast)','LSTM (deep)'])
    lookback = st.sidebar.slider("LSTM lookback (timesteps)", min_value=3, max_value=120, value=24)

    try:
        series = prepare_series(df, date_col, value_col, freq=freq)
    except Exception as e:
        st.error(f"Failed to parse series: {e}")
        st.stop()

    st.subheader("Time series preview")
    st.line_chart(series)

    st.subheader("Decomposition & Anomalies")
    dec = seasonal_decompose_plot(series)
    if dec is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(pd.DataFrame({'observed':dec.observed, 'trend':dec.trend}))
        with col2:
            st.line_chart(pd.DataFrame({'seasonal':dec.seasonal, 'resid':dec.resid}))
    else:
        st.info("Could not decompose series with chosen frequency.")

    anom = detect_anomalies(series)
    if anom.any():
        st.write("Detected anomalies (red) on the chart below")
        fig = px.line(series.reset_index(), x=series.index.name, y=value_col)
        fig.add_scatter(x=series.index[anom].to_list(), y=series[anom].to_list(), mode='markers')
        st.plotly_chart(fig)

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
    st.sidebar.markdown("**Tips**:\n- Use `M` for monthly, `H` for hourly.\n- Auto-ARIMA is faster; LSTM can model nonlinear patterns.")

else:
    st.info("Upload a CSV to get started. Demo CSVs are included in this repo.")
