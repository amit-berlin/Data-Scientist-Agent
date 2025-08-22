# app.py
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
