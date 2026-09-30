import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go
import requests

# ==========================================
# KONFIGURASI HALAMAN & SECRETS
# ==========================================
st.set_page_config(page_title="IDX Stock Prediction Web", page_icon="📈", layout="wide")

BOT_TOKEN = st.secrets.get("BOT_TOKEN", "8888470562:AAHowR8rtYH15by5H8hzwepz7P5aJ6Qm2yA")
CHAT_ID = st.secrets.get("CHAT_ID", "954645250")

def send_telegram_message(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=3600)
def download_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1)
        except Exception:
            df.columns = df.columns.get_level_values(0)
    return df

def recompute_indicators(data):
    df = data.copy()
    
    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Stochastic Oscillator (14, 3)
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-10))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # Lags & Volatility
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    df['Close_Lag5'] = df['Close'].shift(5)

    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()

    return df

def prepare_data(df):
    data = recompute_indicators(df)
    data['Target'] = data['Close'].shift(-1)
    data = data.replace([np.inf, -np.inf], np.nan)
    data.dropna(inplace=True)
    return data

def train_model(df):
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_10', 'SMA_50', 'EMA_20', 'RSI_14',
        'MACD', 'MACD_Signal', 'Stoch_K', 'Stoch_D',
        'Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Close_Lag5',
        'Daily_Return', 'Volatility_10'
    ]

    X = df[feature_cols]
    y = df['Target']

    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42, objective='reg:squarederror'
    )

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    y_test_arr = np.asarray(y_test, dtype=np.float64).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()
    
    valid_mask = ~np.isnan(y_test_arr) & ~np.isnan(y_pred_arr) & ~np.isinf(y_test_arr) & ~np.isinf(y_pred_arr)
    y_test_clean = y_test_arr[valid_mask]
    y_pred_clean = y_pred_arr[valid_mask]

    rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))
    mae = mean_absolute_error(y_test_clean, y_pred_clean)
    mape = mean_absolute_percentage_error(y_test_clean, y_pred_clean) * 100

    return model, X_test, y_test, y_pred, feature_cols, rmse, mae, mape

def get_mape_category(mape_value):
    """Menentukan kategori akurasi berdasarkan nilai MAPE."""
    if mape_value < 10:
        return "Sangat Akurat", "🟢"
    elif mape_value <= 20:
        return "Bagus / Akurat", "🟡"
    elif mape_value <= 50:
        return "Cukup Akurat", "🟠"
    else:
        return "Kurang Akurat", "🔴"

def forecast_future(model, df_raw, feature_cols, days=5):
    sim_df = df_raw.copy()
    future_records = []
    last_date = sim_df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days*2, freq='B')[:days]

    for next_date in future_dates:
        sim_df_updated = recompute_indicators(sim_df)
        last_features = sim_df_updated[feature_cols].iloc[[-1]]
        pred_price = float(model.predict(last_features)[0])

        future_records.append({'Tanggal': next_date.strftime('%Y-%m-%d'), 'Prediksi_Harga': pred_price})

        last_vol = float(sim_df['Volume'].iloc[-1])
        new_row = pd.DataFrame({
            'Open': [pred_price], 'High': [pred_price * 1.003], 'Low': [pred_price * 0.997],
            'Close': [pred_price], 'Volume': [last_vol]
        }, index=[next_date])

        sim_df = pd.concat([sim_df, new_row])

    return pd.DataFrame(future_records)

# UI STREAMLIT
st.title("📈 Dashboard Prediksi Saham BEI (XGBoost)")

TICKER = st.sidebar.text_input("Kode Emiten", value="BBCA.JK")
START_DATE = st.sidebar.date_input("Tanggal Mulai", value=pd.to_datetime("2020-01-01"))
END_DATE = st.sidebar.date_input("Tanggal Akhir", value=pd.to_datetime("2026-01-01"))

if st.sidebar.button("🚀 Jalankan Prediksi", type="primary"):
    with st.spinner("Memproses data & melatih model XGBoost..."):
        df_raw = download_stock_data(TICKER, START_DATE, END_DATE)

        if not df_raw.empty:
            df_processed = prepare_data(df_raw)
            
            if len(df_processed) < 50:
                st.error("Data terlalu sedikit setelah dibersihkan. Harap perluas rentang tanggal mulai/akhir.")
            else:
                model, X_test, y_test, y_pred, feature_cols, rmse, mae, mape = train_model(df_processed)
                pred_df = forecast_future(model, df_raw, feature_cols, days=5)

                last_close = float(df_raw['Close'].iloc[-1])
                last_date = df_raw.index[-1]
                target_1w = float(pred_df['Prediksi_Harga'].iloc[-1])
                pct_change_1w = ((target_1w - last_close) / last_close) * 100

                if pct_change_1w >= 1.5:
                    signal_text = "BUY / ENTRY LONG"
                    take_profit, stop_loss = target_1w, last_close * 0.98
                elif pct_change_1w <= -1.5:
                    signal_text = "SELL / WAIT & SEE"
                    take_profit, stop_loss = None, None
                else:
                    signal_text = "HOLD / NEUTRAL"
                    take_profit, stop_loss = target_1w, last_close * 0.985

                mape_cat, mape_icon = get_mape_category(mape)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Harga Terakhir", f"Rp {last_close:,.2f}")
                c2.metric("Target 1 Minggu", f"Rp {target_1w:,.2f}", f"{pct_change_1w:+.2f}%")
                c3.metric("Sinyal Posisi", signal_text)
                c4.metric("Error Model (MAPE)", f"{mape:.2f}%", f"{mape_icon} {mape_cat}", delta_color="normal")

                # Keterangan Acuan MAPE
                with st.expander("ℹ️ Info Acuan Kategori MAPE"):
                    st.markdown("""
                    - **< 10%**: **Sangat Akurat** (Kemampuan prediksi sangat tinggi)
                    - **10% - 20%**: **Bagus / Akurat** (Prediksi layak digunakan)
                    - **20% - 50%**: **Cukup Akurat** (Ekspektasi moderat)
                    - **> 50%**: **Kurang Akurat** (Model kurang direkomendasikan)
                    """)

                # Grafik Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, name='Harga Aktual (Test)', line=dict(color='#29b6f6')))
                fig.add_trace(go.Scatter(x=y_test.index, y=y_pred, name='Prediksi Model', line=dict(color='#ffa726', dash='dash')))
                fig.add_trace(go.Scatter(x=pd.to_datetime(pred_df['Tanggal']), y=pred_df['Prediksi_Harga'], name='Proyeksi 1 Minggu', line=dict(color='#ff1744', width=3), marker=dict(size=8)))
                fig.update_layout(template="plotly_dark", height=500, title=f"Proyeksi Harga {TICKER}")
                st.plotly_chart(fig, use_container_width=True)

                # Telegram Alert
                if "BUY" in signal_text and BOT_TOKEN and CHAT_ID:
                    pesan = (
                        f"🚨 *SINYAL BUY DETECTED!* 🚨\n\n"
                        f"📈 *Emiten:* `{TICKER}`\n"
                        f"💵 *Harga Terakhir:* Rp {last_close:,.2f}\n"
                        f"🎯 *Target Price (1W):* Rp {take_profit:,.2f}\n"
                        f"🛡️ *Stop Loss:* Rp {stop_loss:,.2f}\n"
                        f"📊 *Proyeksi Return:* +{pct_change_1w:.2f}%\n"
                        f"📊 *MAPE Model:* {mape:.2f}% ({mape_cat})\n"
                        f"📅 *Tanggal:* {last_date.strftime('%Y-%m-%d')}"
                    )
                    if send_telegram_message(BOT_TOKEN, CHAT_ID, pesan):
                        st.success("Notifikasi Sinyal BUY berhasil dikirim ke Telegram!")
        else:
            st.error("Gagal mengunduh data saham. Periksa kode ticker atau koneksi internet.")
