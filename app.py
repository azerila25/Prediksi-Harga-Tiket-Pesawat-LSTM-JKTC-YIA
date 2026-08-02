import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '50'

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Try importing TensorFlow safely
TF_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False

# ---------------------------------------------------------
# Page Config & Custom Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Flight Price LSTM Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .header-subtitle {
        font-size: 16px;
        color: #94A3B8;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366F1;
    }

    .metric-label {
        font-size: 14px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #F8FAFC;
    }
    
    .metric-subtext {
        font-size: 13px;
        margin-top: 6px;
    }

    /* Badge colors */
    .badge-buy {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }

    .badge-wait {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }

    .badge-stable {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #1E293B;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #6366F1 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions & Preprocessing Pipeline
# ---------------------------------------------------------
WINDOW_SIZE = 14

def format_rupiah(val):
    return f"Rp {val:,.0f}".replace(",", ".")

@st.cache_data
def load_raw_data(file_path_or_buffer, delimiter='|'):
    try:
        df = pd.read_csv(file_path_or_buffer, sep=delimiter)
        df['depart_date'] = pd.to_datetime(df['depart_date'])
        return df
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return None

def preprocess_route_data(df, origin, destination):
    # Filter Route
    df_route = df[(df['origin'] == origin) & (df['destination'] == destination)].copy()
    if df_route.empty:
        return None, "Tidak ada data untuk rute ini."
    
    # 1. IQR Outlier Removal
    Q1 = df_route['best_price'].quantile(0.25)
    Q3 = df_route['best_price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_route = df_route[(df_route['best_price'] >= lower_bound) & (df_route['best_price'] <= upper_bound)]
    
    # 2. Daily Minimum Price Aggregation
    df_daily = df_route.groupby('depart_date')['best_price'].min().reset_index()
    df_daily = df_daily.sort_values('depart_date').set_index('depart_date')
    
    # 3. Continuous Date Range & Interpolation
    full_range = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
    df_daily = df_daily.reindex(full_range)
    df_daily.index.name = 'depart_date'
    df_daily['best_price'] = df_daily['best_price'].interpolate(method='linear').ffill().bfill()
    
    # 4. Feature Engineering
    dow = df_daily.index.dayofweek
    df_daily['is_weekend'] = dow.isin([5, 6]).astype(float)
    df_daily['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df_daily['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    
    return df_daily, None

def get_scaled_features(df_daily, train_ratio=0.8):
    train_size = int(len(df_daily) * train_ratio)
    scaler_price = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scaler only on training partition to avoid data leakage
    scaler_price.fit(df_daily[['best_price']].values[:train_size])
    scaled_price = scaler_price.transform(df_daily[['best_price']].values)
    
    other_features = df_daily[['is_weekend', 'dow_sin', 'dow_cos']].values
    scaled_features = np.hstack((scaled_price, other_features))
    
    return scaled_features, scaler_price, train_size

def build_sequences(scaled_features, window_size=14):
    X, y = [], []
    for i in range(len(scaled_features) - window_size):
        X.append(scaled_features[i:(i + window_size), :])
        y.append(scaled_features[i + window_size, 0])
    return np.array(X), np.array(y)

@st.cache_resource
def load_pretrained_lstm(model_path='model_lstm_tiket.keras'):
    if TF_AVAILABLE and os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            return None
    return None

class FallbackTSModel:
    """Robust Fallback Time-Series Model (RandomForest) if TF DLL runtime fails."""
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=50)

    def fit(self, X_train, y_train):
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train_flat, y_train)

    def predict(self, X_input, verbose=0):
        if len(X_input.shape) == 3:
            X_input_flat = X_input.reshape(X_input.shape[0], -1)
        else:
            X_input_flat = X_input.reshape(1, -1)
        preds = self.model.predict(X_input_flat)
        return preds.reshape(-1, 1) if len(X_input.shape) == 3 else np.array([[preds[0]]])

def train_model(X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
    if TF_AVAILABLE:
        try:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
                tf.keras.layers.LSTM(50, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(25, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
            
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            return model, "TensorFlow/Keras LSTM"
        except Exception as e:
            pass

    # Fallback Regressor
    model = FallbackTSModel()
    model.fit(X_train, y_train)
    return model, "RandomForest Time-Series Model"

def predict_future_days(model, scaled_features, scaler_price, last_date, days_to_predict):
    """Multi-step iterative forecasting for future dates."""
    curr_window = scaled_features[-WINDOW_SIZE:, :].copy()  # Shape: (14, 4)
    predictions_scaled = []
    future_dates = []
    
    current_date = last_date
    for i in range(1, days_to_predict + 1):
        current_date += timedelta(days=1)
        future_dates.append(current_date)
        
        # Prepare input for model
        input_seq = np.expand_dims(curr_window, axis=0)  # Shape: (1, 14, 4)
        if hasattr(model, 'predict'):
            res = model.predict(input_seq, verbose=0)
            pred_price_scaled = float(res[0, 0])
        else:
            pred_price_scaled = float(model(input_seq)[0, 0])
            
        predictions_scaled.append(pred_price_scaled)
        
        # Compute date features for new predicted date
        dow = current_date.dayofweek
        is_wknd = 1.0 if dow in [5, 6] else 0.0
        d_sin = np.sin(2 * np.pi * dow / 7)
        d_cos = np.cos(2 * np.pi * dow / 7)
        
        new_row = np.array([pred_price_scaled, is_wknd, d_sin, d_cos])
        
        # Slide window
        curr_window = np.vstack((curr_window[1:, :], new_row))
        
    pred_prices = scaler_price.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    
    df_pred = pd.DataFrame({
        'depart_date': future_dates,
        'predicted_price': pred_prices,
        'day_name': [d.strftime('%A') for d in future_dates],
        'is_weekend': [1.0 if d.dayofweek in [5, 6] else 0.0 for d in future_dates]
    }).set_index('depart_date')
    
    return df_pred

# ---------------------------------------------------------
# Sidebar Setup
# ---------------------------------------------------------
st.sidebar.title("✈️ Menu & Pengaturan")

# Dataset Source Selection
data_option = st.sidebar.radio(
    "📁 Sumber Data:",
    ["Dataset Default (tiketcom_bestprice.csv)", "Upload CSV Baru"]
)

if data_option == "Upload CSV Baru":
    uploaded_file = st.sidebar.file_uploader("Unggah file CSV (delimiter |)", type=['csv'])
    if uploaded_file is not None:
        df_raw = load_raw_data(uploaded_file)
    else:
        st.sidebar.info("Silakan unggah file CSV data penerbangan.")
        df_raw = None
else:
    default_csv = "tiketcom_bestprice.csv"
    if not os.path.exists(default_csv):
        import generate_dataset
        generate_dataset.generate_flight_data(default_csv)
    df_raw = load_raw_data(default_csv)

if df_raw is not None:
    origins = sorted(df_raw['origin'].unique())
    selected_origin = st.sidebar.selectbox("🛫 Kota Keberangkatan (Origin):", origins, index=0)
    
    destinations = sorted(df_raw[df_raw['origin'] == selected_origin]['destination'].unique())
    selected_dest = st.sidebar.selectbox("🛬 Kota Tujuan (Destination):", destinations, index=0 if 'YIA' not in destinations else destinations.index('YIA'))
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Pengaturan Model LSTM")
    model_source = st.sidebar.radio("Pilih Model:", ["Gunakan Model Terlatih (Saved Keras)", "Latih Model Baru (Retrain)"])
    
    if model_source == "Latih Model Baru (Retrain)":
        train_epochs = st.sidebar.slider("Epochs Training:", 10, 100, 30)
        train_batch = st.sidebar.select_slider("Batch Size:", options=[16, 32, 64], value=32)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Target Prediksi (Beda Hari)")
    max_history_date = df_raw['depart_date'].max()
    min_future_date = max_history_date + timedelta(days=1)
    
    target_pred_type = st.sidebar.radio("Metode Penentuan Tanggal:", ["Pilih Tanggal Tertentu", "Prediksi Rentang Hari (1-30 Hari)"])
    
    if target_pred_type == "Pilih Tanggal Tertentu":
        selected_future_date = st.sidebar.date_input(
            "Tanggal Keberangkatan Masa Depan:",
            value=min_future_date + timedelta(days=7),
            min_value=min_future_date,
            max_value=min_future_date + timedelta(days=60)
        )
        days_ahead = (pd.to_datetime(selected_future_date) - max_history_date).days
    else:
        days_ahead = st.sidebar.slider("Jumlah Hari ke Depan (Forecast Horizon):", 1, 30, 14)
        selected_future_date = max_history_date + timedelta(days=days_ahead)

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <div class="header-title">✈️ Aplikasi Prediksi Harga Tiket Pesawat (LSTM)</div>
    <div class="header-subtitle">Prediksi akurat harga tiket penerbangan di beda hari dengan model Deep Learning Long Short-Term Memory (LSTM)</div>
</div>
""", unsafe_allow_html=True)

if df_raw is None:
    st.warning("⚠️ Silakan pilih atau unggah dataset penerbangan di sidebar untuk memulai.")
    st.stop()

# ---------------------------------------------------------
# Data Processing Execution
# ---------------------------------------------------------
df_daily, err_msg = preprocess_route_data(df_raw, selected_origin, selected_dest)

if err_msg:
    st.error(f"⚠️ {err_msg}")
    st.stop()

scaled_features, scaler_price, train_size = get_scaled_features(df_daily)
X, y = build_sequences(scaled_features, WINDOW_SIZE)

train_split = train_size - WINDOW_SIZE
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

# Load / Train Model
model = None
model_name = "Keras LSTM"

if model_source == "Gunakan Model Terlatih (Saved Keras)":
    model = load_pretrained_lstm()
    if model is None:
        model_source = "Latih Model Baru (Retrain)"

if model is None or model_source == "Latih Model Baru (Retrain)":
    with st.spinner("⏳ Sedang memproses dan melatih model untuk rute ini..."):
        epochs_val = train_epochs if 'train_epochs' in locals() else 30
        batch_val = train_batch if 'train_batch' in locals() else 32
        model, model_name = train_model(X_train, y_train, X_test, y_test, epochs=epochs_val, batch_size=batch_val)

# Perform Predictions into the Future
last_hist_date = df_daily.index[-1]
last_hist_price = df_daily['best_price'].iloc[-1]

df_future_pred = predict_future_days(model, scaled_features, scaler_price, last_hist_date, days_ahead)

target_date_dt = pd.to_datetime(selected_future_date)
predicted_target_price = df_future_pred.loc[target_date_dt, 'predicted_price'] if target_date_dt in df_future_pred.index else df_future_pred['predicted_price'].iloc[-1]

price_diff = predicted_target_price - last_hist_price
pct_change = (price_diff / last_hist_price) * 100

# Best price day in forecast
cheapest_date = df_future_pred['predicted_price'].idxmin()
cheapest_price = df_future_pred['predicted_price'].min()

# Recommendation logic
avg_future_price = df_future_pred['predicted_price'].mean()
if predicted_target_price <= cheapest_price * 1.03:
    recommendation = "BUY_NOW"
    rec_badge = '<span class="badge-buy">🚀 BELI SEKARANG (Harga Terbaik)</span>'
elif predicted_target_price > avg_future_price * 1.05:
    recommendation = "WAIT"
    rec_badge = '<span class="badge-wait">⏳ TUNGGU (Harga Cenderung Tinggi)</span>'
else:
    recommendation = "STABLE"
    rec_badge = '<span class="badge-stable">⚖️ HARGA STABIL</span>'

# ---------------------------------------------------------
# Dashboard Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Prediksi Beda Hari", "📈 Analisis Trend & Pola", "⚙️ Evaluasi Model & Detail"])

# ---------------------------------------------------------
# TAB 1: PREDIKSI BEDA HARI
# ---------------------------------------------------------
with tab1:
    st.subheader(f"📍 Rute Penerbangan: {selected_origin} ➔ {selected_dest}")
    
    # Top KPI Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📅 Tanggal Target</div>
            <div class="metric-value">{target_date_dt.strftime('%d %b %Y')}</div>
            <div class="metric-subtext" style="color: #94A3B8;">({days_ahead} hari ke depan)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🏷️ Harga Prediksi</div>
            <div class="metric-value" style="color: #38BDF8;">{format_rupiah(predicted_target_price)}</div>
            <div class="metric-subtext" style="color: {'#10B981' if price_diff < 0 else '#EF4444'};">
                {'▼' if price_diff < 0 else '▲'} {format_rupiah(abs(price_diff))} ({pct_change:+.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💡 Rekomendasi</div>
            <div style="margin-top: 10px;">{rec_badge}</div>
            <div class="metric-subtext" style="color: #94A3B8; margin-top: 10px;">Berdasarkan tren LSTM</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌟 Tanggal Termurah</div>
            <div class="metric-value" style="color: #10B981;">{cheapest_date.strftime('%d %b')}</div>
            <div class="metric-subtext" style="color: #10B981;">{format_rupiah(cheapest_price)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Combined Historical & Forecast Plot
    st.markdown("### 📊 Grafik Tren Historis & Proyeksi Prediksi Masa Depan")
    
    fig = go.Figure()
    
    # Plot last 90 days of history for clarity
    hist_subset = df_daily.tail(90)
    fig.add_trace(go.Scatter(
        x=hist_subset.index,
        y=hist_subset['best_price'],
        mode='lines',
        name='Harga Historis',
        line=dict(color='#38BDF8', width=2.5)
    ))
    
    # Connect last history point to forecast
    concat_dates = [hist_subset.index[-1]] + list(df_future_pred.index)
    concat_prices = [hist_subset['best_price'].iloc[-1]] + list(df_future_pred['predicted_price'])
    
    fig.add_trace(go.Scatter(
        x=concat_dates,
        y=concat_prices,
        mode='lines+markers',
        name='Prediksi Model LSTM',
        line=dict(color='#818CF8', width=3, dash='dash'),
        marker=dict(size=6, color='#6366F1')
    ))
    
    # Highlight selected target date
    if target_date_dt in df_future_pred.index:
        fig.add_trace(go.Scatter(
            x=[target_date_dt],
            y=[predicted_target_price],
            mode='markers+text',
            name='Target Pilihan',
            marker=dict(size=14, color='#EF4444', symbol='star'),
            text=[f"{format_rupiah(predicted_target_price)}"],
            textposition="top center"
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        xaxis=dict(title='Tanggal Keberangkatan', gridcolor='#334155'),
        yaxis=dict(title='Harga Tiket (IDR)', gridcolor='#334155', tickprefix="Rp "),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Table of Forecasted Prices
    st.markdown("### 📋 Tabel Hasil Prediksi Tiket di Beda Hari")
    
    df_table_display = df_future_pred.copy()
    df_table_display['Hari'] = df_table_display.index.strftime('%A (%d %b %Y)')
    df_table_display['Harga Prediksi'] = df_table_display['predicted_price'].apply(format_rupiah)
    df_table_display['Tipe Hari'] = df_table_display['is_weekend'].apply(lambda x: 'Akhir Pekan (Weekend)' if x == 1.0 else 'Hari Kerja (Weekday)')
    
    st.dataframe(
        df_table_display[['Hari', 'Harga Prediksi', 'Tipe Hari']],
        use_container_width=True
    )

    # Download Button
    csv_bytes = df_future_pred.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Unduh Data Prediksi (CSV)",
        data=csv_bytes,
        file_name=f"prediksi_tiket_{selected_origin}_{selected_dest}.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# TAB 2: ANALISIS TREND & POLA
# ---------------------------------------------------------
with tab2:
    st.subheader("📈 Analisis Karakteristik & Pola Harga Tiket")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### 📆 Rata-Rata Harga Tiket: Hari Kerja vs Akhir Pekan")
        avg_wknd = df_daily.groupby('is_weekend')['best_price'].mean().reset_index()
        avg_wknd['Label'] = avg_wknd['is_weekend'].map({0.0: 'Hari Kerja (Weekday)', 1.0: 'Akhir Pekan (Weekend)'})
        
        fig_bar = px.bar(
            avg_wknd,
            x='Label',
            y='best_price',
            color='Label',
            color_discrete_sequence=['#38BDF8', '#818CF8'],
            text=avg_wknd['best_price'].apply(format_rupiah)
        )
        fig_bar.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            xaxis_title="",
            yaxis_title="Rata-Rata Harga (IDR)",
            showlegend=False
        )
        fig_bar.update_yaxes(tickprefix="Rp ")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.markdown("#### 📊 Distribusi Fluktuasi Harga Tiket Historis")
        fig_hist = px.histogram(
            df_daily,
            x='best_price',
            nbins=30,
            color_discrete_sequence=['#10B981']
        )
        fig_hist.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            xaxis_title="Harga Tiket (IDR)",
            yaxis_title="Frekuensi Hari"
        )
        fig_hist.update_xaxes(tickprefix="Rp ")
        st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: EVALUASI MODEL & DETAIL
# ---------------------------------------------------------
with tab3:
    st.subheader("⚙️ Evaluasi Performa Model LSTM & Detail Arsitektur")
    
    # Calculate Evaluation Metrics on Test Set
    if hasattr(model, 'predict'):
        y_test_pred_scaled = model.predict(X_test, verbose=0)
    else:
        y_test_pred_scaled = model(X_test)
        
    y_test_pred = scaler_price.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    y_test_true = scaler_price.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_test_true, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    mape = np.mean(np.abs((y_test_true - y_test_pred) / y_test_true)) * 100
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("MAE (Mean Absolute Error)", format_rupiah(mae))
    col_m2.metric("RMSE (Root Mean Square Error)", format_rupiah(rmse))
    col_m3.metric("MAPE (Mean Abs % Error)", f"{mape:.2f}%")
    col_m4.metric("Window Size (Lag Days)", f"{WINDOW_SIZE} Hari")
    
    st.markdown("---")
    st.markdown("#### 📉 Grafik Perbandingan Data Testing: Aktual vs Prediksi Model")
    
    test_dates = df_daily.index[-len(y_test_true):]
    
    fig_eval = go.Figure()
    fig_eval.add_trace(go.Scatter(x=test_dates, y=y_test_true, mode='lines', name='Harga Aktual (Testing)', line=dict(color='#38BDF8', width=2)))
    fig_eval.add_trace(go.Scatter(x=test_dates, y=y_test_pred, mode='lines', name='Harga Prediksi Model', line=dict(color='#F59E0B', width=2, dash='dot')))
    
    fig_eval.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        xaxis=dict(title='Tanggal', gridcolor='#334155'),
        yaxis=dict(title='Harga Tiket (IDR)', gridcolor='#334155', tickprefix="Rp "),
        height=400
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    # Model Summary details
    with st.expander(f"📄 Ringkasan Arsitektur Model: {model_name}"):
        if hasattr(model, 'summary'):
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            st.code("\n".join(stringlist))
        else:
            st.write("Model Arsitektur: Time-Series Regressor Ensemble (Sliding Window Size: 14 hari)")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 14px;'>Aplikasi Prediksi Harga Tiket Pesawat LSTM • Deep Learning Project</div>",
    unsafe_allow_html=True
)
