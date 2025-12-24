import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import os, math
from datetime import date, timedelta
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False


st.set_page_config(page_title="Aurex - Automated Foreign Exchange Forecasting", layout="wide")

CURRENCY_YAHOO = {
    "EUR/USD": "EURUSD=X",
    "JPY/USD": "JPYUSD=X",
    "CAD/USD": "CADUSD=X",
    "AUD/USD": "AUDUSD=X",
    "GBP/USD": "GBPUSD=X",
}

MODEL_PATHS = {
    ("EUR/USD", "GCN"): "models/EURUSD_baseline_model.pth",
    ("EUR/USD", "GCN + RSA"): "models/EURUSD_rsa_model.pth",
    ("JPY/USD", "GCN"): "models/JPYUSD_baseline_model.pth",
    ("JPY/USD", "GCN + RSA"): "models/JPYUSD_rsa_model.pth",
    ("GBP/USD", "GCN"): "models/GBPUSD_baseline_model.pth",
    ("GBP/USD", "GCN + RSA"): "models/GBPUSD_rsa_model.pth",
    ("CAD/USD", "GCN"): "models/CADUSD_baseline_model.pth",
    ("CAD/USD", "GCN + RSA"): "models/CADUSD_rsa_model.pth",
    ("AUD/USD", "GCN"): "models/AUDUSD_baseline_model.pth",
    ("AUD/USD", "GCN + RSA"): "models/AUDUSD_rsa_model.pth",
}

INTERVAL = {
    "Daily": "1d",
    "Weekly": "1wk",
    "Monthly": "1mo"
}

DEFAULT_FORECAST_STEPS = 5
WINDOW = 30
INPUT_DIM = 4
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.1
FEATURE_COLS = ['Open', 'High', 'Low', 'Close']
TARGET_COL = 'Close'

class GCNForecast(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = global_mean_pool(x, batch)
        out = self.lin(pooled)
        return out.view(-1)

@st.cache_data
def load_price_data(yahoo_ticker, start_date, end_date, interval="1d"):
    end_date_inclusive = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    df = yf.download(yahoo_ticker, start=start_date, end=end_date_inclusive, interval=interval, progress=False)
    if df.empty:
        return df
    if 'Volume' in df.columns:
        df = df.drop('Volume', axis=1)
    df = df.dropna().reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df['Date'] = pd.to_datetime(df['Date'])
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)

    return df

@st.cache_data(show_spinner=False)
def load_historical_data(yahoo_ticker):
    raw = yf.download(yahoo_ticker, start="2015-12-01", end="2025-11-30", progress=False)
    
    if raw.empty:
        return None, None

    raw.dropna(inplace=True)
    raw = raw.reset_index()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.dropna().reset_index(drop=True)
    
    if raw.empty: # Final check after cleanup
        return None, None

    unscaled_target_data = raw[[TARGET_COL]].copy() 
    
    scaler_feats = MinMaxScaler(feature_range=(0, 1))
    raw[FEATURE_COLS] = scaler_feats.fit_transform(raw[FEATURE_COLS])
    
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit(unscaled_target_data) 

    return raw, scaler_target


def create_graph_data(scaled_data, window=WINDOW, feature_cols=FEATURE_COLS, target_col=TARGET_COL):
    num_features = len(feature_cols)
    edge_index = torch.tensor([[i, j] for i in range(num_features) for j in range(num_features)], dtype=torch.long).t().contiguous()

    data_list = []
    for i in range(window, len(scaled_data)):
        x_window = scaled_data.loc[i-window:i-1, feature_cols].values
        x = torch.tensor(x_window, dtype=torch.float32)
        
        y = scaled_data.loc[i, target_col]
        y = torch.tensor([y], dtype=torch.float32)
        
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
        
    n = len(data_list)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, n))

    return data_list, train_indices, val_indices, test_indices

def get_data_loaders(data_list, train_idx, val_idx, test_idx, batch_size=32):
    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    test_data = [data_list[i] for i in test_idx]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def last_window_global_scaled(scaled_feats, window=WINDOW):
    last_window_data = scaled_feats.tail(window)[FEATURE_COLS].values
    
    num_features = len(FEATURE_COLS)
    edge_index = torch.tensor([[i, j] for i in range(num_features) for j in range(num_features)], dtype=torch.long).t().contiguous()
    
    data = Data(x=torch.tensor(last_window_data, dtype=torch.float), edge_index=edge_index)
    
    return DataLoader([data], batch_size=1, shuffle=False)

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        
        return model
    except FileNotFoundError:
        st.error(f"Pretrained model file not found: `{model_path}`. Please ensure the model files are in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def eval_epoch_predictions(model, loader, scaler_target):
    model.eval()
    all_preds, all_actuals = [], []
    
    with torch.no_grad():
        for data in loader:
            output = model(data.x, data.edge_index, data.batch)

            output_np = np.array(output.tolist()).reshape(-1, 1)
            preds = scaler_target.inverse_transform(output_np).flatten()

            actuals_np = np.array(data.y.tolist()).reshape(-1, 1)
            actuals = scaler_target.inverse_transform(actuals_np).flatten()
            
            all_preds.extend(preds)
            all_actuals.extend(actuals)
    
    actuals_np = np.array(all_actuals)
    preds_np = np.array(all_preds)

    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(actuals_np, preds_np)),
        'MAE': mean_absolute_error(actuals_np, preds_np),
        'R2': r2_score(actuals_np, preds_np),
        'MAPE': np.mean(np.abs((actuals_np - preds_np) / actuals_np)) * 100
    }

    return all_actuals, all_preds, metrics

def forecast_steps(model, last_window_loader, steps, scaler_target):
    """Performs multi-step forecasting using a rolling window auto-regression."""
    model.eval()
    
    initial_data = next(iter(last_window_loader))
    
    forecast_scaled = []
    current_x = initial_data.x.clone().detach() # (WINDOW, INPUT_DIM)
    
    num_features = len(FEATURE_COLS)
    edge_index = torch.tensor([[i, j] for i in range(num_features) for j in range(num_features)], dtype=torch.long).t().contiguous()

    with torch.no_grad():
        for _ in range(steps):
            forecast_data = Data(x=current_x, edge_index=edge_index)
            loader = DataLoader([forecast_data], batch_size=1) 
            batch = next(iter(loader))
            predicted_y_scaled = model(batch.x, batch.edge_index, batch.batch).item()
                
            forecast_scaled.append(predicted_y_scaled)
            
            predicted_day_feats = torch.full((1, INPUT_DIM), predicted_y_scaled, dtype=torch.float)
            
            next_x = current_x[1:, :] 
            
            current_x = torch.cat((next_x, predicted_day_feats), dim=0)

    forecast_np = np.array(forecast_scaled).reshape(-1, 1)
    forecast_unscaled = scaler_target.inverse_transform(forecast_np).flatten()
    
    return forecast_unscaled

def load_torch_model_from_path(path):
    if not TORCH_AVAILABLE:
        st.error("PyTorch is not available in this environment.")
        return None
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None
    try:
        model = torch.load(path)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def build_edge_index(window):
    # chain graph 0-1-2-...-(window-1)
    src = list(range(window - 1))
    dst = list(range(1, window))
    edge_idx = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return edge_idx

def create_sequences_from_scaled(scaled_data, window=30):
    X = []
    for i in range(len(scaled_data) - window):
        X.append(scaled_data[i:i+window])
    return np.array(X)

def predict_block_pyg(model, X_block, edge_index):
    model.eval()
    model_cpu = model.to('cpu')
    
    preds = []
    
    edge_index_cpu = edge_index.to('cpu')

    for i in range(len(X_block)):
        xi = torch.tensor(X_block[i], dtype=torch.float32) 
        
        batch_idx = torch.zeros(xi.size(0), dtype=torch.long)
        
        xi_cpu = xi.to('cpu')
        batch_idx_cpu = batch_idx.to('cpu')
        
        with torch.no_grad():
            out = model_cpu(xi_cpu, edge_index_cpu, batch_idx_cpu)
            
        val = out.detach().cpu().np() 
        
        if np.ndim(val) == 0:
            preds.append(float(val))
        else:
            preds.append(float(val.flatten()[0]))
            
    return np.array(preds)

def invert_close_predictions(scaler, scaled_preds, reference_scaled_rows):
    close_idx = FEATURE_COLS.index('Close')
    restored = []
    baseline = reference_scaled_rows[-1].copy()
    for p in scaled_preds:
        tmp = baseline.copy()
        tmp[close_idx] = p
        orig = scaler.inverse_transform([tmp])[0][close_idx]
        restored.append(orig)
    return np.array(restored)

def get_tvt_predictions(model, df, window=WINDOW):
    scaler = MinMaxScaler()
    features = df[FEATURE_COLS].values.astype(float)
    scaler.fit(features)
    scaled_all = scaler.transform(features)

    X_all = create_sequences_from_scaled(scaled_all, window)
    if len(X_all) == 0:
        return (np.array([]),)*9

    n = len(X_all)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train = X_all[:train_end]
    X_val = X_all[train_end:val_end]
    X_test = X_all[val_end:]

    edge_index = build_edge_index(window).to('cpu')

    pred_train_scaled = predict_block_pyg(model, X_train, edge_index) if len(X_train)>0 else np.array([])
    pred_val_scaled   = predict_block_pyg(model, X_val, edge_index) if len(X_val)>0 else np.array([])
    pred_test_scaled  = predict_block_pyg(model, X_test, edge_index) if len(X_test)>0 else np.array([])

    pred_train = invert_close_predictions(scaler, pred_train_scaled, scaled_all[-window:]) if len(pred_train_scaled)>0 else np.array([])
    pred_val   = invert_close_predictions(scaler, pred_val_scaled, scaled_all[-window:]) if len(pred_val_scaled)>0 else np.array([])
    pred_test  = invert_close_predictions(scaler, pred_test_scaled, scaled_all[-window:]) if len(pred_test_scaled)>0 else np.array([])

    y_all = df[TARGET_COL].values[window:]
    y_train = y_all[:train_end]
    y_val = y_all[train_end:val_end]
    y_test = y_all[val_end:]

    return (pred_train, pred_val, pred_test, y_train, y_val, y_test, train_end, val_end, scaler)

def generate_rolling_forecast(model, scaler, df, forecast_days=7, window=WINDOW):
    features = df[FEATURE_COLS].values.astype(float)
    scaled_all = scaler.transform(features)
    last_window = scaled_all[-window:].copy()
    edge_index = build_edge_index(window).to('cpu')

    preds_scaled = []
    curr = last_window.copy()

    for _ in range(forecast_days):
        x = torch.tensor(curr, dtype=torch.float32)
        batch_idx = torch.zeros(x.size(0), dtype=torch.long)
        with torch.no_grad():
            out = model(x, edge_index, batch_idx)
        val = out.detach().cpu().np()
        if np.ndim(val) == 0:
            p = float(val)
        else:
            p = float(val.flatten()[0])

        preds_scaled.append(p)
        next_row = curr[-1].copy()
        close_idx = FEATURE_COLS.index('Close')
        next_row[close_idx] = p
        curr = np.vstack([curr[1:], next_row])

    preds_scaled = np.array(preds_scaled)
    preds = invert_close_predictions(scaler, preds_scaled, scaled_all[-window:])
    return preds

def cell(label, value):
    st.markdown(
        f"""
        <div style="
            border: 1px solid #555;
            padding: 16px;
            text-align: center;
            font-size: 20px;
            border-radius: 6px;
            background-color: rgba(255,255,255,0.02);
        ">
            <div style="font-weight: 600;">{label}</div>
            <div>{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def table_cell(text, bold=False):
    weight = "700" if bold else "400"
    st.markdown(
        f"""
        <div style="
            border: 1px solid #666;
            padding: 12px 20px;
            font-size: 18px;
            font-weight: {weight};
            text-align: center;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<h1 style='text-align: center;'>AUREX 📈</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px; color: gray;'>Automated Foreign Exchange Forecasting</p>", unsafe_allow_html=True)

tabs = st.tabs(["Data", "Prediction", "About Us"])

with tabs[0]:
    st.header("📊 Data")
    default_end_date = date.today()
    default_start_date = default_end_date - timedelta(days=365)
    col1, col2 = st.columns([3,1])
    with col2:
        currency = st.selectbox("Choose currency", list(CURRENCY_YAHOO.keys()), key="data_currency")
        start_date = st.date_input("Start Date", default_start_date, key="data_start")
        end_date = st.date_input("End Date", default_end_date, key="data_end")
        interval = st.selectbox("Interval", list(INTERVAL.keys()), index=0, key="data_interval")
        interval = INTERVAL.get(interval)
        if interval == "1mo":
            start_date = start_date.replace(day=1)  
        if start_date >= end_date:
            st.error("Start Date must be before End Date")
            data = pd.DataFrame()
        else:
            yahoo = CURRENCY_YAHOO[currency]
            data = load_price_data(yahoo, start_date=start_date, end_date=end_date, interval=interval)
    with col1:
        if data is None or data.empty:
            st.warning("No data available")
        else:
            fig = go.Figure(data=[go.Candlestick(x=data['Date'], open=data['Open'], 
                            high=data['High'], low=data['Low'], close=data['Close'],
                            increasing_line_color="#22c55e",
                            decreasing_line_color="#ef4444",
                            increasing_fillcolor="rgba(34,197,94,0.6)",
                            decreasing_fillcolor="rgba(239,68,68,0.6)")])
            fig.update_layout(
                plot_bgcolor='#121212',
                font=dict(color='#e5e7eb', size=12),
                xaxis=dict(
                    gridcolor='#374151',
                    zerolinecolor='#4b5563',
                    showline=False
                ),
                yaxis=dict(
                    gridcolor='#374151',
                    zerolinecolor='#4b5563',
                    showline=False
                ),
                margin=dict(l=20, r=20, t=30, b=20),
                height=600
            )
            # fig.update_layout(height=600, template='plotly_dark')
            # fig.update_xaxes(rangeslider_visible=True, rangebreaks=[dict(bounds=['sat','mon'])])
            if interval in ["1d", "1wk"]:
                fig.update_xaxes(
                    rangeslider_visible=True,
                    rangebreaks=[dict(bounds=["sat", "mon"])]
                )
            else:
                fig.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)
            data_display = data.copy()
            data_display = data_display[['Date', 'Open', 'Low', 'High', 'Close']]
            data_display['Date'] = data_display['Date'].dt.strftime('%d-%m-%Y')
            data_display.set_index('Date', inplace=True)
            st.dataframe(data_display.style.format({'Open':'{:.6f}','High':'{:.6f}','Low':'{:.6f}','Close':'{:.6f}'}))

with tabs[1]:
    st.header("🔮 Prediction Results")
    col_1, col_2 = st.columns([1,1])
    with col_1:
        currency_pred = st.selectbox("Currency To Predict", list(CURRENCY_YAHOO.keys()), key="pred_currency")
        model_choice = st.selectbox("Model", ["GCN","GCN + RSA"], key="model_choice")
        steps = st.number_input("Forecast Days", min_value=1, max_value=10, value=DEFAULT_FORECAST_STEPS)
        model_key = (currency_pred, model_choice)
        model_path_to_load = MODEL_PATHS.get(model_key)
        model_path_found = model_path_to_load is not None

    st.markdown("---")
    if st.button("Run Prediction"):       
        try:
            with st.spinner(f"Loading data and model..."):
                
                # 1. Load Data and Scaler (FIX: Pass the Yahoo Ticker)
                yahoo_ticker = CURRENCY_YAHOO[currency_pred]

                df_historical, scaler_target = load_historical_data(yahoo_ticker)
                
                if df_historical is None or scaler_target is None:
                    st.error(f"Prediction failed: Cannot load sufficient historical data for {currency_pred} ({yahoo_ticker}).")
                    st.stop()
                
                if len(df_historical) < WINDOW + 1:
                    st.error(f"Prediction failed: Not enough data points ({len(df_historical)}) to create time windows. Need at least {WINDOW + 1} after cleaning.")
                    st.stop()

                scaled_feats = df_historical.copy() 

                # st.write("df_historal size = " + str(len(df_historical)))
                # st.write("scaled_feats size = " + str(len(scaled_feats)))

                # 2. Load Model
                model = load_model(model_path_to_load)
                if model is None:
                    st.error("Failed to load model architecture/weights.")
                    st.stop()

                # 3. Create Data Loaders for Prediction/Evaluation                
                data_list, train_idx, val_idx, test_idx = create_graph_data(scaled_feats)
                train_loader, val_loader, test_loader = get_data_loaders(data_list, train_idx, val_idx, test_idx, batch_size=32)
                
            with st.spinner("Generating predictions and forecast..."):
                
                # 4. Get Predictions and Metrics for Train/Val/Test
                train_actuals, train_preds, train_metrics = eval_epoch_predictions(model, train_loader, scaler_target)
                val_actuals, val_preds, val_metrics = eval_epoch_predictions(model, val_loader, scaler_target)
                test_actuals, test_preds, test_metrics = eval_epoch_predictions(model, test_loader, scaler_target)
                
                # Combine metrics for display
                # metrics_data = {
                #     'Metric': ['RMSE', 'MAE', 'R2', 'MAPE (%)'],
                #     'Train': [train_metrics['RMSE'], train_metrics['MAE'], train_metrics['R2'], train_metrics['MAPE']],
                #     'Validation': [val_metrics['RMSE'], val_metrics['MAE'], val_metrics['R2'], val_metrics['MAPE']],
                #     'Test': [test_metrics['RMSE'], test_metrics['MAE'], test_metrics['R2'], test_metrics['MAPE']]
                # }

                # metrics_data = {
                #     'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'R2'],
                #     'Test': [test_metrics['RMSE'], test_metrics['MAE'], test_metrics['MAPE'], test_metrics['R2']]
                # }
                # metrics_df = pd.DataFrame(metrics_data).set_index('Metric').applymap(lambda x: f"{x:.6f}")

                # 5. Generate Forecast
                # st.write("generate forecast ")
                last_window_loader = last_window_global_scaled(scaled_feats)
                forecast_unscaled = forecast_steps(model, last_window_loader, int(steps), scaler_target)

                # 6. Prepare Data for Plotting
                # st.write("prepare data for plotting ")
                historical_data_dates = df_historical['Date'].iloc[WINDOW:].reset_index(drop=True)
                
                # Forecast Dates (Next 'n' Business days)
                last_date = df_historical['Date'].iloc[-1]
                # Use 'B' for business day frequency, which is typical for currency data
                forecast_dates = pd.date_range(start=last_date, periods=int(steps) + 1, inclusive='right', freq='B') 
                
                # Combine all predictions/actuals
                actual_data = pd.Series(train_actuals + val_actuals + test_actuals, index=historical_data_dates)
                train_preds_series = pd.Series(train_preds, index=historical_data_dates[:len(train_preds)])
                val_preds_series = pd.Series(val_preds, index=historical_data_dates[len(train_preds):len(train_preds)+len(val_preds)])
                test_preds_series = pd.Series(test_preds, index=historical_data_dates[len(train_preds)+len(val_preds):])
                forecast_series = pd.Series(forecast_unscaled, index=forecast_dates)
                
                # 7. Create Interactive Plot (Plotly)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data.values, mode='lines', name='Actual Price', line=dict(color='#e5e7eb', width=1)))
                fig.add_trace(go.Scatter(x=train_preds_series.index, y=train_preds_series.values, mode='lines', name='Predicted Train', line=dict(color='#22c55e', dash='dot')))
                fig.add_trace(go.Scatter(x=val_preds_series.index, y=val_preds_series.values, mode='lines', name='Predicted Validation', line=dict(color='#f59e0b', dash='dot')))
                fig.add_trace(go.Scatter(x=test_preds_series.index, y=test_preds_series.values, mode='lines', name='Predicted Test', line=dict(color='#ef4444', dash='dot')))
                fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_series.values, mode='lines+markers', name=f'Forecast (Next {int(steps)} Days)', line=dict(color='#3b82f6', dash='dot'), marker=dict(size=5)))

                fig.update_layout(
                    plot_bgcolor='#121212',
                    font=dict(color='#e5e7eb', size=12),
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        rangebreaks=[dict(bounds=["sat", "mon"])],
                        gridcolor='#374151',
                        zerolinecolor='#4b5563',
                        showline=False
                    ),
                    yaxis=dict(
                        gridcolor='#374151',
                        zerolinecolor='#4b5563',
                        showline=False
                    ),
                    margin=dict(l=20, r=20, t=30, b=20),

                    xaxis_title='Date',
                    yaxis_title=f'{currency_pred} Price',
                    hovermode="x unified",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    height=600
                )

                # Display Results
                st.header(f"Results for {currency_pred} using {model_choice}")
                
                # Chart
                st.subheader("Time Series Chart (Actuals, Predictions, and Forecast)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast Table
                st.subheader(f"Predicted Forecast Values (Next {int(steps)} Days)")
                forecast_table = pd.DataFrame({
                    'Day Ahead': [f"Day {i+1}" for i in range(int(steps))],
                    'Date': forecast_dates.strftime('%Y-%m-%d'),
                    'Forecasted Price': [f"{p:.6f}" for p in forecast_unscaled]
                })
                st.dataframe(forecast_table, use_container_width=True, hide_index=True)
                
                # Metrics Table
                metrics_data = {
                    'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'R2'],
                    'Test': [test_metrics['RMSE'], test_metrics['MAE'], test_metrics['MAPE'], test_metrics['R2']]
                }
                metrics_df = pd.DataFrame(metrics_data).set_index('Metric').T.applymap(lambda x: f"{x:.6f}")
                st.subheader("Model Performance Metrics")
                # st.dataframe(metrics_df, use_container_width=False)

                # cols = st.columns([1, 2, 0.5, 1, 2])  # widths = padding control

                cols = st.columns(4)

                with cols[0]:
                    cell("RMSE", f"{test_metrics['RMSE']:.6f}")
                with cols[1]:
                    cell("MAE", f"{test_metrics['MAE']:.6f}")
                with cols[2]:
                    cell("MAPE (%)", f"{test_metrics['MAPE']:.6f}")
                with cols[3]:
                    cell("R²", f"{test_metrics['R2']:.6f}")


                # cols = st.columns(8)
                # table_cell("MSE", bold=True)
                # table_cell("200")
                # table_cell("")          # padding column
                # table_cell("RMSE", True)
                # table_cell("300")
                # table_cell("")
                # table_cell("MAE", True)
                # table_cell("150")

        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

with tabs[2]:
    st.header("ℹ️ About Us")
    st.write("This application is designed to help users analyze and predict foreign exchange (forex) price movements using advanced deep learning models.")
    st.markdown("---")
    
    st.header("🎯 Purpose of This Application")
    st.markdown("""
    * To demonstrate how advanced deep learning models can be applied to forex forecasting.
    * To provide transparent and understandable results.
    * To support research, learning, and decision analysis.
    """)
    st.markdown("---")

    st.header("🧠 Prediction Models Used")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. GCN")
        st.info("**Graph Convolutional Network**")
        st.write("""
        GCN is a machine learning model that can understand relationships between multiple variables. 
        Instead of looking at data one-by-one, GCN learns how different features influence each other, 
        which is highly effective for financial time-series data.
        """)

    with col2:
        st.subheader("2. GCN + RSA")
        st.info("**Optimized Deep Learning Model**")
        st.write("""
        This model combines the relational power of GCN with the **Reptile Search Algorithm (RSA)**.
        * **RSA is an optimization algorithm** that mimics the hunting behavior of crocodiles.
        * It helps the model **automatically find better parameter values**.
        * This makes the resulting predictions more **accurate and stable** compared to baseline models.
        """)
    st.markdown("---")

    st.header("📊 Evaluation Metrics")
    st.write("To measure how good the predictions are, this application uses several metrics that compare predicted values with actual market prices.")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.markdown("### RMSE")
        st.write("**Root Mean Square Error**")
        st.markdown("""
        * Shows average distance between predictions and real values.
        * Lower RMSE means better accuracy.
        * Large errors are penalized more heavily.
        """)
    
    with row1_col2:
        st.markdown("### MAE")
        st.write("**Mean Absolute Error**")
        st.markdown("""
        * Shows the average absolute difference between prediction and actual value.
        * Easy to understand as it represents average error directly.
        """)
    
    with row2_col1:
        st.markdown("### MAPE")
        st.write("**Mean Absolute Percentage Error**")
        st.markdown("""
        * Shows the error in percentage form.
        * Explains how big the error is relative to the actual value.
        """)
    
    with row2_col2:
        st.markdown("### R²")
        st.write("**R-Squared Score**")
        st.markdown("""
        * Shows how well the model explains the data.
        * Value ranges from **0 to 1**.
        * Closer to 1 shows a better model.
        """)

    st.markdown('---')

    st.header("⚠️ Important Notes")
    st.warning("""
    * Prediction results are based strictly on historical data.
    * This application is intended for analysis and educational purposes only.
    * Predictions should not be considered financial advice.
    * Market conditions can change suddenly and unpredictably.
    """)
    
