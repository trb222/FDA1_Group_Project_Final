import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
# Set page configuration
st.set_page_config(
    page_title="Stock Market Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# For ARIMA models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    statsmodels_available = True
except ImportError:
    statsmodels_available = False
    st.warning("statsmodels package not found. ARIMA models will not be available.")

# Application title and description
st.title("Stock Market Prediction Dashboard")
st.markdown("""
This dashboard uses machine learning to predict next-day stock price movements based on historical data and technical indicators.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predictions", "Reports"])

# Sidebar for stock selection
st.sidebar.title("Stock Selection")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL):", "AAPL")

# Date range selection
st.sidebar.title("Date Range")
today = datetime.date.today()
start_date = st.sidebar.date_input("Start Date", today - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", today)

# Technical Indicators Functions
def calculate_sma(data, window=20):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, window=20):
    """Calculate Exponential Moving Average"""
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    
    # Make two series: one for gains and one for losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and average loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence"""
    # Calculate EMAs
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    # Create DataFrame with explicit index
    result_df = pd.DataFrame({
        'MACD': macd_line,
        'MACD_Signal': signal_line,
        'MACD_Histogram': histogram
    }, index=data.index)  # Use the same index as the input data
    
    return result_df

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    # Create DataFrame with explicit index
    result_df = pd.DataFrame({
        'BB_Middle': sma,
        'BB_Upper': upper_band,
        'BB_Lower': lower_band
    }, index=data.index)  # Use the same index as the input data
    
    return result_df

def add_features(data):
    """Add technical indicators, lag features, and volume-based features"""
    import pandas as pd
    import numpy as np

    if data is None or len(data) == 0:
        st.error("No data available to add features. Please check your data source.")
        return None

    df = data.copy()

    # Ensure the expected columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Missing required column: {col}. Available columns: {df.columns.tolist()}")
            return None

    # =======================
    # Basic Technical Indicators
    # =======================
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma + (std * 2)
    df['BB_Lower'] = sma - (std * 2)

    # =======================
    # Volume-based Indicators
    # =======================
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    df['Daily_Return'] = df['Close'].pct_change()
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Volume_Change'] = df['Volume'].pct_change()

    # =======================
    # Lag Features (memory of past days)
    # =======================
    for lag in [1, 2, 3]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Daily_Return_lag_{lag}'] = df['Daily_Return'].shift(lag)
        df[f'RSI_14_lag_{lag}'] = df['RSI_14'].shift(lag)
        df[f'MACD_lag_{lag}'] = df['MACD'].shift(lag)

    # =======================
    # Target Variable
    # =======================
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df

# Prediction Functions
def prepare_features(data):
    """Prepare features for machine learning models"""
    # Drop non-feature columns
    feature_cols = [col for col in data.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    
    # Create feature matrix and target vector
    X = data[feature_cols].copy()
    y = data['Target']
    
    # Handle any NaN values
    X = X.fillna(method='ffill').fillna(0)
    
    return X, y, feature_cols

def train_random_forest(X_train, X_test, y_train, n_estimators=100, max_depth=5):
    """Train a Random Forest classifier"""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=20,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1 (up)
    
    return model, predictions, probabilities, scaler

def train_xgboost(X_train, X_test, y_train, n_estimators=100, learning_rate=0.1, max_depth=5):
    """Train an XGBoost classifier"""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1 (up)
    
    return model, predictions, probabilities, scaler

def train_arima_model(data, p=5, d=1, q=0):
    """Train an ARIMA model and forecast future prices"""
    if not statsmodels_available:
        st.error("statsmodels package not installed. ARIMA models are not available.")
        return None, None, None
        
    # Prepare data for ARIMA
    prices = data['Close'].values
    
    # Train ARIMA model
    model = ARIMA(prices, order=(p, d, q))
    model_fit = model.fit()
    
    # Forecast next day
    forecast = model_fit.forecast(steps=1)
    
    # Get confidence intervals
    forecast_ci = model_fit.get_forecast(steps=1).conf_int()
    
    # Determine if price will go up or down
    last_price = prices[-1]
    prediction = 1 if forecast[0] > last_price else 0
    
    # Calculate probability (approximate from forecast)
    range_size = forecast_ci[0, 1] - forecast_ci[0, 0]
    midpoint = (forecast_ci[0, 0] + forecast_ci[0, 1]) / 2
    distance_from_mid = abs(forecast[0] - midpoint)
    probability = 0.5 + (distance_from_mid / range_size) * 0.5  # Scale to 0.5-1.0 range
    
    # For testing and comparison
    historical_predictions = []
    for i in range(len(prices) - 10, len(prices)):
        # Fit model on data up to index i
        model_hist = ARIMA(prices[:i], order=(p, d, q))
        model_fit_hist = model_hist.fit()
        
        # Forecast next point
        forecast_hist = model_fit_hist.forecast(steps=1)[0]
        
        # Actual next value
        actual = prices[i]
        
        # Prediction (1 for up, 0 for down)
        pred = 1 if forecast_hist > prices[i-1] else 0
        actual_movement = 1 if actual > prices[i-1] else 0
        
        historical_predictions.append((pred, actual_movement))
    
    # Calculate accuracy
    accuracy = sum(1 for p, a in historical_predictions if p == a) / len(historical_predictions)
    
    return model_fit, prediction, probability

def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return acc, report, conf_matrix

def predict_next_day(model, scaler, latest_data, feature_cols, model_type="RandomForest"):
    """Make prediction for the next day"""
    # Get the latest data point
    latest_features = latest_data[feature_cols].iloc[-1:].copy()
    
    # Handle any NaN values
    latest_features = latest_features.fillna(method='ffill').fillna(0)
    
    # Scale features
    latest_features_scaled = scaler.transform(latest_features)
    
    # Predict
    prediction = model.predict(latest_features_scaled)[0]
    probability = model.predict_proba(latest_features_scaled)[0][1]  # Probability of class 1 (up)
    
    # If prediction is 0, probability should be for class 0
    if prediction == 0:
        probability = 1 - probability
    
    return prediction, probability

# Function to load stock data with fallback logic
@st.cache_data(ttl=3600)
def load_data(ticker_symbol, start_date, end_date):
    tickers_to_try = [
        ticker_symbol,
        f"{ticker_symbol}.US",  # US suffix
        ticker_symbol.upper(),  # uppercase version
        f"{ticker_symbol}.OQ"   # NASDAQ suffix
    ]

    for t in tickers_to_try:
        try:
            data = yf.download(t, start=start_date, end=end_date)

            # Handle MultiIndex fix
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    data = data.xs(key=t, axis=1, level=1, drop_level=True)
                except KeyError:
                    data = data.droplevel(level=0, axis=1)

            if not data.empty:
                return data  # success
        except Exception as e:
            continue  # try next ticker

    return None  # All attempts failed

# Load stock data
with st.spinner(f"Loading data for {ticker_symbol}..."):
    data = load_data(ticker_symbol, start_date, end_date)

# If still None, use fallback test data
if data is None and ticker_symbol:
    st.warning("⚠️ Using test data for demonstration purposes.")
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    test_data = {
        'Open': np.random.normal(100, 5, size=len(date_range)),
        'High': np.random.normal(105, 5, size=len(date_range)),
        'Low': np.random.normal(95, 5, size=len(date_range)),
        'Close': np.random.normal(102, 5, size=len(date_range)),
        'Volume': np.random.normal(1_000_000, 200_000, size=len(date_range))
    }
    data = pd.DataFrame(test_data, index=date_range)

# Dashboard page
if page == "Dashboard" and data is not None:
    st.header(f"{ticker_symbol} Stock Dashboard")
    
    # Display stock information
    try:
        stock_info = yf.Ticker(ticker_symbol).info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${stock_info.get('currentPrice', 'N/A')}")
        
        with col2:
            st.metric("Market Cap", f"${stock_info.get('marketCap', 'N/A'):,}")
        
        with col3:
            st.metric("52 Week Range", f"${stock_info.get('fiftyTwoWeekLow', 'N/A')} - ${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
    except Exception as e:
        st.warning(f"Detailed stock info not available: {e}")
        
    # Show sample of loaded data inside an expander (clean version)
    with st.expander("View Sample of Loaded Data"):
        st.subheader("Sample of Loaded Data")
    
        # Create a combined sample table: first 3 rows + blank row + last 3 rows
        first_part = data.head(3)
        last_part = data.tail(3)
    
        # Create a blank separator with "..." in Date column
        blank_row = pd.DataFrame({col: "" for col in data.columns}, index=["..."])
    
        combined_sample = pd.concat([first_part, blank_row, last_part])

        # Copy to avoid SettingWithCopyWarning
        combined_sample = combined_sample.copy()

        # Format the table:
        for col in combined_sample.columns:
            if pd.api.types.is_numeric_dtype(data[col]):  # use original data to check type
                if col.lower() == 'volume':
                    combined_sample[col] = combined_sample[col].apply(
                        lambda x: f"{int(x):,}" if isinstance(x, (int, float)) else x
                )
                else:
                    combined_sample[col] = combined_sample[col].apply(
                        lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
                )

        # Display the cleaned table
        st.dataframe(combined_sample)

  
    # Price chart tab
    tab1, tab2 = st.tabs(["Price Chart", "Technical Indicators"])
    
    with tab1:
        st.subheader("Historical Price Chart")

        # Create Plotly figure
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Candlestick'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{ticker_symbol} Stock Price',
            yaxis_title='Price (USD)',
            xaxis_title='Date',
            height=600,
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display raw data
        if st.checkbox("Show Raw Data"):
            st.subheader("Raw Price Data")
            st.dataframe(data)

    with tab2:
        st.subheader("Technical Indicators")
        
        # Select indicator
        indicator = st.selectbox(
            "Select Technical Indicator",
            ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"]
        )
        
        # Plot selected indicator
        if indicator == "SMA":
            periods = st.multiselect("Select SMA Periods", [5, 10, 20, 50, 100, 200], default=[20, 50])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
            
            for period in periods:
                sma = calculate_sma(data, period)
                fig.add_trace(go.Scatter(
                    x=data.index, 
                    y=sma, 
                    mode='lines', 
                    name=f'SMA {period}'
                ))
                
            fig.update_layout(
                title=f'Simple Moving Averages ({ticker_symbol})',
                yaxis_title='Price (USD)',
                xaxis_title='Date',
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif indicator == "EMA":
            periods = st.multiselect("Select EMA Periods", [5, 10, 20, 50, 100, 200], default=[12, 26])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
            
            for period in periods:
                ema = calculate_ema(data, period)
                fig.add_trace(go.Scatter(
                    x=data.index, 
                    y=ema, 
                    mode='lines', 
                    name=f'EMA {period}'
                ))
                
            fig.update_layout(
                title=f'Exponential Moving Averages ({ticker_symbol})',
                yaxis_title='Price (USD)',
                xaxis_title='Date',
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif indicator == "RSI":
            rsi = calculate_rsi(data)
                
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.1, row_heights=[0.7, 0.3])
            
            # Add price to top plot
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'),
                row=1, col=1
            )
            
            # Add RSI to bottom plot
            fig.add_trace(
                go.Scatter(x=data.index, y=rsi, mode='lines', name='RSI (14)'),
                row=2, col=1
            )
            
            # Add horizontal lines at 70 and 30
            fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.update_layout(
                title=f'Relative Strength Index ({ticker_symbol})',
                yaxis_title='Price (USD)',
                yaxis2_title='RSI',
                xaxis2_title='Date',
                height=600,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif indicator == "MACD":
            try:
                macd_data = calculate_macd(data)
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.1, row_heights=[0.7, 0.3])
                
                # Add price to top plot
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'),
                    row=1, col=1
                )
                
                # Add MACD to bottom plot
                fig.add_trace(
                    go.Scatter(x=data.index, y=macd_data['MACD'], mode='lines', name='MACD'),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=data.index, y=macd_data['MACD_Signal'], mode='lines', name='Signal Line'),
                    row=2, col=1
                )
                
                # Add histogram
                fig.add_trace(
                    go.Bar(x=data.index, y=macd_data['MACD_Histogram'], name='Histogram'),
                    row=2, col=1
                )
                
                fig.update_layout(
                    title=f'Moving Average Convergence Divergence ({ticker_symbol})',
                    yaxis_title='Price (USD)',
                    yaxis2_title='MACD',
                    xaxis2_title='Date',
                    height=600,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting MACD: {e}")
            
        elif indicator == "Bollinger Bands":
            try:
                bb_data = calculate_bollinger_bands(data)
                
                fig = go.Figure()
                
                # Add price
                fig.add_trace(go.Scatter(
                    x=data.index, 
                    y=data['Close'], 
                    mode='lines', 
                    name='Close Price'
                ))
                
                # Add Bollinger Bands
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=bb_data['BB_Upper'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=bb_data['BB_Middle'],
                    mode='lines',
                    name='Middle Band (20-day SMA)',
                    line=dict(dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=bb_data['BB_Lower'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(dash='dash')
                ))
                
                # Fill between upper and lower bands
                fig.add_trace(go.Scatter(
                    x=data.index.tolist()+data.index.tolist()[::-1],
                    y=bb_data['BB_Upper'].tolist()+bb_data['BB_Lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))
                
                fig.update_layout(
                    title=f'Bollinger Bands ({ticker_symbol})',
                    yaxis_title='Price (USD)',
                    xaxis_title='Date',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting Bollinger Bands: {e}")
elif page == "Dashboard" and data is None:
    st.info("Please enter a valid stock ticker symbol to view the dashboard.")
    st.info("Popular stock tickers include: AAPL (Apple), MSFT (Microsoft), GOOGL (Google), AMZN (Amazon), META (Meta/Facebook)")

# Predictions page
elif page == "Predictions" and data is not None:
    st.header(f"{ticker_symbol} Price Movement Prediction")
    
    # Add technical indicators for prediction
    data_with_features = add_features(data.copy())
    
    if data_with_features is not None:
        # Drop rows with NaN values
        data_with_features.dropna(inplace=True)
        
        # ML model selection - now without LSTM
        model_type = st.radio("Select Prediction Model", 
                             ["Random Forest", "XGBoost", "ARIMA"], 
                             horizontal=True)
        
        # Show a warning if ARIMA is not available
        if model_type == "ARIMA" and not statsmodels_available:
            st.warning("statsmodels package is not installed. ARIMA models are not available. Please install statsmodels to use this model.")
        
        # Common ML models (Random Forest & XGBoost)
        if model_type in ["Random Forest", "XGBoost"]:
            try:
                # Prepare features
                X, y, feature_cols = prepare_features(data_with_features)
                
                # Train-test split
                test_size = st.slider("Select test data size (%)", 10, 40, 20)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42, shuffle=False
                )
                
                # Model-specific parameters
                if model_type == "Random Forest":
                    n_estimators = st.slider("Number of trees", 50, 500, 100)
                    max_depth = st.slider("Maximum tree depth", 2, 20, 5)
                else:  # XGBoost
                    n_estimators = st.slider("Number of estimators", 50, 500, 100)
                    learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1, step=0.01)
                    max_depth = st.slider("Maximum tree depth", 2, 10, 5)
                
                # Add Auto-Tune checkbox
                auto_tune = st.checkbox("Auto-Tune Model Parameters", value=False)

                # Train and evaluate model
                if st.button("Train Model"):
                    with st.spinner(f"Training {model_type} model..."):
                        # Prepare features and split data
                        X, y, feature_cols = prepare_features(data_with_features)
                        test_size_value = test_size / 100
                        X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size_value, random_state=42, shuffle=False
                        )
                        best_params = {}

                        if model_type in ["Random Forest", "XGBoost"]:
                            if auto_tune:
                                st.info("Auto-tuning model parameters...this may take ~10-20 seconds.")
                
                                from sklearn.model_selection import RandomizedSearchCV
                                import numpy as np

                                if model_type == "Random Forest":
                                    from sklearn.ensemble import RandomForestClassifier
                                    base_model = RandomForestClassifier(random_state=42)
                                    param_dist = {
                                        'n_estimators': np.arange(50, 300, 50),
                                        'max_depth': np.arange(2, 20, 2),
                                        'min_samples_split': [5, 10, 20]
                                    }
                                else:  # XGBoost
                                    from xgboost import XGBClassifier
                                    base_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
                                    param_dist = {
                                        'n_estimators': np.arange(50, 300, 50),
                                        'max_depth': np.arange(2, 10, 2),
                                        'learning_rate': np.linspace(0.01, 0.3, 10)
                                    }    

                                search = RandomizedSearchCV(
                                    base_model, param_distributions=param_dist,
                                    n_iter=10, cv=3, scoring='accuracy', random_state=42, verbose=0
                                )
                                search.fit(X_train, y_train)
                                best_params = search.best_params_
                                st.success(f"Best Parameters Found: {best_params}")

                        # Train final model using tuned or user-input parameters
                        if model_type == "Random Forest":
                            model, predictions, probabilities, scaler = train_random_forest(
                                X_train, X_test, y_train,
                                n_estimators=best_params.get('n_estimators', n_estimators),
                                max_depth=best_params.get('max_depth', max_depth)
                            )
                        else:  # XGBoost
                            model, predictions, probabilities, scaler = train_xgboost(
                                X_train, X_test, y_train,
                                n_estimators=best_params.get('n_estimators', n_estimators),
                                learning_rate=best_params.get('learning_rate', learning_rate),
                                max_depth=best_params.get('max_depth', max_depth)
                            )

                        
                        # Display results
                        accuracy, report, conf_matrix = evaluate_model(y_test, predictions)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Model Accuracy", f"{accuracy:.2%}")
                        with col2:
                            st.metric("Baseline Accuracy (Most Frequent Class)", 
                                     f"{max(y_test.mean(), 1-y_test.mean()):.2%}")
                        
                        st.subheader("Classification Report")
                        st.code(report)
                        
                        st.subheader("Confusion Matrix")
                        fig, ax = plt.subplots(figsize=(5,4))
                        ax.matshow(conf_matrix, cmap='Blues')
                        
                        # Add text annotations
                        for i in range(conf_matrix.shape[0]):
                            for j in range(conf_matrix.shape[1]):
                                ax.text(j, i, str(conf_matrix[i, j]), ha='center', va='center')
                        
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_xticks([0, 1])
                        ax.set_yticks([0, 1])
                        ax.set_xticklabels(['Down', 'Up'])
                        ax.set_yticklabels(['Down', 'Up'])
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        importance = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance.head(10)['Feature'], importance.head(10)['Importance'])
                        ax.set_title('Top 10 Most Important Features')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Make prediction for next day
                        if len(X) > 0:
                            next_day_pred, next_day_prob = predict_next_day(model, scaler, data_with_features, feature_cols)
                            
                            st.subheader("Next Day Prediction")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                predicted_direction = "UP ⬆️" if next_day_pred == 1 else "DOWN ⬇️"
                                st.metric("Predicted Direction", predicted_direction)
                            
                            with col2:
                                st.metric("Confidence Score", f"{next_day_prob:.2%}")
                                
                            # Display warning about prediction reliability
                            st.warning("""
                            **Disclaimer:** Stock market predictions involve high uncertainty. This prediction is based on 
                            historical patterns and may not reflect future market movements accurately. Always consider 
                            multiple sources of information for investment decisions.
                            """)
                            
                            # Save prediction to session state for reports
                            if 'predictions' not in st.session_state:
                                st.session_state.predictions = []
                            
                            st.session_state.predictions.append({
                                'Date': datetime.date.today().strftime('%Y-%m-%d'),
                                'Ticker': ticker_symbol,
                                'Prediction': predicted_direction,
                                'Confidence': f"{next_day_prob:.2%}",
                                'Model': model_type
                            })
            except Exception as e:
                st.error(f"Error during model training or prediction: {e}")
                st.info("This could be due to insufficient data or issues with the features. Try selecting a different stock or date range.")
                
        # ARIMA Model (Statistical)
        elif model_type == "ARIMA" and statsmodels_available:
            try:
                # ARIMA-specific parameters
                p = st.slider("AR order (p)", 0, 10, 5)
                d = st.slider("Difference order (d)", 0, 2, 1)
                q = st.slider("MA order (q)", 0, 10, 0)
                
                if st.button("Train Model"):
                    with st.spinner("Training ARIMA model..."):
                        model_fit, prediction, probability = train_arima_model(
                            data_with_features, p=p, d=d, q=q
                        )
                        
                        if model_fit is not None:
                            # Display model summary
                            st.subheader("ARIMA Model Summary")
                            # Convert statsmodels summary to string for display
                            summary_str = str(model_fit.summary())
                            st.code(summary_str)
                            
                            # Create a forecast plot
                            fig = go.Figure()
                            
                            # Historical data
                            fig.add_trace(go.Scatter(
                                x=data_with_features.index[-30:],  # Last 30 days
                                y=data_with_features['Close'][-30:],
                                mode='lines',
                                name='Historical'
                            ))
                            
                            # Forecast next point
                            forecast_date = data_with_features.index[-1] + pd.Timedelta(days=1)
                            forecast_value = model_fit.forecast(steps=1)[0]
                            
                            fig.add_trace(go.Scatter(
                                x=[data_with_features.index[-1], forecast_date],
                                y=[data_with_features['Close'].iloc[-1], forecast_value],
                                mode='lines+markers',
                                line=dict(dash='dash'),
                                name='Forecast'
                            ))
                            
                            fig.update_layout(
                                title=f'ARIMA({p},{d},{q}) Forecast for {ticker_symbol}',
                                xaxis_title='Date',
                                yaxis_title='Price (USD)',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display next day prediction
                            st.subheader("Next Day Prediction")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                predicted_direction = "UP ⬆️" if prediction == 1 else "DOWN ⬇️"
                                st.metric("Predicted Direction", predicted_direction)
                            
                            with col2:
                                st.metric("Confidence Score", f"{probability:.2%}")
                                
                            # Display warning about prediction reliability
                            st.warning("""
                            **Disclaimer:** ARIMA forecasts are based solely on historical price patterns and do not 
                            account for news, market sentiment, or other external factors. Stock market predictions 
                            involve high uncertainty.
                            """)
                            
                            # Save prediction to session state for reports
                            if 'predictions' not in st.session_state:
                                st.session_state.predictions = []
                            
                            st.session_state.predictions.append({
                                'Date': datetime.date.today().strftime('%Y-%m-%d'),
                                'Ticker': ticker_symbol,
                                'Prediction': predicted_direction,
                                'Confidence': f"{probability:.2%}",
                                'Model': f"ARIMA({p},{d},{q})"
                            })
            except Exception as e:
                st.error(f"Error during ARIMA training or prediction: {e}")
                st.info("Try different ARIMA parameters or a different stock with more stable patterns.")
    else:
        st.error("Could not generate features from the data. Please try a different stock or date range.")
elif page == "Predictions" and data is None:
    st.info("Please go to the Dashboard first and select a valid stock ticker to make predictions.")

# Reports page
elif page == "Reports":
    st.header("Prediction Reports")
    
    if 'predictions' in st.session_state and st.session_state.predictions:
        # Display predictions table
        predictions_df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(predictions_df)
        
        # Download as CSV
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name=f"stock_predictions_{datetime.date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Visualization of predictions
        st.subheader("Prediction Visualization")
        
        # Group by model and prediction
        model_counts = predictions_df.groupby(['Model', 'Prediction']).size().reset_index(name='Count')
        
        # Create bar chart
        import plotly.express as px
        fig = px.bar(model_counts, x='Model', y='Count', color='Prediction', 
                    title='Predictions by Model',
                    barmode='group',
                    color_discrete_map={'UP ⬆️': 'green', 'DOWN ⬇️': 'red'})
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions have been made yet. Go to the Predictions page to generate some predictions first.")
    
    # Option to clear predictions
    if 'predictions' in st.session_state and st.session_state.predictions:
        if st.button("Clear All Predictions"):
            st.session_state.predictions = []
            st.success("All predictions cleared.")
            st.experimental_rerun()