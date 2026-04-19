import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# 1. UI Configuration
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 Hybrid AI Stock Decision Support System")

# 2. Load the trained AI and Scaler
@st.cache_resource
def load_ai_components():
    model = load_model("models/lstm_stock_model.keras")
    scaler = joblib.load("models/scaler.save")
    return model, scaler

model, scaler = load_ai_components()

# 3. Sidebar for User Input
st.sidebar.header("Parameters")
ticker = st.sidebar.text_input("Enter NSE Ticker Symbol", "RELIANCE.NS")
st.sidebar.markdown("*(Examples: TCS.NS, INFY.NS, HDFCBANK.NS)*")

if st.sidebar.button("Predict Next Day Price"):
    with st.spinner(f"Fetching live data for {ticker}..."):
        # Fetch the last 150 days to ensure we have enough data after moving averages
        df = yf.download(ticker, period="150d", auto_adjust=True, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            st.error("Invalid Ticker Symbol or No Data Found.")
        else:
            # 4. Feature Engineering (Live Data)
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df.dropna(inplace=True)
            
            # 5. Extract the last 60 days of required features
            features = ['Close', 'SMA_20', 'RSI']
            last_60_days = df[features].tail(60).values
            
            # 6. Preprocessing for the AI
            last_60_days_scaled = scaler.transform(last_60_days)
            X_input = last_60_days_scaled.reshape(1, 60, 3)
            
            # 7. AI Prediction
            pred_scaled = model.predict(X_input)
            
            # 8. Un-scale the prediction back to Rupees
            # The scaler expects 3 columns, so we create a dummy array to inverse transform
            dummy_array = np.zeros((1, 3))
            dummy_array[0, 0] = pred_scaled[0][0]
            predicted_price = scaler.inverse_transform(dummy_array)[0, 0]
            
            current_price = df['Close'].iloc[-1]
            price_difference = predicted_price - current_price
            
            # 9. Dashboard Display
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Closing Price", f"₹ {current_price:.2f}")
            col2.metric("AI Predicted Price (Next Day)", f"₹ {predicted_price:.2f}", f"₹ {price_difference:.2f}")
            
            # Simple Rule-Based Recommendation
            if predicted_price > current_price and df['RSI'].iloc[-1] < 70:
                col3.success("Signal: BUY / HOLD")
            elif predicted_price < current_price:
                col3.error("Signal: SELL")
            else:
                col3.warning("Signal: NEUTRAL")

            st.markdown("---")
            st.markdown(f"### {ticker} - 60 Day Price Trend")
            
            # 10. Interactive Chart using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].tail(60), mode='lines', name='Actual Close Price'))
            fig.add_trace(go.Scatter(x=df.index[-60:], y=df['SMA_20'].tail(60), mode='lines', name='20-Day SMA', line=dict(dash='dash')))
            
            fig.update_layout(xaxis_title="Date", yaxis_title="Price (₹)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)