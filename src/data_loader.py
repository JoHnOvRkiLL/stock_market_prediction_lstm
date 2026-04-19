import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os

def fetch_stock_data(ticker="RELIANCE.NS", period="5y"):
    print(f"Fetching data for {ticker}...")
    
    df = yf.download(ticker, period=period, auto_adjust=True)
    
    if df.empty:
        print(f"Error: No data found for {ticker}.")
        return None
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.dropna(inplace=True)
    
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    
    df.dropna(inplace=True)
    
    os.makedirs("data", exist_ok=True)
    
    file_path = f"data/{ticker.replace('.NS', '')}_historical_data.csv"
    df.to_csv(file_path)
    print(f"Success! Data saved locally to: {file_path}")
    
    return df

if __name__ == "__main__":
    ticker_symbol = "RELIANCE.NS" 
    stock_data = fetch_stock_data(ticker_symbol)
    
    if stock_data is not None:
        print("\n--- Data Preview ---")
        print(stock_data.tail())