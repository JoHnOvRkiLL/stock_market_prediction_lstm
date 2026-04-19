import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def prepare_data(file_path="data/RELIANCE_historical_data.csv", time_step=60):
    print("Loading data for preprocessing...")
    
    # 1. Load the locally saved dataset
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    
    # 2. Feature Selection
    # We will feed the AI the Closing Price, moving average, and RSI
    features = ['Close', 'SMA_20', 'RSI']
    data = df[features].values
    
    # 3. Scaling the Data (MinMaxScaler)
    # Neural Networks perform best when data is scaled between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Save the scaler so we can 'un-scale' predictions later in the web app
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.save")
    print("Scaler saved to models/scaler.save")
    
    # 4. Creating Sequences (Windowing)
    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        # X: The past 60 days of data (Close, SMA_20, RSI)
        X.append(scaled_data[i:(i + time_step)])
        # y: The 61st day's Close price (Index 0 is 'Close')
        y.append(scaled_data[i + time_step, 0]) 
        
    X = np.array(X)
    y = np.array(y)
    
    # 5. Train-Test Split (80% Train, 20% Test)
    # Important for SPPU Viva: Never shuffle time-series data!
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print("\n--- Preprocessing Complete ---")
    print(f"Training Data Shape (X): {X_train.shape} -> (Samples, Time-Steps, Features)")
    print(f"Testing Data Shape (X):  {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    # Test the processor
    X_train, X_test, y_train, y_test, scaler = prepare_data()