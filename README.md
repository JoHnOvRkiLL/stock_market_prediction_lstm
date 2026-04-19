# Hybrid AI Stock Decision Support System

This project builds and serves a stock prediction workflow in this order:
1. Data loader
2. Processor
3. Model training
4. Streamlit app

## Prerequisites

- Python 3.9 or higher
- Internet connection (required for fetching stock data via Yahoo Finance)

## Project Structure

- app.py -> Streamlit app
- src/data_loader.py -> downloads and stores stock data CSV
- src/processor.py -> preprocesses data and saves scaler
- src/model.py -> trains LSTM model and saves it
- data/ -> generated CSV files
- models/ -> saved scaler, model, and training plot

## Setup

### Windows (PowerShell)

Create and activate a virtual environment:

python -m venv venv
venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

## Run Order (Important)

Run the following commands from the project root folder.

### 1) Run Data Loader

python src/data_loader.py

What this does:
- Downloads historical data for RELIANCE.NS (default)
- Adds technical indicators (SMA_20, SMA_50, RSI, EMA_20)
- Saves file to data/RELIANCE_historical_data.csv

### 2) Run Processor

python src/processor.py

What this does:
- Loads data/RELIANCE_historical_data.csv
- Selects features: Close, SMA_20, RSI
- Scales features using MinMaxScaler
- Saves scaler to models/scaler.save

### 3) Train Model

python src/model.py

What this does:
- Calls the processor pipeline
- Trains LSTM model
- Saves model to models/lstm_stock_model.keras
- Saves training-loss chart to models/training_loss.png

### 4) Launch Streamlit App

streamlit run app.py

Then open the local URL shown in terminal (usually http://localhost:8501).

## Notes

- Default training/data ticker is RELIANCE.NS in src/data_loader.py.
- In the app sidebar, you can predict for other NSE tickers like TCS.NS, INFY.NS, HDFCBANK.NS.
- If you retrain the model, rerun model training before opening the app to use the latest model.

## Quick Full Pipeline (one-by-one)

python src/data_loader.py
python src/processor.py
python src/model.py
streamlit run app.py

## Troubleshooting

- If Activate.ps1 is blocked, run:
	Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

- If Streamlit is not found, reinstall requirements:
	pip install -r requirements.txt

- If model/scaler files are missing, ensure steps 2 and 3 completed without errors.