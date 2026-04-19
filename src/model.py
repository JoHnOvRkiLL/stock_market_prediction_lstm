import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from processor import prepare_data
import os

def build_and_train_model():
    # 1. Get the preprocessed data
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    
    print("\nBuilding the LSTM Model...")
    
    # 2. Model Architecture
    model = Sequential()
    
    # First LSTM layer (return_sequences=True because we have a second LSTM layer coming)
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2)) # Prevents overfitting (memorizing the data)
    
    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Output layer (Predicts 1 value: The next day's scaled closing price)
    model.add(Dense(units=1))
    
    # 3. Compile the Model
    # 'adam' is the best default optimizer. 'mean_squared_error' is standard for price prediction.
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("\nStarting Training (This might take a minute or two)...")
    
    # 4. Train the Model
    # epochs=20: The AI will review the data 20 times to learn patterns.
    # batch_size=32: It learns in chunks of 32 days at a time.
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    # 5. Save the trained model
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_stock_model.keras")
    print("\nSuccess! Model trained and saved to models/lstm_stock_model.keras")
    
    # 6. Plot the Training Loss (Great for your SPPU Project Report!)
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation (Test) Loss')
    plt.title('Model Loss Over Time (Lower is Better)')
    plt.xlabel('Epochs (Training Rounds)')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.legend()
    plt.savefig("models/training_loss.png")
    print("Training graph saved as models/training_loss.png (Add this to your report!)")

if __name__ == "__main__":
    build_and_train_model()