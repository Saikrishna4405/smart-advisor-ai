import time
import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Setup dummy app logic to test speed
def train_and_predict(df):
    if len(df) < 200:
        return {"score": 5, "error": "Short history"}
    
    try:
        data = df.filter(['Close']).values
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)
        lookback = 60
        X_train, y_train = [], []
        for i in range(lookback, len(scaled_data)):
            X_train.append(scaled_data[i-lookback:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=2, verbose=0) # reduced to 2 for test
        return {"score": 7}
    except Exception as e:
        return {"error": str(e)}

print("Starting Scanner Diagnostic...")
symbol = "TCS.NS"
start = time.time()

print(f"Fetching data for {symbol}...")
t = yf.Ticker(symbol)
df = t.history(period="2y")
fetch_time = time.time() - start
print(f"Data Fetch Time: {fetch_time:.2f}s")

print(f"Running ML Prediction for {symbol}...")
ml_start = time.time()
res = train_and_predict(df)
ml_time = time.time() - ml_start
print(f"ML Prediction Time: {ml_time:.2f}s")
print(f"Result: {res}")

total = time.time() - start
print(f"Total Time for 1 stock: {total:.2f}s")
print(f"Projected time for 24 stocks: {(total * 24):.2f}s")
