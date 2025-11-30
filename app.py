import streamlit as st
import yfinance as yf
import numpy as np
from src.predict import predict_next_day

st.title("📈 Stock Price Predictor (Python 3.11 + Keras 3 Compatible)")

symbol = st.text_input("Stock Symbol", "AAPL").upper()

if st.button("Predict"):
    data = yf.download(symbol, start="2010-01-01", end="2024-12-31", auto_adjust=True)

    if data.empty:
        st.error("Invalid symbol.")
    else:
        st.line_chart(data["Close"])
        
        price = data["Close"].values
        pred = predict_next_day("models/lstm_model.keras", price)

        st.success(f"Predicted next price for {symbol}: ${pred:.2f}")
