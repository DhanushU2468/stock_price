import numpy as np
from tensorflow import keras

def load_scaler():
    data_min = np.load("models/scaler_min.npy")
    data_max = np.load("models/scaler_max.npy")
    return data_min, data_max

def predict_next_day(model_path, close_prices):
    print("🔍 Loading:", model_path)
    model = keras.models.load_model(model_path)

    data_min, data_max = load_scaler()

    scaled = (close_prices - data_min) / (data_max - data_min)
    last_60 = scaled[-60:].reshape(1, 60, 1)

    pred = model.predict(last_60)
    final = pred * (data_max - data_min) + data_min
    return float(final)
