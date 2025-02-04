import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def evaluate_model(model, data, feature_column, scaler, lookback=60):
    """
    Evaluate the model on the test data.
    :param model: Trained LSTM model
    :param data: Original DataFrame
    :param feature_column: Column used for forecasting
    :param scaler: Scaler used for preprocessing
    :param lookback: Lookback period
    :return: Predictions and actual values
    """
    dataset = data[feature_column].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(dataset)
    
    X_test, y_test = [], []
    for i in range(lookback, len(scaled_data)):
        X_test.append(scaled_data[i-lookback:i, 0])
        y_test.append(scaled_data[i, 0])
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Actual values
    actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return predictions, actual_values

if __name__ == "__main__":
    data = pd.read_csv("data/raw_data.csv")
    feature_column = "Close"
    scaler = MinMaxScaler()
    model = load_model("models/lstm_model.h5")
    predictions, actual_values = evaluate_model(model, data, feature_column, scaler)
    np.save("data/predictions.npy", predictions)
    np.save("data/actual_values.npy", actual_values)