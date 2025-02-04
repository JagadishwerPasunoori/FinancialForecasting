import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, feature_column, lookback=60):
    """
    Preprocess data for LSTM model.
    :param data: DataFrame containing the financial data
    :param feature_column: Column to use for forecasting (e.g., 'Close')
    :param lookback: Number of time steps to look back for training
    :return: X_train, y_train, scaler
    """
    dataset = data[feature_column].values.reshape(-1, 1)
    
    # Scale the data to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create training data
    X_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        X_train.append(scaled_data[i-lookback:i, 0])
        y_train.append(scaled_data[i, 0])
    
    # Convert to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshape X_train for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train, scaler

if __name__ == "__main__":
    data = pd.read_csv("data/raw_data.csv")
    feature_column = "Close"
    X_train, y_train, scaler = preprocess_data(data, feature_column)
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)