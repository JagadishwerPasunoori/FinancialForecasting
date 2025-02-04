import numpy as np
from tensorflow.keras.models import load_model
from model_building import build_lstm_model

def train_model(X_train, y_train, epochs=50, batch_size=32):
    """
    Train the LSTM model.
    :param X_train: Training data (features)
    :param y_train: Training data (labels)
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :return: Trained model
    """
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    model.save("models/lstm_model.h5")
    return model

if __name__ == "__main__":
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    train_model(X_train, y_train)