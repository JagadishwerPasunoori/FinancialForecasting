import matplotlib.pyplot as plt
import numpy as np

def plot_results(actual_values, predictions):
    """
    Plot actual vs predicted values.
    :param actual_values: Actual values
    :param predictions: Predicted values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(actual_values, color='blue', label='Actual')
    plt.plot(predictions, color='red', label='Predicted')
    plt.title('Financial Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig("visuals/predictions_vs_actual.png")
    plt.show()