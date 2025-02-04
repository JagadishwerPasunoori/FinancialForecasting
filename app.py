import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Financial Forecasting App")

# Load predictions and actual values
try:
    predictions = np.load("data/predictions.npy")
    actual_values = np.load("data/actual_values.npy")
    
    # Ensure shapes match
    if predictions.shape != actual_values.shape:
        st.warning(f"Shape mismatch! Predictions: {predictions.shape}, Actual: {actual_values.shape}")
        predictions = predictions[:len(actual_values)]  # Truncate to match shapes
    
 
    
    # Plot results
    st.write("### Actual vs Predicted Values")
    fig, ax = plt.subplots()
    ax.plot(actual_values, label="Actual", color="blue")
    ax.plot(predictions, label="Predicted", color="red")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

except FileNotFoundError:
    st.error("Error: Prediction or actual value files not found. Please run the model evaluation script first.")
except Exception as e:
    st.error(f"An error occurred: {e}")