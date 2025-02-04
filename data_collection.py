import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

def fetch_financial_data(ticker, start_date, end_date):
    """
    Fetch historical financial data from Yahoo Finance.
    :param ticker: Stock ticker symbol (e.g., 'AAPL')
    :param start_date: Start date in 'YYYY-MM-DD' format
    :param end_date: End date in 'YYYY-MM-DD' format
    :return: DataFrame with historical data
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        data.drop(columns=['Ticker'], errors='ignore', inplace=True)
        print("Data downloaded successfully:")
        print(data.head())
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2010-01-01"
    end_date = "2023-01-01"
    data = fetch_financial_data(ticker, start_date, end_date)
    
    if data is not None:
        data.to_csv("data/raw_data.csv", index=True)
        print("Data saved to data/raw_data.csv")
    else:
        print("Failed to fetch data.")