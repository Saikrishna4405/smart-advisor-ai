import yfinance as yf
import pandas as pd

def test_failure():
    # Simulate a non-existent ticker or network failure if possible
    # We'll try some invalid tickers
    tickers = ["INVALID_TICKER", "UPL.NS"] # UPL.NS was failing in logs
    
    for t in tickers:
        print(f"Testing {t}...")
        try:
            ticker = yf.Ticker(t)
            df = ticker.history(period="max")
            print(f"Type of df: {type(df)}")
            print(f"Empty: {df.empty}")
            if not df.empty:
                df = df.reset_index()
                print(f"Columns: {df.columns.tolist()}")
            else:
                print("DF is empty")
        except Exception as e:
            print(f"Caught exception for {t}: {e}")

if __name__ == "__main__":
    test_failure()
