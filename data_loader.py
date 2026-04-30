import pandas as pd
import os

def load_data(filename):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "dataset", "india", filename)

    if not os.path.exists(path):
        raise Exception(f"File not found: {path}")

    data = pd.read_csv(path)

    data.columns = [col.strip().lower() for col in data.columns]

    close_col = next((col for col in data.columns if "close" in col), None)
    date_col = next((col for col in data.columns if "date" in col), None)

    if close_col is None:
        raise Exception("No Close column found")

    if date_col:
        data = data[[date_col, close_col]]
        data.columns = ["Date", "Close"]
        data["Date"] = pd.to_datetime(data["Date"])
    else:
        data = data[[close_col]]
        data.columns = ["Close"]
        data["Date"] = pd.date_range(start="2000-01-01", periods=len(data))
        data = data[["Date", "Close"]]

    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna()

    return data