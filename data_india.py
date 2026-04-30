import pandas as pd
import os

def load_data(filename):
    path = os.path.join("dataset", "india", filename)

    data = pd.read_csv(path)

    # Flatten columns if multi-level
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Strip spaces
    data.columns = [col.strip() for col in data.columns]

    # Try direct match first
    if "Date" in data.columns:
        date_col = "Date"
    else:
        date_col = next((c for c in data.columns if "date" in c.lower()), None)

    if "Close" in data.columns:
        close_col = "Close"
    else:
        close_col = next((c for c in data.columns if "close" in c.lower()), None)

    if date_col is None or close_col is None:
        raise Exception("Dataset must contain Date and Close columns")

    data = data[[date_col, close_col]]
    data.columns = ["Date", "Close"]

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

    data = data.dropna()

    return data
