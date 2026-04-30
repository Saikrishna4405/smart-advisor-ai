import pandas as pd
import numpy as np

def calculate_metrics(data):

    data = data.copy()

    # ================= Daily Returns =================
    data['Return'] = data['Close'].pct_change()

    volatility = data['Return'].std()
    avg_return = data['Return'].mean()

    # ================= Moving Averages =================
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    if len(data) >= 50:
        trend = data['MA20'].iloc[-1] - data['MA50'].iloc[-1]
    else:
        trend = 0

    # ================= Sharpe Ratio =================
    if volatility and not np.isnan(volatility):
        sharpe_ratio = avg_return / volatility
    else:
        sharpe_ratio = 0

    # ================= CAGR =================
    if len(data) > 1:
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        years = len(data) / 252  # trading days

        if start_price > 0 and years > 0:
            cagr = (end_price / start_price) ** (1 / years) - 1
        else:
            cagr = 0
    else:
        cagr = 0

    # ================= Risk Level =================
    if volatility < 0.015:
        risk_level = "Low"
    elif volatility < 0.03:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # ================= RSI (Optional – Trading Style) =================
    window = 14
    delta = data['Close'].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))

    if not rsi_series.empty and not np.isnan(rsi_series.iloc[-1]):
        latest_rsi = round(rsi_series.iloc[-1], 2)
    else:
        latest_rsi = 50

    # ================= Return Dictionary =================
    return {
        "volatility": round(volatility, 5),
        "avg_return": round(avg_return, 5),
        "trend": round(trend, 3),
        "sharpe_ratio": round(sharpe_ratio, 3),
        "cagr": round(cagr * 100, 2),   # percentage
        "risk_level": risk_level,
        "rsi": latest_rsi
    }