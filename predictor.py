from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def predict_next_price(data):

    data = data.copy()

    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['Return'] = data['Close'].pct_change()
    data['Target'] = data['Close'].shift(-1)

    data = data.dropna()

    X = data[['Close', 'MA20', 'MA50', 'Return']]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Next day prediction
    last_row = X.iloc[-1:]
    next_prediction = model.predict(last_row)

    return {
        "prediction": round(float(next_prediction[0]), 2),
        "mae": round(mae, 2),
        "r2": round(r2, 3),
        "actual_test": y_test.tolist(),          # 👈 REQUIRED
        "predicted_test": y_pred.tolist()        # 👈 REQUIRED
    }