import yfinance as yf

try:
    ticker = yf.Ticker('^NSEI')
    news = ticker.news
    if news:
        print(list(news[0].keys()))
        print(news[0])
except Exception as e:
    print("Error:", e)
