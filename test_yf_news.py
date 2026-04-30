import yfinance as yf

try:
    ticker = yf.Ticker('^NSEI')
    news = ticker.news
    print(f"Got {len(news)} news items for NIFTY")
    for n in news[:2]:
        print(n.get('title'), n.get('publisher'))
except Exception as e:
    print("Error:", e)
