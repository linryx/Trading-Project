import pandas as pd
import yfinance as yf
import datetime as dt
import momentum
import moving_avg
import sma
import volitility

# -----------------------------
# Settings
# -----------------------------
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2025-01-01"

df = yf.download(ticker, start=start_date, end=end_date)

# -----------------------------
print(momentum.calculate_momentum(df, [5,10]))

"""
tickers = ['SPY', 'QQQ']
end_date = dt.datetime.fromisocalendar(2025, 1, 2)
start_date = end_date - dt.timedelta(days = 393)
print("From: " + str(start_date))
print("To: " + str(end_date))

data = {}

for ticker in tickers:
    x = yf.download(ticker, start=start_date, end=end_date)
    data[ticker]= x
    data[ticker]['AVG'] = (data[ticker]['Open'] + data[ticker]['Close'] ) / 2
    data[ticker]['SMA_20']=data[ticker]['AVG'].rolling(window=20).mean()
    data[ticker] = data[ticker].drop(data[ticker].index[0:19])
print(data)
"""