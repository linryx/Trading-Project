import pandas as pd
import yfinance as yf
import datetime as dt

if __name__ == "__main__":
    year = int(input('Enter Year: '))
    ticker = input('Stock ticker ')
    name = yf.Ticker(ticker)

    print(name.balance_sheet)

    start_date = dt.datetime(year, 1, 1)
    end_date = dt.datetime(year, 12, 31)
    print('Retrieving info for', ticker)
    print('From: ' + str(start_date))
    print('To: ' + str(end_date))

    # Download full history so SMA_100 has enough warm-up data before the target year.
    # (An alternative date-range download would leave ~100 NaN rows at the start of
    # the year because SMA_100 needs 100 prior trading days to stabilise.)
    data = yf.download(ticker, period='max')
    data['AVG'] = data['Open'] / 2 + data['Close'] / 2
    data['SMA_20'] = data['AVG'].rolling(window=20).mean()
    data['SMA_100'] = data['AVG'].rolling(window=100).mean()

    # Filter down to the requested calendar year
    data = data[data.index > start_date]
    data = data[data.index < end_date]

    # Drop the first 19 rows of the filtered year so SMA_20 is fully initialised
    data = data.drop(data.index[0:19])

    data['hold'] = data['SMA_20'] > data['SMA_100']

    pd.set_option('display.max_rows', None)

    for i in range(len(data) - 1):
        data.at[data.index[i], 'Signal'] = data['hold'].iloc[i] != data['hold'].iloc[i + 1]
        # Signal==True means a crossover occurred; hold==True means SMA_20 just crossed above
        if (data.at[data.index[i], 'Signal'] == True and data['hold'].iloc[i] == True):
            print(data['Close'].iloc[i])

    print(data[['SMA_20', 'SMA_100', 'hold', 'Signal']])
