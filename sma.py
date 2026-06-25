# NOTE: This module uses the Open/Close midpoint (AVG) as the SMA input, unlike moving_avg.py
# which uses Close. This is intentional — the midpoint smooths intraday noise for longer-term scans.
import pandas as pd
import yfinance as yf
import datetime as dt


def compute_avg(df: pd.DataFrame) -> pd.DataFrame:
    """Add a midpoint price column (average of Open and Close).

    Using the Open/Close midpoint rather than Close alone smooths out
    intraday noise and is a common input for longer-term SMA scanners.

    Args:
        df: DataFrame with 'Open' and 'Close' columns.

    Returns:
        DataFrame with an 'AVG' column added.
    """
    df = df.copy()
    df['AVG'] = df['Open'] / 2 + df['Close'] / 2
    return df


def compute_sma(
    df: pd.DataFrame, short_window: int = 20, long_window: int = 100
) -> pd.DataFrame:
    """Compute short and long simple moving averages on the 'AVG' column.

    Args:
        df: DataFrame with an 'AVG' column (output of compute_avg).
        short_window: Lookback period for the fast SMA (default 20).
        long_window: Lookback period for the slow SMA (default 100).

    Returns:
        DataFrame with 'SMA_{short_window}' and 'SMA_{long_window}' columns added.
    """
    df[f'SMA_{short_window}'] = df['AVG'].rolling(window=short_window).mean()
    df[f'SMA_{long_window}'] = df['AVG'].rolling(window=long_window).mean()
    return df


def filter_to_year(
    df: pd.DataFrame, year: int, warm_up_rows: int = 19
) -> pd.DataFrame:
    """Filter a DataFrame to a single calendar year and drop warm-up rows.

    The first `warm_up_rows` rows of the filtered range are dropped so that
    the short SMA (default window 20) is fully initialised before signals
    are evaluated.  Because SMAs are computed on full history before this
    filter, the long SMA is already valid for the entire year.

    Args:
        df: DataFrame with a DatetimeIndex.
        year: Calendar year to keep (inclusive Jan 1 – Dec 31).
        warm_up_rows: Number of leading rows to drop after filtering (default 19).

    Returns:
        Filtered DataFrame with warm-up rows removed.
    """
    start = dt.datetime(year, 1, 1)
    end = dt.datetime(year, 12, 31)
    df = df[(df.index >= start) & (df.index <= end)].copy()
    df = df.drop(df.index[:warm_up_rows])
    return df


def detect_crossover_signals(
    df: pd.DataFrame,
    short_col: str = 'SMA_20',
    long_col: str = 'SMA_100',
) -> pd.DataFrame:
    """Detect SMA crossover events and add 'hold' and 'Signal' columns.

    'hold' is True on every day the short SMA is above the long SMA (uptrend).
    'Signal' is True only on the day a crossover occurs — when 'hold' flips
    value relative to the previous day.

    Args:
        df: DataFrame with short and long SMA columns already computed.
        short_col: Name of the fast SMA column (default 'SMA_20').
        long_col: Name of the slow SMA column (default 'SMA_100').

    Returns:
        DataFrame with 'hold' and 'Signal' columns added.
    """
    df = df.copy()
    df['hold'] = df[short_col] > df[long_col]
    df['Signal'] = df['hold'] != df['hold'].shift(1)

    return df


def print_buy_signals(df: pd.DataFrame) -> None:
    """Print the Close price on every bullish crossover (signal + uptrend day).

    A bullish crossover is when Signal is True and hold is True, meaning the
    short SMA just crossed *above* the long SMA.

    Args:
        df: DataFrame with 'Close', 'hold', and 'Signal' columns.
    """
    buys = df[df['Signal'] & df['hold']]
    print(buys['Close'].to_string())


if __name__ == "__main__":
    year = int(input('Enter Year: '))
    ticker = input('Stock ticker: ')

    name = yf.Ticker(ticker)
    print(name.balance_sheet)

    print(f'Retrieving info for {ticker}')
    print(f'From: {dt.datetime(year, 1, 1)}')
    print(f'To:   {dt.datetime(year, 12, 31)}')

    # Download full history so SMA_100 has enough warm-up data before the target year
    raw = yf.download(ticker, period='max')
    data = compute_avg(raw)
    data = compute_sma(data, short_window=20, long_window=100)
    data = filter_to_year(data, year)
    data = detect_crossover_signals(data)

    print_buy_signals(data)

    pd.set_option('display.max_rows', None)
    print(data[['SMA_20', 'SMA_100', 'hold', 'Signal']])
