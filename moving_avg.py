import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def compute_sma_signals(
    df: pd.DataFrame, short_window: int = 20, long_window: int = 50
) -> pd.DataFrame:
    """Compute SMA crossover trading signals.

    Signal convention:
         1  = Long  (short SMA is above long SMA)
        -1  = Out   (short SMA is below long SMA)
         0  = No position (insufficient history)

    Position column tracks crossovers:
        +2  = buy crossover  (signal changed from -1 to +1)
        -2  = sell crossover (signal changed from +1 to -1)

    Args:
        df: DataFrame with a 'Close' column.
        short_window: Lookback period for the fast SMA (default 20).
        long_window: Lookback period for the slow SMA (default 50).

    Returns:
        New DataFrame containing 'Close', 'SMA_Short', 'SMA_Long',
        'Signal', and 'Position' columns.
    """
    data = df[["Close"]].copy()

    data["SMA_Short"] = data["Close"].rolling(window=short_window).mean()
    data["SMA_Long"] = data["Close"].rolling(window=long_window).mean()

    data["Signal"] = 0
    # Long when the fast average is above the slow average
    data.loc[data["SMA_Short"] > data["SMA_Long"], "Signal"] = 1
    # Out (or short) when the fast average is below the slow average
    data.loc[data["SMA_Short"] < data["SMA_Long"], "Signal"] = -1

    # Diff of the signal column: non-zero entries are crossover events
    data["Position"] = data["Signal"].diff()

    return data


def compute_strategy_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily and cumulative strategy returns vs buy-and-hold.

    Signal is shifted by one day to avoid lookahead bias — we can only act
    on a signal after the close at which it was generated.

    Args:
        df: DataFrame with 'Close' and 'Signal' columns
            (typically the output of compute_sma_signals).

    Returns:
        DataFrame with 'Market_Return', 'Strategy_Return',
        'Cumulative_Market', and 'Cumulative_Strategy' columns added.
    """
    df = df.copy()
    df["Market_Return"] = df["Close"].pct_change()

    # Use previous day's signal so we don't trade on future information
    df["Strategy_Return"] = df["Market_Return"] * df["Signal"].shift(1)

    df["Cumulative_Market"] = (1 + df["Market_Return"]).cumprod()
    df["Cumulative_Strategy"] = (1 + df["Strategy_Return"]).cumprod()

    return df


def plot_sma_signals(
    data: pd.DataFrame, ticker: str = "", short_window: int = 20, long_window: int = 50
) -> None:
    """Plot closing price with SMA lines and buy/sell crossover markers.

    Args:
        data: DataFrame with 'Close', 'SMA_Short', 'SMA_Long', and 'Position' columns
              (output of compute_sma_signals).
        ticker: Ticker symbol used only in the chart title.
        short_window: Fast SMA period — used in the legend label.
        long_window: Slow SMA period — used in the legend label.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data["Close"], label="Close Price", alpha=0.7)
    plt.plot(data["SMA_Short"], label=f"{short_window}-Day SMA")
    plt.plot(data["SMA_Long"], label=f"{long_window}-Day SMA")

    buy_signals = data[data["Position"] == 2]
    plt.scatter(
        buy_signals.index, buy_signals["Close"],
        marker="^", color="green", s=100, label="Buy"
    )

    sell_signals = data[data["Position"] == -2]
    plt.scatter(
        sell_signals.index, sell_signals["Close"],
        marker="v", color="red", s=100, label="Sell"
    )

    title = f"{ticker} SMA Trading Strategy" if ticker else "SMA Trading Strategy"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cumulative_returns(data: pd.DataFrame) -> None:
    """Plot strategy cumulative returns vs buy-and-hold on the same axes.

    Args:
        data: DataFrame with 'Cumulative_Market' and 'Cumulative_Strategy' columns
              (output of compute_strategy_returns).
    """
    print("\nFinal Performance:")
    print(f"Market Return:   {data['Cumulative_Market'].iloc[-1]:.2f}x")
    print(f"Strategy Return: {data['Cumulative_Strategy'].iloc[-1]:.2f}x")

    plt.figure(figsize=(14, 7))
    plt.plot(data["Cumulative_Market"], label="Buy & Hold")
    plt.plot(data["Cumulative_Strategy"], label="SMA Strategy")
    plt.title("Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2025-01-01"
    short_window = 20
    long_window = 50

    raw = yf.download(ticker, start=start_date, end=end_date)
    data = compute_sma_signals(raw, short_window, long_window)
    data = compute_strategy_returns(data)

    plot_sma_signals(data, ticker=ticker, short_window=short_window, long_window=long_window)
    plot_cumulative_returns(data)
