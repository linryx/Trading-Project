import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # -----------------------------
    # Settings
    # -----------------------------
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2025-01-01"

    short_window = 20
    long_window = 50

    # -----------------------------
    # Download data
    # -----------------------------
    data = yf.download(ticker, start=start_date, end=end_date)

    # Keep only adjusted close price
    data = data[["Close"]].copy()
    # -----------------------------
    # Calculate moving averages
    # -----------------------------
    data["SMA_Short"] = data["Close"].rolling(window=short_window).mean()
    data["SMA_Long"] = data["Close"].rolling(window=long_window).mean()

    # -----------------------------
    # Generate trading signals
    # -----------------------------
    # 1 = Buy
    # -1 = Sell
    # 0 = Hold

    data["Signal"] = 0

    data.loc[
        data["SMA_Short"] > data["SMA_Long"],
        "Signal"
    ] = 1

    data.loc[
        data["SMA_Short"] < data["SMA_Long"],
        "Signal"
    ] = -1

    # Generate position changes
    data["Position"] = data["Signal"].diff()

    # -----------------------------
    # Plot results
    # -----------------------------
    plt.figure(figsize=(14, 7))

    # Price
    plt.plot(data["Close"], label="Close Price", alpha=0.7)

    # Moving averages
    plt.plot(data["SMA_Short"], label=f"{short_window}-Day SMA")
    plt.plot(data["SMA_Long"], label=f"{long_window}-Day SMA")

    # Buy signals
    buy_signals = data[data["Position"] == 2]
    plt.scatter(
        buy_signals.index,
        buy_signals["Close"],
        marker="^",
        color="green",
        s=100,
        label="Buy"
    )

    # Sell signals
    sell_signals = data[data["Position"] == -2]
    plt.scatter(
        sell_signals.index,
        sell_signals["Close"],
        marker="v",
        color="red",
        s=100,
        label="Sell"
    )

    plt.title(f"{ticker} SMA Trading Strategy")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    plt.show()

    # -----------------------------
    # Strategy Returns
    # -----------------------------
    data["Market_Return"] = data["Close"].pct_change()

    # Shift signal to avoid lookahead bias
    data["Strategy_Return"] = (
        data["Market_Return"] * data["Signal"].shift(1)
    )

    # Cumulative returns
    data["Cumulative_Market"] = (
        1 + data["Market_Return"]
    ).cumprod()

    data["Cumulative_Strategy"] = (
        1 + data["Strategy_Return"]
    ).cumprod()

    # Print performance
    print("\nFinal Performance:")
    print(f"Market Return:   {data['Cumulative_Market'].iloc[-1]:.2f}x")
    print(f"Strategy Return: {data['Cumulative_Strategy'].iloc[-1]:.2f}x")

    # Plot cumulative returns
    plt.figure(figsize=(14, 7))

    plt.plot(
        data["Cumulative_Market"],
        label="Buy & Hold"
    )

    plt.plot(
        data["Cumulative_Strategy"],
        label="SMA Strategy"
    )

    plt.title("Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True)

    plt.show()
