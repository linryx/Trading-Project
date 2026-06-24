import yfinance as yf
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Download data
    df = yf.download("AAPL", start="2020-01-01")

    # Daily log returns
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # 20-day rolling volatility
    df["vol_20d"] = df["log_return"].rolling(window=20).std()

    # Annualized volatility (assuming 252 trading days)
    df["vol_20d_annualized"] = df["vol_20d"] * np.sqrt(252)

    print(df[["Close", "vol_20d", "vol_20d_annualized"]].tail())

    # Multiple timeframes
    for window in [5, 10, 20, 60]:
        df[f"vol_{window}d"] = (
            df["log_return"]
            .rolling(window)
            .std()
            * np.sqrt(252)
        )

    # Volitility Ratio
    df["vol_ratio_20d_60d"] = df["vol_20d"] / df["vol_60d"]

    # Rolling Variance
    df["var_20d"] = df["log_return"].rolling(window=20).var()

    # Exponentially weighted volatility
    df["vol_ewma"] = df["log_return"].ewm(span=20).std() * np.sqrt(252)

    # Realized volatility (sum of squared log returns)
    df["realized_vol_20d"] = (
        df["log_return"].rolling(window=20).apply(lambda x: np.sqrt(np.sum(x**2)))
    )
    df["realized_vol_20d"] = np.sqrt(
        (df["log_return"] ** 2).rolling(20).sum()
    ) * np.sqrt(252 / 20)

    # Features for machine learning
    features = [
        "vol_5d",
        "vol_10d",
        "vol_20d",
        "vol_60d",
        "vol_ratio",
        "ewm_vol"
    ]

    # Extra
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)

    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()

    df["price_vs_ma20"] = df["Close"] / df["ma_20"]