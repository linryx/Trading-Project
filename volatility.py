import yfinance as yf
import pandas as pd
import numpy as np


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling volatility features and add them to the dataframe.

    Computes log returns, rolling standard-deviation volatility across multiple
    timeframes, a short/long volatility ratio (regime signal), EWMA volatility,
    and realized volatility.  All vol figures are annualized assuming 252 trading days.

    Args:
        df: DataFrame with a 'Close' column.

    Returns:
        DataFrame with volatility feature columns added.
    """
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Rolling volatility across timeframes — loop runs BEFORE the ratio so
    # vol_60d exists when vol_ratio_20d_60d is calculated below.
    for window in [5, 10, 20, 60]:
        df[f"vol_{window}d"] = (
            df["log_return"]
            .rolling(window)
            .std()
            * np.sqrt(252)
        )

    # Short/long vol ratio: values > 1 indicate a recent volatility spike
    df["vol_ratio_20d_60d"] = df["vol_20d"] / df["vol_60d"]

    # Rolling variance (not annualized — raw daily variance)
    df["var_20d"] = df["log_return"].rolling(window=20).var()

    # Exponentially weighted volatility (more weight on recent observations)
    df["vol_ewma"] = df["log_return"].ewm(span=20).std() * np.sqrt(252)

    # Realized volatility: square root of mean squared log returns, annualized
    df["realized_vol_20d"] = (
        np.sqrt((df["log_return"] ** 2).rolling(20).sum()) * np.sqrt(252 / 20)
    )

    # Short-term return features useful alongside vol in ML pipelines
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)

    # Price relative to moving averages
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["price_vs_ma20"] = df["Close"] / df["ma_20"]

    return df


if __name__ == "__main__":
    df = yf.download("AAPL", start="2020-01-01")
    df = compute_volatility_features(df)

    # Feature columns for machine learning
    features = [
        "vol_5d",
        "vol_10d",
        "vol_20d",
        "vol_60d",
        "vol_ratio_20d_60d",  # Fixed: was 'vol_ratio' (column doesn't exist)
        "vol_ewma",           # Fixed: was 'ewm_vol' (column doesn't exist)
    ]

    print(df[features].tail())
