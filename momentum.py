import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_momentum(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    """Compute simple price momentum over multiple lookback periods.

    Args:
        df: DataFrame with a 'Close' column.
        periods: List of integer lookback windows (e.g. [5, 10, 20]).

    Returns:
        DataFrame with a 'mom_{n}d' column added for each period n.
    """
    for period in periods:
        df[f"mom_{period}d"] = df["Close"].pct_change(period)
    return df


def calculate_log_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log-return-based momentum features and add them to the dataframe.

    Includes rolling log-momentum sums, price-vs-MA ratios, risk-adjusted
    momentum (momentum / volatility), RSI, and a 5-day forward return target
    suitable for ML use.

    Args:
        df: DataFrame with a 'Close' column.

    Returns:
        DataFrame with log-momentum feature columns added.
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1))

    # Rolling sums of log returns as momentum signals
    df["log_mom_20d"] = log_ret.rolling(20).sum()
    df["log_mom_60d"] = log_ret.rolling(60).sum()

    # Price relative to its own moving averages (trend position)
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["price_vs_ma20"] = df["Close"] / df["ma_20"]
    df["price_vs_ma50"] = df["Close"] / df["ma_50"]

    # Risk-adjusted momentum: normalise by recent vol to penalise noisy moves
    df["mom_20d"] = df["Close"].pct_change(20)
    df["vol_20d"] = log_ret.rolling(20).std()
    df["risk_adj_mom"] = df["mom_20d"] / df["vol_20d"]

    # RSI (14-day): overbought >70, oversold <30
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # ML target: next 5-day forward return
    df["target"] = df["Close"].pct_change(5).shift(-5)

    return df
