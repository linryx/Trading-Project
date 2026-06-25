import pandas as pd
import numpy as np
from typing import List


# Feature columns produced by calculate_all_features(), ready to pass to an ML model.
ML_FEATURES: List[str] = [
    "mom_5d",
    "mom_20d",
    "mom_60d",
    "log_mom_20d",
    "log_mom_60d",
    "vol_20d",
    "vol_60d",
    "price_vs_ma20",
    "price_vs_ma50",
    "ma_spread",
    "risk_adj_mom",
    "rsi",
]


def calculate_momentum(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """Compute simple percent-change momentum over multiple lookback periods.

    Args:
        df: DataFrame with a 'Close' column.
        periods: Lookback windows in trading days (e.g. [5, 20, 60]).

    Returns:
        DataFrame with a 'mom_{n}d' column added for each period n.
    """
    for period in periods:
        df[f"mom_{period}d"] = df["Close"].pct_change(period)
    return df


def calculate_log_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling sums of daily log returns as momentum signals.

    Log-return sums are a smoother alternative to simple percent-change momentum
    because they are additive over time and less distorted by large single-day moves.

    Args:
        df: DataFrame with a 'Close' column.

    Returns:
        DataFrame with 'log_mom_20d' and 'log_mom_60d' columns added.
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    df["log_mom_20d"] = log_ret.rolling(20).sum()
    df["log_mom_60d"] = log_ret.rolling(60).sum()
    return df


def calculate_price_vs_ma(
    df: pd.DataFrame, windows: List[int] = [20, 50]
) -> pd.DataFrame:
    """Compute price-relative-to-MA ratios and the spread between MAs.

    A ratio above 1.0 means price is above its average — a simple trend filter.
    The MA spread (fast minus slow) is positive in an uptrend and negative in a downtrend.

    Args:
        df: DataFrame with a 'Close' column.
        windows: Ascending list of MA lookback windows (default [20, 50]).

    Returns:
        DataFrame with 'ma_{w}', 'price_vs_ma{w}', and 'ma_spread' columns added.
    """
    for w in windows:
        df[f"ma_{w}"] = df["Close"].rolling(w).mean()
        df[f"price_vs_ma{w}"] = df["Close"] / df[f"ma_{w}"]

    # Spread between the fastest and slowest MA; positive = short-term uptrend
    if len(windows) >= 2:
        df["ma_spread"] = df[f"ma_{windows[0]}"] - df[f"ma_{windows[-1]}"]

    return df


def calculate_cross_sectional_rank(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Rank each ticker by its 60-day return across a multi-ticker price DataFrame.

    Cross-sectional momentum selects assets that are strong *relative to peers*,
    rather than just strong in absolute terms.

    Args:
        prices_df: DataFrame whose columns are Close price series for different
                   tickers and whose index is dates (e.g. the output of
                   yf.download([...], ...)["Close"]).

    Returns:
        DataFrame of the same shape with percentile rank values (0–1) per row,
        where 1.0 means the highest 60-day return among all tickers on that date.
    """
    returns_60d = prices_df.pct_change(60)
    return returns_60d.rank(axis=1, pct=True)


def calculate_risk_adjusted_momentum(
    df: pd.DataFrame, mom_window: int = 20, vol_windows: List[int] = [20, 60]
) -> pd.DataFrame:
    """Compute momentum normalised by rolling volatility.

    Dividing raw momentum by recent volatility penalises choppy, noisy moves and
    rewards smooth trending ones — improving signal quality for mean-reversion and
    trend strategies alike.

    Args:
        df: DataFrame with a 'Close' column.
        mom_window: Lookback for the momentum numerator (default 20 days).
        vol_windows: Lookback windows for rolling volatility columns (default [20, 60]).

    Returns:
        DataFrame with 'mom_{mom_window}d', 'vol_{w}d' for each w, and
        'risk_adj_mom' columns added.
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    df[f"mom_{mom_window}d"] = df["Close"].pct_change(mom_window)

    for w in vol_windows:
        df[f"vol_{w}d"] = log_ret.rolling(w).std()

    # Normalise by the shortest vol window to get risk-adjusted momentum
    df["risk_adj_mom"] = df[f"mom_{mom_window}d"] / df[f"vol_{vol_windows[0]}d"]
    return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute the Relative Strength Index (RSI).

    RSI measures the speed and magnitude of recent price changes.
    Conventional thresholds: >70 = overbought, <30 = oversold.

    Args:
        df: DataFrame with a 'Close' column.
        period: Smoothing window for average gain/loss (default 14 days).

    Returns:
        DataFrame with a 'rsi' column added (values between 0 and 100).
    """
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_ml_target(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """Add a forward-return target column for supervised ML training.

    The target is the percent change in Close over the next `forward_days` days,
    shifted back so it aligns with the features on each row.

    Args:
        df: DataFrame with a 'Close' column.
        forward_days: Number of trading days ahead to predict (default 5).

    Returns:
        DataFrame with a 'target' column added.
    """
    df["target"] = df["Close"].pct_change(forward_days).shift(-forward_days)
    return df


def calculate_all_features(
    df: pd.DataFrame, periods: List[int] = [5, 20, 60]
) -> pd.DataFrame:
    """Run all momentum feature functions in sequence and add an ML target.

    Produces every column listed in ML_FEATURES plus a 'target' column.

    Args:
        df: DataFrame with a 'Close' column.
        periods: Lookback windows passed to calculate_momentum (default [5, 20, 60]).

    Returns:
        DataFrame with all momentum, volatility, and signal columns added.
    """
    df = calculate_momentum(df, periods)
    df = calculate_log_momentum(df)
    df = calculate_price_vs_ma(df, windows=[20, 50])
    df = calculate_risk_adjusted_momentum(df, mom_window=20, vol_windows=[20, 60])
    df = calculate_rsi(df)
    df = add_ml_target(df)
    return df
