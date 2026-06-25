import yfinance as yf
import pandas as pd
import numpy as np
from typing import List


# Core volatility feature columns produced by compute_all_volatility_features().
VOL_FEATURES: List[str] = [
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "vol_60d",
    "vol_ratio_20d_60d",
    "var_20d",
    "vol_ewma",
    "realized_vol_20d",
]


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns and add them as a 'log_return' column.

    Log returns are additive across time and better-behaved statistically
    than simple percent returns for volatility estimation.

    Args:
        df: DataFrame with a 'Close' column.

    Returns:
        DataFrame with a 'log_return' column added.
    """
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def compute_rolling_volatility(
    df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60]
) -> pd.DataFrame:
    """Compute annualized rolling standard-deviation volatility for multiple windows.

    Each window produces a 'vol_{w}d' column.  Values are annualized by
    multiplying by sqrt(252) — the square root of the assumed number of
    trading days per year.

    Requires a 'log_return' column; call compute_log_returns() first.

    Args:
        df: DataFrame with a 'log_return' column.
        windows: Lookback windows in trading days (default [5, 10, 20, 60]).

    Returns:
        DataFrame with 'vol_{w}d' columns added for each window w.
    """
    for w in windows:
        df[f"vol_{w}d"] = df["log_return"].rolling(w).std() * np.sqrt(252)
    return df


def compute_volatility_ratio(
    df: pd.DataFrame, short_window: int = 20, long_window: int = 60
) -> pd.DataFrame:
    """Compute the ratio of short-term to long-term volatility.

    A ratio above 1.0 signals a recent volatility spike relative to the
    longer-term baseline — useful as a regime-detection feature.

    Requires vol_{short_window}d and vol_{long_window}d columns; call
    compute_rolling_volatility() first.

    Args:
        df: DataFrame with the required vol columns already computed.
        short_window: Numerator vol window (default 20).
        long_window: Denominator vol window (default 60).

    Returns:
        DataFrame with a 'vol_ratio_{short}d_{long}d' column added.
    """
    col = f"vol_ratio_{short_window}d_{long_window}d"
    df[col] = df[f"vol_{short_window}d"] / df[f"vol_{long_window}d"]
    return df


def compute_variance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute rolling variance of log returns (not annualized).

    Raw variance complements the annualized standard-deviation columns when
    features are consumed by ML models that perform their own scaling.

    Requires a 'log_return' column; call compute_log_returns() first.

    Args:
        df: DataFrame with a 'log_return' column.
        window: Rolling window in trading days (default 20).

    Returns:
        DataFrame with a 'var_{window}d' column added.
    """
    df[f"var_{window}d"] = df["log_return"].rolling(window=window).var()
    return df


def compute_ewma_volatility(df: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """Compute exponentially weighted moving average (EWMA) volatility.

    EWMA weights recent observations more heavily than a simple rolling window,
    making it more responsive to sudden volatility changes.  Annualized by sqrt(252).

    Requires a 'log_return' column; call compute_log_returns() first.

    Args:
        df: DataFrame with a 'log_return' column.
        span: EWM span parameter — controls the decay speed (default 20).

    Returns:
        DataFrame with a 'vol_ewma' column added.
    """
    df["vol_ewma"] = df["log_return"].ewm(span=span).std() * np.sqrt(252)
    return df


def compute_realized_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute realized volatility as the square root of summed squared log returns.

    Unlike standard-deviation vol, realized vol does not subtract the mean,
    making it a cleaner high-frequency risk estimate.  Annualized by sqrt(252/window).

    Requires a 'log_return' column; call compute_log_returns() first.

    Args:
        df: DataFrame with a 'log_return' column.
        window: Rolling window in trading days (default 20).

    Returns:
        DataFrame with a 'realized_vol_{window}d' column added.
    """
    df[f"realized_vol_{window}d"] = (
        np.sqrt((df["log_return"] ** 2).rolling(window).sum())
        * np.sqrt(252 / window)
    )
    return df


def compute_price_features(
    df: pd.DataFrame, ma_windows: List[int] = [20, 50]
) -> pd.DataFrame:
    """Compute short-term return features and price-relative-to-MA ratios.

    Provides context features that pair naturally with volatility columns
    in ML pipelines — e.g. a stock can be high-vol in an uptrend or a downtrend.

    Args:
        df: DataFrame with a 'Close' column.
        ma_windows: Moving average windows for the price/MA ratios (default [20, 50]).

    Returns:
        DataFrame with 'ret_5d', 'ret_20d', 'ma_{w}', and 'price_vs_ma{w}' columns added.
    """
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)
    for w in ma_windows:
        df[f"ma_{w}"] = df["Close"].rolling(w).mean()
        df[f"price_vs_ma{w}"] = df["Close"] / df[f"ma_{w}"]
    return df


def compute_all_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all volatility feature functions in sequence.

    Produces every column listed in VOL_FEATURES plus supplementary price
    and return features from compute_price_features().

    Args:
        df: DataFrame with 'Close' (and 'Open' if using price midpoint elsewhere).

    Returns:
        DataFrame with all volatility and price feature columns added.
    """
    df = compute_log_returns(df)
    df = compute_rolling_volatility(df, windows=[5, 10, 20, 60])
    df = compute_volatility_ratio(df, short_window=20, long_window=60)
    df = compute_variance(df, window=20)
    df = compute_ewma_volatility(df, span=20)
    df = compute_realized_volatility(df, window=20)
    df = compute_price_features(df, ma_windows=[20, 50])
    return df


if __name__ == "__main__":
    df = yf.download("AAPL", start="2020-01-01")
    df = compute_all_volatility_features(df)
    print(df[VOL_FEATURES].tail())
