"""
AAPL SMA Crossover Strategy — Demonstration Script
====================================================
Downloads 5 years of Apple (AAPL) price data from Yahoo Finance, applies a
20/50-day simple moving average crossover strategy, then evaluates the result
with the Backtest class.  Momentum features are also computed and printed as
an example of the momentum module's importable functions.

Strategy logic:
    - Signal = +1 (long)  when the 20-day SMA crosses above the 50-day SMA.
    - Signal = -1 (out)   when the 20-day SMA crosses below the 50-day SMA.

Backtest metrics printed:
    - Total strategy return vs buy-and-hold
    - Annualised Sharpe ratio (risk-free rate = 0, 252 trading days assumed)
    - Maximum drawdown
    - Win rate (% of active trading days with a positive return)
    - Transaction costs of 0.1% per signal change are baked into returns.
"""

import yfinance as yf
from moving_avg import compute_sma_signals, compute_strategy_returns, plot_sma_signals, plot_cumulative_returns
from backtest import Backtest
from momentum import calculate_all_features, ML_FEATURES
from volatility import compute_all_volatility_features, VOL_FEATURES

# --- Configuration ---
TICKER = "AAPL"
START = "2020-01-01"
END = "2025-01-01"
SHORT_WINDOW = 20   # fast SMA lookback (days)
LONG_WINDOW = 50    # slow SMA lookback (days)

# Download historical price data
print(f"Downloading {TICKER} data from {START} to {END}...")
raw = yf.download(TICKER, start=START, end=END, progress=False)

# Compute SMA crossover signals
df = compute_sma_signals(raw, short_window=SHORT_WINDOW, long_window=LONG_WINDOW)
df = compute_strategy_returns(df)

# Run the backtest and collect metrics
bt = Backtest(df)
metrics = bt.run()

# Print a formatted summary table
print(f"\n{'=' * 46}")
print(f"  Backtest: {TICKER}  |  {START} → {END}")
print(f"  Strategy: SMA {SHORT_WINDOW}/{LONG_WINDOW} Crossover")
print(f"{'=' * 46}")
for key, value in metrics.items():
    label = key.replace("_", " ").title()
    if key in ("total_return", "buy_and_hold_return", "max_drawdown", "win_rate"):
        print(f"  {label:<26} {value:>8.2%}")
    else:
        print(f"  {label:<26} {value:>8.4f}")
print(f"{'=' * 46}\n")

# Display SMA signal chart and cumulative-returns comparison
plot_sma_signals(df, ticker=TICKER, short_window=SHORT_WINDOW, long_window=LONG_WINDOW)
plot_cumulative_returns(df)

# Display backtest drawdown chart
bt.plot_results()

# --- Momentum features ---
print("Computing momentum features...")
mom_df = calculate_all_features(raw.copy())
print(f"\nMomentum features (last 5 rows):\n{mom_df[ML_FEATURES].tail()}\n")

# --- Volatility features ---
print("Computing volatility features...")
vol_df = compute_all_volatility_features(raw.copy())
print(f"\nVolatility features (last 5 rows):\n{vol_df[VOL_FEATURES].tail()}\n")
