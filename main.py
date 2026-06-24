"""
AAPL SMA Crossover Strategy — Demonstration Script
====================================================
Downloads 5 years of Apple (AAPL) price data from Yahoo Finance, applies a
20/50-day simple moving average crossover strategy, then evaluates the result
with the Backtest class.

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
from moving_avg import compute_sma_signals
from backtest import Backtest

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

# Display cumulative-returns and drawdown charts
bt.plot_results()
