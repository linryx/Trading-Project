# Trading Project

A Python algorithmic trading toolkit that implements signal-based strategies, backtests them against a buy-and-hold benchmark, and reports performance analytics.

## Strategies Implemented

| Strategy | Module | Description |
|---|---|---|
| SMA Crossover | `moving_avg.py` | Go long when the 20-day SMA crosses above the 50-day SMA; exit when it crosses below. Signals are generated with `compute_sma_signals()`. |
| Momentum | `momentum.py` | Rate-of-change momentum over configurable lookback periods (default 5d, 20d, 60d) via `calculate_momentum()`. Log-return variants, RSI, price-vs-MA ratios, cross-sectional rank, and risk-adjusted momentum are also available. `calculate_all_features()` runs all of them and produces the columns listed in `ML_FEATURES`. |
| Volatility Features | `volatility.py` | Rolling and EWMA volatility across multiple timeframes (5d, 10d, 20d, 60d), rolling variance, realized volatility, and a short/long volatility ratio for regime detection. `compute_all_volatility_features()` runs the full pipeline and produces the columns listed in `VOL_FEATURES`. |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

This downloads 5 years of AAPL data (2020–2025), runs the SMA 20/50 crossover strategy, prints the backtest metrics table, opens the performance charts, and prints sample momentum and volatility feature tables.

To run individual modules standalone:

```bash
python moving_avg.py   # plots AAPL SMA 20/50 signals and cumulative returns
python volatility.py   # prints volatility feature table for AAPL
python sma.py          # interactive: prompts for year and ticker, prints balance sheet and SMA 20/100 buy signals
```

## Project Structure

```
.
├── main.py          # Entry point — downloads data, runs strategy, prints metrics and feature tables
├── moving_avg.py    # SMA 20/50 crossover signal generation, strategy returns, and charts
├── momentum.py      # Momentum feature engineering (simple, log-return, RSI, cross-sectional, ML target)
├── volatility.py    # Volatility feature engineering (rolling, EWMA, realized, regime ratio)
├── sma.py           # Interactive SMA 20/100 scanner with balance sheet lookup and buy signal printer
├── backtest.py      # Backtest class — performance metrics and cumulative-return/drawdown plots
├── requirements.txt
└── README.md
```

## Importable Functions

### `moving_avg.py`

| Function | Description |
|---|---|
| `compute_sma_signals(df, short_window, long_window)` | Adds `SMA_Short`, `SMA_Long`, `Signal` (+1/-1), and `Position` (crossover events) columns. |
| `compute_strategy_returns(df)` | Adds daily and cumulative return columns vs buy-and-hold. |
| `plot_sma_signals(data, ticker, short_window, long_window)` | Plots price with SMA lines and buy/sell markers. |
| `plot_cumulative_returns(data)` | Plots strategy vs buy-and-hold cumulative return. |

### `momentum.py`

| Function | Description |
|---|---|
| `calculate_momentum(df, periods)` | Percent-change momentum for each period in `periods`. |
| `calculate_log_momentum(df)` | Rolling 20d and 60d log-return sums. |
| `calculate_price_vs_ma(df, windows)` | Price/MA ratios and fast-minus-slow MA spread. |
| `calculate_cross_sectional_rank(prices_df)` | 60d percentile rank across a multi-ticker price DataFrame. |
| `calculate_risk_adjusted_momentum(df, mom_window, vol_windows)` | Momentum divided by rolling volatility. |
| `calculate_rsi(df, period)` | 14-day RSI (overbought >70, oversold <30). |
| `add_ml_target(df, forward_days)` | Forward-return target column for supervised ML. |
| `calculate_all_features(df, periods)` | Runs all of the above; produces columns in `ML_FEATURES`. |

`ML_FEATURES`: `mom_5d`, `mom_20d`, `mom_60d`, `log_mom_20d`, `log_mom_60d`, `vol_20d`, `vol_60d`, `price_vs_ma20`, `price_vs_ma50`, `ma_spread`, `risk_adj_mom`, `rsi`

### `volatility.py`

| Function | Description |
|---|---|
| `compute_log_returns(df)` | Adds `log_return` column. |
| `compute_rolling_volatility(df, windows)` | Annualised rolling std-dev for each window. |
| `compute_volatility_ratio(df, short_window, long_window)` | Short/long vol ratio for regime detection. |
| `compute_variance(df, window)` | Rolling variance (not annualised). |
| `compute_ewma_volatility(df, span)` | EWMA volatility, annualised. |
| `compute_realized_volatility(df, window)` | Realized vol from squared log returns, annualised. |
| `compute_price_features(df, ma_windows)` | Short-term returns and price/MA ratios. |
| `compute_all_volatility_features(df)` | Runs all of the above; produces columns in `VOL_FEATURES`. |

`VOL_FEATURES`: `vol_5d`, `vol_10d`, `vol_20d`, `vol_60d`, `vol_ratio_20d_60d`, `var_20d`, `vol_ewma`, `realized_vol_20d`

### `sma.py`

| Function | Description |
|---|---|
| `compute_avg(df)` | Adds `AVG` column (Open/Close midpoint). |
| `compute_sma(df, short_window, long_window)` | Adds `SMA_20` and `SMA_100` on the `AVG` column. |
| `filter_to_year(df, year, warm_up_rows)` | Filters to a calendar year and drops warm-up rows. |
| `detect_crossover_signals(df, short_col, long_col)` | Adds `hold` (uptrend flag) and `Signal` (crossover day) columns. |
| `print_buy_signals(df)` | Prints Close price on every bullish crossover. |

## Performance Metrics

Results from the SMA 20/50 crossover strategy on AAPL (2020–2025), with 0.1% transaction costs per trade:

| Metric | Value |
|---|---|
| Total Return | — |
| Buy & Hold Return | — |
| Sharpe Ratio | — |
| Max Drawdown | — |
| Win Rate | — |

> Run `python main.py` to populate this table with live results.
