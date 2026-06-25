import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


class Backtest:
    """Evaluate a signal-based trading strategy against a buy-and-hold benchmark.

    Expected input DataFrame columns:
        - 'Close'  : daily closing prices
        - 'Signal' : integer position signals (-1 = short/out, 0 = flat, 1 = long)

    A transaction cost of 0.1% is deducted on every day the signal changes
    (each signal change is counted as one trade).

    Usage::

        bt = Backtest(df)
        metrics = bt.run()
        bt.plot_results()
    """

    TRADING_DAYS: int = 252   # assumed trading days per year for annualization
    COST_PER_TRADE: float = 0.001  # 0.1% round-trip cost per signal change

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Args:
            df: DataFrame with 'Close' and 'Signal' columns.

        Raises:
            ValueError: If required columns are missing.
        """
        missing = {"Close", "Signal"} - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        self.df = df.copy()
        self._results: Dict[str, float] = {}

    def _compute_strategy_returns(self, market_return: pd.Series) -> pd.Series:
        """Compute daily strategy returns with transaction costs.

        Signal is shifted one day forward to prevent lookahead bias.
        A cost of COST_PER_TRADE is deducted on every day the position changes.

        Returns:
            Series of daily net strategy returns.
        """
        # Lag the signal so we can only trade on the *next* day's open
        position = self.df["Signal"].shift(1)

        # Any day where the signal changes incurs a transaction cost
        trade_days = self.df["Signal"].diff().abs() > 0
        cost = trade_days.astype(float) * self.COST_PER_TRADE

        return position * market_return - cost

    def run(self) -> Dict[str, float]:
        """Run the backtest and return a dictionary of performance metrics.

        Metrics returned:
            total_return        : Net strategy return over the period (decimal).
            buy_and_hold_return : Passive buy-and-hold return over the period.
            sharpe_ratio        : Annualised Sharpe ratio (risk-free rate = 0).
            max_drawdown        : Largest peak-to-trough decline (negative decimal).
            win_rate            : Fraction of active trading days with a positive return.

        Returns:
            Dict mapping metric name to float value.
        """
        market_returns = self.df["Close"].pct_change()
        strategy_returns = self._compute_strategy_returns(market_returns)

        # Cumulative wealth curves (start at $1)
        cum_strategy = (1 + strategy_returns).cumprod()
        cum_market = (1 + market_returns).cumprod()

        total_return = float(cum_strategy.iloc[-1] - 1)
        buy_and_hold_return = float(cum_market.iloc[-1] - 1)

        # Annualised Sharpe ratio (assumes risk-free rate of 0)
        ann_return = strategy_returns.mean() * self.TRADING_DAYS
        ann_vol = strategy_returns.std() * np.sqrt(self.TRADING_DAYS)
        sharpe_ratio = float(ann_return / ann_vol) if ann_vol != 0 else 0.0

        # Maximum drawdown: the deepest peak-to-trough decline in the equity curve
        rolling_peak = cum_strategy.cummax()
        drawdown = (cum_strategy - rolling_peak) / rolling_peak
        max_drawdown = float(drawdown.min())

        # Win rate: fraction of days with an active position that earned a gain
        active = strategy_returns[strategy_returns != 0].dropna()
        win_rate = float((active > 0).mean()) if len(active) > 0 else 0.0

        self._results = {
            "total_return": round(total_return, 4),
            "buy_and_hold_return": round(buy_and_hold_return, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "max_drawdown": round(max_drawdown, 4),
            "win_rate": round(win_rate, 4),
        }

        # Cache curves for plotting
        self._cum_strategy = cum_strategy
        self._cum_market = cum_market
        self._drawdown = drawdown

        return self._results

    def plot_results(self) -> None:
        """Plot cumulative returns and the strategy drawdown in two subplots.

        Raises:
            RuntimeError: If run() has not been called yet.
        """
        if not self._results:
            raise RuntimeError("Call run() before plot_results().")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # --- Top panel: cumulative returns ---
        ax1.plot(self._cum_market.index, self._cum_market, label="Buy & Hold", alpha=0.8)
        ax1.plot(self._cum_strategy.index, self._cum_strategy, label="Strategy", alpha=0.8)
        ax1.set_title("Cumulative Returns: Strategy vs Buy & Hold")
        ax1.set_ylabel("Growth of $1")
        ax1.legend()
        ax1.grid(True)

        # --- Bottom panel: drawdown (shaded area below zero) ---
        ax2.fill_between(
            self._drawdown.index, self._drawdown, 0, color="red", alpha=0.4
        )
        ax2.set_title("Strategy Drawdown")
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
