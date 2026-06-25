import pandas as pd
import numpy as np
import pytest

from backtest import Backtest
from moving_avg import compute_sma_signals
from sma import detect_crossover_signals, compute_avg, compute_sma


def _make_price_series(prices, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="B")
    return pd.DataFrame({"Close": prices, "Open": prices}, index=idx)


# ---------------------------------------------------------------------------
# Backtest.run() — synthetic series with known outcome
# ---------------------------------------------------------------------------

class TestBacktestRun:
    def test_buy_and_hold_matches_price_change(self):
        # Steadily rising prices; signal always 1 (long)
        prices = list(range(100, 130))
        df = _make_price_series(prices)
        df["Signal"] = 1
        bt = Backtest(df)
        metrics = bt.run()
        expected_bah = (prices[-1] - prices[0]) / prices[0]
        assert abs(metrics["buy_and_hold_return"] - expected_bah) < 1e-4

    def test_flat_signal_zero_strategy_return(self):
        # Signal = 0 the whole time → strategy earns nothing
        prices = [100, 110, 105, 115, 120]
        df = _make_price_series(prices)
        df["Signal"] = 0
        metrics = Backtest(df).run()
        assert metrics["total_return"] == pytest.approx(0.0, abs=1e-6)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"Close": [100, 110]})
        with pytest.raises(ValueError):
            Backtest(df)

    def test_win_rate_all_up_days(self):
        # Monotonically rising, always long → all active days are wins
        prices = [100 + i for i in range(20)]
        df = _make_price_series(prices)
        df["Signal"] = 1
        metrics = Backtest(df).run()
        assert metrics["win_rate"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_sma_signals — crossover signal values
# ---------------------------------------------------------------------------

class TestComputeSmaSignals:
    def _crossover_df(self):
        # First 20 bars below the 50-bar SMA, then jump above — forces a crossover
        prices = [90] * 30 + [150] * 30
        idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
        return pd.DataFrame({"Close": prices}, index=idx)

    def test_signal_values_are_plus_minus_one_or_zero(self):
        df = self._crossover_df()
        result = compute_sma_signals(df)
        assert set(result["Signal"].dropna().unique()).issubset({-1, 0, 1})

    def test_crossover_produces_position_change(self):
        df = self._crossover_df()
        result = compute_sma_signals(df)
        # At least one buy crossover (+2) must appear after the price jump
        assert (result["Position"] == 2).any()


# ---------------------------------------------------------------------------
# detect_crossover_signals — marks exactly the crossover days
# ---------------------------------------------------------------------------

class TestDetectCrossoverSignals:
    def _df_with_smas(self):
        # SMA_20 < SMA_100 for first 5 rows, then flips above — one crossover
        n = 10
        idx = pd.date_range("2021-01-01", periods=n, freq="B")
        sma_short = [90, 90, 90, 90, 90, 110, 110, 110, 110, 110]
        sma_long  = [100] * n
        return pd.DataFrame({"SMA_20": sma_short, "SMA_100": sma_long}, index=idx)

    def test_exactly_one_crossover_detected(self):
        df = self._df_with_smas()
        result = detect_crossover_signals(df)
        assert result["Signal"].sum() == 1

    def test_crossover_on_correct_day(self):
        df = self._df_with_smas()
        result = detect_crossover_signals(df)
        crossover_day = result.index[result["Signal"]].tolist()
        # The flip happens between index[4] and index[5]; shift(1) marks index[5]
        assert crossover_day == [result.index[5]]

    def test_hold_true_when_short_above_long(self):
        df = self._df_with_smas()
        result = detect_crossover_signals(df)
        assert result["hold"].iloc[-1] is True or result["hold"].iloc[-1] == True
        assert result["hold"].iloc[0] is False or result["hold"].iloc[0] == False
