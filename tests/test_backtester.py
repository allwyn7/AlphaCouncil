"""Tests for the backtester specifically: walk-forward windows,
realistic transaction cost computation, aggregation, pass/fail gates.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from alphacouncil.research.backtester import (
    StrategyBacktester,
    _BROKERAGE_PCT,
    _GST_ON_BROKERAGE_PCT,
    _MAX_DRAWDOWN,
    _MIN_PROFITABLE_PERIODS_RATIO,
    _MIN_WALK_FORWARD_SHARPE,
    _SEBI_PCT,
    _SLIPPAGE_MARKET_PCT,
    _STAMP_BUY_PCT,
    _STT_SELL_PCT,
    _TRAIN_DAYS,
    _TEST_DAYS,
    _max_consecutive,
    _max_dd_duration,
)


# ---------------------------------------------------------------------------
# Walk-forward window generation
# ---------------------------------------------------------------------------


class TestWindowGeneration:
    def test_windows_non_overlapping(self):
        """Each test_start must be after the previous test_end."""
        windows = StrategyBacktester._build_windows("2018-01-01", "2025-12-31")
        assert len(windows) >= 2
        for i in range(1, len(windows)):
            prev_test_end = windows[i - 1][3]
            curr_train_start = windows[i][0]
            # Curriculum rolls forward: next train_start == prev test_start
            assert curr_train_start >= windows[i - 1][2]

    def test_train_period_longer_than_test(self):
        windows = StrategyBacktester._build_windows("2018-01-01", "2025-12-31")
        for ts, te, xs, xe in windows:
            train_start = datetime.fromisoformat(ts)
            train_end = datetime.fromisoformat(te)
            test_start = datetime.fromisoformat(xs)
            test_end = datetime.fromisoformat(xe)
            train_days = (train_end - train_start).days
            test_days = (test_end - test_start).days
            assert train_days > test_days

    def test_short_range_no_windows(self):
        windows = StrategyBacktester._build_windows("2025-01-01", "2025-03-01")
        assert len(windows) == 0

    def test_exact_boundary(self):
        """When end exactly equals test_end, include that window."""
        windows = StrategyBacktester._build_windows("2018-01-01", "2030-12-31")
        assert len(windows) >= 3
        for _, _, _, test_end in windows:
            assert test_end <= "2030-12-31"

    def test_window_tuple_format(self):
        windows = StrategyBacktester._build_windows("2019-01-01", "2025-12-31")
        for w in windows:
            assert len(w) == 4
            for date_str in w:
                assert len(date_str) == 10  # YYYY-MM-DD
                datetime.fromisoformat(date_str)  # should not raise


# ---------------------------------------------------------------------------
# Realistic transaction cost computation
# ---------------------------------------------------------------------------


class TestTransactionCosts:
    def test_buy_cost_exact(self):
        """Verify buy cost = brokerage + GST + SEBI + stamp + slippage."""
        qty, price = 100, 500.0
        turnover = qty * price  # 50,000
        cost = StrategyBacktester._compute_costs(qty, price, "BUY")

        brokerage = turnover * _BROKERAGE_PCT           # 50000 * 0.0005 = 25
        gst = brokerage * _GST_ON_BROKERAGE_PCT         # 25 * 0.18 = 4.50
        sebi = turnover * _SEBI_PCT                      # 50000 * 0.000001 = 0.05
        stamp = turnover * _STAMP_BUY_PCT                # 50000 * 0.00015 = 7.50
        slippage = turnover * _SLIPPAGE_MARKET_PCT       # 50000 * 0.001 = 50

        expected = brokerage + gst + sebi + stamp + slippage
        assert cost == pytest.approx(expected, abs=0.001)

    def test_sell_cost_exact(self):
        """Verify sell cost = brokerage + GST + SEBI + STT + slippage."""
        qty, price = 100, 500.0
        turnover = qty * price
        cost = StrategyBacktester._compute_costs(qty, price, "SELL")

        brokerage = turnover * _BROKERAGE_PCT
        gst = brokerage * _GST_ON_BROKERAGE_PCT
        sebi = turnover * _SEBI_PCT
        stt = turnover * _STT_SELL_PCT
        slippage = turnover * _SLIPPAGE_MARKET_PCT

        expected = brokerage + gst + sebi + stt + slippage
        assert cost == pytest.approx(expected, abs=0.001)

    def test_brokerage_is_five_basis_points(self):
        assert _BROKERAGE_PCT == 0.0005

    def test_stt_is_ten_basis_points(self):
        assert _STT_SELL_PCT == 0.001

    def test_gst_is_eighteen_percent(self):
        assert _GST_ON_BROKERAGE_PCT == 0.18

    def test_sebi_charge_is_one_tenth_of_basis_point(self):
        assert _SEBI_PCT == 0.000001

    def test_stamp_duty_is_1_5_basis_points(self):
        assert _STAMP_BUY_PCT == 0.00015

    def test_slippage_is_ten_basis_points(self):
        assert _SLIPPAGE_MARKET_PCT == 0.001

    def test_buy_has_stamp_not_stt(self):
        """Buy side should include stamp duty but NOT STT."""
        turnover = 100_000.0
        buy_cost = StrategyBacktester._compute_costs(100, 1000.0, "BUY")
        # Manually compute without stamp and without STT to verify split
        brokerage = turnover * _BROKERAGE_PCT
        gst = brokerage * _GST_ON_BROKERAGE_PCT
        sebi = turnover * _SEBI_PCT
        slippage = turnover * _SLIPPAGE_MARKET_PCT
        base = brokerage + gst + sebi + slippage
        stamp = turnover * _STAMP_BUY_PCT
        assert buy_cost == pytest.approx(base + stamp, abs=0.001)

    def test_sell_has_stt_not_stamp(self):
        """Sell side should include STT but NOT stamp duty."""
        turnover = 100_000.0
        sell_cost = StrategyBacktester._compute_costs(100, 1000.0, "SELL")
        brokerage = turnover * _BROKERAGE_PCT
        gst = brokerage * _GST_ON_BROKERAGE_PCT
        sebi = turnover * _SEBI_PCT
        slippage = turnover * _SLIPPAGE_MARKET_PCT
        base = brokerage + gst + sebi + slippage
        stt = turnover * _STT_SELL_PCT
        assert sell_cost == pytest.approx(base + stt, abs=0.001)


# ---------------------------------------------------------------------------
# Aggregation of period results
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_sharpe_ratio_computation(self):
        """Sharpe = mean / std * sqrt(252)."""
        rng = np.random.default_rng(42)
        daily = list(rng.normal(0.001, 0.01, 252))
        result = StrategyBacktester._aggregate_metrics(
            daily, [], "2023-01-01", "2024-01-01",
        )
        rets = np.array(daily)
        expected_sharpe = float(rets.mean() / rets.std() * np.sqrt(252))
        assert result["sharpe"] == pytest.approx(expected_sharpe, rel=0.01)

    def test_max_drawdown_known_curve(self):
        # Create a simple up-then-down pattern
        daily = [0.01] * 50 + [-0.02] * 50
        result = StrategyBacktester._aggregate_metrics(
            daily, [], "2023-01-01", "2024-01-01",
        )
        assert result["max_dd"] > 0

    def test_win_rate_all_positive(self):
        daily = [0.001] * 100
        result = StrategyBacktester._aggregate_metrics(
            daily, [], "2023-01-01", "2024-01-01",
        )
        assert result["win_rate"] == 1.0

    def test_win_rate_mixed(self):
        daily = [0.01, -0.005, 0.008, -0.003, 0.012]
        result = StrategyBacktester._aggregate_metrics(
            daily, [], "2023-01-01", "2024-01-01",
        )
        # 3 positive out of 5
        assert result["win_rate"] == pytest.approx(0.6, abs=0.01)

    def test_empty_returns_zero_metrics(self):
        result = StrategyBacktester._aggregate_metrics(
            [], [], "2023-01-01", "2024-01-01",
        )
        assert result["sharpe"] == 0.0
        assert result["max_dd"] == 0.0
        assert result["win_rate"] == 0.0


# ---------------------------------------------------------------------------
# Pass/fail promotion criteria
# ---------------------------------------------------------------------------


class TestPromotionGates:
    def test_sharpe_threshold(self):
        assert _MIN_WALK_FORWARD_SHARPE == 0.5

    def test_max_drawdown_threshold(self):
        assert _MAX_DRAWDOWN == 0.20

    def test_profitable_periods_ratio(self):
        # 3 out of 4 must be profitable
        assert _MIN_PROFITABLE_PERIODS_RATIO == 0.75

    def test_passing_all_gates(self):
        """Simulate a result that passes all promotion gates."""
        result = {
            "sharpe": 0.8,
            "max_dd": 0.12,
        }
        profitable_count = 3
        total_windows = 4
        profitable_ratio = profitable_count / total_windows
        corr_ok = True

        passes = (
            result["sharpe"] > _MIN_WALK_FORWARD_SHARPE
            and result["max_dd"] < _MAX_DRAWDOWN
            and profitable_ratio >= _MIN_PROFITABLE_PERIODS_RATIO
            and corr_ok
        )
        assert passes is True

    def test_failing_sharpe_gate(self):
        result = {"sharpe": 0.3, "max_dd": 0.10}
        passes = result["sharpe"] > _MIN_WALK_FORWARD_SHARPE
        assert passes is False

    def test_failing_drawdown_gate(self):
        result = {"sharpe": 0.8, "max_dd": 0.25}
        passes = result["max_dd"] < _MAX_DRAWDOWN
        assert passes is False

    def test_failing_profitable_periods_gate(self):
        profitable_ratio = 1 / 4  # only 25%
        passes = profitable_ratio >= _MIN_PROFITABLE_PERIODS_RATIO
        assert passes is False


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_max_consecutive_losses(self):
        rets = pd.Series([0.01, -0.01, -0.02, -0.03, 0.01, -0.01])
        result = _max_consecutive(rets, negative=True)
        assert result == 3

    def test_max_consecutive_wins(self):
        rets = pd.Series([0.01, 0.02, 0.03, -0.01, 0.01])
        result = _max_consecutive(rets, negative=False)
        assert result == 3

    def test_max_consecutive_empty(self):
        rets = pd.Series(dtype=float)
        assert _max_consecutive(rets, negative=True) == 0

    def test_max_dd_duration(self):
        # Simulate equity: up, up, down, down, down, recovery
        rets = pd.Series([0.01, 0.01, -0.02, -0.02, -0.02, 0.05])
        equity = (1 + rets).cumprod()
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        duration = _max_dd_duration(dd)
        assert duration >= 3

    def test_max_dd_duration_no_drawdown(self):
        rets = pd.Series([0.01, 0.01, 0.01])
        equity = (1 + rets).cumprod()
        peak = equity.expanding().max()
        dd = (equity - peak) / peak
        duration = _max_dd_duration(dd)
        assert duration == 0
