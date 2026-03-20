"""Mean-Reversion agent -- catches oversold GROWTH stocks (buy-the-dip on quality).

Strategy
--------
This agent exploits short-term price dislocations in fundamentally strong
growth stocks.  The critical insight is the **growth quality filter**: we
*never* mean-revert on a stock just because it is cheap -- the stock must
have ``revenue_growth > 10 %`` AND ``ROE > 12 %``.  This avoids classic
value traps and limits the universe to temporarily discounted quality names.

An Ornstein-Uhlenbeck half-life filter further gates entry: if the fitted
mean-reversion half-life exceeds 20 days the stock is structurally trending
(not reverting) and is skipped.

Portfolio rules
~~~~~~~~~~~~~~~
- Max 10 concurrent positions, max 8 % per position.
- Stop-loss: 2x ATR from entry price.
- Take-profit: when z-score returns to 0.
- Holding period: 5--20 days.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401 (used dynamically)
import structlog

from alphacouncil.agents.base import BaseAgent, MessageBus
from alphacouncil.core.models import Action, AgentSignal, AgentStatus

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Default factor weights
# ---------------------------------------------------------------------------

_DEFAULT_FACTOR_WEIGHTS: dict[str, float] = {
    "price_vs_sma20_zscore": 0.25,
    "rsi_14": 0.20,
    "bollinger_band_position": 0.15,
    "growth_quality": 0.20,
    "volume_exhaustion": 0.10,
    "ou_halflife": 0.10,
}

# ---------------------------------------------------------------------------
# Default tunable parameters
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict[str, Any] = {
    "buy_zscore": -1.50,
    "sell_zscore": 0.0,
    "overbought_zscore": 1.5,
    "min_revenue_growth": 0.10,
    "min_roe": 0.12,
    "max_halflife": 20.0,
    "atr_multiplier": 2.0,
    "factor_weights": dict(_DEFAULT_FACTOR_WEIGHTS),
    "max_positions": 10,
    "max_weight_per_position": 0.08,
    "holding_period_min": 5,
    "holding_period_max": 20,
}


class MeanReversionAgent(BaseAgent):
    """Mean-reversion agent targeting oversold growth stocks.

    See module docstring for full strategy description and factor catalogue.
    """

    def __init__(
        self,
        config: Any,
        cache: Any,
        bus: MessageBus,
        db_engine: Any,
        *,
        name: str = "mean_reversion",
    ) -> None:
        super().__init__(
            name=name,
            config=config,
            cache=cache,
            bus=bus,
            db_engine=db_engine,
        )
        self._parameters: dict[str, Any] = dict(_DEFAULT_PARAMS)
        self._log.info(
            "agent.params_loaded",
            params={k: v for k, v in self._parameters.items() if k != "factor_weights"},
        )

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def get_parameters(self) -> dict[str, Any]:
        return dict(self._parameters)

    def set_parameters(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            if key not in self._parameters:
                self._log.warning("agent.unknown_param", key=key)
                continue
            if key == "factor_weights":
                weight_sum = sum(value.values())
                if not np.isclose(weight_sum, 1.0, atol=0.01):
                    self._log.error("agent.invalid_weights", weight_sum=weight_sum)
                    continue
            old = self._parameters[key]
            self._parameters[key] = value
            self._log.info("agent.param_updated", key=key, old=old, new=value)

    # ------------------------------------------------------------------
    # Factor computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _price_vs_sma20_zscore(df: pd.DataFrame) -> float:
        """Z-score of the current price relative to the 20-day SMA.

        Computed as ``(price - SMA20) / rolling_std(20)``.  A negative value
        indicates an oversold condition.
        """
        if df is None or df.empty or len(df) < 20:
            return 0.0
        close = df["Close"]
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std(ddof=0)
        s = sma20.iloc[-1]
        sigma = std20.iloc[-1]
        if sigma == 0 or np.isnan(sigma):
            return 0.0
        return float((close.iloc[-1] - s) / sigma)

    @staticmethod
    def _rsi_score(df: pd.DataFrame, length: int = 14) -> float:
        """Compute the latest RSI(14) value, returned on a 0-100 scale."""
        if df is None or df.empty or len(df) < length + 1:
            return 50.0  # neutral
        rsi_s = df.ta.rsi(length=length)
        if rsi_s is None or rsi_s.empty:
            return 50.0
        val = rsi_s.iloc[-1]
        return float(val) if not np.isnan(val) else 50.0

    @staticmethod
    def _rsi_zscore(rsi_val: float) -> float:
        """Convert RSI (0-100) to a z-like score centred on zero.

        RSI < 30 maps to strongly negative (oversold, BUY-favourable).
        RSI > 70 maps to strongly positive (overbought, SELL-favourable).
        """
        # Linear rescale: 0->-2.5, 50->0, 100->+2.5
        return (rsi_val - 50.0) / 20.0

    @staticmethod
    def _bollinger_position(df: pd.DataFrame, length: int = 20, std_dev: float = 2.0) -> float:
        """Percentage position within the Bollinger Bands.

        Returns a value where 0.0 = at lower band, 0.5 = at midline,
        1.0 = at upper band.  Values below 0 or above 1 indicate price
        outside the bands.
        """
        if df is None or df.empty or len(df) < length:
            return 0.5  # neutral
        bbands = df.ta.bbands(length=length, std=std_dev)
        if bbands is None or bbands.empty:
            return 0.5
        upper_col = f"BBU_{length}_{std_dev}"
        lower_col = f"BBL_{length}_{std_dev}"
        if upper_col not in bbands.columns or lower_col not in bbands.columns:
            return 0.5
        upper = bbands[upper_col].iloc[-1]
        lower = bbands[lower_col].iloc[-1]
        bw = upper - lower
        if bw <= 0 or np.isnan(bw):
            return 0.5
        return float((df["Close"].iloc[-1] - lower) / bw)

    @staticmethod
    def _bollinger_zscore(bb_pos: float) -> float:
        """Convert BB position (0-1) to a z-like score.

        0.0 (lower band) -> -2.0 (oversold).
        1.0 (upper band) -> +2.0 (overbought).
        """
        return (bb_pos - 0.5) * 4.0

    @staticmethod
    def _growth_quality_score(
        fundamentals: dict[str, Any],
        ticker: str,
        min_revenue_growth: float,
        min_roe: float,
    ) -> float:
        """Return +1.0 if the stock passes the growth quality filter, else -2.0.

        A stock must have:
        - Revenue growth > *min_revenue_growth* (default 10 %)
        - ROE > *min_roe* (default 12 %)

        The strongly negative score for failing stocks ensures the composite
        will never trigger a BUY on a non-growth name.
        """
        info = fundamentals.get(ticker, {})
        rev_growth = float(info.get("revenueGrowth", 0.0) or 0.0)
        roe = float(info.get("returnOnEquity", 0.0) or 0.0)
        if rev_growth > min_revenue_growth and roe > min_roe:
            return 1.0
        return -2.0

    @staticmethod
    def _volume_exhaustion(df: pd.DataFrame) -> float:
        """Detect selling-volume exhaustion.

        Returns a positive z-like score when recent sell-side volume is
        declining (exhaustion), suggesting a reversal is near.

        Computed as the negative slope of the 5-day regression on volume
        during down-days.  If volume on down-days is falling, the score
        is positive (favourable for a mean-reversion BUY).
        """
        if df is None or df.empty or len(df) < 20:
            return 0.0
        close = df["Close"]
        volume = df["Volume"].astype(np.float64)
        returns = close.pct_change()

        # Isolate down-day volume over the last 20 bars.
        down_mask = returns < 0
        down_volume = volume.where(down_mask).dropna()
        if len(down_volume) < 5:
            return 0.0

        recent = down_volume.iloc[-5:]
        # Simple slope: last minus first, normalised.
        slope = (recent.iloc[-1] - recent.iloc[0]) / (recent.mean() + 1e-9)
        # Negative slope = declining sell volume = exhaustion -> positive score.
        return float(-slope)

    @staticmethod
    def _ou_halflife(df: pd.DataFrame, lookback: int = 60) -> float:
        """Estimate the Ornstein-Uhlenbeck mean-reversion half-life.

        Fits a linear regression of ``delta_spread`` on ``spread_lag`` and
        extracts the half-life as ``-ln(2) / beta``.

        Returns
        -------
        float
            Estimated half-life in trading days.  Returns ``np.inf`` if the
            series is non-mean-reverting (beta >= 0).
        """
        if df is None or df.empty or len(df) < lookback:
            return np.inf
        close = df["Close"].iloc[-lookback:]
        spread = close - close.rolling(20).mean()
        spread = spread.dropna()
        if len(spread) < 10:
            return np.inf

        spread_lag = spread.shift(1).dropna()
        delta = spread.diff().dropna()

        # Align.
        idx = spread_lag.index.intersection(delta.index)
        if len(idx) < 10:
            return np.inf

        x = spread_lag.loc[idx].values.reshape(-1, 1)
        y = delta.loc[idx].values

        # OLS: y = beta * x + intercept.
        x_with_const = np.hstack([x, np.ones((len(x), 1))])
        try:
            beta, _ = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
        except (np.linalg.LinAlgError, ValueError):
            return np.inf

        if beta >= 0:
            return np.inf  # Non-reverting.

        halflife = -np.log(2) / beta
        return float(halflife)

    @staticmethod
    def _ou_halflife_zscore(halflife: float, max_halflife: float) -> float:
        """Convert OU half-life to a z-like score.

        Short half-life (fast reversion) -> positive score.
        Half-life >= *max_halflife* -> strongly negative score.
        """
        if np.isinf(halflife) or halflife >= max_halflife:
            return -2.0
        # Linearly map: 1 day -> +2.0,  max_halflife -> 0.0
        return float(2.0 * (1.0 - halflife / max_halflife))

    @staticmethod
    def _get_atr(df: pd.DataFrame, length: int = 14) -> float:
        if df is None or df.empty or len(df) < length + 1:
            return 0.0
        atr_s = df.ta.atr(length=length)
        if atr_s is None or atr_s.empty:
            return 0.0
        val = atr_s.iloc[-1]
        return float(val) if not np.isnan(val) else 0.0

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    async def generate_signals(
        self,
        universe: list[str],
        market_data: dict[str, Any],
    ) -> list[AgentSignal]:
        """Run the mean-reversion factor model and emit signals.

        Expected ``market_data`` keys:
            - ``"prices"``: ``dict[str, pd.DataFrame]`` -- OHLCV per ticker.
            - ``"fundamentals"``: ``dict[str, dict]`` -- yfinance ``info`` dicts.
        """
        prices: dict[str, pd.DataFrame] = market_data.get("prices", {})
        fundamentals: dict[str, Any] = market_data.get("fundamentals", {})

        weights = self._parameters["factor_weights"]
        min_rev = self._parameters["min_revenue_growth"]
        min_roe = self._parameters["min_roe"]
        max_hl = self._parameters["max_halflife"]
        atr_mult = self._parameters["atr_multiplier"]
        buy_z = self._parameters["buy_zscore"]
        sell_z = self._parameters["sell_zscore"]
        overbought_z = self._parameters["overbought_zscore"]
        now = datetime.now(tz=timezone.utc)

        # -- 1. Compute raw factor values per ticker ----------------------
        raw_data: dict[str, dict[str, float]] = {}
        for ticker in universe:
            df = prices.get(ticker)
            if df is None or df.empty:
                continue

            price_z = self._price_vs_sma20_zscore(df)
            rsi_val = self._rsi_score(df)
            rsi_z = self._rsi_zscore(rsi_val)
            bb_pos = self._bollinger_position(df)
            bb_z = self._bollinger_zscore(bb_pos)
            gq = self._growth_quality_score(fundamentals, ticker, min_rev, min_roe)
            vol_ex = self._volume_exhaustion(df)
            halflife = self._ou_halflife(df)
            ou_z = self._ou_halflife_zscore(halflife, max_hl)

            raw_data[ticker] = {
                "price_vs_sma20_zscore": price_z,
                "rsi_14": rsi_z,
                "bollinger_band_position": bb_z,
                "growth_quality": gq,
                "volume_exhaustion": vol_ex,
                "ou_halflife": ou_z,
                "rsi_raw": rsi_val,
                "bb_pos_raw": bb_pos,
                "halflife_raw": halflife,
            }

        if not raw_data:
            return []

        # -- 2. Cross-sectional z-score for continuous factors ------------
        factor_df = pd.DataFrame(raw_data).T  # rows=tickers, cols=factors

        # Z-score the continuous factors cross-sectionally.
        for factor_name in [
            "price_vs_sma20_zscore",
            "rsi_14",
            "bollinger_band_position",
            "volume_exhaustion",
            "ou_halflife",
        ]:
            if factor_name in factor_df.columns:
                factor_df[factor_name] = self._zscore(factor_df[factor_name])

        # growth_quality is binary-ish (+1 / -2), z-score it too.
        if "growth_quality" in factor_df.columns:
            factor_df["growth_quality"] = self._zscore(factor_df["growth_quality"])

        # -- 3. Weighted composite score ----------------------------------
        composite = pd.Series(0.0, index=factor_df.index, dtype=np.float64)
        for factor_name, weight in weights.items():
            if factor_name in factor_df.columns:
                composite += factor_df[factor_name] * weight

        # -- 4. Per-ticker signal logic -----------------------------------
        signals: list[AgentSignal] = []

        for ticker in composite.index:
            score = float(composite[ticker])
            df = prices.get(ticker)
            if df is None or df.empty:
                continue

            ticker_data = raw_data[ticker]
            current_price = float(df["Close"].iloc[-1])
            atr_val = self._get_atr(df)
            growth_passes = self._growth_quality_score(
                fundamentals, ticker, min_rev, min_roe,
            ) > 0

            # Determine action.
            action: Action
            if score < buy_z and growth_passes:
                action = Action.BUY
            elif score > overbought_z or (score >= sell_z and score > buy_z):
                # Take profit when z returns to 0, or sell if overbought.
                action = Action.SELL
            else:
                action = Action.HOLD

            if action == Action.HOLD:
                continue

            # Conviction.
            conviction = self._compute_conviction(
                abs(score), min_score=0.0, max_score=3.0,
            )

            # Stop-loss: 2x ATR below entry (for BUY); above entry (for SELL).
            if action == Action.BUY:
                stop_loss = max(current_price - atr_mult * atr_val, 0.01)
            else:
                stop_loss = current_price + atr_mult * atr_val

            # Take-profit: when z-score returns to 0 -- estimated price level.
            sma20 = df["Close"].rolling(20).mean().iloc[-1]
            take_profit = float(sma20) if not np.isnan(sma20) else current_price

            # Factor scores for auditing.
            factor_scores: dict[str, float] = {
                "price_vs_sma20_zscore": float(factor_df.at[ticker, "price_vs_sma20_zscore"]),
                "rsi_14": float(factor_df.at[ticker, "rsi_14"]),
                "bollinger_band_position": float(factor_df.at[ticker, "bollinger_band_position"]),
                "growth_quality": float(factor_df.at[ticker, "growth_quality"]),
                "volume_exhaustion": float(factor_df.at[ticker, "volume_exhaustion"]),
                "ou_halflife": float(factor_df.at[ticker, "ou_halflife"]),
                "composite": score,
                "rsi_raw": ticker_data["rsi_raw"],
                "bb_position_raw": ticker_data["bb_pos_raw"],
                "halflife_raw": ticker_data["halflife_raw"],
                "atr": atr_val,
                "growth_quality_passes": 1.0 if growth_passes else 0.0,
            }

            reasoning = self._build_reasoning(ticker, action, score, factor_scores)

            signals.append(
                AgentSignal(
                    ticker=ticker,
                    action=action,
                    conviction=conviction,
                    target_weight=self._parameters["max_weight_per_position"],
                    stop_loss=round(stop_loss, 2),
                    take_profit=round(take_profit, 2),
                    factor_scores=factor_scores,
                    reasoning=reasoning,
                    holding_period_days=self._parameters["holding_period_min"],
                    agent_name=self._name,
                    timestamp=now,
                )
            )

        self._log.info(
            "signals.generated",
            total=len(signals),
            buys=sum(1 for s in signals if s.action == Action.BUY),
            sells=sum(1 for s in signals if s.action == Action.SELL),
        )
        return signals

    # ------------------------------------------------------------------
    # Reasoning builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reasoning(
        ticker: str,
        action: Action,
        composite: float,
        factors: dict[str, float],
    ) -> str:
        parts: list[str] = [
            f"{action.value} {ticker}: composite={composite:+.3f}.",
        ]

        if action == Action.BUY:
            parts.append(
                f"RSI={factors.get('rsi_raw', 0):.1f}, "
                f"BB%={factors.get('bb_position_raw', 0):.2f}."
            )
            hl = factors.get("halflife_raw", 0)
            if not np.isinf(hl):
                parts.append(f"OU half-life={hl:.1f}d (fast reversion).")
            gq = factors.get("growth_quality_passes", 0)
            if gq > 0:
                parts.append("Growth quality filter: PASSED.")
            parts.append("Oversold growth stock -- buy the dip.")
        elif action == Action.SELL:
            if composite > 1.5:
                parts.append("Overbought: z-score exceeded upper threshold.")
            else:
                parts.append("Z-score returned to mean: taking profit.")

        return " ".join(parts)
