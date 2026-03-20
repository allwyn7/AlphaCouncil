"""Volatility-regime detection agent that emits meta-signals to modulate system risk.

:class:`VolatilityRegimeAgent` classifies the current market into one of four
volatility regimes (LOW / MEDIUM / HIGH / TRANSITION) using a blend of five
model components:

1. **India VIX level + trend** (25 %) -- 5-day MA vs 20-day MA of ^INDIAVIX.
2. **Implied-vol vs realised-vol spread** (20 %) -- VIX minus 20-day Nifty
   realised vol.
3. **GARCH(1,1) forecast** (20 %) -- next-day annualised vol from the ``arch``
   library.
4. **VIX term structure proxy** (15 %) -- level vs trend slope.
5. **Hidden Markov Model regime** (20 %) -- 3-state GaussianHMM on Nifty
   returns via ``hmmlearn``.

The agent does **not** produce conventional BUY/SELL signals for individual
stocks.  Instead it emits *meta-signals* that the orchestrator or downstream
agents consume to adjust position sizing, stop widths, and strategy selection.

Regime behaviour
----------------
* **LOW VOL** (VIX < 14, HMM = calm): full risk on, tighter stops.
* **MEDIUM VOL** (14 <= VIX <= 20): normal sizing, standard parameters.
* **HIGH VOL** (VIX > 20): halve all positions, widen stops, only buy quality
  dips (>10 % drawdown).
* **TRANSITION**: highest alpha -- heavy weight when regime is shifting.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

from alphacouncil.agents.base import BaseAgent
from alphacouncil.core.models import Action, AgentSignal, AgentStatus, VolatilityRegime

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies -- graceful fallback if missing.
# ---------------------------------------------------------------------------

try:
    from arch import arch_model  # type: ignore[import-untyped]
    _HAS_ARCH = True
except ImportError:  # pragma: no cover
    _HAS_ARCH = False

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]
    _HAS_HMM = True
except ImportError:  # pragma: no cover
    _HAS_HMM = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HMM_LOW_VOL_STATE = 0     # Calm
_HMM_MED_VOL_STATE = 1     # Normal
_HMM_HIGH_VOL_STATE = 2    # Stress

_NIFTY_TICKER = "^NSEI"
_VIX_TICKER = "^INDIAVIX"


class VolatilityRegimeAgent(BaseAgent):
    """Detects the current volatility regime and emits risk-modulating meta-signals.

    Tunable parameters (via ``set_parameters``)
    --------------------------------------------
    low_vix_threshold : float
        VIX level below which regime is LOW.  Default ``14``.
    high_vix_threshold : float
        VIX level above which regime is HIGH.  Default ``20``.
    hmm_n_states : int
        Number of HMM hidden states.  Default ``3``.
    garch_p : int
        GARCH lag order for past variance terms.  Default ``1``.
    garch_q : int
        GARCH lag order for past residual terms.  Default ``1``.
    vix_weight : float
        Weight for India VIX component.  Default ``0.25``.
    iv_rv_weight : float
        Weight for implied-vs-realised vol spread.  Default ``0.20``.
    garch_weight : float
        Weight for GARCH(1,1) forecast.  Default ``0.20``.
    term_weight : float
        Weight for VIX term-structure proxy.  Default ``0.15``.
    hmm_weight : float
        Weight for HMM regime classification.  Default ``0.20``.
    quality_dip_threshold : float
        Minimum drawdown from recent high to trigger a quality-dip BUY in
        HIGH VOL.  Default ``0.10`` (10 %).
    max_positions : int
        Maximum number of positions in the system.  Default ``15``.
    """

    AGENT_NAME: str = "volatility_regime"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Any,
        cache: Any,
        bus: Any,
        db_engine: Any,
        *,
        name: str = "volatility_regime",
    ) -> None:
        super().__init__(
            name=name,
            config=config,
            cache=cache,
            bus=bus,
            db_engine=db_engine,
        )
        self._parameters: dict[str, Any] = {
            # Thresholds
            "low_vix_threshold": 14.0,
            "high_vix_threshold": 20.0,
            # HMM
            "hmm_n_states": 3,
            # GARCH
            "garch_p": 1,
            "garch_q": 1,
            # Component weights (must sum to 1.0)
            "vix_weight": 0.25,
            "iv_rv_weight": 0.20,
            "garch_weight": 0.20,
            "term_weight": 0.15,
            "hmm_weight": 0.20,
            # Dip-buying
            "quality_dip_threshold": 0.10,
            # Portfolio
            "max_positions": 15,
        }
        self._log = logger.bind(agent=self._name)

        # Cached HMM model -- retrained periodically, not every cycle.
        self._hmm_model: GaussianHMM | None = None
        self._hmm_last_trained: datetime | None = None
        self._hmm_retrain_hours: int = 24

        # Previous regime for transition detection.
        self._prev_regime: VolatilityRegime | None = None

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def get_parameters(self) -> dict[str, Any]:
        return dict(self._parameters)

    def set_parameters(self, params: dict[str, Any]) -> None:
        weight_keys = {"vix_weight", "iv_rv_weight", "garch_weight", "term_weight", "hmm_weight"}
        incoming_weights = {k: params[k] for k in weight_keys if k in params}

        if incoming_weights:
            # Merge with existing to check sum.
            merged = {k: self._parameters[k] for k in weight_keys}
            merged.update(incoming_weights)
            total = sum(merged.values())
            if abs(total - 1.0) > 1e-6:
                self._log.warning(
                    "set_parameters.invalid_weights",
                    total=total,
                    msg="Component weights must sum to 1.0 -- weight changes ignored",
                )
                params = {k: v for k, v in params.items() if k not in weight_keys}

        for key, value in params.items():
            if key in self._parameters:
                old = self._parameters[key]
                self._parameters[key] = value
                self._log.info("parameter_changed", key=key, old=old, new=value)
            else:
                self._log.warning("set_parameters.unknown_key", key=key)

    # ------------------------------------------------------------------
    # Core signal generation
    # ------------------------------------------------------------------

    async def generate_signals(
        self,
        universe: list[str],
        market_data: dict[str, Any],
    ) -> list[AgentSignal]:
        """Detect volatility regime and emit meta-signals.

        Expected *market_data* keys
        ----------------------------
        ``"vix_history"``
            :class:`pd.Series` of daily India VIX levels (DatetimeIndex).
        ``"nifty_history"``
            :class:`pd.Series` of daily Nifty 50 closing prices (DatetimeIndex).
        ``"prices"``
            :class:`pd.DataFrame` of per-ticker closing prices (used for
            quality-dip detection in HIGH VOL).
        ``"current_positions"``
            ``dict[str, float]`` mapping ticker -> current weight.
        ``"portfolio_value"``
            ``float`` total NAV in INR.
        ``"fundamentals"``
            :class:`pd.DataFrame` indexed by ticker (used for quality filtering
            in dip-buying mode).
        """
        now = datetime.now()

        vix_history: pd.Series | None = market_data.get("vix_history")
        nifty_history: pd.Series | None = market_data.get("nifty_history")
        prices: pd.DataFrame | None = market_data.get("prices")
        current_positions: dict[str, float] = market_data.get("current_positions", {})
        portfolio_value: float = market_data.get("portfolio_value", 0.0)
        fundamentals: pd.DataFrame | None = market_data.get("fundamentals")

        # -----------------------------------------------------------------
        # 1. Compute individual model components
        # -----------------------------------------------------------------
        vix_score = self._compute_vix_score(vix_history)
        iv_rv_score = self._compute_iv_rv_spread(vix_history, nifty_history)
        garch_score = self._compute_garch_forecast(nifty_history)
        term_score = self._compute_term_structure_proxy(vix_history)
        hmm_state, hmm_score = self._compute_hmm_regime(nifty_history)

        # -----------------------------------------------------------------
        # 2. Weighted composite vol score (0 = very calm, 1 = very stressed)
        # -----------------------------------------------------------------
        w = self._parameters
        composite = (
            w["vix_weight"] * vix_score
            + w["iv_rv_weight"] * iv_rv_score
            + w["garch_weight"] * garch_score
            + w["term_weight"] * term_score
            + w["hmm_weight"] * hmm_score
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        # -----------------------------------------------------------------
        # 3. Determine regime
        # -----------------------------------------------------------------
        vix_level = self._current_vix_level(vix_history)
        regime = self._classify_regime(vix_level, hmm_state, composite)

        # Detect transition.
        is_transition = (
            self._prev_regime is not None and regime != self._prev_regime
        )
        if is_transition:
            regime = VolatilityRegime.TRANSITION
        self._prev_regime = regime if not is_transition else self._prev_regime

        # Risk multiplier.
        risk_multiplier = self._risk_multiplier(regime)

        garch_forecast = self._garch_forecast_value(nifty_history)

        self._log.info(
            "regime_detected",
            regime=regime.value,
            vix_level=round(vix_level, 2),
            composite=round(composite, 4),
            risk_multiplier=risk_multiplier,
            is_transition=is_transition,
            hmm_state=hmm_state,
            garch_forecast=round(garch_forecast, 4),
        )

        # -----------------------------------------------------------------
        # 4. Emit meta-signals per ticker
        # -----------------------------------------------------------------
        signals: list[AgentSignal] = []

        for ticker in universe:
            action, conviction, reasoning = self._decide_action(
                ticker=ticker,
                regime=regime,
                risk_multiplier=risk_multiplier,
                current_positions=current_positions,
                prices=prices,
                fundamentals=fundamentals,
                vix_level=vix_level,
                composite=composite,
            )

            if action == Action.HOLD:
                continue

            target_weight = self._compute_target_weight(
                action, risk_multiplier, len(universe),
            )

            latest_price = self._latest_price(ticker, prices)
            if latest_price is None or latest_price <= 0:
                continue

            # Stops widen in HIGH VOL, tighten in LOW VOL.
            stop_pct = self._stop_loss_pct(regime)
            tp_pct = self._take_profit_pct(regime)

            signal = AgentSignal(
                ticker=ticker,
                action=action,
                conviction=conviction,
                target_weight=round(target_weight, 4),
                stop_loss=round(max(latest_price * (1.0 - stop_pct), 0.01), 2),
                take_profit=round(latest_price * (1.0 + tp_pct), 2),
                factor_scores={
                    "vix_level": round(vix_level, 2),
                    "vix_score": round(vix_score, 4),
                    "iv_rv_score": round(iv_rv_score, 4),
                    "garch_forecast": round(garch_forecast, 4),
                    "term_score": round(term_score, 4),
                    "hmm_state": float(hmm_state),
                    "hmm_score": round(hmm_score, 4),
                    "composite_vol": round(composite, 4),
                    "regime": float(
                        {VolatilityRegime.LOW: 0, VolatilityRegime.MEDIUM: 1,
                         VolatilityRegime.HIGH: 2, VolatilityRegime.TRANSITION: 3}[regime]
                    ),
                    "risk_multiplier": risk_multiplier,
                },
                reasoning=reasoning,
                holding_period_days=self._holding_period(regime),
                agent_name=self._name,
                timestamp=now,
            )
            signals.append(signal)

        self._log.info(
            "generate_signals.complete",
            regime=regime.value,
            total_signals=len(signals),
            buy_signals=sum(1 for s in signals if s.action == Action.BUY),
            sell_signals=sum(1 for s in signals if s.action == Action.SELL),
        )
        return signals

    # ==================================================================
    # Model component 1: India VIX level + trend (25 %)
    # ==================================================================

    def _compute_vix_score(self, vix_history: pd.Series | None) -> float:
        """Score from 0 (calm) to 1 (stressed) based on VIX level and trend.

        Uses 5-day MA vs 20-day MA to capture directional momentum.
        """
        if vix_history is None or len(vix_history) < 20:
            return 0.5  # Neutral default.

        vix = vix_history.dropna().astype(float)
        if len(vix) < 20:
            return 0.5

        current_vix = float(vix.iloc[-1])
        ma5 = float(vix.iloc[-5:].mean())
        ma20 = float(vix.iloc[-20:].mean())

        # Level score: 0 at VIX=8, 1 at VIX=30+.
        level_score = float(np.clip((current_vix - 8.0) / 22.0, 0.0, 1.0))

        # Trend score: positive when short MA > long MA (vol rising).
        trend_raw = (ma5 - ma20) / max(ma20, 1.0)
        trend_score = float(np.clip(0.5 + trend_raw * 5.0, 0.0, 1.0))

        # Blend level (70 %) and trend (30 %).
        return 0.7 * level_score + 0.3 * trend_score

    # ==================================================================
    # Model component 2: IV vs RV spread (20 %)
    # ==================================================================

    def _compute_iv_rv_spread(
        self,
        vix_history: pd.Series | None,
        nifty_history: pd.Series | None,
    ) -> float:
        """Score from 0 to 1 based on implied-vol minus 20-day realised-vol spread.

        A high positive spread (IV >> RV) suggests fear / hedging demand;
        a negative spread suggests complacency.
        """
        if vix_history is None or nifty_history is None:
            return 0.5

        vix = vix_history.dropna().astype(float)
        nifty = nifty_history.dropna().astype(float)

        if len(vix) < 1 or len(nifty) < 21:
            return 0.5

        current_iv = float(vix.iloc[-1])

        # 20-day realised vol annualised (% terms to match VIX).
        returns = nifty.pct_change().dropna()
        if len(returns) < 20:
            return 0.5
        rv_20d = float(returns.iloc[-20:].std() * np.sqrt(252) * 100)

        spread = current_iv - rv_20d  # Can be negative.

        # Map spread: -10 -> 0, 0 -> 0.5, +10 -> 1.
        return float(np.clip(0.5 + spread / 20.0, 0.0, 1.0))

    # ==================================================================
    # Model component 3: GARCH(1,1) forecast (20 %)
    # ==================================================================

    def _compute_garch_forecast(self, nifty_history: pd.Series | None) -> float:
        """Score from 0 to 1 based on GARCH(1,1) next-day vol forecast.

        Falls back to 0.5 if the ``arch`` library is unavailable or the model
        fails to converge.
        """
        forecast_vol = self._garch_forecast_value(nifty_history)
        if np.isnan(forecast_vol):
            return 0.5
        # Map annualised vol: 10 % -> 0, 30 % -> 1.
        return float(np.clip((forecast_vol * 100 - 10.0) / 20.0, 0.0, 1.0))

    def _garch_forecast_value(self, nifty_history: pd.Series | None) -> float:
        """Return the GARCH(1,1) next-day annualised volatility forecast.

        Returns ``NaN`` on any failure.
        """
        if not _HAS_ARCH:
            self._log.debug("garch.arch_not_installed")
            return np.nan

        if nifty_history is None or len(nifty_history) < 100:
            return np.nan

        try:
            nifty = nifty_history.dropna().astype(float)
            returns = nifty.pct_change().dropna() * 100  # Percentage returns.

            if len(returns) < 100:
                return np.nan

            p = self._parameters["garch_p"]
            q = self._parameters["garch_q"]

            model = arch_model(returns, vol="Garch", p=p, q=q, dist="normal")
            result = model.fit(disp="off", show_warning=False)
            forecast = result.forecast(horizon=1)
            variance = forecast.variance.iloc[-1, 0]
            # Convert from daily percentage variance to annualised decimal vol.
            daily_vol = np.sqrt(variance) / 100.0
            annual_vol = daily_vol * np.sqrt(252)
            return float(annual_vol)
        except Exception:
            self._log.debug("garch.forecast_failed", exc_info=True)
            return np.nan

    # ==================================================================
    # Model component 4: VIX term-structure proxy (15 %)
    # ==================================================================

    def _compute_term_structure_proxy(self, vix_history: pd.Series | None) -> float:
        """Approximate term-structure score using VIX level vs its trend.

        In a normal (contango) term structure the near-month VIX < far-month;
        in backwardation (stress) near > far.  We proxy this with the current
        VIX level vs its 20-day average: if the current level is *above* the
        average the "curve" is in backwardation (score toward 1).
        """
        if vix_history is None or len(vix_history) < 20:
            return 0.5

        vix = vix_history.dropna().astype(float)
        if len(vix) < 20:
            return 0.5

        current = float(vix.iloc[-1])
        avg_20 = float(vix.iloc[-20:].mean())

        if avg_20 == 0:
            return 0.5

        ratio = current / avg_20  # >1 means backwardation (stressed).
        # Map ratio: 0.8 -> 0, 1.0 -> 0.5, 1.2 -> 1.
        return float(np.clip((ratio - 0.8) / 0.4, 0.0, 1.0))

    # ==================================================================
    # Model component 5: Hidden Markov Model (20 %)
    # ==================================================================

    def _compute_hmm_regime(
        self, nifty_history: pd.Series | None,
    ) -> tuple[int, float]:
        """Classify the current regime via a 3-state Gaussian HMM.

        Returns
        -------
        (state, score)
            *state*: 0 (calm), 1 (normal), 2 (stress).
            *score*: mapped to 0.0 / 0.5 / 1.0 respectively.
        """
        if not _HAS_HMM:
            self._log.debug("hmm.hmmlearn_not_installed")
            return _HMM_MED_VOL_STATE, 0.5

        if nifty_history is None or len(nifty_history) < 100:
            return _HMM_MED_VOL_STATE, 0.5

        try:
            nifty = nifty_history.dropna().astype(float)
            returns = nifty.pct_change().dropna().values.reshape(-1, 1)

            if len(returns) < 100:
                return _HMM_MED_VOL_STATE, 0.5

            now = datetime.now()
            needs_retrain = (
                self._hmm_model is None
                or self._hmm_last_trained is None
                or (now - self._hmm_last_trained).total_seconds() > self._hmm_retrain_hours * 3600
            )

            if needs_retrain:
                n_states = self._parameters["hmm_n_states"]
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type="full",
                    n_iter=200,
                    random_state=42,
                    tol=1e-4,
                )
                model.fit(returns)
                self._hmm_model = model
                self._hmm_last_trained = now
                self._log.info("hmm.retrained", n_states=n_states, n_samples=len(returns))

            hidden_states = self._hmm_model.predict(returns)
            current_state = int(hidden_states[-1])

            # Sort states by the mean return (or variance) so that
            # state 0 = calm (low vol), state 2 = stress (high vol).
            means = self._hmm_model.means_.flatten()
            variances = np.array([
                self._hmm_model.covars_[i][0, 0]
                for i in range(self._hmm_model.n_components)
            ])
            # Order by variance ascending -> [calm, normal, stress].
            state_order = np.argsort(variances)
            mapped_state = int(np.where(state_order == current_state)[0][0])

            score_map = {0: 0.0, 1: 0.5, 2: 1.0}
            score = score_map.get(mapped_state, 0.5)

            return mapped_state, score

        except Exception:
            self._log.debug("hmm.prediction_failed", exc_info=True)
            return _HMM_MED_VOL_STATE, 0.5

    # ==================================================================
    # Regime classification
    # ==================================================================

    def _classify_regime(
        self,
        vix_level: float,
        hmm_state: int,
        composite: float,
    ) -> VolatilityRegime:
        """Map VIX level, HMM state, and composite score to a regime label."""
        low_thresh = self._parameters["low_vix_threshold"]
        high_thresh = self._parameters["high_vix_threshold"]

        if vix_level < low_thresh and hmm_state == _HMM_LOW_VOL_STATE:
            return VolatilityRegime.LOW
        if vix_level > high_thresh:
            return VolatilityRegime.HIGH
        if hmm_state == _HMM_HIGH_VOL_STATE and composite > 0.7:
            return VolatilityRegime.HIGH
        return VolatilityRegime.MEDIUM

    @staticmethod
    def _risk_multiplier(regime: VolatilityRegime) -> float:
        """Return the position-sizing multiplier for the given *regime*."""
        return {
            VolatilityRegime.LOW: 1.2,
            VolatilityRegime.MEDIUM: 1.0,
            VolatilityRegime.HIGH: 0.5,
            VolatilityRegime.TRANSITION: 1.0,
        }[regime]

    @staticmethod
    def _current_vix_level(vix_history: pd.Series | None) -> float:
        """Extract the latest VIX level, defaulting to 15 (medium) on failure."""
        if vix_history is None or vix_history.empty:
            return 15.0
        try:
            return float(vix_history.dropna().iloc[-1])
        except (IndexError, TypeError):
            return 15.0

    # ==================================================================
    # Action decision per ticker
    # ==================================================================

    def _decide_action(
        self,
        *,
        ticker: str,
        regime: VolatilityRegime,
        risk_multiplier: float,
        current_positions: dict[str, float],
        prices: pd.DataFrame | None,
        fundamentals: pd.DataFrame | None,
        vix_level: float,
        composite: float,
    ) -> tuple[Action, int, str]:
        """Determine action, conviction, and reasoning for a single *ticker*.

        Returns
        -------
        (action, conviction, reasoning)
        """
        is_held = ticker in current_positions

        # --- LOW VOL: growth-on, add positions ---
        if regime == VolatilityRegime.LOW:
            if not is_held:
                return (
                    Action.BUY,
                    self._regime_conviction(regime, composite),
                    f"LOW VOL regime (VIX={vix_level:.1f}): adding growth position in {ticker}",
                )
            return Action.HOLD, 0, ""

        # --- HIGH VOL: sell weak, buy quality dips ---
        if regime == VolatilityRegime.HIGH:
            # Sell weak positions (low weight = low conviction from other agents).
            if is_held:
                weight = current_positions.get(ticker, 0.0)
                if weight < 0.03:  # Below 3 % -> weak position.
                    return (
                        Action.SELL,
                        self._regime_conviction(regime, composite),
                        (
                            f"HIGH VOL regime (VIX={vix_level:.1f}): "
                            f"trimming weak position in {ticker} (weight={weight:.2%})"
                        ),
                    )

            # Quality dip-buying: only for non-held tickers that dropped >10 %.
            if not is_held and self._is_quality_dip(ticker, prices, fundamentals):
                return (
                    Action.BUY,
                    self._regime_conviction(regime, composite),
                    (
                        f"HIGH VOL regime (VIX={vix_level:.1f}): "
                        f"quality dip-buy in {ticker} (dropped >10 %% from recent high)"
                    ),
                )
            return Action.HOLD, 0, ""

        # --- TRANSITION: heavy weight (highest alpha opportunity) ---
        if regime == VolatilityRegime.TRANSITION:
            if not is_held:
                return (
                    Action.BUY,
                    min(self._regime_conviction(regime, composite) + 15, 100),
                    (
                        f"REGIME TRANSITION (VIX={vix_level:.1f}): "
                        f"high-alpha opportunity -- adding {ticker}"
                    ),
                )
            return Action.HOLD, 0, ""

        # --- MEDIUM VOL: neutral, no action from this agent ---
        return Action.HOLD, 0, ""

    def _is_quality_dip(
        self,
        ticker: str,
        prices: pd.DataFrame | None,
        fundamentals: pd.DataFrame | None,
    ) -> bool:
        """Check if *ticker* is a quality stock that dropped >10 % from its recent high.

        Quality filter: ROE > 12 % and positive EPS growth (if fundamentals
        are available).
        """
        threshold = self._parameters["quality_dip_threshold"]

        # --- Price check: >threshold drop from 20-day high ---
        if prices is None:
            return False

        latest = self._latest_price(ticker, prices)
        high_20d = self._recent_high(ticker, prices, window=20)

        if latest is None or high_20d is None or high_20d <= 0:
            return False

        drawdown = (high_20d - latest) / high_20d
        if drawdown < threshold:
            return False

        # --- Quality check (best-effort) ---
        if fundamentals is not None and ticker in fundamentals.index:
            try:
                row = fundamentals.loc[ticker]
                roe = float(getattr(row, "roe", 0.0))
                eps_g = float(getattr(row, "eps_growth", 0.0))
                if roe < 0.12 or eps_g <= 0:
                    return False
            except (TypeError, ValueError):
                pass  # If data is bad, let it through -- price signal is primary.

        return True

    # ==================================================================
    # Position sizing & stop helpers
    # ==================================================================

    @staticmethod
    def _compute_target_weight(
        action: Action,
        risk_multiplier: float,
        universe_size: int,
    ) -> float:
        """Equal-weight target adjusted by *risk_multiplier*."""
        if action == Action.SELL:
            return 0.0
        if universe_size <= 0:
            return 0.0
        base = 1.0 / max(universe_size, 15)
        return min(base * risk_multiplier, 0.08)

    @staticmethod
    def _stop_loss_pct(regime: VolatilityRegime) -> float:
        """Stop-loss distance (fraction of price) by regime."""
        return {
            VolatilityRegime.LOW: 0.04,         # Tight
            VolatilityRegime.MEDIUM: 0.06,       # Standard
            VolatilityRegime.HIGH: 0.10,          # Wide
            VolatilityRegime.TRANSITION: 0.07,    # Moderate
        }[regime]

    @staticmethod
    def _take_profit_pct(regime: VolatilityRegime) -> float:
        """Take-profit distance (fraction of price) by regime."""
        return {
            VolatilityRegime.LOW: 0.08,
            VolatilityRegime.MEDIUM: 0.10,
            VolatilityRegime.HIGH: 0.15,
            VolatilityRegime.TRANSITION: 0.12,
        }[regime]

    @staticmethod
    def _holding_period(regime: VolatilityRegime) -> int:
        """Typical holding period (calendar days) by regime."""
        return {
            VolatilityRegime.LOW: 15,
            VolatilityRegime.MEDIUM: 10,
            VolatilityRegime.HIGH: 5,
            VolatilityRegime.TRANSITION: 7,
        }[regime]

    @staticmethod
    def _regime_conviction(regime: VolatilityRegime, composite: float) -> int:
        """Map regime + composite vol score to a 0-100 conviction."""
        base = {
            VolatilityRegime.LOW: 70,
            VolatilityRegime.MEDIUM: 50,
            VolatilityRegime.HIGH: 60,       # High conviction to ACT in stress.
            VolatilityRegime.TRANSITION: 75,  # Highest alpha.
        }[regime]
        # Adjust by composite certainty (how clearly we read the regime).
        certainty_bonus = int(abs(composite - 0.5) * 30)
        return min(base + certainty_bonus, 100)

    # ==================================================================
    # Price helpers
    # ==================================================================

    @staticmethod
    def _latest_price(ticker: str, prices: pd.DataFrame | None) -> float | None:
        """Return the latest closing price for *ticker*, or ``None``."""
        if prices is None:
            return None
        try:
            if "ticker" in prices.columns:
                subset = prices[prices["ticker"] == ticker]
                if subset.empty:
                    return None
                return float(subset["close"].iloc[-1])
            elif ticker in prices.columns:
                series = prices[ticker].dropna()
                return float(series.iloc[-1]) if not series.empty else None
        except (IndexError, KeyError, TypeError):
            pass
        return None

    @staticmethod
    def _recent_high(
        ticker: str,
        prices: pd.DataFrame | None,
        window: int = 20,
    ) -> float | None:
        """Return the highest close in the last *window* days for *ticker*."""
        if prices is None:
            return None
        try:
            if "ticker" in prices.columns:
                subset = prices[prices["ticker"] == ticker].sort_index()
                if len(subset) < 2:
                    return None
                return float(subset["close"].iloc[-window:].max())
            elif ticker in prices.columns:
                series = prices[ticker].dropna()
                if len(series) < 2:
                    return None
                return float(series.iloc[-window:].max())
        except (IndexError, KeyError, TypeError):
            pass
        return None
