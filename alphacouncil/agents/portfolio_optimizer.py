"""Meta-allocator agent that optimises risk-adjusted portfolio allocation.

Sits above the five alpha-generating agents and constructs an optimal
portfolio by combining their BUY signals through a mean-variance
optimisation (MVO) framework with practical constraints tailored to
the Indian equity market.

Algorithm overview
------------------
1. **Candidate selection** -- collect BUY signals from all five agents;
   retain tickers with >= ``min_agent_agreement`` agents in agreement.
2. **Growth tilt** -- tickers recommended by the ``GrowthMomentumAgent``
   receive a ``growth_agent_bonus`` (default 1.5x) weight multiplier in the
   expected-return vector.
3. **Expected returns** -- exponentially weighted momentum with a
   configurable half-life (default 60 days) via ``pandas.DataFrame.ewm``.
4. **Covariance estimation** -- Ledoit-Wolf shrinkage estimator from
   ``sklearn.covariance`` for a well-conditioned matrix.
5. **Mean-variance optimisation** -- maximise the Sharpe ratio
   ``(w'mu - rf) / sqrt(w'Sigma w)`` via ``scipy.optimize.minimize``
   (SLSQP).  Constraints: long-only, sum(w) <= ``max_deployed`` (default
   0.80 for a 20 % cash buffer), per-stock weight in
   ``[min_weight, max_weight]``.
6. **Incremental trading** -- move 30 % of the gap toward the target each
   day to reduce market impact.
7. **Marginal VaR** -- logged per position for risk monitoring.

Rebalance triggers
~~~~~~~~~~~~~~~~~~
* Weekly (every Monday).
* Weight drift > ``drift_threshold`` from target.
* Candidate set changes (new entry or exit).

Portfolio profile: 15-25 positions, max 80 % deployed capital.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import optimize as scipy_optimize
from sklearn.covariance import LedoitWolf

from alphacouncil.agents.base import BaseAgent
from alphacouncil.core.models import Action, AgentSignal, AgentStatus

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_NAME = "portfolio_optimizer"

# The five alpha agents whose signals we consume.
_ALPHA_AGENT_NAMES: frozenset[str] = frozenset(
    {
        "growth_momentum",
        "mean_reversion",
        "sentiment_alpha",
        "macro_regime",
        "technical_breakout",
    }
)

_GROWTH_AGENT_NAME = "growth_momentum"

# Trading days per year (NSE calendar).
_TRADING_DAYS_YEAR = 252

# Portfolio constraints.
_MIN_POSITIONS = 15
_MAX_POSITIONS = 25


# ---------------------------------------------------------------------------
# PortfolioOptimizationAgent
# ---------------------------------------------------------------------------


class PortfolioOptimizationAgent(BaseAgent):
    """Meta-allocator that optimises risk-adjusted capital deployment.

    Parameters
    ----------
    name:
        Canonical agent name (defaults to ``"portfolio_optimizer"``).
    config:
        System-wide settings.
    cache:
        Shared cache backend.
    bus:
        Message bus satisfying :class:`~alphacouncil.agents.base.MessageBus`.
    db_engine:
        SQLAlchemy engine for persistence.
    """

    def __init__(
        self,
        name: str = AGENT_NAME,
        config: Any = None,
        cache: Any = None,
        bus: Any = None,
        db_engine: Any = None,
    ) -> None:
        super().__init__(name=name, config=config, cache=cache, bus=bus, db_engine=db_engine)
        self._log = logger.bind(agent=name)

        # ----- Tunable parameters -----------------------------------------
        self._min_agent_agreement: int = 2
        self._growth_agent_bonus: float = 1.5
        self._ewm_halflife: int = 60
        self._max_weight: float = 0.05
        self._min_weight: float = 0.02
        self._max_deployed: float = 0.80
        self._rebalance_gap: float = 0.30
        self._drift_threshold: float = 0.03
        self._risk_free_rate: float = 0.065  # RBI repo rate (annual)

        # Internal state: target weights from the most recent optimisation.
        self._target_weights: dict[str, float] = {}
        self._current_weights: dict[str, float] = {}
        self._last_candidate_set: set[str] = set()
        self._last_rebalance_weekday: int | None = None  # 0 = Monday

        self._log.info(
            "portfolio_optimizer.configured",
            min_agent_agreement=self._min_agent_agreement,
            growth_agent_bonus=self._growth_agent_bonus,
            max_deployed=self._max_deployed,
        )

    # ------------------------------------------------------------------
    # BaseAgent abstract interface
    # ------------------------------------------------------------------

    def get_parameters(self) -> dict[str, Any]:
        """Return the agent's current tunable parameters."""
        return {
            "min_agent_agreement": self._min_agent_agreement,
            "growth_agent_bonus": self._growth_agent_bonus,
            "ewm_halflife": self._ewm_halflife,
            "max_weight": self._max_weight,
            "min_weight": self._min_weight,
            "max_deployed": self._max_deployed,
            "rebalance_gap": self._rebalance_gap,
            "drift_threshold": self._drift_threshold,
            "risk_free_rate": self._risk_free_rate,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Hot-reload tunable parameters.

        Only recognised keys are applied; unknown keys are logged and skipped.
        """
        allowed = {
            "min_agent_agreement",
            "growth_agent_bonus",
            "ewm_halflife",
            "max_weight",
            "min_weight",
            "max_deployed",
            "rebalance_gap",
            "drift_threshold",
            "risk_free_rate",
        }
        for key, value in params.items():
            if key not in allowed:
                self._log.warning("set_parameters.unknown_key", key=key)
                continue
            old_value = getattr(self, f"_{key}")
            setattr(self, f"_{key}", value)
            self._log.info("set_parameters.updated", key=key, old=old_value, new=value)

    # ------------------------------------------------------------------
    # Signal generation (main entry point)
    # ------------------------------------------------------------------

    async def generate_signals(
        self,
        universe: list[str],
        market_data: dict[str, Any],
    ) -> list[AgentSignal]:
        """Optimise portfolio allocation and emit BUY / SELL / HOLD signals.

        Expected ``market_data`` keys
        ------------------------------
        ``"agent_signals"``
            ``dict[str, list[AgentSignal]]`` -- signals grouped by agent
            name.  Each list contains the most recent ``AgentSignal``
            objects emitted by that agent.
        ``"prices"``
            ``dict[str, pd.DataFrame]`` -- OHLCV DataFrames keyed by ticker,
            with at least 120 rows for the EWM half-life calculation.
        ``"current_weights"``
            ``dict[str, float]`` -- current portfolio weights per ticker.
        ``"today"``
            ``datetime`` -- the current trading date (used for weekly
            rebalance checks).
        """
        agent_signals: dict[str, list[AgentSignal]] = market_data.get("agent_signals", {})
        prices: dict[str, pd.DataFrame] = market_data.get("prices", {})
        current_weights: dict[str, float] = market_data.get("current_weights", {})
        today: datetime = market_data.get("today", datetime.now(tz=timezone.utc))

        self._current_weights = dict(current_weights)

        # --- Step 1: Candidate selection -----------------------------------
        candidates, agent_agreement, growth_tilt_tickers = self._select_candidates(
            agent_signals
        )

        if not candidates:
            self._log.info("generate_signals.no_candidates")
            return []

        # --- Check if rebalance is needed ---------------------------------
        if not self._should_rebalance(today, candidates):
            self._log.info("generate_signals.no_rebalance_needed")
            return self._emit_hold_signals(candidates)

        # --- Step 2-4: Compute expected returns and covariance matrix ------
        # Filter to candidates that have sufficient price data.
        valid_candidates = [t for t in candidates if t in prices and len(prices[t]) >= 60]
        if not valid_candidates:
            self._log.warning("generate_signals.insufficient_price_data")
            return []

        returns_df = self._build_returns_dataframe(valid_candidates, prices)
        if returns_df.empty or len(returns_df.columns) < 2:
            self._log.warning("generate_signals.insufficient_returns_data")
            return []

        expected_returns = await asyncio.to_thread(
            self._compute_expected_returns, returns_df, growth_tilt_tickers
        )
        cov_matrix = await asyncio.to_thread(self._compute_covariance, returns_df)

        # --- Step 5: Mean-variance optimisation ----------------------------
        optimal_weights = await asyncio.to_thread(
            self._optimise_weights, expected_returns, cov_matrix
        )

        if optimal_weights is None:
            self._log.error("generate_signals.optimisation_failed")
            return []

        # --- Step 6: Incremental trading (move 30% toward target) ----------
        target_weights = self._apply_incremental_trading(optimal_weights)
        self._target_weights = dict(target_weights)
        self._last_candidate_set = set(valid_candidates)
        self._last_rebalance_weekday = today.weekday()

        # --- Step 7: Log marginal VaR per position -------------------------
        marginal_vars = self._compute_marginal_var(target_weights, cov_matrix)

        # --- Emit signals --------------------------------------------------
        signals = self._build_signals(
            target_weights=target_weights,
            expected_returns=expected_returns,
            marginal_vars=marginal_vars,
            agent_agreement=agent_agreement,
            growth_tilt_tickers=growth_tilt_tickers,
            prices=prices,
        )

        self._log.info(
            "generate_signals.complete",
            n_candidates=len(valid_candidates),
            n_signals=len(signals),
            total_weight=round(sum(target_weights.values()), 4),
        )
        return signals

    # ------------------------------------------------------------------
    # Step 1: Candidate selection
    # ------------------------------------------------------------------

    def _select_candidates(
        self,
        agent_signals: dict[str, list[AgentSignal]],
    ) -> tuple[list[str], dict[str, int], set[str]]:
        """Select tickers with sufficient agent agreement.

        Returns
        -------
        tuple[list[str], dict[str, int], set[str]]
            ``(candidate_tickers, agreement_counts, growth_tilt_tickers)``
        """
        # Count BUY signals per ticker across all alpha agents.
        buy_counts: dict[str, int] = {}
        growth_tilt_tickers: set[str] = set()

        for agent_name, signals in agent_signals.items():
            if agent_name not in _ALPHA_AGENT_NAMES:
                continue
            for sig in signals:
                if sig.action == Action.BUY:
                    buy_counts[sig.ticker] = buy_counts.get(sig.ticker, 0) + 1
                    if agent_name == _GROWTH_AGENT_NAME:
                        growth_tilt_tickers.add(sig.ticker)

        # Filter to tickers meeting the minimum agreement threshold.
        candidates = [
            ticker
            for ticker, count in buy_counts.items()
            if count >= self._min_agent_agreement
        ]

        self._log.info(
            "select_candidates",
            total_buy_tickers=len(buy_counts),
            candidates=len(candidates),
            growth_tilted=len(growth_tilt_tickers),
        )
        return candidates, buy_counts, growth_tilt_tickers

    # ------------------------------------------------------------------
    # Rebalance trigger logic
    # ------------------------------------------------------------------

    def _should_rebalance(self, today: datetime, candidates: list[str]) -> bool:
        """Determine whether a rebalance is needed.

        Triggers:
        - Weekly (Monday).
        - Weight drift > threshold for any held position.
        - Candidate set changed (new ticker entered or exited).
        """
        # Weekly trigger: Monday.
        is_monday = today.weekday() == 0
        if is_monday and self._last_rebalance_weekday != 0:
            self._log.info("rebalance_trigger.weekly")
            return True

        # Drift trigger.
        for ticker, current_w in self._current_weights.items():
            target_w = self._target_weights.get(ticker, 0.0)
            if abs(current_w - target_w) > self._drift_threshold:
                self._log.info(
                    "rebalance_trigger.drift",
                    ticker=ticker,
                    current=round(current_w, 4),
                    target=round(target_w, 4),
                )
                return True

        # Candidate set change trigger.
        current_candidates = set(candidates)
        if current_candidates != self._last_candidate_set:
            new_entries = current_candidates - self._last_candidate_set
            exits = self._last_candidate_set - current_candidates
            self._log.info(
                "rebalance_trigger.candidate_change",
                new_entries=sorted(new_entries),
                exits=sorted(exits),
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Step 2-3: Returns & covariance estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_returns_dataframe(
        tickers: list[str],
        prices: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Build a DataFrame of daily log returns for the candidate set.

        Columns are tickers; rows are dates.  Missing data is forward-filled
        then dropped.
        """
        close_series: dict[str, pd.Series] = {}
        for ticker in tickers:
            df = prices.get(ticker)
            if df is not None and not df.empty and "close" in df.columns:
                close_series[ticker] = df["close"]

        if not close_series:
            return pd.DataFrame()

        close_df = pd.DataFrame(close_series)
        close_df = close_df.ffill().dropna()

        if close_df.empty or len(close_df) < 2:
            return pd.DataFrame()

        # Log returns for better statistical properties.
        returns_df = np.log(close_df / close_df.shift(1)).dropna()
        return returns_df

    def _compute_expected_returns(
        self,
        returns_df: pd.DataFrame,
        growth_tilt_tickers: set[str],
    ) -> pd.Series:
        """Compute expected returns using exponentially weighted momentum.

        The EWM half-life is configurable (default 60 days).  Tickers
        recommended by the ``GrowthMomentumAgent`` receive a
        ``growth_agent_bonus`` multiplier (default 1.5x).
        """
        # Exponentially weighted mean return (annualised).
        ewm_returns: pd.Series = (
            returns_df.ewm(halflife=self._ewm_halflife).mean().iloc[-1]
            * _TRADING_DAYS_YEAR
        )

        # Apply growth tilt.
        for ticker in growth_tilt_tickers:
            if ticker in ewm_returns.index:
                ewm_returns[ticker] *= self._growth_agent_bonus

        return ewm_returns

    def _compute_covariance(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Estimate the covariance matrix using Ledoit-Wolf shrinkage.

        Returns an annualised covariance matrix as a pandas DataFrame with
        ticker labels.
        """
        lw = LedoitWolf()
        lw.fit(returns_df.values)
        cov_annual = lw.covariance_ * _TRADING_DAYS_YEAR

        return pd.DataFrame(
            cov_annual,
            index=returns_df.columns,
            columns=returns_df.columns,
        )

    # ------------------------------------------------------------------
    # Step 5: Mean-variance optimisation
    # ------------------------------------------------------------------

    def _optimise_weights(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
    ) -> dict[str, float] | None:
        """Maximise the Sharpe ratio subject to portfolio constraints.

        Constraints
        -----------
        - Long-only: ``w >= 0``
        - Total deployed: ``sum(w) <= max_deployed`` (default 0.80)
        - Per-stock: ``min_weight <= w_i <= max_weight`` for included stocks
        - Minimum 2 % for any included position to avoid dust positions.

        Uses ``scipy.optimize.minimize`` with the SLSQP method.

        Returns
        -------
        dict[str, float] | None
            Optimal weights keyed by ticker, or ``None`` if the
            optimiser fails to converge.
        """
        tickers = list(expected_returns.index)
        n = len(tickers)

        if n == 0:
            return None

        mu = expected_returns.values.astype(np.float64)
        sigma = cov_matrix.values.astype(np.float64)
        rf_daily = self._risk_free_rate / _TRADING_DAYS_YEAR
        rf_annual = self._risk_free_rate

        # ------ Objective: negative Sharpe (we minimise) ------------------
        def neg_sharpe(w: np.ndarray) -> float:
            port_return = float(w @ mu)
            port_var = float(w @ sigma @ w)
            port_std = np.sqrt(port_var) if port_var > 1e-12 else 1e-6
            sharpe = (port_return - rf_annual) / port_std
            return -sharpe

        # ------ Constraints -----------------------------------------------
        constraints = [
            # sum(w) <= max_deployed
            {"type": "ineq", "fun": lambda w: self._max_deployed - np.sum(w)},
        ]

        # Bounds: each weight in [0, max_weight].
        bounds = [(0.0, self._max_weight) for _ in range(n)]

        # Initial guess: equal weight (capped to max_deployed).
        w0 = np.full(n, min(self._max_deployed / n, self._max_weight))

        result = scipy_optimize.minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
        )

        if not result.success:
            self._log.warning(
                "optimisation.failed",
                message=result.message,
                n_tickers=n,
            )
            # Fall back to equal weight as a safe default.
            return self._equal_weight_fallback(tickers)

        raw_weights = result.x

        # Enforce minimum weight: set positions below min_weight to zero,
        # then re-normalise.
        filtered: dict[str, float] = {}
        for i, ticker in enumerate(tickers):
            w = float(raw_weights[i])
            if w >= self._min_weight:
                filtered[ticker] = w

        if not filtered:
            return self._equal_weight_fallback(tickers)

        # Re-normalise so total <= max_deployed.
        total = sum(filtered.values())
        if total > self._max_deployed:
            scale = self._max_deployed / total
            filtered = {t: w * scale for t, w in filtered.items()}

        # Cap at max_positions.
        if len(filtered) > _MAX_POSITIONS:
            sorted_tickers = sorted(filtered, key=filtered.get, reverse=True)  # type: ignore[arg-type]
            filtered = {t: filtered[t] for t in sorted_tickers[:_MAX_POSITIONS]}

        self._log.info(
            "optimisation.success",
            n_positions=len(filtered),
            total_weight=round(sum(filtered.values()), 4),
            sharpe=-round(float(result.fun), 4),
        )
        return filtered

    def _equal_weight_fallback(self, tickers: list[str]) -> dict[str, float]:
        """Produce equal-weight allocation as a fallback when MVO fails.

        Respects ``max_weight``, ``max_deployed``, and ``_MAX_POSITIONS``.
        """
        n = min(len(tickers), _MAX_POSITIONS)
        if n == 0:
            return {}
        w = min(self._max_deployed / n, self._max_weight)
        selected = tickers[:n]
        self._log.info("optimisation.fallback_equal_weight", n=n, weight=round(w, 4))
        return {t: w for t in selected}

    # ------------------------------------------------------------------
    # Step 6: Incremental trading
    # ------------------------------------------------------------------

    def _apply_incremental_trading(
        self,
        optimal_weights: dict[str, float],
    ) -> dict[str, float]:
        """Move ``rebalance_gap`` fraction (default 30 %) toward the target.

        This dampens trading to reduce market impact.
        """
        result: dict[str, float] = {}
        all_tickers = set(optimal_weights.keys()) | set(self._current_weights.keys())

        for ticker in all_tickers:
            current = self._current_weights.get(ticker, 0.0)
            target = optimal_weights.get(ticker, 0.0)
            gap = target - current
            new_weight = current + gap * self._rebalance_gap

            # Snap small weights to zero.
            if new_weight < self._min_weight / 2.0:
                new_weight = 0.0

            if new_weight > 0.0:
                result[ticker] = round(new_weight, 6)

        return result

    # ------------------------------------------------------------------
    # Step 7: Marginal VaR
    # ------------------------------------------------------------------

    def _compute_marginal_var(
        self,
        weights: dict[str, float],
        cov_matrix: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute marginal VaR per position (parametric, 95 % confidence).

        Marginal VaR_i = z * (Sigma @ w)_i / sqrt(w' Sigma w)
        where z = 1.645 for 95 % confidence.

        Logged for risk monitoring; also passed into signal factor_scores.
        """
        z_95 = 1.645
        tickers = [t for t in weights if t in cov_matrix.index]
        if not tickers:
            return {}

        sigma = cov_matrix.loc[tickers, tickers].values.astype(np.float64)
        w = np.array([weights[t] for t in tickers], dtype=np.float64)

        port_var = float(w @ sigma @ w)
        port_std = np.sqrt(port_var) if port_var > 1e-12 else 1e-6

        sigma_w = sigma @ w  # Component risk contribution vector.
        marginal = z_95 * sigma_w / port_std

        result: dict[str, float] = {}
        for i, ticker in enumerate(tickers):
            mvar = round(float(marginal[i]), 6)
            result[ticker] = mvar
            self._log.debug(
                "marginal_var",
                ticker=ticker,
                weight=round(weights[ticker], 4),
                marginal_var=mvar,
            )

        return result

    # ------------------------------------------------------------------
    # Signal emission helpers
    # ------------------------------------------------------------------

    def _build_signals(
        self,
        target_weights: dict[str, float],
        expected_returns: pd.Series,
        marginal_vars: dict[str, float],
        agent_agreement: dict[str, int],
        growth_tilt_tickers: set[str],
        prices: dict[str, pd.DataFrame],
    ) -> list[AgentSignal]:
        """Translate target weights into ``AgentSignal`` objects.

        Action logic:
        - **BUY**: target weight > current weight (increase position).
        - **SELL**: target weight = 0 (exit position) or target < current.
        - **HOLD**: target ~= current (within ``drift_threshold``).

        Conviction is derived from the improvement in portfolio Sharpe from
        including this stock.
        """
        signals: list[AgentSignal] = []
        now = datetime.now(tz=timezone.utc)

        all_tickers = set(target_weights.keys()) | set(self._current_weights.keys())

        for ticker in all_tickers:
            target_w = target_weights.get(ticker, 0.0)
            current_w = self._current_weights.get(ticker, 0.0)
            delta = target_w - current_w

            # Determine action.
            if abs(delta) < self._drift_threshold / 2.0:
                action = Action.HOLD
            elif delta > 0:
                action = Action.BUY
            else:
                action = Action.SELL

            # Conviction: proportional to target weight relative to max_weight.
            raw_conviction = target_w / self._max_weight if self._max_weight > 0 else 0.0
            conviction = self._compute_conviction(
                composite_score=raw_conviction,
                min_score=0.0,
                max_score=1.5,
            )

            # Price data for stop-loss / take-profit.
            price_df = prices.get(ticker, pd.DataFrame())
            if not price_df.empty and "close" in price_df.columns:
                current_price = float(price_df["close"].iloc[-1])
            else:
                current_price = 100.0  # Placeholder if price unavailable.

            # Conservative stop-loss and take-profit for the meta-allocator.
            stop_loss = round(current_price * 0.92, 2)   # 8 % stop
            take_profit = round(current_price * 1.15, 2)  # 15 % target

            exp_ret = float(expected_returns.get(ticker, 0.0)) if ticker in expected_returns.index else 0.0
            m_var = marginal_vars.get(ticker, 0.0)
            agreement = agent_agreement.get(ticker, 0)
            has_growth_tilt = 1.0 if ticker in growth_tilt_tickers else 0.0

            factor_scores = {
                "expected_return": round(exp_ret, 6),
                "marginal_var": round(m_var, 6),
                "agent_agreement": float(agreement),
                "growth_tilt": has_growth_tilt,
            }

            reasoning = (
                f"Portfolio optimiser: target_weight={target_w:.4f} "
                f"(current={current_w:.4f}, delta={delta:+.4f}). "
                f"Agent agreement={agreement}, "
                f"expected_return={exp_ret:.4f}, "
                f"marginal_VaR={m_var:.4f}."
            )
            if has_growth_tilt:
                reasoning += " Growth tilt applied (1.5x)."

            signals.append(
                AgentSignal(
                    ticker=ticker,
                    action=action,
                    conviction=conviction,
                    target_weight=round(target_w, 4),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    factor_scores=factor_scores,
                    reasoning=reasoning,
                    holding_period_days=7,  # Weekly rebalance cadence.
                    agent_name=self._name,
                    timestamp=now,
                )
            )

        # Sort by conviction descending for the orchestrator.
        signals.sort(key=lambda s: s.conviction, reverse=True)
        return signals

    def _emit_hold_signals(self, candidates: list[str]) -> list[AgentSignal]:
        """Emit HOLD signals when no rebalance is triggered.

        Maintains the current target weights without any trading.
        """
        now = datetime.now(tz=timezone.utc)
        signals: list[AgentSignal] = []

        for ticker in candidates:
            target_w = self._target_weights.get(ticker, 0.0)
            if target_w <= 0:
                continue

            current_price = 100.0  # Placeholder; real price not needed for HOLD.
            signals.append(
                AgentSignal(
                    ticker=ticker,
                    action=Action.HOLD,
                    conviction=50,
                    target_weight=round(target_w, 4),
                    stop_loss=round(current_price * 0.92, 2),
                    take_profit=round(current_price * 1.15, 2),
                    factor_scores={
                        "expected_return": 0.0,
                        "marginal_var": 0.0,
                        "agent_agreement": 0.0,
                        "growth_tilt": 0.0,
                    },
                    reasoning=f"HOLD: no rebalance triggered. Maintaining target_weight={target_w:.4f}.",
                    holding_period_days=7,
                    agent_name=self._name,
                    timestamp=now,
                )
            )

        return signals
