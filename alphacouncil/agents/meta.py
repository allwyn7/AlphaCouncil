"""MetaAgent -- THE COUNCIL.

Central decision-maker that aggregates signals from all 6 quant agents,
blends them using regime-dependent weights (learned via Thompson Sampling),
sizes positions using a half-Kelly criterion, runs risk checks, and
generates executable orders.

The MetaAgent does **not** inherit from :class:`BaseAgent` because it is
not itself a signal-generating agent -- it is the *orchestrator* that sits
above the six alpha agents.

RL Nightly Update (Thompson Sampling contextual bandit)
-------------------------------------------------------
* **State** = (VIX_bucket: low/med/high, Nifty_trend: up/down/flat,
  FII_flow_dir: buying/selling)
* **Action** = weight vector for 6 agents
* **Reward** = daily portfolio Sharpe ratio
* Beta(1,1) priors, updated with realised trade outcomes
* Method: :meth:`update_rl_weights`
"""

from __future__ import annotations

import time as _time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog

from alphacouncil.agents.base import BaseAgent
from alphacouncil.agents.news_shock import NewsShockDetector, NewsShock
from alphacouncil.core.models import (
    Action,
    AgentSignal,
    AgentStatus,
    Exchange,
    MarketRegime,
    Order,
    OrderSide,
    OrderType,
    PortfolioState,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical names of the six alpha agents in the council.
AGENT_NAMES: list[str] = [
    "growth_momentum",
    "mean_reversion",
    "sentiment_alpha",
    "fundamental_value",
    "volatility_regime",
    "macro_flow",
]

#: Agents that receive a 1.5x base-weight boost.
_GROWTH_AGENTS: frozenset[str] = frozenset({
    "growth_momentum",
    "sentiment_alpha",
})

#: Default base weight assigned to each agent.
_DEFAULT_BASE_WEIGHT: float = 1.0

#: Multiplier for growth-tilted agents.
_GROWTH_WEIGHT_MULTIPLIER: float = 1.5

#: Minimum agreement threshold (fraction of agents) to act.
_MIN_AGREEMENT_FRACTION: float = 0.4

#: Hard cap on the Kelly fraction to prevent ruin.
_MAX_KELLY_FRACTION: float = 0.25

#: Minimum conviction (0-100) to consider a signal actionable.
_MIN_CONVICTION: int = 30

#: Thompson Sampling prior parameters (Beta distribution).
_TS_ALPHA_PRIOR: float = 1.0
_TS_BETA_PRIOR: float = 1.0

# ---------------------------------------------------------------------------
# VIX / regime bucketing helpers
# ---------------------------------------------------------------------------

_VIX_BUCKETS: dict[str, tuple[float, float]] = {
    "low": (0.0, 15.0),
    "med": (15.0, 25.0),
    "high": (25.0, float("inf")),
}

_NIFTY_TREND_LABELS: list[str] = ["up", "down", "flat"]
_FII_FLOW_LABELS: list[str] = ["buying", "selling"]


def _regime_key(regime_state: dict[str, str]) -> str:
    """Build a hashable key from a regime state dict."""
    vix = regime_state.get("vix_bucket", "med")
    trend = regime_state.get("nifty_trend", "flat")
    fii = regime_state.get("fii_flow_dir", "buying")
    return f"{vix}|{trend}|{fii}"


# ---------------------------------------------------------------------------
# MetaAgent
# ---------------------------------------------------------------------------


class MetaAgent:
    """The Council -- aggregates, blends, sizes, and routes orders.

    Parameters
    ----------
    agents:
        The six alpha agents (subclasses of :class:`BaseAgent`).
    risk_manager:
        A :class:`~alphacouncil.core.risk_manager.RiskManager` instance for
        pre-trade validation and post-trade checks.
    broker:
        A :class:`~alphacouncil.core.broker.base.BrokerAdapter` for order
        execution (paper or live).
    config:
        System-wide :class:`~alphacouncil.core.config.Settings`.
    bus:
        Shared message bus for publishing council decisions.
    db_engine:
        SQLAlchemy engine for persistence.
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        risk_manager: Any,
        broker: Any,
        config: Any,
        bus: Any,
        db_engine: Any,
    ) -> None:
        self._agents = agents
        self._risk_manager = risk_manager
        self._broker = broker
        self._config = config
        self._bus = bus
        self._db_engine = db_engine

        self._log = logger.bind(component="meta_agent")

        # Map agent name -> BaseAgent for fast lookup.
        self._agent_map: dict[str, BaseAgent] = {a.name: a for a in agents}

        # --- Thompson Sampling bandit state ---
        # Per (regime_key, agent_name) we maintain Beta(alpha, beta) params.
        # Keyed: regime_key -> agent_name -> {"alpha": float, "beta": float}
        self._ts_params: dict[str, dict[str, dict[str, float]]] = defaultdict(
            lambda: {
                name: {"alpha": _TS_ALPHA_PRIOR, "beta": _TS_BETA_PRIOR}
                for name in AGENT_NAMES
            }
        )

        # Historical trade performance (for Kelly computation).
        # Keyed by agent_name.
        self._trade_history: dict[str, list[dict[str, float]]] = defaultdict(list)

        self._log.info(
            "meta_agent.initialised",
            agent_count=len(agents),
            agent_names=[a.name for a in agents],
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def run_council(
        self,
        universe: list[str],
        market_data: dict[str, Any],
        regime: MarketRegime,
    ) -> list[Order]:
        """Execute the full council pipeline and return executable orders.

        Steps
        -----
        1. Collect signals from all agents.
        2. Compute agreement scores per ticker.
        3. Compute weighted blend per ticker.
        4. Size positions via half-Kelly.
        5. Run risk checks via RiskManager.
        6. Generate and return orders.
        """
        cycle_start = _time.perf_counter_ns()
        self._log.info(
            "council.cycle_start",
            universe_size=len(universe),
            regime=regime.value,
        )

        # 1. Collect signals ------------------------------------------------
        signals_by_ticker: dict[str, list[AgentSignal]] = defaultdict(list)

        for agent in self._agents:
            try:
                agent_signals = await agent.run_cycle(universe, market_data)
                for sig in agent_signals:
                    if sig.conviction >= _MIN_CONVICTION:
                        signals_by_ticker[sig.ticker].append(sig)
            except Exception:
                self._log.exception(
                    "council.agent_cycle_failed",
                    agent=agent.name,
                )

        if not signals_by_ticker:
            self._log.info("council.no_signals")
            return []

        # 1b. News shock filter ------------------------------------------
        shock_detector = NewsShockDetector()
        sentiment_data = market_data.get("sentiment", {})
        sentiment_signals_data = market_data.get("sentiment_signals", {})

        for ticker, sigs in list(signals_by_ticker.items()):
            ticker_sent = sentiment_data.get(ticker, {})
            ticker_sent_signals = sentiment_signals_data.get(ticker, {})
            shock = shock_detector.detect_shock(ticker, ticker_sent_signals, ticker_sent)

            if not shock.is_shock:
                continue

            filtered_sigs: list[AgentSignal] = []
            for sig in sigs:
                new_conviction = sig.conviction
                new_action = sig.action
                new_reasoning = sig.reasoning

                if shock.severity > 0.8:
                    # Extreme shock: override to HOLD
                    new_action = Action.HOLD
                    new_conviction = max(int(sig.conviction * 0.2), 0)
                    new_reasoning = f"[NEWS SHOCK OVERRIDE severity={shock.severity:.2f}] {sig.reasoning}"
                    self._log.warning(
                        "council.news_shock_override",
                        ticker=ticker,
                        agent=sig.agent_name,
                        severity=shock.severity,
                    )
                elif shock.direction == "negative" and sig.action == Action.BUY:
                    dampening = shock.severity * 0.70
                    new_conviction = max(int(sig.conviction * (1 - dampening)), 0)
                    new_reasoning = f"[NEWS SHOCK DAMPENED -{dampening:.0%}] {sig.reasoning}"
                elif shock.direction == "positive" and sig.action == Action.SELL:
                    dampening = shock.severity * 0.50
                    new_conviction = max(int(sig.conviction * (1 - dampening)), 0)
                    new_reasoning = f"[NEWS SHOCK DAMPENED -{dampening:.0%}] {sig.reasoning}"

                if new_action == Action.HOLD and new_conviction < _MIN_CONVICTION:
                    continue  # drop the signal entirely

                # Rebuild signal with adjusted values (frozen model, so recreate)
                filtered_sigs.append(AgentSignal(
                    ticker=sig.ticker,
                    action=new_action,
                    conviction=new_conviction,
                    target_weight=sig.target_weight,
                    stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit,
                    factor_scores=sig.factor_scores,
                    reasoning=new_reasoning,
                    holding_period_days=sig.holding_period_days,
                    agent_name=sig.agent_name,
                    timestamp=sig.timestamp,
                ))

            if filtered_sigs:
                signals_by_ticker[ticker] = filtered_sigs
            else:
                del signals_by_ticker[ticker]

        if not signals_by_ticker:
            self._log.info("council.all_signals_filtered_by_news_shock")
            return []

        # 2. Agreement scores -----------------------------------------------
        agreement = self._compute_agreement(signals_by_ticker)

        # 3. Weighted blend -------------------------------------------------
        blended = self._weighted_blend(signals_by_ticker, regime)

        # 4. Position sizing via half-Kelly ---------------------------------
        decisions: dict[str, dict[str, Any]] = {}
        for ticker, score in blended.items():
            if abs(score) < 0.05:
                continue  # too weak to act on
            if agreement.get(ticker, 0.0) < _MIN_AGREEMENT_FRACTION:
                self._log.debug(
                    "council.low_agreement",
                    ticker=ticker,
                    agreement=agreement[ticker],
                )
                continue

            # Determine dominant direction.
            direction = Action.BUY if score > 0 else Action.SELL

            # Lookup best signal for stop/take-profit.
            best_signal = max(
                signals_by_ticker[ticker],
                key=lambda s: s.conviction,
            )

            # Compute Kelly fraction from agent's historical performance.
            agent_name = best_signal.agent_name
            win_rate, wl_ratio = self._get_win_loss_stats(agent_name)
            kelly_frac = self._half_kelly(win_rate, wl_ratio)

            decisions[ticker] = {
                "direction": direction,
                "score": score,
                "kelly_fraction": kelly_frac,
                "conviction": best_signal.conviction,
                "stop_loss": best_signal.stop_loss,
                "take_profit": best_signal.take_profit,
                "reasoning": best_signal.reasoning,
                "agent_name": agent_name,
            }

        if not decisions:
            self._log.info("council.no_decisions_after_sizing")
            return []

        # 5. Generate orders ------------------------------------------------
        portfolio = await self._get_portfolio_state()
        raw_orders = await self._generate_orders(decisions, portfolio)

        # 6. Risk check -----------------------------------------------------
        validated = await self._risk_manager.validate_orders(
            raw_orders, portfolio,
        )

        approved_orders: list[Order] = []
        for order, passed, reason in validated:
            if passed:
                approved_orders.append(order)
            else:
                self._log.warning(
                    "council.order_rejected",
                    ticker=order.ticker,
                    reason=reason,
                )

        # 7. Execute via broker ---------------------------------------------
        executed_orders: list[Order] = []
        for order in approved_orders:
            try:
                broker_id = await self._broker.place_order(order)
                executed_orders.append(order)
                self._log.info(
                    "council.order_executed",
                    ticker=order.ticker,
                    side=order.side.value,
                    quantity=order.quantity,
                    broker_id=broker_id,
                )
            except Exception:
                self._log.exception(
                    "council.order_execution_failed",
                    ticker=order.ticker,
                )

        # 8. Post-trade risk check ------------------------------------------
        try:
            updated_portfolio = await self._get_portfolio_state()
            await self._risk_manager.post_trade_check(updated_portfolio)
        except Exception:
            self._log.exception("council.post_trade_check_failed")

        # Publish results to bus.
        try:
            await self._bus.publish(
                "signal",
                {
                    "event": "council_cycle_complete",
                    "orders_generated": len(raw_orders),
                    "orders_approved": len(approved_orders),
                    "orders_executed": len(executed_orders),
                    "regime": regime.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                publisher="meta_agent",
            )
        except Exception:
            self._log.exception("council.bus_publish_failed")

        elapsed_ms = (_time.perf_counter_ns() - cycle_start) / 1_000_000
        self._log.info(
            "council.cycle_complete",
            decisions=len(decisions),
            orders_approved=len(approved_orders),
            orders_executed=len(executed_orders),
            elapsed_ms=round(elapsed_ms, 2),
        )
        return executed_orders

    # ------------------------------------------------------------------
    # Agreement scoring
    # ------------------------------------------------------------------

    def _compute_agreement(
        self,
        signals_by_ticker: dict[str, list[AgentSignal]],
    ) -> dict[str, float]:
        """Compute inter-agent agreement score per ticker.

        The agreement score is the fraction of signalling agents that agree
        on the dominant direction (BUY or SELL).  HOLD signals are excluded.

        Returns
        -------
        dict[str, float]
            Ticker -> agreement score in [0.0, 1.0].
        """
        result: dict[str, float] = {}

        for ticker, signals in signals_by_ticker.items():
            directional = [s for s in signals if s.action != Action.HOLD]
            if not directional:
                result[ticker] = 0.0
                continue

            buy_count = sum(1 for s in directional if s.action == Action.BUY)
            sell_count = len(directional) - buy_count
            dominant_count = max(buy_count, sell_count)
            result[ticker] = dominant_count / len(self._agents)

        return result

    # ------------------------------------------------------------------
    # Weighted blend
    # ------------------------------------------------------------------

    def _weighted_blend(
        self,
        signals_by_ticker: dict[str, list[AgentSignal]],
        regime: MarketRegime,
    ) -> dict[str, float]:
        """Compute weighted blended score per ticker.

        Each agent's contribution:
            agent_weight[regime] * conviction_normalized * action_sign

        Where ``action_sign`` is +1 (BUY), -1 (SELL), 0 (HOLD).

        Growth agents (GrowthMomentumAgent, SentimentAlphaAgent) start
        with a 1.5x base-weight multiplier.

        Returns
        -------
        dict[str, float]
            Ticker -> blended score (positive = BUY bias, negative = SELL).
        """
        weights = self.get_agent_weights(regime)
        result: dict[str, float] = {}

        for ticker, signals in signals_by_ticker.items():
            blended_score = 0.0
            for sig in signals:
                agent_w = weights.get(sig.agent_name, _DEFAULT_BASE_WEIGHT)
                conviction_norm = sig.conviction / 100.0
                action_sign = self._action_to_sign(sig.action)
                blended_score += agent_w * conviction_norm * action_sign
            result[ticker] = blended_score

        return result

    # ------------------------------------------------------------------
    # Half-Kelly position sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _half_kelly(win_rate: float, win_loss_ratio: float) -> float:
        """Compute the half-Kelly fraction for position sizing.

        Formula
        -------
        f = 0.5 * (p * b - q) / b

        Where:
            p = historical win rate
            b = avg_win / avg_loss ratio
            q = 1 - p

        The result is clamped to [0, MAX_KELLY_FRACTION] to prevent ruin.

        Parameters
        ----------
        win_rate:
            Fraction of winning trades, in [0.0, 1.0].
        win_loss_ratio:
            Average win / average loss magnitude ratio.

        Returns
        -------
        float
            Position size as fraction of capital, in [0.0, MAX_KELLY_FRACTION].
        """
        if win_loss_ratio <= 0 or win_rate <= 0:
            return 0.0

        p = np.clip(win_rate, 0.0, 1.0)
        q = 1.0 - p
        b = win_loss_ratio

        kelly = 0.5 * (p * b - q) / b
        return float(np.clip(kelly, 0.0, _MAX_KELLY_FRACTION))

    # ------------------------------------------------------------------
    # Order generation
    # ------------------------------------------------------------------

    async def _generate_orders(
        self,
        decisions: dict[str, dict[str, Any]],
        portfolio: PortfolioState,
    ) -> list[Order]:
        """Convert sizing decisions to Order objects.

        Parameters
        ----------
        decisions:
            Ticker -> decision dict with direction, kelly_fraction, etc.
        portfolio:
            Current portfolio state for capital calculations.

        Returns
        -------
        list[Order]
            Ready-to-validate order tickets.
        """
        orders: list[Order] = []
        capital = portfolio.total_value

        for ticker, dec in decisions.items():
            direction: Action = dec["direction"]
            kelly_frac: float = dec["kelly_fraction"]

            if direction == Action.HOLD:
                continue

            # Capital to allocate = Kelly fraction * total capital.
            alloc = kelly_frac * capital

            # Use stop-loss as a conservative price estimate for quantity.
            price_estimate = dec.get("stop_loss", 0.0)
            if price_estimate <= 0:
                self._log.warning(
                    "council.no_price_estimate",
                    ticker=ticker,
                )
                continue

            # For sells, use the take-profit side as price estimate.
            if direction == Action.SELL:
                price_estimate = dec.get("take_profit", price_estimate)

            quantity = max(1, int(alloc / price_estimate))

            side = OrderSide.BUY if direction == Action.BUY else OrderSide.SELL

            # For SELL orders, cap quantity at existing holdings.
            if side == OrderSide.SELL:
                existing_qty = sum(
                    pos.quantity
                    for pos in portfolio.positions
                    if pos.ticker == ticker
                )
                if existing_qty <= 0:
                    self._log.debug(
                        "council.skip_sell_no_position",
                        ticker=ticker,
                    )
                    continue
                quantity = min(quantity, existing_qty)

            order = Order(
                order_id=f"council-{uuid.uuid4().hex[:12]}",
                ticker=ticker,
                exchange=Exchange.NSE,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=None,
                trigger_price=None,
                agent_name="meta_agent",
                reasoning=(
                    f"Council decision: {direction.value} "
                    f"score={dec['score']:.3f} "
                    f"kelly={kelly_frac:.3f} "
                    f"conviction={dec['conviction']} | "
                    f"{dec['reasoning']}"
                ),
                timestamp=datetime.now(timezone.utc),
            )
            orders.append(order)

        return orders

    # ------------------------------------------------------------------
    # Thompson Sampling RL weight update
    # ------------------------------------------------------------------

    async def update_rl_weights(
        self,
        daily_returns: pd.Series,
        regime_state: dict[str, str],
    ) -> None:
        """Nightly Thompson Sampling update for agent weights.

        Uses a contextual bandit formulation:
        * State = regime_state (VIX bucket, Nifty trend, FII flow dir).
        * Action = which agent to up-weight.
        * Reward = whether daily Sharpe improvement is positive.

        Each agent's Beta(alpha, beta) posterior is updated based on
        whether the portfolio performed positively today.

        Parameters
        ----------
        daily_returns:
            Per-agent daily return series (index = agent names).
        regime_state:
            Dict with keys ``vix_bucket``, ``nifty_trend``, ``fii_flow_dir``.
        """
        rkey = _regime_key(regime_state)
        params = self._ts_params[rkey]

        # Compute reward: 1 if daily Sharpe > 0, else 0.
        if len(daily_returns) < 2:
            self._log.debug("council.rl_update_skip_insufficient_data")
            return

        portfolio_mean = daily_returns.mean()
        portfolio_std = daily_returns.std(ddof=1)
        daily_sharpe = (
            portfolio_mean / portfolio_std if portfolio_std > 0 else 0.0
        )
        reward = 1.0 if daily_sharpe > 0 else 0.0

        for agent_name in AGENT_NAMES:
            agent_params = params[agent_name]

            # Check if this agent contributed positively today.
            agent_return = daily_returns.get(agent_name, 0.0)
            agent_reward = 1.0 if agent_return > 0 else 0.0

            # Blend portfolio-level and agent-level reward.
            combined_reward = 0.6 * reward + 0.4 * agent_reward

            if combined_reward > 0.5:
                agent_params["alpha"] += combined_reward
            else:
                agent_params["beta"] += 1.0 - combined_reward

        self._log.info(
            "council.rl_weights_updated",
            regime_key=rkey,
            daily_sharpe=round(daily_sharpe, 4),
            reward=reward,
        )

    def get_agent_weights(self, regime: MarketRegime) -> dict[str, float]:
        """Sample agent weights from Thompson Sampling posteriors.

        For each agent, sample from Beta(alpha, beta) to get a raw weight,
        then apply the growth-agent 1.5x multiplier.

        Parameters
        ----------
        regime:
            Current market regime (used to derive regime_state key).

        Returns
        -------
        dict[str, float]
            Agent name -> weight.
        """
        # Map MarketRegime enum to a crude regime_state for TS lookup.
        rstate = self._regime_to_state(regime)
        rkey = _regime_key(rstate)
        params = self._ts_params[rkey]

        weights: dict[str, float] = {}
        for agent_name in AGENT_NAMES:
            ap = params[agent_name]
            # Sample from Beta posterior.
            sampled = float(
                np.random.beta(ap["alpha"], ap["beta"])
            )
            # Apply growth multiplier.
            base = (
                _DEFAULT_BASE_WEIGHT * _GROWTH_WEIGHT_MULTIPLIER
                if agent_name in _GROWTH_AGENTS
                else _DEFAULT_BASE_WEIGHT
            )
            weights[agent_name] = base * sampled

        return weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_to_sign(action: Action) -> float:
        """Convert an Action enum to a numeric sign."""
        if action == Action.BUY:
            return 1.0
        elif action == Action.SELL:
            return -1.0
        return 0.0

    def _get_win_loss_stats(
        self,
        agent_name: str,
    ) -> tuple[float, float]:
        """Retrieve historical win rate and win/loss ratio for an agent.

        Returns
        -------
        tuple[float, float]
            (win_rate, avg_win / avg_loss ratio). Defaults to (0.5, 1.5)
            when insufficient history.
        """
        history = self._trade_history.get(agent_name, [])

        if len(history) < 10:
            # Insufficient data -- use conservative defaults.
            return 0.5, 1.5

        wins = [t["pnl"] for t in history if t["pnl"] > 0]
        losses = [abs(t["pnl"]) for t in history if t["pnl"] < 0]

        win_rate = len(wins) / len(history) if history else 0.5
        avg_win = np.mean(wins) if wins else 1.0
        avg_loss = np.mean(losses) if losses else 1.0
        wl_ratio = float(avg_win / avg_loss) if avg_loss > 0 else 1.5

        return win_rate, wl_ratio

    def record_trade_outcome(
        self,
        agent_name: str,
        pnl: float,
        holding_days: int,
    ) -> None:
        """Record a closed trade outcome for Kelly and RL computations.

        Parameters
        ----------
        agent_name:
            The agent that originated the trade.
        pnl:
            Realised profit/loss in INR.
        holding_days:
            Number of days the position was held.
        """
        self._trade_history[agent_name].append({
            "pnl": pnl,
            "holding_days": holding_days,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def _get_portfolio_state(self) -> PortfolioState:
        """Retrieve current portfolio state from the broker.

        Falls back to a minimal empty portfolio if the broker call fails.
        """
        try:
            positions = await self._broker.get_positions()
            funds = await self._broker.get_funds()
            cash = funds.get("cash", 0.0)
            total = funds.get("total", cash)
            deployed = funds.get("deployed", 0.0)
            deployed_pct = deployed / total if total > 0 else 0.0

            # Compute daily P&L from positions.
            daily_pnl = sum(p.pnl for p in positions)
            daily_pnl_pct = daily_pnl / total if total > 0 else 0.0

            return PortfolioState(
                cash=cash,
                positions=positions,
                total_value=max(total, 1.0),
                deployed_pct=min(deployed_pct, 1.0),
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                drawdown_from_peak=0.0,  # computed externally
            )
        except Exception:
            self._log.exception("council.portfolio_state_fetch_failed")
            capital = getattr(self._config, "INITIAL_CAPITAL", 1_000_000.0)
            return PortfolioState(
                cash=capital,
                positions=[],
                total_value=capital,
                deployed_pct=0.0,
                daily_pnl=0.0,
                daily_pnl_pct=0.0,
                drawdown_from_peak=0.0,
            )

    @staticmethod
    def _regime_to_state(regime: MarketRegime) -> dict[str, str]:
        """Map a :class:`MarketRegime` enum to a Thompson Sampling state dict.

        This is a heuristic mapping; a more sophisticated version would
        consume the raw VIX / Nifty / FII values directly.
        """
        mapping: dict[MarketRegime, dict[str, str]] = {
            MarketRegime.BULL_LOW_VOL: {
                "vix_bucket": "low",
                "nifty_trend": "up",
                "fii_flow_dir": "buying",
            },
            MarketRegime.BULL_HIGH_VOL: {
                "vix_bucket": "high",
                "nifty_trend": "up",
                "fii_flow_dir": "buying",
            },
            MarketRegime.BEAR_LOW_VOL: {
                "vix_bucket": "low",
                "nifty_trend": "down",
                "fii_flow_dir": "selling",
            },
            MarketRegime.BEAR_HIGH_VOL: {
                "vix_bucket": "high",
                "nifty_trend": "down",
                "fii_flow_dir": "selling",
            },
            MarketRegime.SIDEWAYS: {
                "vix_bucket": "med",
                "nifty_trend": "flat",
                "fii_flow_dir": "buying",
            },
            MarketRegime.FII_BUYING: {
                "vix_bucket": "med",
                "nifty_trend": "up",
                "fii_flow_dir": "buying",
            },
            MarketRegime.FII_SELLING: {
                "vix_bucket": "med",
                "nifty_trend": "down",
                "fii_flow_dir": "selling",
            },
            MarketRegime.PRE_EXPIRY: {
                "vix_bucket": "high",
                "nifty_trend": "flat",
                "fii_flow_dir": "selling",
            },
            MarketRegime.EARNINGS_SEASON: {
                "vix_bucket": "high",
                "nifty_trend": "flat",
                "fii_flow_dir": "buying",
            },
            MarketRegime.BUDGET_POLICY: {
                "vix_bucket": "high",
                "nifty_trend": "flat",
                "fii_flow_dir": "buying",
            },
        }
        return mapping.get(regime, {
            "vix_bucket": "med",
            "nifty_trend": "flat",
            "fii_flow_dir": "buying",
        })

    def __repr__(self) -> str:
        return (
            f"<MetaAgent agents={len(self._agents)} "
            f"ts_regimes={len(self._ts_params)}>"
        )
