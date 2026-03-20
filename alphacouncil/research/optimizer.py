"""Nightly Bayesian parameter optimisation for trading agents.

Uses Optuna with TPE (Tree-structured Parzen Estimator) sampling to
search over each agent's tunable parameter space.  The objective is
walk-forward Sharpe ratio (via :class:`StrategyBacktester`), subject to
a drawdown constraint and a growth-bias penalty.

Safety rails:
* Parameters are bounded to +/-30% of their current values.
* Per-run application is capped at +/-15% change per parameter per week.
* Auto-application requires ``Settings.AUTO_TUNE = True``; otherwise the
  optimiser flags the improvement for human review.
* Every change is logged to the ``research_log`` table.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import optuna
import structlog
from sqlalchemy import text
from sqlalchemy.engine import Engine

from alphacouncil.agents.base import BaseAgent
from alphacouncil.core.config import get_settings
from alphacouncil.core.models import ResearchLog
from alphacouncil.research.backtester import StrategyBacktester

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_N_TRIALS: int = 50
_PARAM_SEARCH_RANGE_PCT: float = 0.30   # +/- 30%
_MAX_CHANGE_PER_WEEK_PCT: float = 0.15  # +/- 15%
_MIN_IMPROVEMENT_PCT: float = 0.10       # 10% improvement to apply
_GROWTH_PENALTY_WEIGHT: float = 0.05     # growth-bias penalty coefficient


# ---------------------------------------------------------------------------
# AgentParameterOptimizer
# ---------------------------------------------------------------------------


class AgentParameterOptimizer:
    """Bayesian hyper-parameter optimiser for trading agents.

    Runs nightly, taking each agent's current parameter set, defining
    a +/-30% search space around each value, and using Optuna TPE to
    maximise walk-forward Sharpe while constraining drawdown.

    Parameters
    ----------
    db_engine:
        SQLAlchemy :class:`Engine` for reading backtest data and
        writing optimisation logs.
    backtester:
        :class:`StrategyBacktester` used to evaluate candidate
        parameter sets via walk-forward simulation.
    """

    def __init__(
        self,
        db_engine: Engine,
        backtester: StrategyBacktester,
    ) -> None:
        self._db_engine = db_engine
        self._backtester = backtester
        self._log = logger.bind(component="optimizer")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def optimize(
        self,
        agent: BaseAgent,
        universe: list[str],
        n_trials: int = _DEFAULT_N_TRIALS,
    ) -> dict:
        """Run Bayesian optimisation for *agent*.

        Parameters
        ----------
        agent:
            The agent whose parameters will be tuned.
        universe:
            Stock universe for the backtester.
        n_trials:
            Number of Optuna trials.  Defaults to 50.

        Returns
        -------
        dict
            ``best_params`` : dict of optimised parameter values.
            ``improvement_pct`` : relative Sharpe improvement (%).
            ``applied`` : whether the new parameters were applied.
            ``baseline_sharpe`` : Sharpe before optimisation.
            ``optimized_sharpe`` : Sharpe after optimisation.
        """
        self._log.info(
            "optimize.start",
            agent=agent.name,
            n_trials=n_trials,
        )

        current_params = agent.get_parameters()
        if not current_params:
            self._log.warning("optimize.no_parameters", agent=agent.name)
            return {
                "best_params": {},
                "improvement_pct": 0.0,
                "applied": False,
                "baseline_sharpe": 0.0,
                "optimized_sharpe": 0.0,
            }

        # --- Baseline backtest -----------------------------------------------
        settings = get_settings()
        baseline_result = await self._backtester.backtest(
            agent=agent,
            universe=universe,
            start=self._default_start(),
            end=self._default_end(),
        )
        baseline_sharpe = baseline_result.get("sharpe", 0.0)
        baseline_max_dd = baseline_result.get("max_dd", 1.0)

        # --- Build search space ----------------------------------------------
        search_space = self._create_search_space(current_params)
        self._log.info(
            "optimize.search_space",
            agent=agent.name,
            n_params=len(search_space),
        )

        # --- Run Optuna study ------------------------------------------------
        # Optuna is synchronous so we wrap in executor
        best_params, best_sharpe = await self._run_optuna_study(
            agent=agent,
            universe=universe,
            search_space=search_space,
            current_params=current_params,
            baseline_max_dd=baseline_max_dd,
            n_trials=n_trials,
        )

        # --- Evaluate improvement --------------------------------------------
        improvement_pct = 0.0
        if baseline_sharpe > 0:
            improvement_pct = (best_sharpe - baseline_sharpe) / baseline_sharpe
        elif best_sharpe > 0:
            improvement_pct = 1.0  # any positive is an improvement from zero

        applied = False

        if improvement_pct > _MIN_IMPROVEMENT_PCT:
            if settings.AUTO_TUNE:
                self._apply_with_bounds(
                    agent, best_params, max_change=_MAX_CHANGE_PER_WEEK_PCT,
                )
                applied = True
                self._log.info(
                    "optimize.auto_applied",
                    agent=agent.name,
                    improvement_pct=round(improvement_pct * 100, 2),
                )
            else:
                self._log.info(
                    "optimize.flagged_for_review",
                    agent=agent.name,
                    improvement_pct=round(improvement_pct * 100, 2),
                    best_params=best_params,
                )

            # Log every change
            await self._log_changes(
                agent=agent,
                current_params=current_params,
                best_params=best_params,
                improvement_pct=improvement_pct,
                applied=applied,
            )
        else:
            self._log.info(
                "optimize.no_improvement",
                agent=agent.name,
                improvement_pct=round(improvement_pct * 100, 2),
            )

        return {
            "best_params": best_params,
            "improvement_pct": round(improvement_pct, 6),
            "applied": applied,
            "baseline_sharpe": round(baseline_sharpe, 4),
            "optimized_sharpe": round(best_sharpe, 4),
        }

    # ------------------------------------------------------------------
    # Search space construction
    # ------------------------------------------------------------------

    @staticmethod
    def _create_search_space(current_params: dict[str, Any]) -> dict[str, dict]:
        """Define Optuna search ranges: +/-30% around each current value.

        Parameters
        ----------
        current_params:
            The agent's current parameter dictionary.

        Returns
        -------
        dict[str, dict]
            ``{param_name: {"type": "float"|"int", "low": X, "high": Y,
            "current": Z}}``
        """
        space: dict[str, dict] = {}

        for name, value in current_params.items():
            if isinstance(value, bool):
                # Booleans are not tunable via continuous search
                continue

            if isinstance(value, float):
                low = value * (1.0 - _PARAM_SEARCH_RANGE_PCT)
                high = value * (1.0 + _PARAM_SEARCH_RANGE_PCT)
                # Handle negative values
                if value < 0:
                    low, high = high, low
                # Handle zero
                if value == 0.0:
                    low, high = -0.1, 0.1
                space[name] = {
                    "type": "float",
                    "low": low,
                    "high": high,
                    "current": value,
                }

            elif isinstance(value, int):
                delta = max(1, int(abs(value) * _PARAM_SEARCH_RANGE_PCT))
                space[name] = {
                    "type": "int",
                    "low": value - delta,
                    "high": value + delta,
                    "current": value,
                }

        return space

    # ------------------------------------------------------------------
    # Bounded parameter application
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_with_bounds(
        agent: BaseAgent,
        best_params: dict[str, Any],
        max_change: float = _MAX_CHANGE_PER_WEEK_PCT,
    ) -> None:
        """Apply *best_params* to *agent*, capping change per parameter.

        Each parameter is clamped so that its absolute change does not
        exceed ``max_change`` (default 15%) of the current value.  This
        prevents sudden regime shocks from destabilising a live agent.

        Parameters
        ----------
        agent:
            Target agent.
        best_params:
            Optimised parameter values from Optuna.
        max_change:
            Maximum fractional change allowed per parameter per week.
        """
        current = agent.get_parameters()
        bounded: dict[str, Any] = {}

        for name, new_val in best_params.items():
            old_val = current.get(name)
            if old_val is None:
                bounded[name] = new_val
                continue

            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                if old_val == 0:
                    # Cannot compute fractional change; apply directly if small
                    bounded[name] = type(old_val)(np.clip(new_val, -0.15, 0.15))
                    continue

                pct_change = (new_val - old_val) / abs(old_val)
                clamped_pct = float(np.clip(pct_change, -max_change, max_change))
                clamped_val = old_val * (1.0 + clamped_pct)

                # Preserve type
                if isinstance(old_val, int):
                    clamped_val = int(round(clamped_val))

                bounded[name] = clamped_val
            else:
                bounded[name] = new_val

        agent.set_parameters(bounded)

    # ------------------------------------------------------------------
    # Optuna study
    # ------------------------------------------------------------------

    async def _run_optuna_study(
        self,
        agent: BaseAgent,
        universe: list[str],
        search_space: dict[str, dict],
        current_params: dict[str, Any],
        baseline_max_dd: float,
        n_trials: int,
    ) -> tuple[dict[str, Any], float]:
        """Run the Optuna study in a thread pool.

        Returns ``(best_params, best_sharpe)``.
        """
        loop = asyncio.get_running_loop()

        # Create study
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"optimize_{agent.name}",
        )

        # Suppress Optuna logs (structlog is primary)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        best_sharpe: float = 0.0
        best_params: dict[str, Any] = {}
        dd_constraint = baseline_max_dd * 1.2  # allow 20% more DD

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_sharpe, best_params

            # Sample parameters
            sampled: dict[str, Any] = {}
            for name, spec in search_space.items():
                if spec["type"] == "float":
                    sampled[name] = trial.suggest_float(
                        name, spec["low"], spec["high"],
                    )
                elif spec["type"] == "int":
                    sampled[name] = trial.suggest_int(
                        name, spec["low"], spec["high"],
                    )

            # Apply to agent temporarily
            original = agent.get_parameters()
            agent.set_parameters({**original, **sampled})

            try:
                # Run backtest synchronously (we are in executor)
                result = asyncio.run(
                    self._backtester.backtest(
                        agent=agent,
                        universe=universe,
                        start=self._default_start(),
                        end=self._default_end(),
                    )
                )
            except Exception:
                agent.set_parameters(original)
                return float("-inf")

            sharpe = result.get("sharpe", 0.0)
            max_dd = result.get("max_dd", 1.0)

            # Restore original parameters
            agent.set_parameters(original)

            # Constraint: max DD
            if max_dd > dd_constraint:
                return float("-inf")

            # Growth bias: penalize parameter changes that reduce growth allocation
            growth_penalty = self._growth_allocation_penalty(
                current_params, sampled,
            )
            adjusted_sharpe = sharpe - growth_penalty

            if adjusted_sharpe > best_sharpe:
                best_sharpe = adjusted_sharpe
                best_params = dict(sampled)

            return adjusted_sharpe

        await loop.run_in_executor(
            None,
            lambda: study.optimize(objective, n_trials=n_trials),
        )

        return best_params, best_sharpe

    # ------------------------------------------------------------------
    # Growth-bias penalty
    # ------------------------------------------------------------------

    @staticmethod
    def _growth_allocation_penalty(
        current_params: dict[str, Any],
        new_params: dict[str, Any],
    ) -> float:
        """Penalize parameter changes that reduce allocation to high-growth stocks.

        Heuristic: if any parameter named with 'growth', 'momentum', or
        'revenue' decreases, impose a penalty proportional to the decrease.
        """
        growth_keywords = {"growth", "momentum", "revenue", "earnings", "eps"}
        penalty = 0.0

        for name, new_val in new_params.items():
            name_lower = name.lower()
            is_growth_param = any(kw in name_lower for kw in growth_keywords)

            if not is_growth_param:
                continue

            old_val = current_params.get(name)
            if old_val is None or not isinstance(old_val, (int, float)):
                continue
            if not isinstance(new_val, (int, float)):
                continue

            if old_val > 0 and new_val < old_val:
                reduction_pct = (old_val - new_val) / old_val
                penalty += reduction_pct * _GROWTH_PENALTY_WEIGHT

        return penalty

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    async def _log_changes(
        self,
        agent: BaseAgent,
        current_params: dict[str, Any],
        best_params: dict[str, Any],
        improvement_pct: float,
        applied: bool,
    ) -> None:
        """Persist optimisation results to the research_log table."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._log_changes_sync,
            agent.name,
            current_params,
            best_params,
            improvement_pct,
            applied,
        )

    def _log_changes_sync(
        self,
        agent_name: str,
        current_params: dict[str, Any],
        best_params: dict[str, Any],
        improvement_pct: float,
        applied: bool,
    ) -> None:
        """Write each changed parameter to research_logs."""
        try:
            with self._db_engine.begin() as conn:
                for name, new_val in best_params.items():
                    old_val = current_params.get(name, "N/A")
                    conn.execute(
                        text("""
                            INSERT INTO research_logs
                                (agent_id, query, result_summary, sources)
                            VALUES
                                (:agent, :query, :result, :sources)
                        """),
                        {
                            "agent": agent_name,
                            "query": f"optimize_param:{name}",
                            "result": json.dumps({
                                "old_value": str(old_val),
                                "new_value": str(new_val),
                                "improvement_pct": round(improvement_pct * 100, 2),
                                "applied": applied,
                            }),
                            "sources": "optuna_tpe_bayesian",
                        },
                    )

            self._log.info(
                "optimize.logged",
                agent=agent_name,
                n_params=len(best_params),
            )
        except Exception as exc:
            self._log.error(
                "optimize.log_failed",
                agent=agent_name,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_start() -> str:
        """Default backtest start: 3 years ago."""
        dt = datetime.now(tz=timezone.utc) - __import__("datetime").timedelta(days=3 * 365)
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def _default_end() -> str:
        """Default backtest end: today."""
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
