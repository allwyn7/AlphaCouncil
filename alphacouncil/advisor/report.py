"""Report generation module for the Investment Advisor.

Provides high-level reporting capabilities including individual stock
reports, portfolio suggestions with risk-parity allocation, and a
broad market overview snapshot. All methods are async and rely on
:class:`InvestmentAdvisor` for deep analysis and :class:`StockScreener`
for rapid universe screening.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import structlog
import yfinance as yf

from alphacouncil.advisor.engine import InvestmentAdvisor
from alphacouncil.advisor.models import (
    AdvisorAction,
    InvestmentHorizon,
    MarketOverview,
    PortfolioAllocation,
    PortfolioSuggestion,
    RiskAppetite,
    ScreenerResultItem,
)
from alphacouncil.advisor.screener import StockScreener
from alphacouncil.advisor.universes import get_sector, get_universe  # noqa: F401

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MARKET_TICKERS: dict[str, str] = {
    "nifty50": "^NSEI",
    "sensex": "^BSESN",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "india_vix": "^INDIAVIX",
    "us_vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "gold": "GC=F",
    "brent": "BZ=F",
}

_NIFTY_SECTOR_INDICES: dict[str, str] = {
    "NIFTY_IT": "^CNXIT",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY_REALTY": "^CNXREALTY",
    "NIFTY_INFRA": "^CNXINFRA",
    "NIFTY_PSE": "^CNXPSE",
}

# VIX signal thresholds (India VIX)
_VIX_LOW = 14.0
_VIX_NORMAL = 20.0
_VIX_ELEVATED = 25.0


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generates comprehensive reports, portfolio suggestions, and market overviews.

    Parameters
    ----------
    advisor:
        An :class:`InvestmentAdvisor` instance for full stock analysis.
    screener:
        A :class:`StockScreener` instance for rapid universe screening.
    """

    def __init__(self, advisor: InvestmentAdvisor, screener: StockScreener) -> None:
        self._advisor = advisor
        self._screener = screener
        self._log = logger.bind(component="ReportGenerator")

    # -----------------------------------------------------------------------
    # 1. Individual stock report
    # -----------------------------------------------------------------------

    async def generate_stock_report(self, ticker: str) -> dict:
        """Run a full analysis on *ticker* and augment with price-history extras.

        Returns a dict containing the complete :class:`StockRecommendation`
        data plus additional fields:

        * ``price_history`` -- 1M, 3M, 6M, 1Y percentage returns
        * ``week_52_high`` / ``week_52_low``
        * ``avg_volume_30d``
        """
        self._log.info("report.stock.start", ticker=ticker)

        # Run full advisor analysis
        recommendation = await self._advisor.analyze(ticker)
        report: dict = recommendation.model_dump()

        # Augment with price-history extras via yfinance
        extras = await self._fetch_price_extras(ticker)
        report.update(extras)

        self._log.info("report.stock.done", ticker=ticker)
        return report

    async def _fetch_price_extras(self, ticker: str) -> dict:
        """Fetch supplementary price data for the stock report."""

        def _get() -> dict:
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(period="1y")
                if hist.empty:
                    return {}

                closes = hist["Close"]
                current = float(closes.iloc[-1])

                def _pct_return(days: int) -> Optional[float]:
                    if len(closes) < days:
                        return None
                    past = float(closes.iloc[-days])
                    if past == 0:
                        return None
                    return round((current - past) / past, 4)

                result: dict = {
                    "price_history": {
                        "return_1m": _pct_return(21),
                        "return_3m": _pct_return(63),
                        "return_6m": _pct_return(126),
                        "return_1y": _pct_return(252) if len(closes) >= 252 else _pct_return(len(closes)),
                    },
                    "week_52_high": float(closes.max()),
                    "week_52_low": float(closes.min()),
                }

                # Average daily volume over last 30 trading days
                if "Volume" in hist.columns and len(hist) >= 30:
                    result["avg_volume_30d"] = int(hist["Volume"].iloc[-30:].mean())
                else:
                    result["avg_volume_30d"] = (
                        int(hist["Volume"].mean()) if "Volume" in hist.columns else 0
                    )

                return result
            except Exception:
                self._log.warning(
                    "report.stock.extras_failed",
                    ticker=ticker,
                    exc_info=True,
                )
                return {}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _get)

    # -----------------------------------------------------------------------
    # 2. Portfolio suggestion
    # -----------------------------------------------------------------------

    async def generate_portfolio_suggestion(
        self,
        universe: list[str],
        capital: float,
        risk_appetite: RiskAppetite,
        horizon: InvestmentHorizon = InvestmentHorizon.MID_TERM,
    ) -> PortfolioSuggestion:
        """Screen *universe* and build a risk-parity portfolio suggestion.

        The allocation strategy varies by *risk_appetite*:

        * **CONSERVATIVE** -- max 30 pct equity, favour ``dividend_steady``,
          large-cap bias, higher cash reserve.
        * **MODERATE** -- max 60 pct equity, blend ``growth_picks`` +
          ``value_dips``, balanced cash.
        * **AGGRESSIVE** -- max 80 pct equity, favour
          ``momentum_breakouts`` + ``growth_picks``, low cash.
        """
        self._log.info(
            "report.portfolio.start",
            universe_size=len(universe),
            capital=capital,
            risk_appetite=risk_appetite.value,
            horizon=horizon.value,
        )

        # --- Select profiles and equity cap based on risk appetite ---------
        profiles, max_equity_pct, cash_pct = self._risk_params(risk_appetite)

        # --- Screen with each profile and merge results --------------------
        all_picks: dict[str, ScreenerResultItem] = {}
        for profile_name in profiles:
            result = await self._screener.screen(
                universe, profile=profile_name,
            )
            for item in result.results:
                existing = all_picks.get(item.ticker)
                if existing is None or item.composite_score > existing.composite_score:
                    all_picks[item.ticker] = item

        # Sort by composite descending and cap the number of holdings
        sorted_picks = sorted(
            all_picks.values(),
            key=lambda p: p.composite_score,
            reverse=True,
        )
        max_holdings = self._max_holdings(risk_appetite)
        top_picks = sorted_picks[:max_holdings]

        if not top_picks:
            self._log.warning("report.portfolio.no_picks")
            return self._empty_portfolio(capital, risk_appetite, horizon, cash_pct=1.0)

        # --- Fetch volatility for risk-parity weighting --------------------
        vol_map = await self._fetch_volatilities([p.ticker for p in top_picks])

        # --- Inverse-volatility weights ------------------------------------
        weights = self._inverse_volatility_weights(top_picks, vol_map)

        # --- Sector diversification check ----------------------------------
        sector_map: dict[str, float] = {}
        for pick, w in zip(top_picks, weights):
            sec = get_sector(pick.ticker)
            sector_map[sec] = sector_map.get(sec, 0.0) + w

        diversification_warnings: list[str] = []
        for sec, w in sector_map.items():
            if w > 0.30:
                diversification_warnings.append(
                    f"Sector '{sec}' has {w:.0%} allocation (>30%)"
                )

        # Diversification score: 100 = perfectly spread, penalise concentration
        if sector_map:
            max_sector_w = max(sector_map.values())
            diversification_score = max(0.0, 100.0 * (1 - max_sector_w))
        else:
            diversification_score = 0.0

        # --- Build allocations ---------------------------------------------
        equity_capital = capital * (1 - cash_pct)
        allocations: list[PortfolioAllocation] = []
        for pick, w in zip(top_picks, weights):
            amount = round(equity_capital * w, 2)
            allocations.append(
                PortfolioAllocation(
                    ticker=pick.ticker,
                    name=pick.name,
                    weight=round(w, 4),
                    amount=amount,
                    action=pick.action,
                    conviction=pick.conviction,
                    sector=get_sector(pick.ticker),
                    rationale="; ".join(pick.key_factors[:3]) if pick.key_factors else "",
                )
            )

        # --- Risk / return estimates ---------------------------------------
        avg_vol = float(np.mean(list(vol_map.values()))) if vol_map else 0.15
        exp_return_low, exp_return_high = self._expected_return_range(
            risk_appetite, avg_vol,
        )
        max_dd = self._estimate_max_drawdown(avg_vol)
        sharpe = self._estimate_sharpe(exp_return_high, avg_vol)

        # --- Reasoning string ----------------------------------------------
        reasoning_parts = [
            f"Selected {len(top_picks)} stocks from {len(universe)} universe "
            f"using {', '.join(profiles)} profiles.",
            f"Equity allocation capped at {(1-cash_pct)*100:.0f}% ({risk_appetite.value}).",
            f"Weights assigned via inverse-volatility risk parity.",
        ]
        if diversification_warnings:
            reasoning_parts.append(
                "Diversification warnings: " + "; ".join(diversification_warnings)
            )

        suggestion = PortfolioSuggestion(
            capital=capital,
            risk_appetite=risk_appetite,
            horizon=horizon,
            allocations=allocations,
            cash_reserve_pct=round(cash_pct, 4),
            expected_annual_return_low=round(exp_return_low, 4),
            expected_annual_return_high=round(exp_return_high, 4),
            expected_max_drawdown=round(max_dd, 4),
            expected_sharpe=round(sharpe, 2),
            sector_breakdown={k: round(v, 4) for k, v in sector_map.items()},
            diversification_score=round(diversification_score, 2),
            reasoning=" ".join(reasoning_parts),
            timestamp=datetime.now(tz=timezone.utc),
        )

        self._log.info(
            "report.portfolio.done",
            holdings=len(allocations),
            cash_pct=cash_pct,
        )
        return suggestion

    # -- portfolio helpers --------------------------------------------------

    @staticmethod
    def _risk_params(
        risk_appetite: RiskAppetite,
    ) -> tuple[list[str], float, float]:
        """Return (profile_names, max_equity_pct, cash_reserve_pct)."""
        if risk_appetite == RiskAppetite.CONSERVATIVE:
            return ["dividend_steady"], 0.30, 0.70
        if risk_appetite == RiskAppetite.MODERATE:
            return ["growth_picks", "value_dips"], 0.60, 0.40
        # AGGRESSIVE
        return ["momentum_breakouts", "growth_picks"], 0.80, 0.20

    @staticmethod
    def _max_holdings(risk_appetite: RiskAppetite) -> int:
        if risk_appetite == RiskAppetite.CONSERVATIVE:
            return 8
        if risk_appetite == RiskAppetite.MODERATE:
            return 12
        return 15

    @staticmethod
    def _inverse_volatility_weights(
        picks: list[ScreenerResultItem],
        vol_map: dict[str, float],
    ) -> list[float]:
        """Compute normalised inverse-volatility weights."""
        inv_vols = []
        for pick in picks:
            vol = vol_map.get(pick.ticker, 0.25)  # default 25% annualised
            inv_vols.append(1.0 / max(vol, 0.01))
        total = sum(inv_vols)
        if total == 0:
            n = len(picks)
            return [1.0 / n] * n
        return [iv / total for iv in inv_vols]

    async def _fetch_volatilities(
        self,
        tickers: list[str],
        period: str = "6mo",
    ) -> dict[str, float]:
        """Fetch annualised volatility for each ticker."""

        def _compute() -> dict[str, float]:
            vols: dict[str, float] = {}
            try:
                data = yf.download(
                    tickers,
                    period=period,
                    progress=False,
                    threads=True,
                )
                if data.empty:
                    return vols

                for t in tickers:
                    try:
                        if len(tickers) == 1:
                            closes = data["Close"]
                        else:
                            closes = data[t]["Close"] if t in data.columns.get_level_values(0) else data["Close"][t]
                        closes = closes.dropna()
                        if len(closes) < 20:
                            continue
                        log_ret = np.log(closes / closes.shift(1)).dropna()
                        vols[t] = float(log_ret.std() * np.sqrt(252))
                    except Exception:
                        pass
            except Exception:
                self._log.warning("report.vol_fetch_failed", exc_info=True)
            return vols

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _compute)

    @staticmethod
    def _expected_return_range(
        risk_appetite: RiskAppetite,
        avg_vol: float,
    ) -> tuple[float, float]:
        """Rough expected annual return range based on risk and volatility."""
        if risk_appetite == RiskAppetite.CONSERVATIVE:
            base = 0.06
        elif risk_appetite == RiskAppetite.MODERATE:
            base = 0.10
        else:
            base = 0.15
        spread = avg_vol * 0.5
        return (base - spread, base + spread)

    @staticmethod
    def _estimate_max_drawdown(avg_vol: float) -> float:
        """Heuristic max drawdown ~ 2x annualised volatility."""
        return min(round(avg_vol * 2.0, 4), 0.60)

    @staticmethod
    def _estimate_sharpe(expected_return: float, avg_vol: float) -> float:
        """Simplified Sharpe = (return - risk_free) / vol."""
        risk_free = 0.05  # approximate
        if avg_vol <= 0:
            return 0.0
        return (expected_return - risk_free) / avg_vol

    def _empty_portfolio(
        self,
        capital: float,
        risk_appetite: RiskAppetite,
        horizon: InvestmentHorizon,
        cash_pct: float = 1.0,
    ) -> PortfolioSuggestion:
        """Return an all-cash portfolio when no picks pass the screener."""
        return PortfolioSuggestion(
            capital=capital,
            risk_appetite=risk_appetite,
            horizon=horizon,
            allocations=[],
            cash_reserve_pct=cash_pct,
            expected_annual_return_low=0.0,
            expected_annual_return_high=0.0,
            expected_max_drawdown=0.0,
            expected_sharpe=0.0,
            sector_breakdown={},
            diversification_score=0.0,
            reasoning="No stocks passed the screening filters. Holding 100% cash.",
            timestamp=datetime.now(tz=timezone.utc),
        )

    # -----------------------------------------------------------------------
    # 3. Market overview
    # -----------------------------------------------------------------------

    async def generate_market_overview(self) -> MarketOverview:
        """Build a broad market snapshot for Indian and global benchmarks."""
        self._log.info("report.market_overview.start")

        # Fetch all market tickers + sectoral indices in parallel
        all_symbols = list(_MARKET_TICKERS.values()) + list(_NIFTY_SECTOR_INDICES.values())
        market_data = await self._fetch_market_data(all_symbols)

        # --- Extract levels and daily % changes ----------------------------
        def _level_and_change(symbol: str) -> tuple[float, float]:
            df = market_data.get(symbol)
            if df is None or df.empty or len(df) < 2:
                return 0.0, 0.0
            closes = df["Close"].dropna()
            if len(closes) < 2:
                return float(closes.iloc[-1]) if len(closes) == 1 else 0.0, 0.0
            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2])
            change = ((last - prev) / prev) if prev != 0 else 0.0
            return last, round(change * 100, 2)

        nifty_level, nifty_chg = _level_and_change(_MARKET_TICKERS["nifty50"])
        sensex_level, sensex_chg = _level_and_change(_MARKET_TICKERS["sensex"])
        sp500_level, sp500_chg = _level_and_change(_MARKET_TICKERS["sp500"])
        nasdaq_level, nasdaq_chg = _level_and_change(_MARKET_TICKERS["nasdaq"])

        india_vix_level, _ = _level_and_change(_MARKET_TICKERS["india_vix"])
        us_vix_level, _ = _level_and_change(_MARKET_TICKERS["us_vix"])
        dxy_level, _ = _level_and_change(_MARKET_TICKERS["dxy"])
        gold_level, _ = _level_and_change(_MARKET_TICKERS["gold"])
        brent_level, _ = _level_and_change(_MARKET_TICKERS["brent"])

        # --- India VIX signal ---------------------------------------------
        india_vix_signal = self._vix_signal(india_vix_level)

        # --- India market regime ------------------------------------------
        nifty_sma200 = await self._nifty_sma200()
        india_regime = self._determine_regime(
            nifty_level, nifty_sma200, india_vix_level,
        )

        # --- Sector performance -------------------------------------------
        sector_perf: dict[str, float] = {}
        for name, symbol in _NIFTY_SECTOR_INDICES.items():
            _, chg = _level_and_change(symbol)
            sector_perf[name] = chg

        # --- Summaries ----------------------------------------------------
        india_summary = self._build_india_summary(
            nifty_level, nifty_chg, sensex_chg, india_vix_level,
            india_vix_signal, india_regime,
        )
        global_summary = self._build_global_summary(
            sp500_level, sp500_chg, nasdaq_chg,
            dxy_level, gold_level, brent_level, us_vix_level,
        )
        risk_outlook = self._risk_outlook(india_vix_level, us_vix_level, india_regime)

        overview = MarketOverview(
            nifty50_level=nifty_level,
            nifty50_change_pct=nifty_chg,
            sensex_level=sensex_level,
            sensex_change_pct=sensex_chg,
            india_vix=india_vix_level,
            india_vix_signal=india_vix_signal,
            fii_net_flow=0.0,  # requires separate data source (not in yfinance)
            dii_net_flow=0.0,
            india_regime=india_regime,
            sp500_level=sp500_level,
            sp500_change_pct=sp500_chg,
            nasdaq_level=nasdaq_level,
            nasdaq_change_pct=nasdaq_chg,
            us_vix=us_vix_level,
            dxy=dxy_level,
            gold_price=gold_level,
            brent_crude=brent_level,
            sector_performance=sector_perf,
            india_summary=india_summary,
            global_summary=global_summary,
            risk_outlook=risk_outlook,
            timestamp=datetime.now(tz=timezone.utc),
        )

        self._log.info("report.market_overview.done")
        return overview

    # -- market overview helpers --------------------------------------------

    async def _fetch_market_data(
        self,
        symbols: list[str],
        period: str = "5d",
    ) -> dict[str, pd.DataFrame]:
        """Batch-download short-period data for market tickers."""

        def _download() -> dict[str, pd.DataFrame]:
            result: dict[str, pd.DataFrame] = {}
            try:
                raw = yf.download(
                    symbols,
                    period=period,
                    group_by="ticker",
                    progress=False,
                    threads=True,
                )
                if raw.empty:
                    return result
                if len(symbols) == 1:
                    result[symbols[0]] = raw
                else:
                    for s in symbols:
                        try:
                            df = raw[s].dropna(how="all")
                            if not df.empty:
                                result[s] = df
                        except (KeyError, AttributeError):
                            pass
            except Exception:
                self._log.warning("report.market_download_failed", exc_info=True)
            return result

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _download)

    async def _nifty_sma200(self) -> Optional[float]:
        """Fetch Nifty 50 closing prices (1y) and return 200-SMA."""

        def _get() -> Optional[float]:
            try:
                hist = yf.Ticker("^NSEI").history(period="1y")
                if hist.empty or len(hist) < 200:
                    return None
                return float(hist["Close"].rolling(200).mean().iloc[-1])
            except Exception:
                return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _get)

    @staticmethod
    def _vix_signal(vix: float) -> str:
        if vix <= 0:
            return "UNKNOWN"
        if vix < _VIX_LOW:
            return "LOW"
        if vix < _VIX_NORMAL:
            return "NORMAL"
        if vix < _VIX_ELEVATED:
            return "ELEVATED"
        return "HIGH"

    @staticmethod
    def _determine_regime(
        nifty_level: float,
        nifty_sma200: Optional[float],
        india_vix: float,
    ) -> str:
        """Simple regime determination based on Nifty vs 200-SMA and VIX."""
        if nifty_sma200 is None:
            return "NEUTRAL"
        above_sma = nifty_level > nifty_sma200
        low_vix = india_vix < _VIX_NORMAL

        if above_sma and low_vix:
            return "BULL"
        if above_sma and not low_vix:
            return "BULL_VOLATILE"
        if not above_sma and low_vix:
            return "BEAR_CALM"
        return "BEAR"

    @staticmethod
    def _build_india_summary(
        nifty_level: float,
        nifty_chg: float,
        sensex_chg: float,
        india_vix: float,
        vix_signal: str,
        regime: str,
    ) -> str:
        direction = "up" if nifty_chg >= 0 else "down"
        return (
            f"Nifty 50 at {nifty_level:,.0f} ({direction} {abs(nifty_chg):.2f}%), "
            f"Sensex {'+' if sensex_chg >= 0 else ''}{sensex_chg:.2f}%. "
            f"India VIX at {india_vix:.1f} ({vix_signal}). "
            f"Market regime: {regime}."
        )

    @staticmethod
    def _build_global_summary(
        sp500_level: float,
        sp500_chg: float,
        nasdaq_chg: float,
        dxy: float,
        gold: float,
        brent: float,
        us_vix: float,
    ) -> str:
        return (
            f"S&P 500 at {sp500_level:,.0f} ({'+' if sp500_chg >= 0 else ''}{sp500_chg:.2f}%), "
            f"Nasdaq {'+' if nasdaq_chg >= 0 else ''}{nasdaq_chg:.2f}%. "
            f"US VIX {us_vix:.1f}. "
            f"DXY {dxy:.2f}, Gold ${gold:,.0f}, Brent ${brent:.1f}."
        )

    @staticmethod
    def _risk_outlook(india_vix: float, us_vix: float, regime: str) -> str:
        """High-level risk outlook string."""
        if india_vix > _VIX_ELEVATED or us_vix > 25:
            return "RISK_OFF"
        if regime in ("BEAR", "BEAR_CALM"):
            return "CAUTIOUS"
        if regime == "BULL_VOLATILE":
            return "MIXED"
        if regime == "BULL" and india_vix < _VIX_LOW:
            return "RISK_ON"
        return "NEUTRAL"
