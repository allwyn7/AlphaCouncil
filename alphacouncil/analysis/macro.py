"""Macro-economic analysis engine for Indian equity market regime detection.

Combines FRED macro data, yfinance market indicators, and calendar heuristics
to classify the current market regime into a
:class:`~alphacouncil.core.models.MarketRegime` enum value.  The result is
returned as a frozen :class:`~alphacouncil.core.models.MacroSignal`.
"""

from __future__ import annotations

import asyncio
import calendar
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pandas as pd
import structlog
import yfinance as yf
from fredapi import Fred

from alphacouncil.core.models import MacroSignal, MarketRegime

if TYPE_CHECKING:
    from alphacouncil.core.cache import TieredCache

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_NS = "macro"
_CACHE_TTL_S = 1800  # 30 minutes

# FRED series identifiers
_FRED_INDIA_RATE = "INDIRLTLT01STM"    # India long-term interest rate (proxy for repo rate)
_FRED_INDIA_CPI = "INDCPIALLMINMEI"    # India CPI All Items
_FRED_US_FED_RATE = "FEDFUNDS"          # US Federal Funds Rate

# yfinance tickers for market indicators
_YF_DXY = "DX-Y.NYB"
_YF_BRENT = "BZ=F"
_YF_INDIA_VIX = "^INDIAVIX"
_YF_GOLD = "GOLDBEES.NS"
_YF_NIFTY = "^NSEI"

# India VIX thresholds for volatility regime
_VIX_HIGH_THRESHOLD = 20.0
_VIX_LOW_THRESHOLD = 14.0

# Months considered earnings season in India
# (Jan-Feb: Q3 results, Jul-Aug: Q1 results, Oct-Nov: Q2 results)
_EARNINGS_MONTHS = {1, 2, 7, 8, 10, 11}


# ---------------------------------------------------------------------------
# MacroEngine
# ---------------------------------------------------------------------------


class MacroEngine:
    """Fetches macro-economic data and classifies market regime.

    Parameters
    ----------
    cache:
        A :class:`TieredCache` instance for caching macro snapshots.
    fred_api_key:
        API key for the FRED (Federal Reserve Economic Data) service.
    """

    def __init__(self, cache: TieredCache, fred_api_key: str) -> None:
        self._cache = cache
        self._fred = Fred(api_key=fred_api_key)

    # ------------------------------------------------------------------
    # India macro (FRED-backed)
    # ------------------------------------------------------------------

    async def fetch_india_macro(self) -> dict:
        """Fetch Indian macro indicators from FRED.

        Returns
        -------
        dict
            ``{"repo_rate": float, "india_cpi": float, "india_iip": float}``
        """
        cache_key = f"{_CACHE_NS}:india_macro"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = await asyncio.to_thread(self._fetch_india_macro_sync)
        await self._cache.set(cache_key, result, ttl=_CACHE_TTL_S)
        return result

    def _fetch_india_macro_sync(self) -> dict:
        """Synchronous FRED fetch for India macro series."""
        repo_rate = self._latest_fred_value(_FRED_INDIA_RATE)
        india_cpi = self._latest_fred_value(_FRED_INDIA_CPI)
        # India IIP is not directly on FRED with a stable series; placeholder.
        india_iip = 0.0

        logger.debug(
            "india_macro_fetched",
            repo_rate=repo_rate,
            cpi=india_cpi,
            iip=india_iip,
        )
        return {
            "repo_rate": repo_rate,
            "india_cpi": india_cpi,
            "india_iip": india_iip,
        }

    # ------------------------------------------------------------------
    # Global macro (FRED + yfinance)
    # ------------------------------------------------------------------

    async def fetch_global_macro(self) -> dict:
        """Fetch global macro indicators.

        Returns
        -------
        dict
            ``{"fed_rate": float, "dxy": float, "brent_crude": float}``
        """
        cache_key = f"{_CACHE_NS}:global_macro"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        fed_rate_task = asyncio.to_thread(self._latest_fred_value, _FRED_US_FED_RATE)
        dxy_task = asyncio.to_thread(_yf_last_close, _YF_DXY)
        brent_task = asyncio.to_thread(_yf_last_close, _YF_BRENT)

        fed_rate, dxy, brent = await asyncio.gather(
            fed_rate_task, dxy_task, brent_task,
        )

        result = {
            "fed_rate": fed_rate,
            "dxy": dxy,
            "brent_crude": brent,
        }

        await self._cache.set(cache_key, result, ttl=_CACHE_TTL_S)
        logger.debug("global_macro_fetched", **result)
        return result

    # ------------------------------------------------------------------
    # Market indicators (yfinance)
    # ------------------------------------------------------------------

    async def fetch_market_indicators(self) -> dict:
        """Fetch India-specific market indicators.

        Returns
        -------
        dict
            ``{"india_vix": float, "gold_price": float, "nifty_level": float}``
        """
        cache_key = f"{_CACHE_NS}:market_indicators"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        vix_task = asyncio.to_thread(_yf_last_close, _YF_INDIA_VIX)
        gold_task = asyncio.to_thread(_yf_last_close, _YF_GOLD)
        nifty_task = asyncio.to_thread(_yf_last_close, _YF_NIFTY)

        vix, gold, nifty = await asyncio.gather(vix_task, gold_task, nifty_task)

        result = {
            "india_vix": vix,
            "gold_price": gold,
            "nifty_level": nifty,
        }

        await self._cache.set(cache_key, result, ttl=_CACHE_TTL_S)
        logger.debug("market_indicators_fetched", **result)
        return result

    # ------------------------------------------------------------------
    # FII / DII flow data
    # ------------------------------------------------------------------

    async def fetch_fii_dii_flows(self) -> dict:
        """Fetch FII and DII net flow data for the day.

        Attempts to use ``jugaad_data`` or ``nsetools`` if available; falls
        back to a zero-valued placeholder when neither is installed.

        Returns
        -------
        dict
            ``{"fii_net_flow": float, "dii_net_flow": float}``
        """
        cache_key = f"{_CACHE_NS}:fii_dii_flows"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = await asyncio.to_thread(self._fetch_fii_dii_sync)
        await self._cache.set(cache_key, result, ttl=_CACHE_TTL_S)
        return result

    @staticmethod
    def _fetch_fii_dii_sync() -> dict:
        """Synchronous attempt to fetch FII/DII data."""
        # --- Attempt 1: jugaad-data ---
        try:
            from jugaad_data.nse import NSELive  # type: ignore[import-untyped]

            nse = NSELive()
            fii_dii = nse.fii_dii_data()
            if fii_dii and isinstance(fii_dii, list) and len(fii_dii) >= 2:
                fii_net = float(fii_dii[0].get("netValue", 0))
                dii_net = float(fii_dii[1].get("netValue", 0))
                return {"fii_net_flow": fii_net, "dii_net_flow": dii_net}
        except Exception:  # noqa: BLE001
            pass

        # --- Attempt 2: nsetools ---
        try:
            from nsetools import Nse  # type: ignore[import-untyped]

            nse = Nse()
            # nsetools does not directly expose FII/DII; skip.
        except Exception:  # noqa: BLE001
            pass

        # --- Fallback: placeholder ---
        logger.warning("fii_dii_data_unavailable", fallback="zeros")
        return {"fii_net_flow": 0.0, "dii_net_flow": 0.0}

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def determine_regime(self, macro_data: dict) -> MarketRegime:
        """Classify the current market regime.

        Decision hierarchy (first match wins after calendar overrides):

        1. **Calendar overrides** -- ``PRE_EXPIRY``, ``EARNINGS_SEASON``,
           ``BUDGET_POLICY`` take precedence when their conditions hold.
        2. **FII flow** -- extreme net buying/selling dominates.
        3. **Trend + volatility** -- combines Nifty 50 vs 200 SMA direction
           with India VIX level.

        Parameters
        ----------
        macro_data:
            A merged dict containing at least the keys produced by
            :meth:`fetch_india_macro`, :meth:`fetch_global_macro`,
            :meth:`fetch_market_indicators`, and :meth:`fetch_fii_dii_flows`.
        """
        today = date.today()

        # --- Calendar-based overrides ---
        if self._is_pre_expiry(today):
            return MarketRegime.PRE_EXPIRY

        if self._is_budget_policy_week(today):
            return MarketRegime.BUDGET_POLICY

        if today.month in _EARNINGS_MONTHS:
            return MarketRegime.EARNINGS_SEASON

        # --- FII flow-based ---
        fii_net = macro_data.get("fii_net_flow", 0.0)
        if fii_net > 1000:  # net buying > INR 1000 crore
            return MarketRegime.FII_BUYING
        if fii_net < -1000:
            return MarketRegime.FII_SELLING

        # --- Trend + volatility ---
        nifty = macro_data.get("nifty_level", 0.0)
        nifty_sma200 = macro_data.get("nifty_sma200", None)
        india_vix = macro_data.get("india_vix", 15.0)

        # Determine bull/bear from Nifty vs 200 SMA
        is_bull = True
        if nifty_sma200 is not None and nifty_sma200 > 0:
            is_bull = nifty > nifty_sma200

        # Determine vol regime from India VIX
        is_high_vol = india_vix >= _VIX_HIGH_THRESHOLD

        if is_bull and is_high_vol:
            return MarketRegime.BULL_HIGH_VOL
        if is_bull and not is_high_vol:
            return MarketRegime.BULL_LOW_VOL
        if not is_bull and is_high_vol:
            return MarketRegime.BEAR_HIGH_VOL
        if not is_bull and not is_high_vol:
            return MarketRegime.BEAR_LOW_VOL

        return MarketRegime.SIDEWAYS

    # ------------------------------------------------------------------
    # Main signal builder
    # ------------------------------------------------------------------

    async def get_signal(self) -> MacroSignal:
        """Fetch all macro data and return a frozen :class:`MacroSignal`."""
        cache_key = f"{_CACHE_NS}:signal"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.debug("cache_hit", key=cache_key)
            return cached

        # Fetch all data sources concurrently
        india_task = self.fetch_india_macro()
        global_task = self.fetch_global_macro()
        market_task = self.fetch_market_indicators()
        flow_task = self.fetch_fii_dii_flows()

        india, global_m, market, flows = await asyncio.gather(
            india_task, global_task, market_task, flow_task,
        )

        # Compute Nifty 200 SMA for regime classification
        nifty_sma200 = await asyncio.to_thread(_nifty_sma200)

        # Merge all data
        merged: dict = {**india, **global_m, **market, **flows, "nifty_sma200": nifty_sma200}

        regime = self.determine_regime(merged)

        signal = MacroSignal(
            repo_rate=india.get("repo_rate", 0.0),
            india_cpi=india.get("india_cpi", 0.0),
            india_iip=india.get("india_iip", 0.0),
            fed_rate=global_m.get("fed_rate", 0.0),
            dxy=global_m.get("dxy", 0.0),
            brent_crude=global_m.get("brent_crude", 0.0),
            india_vix=market.get("india_vix", 0.0),
            gold_price=market.get("gold_price", 0.0),
            nifty_level=market.get("nifty_level", 0.0),
            fii_net_flow=flows.get("fii_net_flow", 0.0),
            dii_net_flow=flows.get("dii_net_flow", 0.0),
            regime=regime,
            timestamp=datetime.now(tz=timezone.utc),
        )

        await self._cache.set(cache_key, signal, ttl=_CACHE_TTL_S)
        logger.info("macro_signal_complete", regime=regime.value)
        return signal

    # ------------------------------------------------------------------
    # FRED helper
    # ------------------------------------------------------------------

    def _latest_fred_value(self, series_id: str) -> float:
        """Fetch the most recent observation for a FRED series.

        Returns 0.0 on any failure so the caller always gets a numeric
        result.
        """
        try:
            data: pd.Series = self._fred.get_series(series_id)
            if data is not None and not data.empty:
                val = data.dropna().iloc[-1]
                return float(val)
        except Exception:  # noqa: BLE001
            logger.warning("fred_fetch_failed", series=series_id)
        return 0.0

    # ------------------------------------------------------------------
    # Calendar heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _is_pre_expiry(today: date) -> bool:
        """Return True if *today* is within 3 calendar days of the month's
        last Thursday (F&O expiry day on NSE).
        """
        year, month = today.year, today.month
        # Find the last Thursday of the month
        last_day = calendar.monthrange(year, month)[1]
        dt = date(year, month, last_day)
        while dt.weekday() != 3:  # Thursday = 3
            dt -= timedelta(days=1)
        # Within 3 calendar days (inclusive) before expiry
        return 0 <= (dt - today).days <= 3

    @staticmethod
    def _is_budget_policy_week(today: date) -> bool:
        """Return True if *today* falls in the Union Budget week or an
        RBI monetary policy week.

        Heuristics:
        * Budget is typically presented on Feb 1.  We flag Jan 29 -- Feb 3.
        * RBI policy is announced ~6 times/year (Feb, Apr, Jun, Aug, Oct,
          Dec), usually on the first Friday.  We flag the surrounding Mon-Fri
          of the first week.
        """
        month, day = today.month, today.day

        # Budget week: last few days of January through early February
        if (month == 1 and day >= 29) or (month == 2 and day <= 3):
            return True

        # RBI policy months: flag the first 7 days
        rbi_months = {2, 4, 6, 8, 10, 12}
        if month in rbi_months and day <= 7:
            return True

        return False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _yf_last_close(ticker: str) -> float:
    """Return the last closing price for a yfinance ticker.

    Falls back to 0.0 on any error.
    """
    try:
        data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if data is not None and not data.empty:
            close_col = "Close"
            if close_col in data.columns:
                return float(data[close_col].dropna().iloc[-1])
    except Exception:  # noqa: BLE001
        logger.warning("yf_fetch_failed", ticker=ticker)
    return 0.0


def _nifty_sma200() -> float:
    """Compute the Nifty 50 200-day SMA from yfinance historical data.

    Returns 0.0 when data is insufficient.
    """
    try:
        data = yf.download(_YF_NIFTY, period="1y", progress=False, auto_adjust=True)
        if data is not None and len(data) >= 200:
            close_col = "Close"
            if close_col in data.columns:
                sma = data[close_col].rolling(window=200).mean()
                val = sma.dropna().iloc[-1]
                return float(val)
    except Exception:  # noqa: BLE001
        logger.warning("nifty_sma200_failed")
    return 0.0
