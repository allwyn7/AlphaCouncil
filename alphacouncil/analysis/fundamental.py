"""Fundamental analysis engine backed by yfinance for Indian equities.

Fetches valuation, profitability, leverage, growth, and ownership data for
NSE-listed stocks and produces a frozen
:class:`~alphacouncil.core.models.FundamentalSignal`.  A simple DCF
intrinsic-value estimate is included.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog
import yfinance as yf

from alphacouncil.core.models import FundamentalSignal

if TYPE_CHECKING:
    from alphacouncil.core.cache import TieredCache

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_NS = "fundamental"
_CACHE_TTL_S = 3600  # 1 hour -- fundamental data is slow-moving

# DCF assumptions (India-specific)
_RISK_FREE_RATE = 0.07       # 10Y G-Sec yield proxy
_EQUITY_RISK_PREMIUM = 0.06  # Indian equity risk premium
_DISCOUNT_RATE = _RISK_FREE_RATE + _EQUITY_RISK_PREMIUM  # 13%
_TERMINAL_GROWTH = 0.03      # long-term nominal GDP growth
_DCF_PROJECTION_YEARS = 10


# ---------------------------------------------------------------------------
# FundamentalEngine
# ---------------------------------------------------------------------------


class FundamentalEngine:
    """Retrieves and analyses fundamental data for NSE-listed companies.

    Parameters
    ----------
    cache:
        A :class:`TieredCache` instance for caching yfinance responses.
    """

    def __init__(self, cache: TieredCache) -> None:
        self._cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, ticker: str) -> FundamentalSignal:
        """Fetch fundamentals via yfinance and return a frozen signal.

        Parameters
        ----------
        ticker:
            NSE ticker symbol **without** the ``.NS`` suffix (e.g.
            ``"RELIANCE"``).  The suffix is appended automatically.
        """
        ns_ticker = ticker if ticker.endswith(".NS") else f"{ticker}.NS"

        cache_key = f"{_CACHE_NS}:{ns_ticker}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.debug("cache_hit", key=cache_key)
            return cached

        # yfinance is synchronous / network-bound -- run off the event loop.
        signal = await asyncio.to_thread(self._fetch_and_build, ns_ticker, ticker)

        await self._cache.set(cache_key, signal, ttl=_CACHE_TTL_S)
        logger.info("fundamental_analysis_complete", ticker=ticker)
        return signal

    def growth_quality_score(self, signal: FundamentalSignal) -> float:
        """Compute a composite growth-quality score in [0, 100].

        The score is a weighted blend of:
        * Revenue growth YoY       (25%)
        * EPS growth YoY           (25%)
        * Return on Equity         (25%)
        * Margin expansion proxy   (25%) -- operating margin as a
          stand-in for margin trajectory.

        Each component is clamped to [0, 100] before weighting so that a
        single extreme value cannot dominate.
        """
        rev_score = _clamp(_normalise_growth(signal.revenue_growth), 0.0, 100.0)
        eps_score = _clamp(_normalise_growth(signal.eps_growth), 0.0, 100.0)

        # ROE: treat 20%+ as top-tier → score 100
        roe_score = _clamp(signal.roe * 100.0 / 0.20, 0.0, 100.0) if signal.roe else 0.0

        # Operating margin as a proxy (30%+ → score 100)
        margin_score = _clamp(
            signal.operating_margin * 100.0 / 0.30, 0.0, 100.0,
        ) if signal.operating_margin else 0.0

        composite = 0.25 * rev_score + 0.25 * eps_score + 0.25 * roe_score + 0.25 * margin_score
        return round(composite, 2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_and_build(self, ns_ticker: str, raw_ticker: str) -> FundamentalSignal:
        """Synchronous: hit yfinance and build the signal model."""
        stock = yf.Ticker(ns_ticker)
        info: dict = stock.info or {}

        pe = _safe_float(info.get("trailingPE"))
        peg = _safe_float(info.get("pegRatio"))
        pb = _safe_float(info.get("priceToBook"))
        roe = _safe_float(info.get("returnOnEquity"))
        roa = _safe_float(info.get("returnOnAssets"))
        de = _safe_float(info.get("debtToEquity"))
        fcf = _safe_float(info.get("freeCashflow"))
        gross_margin = _safe_float(info.get("grossMargins"))
        operating_margin = _safe_float(info.get("operatingMargins"))
        net_margin = _safe_float(info.get("profitMargins"))
        revenue_growth = _safe_float(info.get("revenueGrowth"))
        eps_growth = _safe_float(info.get("earningsGrowth"))

        # ----- India-specific ownership data -----
        # yfinance sometimes exposes major-holder data; try gracefully.
        promoter_holding = self._try_promoter_holding(stock)
        fii_holding = self._try_institutional_holding(stock, kind="fii")
        dii_holding = self._try_institutional_holding(stock, kind="dii")

        # ----- Revenue acceleration (QoQ change in YoY growth) -----
        # Not directly in info; derive from quarterly financials if possible.
        _revenue_accel = self._revenue_acceleration(stock)

        # ----- DCF intrinsic value -----
        intrinsic = self._dcf_intrinsic_value(fcf, revenue_growth)

        now = datetime.now(tz=timezone.utc)

        return FundamentalSignal(
            ticker=raw_ticker,
            pe_ratio=pe,
            peg_ratio=peg,
            pb_ratio=pb,
            roe=roe,
            roa=roa,
            debt_to_equity=de,
            fcf=fcf,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            revenue_growth=revenue_growth,
            eps_growth=eps_growth,
            promoter_holding=promoter_holding,
            fii_holding=fii_holding,
            dii_holding=dii_holding,
            intrinsic_value=intrinsic,
            timestamp=now,
        )

    # ----- Ownership helpers --------------------------------------------------

    @staticmethod
    def _try_promoter_holding(stock: yf.Ticker) -> float:
        """Attempt to extract promoter holding % from yfinance major_holders."""
        try:
            holders = stock.major_holders
            if holders is not None and not holders.empty:
                for _, row in holders.iterrows():
                    label = str(row.iloc[-1]).lower()
                    if "insider" in label or "promoter" in label:
                        return float(str(row.iloc[0]).replace("%", ""))
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    @staticmethod
    def _try_institutional_holding(stock: yf.Ticker, *, kind: str) -> float:
        """Attempt to extract FII / DII holding % from yfinance.

        ``kind`` should be ``"fii"`` or ``"dii"``.
        """
        try:
            holders = stock.institutional_holders
            if holders is not None and not holders.empty:
                # yfinance does not cleanly separate FII vs DII; return
                # aggregate institutional as a proxy until a dedicated
                # data source (e.g. NSE bulk deals) is integrated.
                mh = stock.major_holders
                if mh is not None and not mh.empty:
                    for _, row in mh.iterrows():
                        label = str(row.iloc[-1]).lower()
                        if kind == "fii" and "institution" in label:
                            return float(str(row.iloc[0]).replace("%", ""))
                        if kind == "dii" and ("mutual" in label or "domestic" in label):
                            return float(str(row.iloc[0]).replace("%", ""))
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    @staticmethod
    def _revenue_acceleration(stock: yf.Ticker) -> float | None:
        """Compute QoQ change in YoY revenue growth from quarterly financials.

        Returns ``None`` when insufficient data is available.
        """
        try:
            qtr = stock.quarterly_financials
            if qtr is None or qtr.empty:
                return None

            # yfinance quarterly_financials: columns = quarter dates, rows = line items
            if "Total Revenue" not in qtr.index:
                return None

            rev = qtr.loc["Total Revenue"].dropna().sort_index()
            if len(rev) < 5:
                return None

            # YoY growth for the last two quarters (current vs same quarter YoY)
            yoy_current = (rev.iloc[-1] - rev.iloc[-4]) / abs(rev.iloc[-4]) if rev.iloc[-4] != 0 else 0.0
            yoy_prior = (rev.iloc[-2] - rev.iloc[-5]) / abs(rev.iloc[-5]) if rev.iloc[-5] != 0 else 0.0

            return float(yoy_current - yoy_prior)
        except Exception:  # noqa: BLE001
            return None

    # ----- DCF valuation ------------------------------------------------------

    @staticmethod
    def _dcf_intrinsic_value(fcf: float, revenue_growth: float | None) -> float:
        """Simple DCF intrinsic value estimate.

        Uses projected free cash flow grown at a rate derived from
        ``revenue_growth``, discounted at the Indian-market discount rate
        (risk-free + equity risk premium = 13%).  Terminal value assumes
        3% perpetuity growth.

        Returns 0.0 when FCF data is unavailable or negative (a DCF is
        meaningless for cash-burning companies).
        """
        if not fcf or fcf <= 0:
            return 0.0

        growth = float(revenue_growth) if revenue_growth and revenue_growth > 0 else 0.05

        # Cap growth at a reasonable ceiling for stability
        growth = min(growth, 0.30)

        pv_sum = 0.0
        projected_fcf = fcf
        for year in range(1, _DCF_PROJECTION_YEARS + 1):
            projected_fcf *= (1 + growth)
            pv_sum += projected_fcf / ((1 + _DISCOUNT_RATE) ** year)

        # Terminal value at the end of projection period
        terminal_fcf = projected_fcf * (1 + _TERMINAL_GROWTH)
        terminal_value = terminal_fcf / (_DISCOUNT_RATE - _TERMINAL_GROWTH)
        pv_terminal = terminal_value / ((1 + _DISCOUNT_RATE) ** _DCF_PROJECTION_YEARS)

        intrinsic = pv_sum + pv_terminal
        return round(intrinsic, 2)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _safe_float(val: object) -> float:
    """Convert a possibly-None / non-numeric value to float, defaulting to 0."""
    if val is None:
        return 0.0
    try:
        result = float(val)
        # Guard against inf / nan from yfinance quirks
        if result != result or result in (float("inf"), float("-inf")):
            return 0.0
        return result
    except (ValueError, TypeError):
        return 0.0


def _normalise_growth(rate: float) -> float:
    """Map a growth rate (e.g. 0.25 = 25%) to a 0-100 score.

    Linear mapping: 0% -> 0, 50%+ -> 100.
    """
    if rate <= 0:
        return 0.0
    return min(rate / 0.50, 1.0) * 100.0


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the interval [lo, hi]."""
    return max(lo, min(hi, value))
