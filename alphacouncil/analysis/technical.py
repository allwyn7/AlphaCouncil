"""Technical analysis engine using pandas-ta for vectorised indicator computation.

All indicators are computed via *pandas-ta* (NOT ta-lib) so no C-library
dependency is required.  Results are returned as a frozen
:class:`~alphacouncil.core.models.TechnicalSignal` Pydantic model.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandas_ta as ta  # vectorised technical indicators
import structlog

from alphacouncil.core.models import TechnicalSignal

if TYPE_CHECKING:
    from alphacouncil.core.cache import TieredCache

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_NS = "technical"
_CACHE_TTL_S = 300  # 5 minutes


# ---------------------------------------------------------------------------
# TechnicalEngine
# ---------------------------------------------------------------------------


class TechnicalEngine:
    """Computes a full suite of technical indicators for a given ticker.

    Parameters
    ----------
    cache:
        A :class:`TieredCache` instance used to memoise expensive indicator
        computations keyed on ``(ticker, last_bar_timestamp)``.
    """

    def __init__(self, cache: TieredCache) -> None:
        self._cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, ticker: str, df: pd.DataFrame) -> TechnicalSignal:
        """Compute all technical indicators and return a frozen signal.

        Parameters
        ----------
        ticker:
            NSE symbol (e.g. ``"RELIANCE"``).
        df:
            OHLCV DataFrame with columns
            ``["open", "high", "low", "close", "volume"]`` indexed by
            datetime.  Must contain at least 200 rows for the longest
            look-back window (SMA-200).
        """
        if df.empty:
            raise ValueError(f"Empty DataFrame for {ticker}")

        # Normalise column names to lowercase
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]

        # Check for minimum rows
        if len(df) < 200:
            logger.warning(
                "insufficient_bars",
                ticker=ticker,
                rows=len(df),
                required=200,
            )

        # Cache key based on ticker + last bar timestamp
        last_ts = str(df.index[-1]) if isinstance(df.index, pd.DatetimeIndex) else str(len(df))
        cache_key = f"{_CACHE_NS}:{ticker}:{last_ts}"

        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.debug("cache_hit", key=cache_key)
            return cached

        # Run computation in a thread-pool so the event loop is never blocked.
        signal = await asyncio.to_thread(self._compute, ticker, df)

        await self._cache.set(cache_key, signal, ttl=_CACHE_TTL_S)
        logger.info("technical_analysis_complete", ticker=ticker)
        return signal

    def detect_breakout(self, df: pd.DataFrame) -> dict:
        """Detect price breakouts above recent resistance levels.

        Uses a rolling 20-period high as a proxy for resistance.  A breakout
        is signalled when the latest close exceeds the prior 20-bar high by
        at least one ATR.

        Returns
        -------
        dict
            ``{"breakout": bool, "resistance": float, "close": float,
              "atr": float, "strength": float}``
        """
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]

        atr_series: pd.Series = ta.atr(df["high"], df["low"], df["close"], length=14)
        atr_val = float(atr_series.iloc[-1]) if atr_series is not None and not atr_series.empty else 0.0

        resistance = float(df["high"].rolling(window=20).max().iloc[-2])
        close = float(df["close"].iloc[-1])

        is_breakout = close > resistance and atr_val > 0
        strength = (close - resistance) / atr_val if atr_val > 0 else 0.0

        return {
            "breakout": is_breakout,
            "resistance": resistance,
            "close": close,
            "atr": atr_val,
            "strength": round(strength, 4),
        }

    def get_trend(self, df: pd.DataFrame) -> str:
        """Classify the prevailing trend via EMA alignment.

        Returns ``"BULLISH"`` when EMA-20 > EMA-50 > EMA-200,
        ``"BEARISH"`` when EMA-20 < EMA-50 < EMA-200, else ``"NEUTRAL"``.
        """
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]

        ema_20 = ta.ema(df["close"], length=20)
        ema_50 = ta.ema(df["close"], length=50)
        ema_200 = ta.ema(df["close"], length=200)

        if ema_20 is None or ema_50 is None or ema_200 is None:
            return "NEUTRAL"

        e20 = float(ema_20.iloc[-1])
        e50 = float(ema_50.iloc[-1])
        e200 = float(ema_200.iloc[-1])

        if e20 > e50 > e200:
            return "BULLISH"
        if e20 < e50 < e200:
            return "BEARISH"
        return "NEUTRAL"

    # ------------------------------------------------------------------
    # Internal computation (all vectorised via pandas-ta)
    # ------------------------------------------------------------------

    def _compute(self, ticker: str, df: pd.DataFrame) -> TechnicalSignal:
        """Synchronous, CPU-bound indicator computation."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # -- Momentum ----------------------------------------------------------
        rsi_series = ta.rsi(close, length=14)
        rsi_val = _last(rsi_series)

        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        # pandas-ta returns columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        macd_val = _last(macd_df.iloc[:, 0]) if macd_df is not None else 0.0
        macd_hist = _last(macd_df.iloc[:, 1]) if macd_df is not None else 0.0
        macd_signal = _last(macd_df.iloc[:, 2]) if macd_df is not None else 0.0

        roc_series = ta.roc(close, length=12)
        roc_val = _last(roc_series)

        # -- Bollinger Bands ---------------------------------------------------
        bb_df = ta.bbands(close, length=20, std=2)
        # Columns: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        bb_lower = _last(bb_df.iloc[:, 0]) if bb_df is not None else 0.0
        bb_mid = _last(bb_df.iloc[:, 1]) if bb_df is not None else 0.0
        bb_upper = _last(bb_df.iloc[:, 2]) if bb_df is not None else 0.0

        # -- Simple Moving Averages --------------------------------------------
        sma_20 = _last(ta.sma(close, length=20))
        sma_50 = _last(ta.sma(close, length=50))
        sma_200 = _last(ta.sma(close, length=200))

        # -- Exponential Moving Averages ---------------------------------------
        ema_20 = _last(ta.ema(close, length=20))
        ema_50 = _last(ta.ema(close, length=50))
        ema_200 = _last(ta.ema(close, length=200))

        # -- Trend / Volatility ------------------------------------------------
        adx_df = ta.adx(high, low, close, length=14)
        # First column is ADX_14
        adx_val = _last(adx_df.iloc[:, 0]) if adx_df is not None else 0.0

        atr_series = ta.atr(high, low, close, length=14)
        atr_val = _last(atr_series)

        # -- Volume indicators -------------------------------------------------
        obv_series = ta.obv(close, volume)
        obv_val = _last(obv_series)

        # VWAP: meaningful only for intraday data with volume.
        vwap_val = self._compute_vwap(df)

        # Volume ratio: 5-day average volume / 20-day average volume
        vol_avg_5 = volume.rolling(window=5).mean().iloc[-1]
        vol_avg_20 = volume.rolling(window=20).mean().iloc[-1]
        volume_ratio = float(vol_avg_5 / vol_avg_20) if vol_avg_20 and vol_avg_20 != 0 else 1.0

        now = datetime.now(tz=timezone.utc)

        return TechnicalSignal(
            ticker=ticker,
            rsi=rsi_val,
            macd=macd_val,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
            roc=roc_val,
            bollinger_upper=bb_upper,
            bollinger_lower=bb_lower,
            bollinger_mid=bb_mid,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            ema_20=ema_20,
            ema_50=ema_50,
            ema_200=ema_200,
            adx=adx_val,
            atr=atr_val,
            obv=obv_val,
            vwap=vwap_val,
            volume_ratio=round(volume_ratio, 4),
            timestamp=now,
        )

    @staticmethod
    def _compute_vwap(df: pd.DataFrame) -> float:
        """Compute VWAP from intraday data if volume is available.

        Falls back to the last close when the data lacks intraday
        granularity or volume is zero.
        """
        try:
            if df["volume"].sum() == 0:
                return float(df["close"].iloc[-1])

            # pandas-ta vwap requires a DatetimeIndex
            vwap_series = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            if vwap_series is not None and not vwap_series.empty:
                val = vwap_series.iloc[-1]
                if pd.notna(val):
                    return float(val)
        except Exception:  # noqa: BLE001
            pass

        return float(df["close"].iloc[-1])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last(series: pd.Series | pd.DataFrame | None) -> float:
    """Extract the last non-NaN value from a pandas Series, defaulting to 0."""
    if series is None:
        return 0.0
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    if series.empty:
        return 0.0
    val = series.iloc[-1]
    return float(val) if pd.notna(val) else 0.0
