"""Price prediction engine — ensemble of statistical models.

Uses three complementary approaches (no new dependencies):
1. Linear regression on technical features (sklearn)
2. Exponential smoothing with mean-reversion pull
3. Technical level projection (support/resistance)

All models are combined into an ensemble average with volatility-based
confidence bounds.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from alphacouncil.advisor.models import PredictionPoint, PricePrediction

logger = structlog.get_logger(__name__)

# Default horizons in trading days
DEFAULT_HORIZONS: list[int] = [7, 14, 30, 60, 90]


class PricePredictor:
    """Ensemble price predictor combining regression, smoothing, and technicals."""

    def __init__(self, cache: Optional[object] = None) -> None:
        self._cache = cache

    async def predict(
        self,
        ticker: str,
        horizon_days: list[int] | None = None,
        df: pd.DataFrame | None = None,
    ) -> PricePrediction:
        """Generate price predictions for multiple horizons.

        Parameters
        ----------
        ticker:
            Stock ticker symbol.
        horizon_days:
            List of days-ahead to predict. Defaults to [7, 14, 30, 60, 90].
        df:
            Optional pre-fetched OHLCV DataFrame. If None, fetches via yfinance.
        """
        import asyncio

        horizons = horizon_days or DEFAULT_HORIZONS

        if df is None:
            df = await asyncio.to_thread(self._fetch_data, ticker)

        if df is None or df.empty or len(df) < 60:
            raise ValueError(f"Insufficient data for prediction: {ticker}")

        # Normalize columns
        df = df.copy()
        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        close = df["Close"].dropna()
        current_price = float(close.iloc[-1])

        # Run 3 models
        lr_preds = self._linear_regression_model(df, horizons)
        ewm_preds = self._exponential_smoothing_model(close, horizons)
        tech_preds = self._technical_projection_model(df, horizons)

        # Ensemble: average the three models
        predictions: list[PredictionPoint] = []
        all_model_prices: list[list[float]] = []

        daily_vol = float(close.pct_change().dropna().std())
        annual_vol = daily_vol * math.sqrt(252)

        for i, days in enumerate(horizons):
            prices = []
            if lr_preds and i < len(lr_preds):
                prices.append(lr_preds[i])
            if ewm_preds and i < len(ewm_preds):
                prices.append(ewm_preds[i])
            if tech_preds and i < len(tech_preds):
                prices.append(tech_preds[i])

            if not prices:
                prices = [current_price]

            all_model_prices.append(prices)
            pred_price = float(np.mean(prices))

            # Confidence bounds: historical vol scaled to horizon
            vol_spread = daily_vol * math.sqrt(days) * current_price * 1.5
            low = pred_price - vol_spread
            high = pred_price + vol_spread

            change_pct = ((pred_price - current_price) / current_price) * 100

            predictions.append(PredictionPoint(
                days_ahead=days,
                predicted_price=round(pred_price, 2),
                low_bound=round(max(low, current_price * 0.5), 2),
                high_bound=round(high, 2),
                change_pct=round(change_pct, 2),
            ))

        # Model confidence: based on agreement between the 3 models
        confidence = self._compute_confidence(all_model_prices, current_price)

        return PricePrediction(
            ticker=ticker,
            current_price=round(current_price, 2),
            predictions=predictions,
            model_confidence=round(confidence, 2),
            timestamp=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Model 1: Linear regression on technical features
    # ------------------------------------------------------------------

    def _linear_regression_model(
        self, df: pd.DataFrame, horizons: list[int],
    ) -> list[float]:
        """Fit linear regression on log-returns with technical features."""
        try:
            from sklearn.linear_model import LinearRegression

            close = df["Close"].astype(float)
            log_returns = np.log(close / close.shift(1)).dropna()

            if len(log_returns) < 60:
                return []

            # Build feature matrix
            features = pd.DataFrame(index=log_returns.index)
            features["day_of_week"] = features.index.dayofweek if hasattr(features.index, "dayofweek") else 0
            features["month"] = features.index.month if hasattr(features.index, "month") else 1
            features["ret_20d"] = log_returns.rolling(20).mean()
            features["ret_50d"] = log_returns.rolling(50).mean()

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / loss.replace(0, float("nan"))
            features["rsi"] = (100 - (100 / (1 + rs))).reindex(features.index)

            # MACD histogram
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
            features["macd_hist"] = macd_hist.reindex(features.index)

            # Bollinger position
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            features["bb_pos"] = ((close - sma20) / std20.replace(0, float("nan"))).reindex(features.index)

            # Drop NaN rows
            features = features.dropna()
            aligned_returns = log_returns.reindex(features.index).dropna()
            features = features.loc[aligned_returns.index]

            if len(features) < 30:
                return []

            X = features.values
            y = aligned_returns.values

            model = LinearRegression()
            model.fit(X, y)

            # Project forward: use last row's features with decaying returns
            last_features = features.iloc[-1].values.reshape(1, -1)
            daily_pred = float(model.predict(last_features)[0])

            current_price = float(close.iloc[-1])
            predictions = []
            for days in horizons:
                # Compound the daily prediction, decaying toward 0
                decay = math.exp(-days / 120)  # decay half-life ~83 days
                cum_return = daily_pred * days * decay
                pred_price = current_price * math.exp(cum_return)
                predictions.append(pred_price)

            return predictions

        except Exception as e:
            logger.debug("lr_model_failed", error=str(e))
            return []

    # ------------------------------------------------------------------
    # Model 2: Exponential smoothing with mean-reversion
    # ------------------------------------------------------------------

    def _exponential_smoothing_model(
        self, close: pd.Series, horizons: list[int],
    ) -> list[float]:
        """EWM-based projection with mean-reversion pull toward SMA200."""
        try:
            current_price = float(close.iloc[-1])
            returns = close.pct_change().dropna()

            if len(returns) < 60:
                return []

            # EWM trend from last 60 days
            ewm_mean = float(returns.tail(60).ewm(span=20).mean().iloc[-1])

            # SMA200 as fair value anchor
            sma200 = float(close.rolling(min(200, len(close))).mean().iloc[-1])

            # Mean-reversion strength: how far from SMA200
            std_price = float(close.tail(200).std()) if len(close) >= 50 else current_price * 0.1
            distance = (current_price - sma200) / max(std_price, 0.01)

            predictions = []
            for days in horizons:
                # Base: compound EWM trend forward, decaying toward 0
                decay = math.exp(-days / 90)
                trend_component = current_price * (1 + ewm_mean * days * decay)

                # Mean-reversion pull: stronger when far from SMA200
                reversion_strength = min(abs(distance) * 0.1, 0.5) * (days / 60)
                if current_price > sma200:
                    reversion = -reversion_strength * (current_price - sma200)
                else:
                    reversion = reversion_strength * (sma200 - current_price)

                pred_price = trend_component + reversion
                predictions.append(max(pred_price, current_price * 0.3))

            return predictions

        except Exception as e:
            logger.debug("ewm_model_failed", error=str(e))
            return []

    # ------------------------------------------------------------------
    # Model 3: Technical level projection
    # ------------------------------------------------------------------

    def _technical_projection_model(
        self, df: pd.DataFrame, horizons: list[int],
    ) -> list[float]:
        """Project price toward support/resistance using trend direction."""
        try:
            close = df["Close"].astype(float)
            high = df["High"].astype(float)
            low = df["Low"].astype(float)
            current_price = float(close.iloc[-1])

            # SMA trend
            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(min(200, len(close))).mean().iloc[-1])
            uptrend = sma50 > sma200

            # Swing highs/lows for support/resistance (last 60 bars)
            recent = df.tail(60)
            resistance = float(recent["High"].max())
            support = float(recent["Low"].min())

            # ATR for volatility scaling
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])

            predictions = []
            for days in horizons:
                if uptrend:
                    # Project toward resistance, proportional to time
                    progress = min(days / 60, 1.0)
                    target = current_price + (resistance - current_price) * progress
                    # Add ATR-based drift
                    target += atr * 0.05 * days
                else:
                    # Project toward support
                    progress = min(days / 60, 1.0)
                    target = current_price - (current_price - support) * progress * 0.5
                    target -= atr * 0.03 * days

                predictions.append(max(target, current_price * 0.3))

            return predictions

        except Exception as e:
            logger.debug("tech_projection_failed", error=str(e))
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_confidence(
        self, all_model_prices: list[list[float]], current_price: float,
    ) -> float:
        """Compute model confidence based on inter-model agreement."""
        if not all_model_prices or current_price <= 0:
            return 0.3

        dispersions = []
        for prices in all_model_prices:
            if len(prices) >= 2:
                std = float(np.std(prices))
                dispersions.append(std / current_price)

        if not dispersions:
            return 0.5

        avg_dispersion = float(np.mean(dispersions))

        # Low dispersion = high confidence
        if avg_dispersion < 0.02:
            return 0.85
        elif avg_dispersion < 0.05:
            return 0.70
        elif avg_dispersion < 0.10:
            return 0.55
        elif avg_dispersion < 0.20:
            return 0.40
        else:
            return 0.25

    def _fetch_data(self, ticker: str) -> pd.DataFrame | None:
        """Fetch 2 years of OHLCV data via yfinance."""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y")
            if df is not None and not df.empty:
                return df
            # Try .NS
            if "." not in ticker:
                stock = yf.Ticker(f"{ticker}.NS")
                return stock.history(period="2y")
        except Exception:
            pass
        return None
