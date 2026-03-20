"""Angel One broker adapter using the SmartAPI (smartapi-python) SDK.

This adapter provides live trading against Angel One (formerly Angel
Broking) via their SmartConnect REST API and SmartWebSocketV2 for
real-time price streaming.

Features
--------
* TOTP-based 2FA login via *pyotp*.
* Automatic symbol-token mapping from the Angel One instrument master.
* WebSocket auto-reconnect with exponential back-off.
* Rate limiting: max 10 orders / second to stay within API quotas.

Dependencies
------------
* ``smartapi-python`` (``SmartApi``)
* ``pyotp``
* ``websocket-client``  (pulled in by SmartApi)
"""

from __future__ import annotations

import asyncio
import time as _time
from collections import deque
from datetime import datetime
from typing import Any, Callable

import pandas as pd
import pyotp
import structlog
from SmartApi.smartConnect import SmartConnect  # type: ignore[import-untyped]
from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # type: ignore[import-untyped]

from alphacouncil.core.broker.base import (
    AuthenticationError,
    BrokerAdapter,
    BrokerError,
    ConnectionError,
    OrderRejectedError,
    RateLimitError,
)
from alphacouncil.core.config import IST
from alphacouncil.core.models import (
    Exchange,
    Order,
    OrderSide,
    OrderType,
    Position,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

_EXCHANGE_MAP: dict[Exchange, str] = {
    Exchange.NSE: "NSE",
    Exchange.NFO: "NFO",
}

_SIDE_MAP: dict[OrderSide, str] = {
    OrderSide.BUY: "BUY",
    OrderSide.SELL: "SELL",
}

_ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.SL: "STOPLOSS_LIMIT",
    OrderType.SL_M: "STOPLOSS_MARKET",
    OrderType.AMO: "AMO",
}

# Maximum orders per second (Angel One rate limit)
_MAX_ORDERS_PER_SECOND = 10


# ---------------------------------------------------------------------------
# AngelOneBroker
# ---------------------------------------------------------------------------


class AngelOneBroker(BrokerAdapter):
    """Live broker adapter for Angel One via SmartAPI.

    Parameters
    ----------
    api_key:
        SmartAPI developer key.
    client_id:
        Angel One login / client ID.
    password:
        Angel One login password.
    totp_secret:
        Base-32 TOTP secret for two-factor authentication.
    """

    def __init__(
        self,
        api_key: str,
        client_id: str,
        password: str,
        totp_secret: str,
    ) -> None:
        super().__init__(name="angelone")
        self._api_key = api_key
        self._client_id = client_id
        self._password = password
        self._totp_secret = totp_secret

        self._smart: SmartConnect = SmartConnect(api_key=api_key)
        self._auth_token: str | None = None
        self._refresh_token: str | None = None
        self._feed_token: str | None = None

        # Symbol -> token mapping (populated lazily)
        self._token_map: dict[str, str] = {}

        # Rate-limiter: timestamps of recent order submissions
        self._order_timestamps: deque[float] = deque(maxlen=_MAX_ORDERS_PER_SECOND)

        # WebSocket handle
        self._ws: SmartWebSocketV2 | None = None
        self._ws_running = False

        self._log = structlog.get_logger(broker="angelone")

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Authenticate with Angel One using TOTP-based 2FA."""
        await self.login()

    async def login(self) -> None:
        """Generate TOTP, authenticate, and store session tokens."""
        loop = asyncio.get_running_loop()

        totp = pyotp.TOTP(self._totp_secret).now()
        self._log.info("angelone.login_start", client_id=self._client_id)

        try:
            data = await loop.run_in_executor(
                None,
                lambda: self._smart.generateSession(
                    self._client_id,
                    self._password,
                    totp,
                ),
            )
        except Exception as exc:
            self._log.error("angelone.login_failed", error=str(exc))
            raise AuthenticationError(f"Angel One login failed: {exc}") from exc

        if data.get("status") is False:
            msg = data.get("message", "Unknown authentication error")
            self._log.error("angelone.login_rejected", message=msg)
            raise AuthenticationError(f"Angel One login rejected: {msg}")

        tokens = data.get("data", {})
        self._auth_token = tokens.get("jwtToken")
        self._refresh_token = tokens.get("refreshToken")
        self._feed_token = tokens.get("feedToken") or self._smart.getfeedToken()

        self._log.info("angelone.login_success", client_id=self._client_id)

    # ------------------------------------------------------------------
    # Token mapping
    # ------------------------------------------------------------------

    def _resolve_token(self, ticker: str, exchange: Exchange = Exchange.NSE) -> str:
        """Resolve an NSE trading symbol to the Angel One symbol token.

        Falls back to the ``ltpData`` endpoint if the local map is empty.
        """
        key = f"{exchange.value}:{ticker}"
        if key in self._token_map:
            return self._token_map[key]

        # Attempt a lookup via ltpData which returns the token
        try:
            resp = self._smart.ltpData(exchange.value, ticker, "0")
            if resp and resp.get("data"):
                token = str(resp["data"].get("symboltoken", ""))
                if token:
                    self._token_map[key] = token
                    return token
        except Exception as exc:  # noqa: BLE001
            self._log.warning("angelone.token_resolve_failed", ticker=ticker, error=str(exc))

        raise BrokerError(
            f"Cannot resolve symbol token for {ticker} on {exchange.value}. "
            "Ensure the instrument master is loaded or the symbol is correct."
        )

    def load_token_map(self, mapping: dict[str, str]) -> None:
        """Bulk-load a ``{"EXCHANGE:SYMBOL": "token"}`` mapping.

        This is typically built from Angel One's downloadable instrument
        master CSV at session start.
        """
        self._token_map.update(mapping)
        self._log.info("angelone.token_map_loaded", count=len(mapping))

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _enforce_rate_limit(self) -> None:
        """Block until we are within the 10-orders/second budget."""
        now = _time.monotonic()
        while (
            len(self._order_timestamps) >= _MAX_ORDERS_PER_SECOND
            and now - self._order_timestamps[0] < 1.0
        ):
            wait = 1.0 - (now - self._order_timestamps[0])
            self._log.debug("angelone.rate_limit_wait", wait_s=round(wait, 3))
            await asyncio.sleep(wait)
            now = _time.monotonic()
        self._order_timestamps.append(now)

    # ------------------------------------------------------------------
    # BrokerAdapter: place_order
    # ------------------------------------------------------------------

    async def place_order(self, order: Order) -> str:
        """Submit an order to Angel One via SmartConnect.placeOrder()."""
        await self._enforce_rate_limit()

        token = self._resolve_token(order.ticker, order.exchange)
        variety = "AMO" if order.order_type == OrderType.AMO else "NORMAL"

        params: dict[str, Any] = {
            "variety": variety,
            "tradingsymbol": order.ticker,
            "symboltoken": token,
            "transactiontype": _SIDE_MAP[order.side],
            "exchange": _EXCHANGE_MAP.get(order.exchange, order.exchange.value),
            "ordertype": _ORDER_TYPE_MAP.get(order.order_type, "MARKET"),
            "producttype": "CNC",  # delivery
            "duration": "DAY",
            "quantity": order.quantity,
            "price": order.price or 0,
            "squareoff": 0,
            "stoploss": 0,
            "triggerprice": order.trigger_price or 0,
        }

        self._log.info("angelone.place_order", ticker=order.ticker, params=params)
        loop = asyncio.get_running_loop()

        try:
            resp = await loop.run_in_executor(
                None,
                lambda: self._smart.placeOrder(params),
            )
        except Exception as exc:
            self._log.error("angelone.place_order_failed", error=str(exc))
            raise BrokerError(f"placeOrder failed: {exc}") from exc

        if resp is None:
            raise OrderRejectedError("Angel One returned None for placeOrder")

        # SmartAPI returns the order ID directly as a string on success
        order_id = str(resp)
        self._log.info("angelone.order_placed", order_id=order_id, ticker=order.ticker)
        return order_id

    # ------------------------------------------------------------------
    # BrokerAdapter: cancel_order
    # ------------------------------------------------------------------

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order on Angel One."""
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: self._smart.cancelOrder(order_id, "NORMAL"),
            )
            self._log.info("angelone.order_cancelled", order_id=order_id, response=resp)
            return True
        except Exception as exc:
            self._log.error("angelone.cancel_failed", order_id=order_id, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # BrokerAdapter: get_positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Fetch open positions from Angel One and convert to Position models."""
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(None, self._smart.position)
        except Exception as exc:
            self._log.error("angelone.positions_failed", error=str(exc))
            raise BrokerError(f"Failed to fetch positions: {exc}") from exc

        if not resp or resp.get("status") is False:
            return []

        raw_positions = resp.get("data", []) or []
        positions: list[Position] = []

        for p in raw_positions:
            try:
                qty = int(p.get("netqty", 0) or p.get("quantity", 0))
                if qty == 0:
                    continue
                avg = float(p.get("avgnetprice", 0) or p.get("averageprice", 0))
                ltp = float(p.get("ltp", 0))
                pnl = float(p.get("pnl", 0) or p.get("unrealised", 0))
                pnl_pct = ((ltp / avg) - 1.0) * 100.0 if avg else 0.0

                positions.append(
                    Position(
                        ticker=str(p.get("tradingsymbol", "")),
                        quantity=abs(qty),
                        avg_price=avg,
                        current_price=ltp,
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                    )
                )
            except (ValueError, TypeError) as exc:
                self._log.warning("angelone.position_parse_error", raw=p, error=str(exc))

        return positions

    # ------------------------------------------------------------------
    # BrokerAdapter: get_holdings
    # ------------------------------------------------------------------

    async def get_holdings(self) -> list[dict[str, Any]]:
        """Fetch delivery holdings from Angel One."""
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(None, self._smart.holding)
        except Exception as exc:
            self._log.error("angelone.holdings_failed", error=str(exc))
            raise BrokerError(f"Failed to fetch holdings: {exc}") from exc

        if not resp or resp.get("status") is False:
            return []

        return resp.get("data", []) or []

    # ------------------------------------------------------------------
    # BrokerAdapter: get_funds
    # ------------------------------------------------------------------

    async def get_funds(self) -> dict[str, float]:
        """Fetch fund/margin details from Angel One RMS limits."""
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(None, self._smart.rmsLimit)
        except Exception as exc:
            self._log.error("angelone.funds_failed", error=str(exc))
            raise BrokerError(f"Failed to fetch funds: {exc}") from exc

        if not resp or resp.get("status") is False:
            return {"cash": 0.0, "margin": 0.0, "deployed": 0.0, "total": 0.0}

        data = resp.get("data", {}) or {}
        cash = float(data.get("availablecash", 0))
        margin = float(data.get("availableintradaypayin", 0))
        collateral = float(data.get("collateral", 0))

        return {
            "cash": round(cash, 2),
            "margin": round(margin, 2),
            "deployed": round(collateral, 2),
            "total": round(cash + collateral, 2),
        }

    # ------------------------------------------------------------------
    # BrokerAdapter: stream_prices (WebSocket)
    # ------------------------------------------------------------------

    async def stream_prices(
        self,
        tickers: list[str],
        callback: Callable[[str, float], Any],
    ) -> None:
        """Stream live prices via Angel One SmartWebSocketV2.

        Reconnects automatically on disconnect with exponential back-off.
        """
        if not self._feed_token or not self._auth_token:
            raise ConnectionError("Must call login() before streaming prices")

        # Build token list for subscription
        # Angel One expects [exchange_type, token] pairs
        token_list: list[list[Any]] = []
        ticker_by_token: dict[str, str] = {}
        for ticker in tickers:
            try:
                token = self._resolve_token(ticker)
                token_list.append([1, token])  # 1 = NSE exchange type
                ticker_by_token[token] = ticker
            except BrokerError:
                self._log.warning("angelone.stream_skip", ticker=ticker)

        if not token_list:
            raise BrokerError("No valid tokens to subscribe for streaming")

        backoff = 1.0
        max_backoff = 60.0

        while True:
            try:
                self._log.info("angelone.ws_connect", num_tokens=len(token_list))

                ws = SmartWebSocketV2(
                    self._auth_token,
                    self._api_key,
                    self._client_id,
                    self._feed_token,
                )

                def on_data(wsapp: Any, message: dict[str, Any]) -> None:
                    try:
                        token_str = str(message.get("token", ""))
                        ltp = float(message.get("last_traded_price", 0)) / 100.0
                        resolved_ticker = ticker_by_token.get(token_str, token_str)
                        callback(resolved_ticker, ltp)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("angelone.ws_data_error", error=str(exc))

                def on_open(wsapp: Any) -> None:
                    logger.info("angelone.ws_opened")
                    # Subscribe: mode 1 = LTP, 2 = Quote, 3 = Snap Quote
                    ws.subscribe("abc123", 1, token_list)
                    nonlocal backoff
                    backoff = 1.0  # Reset on successful connect

                def on_error(wsapp: Any, error: Any) -> None:
                    logger.error("angelone.ws_error", error=str(error))

                def on_close(wsapp: Any) -> None:
                    logger.info("angelone.ws_closed")

                ws.on_data = on_data
                ws.on_open = on_open
                ws.on_error = on_error
                ws.on_close = on_close

                self._ws = ws
                self._ws_running = True

                # Run the blocking WebSocket in a thread
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, ws.connect)

            except asyncio.CancelledError:
                self._log.info("angelone.ws_cancelled")
                self._ws_running = False
                if self._ws:
                    try:
                        self._ws.close_connection()
                    except Exception:  # noqa: BLE001
                        pass
                return

            except Exception as exc:  # noqa: BLE001
                self._log.error(
                    "angelone.ws_reconnect",
                    error=str(exc),
                    backoff_s=backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    # ------------------------------------------------------------------
    # BrokerAdapter: get_historical
    # ------------------------------------------------------------------

    async def get_historical(
        self,
        ticker: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        """Fetch historical candle data from Angel One.

        Parameters
        ----------
        interval:
            One of ``"ONE_MINUTE"``, ``"FIVE_MINUTE"``, ``"FIFTEEN_MINUTE"``,
            ``"THIRTY_MINUTE"``, ``"ONE_HOUR"``, ``"ONE_DAY"``.
        """
        token = self._resolve_token(ticker)
        exchange = "NSE"

        # Map common shorthand intervals to Angel One format
        interval_map: dict[str, str] = {
            "1m": "ONE_MINUTE",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "30m": "THIRTY_MINUTE",
            "1h": "ONE_HOUR",
            "1d": "ONE_DAY",
        }
        ao_interval = interval_map.get(interval, interval)

        params: dict[str, str] = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": ao_interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M"),
        }

        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: self._smart.getCandleData(params),
            )
        except Exception as exc:
            self._log.error("angelone.historical_failed", ticker=ticker, error=str(exc))
            raise BrokerError(f"getCandleData failed for {ticker}: {exc}") from exc

        if not resp or resp.get("status") is False:
            self._log.warning("angelone.historical_empty", ticker=ticker)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        candles = resp.get("data", []) or []
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        return df

    # ------------------------------------------------------------------
    # BrokerAdapter: get_ltp
    # ------------------------------------------------------------------

    async def get_ltp(self, ticker: str) -> float:
        """Get last traded price from Angel One."""
        token = self._resolve_token(ticker)
        exchange = "NSE"

        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: self._smart.ltpData(exchange, ticker, token),
            )
        except Exception as exc:
            self._log.error("angelone.ltp_failed", ticker=ticker, error=str(exc))
            raise BrokerError(f"ltpData failed for {ticker}: {exc}") from exc

        if not resp or resp.get("status") is False:
            raise BrokerError(f"No LTP data returned for {ticker}")

        data = resp.get("data", {}) or {}
        ltp = data.get("ltp")
        if ltp is None:
            raise BrokerError(f"LTP missing in response for {ticker}")

        return float(ltp)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def disconnect(self) -> None:
        """Close WebSocket and clear session."""
        if self._ws and self._ws_running:
            try:
                self._ws.close_connection()
            except Exception:  # noqa: BLE001
                pass
            self._ws_running = False
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._smart.terminateSession, self._client_id)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("angelone.disconnect_error", error=str(exc))
        self._log.info("angelone.disconnected")

    def __repr__(self) -> str:
        return f"<AngelOneBroker client_id={self._client_id!r}>"
