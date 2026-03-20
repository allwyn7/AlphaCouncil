"""Fyers broker adapter using the fyers-apiv3 SDK.

This is positioned as a **backup broker** -- it implements the full
:class:`~alphacouncil.core.broker.base.BrokerAdapter` interface but is
intended for use when Angel One is unavailable or for portfolio
diversification across brokers.

Dependencies
------------
* ``fyers-apiv3``
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable

import pandas as pd
import structlog
from fyers_apiv3 import fyersModel  # type: ignore[import-untyped]
from fyers_apiv3.FyersWebsocket import data_ws  # type: ignore[import-untyped]

from alphacouncil.core.broker.base import (
    AuthenticationError,
    BrokerAdapter,
    BrokerError,
    ConnectionError,
    OrderRejectedError,
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

_SIDE_MAP: dict[OrderSide, int] = {
    OrderSide.BUY: 1,
    OrderSide.SELL: -1,
}

_ORDER_TYPE_MAP: dict[OrderType, int] = {
    OrderType.MARKET: 2,
    OrderType.LIMIT: 1,
    OrderType.SL: 3,
    OrderType.SL_M: 4,
}

_EXCHANGE_PREFIX: dict[Exchange, str] = {
    Exchange.NSE: "NSE",
    Exchange.NFO: "NFO",
}


def _fyers_symbol(ticker: str, exchange: Exchange = Exchange.NSE) -> str:
    """Convert an AlphaCouncil ticker to a Fyers-format symbol.

    Fyers expects ``NSE:RELIANCE-EQ`` for equities on NSE.
    If the ticker already contains a colon, return it as-is.
    """
    if ":" in ticker:
        return ticker
    # Strip .NS / .BO suffixes if present
    clean = ticker.replace(".NS", "").replace(".BO", "")
    prefix = _EXCHANGE_PREFIX.get(exchange, "NSE")
    return f"{prefix}:{clean}-EQ"


# ---------------------------------------------------------------------------
# FyersBroker
# ---------------------------------------------------------------------------


class FyersBroker(BrokerAdapter):
    """Backup broker adapter for Fyers using fyers-apiv3.

    Parameters
    ----------
    app_id:
        Fyers API v3 application ID (e.g. ``"XXXX-100"``).
    secret_id:
        Fyers API v3 secret key.
    redirect_uri:
        OAuth2 redirect URI registered with Fyers.
    access_token:
        Pre-generated access token.  If not provided, the caller must
        complete the OAuth2 flow and call :meth:`set_access_token` before
        using the adapter.
    """

    def __init__(
        self,
        app_id: str,
        secret_id: str,
        redirect_uri: str = "https://127.0.0.1",
        access_token: str | None = None,
    ) -> None:
        super().__init__(name="fyers")
        self._app_id = app_id
        self._secret_id = secret_id
        self._redirect_uri = redirect_uri
        self._access_token = access_token or ""

        self._fyers: fyersModel.FyersModel | None = None
        self._ws: data_ws.FyersDataSocket | None = None
        self._ws_running = False

        self._log = structlog.get_logger(broker="fyers")

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def get_auth_url(self) -> str:
        """Generate the Fyers OAuth2 authorization URL.

        The user must open this URL in a browser, log in, and extract the
        ``auth_code`` from the redirect to complete authentication.
        """
        session = fyersModel.SessionModel(
            client_id=self._app_id,
            secret_key=self._secret_id,
            redirect_uri=self._redirect_uri,
            response_type="code",
            grant_type="authorization_code",
        )
        return session.generate_authcode()

    async def authenticate(self, auth_code: str) -> None:
        """Exchange an authorization code for an access token."""
        session = fyersModel.SessionModel(
            client_id=self._app_id,
            secret_key=self._secret_id,
            redirect_uri=self._redirect_uri,
            response_type="code",
            grant_type="authorization_code",
        )
        session.set_token(auth_code)

        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(None, session.generate_token)
        except Exception as exc:
            raise AuthenticationError(f"Fyers token generation failed: {exc}") from exc

        token = resp.get("access_token")
        if not token:
            msg = resp.get("message", "Unknown error")
            raise AuthenticationError(f"Fyers authentication failed: {msg}")

        self.set_access_token(token)
        self._log.info("fyers.authenticated")

    def set_access_token(self, token: str) -> None:
        """Set or replace the access token and (re-)initialise the client."""
        self._access_token = token
        self._fyers = fyersModel.FyersModel(
            client_id=self._app_id,
            is_async=False,
            token=token,
            log_path="",
        )
        self._log.info("fyers.client_initialized")

    async def connect(self) -> None:
        """Ensure the Fyers client is ready.  Raises if no access token."""
        if not self._access_token:
            raise ConnectionError(
                "No access token set.  Call authenticate() or set_access_token() first."
            )
        if self._fyers is None:
            self.set_access_token(self._access_token)
        self._log.info("fyers.connected")

    def _require_client(self) -> fyersModel.FyersModel:
        if self._fyers is None:
            raise ConnectionError("Fyers client not initialized. Call connect() first.")
        return self._fyers

    # ------------------------------------------------------------------
    # BrokerAdapter: place_order
    # ------------------------------------------------------------------

    async def place_order(self, order: Order) -> str:
        """Place an order via the Fyers API."""
        client = self._require_client()
        symbol = _fyers_symbol(order.ticker, order.exchange)

        payload: dict[str, Any] = {
            "symbol": symbol,
            "qty": order.quantity,
            "type": _ORDER_TYPE_MAP.get(order.order_type, 2),
            "side": _SIDE_MAP[order.side],
            "productType": "CNC",
            "limitPrice": order.price or 0,
            "stopPrice": order.trigger_price or 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
        }

        self._log.info("fyers.place_order", symbol=symbol, payload=payload)
        loop = asyncio.get_running_loop()

        try:
            resp = await loop.run_in_executor(None, lambda: client.place_order(data=payload))
        except Exception as exc:
            self._log.error("fyers.place_order_failed", error=str(exc))
            raise BrokerError(f"Fyers placeOrder failed: {exc}") from exc

        if resp.get("s") != "ok":
            msg = resp.get("message", "Order rejected")
            raise OrderRejectedError(f"Fyers order rejected: {msg}")

        order_id = str(resp.get("id", ""))
        self._log.info("fyers.order_placed", order_id=order_id, symbol=symbol)
        return order_id

    # ------------------------------------------------------------------
    # BrokerAdapter: cancel_order
    # ------------------------------------------------------------------

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order on Fyers."""
        client = self._require_client()
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: client.cancel_order(data={"id": order_id}),
            )
            success = resp.get("s") == "ok"
            self._log.info("fyers.order_cancelled", order_id=order_id, success=success)
            return success
        except Exception as exc:
            self._log.error("fyers.cancel_failed", order_id=order_id, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # BrokerAdapter: get_positions
    # ------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Fetch open positions from Fyers."""
        client = self._require_client()
        loop = asyncio.get_running_loop()

        try:
            resp = await loop.run_in_executor(None, client.positions)
        except Exception as exc:
            self._log.error("fyers.positions_failed", error=str(exc))
            raise BrokerError(f"Failed to fetch positions: {exc}") from exc

        if resp.get("s") != "ok":
            return []

        raw = resp.get("netPositions", []) or []
        positions: list[Position] = []

        for p in raw:
            try:
                qty = int(p.get("netQty", 0))
                if qty == 0:
                    continue
                avg = float(p.get("avgPrice", 0))
                ltp = float(p.get("ltp", 0))
                pnl = float(p.get("pl", 0))
                pnl_pct = ((ltp / avg) - 1.0) * 100.0 if avg else 0.0

                # Extract ticker from Fyers symbol (e.g. "NSE:RELIANCE-EQ" -> "RELIANCE")
                symbol = str(p.get("symbol", ""))
                ticker_clean = symbol.split(":")[-1].replace("-EQ", "") if symbol else symbol

                positions.append(
                    Position(
                        ticker=ticker_clean,
                        quantity=abs(qty),
                        avg_price=avg,
                        current_price=ltp,
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                    )
                )
            except (ValueError, TypeError) as exc:
                self._log.warning("fyers.position_parse_error", raw=p, error=str(exc))

        return positions

    # ------------------------------------------------------------------
    # BrokerAdapter: get_holdings
    # ------------------------------------------------------------------

    async def get_holdings(self) -> list[dict[str, Any]]:
        """Fetch delivery holdings from Fyers."""
        client = self._require_client()
        loop = asyncio.get_running_loop()

        try:
            resp = await loop.run_in_executor(None, client.holdings)
        except Exception as exc:
            self._log.error("fyers.holdings_failed", error=str(exc))
            raise BrokerError(f"Failed to fetch holdings: {exc}") from exc

        if resp.get("s") != "ok":
            return []

        return resp.get("holdings", []) or []

    # ------------------------------------------------------------------
    # BrokerAdapter: get_funds
    # ------------------------------------------------------------------

    async def get_funds(self) -> dict[str, float]:
        """Fetch fund details from Fyers."""
        client = self._require_client()
        loop = asyncio.get_running_loop()

        try:
            resp = await loop.run_in_executor(None, client.funds)
        except Exception as exc:
            self._log.error("fyers.funds_failed", error=str(exc))
            raise BrokerError(f"Failed to fetch funds: {exc}") from exc

        if resp.get("s") != "ok":
            return {"cash": 0.0, "margin": 0.0, "deployed": 0.0, "total": 0.0}

        fund_limits = resp.get("fund_limit", []) or []

        # Fyers returns a list of fund limit entries keyed by "title"
        fund_map: dict[str, float] = {}
        for entry in fund_limits:
            title = str(entry.get("title", "")).lower()
            value = float(entry.get("equityAmount", 0))
            fund_map[title] = value

        cash = fund_map.get("available balance", 0.0)
        margin = fund_map.get("total margin available", cash)
        collateral = fund_map.get("collateral", 0.0)

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
        """Stream live prices via Fyers WebSocket.

        Reconnects automatically on disconnect.
        """
        if not self._access_token:
            raise ConnectionError("Must authenticate before streaming prices")

        symbols = [_fyers_symbol(t) for t in tickers]
        # Build a reverse map: fyers_symbol -> original ticker
        symbol_to_ticker: dict[str, str] = {
            _fyers_symbol(t): t for t in tickers
        }

        backoff = 1.0
        max_backoff = 60.0

        while True:
            try:
                self._log.info("fyers.ws_connect", symbols=symbols)

                def on_message(message: dict[str, Any]) -> None:
                    try:
                        symbol = str(message.get("symbol", ""))
                        ltp = float(message.get("ltp", 0))
                        original = symbol_to_ticker.get(symbol, symbol)
                        callback(original, ltp)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("fyers.ws_data_error", error=str(exc))

                def on_error(message: Any) -> None:
                    logger.error("fyers.ws_error", message=str(message))

                def on_close(message: Any) -> None:
                    logger.info("fyers.ws_closed", message=str(message))

                def on_open() -> None:
                    logger.info("fyers.ws_opened")
                    nonlocal backoff
                    backoff = 1.0

                fyers_ws = data_ws.FyersDataSocket(
                    access_token=f"{self._app_id}:{self._access_token}",
                    log_path="",
                    litemode=True,
                    write_to_file=False,
                    reconnect=True,
                    on_connect=on_open,
                    on_close=on_close,
                    on_error=on_error,
                    on_message=on_message,
                )

                self._ws = fyers_ws

                fyers_ws.subscribe(symbols)
                fyers_ws.keep_running()

                # Run the blocking WebSocket in a thread
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, fyers_ws.connect)

            except asyncio.CancelledError:
                self._log.info("fyers.ws_cancelled")
                self._ws_running = False
                if self._ws:
                    try:
                        self._ws.unsubscribe(symbols)
                    except Exception:  # noqa: BLE001
                        pass
                return

            except Exception as exc:  # noqa: BLE001
                self._log.error(
                    "fyers.ws_reconnect",
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
        """Fetch historical candle data from Fyers.

        Parameters
        ----------
        interval:
            Resolution string, e.g. ``"1"``, ``"5"``, ``"15"``, ``"60"``,
            ``"D"`` (daily).  Common shorthands like ``"1d"`` and ``"5m"``
            are auto-mapped.
        """
        client = self._require_client()
        symbol = _fyers_symbol(ticker)

        # Map common shorthand intervals to Fyers format
        interval_map: dict[str, str] = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "1d": "D",
        }
        fyers_interval = interval_map.get(interval, interval)

        payload: dict[str, Any] = {
            "symbol": symbol,
            "resolution": fyers_interval,
            "date_format": "1",
            "range_from": from_date.strftime("%Y-%m-%d"),
            "range_to": to_date.strftime("%Y-%m-%d"),
            "cont_flag": "1",
        }

        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(None, lambda: client.history(data=payload))
        except Exception as exc:
            self._log.error("fyers.historical_failed", ticker=ticker, error=str(exc))
            raise BrokerError(f"Fyers history() failed for {ticker}: {exc}") from exc

        if resp.get("s") != "ok":
            self._log.warning("fyers.historical_empty", ticker=ticker, resp=resp)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        candles = resp.get("candles", []) or []
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp")
        return df

    # ------------------------------------------------------------------
    # BrokerAdapter: get_ltp
    # ------------------------------------------------------------------

    async def get_ltp(self, ticker: str) -> float:
        """Get last traded price from Fyers."""
        client = self._require_client()
        symbol = _fyers_symbol(ticker)

        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: client.quotes(data={"symbols": symbol}),
            )
        except Exception as exc:
            self._log.error("fyers.ltp_failed", ticker=ticker, error=str(exc))
            raise BrokerError(f"Fyers quotes() failed for {ticker}: {exc}") from exc

        if resp.get("s") != "ok":
            raise BrokerError(f"No quote data returned for {ticker}")

        quotes = resp.get("d", []) or []
        if not quotes:
            raise BrokerError(f"Empty quote response for {ticker}")

        ltp = quotes[0].get("v", {}).get("lp")
        if ltp is None:
            raise BrokerError(f"LTP not found in quote response for {ticker}")

        return float(ltp)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def disconnect(self) -> None:
        """Shut down the Fyers WebSocket and clear state."""
        if self._ws:
            try:
                self._ws.unsubscribe([])
            except Exception:  # noqa: BLE001
                pass
            self._ws = None
        self._ws_running = False
        self._fyers = None
        self._log.info("fyers.disconnected")

    def __repr__(self) -> str:
        return f"<FyersBroker app_id={self._app_id!r}>"
