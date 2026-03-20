"""AlphaCouncil -- Streamlit Dashboard.

A comprehensive 8-tab monitoring dashboard for the AlphaCouncil multi-agent
quantitative trading system.  The dashboard is **self-contained**: it connects
directly to the SQLite database and reads cached state -- it does NOT import
any agent or orchestration code.

Launch::

    streamlit run alphacouncil/dashboard/app.py

Requirements::

    pip install streamlit plotly pandas sqlalchemy streamlit-autorefresh
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import MetaData, Table, create_engine, desc, func, select, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH: Final[str] = "data/alphacouncil.db"
DB_URL: Final[str] = f"sqlite:///{DB_PATH}"
INITIAL_CAPITAL: Final[float] = 1_000_000.0  # Rs 10 lakh
REFRESH_INTERVAL_SECONDS: Final[int] = 30

AGENT_NAMES: Final[list[str]] = [
    "growth_momentum",
    "value_contrarian",
    "technical_swing",
    "sentiment_nlp",
    "macro_regime",
    "statistical_arb",
]

ACTION_COLORS: Final[dict[str, str]] = {
    "BUY": "#00c853",
    "SELL": "#ff1744",
    "HOLD": "#9e9e9e",
    "buy": "#00c853",
    "sell": "#ff1744",
    "hold": "#9e9e9e",
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AlphaCouncil",
    page_icon="\u265f",  # chess pawn
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Auto-refresh via streamlit-autorefresh (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore[import-untyped]

    st_autorefresh(
        interval=REFRESH_INTERVAL_SECONDS * 1000,
        limit=None,
        key="dashboard_autorefresh",
    )
except ImportError:
    pass  # auto-refresh not available; user can refresh manually

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


@st.cache_resource(ttl=300)
def _get_engine() -> Engine:
    """Return a cached SQLAlchemy Engine for the dashboard SQLite database."""
    db_file = Path(DB_PATH)
    if not db_file.exists():
        db_file.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        DB_URL,
        echo=False,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False},
    )


def _reflect_tables(engine: Engine) -> dict[str, Table]:
    """Reflect existing tables from the database. Returns a dict of name -> Table."""
    meta = MetaData()
    meta.reflect(bind=engine)
    return dict(meta.tables)


def _table_exists(tables: dict[str, Table], name: str) -> bool:
    return name in tables


def _query_df(engine: Engine, stmt: Any) -> pd.DataFrame:
    """Execute a SQLAlchemy select statement and return a DataFrame."""
    try:
        with engine.connect() as conn:
            result = conn.execute(stmt)
            rows = result.fetchall()
            if not rows:
                return pd.DataFrame()
            columns = list(result.keys())
            return pd.DataFrame(rows, columns=columns)
    except Exception as exc:
        logger.warning("Query failed: %s", exc)
        return pd.DataFrame()


def _query_scalar(engine: Engine, stmt: Any) -> Any:
    """Execute a statement and return the first scalar value."""
    try:
        with engine.connect() as conn:
            return conn.execute(stmt).scalar()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _no_data(message: str = "No data available.") -> None:
    """Display a standardized 'no data' placeholder."""
    st.info(message)


def _format_inr(value: float) -> str:
    """Format a float as an Indian-style currency string."""
    if abs(value) >= 1_00_00_000:  # 1 crore
        return f"Rs {value / 1_00_00_000:,.2f} Cr"
    if abs(value) >= 1_00_000:  # 1 lakh
        return f"Rs {value / 1_00_000:,.2f} L"
    return f"Rs {value:,.2f}"


def _pnl_color(value: float) -> str:
    """Return a CSS color string for P&L values."""
    if value > 0:
        return "color: #00c853"
    if value < 0:
        return "color: #ff1744"
    return "color: #9e9e9e"


def _conviction_opacity(conviction: float) -> float:
    """Map conviction (0-100) to an opacity value (0.2-1.0)."""
    return 0.2 + 0.8 * (min(max(conviction, 0), 100) / 100)


def _safe_json_loads(raw: str | None) -> dict[str, Any]:
    """Parse a JSON string, returning an empty dict on failure."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar(engine: Engine, tables: dict[str, Table]) -> None:
    """Render the sidebar with system status, last update, and quick actions."""
    st.sidebar.title("AlphaCouncil")
    st.sidebar.caption("Multi-Agent Quantitative Trading")

    # --- System status ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")

    db_exists = Path(DB_PATH).exists()
    if db_exists:
        st.sidebar.markdown(":green_circle: **Database**: Connected")
    else:
        st.sidebar.markdown(":red_circle: **Database**: Not found")

    # Kill switch status
    kill_active = _get_kill_switch_status(engine, tables)
    if kill_active is True:
        st.sidebar.markdown(":red_circle: **Kill Switch**: ACTIVE")
    elif kill_active is False:
        st.sidebar.markdown(":green_circle: **Kill Switch**: Inactive")
    else:
        st.sidebar.markdown(":white_circle: **Kill Switch**: Unknown")

    # Latest portfolio value
    if _table_exists(tables, "portfolio_snapshots"):
        tbl = tables["portfolio_snapshots"]
        latest_val = _query_scalar(
            engine,
            select(tbl.c.total_value).order_by(desc(tbl.c.timestamp)).limit(1),
        )
        if latest_val is not None:
            st.sidebar.metric("Portfolio Value", _format_inr(latest_val))

    # --- Last update ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Last Update")
    if _table_exists(tables, "audit_trail"):
        tbl = tables["audit_trail"]
        last_ts = _query_scalar(
            engine,
            select(func.max(tbl.c.timestamp)),
        )
        if last_ts is not None:
            st.sidebar.write(f"{last_ts}")
        else:
            st.sidebar.write("No activity recorded")
    else:
        st.sidebar.write("N/A")

    st.sidebar.write(f"Dashboard refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")

    # --- Quick actions ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Actions")

    if st.sidebar.button("Refresh Dashboard", use_container_width=True):
        st.rerun()

    if st.sidebar.button(
        "KILL SWITCH",
        type="primary",
        use_container_width=True,
    ):
        st.sidebar.warning(
            "Kill switch activation requires the running system. "
            "Use the CLI or API: `POST /api/kill-switch/activate`"
        )

    # Agent count
    if _table_exists(tables, "agent_signals"):
        tbl = tables["agent_signals"]
        agent_count = _query_scalar(
            engine,
            select(func.count(func.distinct(tbl.c.agent_id))),
        )
        if agent_count:
            st.sidebar.metric("Active Agents", int(agent_count))


def _get_kill_switch_status(engine: Engine, tables: dict[str, Table]) -> bool | None:
    """Check the audit trail for the most recent kill switch event."""
    if not _table_exists(tables, "audit_trail"):
        return None
    tbl = tables["audit_trail"]
    df = _query_df(
        engine,
        select(tbl.c.action, tbl.c.timestamp)
        .where(tbl.c.action.in_(["KILL_SWITCH_ACTIVATED", "KILL_SWITCH_RESET"]))
        .order_by(desc(tbl.c.timestamp))
        .limit(1),
    )
    if df.empty:
        return False
    return str(df.iloc[0]["action"]) == "KILL_SWITCH_ACTIVATED"


# ===========================================================================
# Tab 1 -- The Council
# ===========================================================================


def _render_tab_council(engine: Engine, tables: dict[str, Table]) -> None:
    """Tab 1: Signal grid, agreement heatmap, and risk veto flags."""
    st.header("The Council")

    if not _table_exists(tables, "agent_signals"):
        _no_data("Agent signals table not found.")
        return

    tbl = tables["agent_signals"]

    # Fetch the most recent signal per (agent, ticker) combination
    subq = (
        select(
            tbl.c.agent_id,
            tbl.c.symbol,
            tbl.c.action,
            tbl.c.confidence,
            tbl.c.payload,
            tbl.c.timestamp,
            func.row_number()
            .over(partition_by=[tbl.c.agent_id, tbl.c.symbol], order_by=desc(tbl.c.timestamp))
            .label("rn"),
        )
        .where(tbl.c.symbol.is_not(None))
        .subquery()
    )
    stmt = select(subq).where(subq.c.rn == 1)
    df = _query_df(engine, stmt)

    if df.empty:
        _no_data("No agent signals recorded yet.")
        return

    # Normalize action strings to uppercase
    df["action"] = df["action"].str.upper()

    # --- Council final decision (majority vote per ticker) ---
    st.subheader("Council Final Decision")

    tickers = sorted(df["symbol"].dropna().unique())
    agents = sorted(df["agent_id"].dropna().unique())

    if tickers:
        council_decisions: list[dict[str, Any]] = []
        for ticker in tickers:
            ticker_df = df[df["symbol"] == ticker]
            action_counts = ticker_df["action"].value_counts()
            total_signals = len(ticker_df)
            majority_action = action_counts.index[0] if len(action_counts) > 0 else "HOLD"
            agreement = (
                action_counts.iloc[0] / total_signals * 100 if total_signals > 0 else 0
            )
            avg_conviction = ticker_df["confidence"].mean()

            # Check for risk veto in payload
            risk_vetoed = False
            for _, row in ticker_df.iterrows():
                payload = _safe_json_loads(row.get("payload"))
                if payload.get("risk_vetoed", False):
                    risk_vetoed = True
                    break

            council_decisions.append(
                {
                    "Ticker": ticker,
                    "Decision": majority_action,
                    "Agreement %": f"{agreement:.0f}%",
                    "Avg Conviction": f"{avg_conviction:.0f}",
                    "Signals": total_signals,
                    "Risk Veto": "VETOED" if risk_vetoed else "",
                }
            )

        council_df = pd.DataFrame(council_decisions)
        st.dataframe(
            council_df.style.map(
                lambda v: f"background-color: {ACTION_COLORS.get(v, 'transparent')}; color: white"
                if v in ACTION_COLORS
                else "",
                subset=["Decision"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    # --- Signal grid: rows=tickers, columns=agents ---
    st.subheader("Per-Ticker Signal Grid")

    if tickers and agents:
        grid_data: dict[str, dict[str, str]] = {ticker: {} for ticker in tickers}
        conviction_data: dict[str, dict[str, float]] = {ticker: {} for ticker in tickers}

        for _, row in df.iterrows():
            ticker = row["symbol"]
            agent = row["agent_id"]
            action = row["action"]
            conviction = row["confidence"] if pd.notna(row["confidence"]) else 50.0
            if ticker in grid_data:
                grid_data[ticker][agent] = action
                conviction_data[ticker][agent] = conviction

        grid_df = pd.DataFrame(grid_data).T
        grid_df.index.name = "Ticker"

        # Fill missing with HOLD
        for agent in agents:
            if agent not in grid_df.columns:
                grid_df[agent] = "HOLD"
        grid_df = grid_df.fillna("HOLD")

        # Style cells
        def _color_cell(val: str) -> str:
            bg = ACTION_COLORS.get(str(val).upper(), "#424242")
            return f"background-color: {bg}; color: white; text-align: center; font-weight: bold"

        st.dataframe(
            grid_df.style.map(_color_cell),
            use_container_width=True,
        )

        # --- Agreement heatmap ---
        st.subheader("Agent Agreement Heatmap")

        # Build a numeric matrix: +1 BUY, -1 SELL, 0 HOLD
        action_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
        numeric_matrix = grid_df.replace(action_map).apply(pd.to_numeric, errors="coerce").fillna(0)

        # Compute pairwise agreement between agents
        if len(agents) > 1:
            agreement_matrix = pd.DataFrame(
                index=agents, columns=agents, dtype=float
            )
            for a1 in agents:
                for a2 in agents:
                    if a1 in numeric_matrix.columns and a2 in numeric_matrix.columns:
                        matches = (numeric_matrix[a1] == numeric_matrix[a2]).sum()
                        total = len(numeric_matrix)
                        agreement_matrix.loc[a1, a2] = (
                            matches / total * 100 if total > 0 else 0
                        )
                    else:
                        agreement_matrix.loc[a1, a2] = 0

            agreement_matrix = agreement_matrix.astype(float)

            fig_heatmap = px.imshow(
                agreement_matrix,
                labels=dict(x="Agent", y="Agent", color="Agreement %"),
                color_continuous_scale="RdYlGn",
                zmin=0,
                zmax=100,
                text_auto=".0f",
                title="Pairwise Agent Agreement (%)",
            )
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            _no_data("Need at least 2 agents for an agreement heatmap.")
    else:
        _no_data("Insufficient data for signal grid.")


# ===========================================================================
# Tab 2 -- Portfolio Battle
# ===========================================================================


def _render_tab_portfolio_battle(engine: Engine, tables: dict[str, Table]) -> None:
    """Tab 2: Comparative equity curves and leaderboard."""
    st.header("Portfolio Battle")

    has_agent_snaps = _table_exists(tables, "agent_portfolio_snapshots")
    has_portfolio_snaps = _table_exists(tables, "portfolio_snapshots")

    if not has_agent_snaps and not has_portfolio_snaps:
        _no_data("No portfolio snapshot tables found.")
        return

    # --- Load agent equity curves ---
    curves: dict[str, pd.DataFrame] = {}

    if has_agent_snaps:
        tbl = tables["agent_portfolio_snapshots"]
        agent_df = _query_df(
            engine,
            select(tbl.c.timestamp, tbl.c.agent_id, tbl.c.total_value).order_by(
                tbl.c.timestamp
            ),
        )
        if not agent_df.empty:
            for agent_id, grp in agent_df.groupby("agent_id"):
                curves[str(agent_id)] = grp[["timestamp", "total_value"]].reset_index(
                    drop=True
                )

    # Council = main portfolio
    if has_portfolio_snaps:
        tbl = tables["portfolio_snapshots"]
        council_df = _query_df(
            engine,
            select(tbl.c.timestamp, tbl.c.total_value).order_by(tbl.c.timestamp),
        )
        if not council_df.empty:
            curves["Council"] = council_df

    if not curves:
        _no_data("No equity curve data available.")
        return

    # --- Plot equity curves ---
    st.subheader("Equity Curves")
    fig = go.Figure()
    for name, curve_df in sorted(curves.items()):
        fig.add_trace(
            go.Scatter(
                x=curve_df["timestamp"],
                y=curve_df["total_value"],
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title="Equity Curves (6 Agents + Council + Nifty 50)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (INR)",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Leaderboard table ---
    st.subheader("Leaderboard")
    leaderboard_rows: list[dict[str, Any]] = []

    for name, curve_df in sorted(curves.items()):
        values = curve_df["total_value"].astype(float)
        if len(values) < 2:
            continue

        start_val = values.iloc[0]
        end_val = values.iloc[-1]
        total_return = (end_val - start_val) / start_val if start_val else 0

        # Daily returns for Sharpe / Sortino
        daily_returns = values.pct_change().dropna()

        # CAGR approximation (assume 252 trading days per year)
        n_days = len(values)
        years = n_days / 252 if n_days > 0 else 1
        cagr = (end_val / start_val) ** (1 / years) - 1 if start_val > 0 and years > 0 else 0

        # Sharpe (annualized, risk-free ~ 6% for India)
        rf_daily = 0.06 / 252
        excess = daily_returns - rf_daily
        sharpe = (
            excess.mean() / excess.std() * math.sqrt(252)
            if len(excess) > 1 and excess.std() > 0
            else 0
        )

        # Sortino
        downside = excess[excess < 0]
        sortino = (
            excess.mean() / downside.std() * math.sqrt(252)
            if len(downside) > 1 and downside.std() > 0
            else 0
        )

        # Max drawdown
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax
        max_dd = drawdown.min() if len(drawdown) > 0 else 0

        # Win rate (percentage of positive-return days)
        win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100 if len(daily_returns) > 0 else 0

        # Rs 10L invested ending balance
        ending_10l = INITIAL_CAPITAL * (1 + total_return)

        leaderboard_rows.append(
            {
                "Strategy": name,
                "CAGR": f"{cagr:.1%}",
                "Sharpe": f"{sharpe:.2f}",
                "Sortino": f"{sortino:.2f}",
                "Max DD": f"{max_dd:.1%}",
                "Win Rate": f"{win_rate:.1f}%",
                "Rs 10L Ending": _format_inr(ending_10l),
            }
        )

    if leaderboard_rows:
        lb_df = pd.DataFrame(leaderboard_rows)

        def _color_metric(val: str) -> str:
            """Color positive values green, negative red."""
            try:
                numeric = float(val.strip("%").replace(",", "").replace("Rs ", "").replace(" L", "e5").replace(" Cr", "e7"))
                if numeric > 0:
                    return "color: #00c853"
                if numeric < 0:
                    return "color: #ff1744"
            except (ValueError, AttributeError):
                pass
            return ""

        st.dataframe(
            lb_df.style.map(_color_metric, subset=["CAGR", "Sharpe", "Sortino", "Max DD"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        _no_data("Insufficient data to build leaderboard.")


# ===========================================================================
# Tab 3 -- Live Positions
# ===========================================================================


def _render_tab_live_positions(engine: Engine, tables: dict[str, Table]) -> None:
    """Tab 3: Current positions, portfolio summary, and pending orders."""
    st.header("Live Positions")

    # --- Portfolio summary ---
    if _table_exists(tables, "portfolio_snapshots"):
        tbl = tables["portfolio_snapshots"]
        snap_df = _query_df(
            engine,
            select(tbl).order_by(desc(tbl.c.timestamp)).limit(1),
        )
        if not snap_df.empty:
            snap = snap_df.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            total_value = float(snap.get("total_value", 0))
            cash = float(snap.get("cash", 0))
            invested = float(snap.get("invested", 0))
            deployed_pct = invested / total_value * 100 if total_value > 0 else 0

            col1.metric("Total Portfolio Value", _format_inr(total_value))
            col2.metric("Cash Available", _format_inr(cash))
            col3.metric("Deployed", f"{deployed_pct:.1f}%")
            daily_pnl = float(snap.get("unrealised_pnl", 0) or 0)
            col4.metric(
                "Unrealised P&L",
                _format_inr(daily_pnl),
                delta=f"{daily_pnl:+,.0f}",
                delta_color="normal",
            )

    # --- Positions table ---
    st.subheader("Open Positions")

    if not _table_exists(tables, "positions"):
        _no_data("Positions table not found.")
        return

    tbl = tables["positions"]
    # Latest positions (most recent timestamp per symbol)
    subq = (
        select(
            tbl,
            func.row_number()
            .over(partition_by=tbl.c.symbol, order_by=desc(tbl.c.timestamp))
            .label("rn"),
        ).subquery()
    )
    pos_df = _query_df(engine, select(subq).where(subq.c.rn == 1))

    if pos_df.empty:
        _no_data("No open positions.")
    else:
        # Compute P&L columns if missing
        if "current_price" in pos_df.columns and "avg_entry_price" in pos_df.columns:
            pos_df["P&L"] = (
                (pos_df["current_price"].astype(float) - pos_df["avg_entry_price"].astype(float))
                * pos_df["quantity"].astype(float)
            )
            pos_df["P&L %"] = (
                (pos_df["current_price"].astype(float) - pos_df["avg_entry_price"].astype(float))
                / pos_df["avg_entry_price"].astype(float)
                * 100
            )

        display_cols = [
            c
            for c in [
                "symbol",
                "quantity",
                "avg_entry_price",
                "current_price",
                "P&L",
                "P&L %",
                "unrealised_pnl",
                "exchange",
            ]
            if c in pos_df.columns
        ]

        if display_cols:
            styled = pos_df[display_cols].style
            for col in ["P&L", "P&L %", "unrealised_pnl"]:
                if col in display_cols:
                    styled = styled.map(
                        lambda v: "color: #00c853" if isinstance(v, (int, float)) and v > 0
                        else ("color: #ff1744" if isinstance(v, (int, float)) and v < 0 else ""),
                        subset=[col],
                    )
            st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Stop-loss levels (from recent signals) ---
    st.subheader("Stop-Loss Levels")
    if _table_exists(tables, "agent_signals"):
        sig_tbl = tables["agent_signals"]
        sl_df = _query_df(
            engine,
            select(sig_tbl.c.symbol, sig_tbl.c.payload, sig_tbl.c.timestamp)
            .where(sig_tbl.c.symbol.is_not(None))
            .order_by(desc(sig_tbl.c.timestamp))
            .limit(100),
        )
        if not sl_df.empty:
            sl_rows: list[dict[str, Any]] = []
            seen_tickers: set[str] = set()
            for _, row in sl_df.iterrows():
                ticker = row["symbol"]
                if ticker in seen_tickers:
                    continue
                payload = _safe_json_loads(row.get("payload"))
                sl = payload.get("stop_loss") or payload.get("stoploss")
                tp = payload.get("take_profit") or payload.get("target")
                if sl or tp:
                    sl_rows.append(
                        {
                            "Ticker": ticker,
                            "Stop Loss": f"Rs {sl:,.2f}" if sl else "N/A",
                            "Take Profit": f"Rs {tp:,.2f}" if tp else "N/A",
                        }
                    )
                    seen_tickers.add(ticker)
            if sl_rows:
                st.dataframe(pd.DataFrame(sl_rows), use_container_width=True, hide_index=True)
            else:
                _no_data("No stop-loss data in signal payloads.")
        else:
            _no_data("No signal data for stop-loss levels.")
    else:
        _no_data("Agent signals table not found.")

    # --- Pending orders ---
    st.subheader("Pending Orders")
    if _table_exists(tables, "trades"):
        trades_tbl = tables["trades"]
        # Show recent trades that might represent pending orders (order_type != MARKET recently placed)
        pending_df = _query_df(
            engine,
            select(trades_tbl)
            .where(trades_tbl.c.order_type.in_(["LIMIT", "SL", "SL_M", "limit", "sl", "sl_m"]))
            .order_by(desc(trades_tbl.c.timestamp))
            .limit(20),
        )
        if not pending_df.empty:
            display_cols = [
                c for c in ["symbol", "side", "order_type", "quantity", "price", "timestamp", "agent_id"]
                if c in pending_df.columns
            ]
            st.dataframe(pending_df[display_cols], use_container_width=True, hide_index=True)
        else:
            _no_data("No pending orders.")
    else:
        _no_data("Trades table not found.")


# ===========================================================================
# Tab 4 -- Agent Deep Dive
# ===========================================================================


def _render_tab_agent_deep_dive(engine: Engine, tables: dict[str, Table]) -> None:
    """Tab 4: Single-agent deep dive with holdings, trades, and factor breakdown."""
    st.header("Agent Deep Dive")

    # Determine available agents
    available_agents: list[str] = []
    if _table_exists(tables, "agent_signals"):
        tbl = tables["agent_signals"]
        agents_df = _query_df(
            engine,
            select(func.distinct(tbl.c.agent_id)),
        )
        if not agents_df.empty:
            available_agents = sorted(agents_df.iloc[:, 0].dropna().tolist())

    if not available_agents:
        available_agents = AGENT_NAMES

    selected_agent: str = st.selectbox(
        "Select Agent",
        options=available_agents,
        index=0,
    )

    if not selected_agent:
        _no_data("No agent selected.")
        return

    # --- Holdings ---
    st.subheader(f"Holdings: {selected_agent}")
    if _table_exists(tables, "agent_portfolio_snapshots"):
        tbl = tables["agent_portfolio_snapshots"]
        holdings_df = _query_df(
            engine,
            select(tbl)
            .where(tbl.c.agent_id == selected_agent)
            .order_by(desc(tbl.c.timestamp))
            .limit(1),
        )
        if not holdings_df.empty:
            row = holdings_df.iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Value", _format_inr(float(row.get("total_value", 0))))
            col2.metric("Cash", _format_inr(float(row.get("cash", 0))))
            col3.metric("Invested", _format_inr(float(row.get("invested", 0))))

            alloc = _safe_json_loads(row.get("allocation_json"))
            if alloc:
                alloc_df = pd.DataFrame(
                    [{"Ticker": k, "Value": v} for k, v in alloc.items()]
                )
                st.dataframe(alloc_df, use_container_width=True, hide_index=True)
        else:
            _no_data(f"No portfolio snapshot for {selected_agent}.")
    else:
        _no_data("Agent portfolio snapshots table not found.")

    # --- Recent trades ---
    st.subheader(f"Recent Trades: {selected_agent}")
    if _table_exists(tables, "trades"):
        tbl = tables["trades"]
        trades_df = _query_df(
            engine,
            select(tbl)
            .where(tbl.c.agent_id == selected_agent)
            .order_by(desc(tbl.c.timestamp))
            .limit(50),
        )
        if not trades_df.empty:
            display_cols = [
                c for c in ["timestamp", "symbol", "side", "quantity", "price", "order_type", "fees"]
                if c in trades_df.columns
            ]
            st.dataframe(trades_df[display_cols], use_container_width=True, hide_index=True)
        else:
            _no_data(f"No trades recorded for {selected_agent}.")
    else:
        _no_data("Trades table not found.")

    # --- Factor decomposition bar chart ---
    st.subheader(f"Factor Decomposition: {selected_agent}")
    if _table_exists(tables, "agent_signals"):
        tbl = tables["agent_signals"]
        factor_df = _query_df(
            engine,
            select(tbl.c.payload)
            .where(tbl.c.agent_id == selected_agent)
            .where(tbl.c.payload.is_not(None))
            .order_by(desc(tbl.c.timestamp))
            .limit(1),
        )
        if not factor_df.empty:
            payload = _safe_json_loads(factor_df.iloc[0]["payload"])
            factor_scores = payload.get("factor_scores", {})
            if factor_scores:
                factors_list = [
                    {"Factor": k, "Score": float(v)} for k, v in factor_scores.items()
                ]
                fdf = pd.DataFrame(factors_list).sort_values("Score", ascending=True)
                fig = px.bar(
                    fdf,
                    x="Score",
                    y="Factor",
                    orientation="h",
                    title=f"Factor Scores for {selected_agent}",
                    color="Score",
                    color_continuous_scale="RdYlGn",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                _no_data("No factor scores in latest signal payload.")
        else:
            _no_data("No signal payloads found.")
    else:
        _no_data("Agent signals table not found.")

    # --- Signal accuracy ---
    st.subheader(f"Signal Accuracy: {selected_agent}")
    if _table_exists(tables, "agent_signals"):
        tbl = tables["agent_signals"]
        accuracy_df = _query_df(
            engine,
            select(tbl.c.action, tbl.c.confidence, tbl.c.payload, tbl.c.timestamp)
            .where(tbl.c.agent_id == selected_agent)
            .where(tbl.c.action.in_(["BUY", "buy", "SELL", "sell"]))
            .order_by(desc(tbl.c.timestamp))
            .limit(200),
        )
        if not accuracy_df.empty:
            # Try to extract accuracy from payload or compute from available data
            accuracy_data: list[dict[str, Any]] = []
            for _, row in accuracy_df.iterrows():
                payload = _safe_json_loads(row.get("payload"))
                acc_5 = payload.get("accuracy_5d")
                acc_10 = payload.get("accuracy_10d")
                acc_20 = payload.get("accuracy_20d")
                if acc_5 is not None or acc_10 is not None or acc_20 is not None:
                    accuracy_data.append(
                        {
                            "Date": row["timestamp"],
                            "5-Day": f"{acc_5:.1%}" if acc_5 is not None else "N/A",
                            "10-Day": f"{acc_10:.1%}" if acc_10 is not None else "N/A",
                            "20-Day": f"{acc_20:.1%}" if acc_20 is not None else "N/A",
                        }
                    )
            if accuracy_data:
                st.dataframe(
                    pd.DataFrame(accuracy_data).head(20),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                total_signals = len(accuracy_df)
                buy_signals = len(accuracy_df[accuracy_df["action"].str.upper() == "BUY"])
                sell_signals = total_signals - buy_signals
                st.write(
                    f"Total directional signals: **{total_signals}** "
                    f"(BUY: {buy_signals}, SELL: {sell_signals}). "
                    "Accuracy lookback data not yet available in payloads."
                )
        else:
            _no_data("No directional signals to measure accuracy.")
    else:
        _no_data("Agent signals table not found.")

    # --- Performance by regime ---
    st.subheader(f"Performance by Regime: {selected_agent}")
    if _table_exists(tables, "agent_signals"):
        tbl = tables["agent_signals"]
        regime_df = _query_df(
            engine,
            select(tbl.c.payload, tbl.c.confidence, tbl.c.timestamp)
            .where(tbl.c.agent_id == selected_agent)
            .where(tbl.c.payload.is_not(None))
            .order_by(desc(tbl.c.timestamp))
            .limit(500),
        )
        if not regime_df.empty:
            regime_perf: dict[str, list[float]] = {}
            for _, row in regime_df.iterrows():
                payload = _safe_json_loads(row.get("payload"))
                regime = payload.get("regime") or payload.get("market_regime")
                if regime:
                    regime_perf.setdefault(regime, []).append(float(row["confidence"]))

            if regime_perf:
                regime_rows = [
                    {
                        "Regime": regime,
                        "Avg Conviction": f"{sum(vals) / len(vals):.1f}",
                        "Signal Count": len(vals),
                    }
                    for regime, vals in sorted(regime_perf.items())
                ]
                st.dataframe(
                    pd.DataFrame(regime_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                _no_data("No regime information in signal payloads.")
        else:
            _no_data("No payload data to extract regime performance.")
    else:
        _no_data("Agent signals table not found.")

    # --- Auto-research parameter changes ---
    st.subheader(f"Auto-Research Changes: {selected_agent}")
    if _table_exists(tables, "research_logs"):
        tbl = tables["research_logs"]
        research_df = _query_df(
            engine,
            select(tbl)
            .where(tbl.c.agent_id == selected_agent)
            .order_by(desc(tbl.c.timestamp))
            .limit(20),
        )
        if not research_df.empty:
            display_cols = [
                c for c in ["timestamp", "query", "result_summary", "tokens_used", "cost_usd"]
                if c in research_df.columns
            ]
            st.dataframe(research_df[display_cols], use_container_width=True, hide_index=True)
        else:
            _no_data(f"No research log entries for {selected_agent}.")
    else:
        _no_data("Research logs table not found.")


# ===========================================================================
# Tab 5 -- Market Analysis
# ===========================================================================


def _render_tab_market_analysis(engine: Engine, tables: dict[str, Table]) -> None:
    """Tab 5: Technical heatmap, sentiment feed, FII/DII flows, VIX gauge."""
    st.header("Market Analysis")

    # --- Technical heatmap ---
    st.subheader("Technical Indicator Heatmap")
    if _table_exists(tables, "agent_signals"):
        tbl = tables["agent_signals"]
        # Fetch recent technical signals
        tech_df = _query_df(
            engine,
            select(tbl.c.symbol, tbl.c.payload, tbl.c.timestamp)
            .where(tbl.c.signal_type.in_(["technical", "TECHNICAL"]))
            .where(tbl.c.symbol.is_not(None))
            .order_by(desc(tbl.c.timestamp))
            .limit(500),
        )
        if not tech_df.empty:
            # Extract indicator data from payloads
            indicator_rows: list[dict[str, Any]] = []
            seen_tickers: set[str] = set()
            for _, row in tech_df.iterrows():
                ticker = row["symbol"]
                if ticker in seen_tickers:
                    continue
                seen_tickers.add(ticker)
                payload = _safe_json_loads(row.get("payload"))
                if not payload:
                    continue

                rsi = payload.get("rsi", 50)
                macd_hist = payload.get("macd_hist", 0)
                bb_pos = payload.get("bb_position") or payload.get("bollinger_position", 0.5)
                adx = payload.get("adx", 0)

                # Classify indicators
                rsi_class = "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else "Neutral")
                macd_dir = "Bullish" if macd_hist > 0 else ("Bearish" if macd_hist < 0 else "Neutral")
                trend = "Strong" if adx > 25 else "Weak"

                indicator_rows.append(
                    {
                        "Ticker": ticker,
                        "RSI": round(rsi, 1),
                        "RSI Signal": rsi_class,
                        "MACD Direction": macd_dir,
                        "BB Position": round(float(bb_pos), 2) if bb_pos else 0.5,
                        "Trend (ADX)": trend,
                    }
                )

            if indicator_rows:
                ind_df = pd.DataFrame(indicator_rows)

                # Build a numeric heatmap
                heatmap_data: dict[str, list[float]] = {
                    "RSI": [],
                    "MACD": [],
                    "BB Position": [],
                    "ADX Trend": [],
                }
                heatmap_tickers: list[str] = []
                for irow in indicator_rows:
                    heatmap_tickers.append(irow["Ticker"])
                    # Normalize RSI to -1..+1 (30-centered)
                    rsi_norm = (irow["RSI"] - 50) / 50
                    heatmap_data["RSI"].append(rsi_norm)
                    heatmap_data["MACD"].append(
                        1.0 if irow["MACD Direction"] == "Bullish" else (-1.0 if irow["MACD Direction"] == "Bearish" else 0.0)
                    )
                    heatmap_data["BB Position"].append(float(irow["BB Position"]) * 2 - 1)
                    heatmap_data["ADX Trend"].append(1.0 if irow["Trend (ADX)"] == "Strong" else 0.0)

                heatmap_df = pd.DataFrame(heatmap_data, index=heatmap_tickers)
                fig_tech = px.imshow(
                    heatmap_df.T,
                    labels=dict(x="Stock", y="Indicator", color="Signal"),
                    color_continuous_scale="RdYlGn",
                    zmin=-1,
                    zmax=1,
                    title="Technical Indicators (Green=Bullish, Red=Bearish)",
                    aspect="auto",
                )
                fig_tech.update_layout(height=400)
                st.plotly_chart(fig_tech, use_container_width=True)

                # Also show the table
                st.dataframe(ind_df, use_container_width=True, hide_index=True)
            else:
                _no_data("No technical indicators found in signal payloads.")
        else:
            _no_data("No technical signals recorded.")
    else:
        _no_data("Agent signals table not found.")

    # --- Sentiment feed ---
    st.subheader("Sentiment Feed")
    if _table_exists(tables, "sentiment_cache"):
        tbl = tables["sentiment_cache"]
        sent_df = _query_df(
            engine,
            select(tbl).order_by(desc(tbl.c.timestamp)).limit(50),
        )
        if not sent_df.empty:
            # Scrollable sentiment feed
            display_cols = [
                c for c in ["timestamp", "symbol", "source", "score", "magnitude", "raw_text"]
                if c in sent_df.columns
            ]
            for col in ["score"]:
                if col in sent_df.columns:
                    sent_df[col] = sent_df[col].astype(float).round(3)

            st.dataframe(
                sent_df[display_cols].style.map(
                    lambda v: "color: #00c853" if isinstance(v, (int, float)) and v > 0.1
                    else ("color: #ff1744" if isinstance(v, (int, float)) and v < -0.1 else ""),
                    subset=["score"] if "score" in display_cols else [],
                ),
                use_container_width=True,
                hide_index=True,
                height=400,
            )
        else:
            _no_data("No sentiment data cached.")
    else:
        _no_data("Sentiment cache table not found.")

    # --- FII/DII flow bar chart ---
    st.subheader("FII / DII Flows")
    if _table_exists(tables, "agent_signals"):
        tbl = tables["agent_signals"]
        macro_df = _query_df(
            engine,
            select(tbl.c.payload, tbl.c.timestamp)
            .where(tbl.c.signal_type.in_(["macro", "MACRO"]))
            .order_by(desc(tbl.c.timestamp))
            .limit(30),
        )
        if not macro_df.empty:
            flow_rows: list[dict[str, Any]] = []
            for _, row in macro_df.iterrows():
                payload = _safe_json_loads(row.get("payload"))
                fii = payload.get("fii_net_flow")
                dii = payload.get("dii_net_flow")
                if fii is not None or dii is not None:
                    flow_rows.append(
                        {
                            "Date": row["timestamp"],
                            "FII Net (Cr)": float(fii) if fii else 0,
                            "DII Net (Cr)": float(dii) if dii else 0,
                        }
                    )
            if flow_rows:
                flow_df = pd.DataFrame(flow_rows)
                fig_flow = go.Figure()
                fig_flow.add_trace(
                    go.Bar(
                        x=flow_df["Date"],
                        y=flow_df["FII Net (Cr)"],
                        name="FII",
                        marker_color="#2196f3",
                    )
                )
                fig_flow.add_trace(
                    go.Bar(
                        x=flow_df["Date"],
                        y=flow_df["DII Net (Cr)"],
                        name="DII",
                        marker_color="#ff9800",
                    )
                )
                fig_flow.update_layout(
                    title="FII / DII Net Flows (INR Crore)",
                    barmode="group",
                    height=400,
                    xaxis_title="Date",
                    yaxis_title="Net Flow (Cr)",
                )
                st.plotly_chart(fig_flow, use_container_width=True)
            else:
                _no_data("No FII/DII flow data in macro signals.")
        else:
            _no_data("No macro signals recorded.")
    else:
        _no_data("Agent signals table not found.")

    # --- India VIX gauge ---
    st.subheader("India VIX & Regime")
    if _table_exists(tables, "agent_signals"):
        tbl = tables["agent_signals"]
        vix_df = _query_df(
            engine,
            select(tbl.c.payload, tbl.c.timestamp)
            .where(tbl.c.signal_type.in_(["macro", "MACRO"]))
            .order_by(desc(tbl.c.timestamp))
            .limit(1),
        )
        if not vix_df.empty:
            payload = _safe_json_loads(vix_df.iloc[0].get("payload"))
            vix_val = payload.get("india_vix")
            regime = payload.get("regime") or payload.get("market_regime", "Unknown")

            if vix_val is not None:
                col1, col2 = st.columns(2)
                with col1:
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=float(vix_val),
                            title={"text": "India VIX"},
                            gauge={
                                "axis": {"range": [0, 50]},
                                "bar": {"color": "#1a237e"},
                                "steps": [
                                    {"range": [0, 15], "color": "#c8e6c9"},
                                    {"range": [15, 25], "color": "#fff9c4"},
                                    {"range": [25, 50], "color": "#ffcdd2"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.75,
                                    "value": 25,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col2:
                    # Regime indicator
                    regime_colors: dict[str, str] = {
                        "BULL_LOW_VOL": "#00c853",
                        "BULL_HIGH_VOL": "#69f0ae",
                        "BEAR_LOW_VOL": "#ff5252",
                        "BEAR_HIGH_VOL": "#ff1744",
                        "SIDEWAYS": "#ffc107",
                        "FII_BUYING": "#2196f3",
                        "FII_SELLING": "#e91e63",
                        "PRE_EXPIRY": "#ff9800",
                        "EARNINGS_SEASON": "#9c27b0",
                        "BUDGET_POLICY": "#795548",
                    }
                    r_color = regime_colors.get(str(regime), "#9e9e9e")
                    st.markdown(
                        f'<div style="background-color: {r_color}; color: white; '
                        f'padding: 2rem; border-radius: 1rem; text-align: center; '
                        f'margin-top: 2rem;">'
                        f'<h2 style="margin: 0;">Market Regime</h2>'
                        f'<h1 style="margin: 0.5rem 0 0 0;">{regime}</h1>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                _no_data("India VIX value not found in macro payload.")
        else:
            _no_data("No macro signals for VIX data.")
    else:
        _no_data("Agent signals table not found.")


# ===========================================================================
# Tab 6 -- Research Lab
# ===========================================================================


def _render_tab_research_lab(engine: Engine, tables: dict[str, Table]) -> None:
    """Tab 6: Discovered signals, optimization history, regime heatmap, attribution."""
    st.header("Research Lab")

    # --- Discovered signals ---
    st.subheader("Discovered Signals")
    if _table_exists(tables, "research_logs"):
        tbl = tables["research_logs"]
        disc_df = _query_df(
            engine,
            select(tbl).order_by(desc(tbl.c.timestamp)).limit(50),
        )
        if not disc_df.empty:
            # Try to extract IC and robustness from result_summary
            signal_rows: list[dict[str, Any]] = []
            for _, row in disc_df.iterrows():
                summary = _safe_json_loads(row.get("result_summary"))
                if isinstance(summary, dict) and (summary.get("ic") is not None or summary.get("signal_name")):
                    signal_rows.append(
                        {
                            "Signal": summary.get("signal_name", row.get("query", "Unknown")),
                            "IC": f"{summary.get('ic', 0):.4f}" if summary.get("ic") is not None else "N/A",
                            "Robustness": f"{summary.get('robustness', 0):.2f}" if summary.get("robustness") is not None else "N/A",
                            "Agent": row.get("agent_id", ""),
                            "Date": row.get("timestamp", ""),
                        }
                    )

            if signal_rows:
                st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)
            else:
                # Show raw research logs
                display_cols = [
                    c for c in ["timestamp", "agent_id", "query", "result_summary", "tokens_used"]
                    if c in disc_df.columns
                ]
                st.dataframe(disc_df[display_cols], use_container_width=True, hide_index=True)
        else:
            _no_data("No research discoveries recorded.")
    else:
        _no_data("Research logs table not found.")

    # --- Parameter optimization history ---
    st.subheader("Parameter Optimization History")
    if _table_exists(tables, "audit_trail"):
        tbl = tables["audit_trail"]
        opt_df = _query_df(
            engine,
            select(tbl)
            .where(
                tbl.c.action.in_(
                    ["PARAM_CHANGE", "param_change", "AUTO_TUNE", "auto_tune", "parameter_update"]
                )
            )
            .order_by(desc(tbl.c.timestamp))
            .limit(100),
        )
        if not opt_df.empty:
            opt_rows: list[dict[str, Any]] = []
            for _, row in opt_df.iterrows():
                detail = _safe_json_loads(row.get("detail"))
                opt_rows.append(
                    {
                        "Date": row.get("timestamp", ""),
                        "Agent": row.get("actor", ""),
                        "Parameter": detail.get("parameter", row.get("resource", "")),
                        "Old Value": detail.get("old_value", ""),
                        "New Value": detail.get("new_value", ""),
                        "Evidence": detail.get("evidence", ""),
                    }
                )

            if opt_rows:
                opt_hist_df = pd.DataFrame(opt_rows)
                # Timeline chart of changes
                if len(opt_hist_df) > 1:
                    fig_opt = px.scatter(
                        opt_hist_df,
                        x="Date",
                        y="Agent",
                        color="Parameter",
                        hover_data=["Old Value", "New Value", "Evidence"],
                        title="Parameter Change Timeline",
                    )
                    fig_opt.update_layout(height=400)
                    st.plotly_chart(fig_opt, use_container_width=True)
                st.dataframe(opt_hist_df, use_container_width=True, hide_index=True)
            else:
                _no_data("No parameter optimization entries in audit trail.")
        else:
            _no_data("No parameter optimization events found.")
    else:
        _no_data("Audit trail table not found.")

    # --- Regime-Agent Sharpe heatmap ---
    st.subheader("Regime-Agent Sharpe Heatmap")
    if _table_exists(tables, "agent_portfolio_snapshots") and _table_exists(tables, "agent_signals"):
        sig_tbl = tables["agent_signals"]
        snap_tbl = tables["agent_portfolio_snapshots"]

        # Fetch all agent performance + regime data
        sig_df = _query_df(
            engine,
            select(sig_tbl.c.agent_id, sig_tbl.c.payload, sig_tbl.c.confidence, sig_tbl.c.timestamp)
            .where(sig_tbl.c.payload.is_not(None))
            .order_by(sig_tbl.c.timestamp)
            .limit(2000),
        )

        if not sig_df.empty:
            # Build regime x agent matrix
            regime_agent: dict[str, dict[str, list[float]]] = {}
            for _, row in sig_df.iterrows():
                payload = _safe_json_loads(row.get("payload"))
                regime = payload.get("regime") or payload.get("market_regime")
                sharpe_est = payload.get("sharpe") or payload.get("sharpe_ratio")
                agent = row["agent_id"]
                if regime and sharpe_est is not None:
                    regime_agent.setdefault(regime, {}).setdefault(agent, []).append(float(sharpe_est))

            if regime_agent:
                regimes = sorted(regime_agent.keys())
                all_agents = sorted({a for r in regime_agent.values() for a in r})
                matrix = []
                for regime in regimes:
                    row_data: list[float] = []
                    for agent in all_agents:
                        vals = regime_agent.get(regime, {}).get(agent, [])
                        row_data.append(sum(vals) / len(vals) if vals else 0)
                    matrix.append(row_data)

                matrix_df = pd.DataFrame(matrix, index=regimes, columns=all_agents)
                fig_regime = px.imshow(
                    matrix_df,
                    labels=dict(x="Agent", y="Regime", color="Sharpe"),
                    color_continuous_scale="RdYlGn",
                    text_auto=".2f",
                    title="Sharpe Ratio by Regime and Agent",
                )
                fig_regime.update_layout(height=500)
                st.plotly_chart(fig_regime, use_container_width=True)
            else:
                _no_data("No regime-Sharpe data found in signal payloads.")
        else:
            _no_data("No signal data available for regime heatmap.")
    else:
        _no_data("Required tables not found for regime-agent analysis.")

    # --- Attribution breakdown ---
    st.subheader("Attribution Breakdown per Agent")
    if _table_exists(tables, "agent_portfolio_snapshots"):
        tbl = tables["agent_portfolio_snapshots"]
        attr_df = _query_df(
            engine,
            select(tbl.c.agent_id, tbl.c.total_value, tbl.c.invested, tbl.c.unrealised_pnl)
            .order_by(desc(tbl.c.timestamp))
            .limit(100),
        )
        if not attr_df.empty:
            # Latest snapshot per agent
            attr_latest = attr_df.groupby("agent_id").first().reset_index()
            if "unrealised_pnl" in attr_latest.columns:
                fig_attr = px.bar(
                    attr_latest,
                    x="agent_id",
                    y="unrealised_pnl",
                    color="unrealised_pnl",
                    color_continuous_scale="RdYlGn",
                    title="Unrealised P&L Attribution by Agent",
                    labels={"agent_id": "Agent", "unrealised_pnl": "Unrealised P&L (INR)"},
                )
                fig_attr.update_layout(height=400)
                st.plotly_chart(fig_attr, use_container_width=True)
        else:
            _no_data("No agent portfolio snapshots for attribution.")
    else:
        _no_data("Agent portfolio snapshots table not found.")

    # --- Auto-tune change log ---
    st.subheader("Auto-Tune Change Log")
    if _table_exists(tables, "audit_trail"):
        tbl = tables["audit_trail"]
        tune_df = _query_df(
            engine,
            select(tbl)
            .where(
                tbl.c.action.in_(
                    [
                        "AUTO_TUNE",
                        "auto_tune",
                        "PARAM_CHANGE",
                        "param_change",
                        "parameter_update",
                        "self_tune",
                    ]
                )
            )
            .order_by(desc(tbl.c.timestamp))
            .limit(30),
        )
        if not tune_df.empty:
            display_cols = [
                c for c in ["timestamp", "actor", "action", "resource", "detail"]
                if c in tune_df.columns
            ]
            st.dataframe(tune_df[display_cols], use_container_width=True, hide_index=True, height=300)
        else:
            _no_data("No auto-tune events logged.")
    else:
        _no_data("Audit trail table not found.")


# ===========================================================================
# Tab 7 -- Safety Dashboard
# ===========================================================================


def _render_tab_safety(engine: Engine, tables: dict[str, Table]) -> None:
    """Tab 7: Kill switch, daily P&L gauge, utilization, agent status, audit log."""
    st.header("Safety Dashboard")

    # --- Kill switch big indicator ---
    st.subheader("Kill Switch Status")
    kill_active = _get_kill_switch_status(engine, tables)

    if kill_active is True:
        st.markdown(
            '<div style="background-color: #ff1744; color: white; padding: 2rem; '
            'border-radius: 1rem; text-align: center;">'
            '<h1 style="margin: 0; font-size: 3rem;">KILL SWITCH ACTIVE</h1>'
            '<p style="margin: 0.5rem 0 0 0;">All trading halted. Manual reset required.</p>'
            "</div>",
            unsafe_allow_html=True,
        )
    elif kill_active is False:
        st.markdown(
            '<div style="background-color: #00c853; color: white; padding: 2rem; '
            'border-radius: 1rem; text-align: center;">'
            '<h1 style="margin: 0; font-size: 3rem;">SYSTEM NOMINAL</h1>'
            '<p style="margin: 0.5rem 0 0 0;">Kill switch inactive. Trading enabled.</p>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background-color: #9e9e9e; color: white; padding: 2rem; '
            'border-radius: 1rem; text-align: center;">'
            '<h1 style="margin: 0; font-size: 3rem;">STATUS UNKNOWN</h1>'
            '<p style="margin: 0.5rem 0 0 0;">No audit trail data available.</p>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacer

    # --- Daily P&L vs limit gauge ---
    st.subheader("Daily P&L vs Limit")
    daily_loss_limit_pct: float = 0.03  # 3% of capital from config defaults
    daily_loss_limit_inr: float = INITIAL_CAPITAL * daily_loss_limit_pct

    daily_pnl: float = 0.0
    if _table_exists(tables, "portfolio_snapshots"):
        tbl = tables["portfolio_snapshots"]
        snap = _query_df(
            engine,
            select(tbl.c.unrealised_pnl, tbl.c.realised_pnl, tbl.c.total_value)
            .order_by(desc(tbl.c.timestamp))
            .limit(1),
        )
        if not snap.empty:
            daily_pnl = float(snap.iloc[0].get("unrealised_pnl") or 0) + float(
                snap.iloc[0].get("realised_pnl") or 0
            )

    col1, col2 = st.columns(2)
    with col1:
        # P&L gauge
        pnl_pct_of_limit = abs(daily_pnl) / daily_loss_limit_inr * 100 if daily_loss_limit_inr else 0
        gauge_color = "#00c853" if daily_pnl >= 0 else ("#ff9800" if pnl_pct_of_limit < 80 else "#ff1744")

        fig_pnl_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=daily_pnl,
                number={"prefix": "Rs ", "valueformat": ",.0f"},
                delta={"reference": 0, "valueformat": ",.0f"},
                title={"text": "Daily P&L"},
                gauge={
                    "axis": {"range": [-daily_loss_limit_inr, daily_loss_limit_inr]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [-daily_loss_limit_inr, -daily_loss_limit_inr * 0.8], "color": "#ffcdd2"},
                        {"range": [-daily_loss_limit_inr * 0.8, 0], "color": "#fff9c4"},
                        {"range": [0, daily_loss_limit_inr], "color": "#c8e6c9"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": -daily_loss_limit_inr,
                    },
                },
            )
        )
        fig_pnl_gauge.update_layout(height=350)
        st.plotly_chart(fig_pnl_gauge, use_container_width=True)

    with col2:
        st.metric("Daily P&L", _format_inr(daily_pnl))
        st.metric("Loss Limit", _format_inr(daily_loss_limit_inr))
        st.metric(
            "Utilization",
            f"{pnl_pct_of_limit:.1f}%",
            delta=f"{'DANGER' if pnl_pct_of_limit > 80 else 'OK'}",
            delta_color="inverse" if pnl_pct_of_limit > 80 else "normal",
        )

    # --- Position utilization bars ---
    st.subheader("Position Utilization")
    if _table_exists(tables, "positions"):
        pos_tbl = tables["positions"]
        subq = (
            select(
                pos_tbl,
                func.row_number()
                .over(partition_by=pos_tbl.c.symbol, order_by=desc(pos_tbl.c.timestamp))
                .label("rn"),
            ).subquery()
        )
        pos_df = _query_df(engine, select(subq).where(subq.c.rn == 1))

        if not pos_df.empty and "current_price" in pos_df.columns:
            max_per_stock = INITIAL_CAPITAL * 0.05  # 5%
            max_per_sector = INITIAL_CAPITAL * 0.25  # 25%
            max_deployed = INITIAL_CAPITAL * 0.80  # 80%

            # Per-stock utilization
            pos_df["notional"] = pos_df["quantity"].astype(float) * pos_df["current_price"].astype(float)
            pos_df["stock_util_pct"] = pos_df["notional"] / max_per_stock * 100

            fig_stock_util = px.bar(
                pos_df,
                x="symbol",
                y="stock_util_pct",
                title="Per-Stock Utilization (% of 5% limit)",
                labels={"symbol": "Stock", "stock_util_pct": "Utilization %"},
                color="stock_util_pct",
                color_continuous_scale=["#00c853", "#ffc107", "#ff1744"],
            )
            fig_stock_util.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Limit")
            fig_stock_util.update_layout(height=350)
            st.plotly_chart(fig_stock_util, use_container_width=True)

            # Total deployment
            total_deployed = pos_df["notional"].sum()
            deploy_pct = total_deployed / max_deployed * 100

            st.progress(
                min(deploy_pct / 100, 1.0),
                text=f"Total Deployment: {_format_inr(total_deployed)} / {_format_inr(max_deployed)} ({deploy_pct:.1f}%)",
            )
        else:
            _no_data("No position data with prices for utilization calculation.")
    else:
        _no_data("Positions table not found.")

    # --- Agent promotion status ---
    st.subheader("Agent Promotion Status")
    if _table_exists(tables, "audit_trail"):
        tbl = tables["audit_trail"]
        promo_df = _query_df(
            engine,
            select(tbl.c.actor, tbl.c.action, tbl.c.detail, tbl.c.timestamp)
            .where(
                tbl.c.action.in_(
                    [
                        "AGENT_PROMOTED",
                        "AGENT_DEMOTED",
                        "agent_promoted",
                        "agent_demoted",
                        "status_change",
                        "STATUS_CHANGE",
                    ]
                )
            )
            .order_by(desc(tbl.c.timestamp))
            .limit(50),
        )

        # Build latest status per agent
        agent_statuses: dict[str, dict[str, Any]] = {}
        if not promo_df.empty:
            for _, row in promo_df.iterrows():
                agent = row["actor"]
                if agent not in agent_statuses:
                    detail = _safe_json_loads(row.get("detail"))
                    status = detail.get("new_status") or detail.get("status") or row.get("action", "")
                    agent_statuses[agent] = {
                        "Agent": agent,
                        "Status": status.upper(),
                        "Last Changed": row["timestamp"],
                    }

        if agent_statuses:
            status_df = pd.DataFrame(list(agent_statuses.values()))
            status_colors = {
                "LIVE": "background-color: #00c853; color: white",
                "PAPER": "background-color: #ffc107; color: black",
                "BACKTEST": "background-color: #2196f3; color: white",
                "DEMOTED": "background-color: #ff1744; color: white",
            }
            st.dataframe(
                status_df.style.map(
                    lambda v: status_colors.get(str(v).upper(), ""),
                    subset=["Status"],
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            # Show default statuses for known agents
            default_status = pd.DataFrame(
                [{"Agent": a, "Status": "BACKTEST", "Last Changed": "N/A"} for a in AGENT_NAMES]
            )
            st.dataframe(default_status, use_container_width=True, hide_index=True)
    else:
        _no_data("Audit trail table not found.")

    # --- Audit log ---
    st.subheader("Audit Log (Last 50 Entries)")
    if _table_exists(tables, "audit_trail"):
        tbl = tables["audit_trail"]
        audit_df = _query_df(
            engine,
            select(tbl).order_by(desc(tbl.c.timestamp)).limit(50),
        )
        if not audit_df.empty:
            display_cols = [
                c for c in ["timestamp", "actor", "action", "resource", "detail", "severity"]
                if c in audit_df.columns
            ]

            severity_colors: dict[str, str] = {
                "critical": "background-color: #ff1744; color: white",
                "error": "background-color: #ff5252; color: white",
                "warning": "background-color: #ffc107; color: black",
                "info": "",
            }

            st.dataframe(
                audit_df[display_cols].style.map(
                    lambda v: severity_colors.get(str(v).lower(), ""),
                    subset=["severity"] if "severity" in display_cols else [],
                ),
                use_container_width=True,
                hide_index=True,
                height=400,
            )
        else:
            _no_data("No audit trail entries.")
    else:
        _no_data("Audit trail table not found.")


# ===========================================================================
# Tab 8 -- System Performance
# ===========================================================================


def _render_tab_system_performance(engine: Engine, tables: dict[str, Table]) -> None:
    """Tab 8: Latency stats, cache hit rates, API call counts, uptime."""
    st.header("System Performance")

    # --- Latency p50/p95/p99 bar chart ---
    st.subheader("Pipeline Latency (p50 / p95 / p99)")
    if _table_exists(tables, "latency_logs"):
        tbl = tables["latency_logs"]
        lat_df = _query_df(
            engine,
            select(tbl.c.stage, tbl.c.latency_ns).order_by(desc(tbl.c.timestamp)).limit(10000),
        )
        if not lat_df.empty:
            lat_df["latency_ms"] = lat_df["latency_ns"].astype(float) / 1_000_000

            # Compute percentiles per stage
            perc_rows: list[dict[str, Any]] = []
            for stage, grp in lat_df.groupby("stage"):
                values = grp["latency_ms"].sort_values()
                n = len(values)
                if n == 0:
                    continue
                p50 = values.quantile(0.50)
                p95 = values.quantile(0.95)
                p99 = values.quantile(0.99)
                perc_rows.append({"Stage": stage, "p50 (ms)": p50, "p95 (ms)": p95, "p99 (ms)": p99})

            if perc_rows:
                perc_df = pd.DataFrame(perc_rows)
                fig_lat = go.Figure()
                fig_lat.add_trace(
                    go.Bar(
                        x=perc_df["Stage"],
                        y=perc_df["p50 (ms)"],
                        name="p50",
                        marker_color="#4caf50",
                    )
                )
                fig_lat.add_trace(
                    go.Bar(
                        x=perc_df["Stage"],
                        y=perc_df["p95 (ms)"],
                        name="p95",
                        marker_color="#ff9800",
                    )
                )
                fig_lat.add_trace(
                    go.Bar(
                        x=perc_df["Stage"],
                        y=perc_df["p99 (ms)"],
                        name="p99",
                        marker_color="#f44336",
                    )
                )
                fig_lat.update_layout(
                    title="Latency Percentiles by Pipeline Stage",
                    barmode="group",
                    xaxis_title="Stage",
                    yaxis_title="Latency (ms)",
                    height=450,
                )
                st.plotly_chart(fig_lat, use_container_width=True)

                # Also show the table
                st.dataframe(
                    perc_df.style.format({"p50 (ms)": "{:.2f}", "p95 (ms)": "{:.2f}", "p99 (ms)": "{:.2f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                _no_data("No latency measurements to compute percentiles.")
        else:
            _no_data("No latency logs recorded.")
    else:
        _no_data("Latency logs table not found.")

    # --- Cache hit rates pie chart ---
    st.subheader("Cache Hit Rates")
    if _table_exists(tables, "audit_trail"):
        tbl = tables["audit_trail"]
        cache_df = _query_df(
            engine,
            select(tbl.c.detail)
            .where(tbl.c.action.in_(["CACHE_HIT", "CACHE_MISS", "cache_hit", "cache_miss"]))
            .order_by(desc(tbl.c.timestamp))
            .limit(1000),
        )
        if not cache_df.empty:
            cache_levels: dict[str, int] = {"L0 (Memory)": 0, "L1 (Disk)": 0, "L2 (SQLite)": 0, "L3 (Remote)": 0}
            for _, row in cache_df.iterrows():
                detail = _safe_json_loads(row.get("detail"))
                level = detail.get("level") or detail.get("cache_level", "L0")
                level_str = str(level)
                if "0" in level_str or "memory" in level_str.lower():
                    cache_levels["L0 (Memory)"] += 1
                elif "1" in level_str or "disk" in level_str.lower():
                    cache_levels["L1 (Disk)"] += 1
                elif "2" in level_str or "sqlite" in level_str.lower():
                    cache_levels["L2 (SQLite)"] += 1
                else:
                    cache_levels["L3 (Remote)"] += 1

            if sum(cache_levels.values()) > 0:
                fig_cache = px.pie(
                    names=list(cache_levels.keys()),
                    values=list(cache_levels.values()),
                    title="Cache Hit Distribution by Level",
                    color_discrete_sequence=["#4caf50", "#2196f3", "#ff9800", "#9c27b0"],
                    hole=0.4,
                )
                fig_cache.update_layout(height=400)
                st.plotly_chart(fig_cache, use_container_width=True)
            else:
                _no_data("No cache hit/miss events found.")
        else:
            _no_data("No cache events in audit trail.")

        # --- API call counts ---
        st.subheader("API Call Counts")
        api_df = _query_df(
            engine,
            select(tbl.c.resource, func.count().label("call_count"))
            .where(tbl.c.action.in_(["API_CALL", "api_call", "api_request"]))
            .group_by(tbl.c.resource)
            .order_by(desc("call_count"))
            .limit(20),
        )
        if not api_df.empty:
            fig_api = px.bar(
                api_df,
                x="resource",
                y="call_count",
                title="API Call Counts by Endpoint",
                labels={"resource": "API Endpoint", "call_count": "Calls"},
                color="call_count",
                color_continuous_scale="Blues",
            )
            fig_api.update_layout(height=400)
            st.plotly_chart(fig_api, use_container_width=True)
        else:
            # Fallback: show latency logs by stage as a proxy for API calls
            if _table_exists(tables, "latency_logs"):
                lat_tbl = tables["latency_logs"]
                stage_counts = _query_df(
                    engine,
                    select(lat_tbl.c.stage, func.count().label("call_count"))
                    .group_by(lat_tbl.c.stage)
                    .order_by(desc("call_count"))
                    .limit(20),
                )
                if not stage_counts.empty:
                    fig_api = px.bar(
                        stage_counts,
                        x="stage",
                        y="call_count",
                        title="Pipeline Stage Invocations (from latency logs)",
                        labels={"stage": "Stage", "call_count": "Invocations"},
                        color="call_count",
                        color_continuous_scale="Blues",
                    )
                    fig_api.update_layout(height=400)
                    st.plotly_chart(fig_api, use_container_width=True)
                else:
                    _no_data("No API call data or latency stage data.")
            else:
                _no_data("No API call events in audit trail.")
    else:
        _no_data("Audit trail table not found.")

    # --- Uptime and error count ---
    st.subheader("Uptime & Error Summary")

    col1, col2, col3, col4 = st.columns(4)

    # Total run time (approximate from first to last audit entry)
    if _table_exists(tables, "audit_trail"):
        tbl = tables["audit_trail"]
        first_ts = _query_scalar(engine, select(func.min(tbl.c.timestamp)))
        last_ts = _query_scalar(engine, select(func.max(tbl.c.timestamp)))

        if first_ts and last_ts:
            if isinstance(first_ts, str):
                try:
                    first_ts = datetime.fromisoformat(first_ts)
                except ValueError:
                    first_ts = None
            if isinstance(last_ts, str):
                try:
                    last_ts = datetime.fromisoformat(last_ts)
                except ValueError:
                    last_ts = None

            if first_ts and last_ts:
                uptime = last_ts - first_ts
                col1.metric("System Uptime", str(uptime).split(".")[0])
            else:
                col1.metric("System Uptime", "N/A")
        else:
            col1.metric("System Uptime", "N/A")

        # Error count
        error_count = _query_scalar(
            engine,
            select(func.count())
            .select_from(tbl)
            .where(tbl.c.severity.in_(["error", "critical", "ERROR", "CRITICAL"])),
        )
        col2.metric("Total Errors", int(error_count) if error_count else 0)

        # Warning count
        warn_count = _query_scalar(
            engine,
            select(func.count())
            .select_from(tbl)
            .where(tbl.c.severity.in_(["warning", "WARNING"])),
        )
        col3.metric("Warnings", int(warn_count) if warn_count else 0)

        # Total events
        total_events = _query_scalar(
            engine,
            select(func.count()).select_from(tbl),
        )
        col4.metric("Total Events", int(total_events) if total_events else 0)
    else:
        col1.metric("System Uptime", "N/A")
        col2.metric("Total Errors", "N/A")
        col3.metric("Warnings", "N/A")
        col4.metric("Total Events", "N/A")


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    """Entry point for the AlphaCouncil Streamlit dashboard."""
    engine = _get_engine()
    tables = _reflect_tables(engine)

    _render_sidebar(engine, tables)

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "The Council",
            "Portfolio Battle",
            "Live Positions",
            "Agent Deep Dive",
            "Market Analysis",
            "Research Lab",
            "Safety Dashboard",
            "System Performance",
        ]
    )

    with tab1:
        _render_tab_council(engine, tables)

    with tab2:
        _render_tab_portfolio_battle(engine, tables)

    with tab3:
        _render_tab_live_positions(engine, tables)

    with tab4:
        _render_tab_agent_deep_dive(engine, tables)

    with tab5:
        _render_tab_market_analysis(engine, tables)

    with tab6:
        _render_tab_research_lab(engine, tables)

    with tab7:
        _render_tab_safety(engine, tables)

    with tab8:
        _render_tab_system_performance(engine, tables)


if __name__ == "__main__":
    main()
