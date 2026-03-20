"""AlphaCouncil Advisor — Premium Investment Dashboard.

Award-winning glassmorphism UI with company name autocomplete search,
per-horizon buy ratings, and interactive technical charts.

Launch::
    streamlit run alphacouncil/dashboard/advisor_app.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Final

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
from sqlalchemy import text

# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------

def run_async(coro):  # noqa: ANN001,ANN201
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)

# ---------------------------------------------------------------------------
# Imports (lazy, resilient)
# ---------------------------------------------------------------------------

_ADVISOR_OK = False
_SCREENER_OK = False

try:
    from alphacouncil.advisor.engine import InvestmentAdvisor
    _ADVISOR_OK = True
except ImportError:
    InvestmentAdvisor = None  # type: ignore[assignment,misc]

try:
    from alphacouncil.advisor.screener import StockScreener
    _SCREENER_OK = True
except ImportError:
    StockScreener = None  # type: ignore[assignment,misc]

try:
    from alphacouncil.advisor.report import ReportGenerator
except ImportError:
    ReportGenerator = None  # type: ignore[assignment,misc]

from alphacouncil.advisor.models import (
    AdvisorAction, InvestmentHorizon, RiskAppetite,
    ScreenerFilter, ScreenerResult, StockRecommendation,
)
from alphacouncil.advisor.universes import (
    COMPANY_NAMES, TICKER_TO_NAME,
    get_sector, get_universe, list_universes, search_stocks, get_company_name,
)
from alphacouncil.core.cache_manager import TieredCache
from alphacouncil.core.database import init_db, metadata

logger = logging.getLogger(__name__)

DB_URL: Final[str] = "sqlite:///data/alphacouncil.db"

# ---------------------------------------------------------------------------
# Action styling
# ---------------------------------------------------------------------------

ACTION_GRADIENT: dict[str, str] = {
    "STRONG_BUY":  "linear-gradient(135deg, #00c853, #00e676)",
    "BUY":         "linear-gradient(135deg, #26a69a, #4caf50)",
    "HOLD":        "linear-gradient(135deg, #f57c00, #ffa726)",
    "SELL":        "linear-gradient(135deg, #e64a19, #ff5722)",
    "STRONG_SELL": "linear-gradient(135deg, #c62828, #ff1744)",
}

ACTION_COLOR: dict[str, str] = {
    "STRONG_BUY": "#00e676", "BUY": "#4caf50",
    "HOLD": "#ffa726", "SELL": "#ff5722", "STRONG_SELL": "#ff1744",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AlphaCouncil Advisor",
    page_icon="https://em-content.zobj.net/source/apple/391/chart-increasing_1f4c8.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — glassmorphism + dark premium theme
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ---- root vars ---- */
:root {
    --bg-primary: #0a1628;
    --bg-card: rgba(255,255,255,0.04);
    --border-card: rgba(255,255,255,0.08);
    --accent: #00d4ff;
    --accent2: #7c4dff;
    --text-primary: #e8eaf6;
    --text-secondary: #90a4ae;
    --green: #00e676;
    --red: #ff5252;
    --yellow: #ffa726;
}

/* ---- glass card ---- */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,212,255,0.08);
}

/* ---- action badge ---- */
.action-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.95rem;
    color: #fff;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ---- horizon rating card ---- */
.horizon-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.horizon-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-secondary);
    margin-bottom: 8px;
}
.horizon-card .action-pill {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.85rem;
    color: #fff;
    margin: 8px 0;
}
.horizon-card .conviction-num {
    font-size: 2rem;
    font-weight: 800;
    margin: 4px 0;
}
.horizon-card .target {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 4px;
}
.horizon-card .expected-return {
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 2px;
}

/* ---- section title ---- */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    padding-bottom: 8px;
    margin-bottom: 16px;
    border-bottom: 3px solid var(--accent);
    display: inline-block;
}

/* ---- hero header ---- */
.hero-header {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,77,255,0.08));
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 24px;
}
.hero-header .company-name {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0;
}
.hero-header .ticker {
    font-size: 1rem;
    color: var(--text-secondary);
}
.hero-header .price-big {
    font-size: 2.4rem;
    font-weight: 800;
}

/* ---- price ladder ---- */
.price-ladder {
    position: relative;
    padding: 16px 0;
}
.price-level {
    display: flex;
    align-items: center;
    padding: 8px 16px;
    margin: 4px 0;
    border-radius: 8px;
    font-size: 0.9rem;
}
.price-level .level-label { flex: 1; font-weight: 600; }
.price-level .level-price { font-weight: 700; font-size: 1rem; }
.price-level .level-rr { font-size: 0.8rem; color: var(--text-secondary); margin-left: 12px; }

/* ---- verdict card ---- */
.verdict-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-card);
    border-radius: 16px;
    padding: 20px;
    height: 100%;
}
.verdict-card h4 {
    margin: 0 0 12px;
    font-size: 1rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1px;
}
.verdict-card .stat-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.85rem;
}
.verdict-card .stat-label { color: var(--text-secondary); }
.verdict-card .stat-value { font-weight: 600; }

/* ---- metric glass ---- */
.metric-glass {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-glass .metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-secondary);
}
.metric-glass .metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 4px 0;
}
.metric-glass .metric-delta {
    font-size: 0.85rem;
    font-weight: 600;
}

/* ---- reasoning box ---- */
.reasoning-box {
    background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(124,77,255,0.05));
    border-left: 4px solid var(--accent);
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ---- regime badge ---- */
.regime-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1rem;
    color: #fff;
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

@st.cache_resource
def _init_db():
    return init_db(DB_URL)

@st.cache_resource
def _cache():
    return TieredCache()

@st.cache_resource
def _advisor():
    if InvestmentAdvisor is None:
        return None
    try:
        return InvestmentAdvisor(cache=_cache())
    except Exception:
        try:
            return InvestmentAdvisor()
        except Exception:
            return None

@st.cache_resource
def _screener():
    if StockScreener is None:
        return None
    try:
        return StockScreener(_advisor())
    except Exception:
        return None

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _db():
    return _init_db()

def _query(sql: str, params: dict | None = None) -> pd.DataFrame:
    try:
        with _db().connect() as conn:
            r = conn.execute(text(sql), params or {})
            rows = r.fetchall()
            return pd.DataFrame(rows, columns=list(r.keys())) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _exec(sql: str, params: dict | None = None) -> None:
    try:
        with _db().begin() as conn:
            conn.execute(text(sql), params or {})
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_price(v: float, cur: str = "INR") -> str:
    if cur == "INR":
        if abs(v) >= 1_00_00_000:
            return f"\u20b9{v / 1_00_00_000:,.2f} Cr"
        if abs(v) >= 1_00_000:
            return f"\u20b9{v / 1_00_000:,.2f} L"
        return f"\u20b9{v:,.2f}"
    return f"${v:,.2f}"

def _action_html(action: str, size: str = "0.95rem") -> str:
    grad = ACTION_GRADIENT.get(action, "linear-gradient(135deg,#666,#888)")
    label = action.replace("_", " ")
    return f'<span class="action-badge" style="background:{grad};font-size:{size}">{label}</span>'

def _conviction_ring(value: int, size: int = 80) -> str:
    """SVG circular progress ring."""
    r = size // 2 - 4
    c = math.pi * 2 * r
    pct = value / 100
    offset = c * (1 - pct)
    color = ACTION_COLOR.get("BUY" if value >= 60 else ("HOLD" if value >= 40 else "SELL"), "#888")
    return f'''<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <circle cx="{size//2}" cy="{size//2}" r="{r}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="6"/>
      <circle cx="{size//2}" cy="{size//2}" r="{r}" fill="none" stroke="{color}" stroke-width="6"
        stroke-dasharray="{c:.1f}" stroke-dashoffset="{offset:.1f}"
        stroke-linecap="round" transform="rotate(-90 {size//2} {size//2})"/>
      <text x="{size//2}" y="{size//2 + 5}" text-anchor="middle" font-size="16" font-weight="800" fill="white">{value}%</text>
    </svg>'''

def _fetch_yf(ticker: str, period: str = "6mo"):
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar():
    st.sidebar.markdown(
        '<h2 style="background:linear-gradient(135deg,#00d4ff,#7c4dff);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'font-weight:900;">AlphaCouncil</h2>'
        '<p style="color:#90a4ae;margin-top:-10px;">AI Investment Advisor</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"_Last updated: {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")

# ===========================================================================
# Tab 1: Stock Analyzer
# ===========================================================================

def _build_search_options() -> list[str]:
    """Build sorted search options: 'Company Name  (TICKER)'."""
    return sorted([f"{name}  ({ticker})" for name, ticker in COMPANY_NAMES.items()])

def _tab_analyzer():
    st.markdown('<div class="section-title">Stock Analyzer</div>', unsafe_allow_html=True)

    # --- Search with autocomplete ---
    all_options = _build_search_options()

    col_search, col_or, col_manual = st.columns([4, 1, 3])
    with col_search:
        selected = st.selectbox(
            "Search by company name or ticker",
            options=["Type to search..."] + all_options,
            index=0,
            key="company_search",
        )
    with col_or:
        st.markdown('<div style="text-align:center;padding-top:28px;color:#90a4ae;">or</div>', unsafe_allow_html=True)
    with col_manual:
        manual_ticker = st.text_input("Enter any ticker directly", value="", key="manual_ticker", placeholder="e.g. RELIANCE.NS, AAPL")

    # Determine which ticker to analyze
    ticker = ""
    if selected and selected != "Type to search...":
        # Parse ticker from "Company Name  (TICKER)"
        ticker = selected.split("(")[-1].rstrip(") ")
    if manual_ticker.strip():
        ticker = manual_ticker.strip().upper()

    analyze_btn = st.button("Analyze", type="primary", use_container_width=True, key="analyze_btn")

    if not analyze_btn or not ticker:
        st.info("Search for a company above, or type any ticker/company name directly. "
                "Indian stocks work with or without the .NS suffix (e.g. MOTHERSON or MOTHERSON.NS).")
        return

    # --- Run analysis ---
    advisor = _advisor()
    rec: StockRecommendation | None = None

    if advisor is not None:
        with st.spinner(f"Analyzing {ticker} ..."):
            try:
                rec = run_async(advisor.analyze(ticker))
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    price_df = _fetch_yf(ticker, "1y")
    if price_df is None or price_df.empty:
        st.warning(f"No price data for **{ticker}**.")
        return
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in price_df.columns:
            price_df[c] = pd.to_numeric(price_df[c], errors="coerce")

    current_price = float(price_df["Close"].iloc[-1])

    if rec is None:
        st.subheader(ticker)
        st.metric("Last Close", f"{current_price:,.2f}")
        _render_chart(price_df, ticker, None)
        return

    # --- Hero Header ---
    daily_chg = 0.0
    if len(price_df) >= 2:
        prev = float(price_df["Close"].iloc[-2])
        if prev > 0:
            daily_chg = (current_price - prev) / prev * 100

    chg_color = "var(--green)" if daily_chg >= 0 else "var(--red)"
    chg_sign = "+" if daily_chg >= 0 else ""

    st.markdown(f'''
    <div class="hero-header">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;">
        <div>
          <p class="company-name">{rec.name or ticker}</p>
          <p class="ticker">{ticker} &middot; {rec.exchange}</p>
        </div>
        <div style="text-align:right;">
          <div class="price-big">{_fmt_price(rec.current_price, rec.currency)}</div>
          <div style="color:{chg_color};font-weight:700;font-size:1.1rem;">{chg_sign}{daily_chg:.2f}%</div>
        </div>
        <div>{_action_html(rec.action.value, "1.1rem")}</div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    # --- Horizon Rating Cards (replacing risk level) ---
    st.markdown('<div class="section-title">Buy Rating by Horizon</div>', unsafe_allow_html=True)

    h_cols = st.columns(3)
    horizons = rec.horizon_ratings if rec.horizon_ratings else []

    horizon_labels = {"SHORT_TERM": "Short Term", "MID_TERM": "Mid Term", "LONG_TERM": "Long Term"}
    horizon_sub = {"SHORT_TERM": "1-4 Weeks", "MID_TERM": "1-6 Months", "LONG_TERM": "6-24 Months"}

    for i, h in enumerate(horizons[:3]):
        with h_cols[i]:
            grad = ACTION_GRADIENT.get(h.action.value, "linear-gradient(135deg,#666,#888)")
            ac = ACTION_COLOR.get(h.action.value, "#888")
            label = horizon_labels.get(h.horizon.value, h.horizon.value)
            sub = horizon_sub.get(h.horizon.value, "")
            target_str = f"{_fmt_price(h.target_price, rec.currency)}" if h.target_price else "—"
            ret_str = ""
            if h.expected_return_pct is not None:
                ret_sign = "+" if h.expected_return_pct >= 0 else ""
                ret_color = "var(--green)" if h.expected_return_pct >= 0 else "var(--red)"
                ret_str = f'<div class="expected-return" style="color:{ret_color}">{ret_sign}{h.expected_return_pct:.1f}%</div>'

            st.markdown(f'''
            <div class="horizon-card">
              <div class="label">{label} <span style="opacity:0.5">({sub})</span></div>
              <div class="action-pill" style="background:{grad}">{h.action.value.replace("_", " ")}</div>
              <div class="conviction-num" style="color:{ac}">{h.conviction}%</div>
              <div class="target">Target: {target_str}</div>
              {ret_str}
              <div style="font-size:0.8rem;color:var(--text-secondary);margin-top:8px;">{h.reasoning}</div>
            </div>
            ''', unsafe_allow_html=True)

    # If no horizon ratings were computed, show a fallback
    if len(horizons) == 0:
        st.info("Horizon ratings not available for this stock.")

    # --- Technical Chart ---
    _render_chart(price_df, ticker, rec)

    # --- Verdict Cards ---
    st.markdown('<div class="section-title">Analysis Breakdown</div>', unsafe_allow_html=True)
    vc1, vc2, vc3 = st.columns(3)

    with vc1:
        t = rec.technical
        st.markdown(f'''
        <div class="verdict-card">
          <h4>Technical</h4>
          <div class="stat-row"><span class="stat-label">Trend</span><span class="stat-value">{t.trend}</span></div>
          <div class="stat-row"><span class="stat-label">RSI (14)</span><span class="stat-value">{t.rsi:.0f} ({t.rsi_signal})</span></div>
          <div class="stat-row"><span class="stat-label">MACD</span><span class="stat-value">{t.macd_signal}</span></div>
          <div class="stat-row"><span class="stat-label">MA Align</span><span class="stat-value">{t.ma_alignment}</span></div>
          <div class="stat-row"><span class="stat-label">ADX</span><span class="stat-value">{t.adx:.0f} ({t.adx_signal})</span></div>
          <div class="stat-row"><span class="stat-label">Volume</span><span class="stat-value">{t.volume_signal}</span></div>
          <div class="stat-row"><span class="stat-label">Breakout</span><span class="stat-value">{"Yes" if t.breakout else "No"}</span></div>
          <div style="margin-top:12px;font-size:0.85rem;color:var(--text-secondary);font-style:italic;">{t.summary}</div>
        </div>
        ''', unsafe_allow_html=True)

    with vc2:
        f = rec.fundamental
        pe_s = f"{f.pe_ratio:.1f}" if f.pe_ratio else "N/A"
        roe_s = f"{f.roe:.1%}" if f.roe else "N/A"
        rg_s = f"{f.revenue_growth:.1%}" if f.revenue_growth else "N/A"
        de_s = f"{f.debt_to_equity:.1f}" if f.debt_to_equity else "N/A"
        mos_s = f"{f.margin_of_safety:.1%}" if f.margin_of_safety is not None else "N/A"
        st.markdown(f'''
        <div class="verdict-card">
          <h4>Fundamental</h4>
          <div class="stat-row"><span class="stat-label">Valuation</span><span class="stat-value">{f.valuation.value.replace("_"," ")}</span></div>
          <div class="stat-row"><span class="stat-label">P/E</span><span class="stat-value">{pe_s}</span></div>
          <div class="stat-row"><span class="stat-label">ROE</span><span class="stat-value">{roe_s}</span></div>
          <div class="stat-row"><span class="stat-label">Rev Growth</span><span class="stat-value">{rg_s}</span></div>
          <div class="stat-row"><span class="stat-label">Debt/Equity</span><span class="stat-value">{de_s}</span></div>
          <div class="stat-row"><span class="stat-label">Health</span><span class="stat-value">{f.financial_health}</span></div>
          <div class="stat-row"><span class="stat-label">Margin of Safety</span><span class="stat-value">{mos_s}</span></div>
          <div class="stat-row"><span class="stat-label">Growth Score</span><span class="stat-value">{f.growth_quality_score:.0f}/100</span></div>
          <div style="margin-top:12px;font-size:0.85rem;color:var(--text-secondary);font-style:italic;">{f.summary}</div>
        </div>
        ''', unsafe_allow_html=True)

    with vc3:
        s = rec.sentiment
        kw_str = ", ".join(s.top_keywords[:4]) if s.top_keywords else "—"
        st.markdown(f'''
        <div class="verdict-card">
          <h4>Sentiment</h4>
          <div class="stat-row"><span class="stat-label">Score</span><span class="stat-value">{s.score:.2f} ({s.signal})</span></div>
          <div class="stat-row"><span class="stat-label">Articles</span><span class="stat-value">{s.article_count}</span></div>
          <div class="stat-row"><span class="stat-label">Social Buzz</span><span class="stat-value">{s.social_buzz}</span></div>
          <div class="stat-row"><span class="stat-label">Momentum</span><span class="stat-value">{s.trend:.2f}</span></div>
          <div class="stat-row"><span class="stat-label">Keywords</span><span class="stat-value">{kw_str}</span></div>
          <div style="margin-top:12px;font-size:0.85rem;color:var(--text-secondary);font-style:italic;">{s.summary}</div>
        </div>
        ''', unsafe_allow_html=True)

    # --- Entry/Exit Price Ladder ---
    st.markdown('<div class="section-title">Entry / Exit Levels</div>', unsafe_allow_html=True)
    lv = rec.levels

    levels_list = []
    if lv.target_long_term:
        levels_list.append(("Target (Long)", lv.target_long_term, "#69f0ae", f"R:R {lv.risk_reward_long:.1f}" if lv.risk_reward_long else ""))
    if lv.target_mid_term:
        levels_list.append(("Target (Mid)", lv.target_mid_term, "#00e676", f"R:R {lv.risk_reward_mid:.1f}" if lv.risk_reward_mid else ""))
    if lv.target_short_term:
        levels_list.append(("Target (Short)", lv.target_short_term, "#4caf50", f"R:R {lv.risk_reward_short:.1f}" if lv.risk_reward_short else ""))
    levels_list.append(("Entry High", lv.entry_zone_high, "#42a5f5", ""))
    levels_list.append(("Current Price", rec.current_price, "#ffffff", ""))
    levels_list.append(("Entry Low", lv.entry_zone_low, "#42a5f5", ""))
    levels_list.append(("Stop Loss", lv.stop_loss, "#ff5252", ""))

    ladder_html = '<div class="price-ladder">'
    for label, price, color, rr in levels_list:
        bg = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)" if color.startswith("#") else "rgba(255,255,255,0.04)"
        ladder_html += f'''
        <div class="price-level" style="background:{bg};border-left:4px solid {color};">
          <span class="level-label" style="color:{color}">{label}</span>
          <span class="level-price">{_fmt_price(price, rec.currency)}</span>
          <span class="level-rr">{rr}</span>
        </div>'''
    ladder_html += '</div>'
    st.markdown(ladder_html, unsafe_allow_html=True)

    # --- Reasoning ---
    st.markdown('<div class="section-title">Analysis Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="reasoning-box">{rec.reasoning}</div>', unsafe_allow_html=True)

    # --- Price Prediction ---
    _render_price_prediction(ticker, rec, price_df)

    # --- Latest News ---
    st.markdown('<div class="section-title">Latest News</div>', unsafe_allow_html=True)
    _render_stock_news(ticker)

    # Persist
    try:
        _exec(
            "INSERT INTO advisor_recommendations (timestamp,ticker,action,horizon,conviction,reasoning,current_price,currency) VALUES (:ts,:t,:a,:h,:c,:r,:p,:cu)",
            {"ts": rec.timestamp.isoformat(), "t": rec.ticker, "a": rec.action.value,
             "h": rec.horizon.value, "c": rec.conviction, "r": rec.reasoning,
             "p": rec.current_price, "cu": rec.currency},
        )
    except Exception:
        pass


def _render_price_prediction(ticker: str, rec: StockRecommendation, price_df: pd.DataFrame) -> None:
    """Show ensemble price predictions with confidence cone chart."""
    st.markdown('<div class="section-title">Price Prediction</div>', unsafe_allow_html=True)

    try:
        from alphacouncil.advisor.predictor import PricePredictor
        predictor = PricePredictor()
        prediction = run_async(predictor.predict(ticker, df=price_df))
    except Exception as e:
        st.warning(f"Price prediction unavailable: {e}")
        return

    if not prediction or not prediction.predictions:
        st.info("Not enough data to generate predictions.")
        return

    # --- Prediction summary cards ---
    horizon_labels = {7: "1W", 14: "2W", 30: "1M", 60: "2M", 90: "3M"}
    cols = st.columns(len(prediction.predictions))

    for i, pt in enumerate(prediction.predictions):
        label = horizon_labels.get(pt.days_ahead, f"{pt.days_ahead}D")
        color = "var(--green)" if pt.change_pct >= 0 else "var(--red)"
        sign = "+" if pt.change_pct >= 0 else ""
        cur = rec.currency

        with cols[i]:
            st.markdown(f'''
            <div class="horizon-card">
              <div class="label">{label}</div>
              <div style="font-size:1.4rem;font-weight:800;margin:6px 0;">{_fmt_price(pt.predicted_price, cur)}</div>
              <div class="expected-return" style="color:{color};font-size:1rem;">{sign}{pt.change_pct:.1f}%</div>
              <div class="target" style="font-size:0.75rem;">
                {_fmt_price(pt.low_bound, cur)} — {_fmt_price(pt.high_bound, cur)}
              </div>
            </div>
            ''', unsafe_allow_html=True)

    # --- Confidence badge ---
    conf = prediction.model_confidence
    conf_color = "#00e676" if conf >= 0.7 else ("#ffa726" if conf >= 0.4 else "#ff5252")
    st.markdown(
        f'<div style="text-align:center;margin:12px 0;">'
        f'<span style="background:{conf_color};color:#000;padding:4px 16px;border-radius:20px;'
        f'font-weight:700;font-size:0.85rem;">Model Confidence: {conf:.0%}</span></div>',
        unsafe_allow_html=True,
    )

    # --- Prediction chart: historical + forecast cone ---
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _ms

    close = price_df["Close"].astype(float)
    fig = _ms(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
              row_heights=[0.75, 0.25])

    # Historical candlestick (last 6 months)
    fig.add_trace(_go.Candlestick(
        x=price_df.index, open=price_df["Open"], high=price_df["High"],
        low=price_df["Low"], close=price_df["Close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Build forecast dates and values
    last_date = price_df.index[-1]
    forecast_dates = [last_date]
    forecast_prices = [prediction.current_price]
    forecast_low = [prediction.current_price]
    forecast_high = [prediction.current_price]

    for pt in prediction.predictions:
        fwd_date = last_date + pd.Timedelta(days=pt.days_ahead)
        forecast_dates.append(fwd_date)
        forecast_prices.append(pt.predicted_price)
        forecast_low.append(pt.low_bound)
        forecast_high.append(pt.high_bound)

    # Confidence cone (shaded area)
    fig.add_trace(_go.Scatter(
        x=forecast_dates, y=forecast_high, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(_go.Scatter(
        x=forecast_dates, y=forecast_low, mode="lines",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(0,212,255,0.12)", name="Confidence Cone",
    ), row=1, col=1)

    # Forecast line (dashed)
    fig.add_trace(_go.Scatter(
        x=forecast_dates, y=forecast_prices, mode="lines+markers",
        line=dict(color="#00d4ff", width=2.5, dash="dash"),
        marker=dict(size=8, color="#00d4ff", symbol="circle"),
        name="Prediction",
    ), row=1, col=1)

    # Volume bars (historical only)
    vol_colors = [
        "#26a69a" if price_df["Close"].iloc[i] >= price_df["Open"].iloc[i] else "#ef5350"
        for i in range(len(price_df))
    ]
    fig.add_trace(_go.Bar(
        x=price_df.index, y=price_df["Volume"], name="Volume", marker_color=vol_colors,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=600,
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(t=30, b=30, l=50, r=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(title_text="Vol", row=2, col=1, gridcolor="rgba(255,255,255,0.04)")
    for r in [1, 2]:
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", row=r, col=1)

    st.plotly_chart(fig, width="stretch")

    # --- Disclaimer ---
    st.markdown(
        '<div class="glass-card" style="font-size:0.8rem;color:var(--text-secondary);padding:16px;">'
        'Predictions are based on an ensemble of statistical models '
        '(linear regression, exponential smoothing, and technical projection). '
        'These are probabilistic estimates, not guarantees. Past performance does not '
        'predict future results. Always do your own research.</div>',
        unsafe_allow_html=True,
    )


def _render_stock_news(ticker: str) -> None:
    """Fetch and display recent news for a stock via yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        raw_news = stock.news or []
    except Exception:
        raw_news = []

    if not raw_news:
        st.markdown(
            '<div class="glass-card" style="text-align:center;color:var(--text-secondary);">'
            'No recent news available for this stock.</div>',
            unsafe_allow_html=True,
        )
        return

    # Parse news items — yfinance may nest data under "content" key
    parsed: list[dict] = []
    for item in raw_news[:8]:
        content = item.get("content", item)  # newer yfinance nests under "content"
        title = content.get("title", "")
        if not title:
            continue

        provider = content.get("provider", {})
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else content.get("publisher", "")

        # Get link
        click_url = content.get("clickThroughUrl", content.get("canonicalUrl", {}))
        if isinstance(click_url, dict):
            link = click_url.get("url", "#")
        else:
            link = content.get("link", "#")

        # Publication time
        pub_str = ""
        pub_date = content.get("pubDate", content.get("displayTime", ""))
        if pub_date:
            try:
                from datetime import datetime as _dt
                if isinstance(pub_date, str):
                    dt = _dt.fromisoformat(pub_date.replace("Z", "+00:00"))
                    pub_str = dt.strftime("%b %d, %H:%M")
                elif isinstance(pub_date, (int, float)):
                    pub_str = _dt.fromtimestamp(pub_date).strftime("%b %d, %H:%M")
            except Exception:
                pass

        # Thumbnail
        thumb_url = ""
        thumbnail = content.get("thumbnail", {})
        if isinstance(thumbnail, dict):
            resolutions = thumbnail.get("resolutions", [])
            if resolutions:
                thumb_url = resolutions[0].get("url", "")

        summary = content.get("summary", "")[:150]
        if summary:
            summary = summary.rstrip(".") + "..."

        parsed.append({"title": title, "publisher": publisher, "link": link,
                        "pub_str": pub_str, "thumb_url": thumb_url, "summary": summary})

    if not parsed:
        st.markdown(
            '<div class="glass-card" style="text-align:center;color:var(--text-secondary);">'
            'No recent news available for this stock.</div>',
            unsafe_allow_html=True,
        )
        return

    for i in range(0, len(parsed), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(parsed):
                break
            n = parsed[idx]
            thumb_html = ""
            if n["thumb_url"]:
                thumb_html = (
                    f'<img src="{n["thumb_url"]}" style="width:100%;border-radius:8px;'
                    f'margin-bottom:8px;max-height:120px;object-fit:cover;" '
                    f'onerror="this.style.display=\'none\'">'
                )
            summary_html = f'<div style="font-size:0.8rem;color:var(--text-secondary);margin-top:4px;">{n["summary"]}</div>' if n["summary"] else ""

            with col:
                st.markdown(f'''
                <div class="glass-card" style="padding:16px;min-height:140px;">
                  {thumb_html}
                  <div style="font-size:0.9rem;font-weight:600;line-height:1.3;margin-bottom:6px;">
                    <a href="{n["link"]}" target="_blank" style="color:var(--text-primary);text-decoration:none;">{n["title"]}</a>
                  </div>
                  {summary_html}
                  <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:6px;">
                    {n["publisher"]}{(" &middot; " + n["pub_str"]) if n["pub_str"] else ""}
                  </div>
                </div>
                ''', unsafe_allow_html=True)


def _render_chart(df: pd.DataFrame, ticker: str, rec: StockRecommendation | None):
    """Candlestick + RSI + MACD + Volume with dark theme."""
    st.markdown('<div class="section-title">Technical Chart</div>', unsafe_allow_html=True)

    close = df["Close"].astype(float)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    bb_mid = sma20
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.50, 0.15, 0.15, 0.20], subplot_titles=("", "", "", ""))

    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                  name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma20, name="SMA 20", line=dict(color="#ffa726", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA 50", line=dict(color="#42a5f5", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma200, name="SMA 200", line=dict(color="#ab47bc", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name="BB Upper", line=dict(color="rgba(150,150,150,0.3)", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name="BB Lower", line=dict(color="rgba(150,150,150,0.3)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(150,150,150,0.03)"), row=1, col=1)

    if rec is not None:
        lv = rec.levels
        fig.add_hline(y=lv.stop_loss, line_dash="dash", line_color="#ff5252", annotation_text="Stop Loss", row=1, col=1)
        fig.add_hline(y=lv.entry_zone_low, line_dash="dash", line_color="#42a5f5", annotation_text="Entry", row=1, col=1)
        if lv.target_short_term:
            fig.add_hline(y=lv.target_short_term, line_dash="dot", line_color="#00e676", annotation_text="Target ST", row=1, col=1)
        if lv.target_mid_term:
            fig.add_hline(y=lv.target_mid_term, line_dash="dot", line_color="#69f0ae", annotation_text="Target MT", row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color="#7c4dff", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ff5252", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00e676", row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD", line=dict(color="#42a5f5", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_signal, name="Signal", line=dict(color="#ef5350", width=1)), row=3, col=1)
    mc = ["#26a69a" if v >= 0 else "#ef5350" for v in macd_hist.fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=macd_hist, name="Hist", marker_color=mc), row=3, col=1)

    vc = ["#26a69a" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef5350" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Vol", marker_color=vc), row=4, col=1)

    fig.update_layout(
        template="plotly_dark", height=900,
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(t=40, b=30, l=50, r=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100], gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(title_text="MACD", row=3, col=1, gridcolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(title_text="Vol", row=4, col=1, gridcolor="rgba(255,255,255,0.04)")
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", row=i, col=1)

    st.plotly_chart(fig, width="stretch")


# ===========================================================================
# Tab 2: Screener
# ===========================================================================

PROFILES: Final[list[str]] = ["growth_picks", "value_dips", "momentum_breakouts", "dividend_steady", "turnaround_candidates", "custom"]

def _tab_screener():
    st.markdown('<div class="section-title">Stock Screener</div>', unsafe_allow_html=True)

    screener = _screener()
    if screener is None:
        st.warning("Screener module not available.")
        return

    universes = list_universes()
    c1, c2 = st.columns(2)
    with c1:
        uni = st.selectbox("Universe", list(universes.keys()), format_func=lambda n: f"{n} ({universes[n]} stocks)", key="scr_uni")
    with c2:
        profile = st.selectbox("Profile", PROFILES, key="scr_prof")

    custom_filter = None
    if profile == "custom":
        with st.expander("Custom Filters", expanded=True):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                rsi_min = st.number_input("RSI Min", 0.0, 100.0, 30.0, 5.0)
                rsi_max = st.number_input("RSI Max", 0.0, 100.0, 70.0, 5.0)
                above_sma = st.checkbox("Above SMA 200")
            with fc2:
                macd_b = st.checkbox("MACD Bullish")
                adx_min = st.number_input("ADX Min", 0.0, 100.0, 20.0, 5.0)
                min_rg = st.number_input("Min Revenue Growth %", 0.0, step=5.0)
            with fc3:
                max_pe = st.number_input("Max P/E", 0.0, value=50.0, step=5.0)
                min_roe = st.number_input("Min ROE %", 0.0, value=10.0, step=5.0)
                pos_fcf = st.checkbox("Positive FCF")
            custom_filter = ScreenerFilter(
                rsi_min=rsi_min, rsi_max=rsi_max,
                above_sma_200=above_sma or None, macd_bullish=macd_b or None,
                adx_min=adx_min if adx_min > 0 else None,
                min_revenue_growth=min_rg if min_rg > 0 else None,
                max_pe=max_pe if max_pe > 0 else None,
                min_roe=min_roe if min_roe > 0 else None,
                positive_fcf=pos_fcf or None,
            )

    if st.button("Screen Stocks", type="primary", use_container_width=True, key="screen_go"):
        tickers = get_universe(uni)
        with st.spinner("Screening ..."):
            try:
                if profile == "custom" and custom_filter:
                    result = run_async(screener.screen(tickers, profile="custom", custom_filter=custom_filter))
                else:
                    result = run_async(screener.screen(tickers, profile=profile))
            except Exception as e:
                st.error(f"Screening failed: {e}")
                return

        if not result.results:
            st.warning("No stocks matched.")
            return

        st.success(f"**{len(result.results)}** stocks passed from {result.total_screened} screened.")

        rows = []
        for it in result.results:
            rows.append({
                "Ticker": it.ticker, "Score": f"{it.composite_score:.0f}",
                "Tech": f"{it.technical_score:.0f}", "Fund": f"{it.fundamental_score:.0f}",
                "Action": it.action.value, "Conviction": f"{it.conviction}%",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("Select a universe and profile, then click **Screen Stocks**.")


# ===========================================================================
# Tab 3: Market Pulse
# ===========================================================================

def _tab_market_pulse():
    st.markdown('<div class="section-title">Market Pulse</div>', unsafe_allow_html=True)

    with st.spinner("Loading markets ..."):
        data = _fetch_markets()

    if not data:
        st.warning("Could not load market data.")
        return

    # Indices
    indices = [
        ("Nifty 50", data.get("nifty50_level", 0), data.get("nifty50_change_pct", 0)),
        ("Sensex", data.get("sensex_level", 0), data.get("sensex_change_pct", 0)),
        ("S&P 500", data.get("sp500_level", 0), data.get("sp500_change_pct", 0)),
        ("Nasdaq", data.get("nasdaq_level", 0), data.get("nasdaq_change_pct", 0)),
    ]

    cols = st.columns(4)
    for i, (name, level, chg) in enumerate(indices):
        with cols[i]:
            chg_color = "var(--green)" if chg >= 0 else "var(--red)"
            sign = "+" if chg >= 0 else ""
            st.markdown(f'''
            <div class="metric-glass">
              <div class="metric-label">{name}</div>
              <div class="metric-value">{level:,.1f}</div>
              <div class="metric-delta" style="color:{chg_color}">{sign}{chg:.2f}%</div>
            </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # VIX
    v1, v2 = st.columns(2)
    for col, (label, val, max_r) in zip([v1, v2], [("India VIX", data.get("india_vix", 0), 50), ("US VIX", data.get("us_vix", 0), 60)]):
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=val, title={"text": label},
                gauge={"axis": {"range": [0, max_r]}, "bar": {"color": "#00d4ff"},
                       "steps": [{"range": [0, 15], "color": "rgba(0,230,118,0.15)"},
                                 {"range": [15, 25], "color": "rgba(255,167,38,0.15)"},
                                 {"range": [25, max_r], "color": "rgba(255,82,82,0.15)"}]}))
            fig.update_layout(template="plotly_dark", height=280, margin=dict(t=50, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

    # Global
    g1, g2, g3 = st.columns(3)
    for col, (name, val, prefix) in zip([g1, g2, g3], [("DXY", data.get("dxy", 0), ""), ("Gold (USD)", data.get("gold_price", 0), "$"), ("Brent (USD)", data.get("brent_crude", 0), "$")]):
        with col:
            st.markdown(f'''
            <div class="metric-glass">
              <div class="metric-label">{name}</div>
              <div class="metric-value">{prefix}{val:,.2f}</div>
            </div>''', unsafe_allow_html=True)

    # Regime
    regime = data.get("india_regime", "NEUTRAL")
    reg_colors = {"BULL_LOW_VOL": "#00c853", "BULL_HIGH_VOL": "#69f0ae", "BEAR_LOW_VOL": "#ff5252", "BEAR_HIGH_VOL": "#ff1744", "SIDEWAYS": "#ffa726"}
    rc = reg_colors.get(regime, "#888")
    st.markdown(f'<br><div class="regime-badge" style="background:{rc}">India Regime: {regime.replace("_"," ")}</div>', unsafe_allow_html=True)


def _fetch_markets() -> dict[str, Any] | None:
    try:
        import yfinance as yf
        tmap = {
            "^NSEI": ("nifty50_level", "nifty50_change_pct"),
            "^BSESN": ("sensex_level", "sensex_change_pct"),
            "^GSPC": ("sp500_level", "sp500_change_pct"),
            "^IXIC": ("nasdaq_level", "nasdaq_change_pct"),
            "^INDIAVIX": ("india_vix", None), "^VIX": ("us_vix", None),
            "DX-Y.NYB": ("dxy", None), "GC=F": ("gold_price", None), "BZ=F": ("brent_crude", None),
        }
        data: dict[str, Any] = {"india_regime": "NEUTRAL"}
        raw = yf.download(list(tmap.keys()), period="5d", progress=False, auto_adjust=True, group_by="ticker")
        for sym, (lk, ck) in tmap.items():
            try:
                sd = raw[sym] if sym in raw.columns.get_level_values(0) else None
                if sd is None or sd.empty:
                    continue
                if isinstance(sd.columns, pd.MultiIndex):
                    sd.columns = [c[0] for c in sd.columns]
                cl = sd["Close"].dropna()
                if len(cl) < 1:
                    continue
                data[lk] = float(cl.iloc[-1])
                if ck and len(cl) >= 2:
                    prev = float(cl.iloc[-2])
                    if prev > 0:
                        data[ck] = (float(cl.iloc[-1]) - prev) / prev * 100
            except Exception:
                continue
        nc = data.get("nifty50_change_pct", 0)
        vx = data.get("india_vix", 15)
        if nc > 0.5 and vx < 18:
            data["india_regime"] = "BULL_LOW_VOL"
        elif nc > 0.5:
            data["india_regime"] = "BULL_HIGH_VOL"
        elif nc < -0.5 and vx < 18:
            data["india_regime"] = "BEAR_LOW_VOL"
        elif nc < -0.5:
            data["india_regime"] = "BEAR_HIGH_VOL"
        else:
            data["india_regime"] = "SIDEWAYS"
        return data
    except Exception:
        return None


# ===========================================================================
# Tab 4: Portfolio Builder
# ===========================================================================

def _tab_portfolio():
    st.markdown('<div class="section-title">Portfolio Builder</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        capital = st.number_input("Capital (INR)", 10_000.0, 10_00_00_000.0, 5_00_000.0, 50_000.0, format="%.0f", key="pb_cap")
    with c2:
        risk = st.selectbox("Risk Appetite", ["Conservative", "Moderate", "Aggressive"], index=1, key="pb_risk")
    with c3:
        markets = st.multiselect("Markets", ["India Large Cap", "India Mid Cap", "US Top 30", "US Tech"], default=["India Large Cap"], key="pb_mkt")
    with c4:
        horizon = st.selectbox("Horizon", ["Short Term", "Mid Term", "Long Term"], index=1, key="pb_hor")

    if not st.button("Build Portfolio", type="primary", use_container_width=True, key="pb_go"):
        st.info("Configure and click **Build Portfolio**.")
        return

    mkt_map = {"India Large Cap": "india_nifty50", "India Mid Cap": "india_midcap_growth", "US Top 30": "us_sp500_top30", "US Tech": "us_tech_growth"}
    tickers = []
    for m in markets:
        if m in mkt_map:
            try:
                tickers.extend(get_universe(mkt_map[m]))
            except Exception:
                pass

    if not tickers:
        st.warning("No tickers for selected markets.")
        return

    # --- Smart allocation using screener scores + inverse volatility ---
    max_n = {"Conservative": 8, "Moderate": 12, "Aggressive": 18}.get(risk, 12)
    cash_pct = {"Conservative": 0.30, "Moderate": 0.20, "Aggressive": 0.10}.get(risk, 0.20)
    profile_map = {"Conservative": "dividend_steady", "Moderate": "growth_picks", "Aggressive": "momentum_breakouts"}
    profile = profile_map.get(risk, "growth_picks")

    screener = _screener()
    scored_tickers: list[dict] = []

    with st.spinner("Screening and scoring stocks..."):
        if screener is not None:
            try:
                result = run_async(screener.screen(tickers, profile=profile))
                for item in result.results[:max_n]:
                    scored_tickers.append({
                        "ticker": item.ticker,
                        "name": item.name or get_company_name(item.ticker),
                        "score": item.composite_score,
                        "action": item.action.value,
                        "conviction": item.conviction,
                        "sector": get_sector(item.ticker),
                    })
            except Exception as e:
                st.warning(f"Screener failed ({e}), falling back to volatility-based allocation.")

    # Fallback: if screener didn't produce results, use raw tickers
    if not scored_tickers:
        for t in tickers[:max_n]:
            scored_tickers.append({
                "ticker": t, "name": get_company_name(t), "score": 50.0,
                "action": "HOLD", "conviction": 50, "sector": get_sector(t),
            })

    # Fetch volatility for inverse-vol weighting
    import yfinance as _yf
    vols: dict[str, float] = {}
    with st.spinner("Computing optimal weights (inverse-volatility)..."):
        try:
            sym_list = [s["ticker"] for s in scored_tickers]
            hist = _yf.download(sym_list, period="3mo", progress=False, auto_adjust=True, group_by="ticker")
            for s in scored_tickers:
                t = s["ticker"]
                try:
                    if len(sym_list) == 1:
                        cl = hist["Close"].dropna()
                    else:
                        sub = hist[t] if t in hist.columns.get_level_values(0) else None
                        if sub is None:
                            vols[t] = 0.25
                            continue
                        if isinstance(sub.columns, pd.MultiIndex):
                            sub.columns = [c[0] for c in sub.columns]
                        cl = sub["Close"].dropna()
                    if len(cl) > 10:
                        ret = cl.pct_change().dropna()
                        vol = float(ret.std() * (252 ** 0.5))  # annualized
                        vols[t] = max(vol, 0.05)  # floor at 5%
                    else:
                        vols[t] = 0.25
                except Exception:
                    vols[t] = 0.25
        except Exception:
            for s in scored_tickers:
                vols[s["ticker"]] = 0.25

    # Compute weights: inverse-vol * score tilt
    raw_weights: dict[str, float] = {}
    for s in scored_tickers:
        t = s["ticker"]
        inv_vol = 1.0 / vols.get(t, 0.25)
        score_mult = s["score"] / 50.0  # >50 gets more, <50 gets less
        raw_weights[t] = inv_vol * score_mult

    total_raw = sum(raw_weights.values())
    equity_budget = 1.0 - cash_pct

    # Normalize and cap at 15% per stock
    max_single = 0.15
    for _ in range(5):  # iterate to redistribute excess
        total_raw = sum(raw_weights.values())
        if total_raw <= 0:
            break
        excess = 0.0
        for t in raw_weights:
            w = (raw_weights[t] / total_raw) * equity_budget
            if w > max_single:
                excess += w - max_single
                raw_weights[t] = max_single * total_raw / equity_budget
        if excess < 0.001:
            break

    total_raw = sum(raw_weights.values())
    weights = {t: (v / total_raw * equity_budget) if total_raw > 0 else equity_budget / len(raw_weights) for t, v in raw_weights.items()}

    # Build display
    alloc_names, alloc_vals, rows = [], [], []
    for s in scored_tickers:
        t = s["ticker"]
        w = weights.get(t, 0)
        alloc_names.append(s["name"])
        alloc_vals.append(w * 100)
        rows.append({
            "Ticker": t,
            "Name": s["name"],
            "Weight": f"{w*100:.1f}%",
            "Amount": _fmt_price(capital * w),
            "Score": f"{s['score']:.0f}",
            "Rating": s["action"],
            "Vol (ann)": f"{vols.get(t, 0):.0%}",
            "Sector": s["sector"],
        })

    alloc_names.append("Cash Reserve")
    alloc_vals.append(cash_pct * 100)
    rows.append({"Ticker": "CASH", "Name": "Cash Reserve", "Weight": f"{cash_pct*100:.0f}%",
                 "Amount": _fmt_price(capital * cash_pct), "Score": "—", "Rating": "—", "Vol (ann)": "—", "Sector": "—"})

    # Pie chart
    fig = px.pie(names=alloc_names, values=alloc_vals, hole=0.5, title="Portfolio Allocation")
    fig.update_layout(template="plotly_dark", height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    # Allocation table
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Sector concentration check
    sector_weights: dict[str, float] = {}
    for s in scored_tickers:
        sec = s["sector"]
        sector_weights[sec] = sector_weights.get(sec, 0) + weights.get(s["ticker"], 0)

    if sector_weights:
        st.markdown('<div class="section-title">Sector Breakdown</div>', unsafe_allow_html=True)
        sec_fig = px.bar(
            x=list(sector_weights.keys()), y=[v * 100 for v in sector_weights.values()],
            labels={"x": "Sector", "y": "Weight %"}, color=[v * 100 for v in sector_weights.values()],
            color_continuous_scale="Viridis",
        )
        sec_fig.update_layout(template="plotly_dark", height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        sec_fig.add_hline(y=30, line_dash="dash", line_color="#ff5252", annotation_text="30% concentration limit")
        st.plotly_chart(sec_fig, width="stretch")

        # Warn if over-concentrated
        for sec, sw in sector_weights.items():
            if sw > 0.30:
                st.warning(f"Sector **{sec}** is at {sw:.0%} — consider diversifying. Limit is 30%.")

    # Portfolio summary metrics
    st.markdown('<div class="section-title">Portfolio Metrics</div>', unsafe_allow_html=True)
    avg_score = sum(s["score"] * weights.get(s["ticker"], 0) for s in scored_tickers) / equity_budget if equity_budget > 0 else 0
    avg_vol = sum(vols.get(s["ticker"], 0.25) * weights.get(s["ticker"], 0) for s in scored_tickers) / equity_budget if equity_budget > 0 else 0
    n_sectors = len(set(s["sector"] for s in scored_tickers))

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Stocks", len(scored_tickers))
    mc2.metric("Weighted Score", f"{avg_score:.0f}/100")
    mc3.metric("Portfolio Vol (est)", f"{avg_vol:.0%}")
    mc4.metric("Sectors", n_sectors)


# ===========================================================================
# Tab 5: Watchlist
# ===========================================================================

def _tab_watchlist():
    st.markdown('<div class="section-title">Watchlist</div>', unsafe_allow_html=True)

    engine = _db()
    metadata.create_all(engine)

    # Add
    ac1, ac2 = st.columns([4, 1])
    with ac1:
        # Searchable add
        all_opts = _build_search_options()
        new_sel = st.selectbox("Add stock to watchlist", [""] + all_opts, key="wl_add_search")
    with ac2:
        st.markdown('<div style="padding-top:28px"></div>', unsafe_allow_html=True)
        add_btn = st.button("Add", type="primary", use_container_width=True, key="wl_add_btn")

    if add_btn and new_sel:
        ticker_clean = new_sel.split("(")[-1].rstrip(") ").strip().upper()
        if ticker_clean:
            _exec("INSERT OR IGNORE INTO advisor_watchlist (ticker, added_at) VALUES (:t, :n)", {"t": ticker_clean, "n": datetime.now(timezone.utc).isoformat()})
            st.success(f"Added **{ticker_clean}**")
            st.rerun()

    refresh = st.button("Refresh All Prices", use_container_width=True, key="wl_refresh")

    wl = _query("SELECT id, ticker, last_recommendation, last_checked FROM advisor_watchlist ORDER BY added_at DESC")
    if wl.empty:
        st.info("Watchlist empty. Add stocks above.")
        return

    if refresh:
        advisor = _advisor()
        if advisor:
            prog = st.progress(0, "Refreshing...")
            for idx, row in wl.iterrows():
                prog.progress((int(idx) + 1) / len(wl), f"Analyzing {row['ticker']}...")
                try:
                    rec = run_async(advisor.analyze(row["ticker"]))
                    if rec:
                        _exec("UPDATE advisor_watchlist SET last_recommendation=:a, last_checked=:n WHERE ticker=:t",
                              {"a": rec.action.value, "n": datetime.now(timezone.utc).isoformat(), "t": row["ticker"]})
                except Exception:
                    pass
            prog.empty()
            st.rerun()

    # Display
    display = []
    for _, row in wl.iterrows():
        t = row["ticker"]
        price, chg = 0.0, 0.0
        try:
            import yfinance as yf
            d = yf.download(t, period="5d", progress=False, auto_adjust=True)
            if d is not None and not d.empty:
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = [c[0] for c in d.columns]
                cl = d["Close"].dropna()
                if len(cl) >= 1:
                    price = float(cl.iloc[-1])
                if len(cl) >= 2 and float(cl.iloc[-2]) > 0:
                    chg = (price - float(cl.iloc[-2])) / float(cl.iloc[-2]) * 100
        except Exception:
            pass

        act = row.get("last_recommendation") or "—"
        display.append({"id": row["id"], "Ticker": t, "Name": get_company_name(t),
                         "Price": f"{price:,.2f}" if price > 0 else "—",
                         "Change": f"{chg:+.2f}%" if price > 0 else "—",
                         "Rating": act, "Updated": str(row.get("last_checked") or "Never")[:16]})

    df = pd.DataFrame(display)

    def _style_chg(v: str) -> str:
        try:
            n = float(v.replace("%", "").replace("+", ""))
            return f"color:{'#00e676' if n > 0 else '#ff5252' if n < 0 else '#fff'}"
        except Exception:
            return ""

    def _style_act(v: str) -> str:
        c = ACTION_COLOR.get(v, "")
        return f"background:{c};color:#fff;font-weight:700;border-radius:4px;padding:2px 8px" if c else ""

    show = ["Ticker", "Name", "Price", "Change", "Rating", "Updated"]
    st.dataframe(df[show].style.map(_style_chg, subset=["Change"]).map(_style_act, subset=["Rating"]),
                 width="stretch", hide_index=True)

    # Remove buttons
    rm_cols = st.columns(min(len(display), 6))
    for i, r in enumerate(display):
        with rm_cols[i % min(len(display), 6)]:
            if st.button(f"Remove {r['Ticker']}", key=f"rm_{r['id']}"):
                _exec("DELETE FROM advisor_watchlist WHERE id=:id", {"id": r["id"]})
                st.rerun()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    _init_db()
    _sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Stock Analyzer", "Screener", "Market Pulse", "Portfolio Builder", "Watchlist",
    ])

    with tab1:
        _tab_analyzer()
    with tab2:
        _tab_screener()
    with tab3:
        _tab_market_pulse()
    with tab4:
        _tab_portfolio()
    with tab5:
        _tab_watchlist()


if __name__ == "__main__":
    main()
