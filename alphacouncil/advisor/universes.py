"""Pre-built stock universes for Indian and global markets."""

from __future__ import annotations

INDIA_NIFTY50: list[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "BAJFINANCE.NS",
    "LT.NS", "HCLTECH.NS", "KOTAKBANK.NS", "AXISBANK.NS", "TITAN.NS",
    "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "WIPRO.NS",
    "ULTRACEMCO.NS", "NESTLEIND.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
    "TATASTEEL.NS", "ONGC.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "TECHM.NS", "HDFCLIFE.NS", "BAJAJFINSV.NS", "SBILIFE.NS", "GRASIM.NS",
    "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS", "BPCL.NS", "BRITANNIA.NS",
    "APOLLOHOSP.NS", "EICHERMOT.NS", "INDUSINDBK.NS", "TATACONSUM.NS",
    "COALINDIA.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "BAJAJ-AUTO.NS",
    "UPL.NS", "LTIM.NS",
]

INDIA_MIDCAP_GROWTH: list[str] = [
    "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "DIXON.NS", "AFFLE.NS",
    "HAPPSTMNDS.NS", "ROUTE.NS", "CDSL.NS", "TRENT.NS", "ZOMATO.NS",
    "POLICYBZR.NS", "DMART.NS", "CLEAN.NS", "ASTRAL.NS", "DEEPAKNTR.NS",
    "ATUL.NS", "PIIND.NS", "SYNGENE.NS", "METROPOLIS.NS", "LALPATHLAB.NS",
    "IDFCFIRSTB.NS", "FEDERALBNK.NS", "MUTHOOTFIN.NS", "JUBLFOOD.NS",
    "PAGEIND.NS", "CROMPTON.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "ESCORTS.NS",
    "CUMMINSIND.NS",
]

INDIA_SMALLCAP_EMERGING: list[str] = [
    "CAMPUS.NS", "MEDPLUS.NS", "KAYNES.NS", "DOMS.NS", "JYOTHYLAB.NS",
    "ELECTCAST.NS", "FINEORG.NS", "PPLPHARMA.NS", "AMIORG.NS", "MASTEK.NS",
    "KPITTECH.NS", "BIRLASOFT.NS", "RATNAMANI.NS", "GRINDWELL.NS",
    "CERA.NS", "SUPRAJIT.NS", "GARFIBRES.NS", "CARBORUNIV.NS",
    "IIFL.NS", "NUVAMA.NS",
]

US_SP500_TOP30: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "AVGO", "CVX",
    "MRK", "ABBV", "LLY", "PEP", "KO", "COST", "WMT", "CSCO", "TMO",
    "MCD", "ABT", "ACN",
]

US_TECH_GROWTH: list[str] = [
    "NVDA", "AMD", "AVGO", "CRM", "NOW", "PANW", "CRWD", "SNOW",
    "DDOG", "NET", "ZS", "MDB", "PLTR", "COIN", "AFRM", "SQ",
    "SHOP", "TTD", "RBLX", "U",
]

GLOBAL_DIVIDEND: list[str] = [
    "JNJ", "PG", "KO", "PEP", "MMM", "ABT", "XOM", "CVX",
    "NESN.SW", "NOVN.SW", "ROG.SW", "ULVR.L", "AZN.L",
    "SHELL.L", "BP.L",
]

# ---------------------------------------------------------------------------
# Company name → ticker mapping (for autocomplete search)
# ---------------------------------------------------------------------------

COMPANY_NAMES: dict[str, str] = {
    # India - Nifty 50
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "State Bank of India": "SBIN.NS",
    "ITC": "ITC.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Larsen & Toubro": "LT.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Titan Company": "TITAN.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Wipro": "WIPRO.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Tata Steel": "TATASTEEL.NS",
    "ONGC": "ONGC.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Tech Mahindra": "TECHM.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "Grasim Industries": "GRASIM.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Dr Reddy's Laboratories": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "BPCL": "BPCL.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Coal India": "COALINDIA.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco": "HINDALCO.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "UPL": "UPL.NS",
    "LTIMindtree": "LTIM.NS",
    # India - Midcap growth
    "Persistent Systems": "PERSISTENT.NS",
    "Coforge": "COFORGE.NS",
    "Mphasis": "MPHASIS.NS",
    "Dixon Technologies": "DIXON.NS",
    "Affle India": "AFFLE.NS",
    "Happiest Minds": "HAPPSTMNDS.NS",
    "Route Mobile": "ROUTE.NS",
    "CDSL": "CDSL.NS",
    "Trent": "TRENT.NS",
    "Zomato": "ZOMATO.NS",
    "PB Fintech (PolicyBazaar)": "POLICYBZR.NS",
    "Avenue Supermarts (DMart)": "DMART.NS",
    "Clean Science": "CLEAN.NS",
    "Astral": "ASTRAL.NS",
    "Deepak Nitrite": "DEEPAKNTR.NS",
    "KPIT Technologies": "KPITTECH.NS",
    "Birlasoft": "BIRLASOFT.NS",
    # US - S&P 500 Top
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet (Google)": "GOOGL",
    "Amazon": "AMZN",
    "NVIDIA": "NVDA",
    "Meta (Facebook)": "META",
    "Tesla": "TSLA",
    "Berkshire Hathaway": "BRK-B",
    "UnitedHealth": "UNH",
    "Johnson & Johnson": "JNJ",
    "Visa": "V",
    "ExxonMobil": "XOM",
    "JPMorgan Chase": "JPM",
    "Procter & Gamble": "PG",
    "Mastercard": "MA",
    "Home Depot": "HD",
    "Broadcom": "AVGO",
    "Chevron": "CVX",
    "Merck": "MRK",
    "AbbVie": "ABBV",
    "Eli Lilly": "LLY",
    "PepsiCo": "PEP",
    "Coca-Cola": "KO",
    "Costco": "COST",
    "Walmart": "WMT",
    "Cisco": "CSCO",
    "McDonald's": "MCD",
    # US - Tech Growth
    "AMD": "AMD",
    "Salesforce": "CRM",
    "ServiceNow": "NOW",
    "Palo Alto Networks": "PANW",
    "CrowdStrike": "CRWD",
    "Snowflake": "SNOW",
    "Datadog": "DDOG",
    "Cloudflare": "NET",
    "Zscaler": "ZS",
    "MongoDB": "MDB",
    "Palantir": "PLTR",
    "Coinbase": "COIN",
    "Shopify": "SHOP",
    "The Trade Desk": "TTD",
}

# Reverse mapping: ticker -> company name
TICKER_TO_NAME: dict[str, str] = {v: k for k, v in COMPANY_NAMES.items()}

# Sector mapping for Indian stocks
INDIA_SECTOR_MAP: dict[str, str] = {
    "RELIANCE.NS": "Energy", "TCS.NS": "IT", "HDFCBANK.NS": "Banking",
    "INFY.NS": "IT", "ICICIBANK.NS": "Banking", "HINDUNILVR.NS": "FMCG",
    "BHARTIARTL.NS": "Telecom", "SBIN.NS": "Banking", "ITC.NS": "FMCG",
    "BAJFINANCE.NS": "Finance", "LT.NS": "Infrastructure", "HCLTECH.NS": "IT",
    "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking", "TITAN.NS": "Consumer",
    "ASIANPAINT.NS": "Consumer", "MARUTI.NS": "Auto", "SUNPHARMA.NS": "Pharma",
    "TATAMOTORS.NS": "Auto", "WIPRO.NS": "IT", "ULTRACEMCO.NS": "Cement",
    "NESTLEIND.NS": "FMCG", "NTPC.NS": "Power", "POWERGRID.NS": "Power",
    "M&M.NS": "Auto", "TATASTEEL.NS": "Metals", "ONGC.NS": "Energy",
    "JSWSTEEL.NS": "Metals", "ADANIENT.NS": "Conglomerate",
    "ADANIPORTS.NS": "Infrastructure", "TECHM.NS": "IT",
    "PERSISTENT.NS": "IT", "COFORGE.NS": "IT", "MPHASIS.NS": "IT",
    "DIXON.NS": "Electronics", "AFFLE.NS": "Ad Tech", "TRENT.NS": "Retail",
    "ZOMATO.NS": "Consumer Tech", "DMART.NS": "Retail", "CDSL.NS": "Finance",
}

# All available universes
_UNIVERSES: dict[str, list[str]] = {
    "india_nifty50": INDIA_NIFTY50,
    "india_midcap_growth": INDIA_MIDCAP_GROWTH,
    "india_smallcap_emerging": INDIA_SMALLCAP_EMERGING,
    "us_sp500_top30": US_SP500_TOP30,
    "us_tech_growth": US_TECH_GROWTH,
    "global_dividend": GLOBAL_DIVIDEND,
}


def get_universe(name: str) -> list[str]:
    """Retrieve a stock universe by name.

    Raises KeyError if the universe is not found.
    Available: india_nifty50, india_midcap_growth, india_smallcap_emerging,
    us_sp500_top30, us_tech_growth, global_dividend
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in _UNIVERSES:
        available = ", ".join(sorted(_UNIVERSES.keys()))
        raise KeyError(f"Unknown universe '{name}'. Available: {available}")
    return _UNIVERSES[key].copy()


def list_universes() -> dict[str, int]:
    """Return dict of {universe_name: stock_count}."""
    return {name: len(tickers) for name, tickers in _UNIVERSES.items()}


def get_sector(ticker: str) -> str:
    """Get sector for an Indian stock ticker. Returns 'Other' if not mapped."""
    return INDIA_SECTOR_MAP.get(ticker, "Other")


def search_stocks(query: str, limit: int = 10) -> list[dict[str, str]]:
    """Search stocks by company name or ticker symbol.

    Returns list of {name, ticker} dicts matching the query, sorted by relevance.
    """
    if not query or len(query) < 1:
        return []

    query_lower = query.lower()
    results: list[tuple[int, str, str]] = []  # (priority, name, ticker)

    for name, ticker in COMPANY_NAMES.items():
        name_lower = name.lower()
        ticker_lower = ticker.lower().replace(".ns", "").replace(".bo", "")

        # Exact ticker match (highest priority)
        if ticker_lower == query_lower or ticker.lower() == query_lower:
            results.append((0, name, ticker))
        # Ticker starts with query
        elif ticker_lower.startswith(query_lower):
            results.append((1, name, ticker))
        # Company name starts with query
        elif name_lower.startswith(query_lower):
            results.append((2, name, ticker))
        # Query found anywhere in company name
        elif query_lower in name_lower:
            results.append((3, name, ticker))
        # Query found in ticker
        elif query_lower in ticker_lower:
            results.append((4, name, ticker))

    # Sort by priority, then alphabetically
    results.sort(key=lambda x: (x[0], x[1]))
    return [{"name": r[1], "ticker": r[2]} for r in results[:limit]]


def get_company_name(ticker: str) -> str:
    """Get company name for a ticker. Returns the ticker itself if not found."""
    return TICKER_TO_NAME.get(ticker, ticker.split(".")[0])
