# =========================================
# TradeEdge Pro - Clean Yahoo Version
# =========================================
# pip install streamlit yfinance pandas numpy plotly lxml scipy

import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import norm


# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="TradeEdge Pro", page_icon="📈", layout="wide")

JOURNAL_FILE = "trade_journal.csv"
SCAN_DEFAULT = "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, AMD, SPY, QQQ"


# =========================
# STYLE
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: #f3f4f6;
        color: #111827;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        min-height: 108px;
    }

    .signal-box {
        padding: 28px;
        border-radius: 22px;
        text-align: center;
        color: white;
        margin: 18px 0 20px 0;
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }

    .signal-title {
        font-size: 40px;
        font-weight: 900;
        line-height: 1.1;
    }

    .signal-sub {
        font-size: 18px;
        margin-top: 8px;
    }

    .card {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .small-note {
        color: #6b7280;
        font-size: 13px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        padding: 10px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def safe_num(x, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def fmt_money(x) -> str:
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"${float(x):,.2f}"
    except Exception:
        return "N/A"

def fmt_money_short(x):
    try:
        x = float(x)

        if abs(x) >= 999_500:
            return f"${x / 1_000_000:.1f}M"

        if abs(x) >= 1_000:
            return f"${x / 1_000:.1f}K"

        return f"${x:,.0f}"

    except Exception:
        return "N/A"


def pnl_color(x):
    try:
        x = float(x)
        if x > 0:
            return "#16a34a"
        if x < 0:
            return "#dc2626"
        return "#6b7280"
    except Exception:
        return "#6b7280"


def pnl_metric_card(label, value):
    color = pnl_color(value)
    display_value = fmt_money_short(value)

    st.markdown(
        f"""
        <div style="
            background: white;
            border: 1px solid #d1d5db;
            border-radius: 16px;
            padding: 18px;
            min-height: 108px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        ">
            <div style="font-size: 16px; color: #374151; margin-bottom: 18px;">
                {label}
            </div>
            <div style="font-size: 42px; font-weight: 700; color: {color}; line-height: 1;">
                {display_value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_journal_table(df: pd.DataFrame):
    """
    Display-only formatting for journal tables.
    Keeps calculations unchanged but shows prices/P&L cleanly.
    """
    format_map = {}

    for col in ["Strike", "Entry Price", "Exit Price", "Profit/Loss", "Real POP %", "Chance ITM %", "Touch %", "Delta"]:
        if col in df.columns:
            format_map[col] = "{:.2f}"

    if "Trade Rank" in df.columns:
        format_map["Trade Rank"] = "{:.0f}"

    if "Contracts" in df.columns:
        format_map["Contracts"] = "{:.0f}"

    try:
        return df.style.format(format_map)
    except Exception:
        return df


def fmt_number(x) -> str:
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"{float(x):,.0f}"
    except Exception:
        return "N/A"


def fmt_pct(x) -> str:
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "N/A"


def clean_iv(iv) -> float:
    try:
        iv = float(iv)
        if pd.isna(iv) or iv <= 0.01:
            return 0.30
        return iv
    except Exception:
        return 0.30


def get_latest_price(df: pd.DataFrame):
    if df is None or df.empty or "Close" not in df.columns:
        return None

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return None

    return float(close.iloc[-1])


def get_price_change(df: pd.DataFrame):
    if df is None or df.empty or "Close" not in df.columns:
        return None, None

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 2:
        return None, None

    latest = float(close.iloc[-1])
    previous = float(close.iloc[-2])

    if previous == 0:
        return None, None

    change = latest - previous
    change_pct = change / previous

    return change, change_pct


def signal_color(signal: str) -> str:
    if signal == "STRONG BUY CALL":
        return "linear-gradient(135deg,#16a34a,#22c55e)"
    if signal == "BUY CALL WATCH":
        return "linear-gradient(135deg,#2563eb,#38bdf8)"
    if signal == "STRONG BUY PUT":
        return "linear-gradient(135deg,#dc2626,#ef4444)"
    if signal == "PUT WATCH":
        return "linear-gradient(135deg,#ea580c,#f97316)"
    return "linear-gradient(135deg,#6b7280,#9ca3af)"


# =========================
# S&P 500 UNIVERSE
# =========================
@st.cache_data(ttl=24 * 60 * 60)
def load_sp500_universe() -> pd.DataFrame:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp500 = tables[0].rename(columns={"Symbol": "Ticker", "Security": "Company"})

        sp500["Ticker"] = sp500["Ticker"].astype(str).str.replace(".", "-", regex=False)
        sp500["Company"] = sp500["Company"].astype(str)
        sp500["Display"] = sp500["Company"] + " (" + sp500["Ticker"] + ")"

        return sp500[["Ticker", "Company", "Display"]].dropna()

    except Exception:
        return pd.DataFrame(
            [
                {"Ticker": "AAPL", "Company": "Apple", "Display": "Apple (AAPL)"},
                {"Ticker": "MSFT", "Company": "Microsoft", "Display": "Microsoft (MSFT)"},
                {"Ticker": "NVDA", "Company": "Nvidia", "Display": "Nvidia (NVDA)"},
                {"Ticker": "TSLA", "Company": "Tesla", "Display": "Tesla (TSLA)"},
                {"Ticker": "BAC", "Company": "Bank of America", "Display": "Bank of America (BAC)"},
                {"Ticker": "SPY", "Company": "SPY ETF", "Display": "SPY ETF (SPY)"},
                {"Ticker": "QQQ", "Company": "QQQ ETF", "Display": "QQQ ETF (QQQ)"},
            ]
        )


def resolve_ticker(user_input: str, selected_display: str, sp500_df: pd.DataFrame) -> str:
    if user_input and user_input.strip():
        query = user_input.strip().lower()

        exact_ticker = sp500_df[sp500_df["Ticker"].str.lower() == query]
        if not exact_ticker.empty:
            return str(exact_ticker.iloc[0]["Ticker"]).upper()

        exact_company = sp500_df[sp500_df["Company"].str.lower() == query]
        if not exact_company.empty:
            return str(exact_company.iloc[0]["Ticker"]).upper()

        contains_company = sp500_df[
            sp500_df["Company"].str.lower().str.contains(query, na=False, regex=False)
        ]
        if not contains_company.empty:
            return str(contains_company.iloc[0]["Ticker"]).upper()

        return user_input.upper().strip().replace(".", "-")

    return selected_display.split("(")[-1].replace(")", "").strip().upper()


# =========================
# DATA LOADERS
# =========================
@st.cache_data(ttl=60)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        if interval in ["15m", "30m"] and period in ["6mo", "1y"]:
            period = "60d"

        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        df = flatten_columns(df)

        if df is None or df.empty:
            return pd.DataFrame()

        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                df[col] = np.nan

        return df.dropna(subset=["Close"])

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def load_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=120)
def get_options_expirations(ticker: str):
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []


def load_option_chain(ticker: str, expiration: str, typ: str) -> pd.DataFrame:
    try:
        chain = yf.Ticker(ticker).option_chain(expiration)
        return chain.calls.copy() if typ == "CALL" else chain.puts.copy()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_news(ticker: str):
    try:
        return yf.Ticker(ticker).news or []
    except Exception:
        return []


# =========================
# INDICATORS / SIGNALS
# =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").replace(0, np.nan).ffill().fillna(1)

    df["MA20"] = close.rolling(20, min_periods=5).mean()
    df["MA50"] = close.rolling(50, min_periods=10).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    typical_price = (high + low + close) / 3
    df["VWAP"] = ((typical_price * volume).cumsum() / volume.cumsum()).ffill()

    df["Volume_MA20"] = pd.to_numeric(df["Volume"], errors="coerce").rolling(20, min_periods=5).mean()

    return df


def calculate_signal(df: pd.DataFrame):
    df = add_indicators(df)

    clean = df.dropna(subset=["Close"])
    if clean.empty:
        raise ValueError("No valid price data found.")

    latest = clean.iloc[-1]

    price = safe_num(latest.get("Close"), 0)
    rsi = safe_num(latest.get("RSI"), 50)
    ma20 = safe_num(latest.get("MA20"), price)
    ma50 = safe_num(latest.get("MA50"), price)
    macd = safe_num(latest.get("MACD"), 0)
    macd_signal = safe_num(latest.get("MACD_SIGNAL"), 0)
    vwap = safe_num(latest.get("VWAP"), price)

    support = safe_num(df["Low"].tail(30).min(), price)
    resistance = safe_num(df["High"].tail(30).max(), price)

    entry_low = support * 1.005
    entry_high = support * 1.035
    bullish_exit = resistance * 0.985
    bearish_entry = resistance * 0.995
    bearish_exit = support * 1.015
    bearish_stop = support * 0.985

    score = 50

    if rsi < 30:
        score += 18
    elif 30 <= rsi <= 45:
        score += 10
    elif 45 < rsi <= 60:
        score += 5
    elif rsi > 70:
        score -= 15

    score += 12 if price > ma20 else -8
    score += 10 if price > ma50 else -8
    score += 10 if price > vwap else -5
    score += 12 if macd > macd_signal else -8

    vol = safe_num(latest.get("Volume"), 0)
    vol_ma = safe_num(latest.get("Volume_MA20"), 0)
    if vol_ma > 0 and vol > vol_ma:
        score += 5

    score = max(0, min(100, int(score)))

    if score >= 80:
        signal = "STRONG BUY CALL"
    elif score >= 65:
        signal = "BUY CALL WATCH"
    elif score <= 20:
        signal = "STRONG BUY PUT"
    elif score <= 35:
        signal = "PUT WATCH"
    else:
        signal = "NEUTRAL"

    return df, {
        "price": price,
        "rsi": rsi,
        "ma20": ma20,
        "ma50": ma50,
        "macd": macd,
        "macd_signal": macd_signal,
        "vwap": vwap,
        "support": support,
        "resistance": resistance,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "bullish_exit": bullish_exit,
        "bearish_entry": bearish_entry,
        "bearish_exit": bearish_exit,
        "bearish_stop": bearish_stop,
        "score": score,
        "signal": signal,
    }


# =========================
# OPTIONS ENGINE
# =========================
def calc_greeks(S: float, K: float, T: float, iv: float, typ: str):
    try:
        if S <= 0 or K <= 0 or T <= 0 or iv <= 0:
            return np.nan, np.nan, np.nan, np.nan

        r = 0.045
        d1 = (math.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
        d2 = d1 - iv * math.sqrt(T)
        pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)

        if typ == "CALL":
            delta = norm.cdf(d1)
            theta = (-S * pdf * iv / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            delta = norm.cdf(d1) - 1
            theta = (-S * pdf * iv / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365

        gamma = pdf / (S * iv * math.sqrt(T))
        vega = S * pdf * math.sqrt(T) / 100

        return delta, gamma, theta, vega

    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def calc_pop(S: float, K: float, premium: float, T: float, iv: float, typ: str):
    try:
        if S <= 0 or K <= 0 or premium <= 0 or T <= 0 or iv <= 0:
            return np.nan

        if typ == "CALL":
            breakeven = K + premium
            z = math.log(breakeven / S) / (iv * math.sqrt(T))
            return max(0, min(1, 1 - norm.cdf(z)))

        breakeven = K - premium
        if breakeven <= 0:
            return np.nan

        z = math.log(breakeven / S) / (iv * math.sqrt(T))
        return max(0, min(1, norm.cdf(z)))

    except Exception:
        return np.nan




# =========================
# ADVANCED PROBABILITY MODEL (ADDED)
# =========================
def calculate_real_pop(option_type, S, K, premium, days, iv, r=0.045):
    """
    Returns:
    - Chance ITM %: probability the option expires in the money
    - Real POP %: probability the option expires beyond break-even
    """
    try:
        option_type = str(option_type).upper().strip()

        S = float(S)
        K = float(K)
        premium = float(premium)
        days = max(float(days), 1)
        T = days / 365
        iv = float(iv)

        if iv > 3:
            iv = iv / 100

        if S <= 0 or K <= 0 or premium <= 0 or T <= 0 or iv <= 0:
            return None, None

        sigma_sqrt_t = iv * math.sqrt(T)
        if sigma_sqrt_t <= 0:
            return None, None

        d2 = (math.log(S / K) + (r - 0.5 * iv**2) * T) / sigma_sqrt_t

        if option_type == "CALL":
            breakeven = K + premium
            prob_itm = norm.cdf(d2)

            d2_be = (
                math.log(S / breakeven) + (r - 0.5 * iv**2) * T
            ) / sigma_sqrt_t

            prob_be = norm.cdf(d2_be)

        elif option_type == "PUT":
            breakeven = max(K - premium, 0.01)
            prob_itm = norm.cdf(-d2)

            d2_be = (
                math.log(S / breakeven) + (r - 0.5 * iv**2) * T
            ) / sigma_sqrt_t

            prob_be = norm.cdf(-d2_be)

        else:
            return None, None

        prob_itm = max(0, min(prob_itm, 1)) * 100
        prob_be = max(0, min(prob_be, 1)) * 100

        return round(prob_itm, 2), round(prob_be, 2)

    except Exception:
        return None, None


def calculate_probability_touch(option_type, S, K, days, iv, r=0.045):
    """
    Quant-style probability of touch using a lognormal barrier approximation,
    scaled to a practical trading estimate.

    Calls: chance price touches strike above current price.
    Puts: chance price touches strike below current price.
    """
    try:
        option_type = str(option_type).upper().strip()

        S = float(S)
        K = float(K)
        days = max(float(days), 1)
        T = days / 365
        iv = float(iv)

        if iv > 3:
            iv = iv / 100

        if S <= 0 or K <= 0 or T <= 0 or iv <= 0:
            return None

        mu = r - 0.5 * iv**2
        sigma_sqrt_t = iv * math.sqrt(T)

        if sigma_sqrt_t <= 0:
            return None

        epsilon = 1e-6

        if option_type == "CALL":
            if S > K + epsilon:
                return 99.0

            barrier = math.log(K / S)
            z1 = (-barrier + mu * T) / sigma_sqrt_t
            z2 = (-barrier - mu * T) / sigma_sqrt_t
            adjustment = math.exp((2 * mu * barrier) / (iv**2))
            raw_touch = norm.cdf(z1) + adjustment * norm.cdf(z2)

        elif option_type == "PUT":
            if S < K - epsilon:
                return 99.0

            barrier = math.log(S / K)
            z1 = (-barrier - mu * T) / sigma_sqrt_t
            z2 = (-barrier + mu * T) / sigma_sqrt_t
            adjustment = math.exp((-2 * mu * barrier) / (iv**2))
            raw_touch = norm.cdf(z1) + adjustment * norm.cdf(z2)

        else:
            return None

        # Practical scaling: ATM theoretical touch is ~100%, but short-term trading target
        # realism is closer to 70-80% under this simplified model.
        probability_touch = raw_touch * 0.75
        probability_touch = max(0, min(probability_touch, 0.99))

        return round(probability_touch * 100, 2)

    except Exception:
        return None


def score_contract(row: pd.Series, price: float) -> int:
    score = 0

    volume = safe_num(row.get("volume"), 0)
    oi = safe_num(row.get("openInterest"), 0)
    strike = safe_num(row.get("strike"), 0)
    last = safe_num(row.get("lastPrice"), 0)
    spread = row.get("Spread")
    delta = abs(safe_num(row.get("Delta"), np.nan))
    pop = row.get("POP")

    if volume >= 500:
        score += 18
    elif volume >= 100:
        score += 12
    elif volume >= 25:
        score += 6

    if oi >= 1000:
        score += 18
    elif oi >= 500:
        score += 12
    elif oi >= 100:
        score += 6

    if not pd.isna(spread):
        if spread <= 0.10:
            score += 18
        elif spread <= 0.30:
            score += 12
        elif spread <= 0.75:
            score += 6

    if price > 0 and strike > 0:
        distance = abs(strike - price) / price
        if distance <= 0.03:
            score += 18
        elif distance <= 0.07:
            score += 12
        elif distance <= 0.12:
            score += 6

    if not pd.isna(delta):
        if 0.25 <= delta <= 0.70:
            score += 18
        elif 0.15 <= delta <= 0.85:
            score += 10

    if not pd.isna(pop):
        if pop >= 0.40:
            score += 12
        elif pop >= 0.25:
            score += 8

    if last > 0:
        score += 5

    return max(0, min(100, int(score)))



def rank_trade(row):
    """
    Ranks option contracts using tradability + probability + liquidity.

    Higher = better.
    This version intentionally penalizes contracts that look attractive only
    because they are deep ITM, too expensive, too low POP, or too wide spread.
    """
    score = 0

    contract_score = safe_num(row.get("Score"), 0)
    real_pop = safe_num(row.get("Real POP %"), 0)
    touch = safe_num(row.get("Touch %"), 0)
    delta = abs(safe_num(row.get("Delta"), 0))
    cost = safe_num(row.get("Contract Cost"), 0)
    volume = safe_num(row.get("volume"), 0)
    oi = safe_num(row.get("openInterest"), 0)
    spread = safe_num(row.get("Spread"), 999)

    # Base contract score: useful, but not allowed to dominate.
    score += contract_score * 0.20

    # Real POP is the most important probability metric.
    if real_pop >= 60:
        score += 30
    elif real_pop >= 50:
        score += 20
    elif real_pop >= 40:
        score += 10
    else:
        score -= 15

    # Touch is useful, but secondary to Real POP.
    if touch >= 75:
        score += 15
    elif touch >= 60:
        score += 10
    elif touch >= 45:
        score += 5
    else:
        score -= 10

    # Delta sweet spot: avoid too deep ITM and far OTM contracts.
    if 0.30 <= delta <= 0.60:
        score += 20
    elif 0.20 <= delta <= 0.70:
        score += 10
    else:
        score -= 15

    # Cost efficiency.
    if 0 < cost <= 200:
        score += 10
    elif cost > 500:
        score -= 10

    # Liquidity.
    if volume >= 1000 and oi >= 2000:
        score += 10
    elif volume >= 200 and oi >= 500:
        score += 7
    elif volume >= 50 and oi >= 100:
        score += 4

    # Spread quality.
    if spread <= 0.10:
        score += 10
    elif spread <= 0.30:
        score += 5
    elif spread > 0.50:
        score -= 10

    return max(0, min(100, int(score)))


def is_valid_trade(row):
    """
    Quality gate for ranked option contracts.
    Filters out contracts that are too low-probability, too deep ITM/OTM,
    too wide-spread, or too illiquid.
    """
    real_pop = safe_num(row.get("Real POP %"), 0)
    touch = safe_num(row.get("Touch %"), 0)
    delta = abs(safe_num(row.get("Delta"), 0))
    spread = safe_num(row.get("Spread"), 999)
    volume = safe_num(row.get("volume"), 0)

    return (
        real_pop >= 40 and
        touch >= 55 and
        0.25 <= delta <= 0.70 and
        spread <= 0.50 and
        volume >= 25
    )


def prepare_options(df: pd.DataFrame, price: float, expiration: str, typ: str):
    df = df.copy()

    required = [
        "contractSymbol",
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
        "impliedVolatility",
    ]

    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    numeric = ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = df["volume"].fillna(0)
    df["openInterest"] = df["openInterest"].fillna(0)
    df["lastPrice"] = df["lastPrice"].fillna(0)
    df["bid"] = df["bid"].fillna(0)
    df["ask"] = df["ask"].fillna(0)
    df["IV"] = df["impliedVolatility"].apply(clean_iv)

    days = max((datetime.strptime(expiration, "%Y-%m-%d").date() - date.today()).days, 1)
    T = days / 365

    greeks = df.apply(lambda r: calc_greeks(price, r["strike"], T, r["IV"], typ), axis=1)

    df["Delta"] = [g[0] for g in greeks]
    df["Gamma"] = [g[1] for g in greeks]
    df["Theta"] = [g[2] for g in greeks]
    df["Vega"] = [g[3] for g in greeks]
    df["POP"] = df.apply(lambda r: calc_pop(price, r["strike"], r["lastPrice"], T, r["IV"], typ), axis=1)
    df["Spread"] = np.where((df["ask"] > 0) & (df["bid"] > 0), df["ask"] - df["bid"], np.nan)
    df["Contract Cost"] = df["lastPrice"] * 100
    df["Score"] = df.apply(lambda r: score_contract(r, price), axis=1)

    # Probability metrics must be calculated before Trade Rank.
    pops = df.apply(
        lambda r: calculate_real_pop(typ, price, r["strike"], r["lastPrice"], days, r["IV"]),
        axis=1,
    )

    df["Real POP %"] = [p[1] if p and p[1] is not None else None for p in pops]
    df["Chance ITM %"] = [p[0] if p and p[0] is not None else None for p in pops]

    df["Touch %"] = df.apply(
        lambda r: calculate_probability_touch(typ, price, r["strike"], days, r["IV"]),
        axis=1,
    )

    # Trade Rank must come after Spread, Contract Cost, Score, Real POP, ITM, and Touch exist.
    df["Trade Rank"] = df.apply(rank_trade, axis=1)

    return df, days


# =========================
# STRATEGY SCANNER
# =========================
STRATEGY_CHOICES = [
    "Credit Spreads",
    "Iron Condors",
    "Covered Calls",
    "Cash Secured Puts",
]


def expiration_dte(expiration: str):
    try:
        return max((datetime.strptime(expiration, "%Y-%m-%d").date() - date.today()).days, 1)
    except Exception:
        return None


def select_strategy_expirations(expirations, min_dte: int, max_dte: int, limit: int):
    selected = []

    for exp in expirations:
        dte = expiration_dte(exp)
        if dte is None:
            continue
        if min_dte <= dte <= max_dte:
            selected.append((exp, dte))

    return selected[: max(int(limit), 1)]


def option_sell_price(row) -> float:
    bid = safe_num(row.get("bid"), 0)
    return bid if bid > 0 else 0.0


def option_buy_price(row) -> float:
    ask = safe_num(row.get("ask"), 0)
    last = safe_num(row.get("lastPrice"), 0)

    if ask > 0:
        return ask
    if last > 0:
        return last
    return 0.0


def option_leg_spread(row) -> float:
    bid = safe_num(row.get("bid"), 0)
    ask = safe_num(row.get("ask"), 0)

    if bid > 0 and ask > 0 and ask >= bid:
        return ask - bid
    return np.nan


def add_trade_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["Sell Price"] = out.apply(option_sell_price, axis=1)
    out["Buy Price"] = out.apply(option_buy_price, axis=1)
    out["Leg Spread"] = out.apply(option_leg_spread, axis=1)
    return out


def chance_option_expires_otm(row) -> float:
    chance_itm = safe_num(row.get("Chance ITM %"), np.nan)

    if pd.isna(chance_itm):
        chance_itm = abs(safe_num(row.get("Delta"), 0)) * 100

    return max(0, min(100, 100 - chance_itm))


def probability_between_strikes(S: float, lower: float, upper: float, days: int, iv: float, r: float = 0.045):
    try:
        S = float(S)
        lower = float(lower)
        upper = float(upper)
        days = max(float(days), 1)
        T = days / 365
        iv = float(iv)

        if iv > 3:
            iv = iv / 100

        if S <= 0 or lower <= 0 or upper <= lower or T <= 0 or iv <= 0:
            return None

        sigma_sqrt_t = iv * math.sqrt(T)
        if sigma_sqrt_t <= 0:
            return None

        d2_lower = (math.log(S / lower) + (r - 0.5 * iv**2) * T) / sigma_sqrt_t
        d2_upper = (math.log(S / upper) + (r - 0.5 * iv**2) * T) / sigma_sqrt_t
        prob = norm.cdf(d2_lower) - norm.cdf(d2_upper)

        return round(max(0, min(prob, 1)) * 100, 2)

    except Exception:
        return None


def leg_liquidity_stats(legs):
    volumes = [safe_num(leg.get("volume"), 0) for leg in legs]
    open_interests = [safe_num(leg.get("openInterest"), 0) for leg in legs]
    spreads = [safe_num(leg.get("Leg Spread"), np.nan) for leg in legs]
    ivs = [clean_iv(leg.get("IV")) for leg in legs]

    valid_spreads = [s for s in spreads if not pd.isna(s)]

    return {
        "Min Volume": min(volumes) if volumes else 0,
        "Min Open Interest": min(open_interests) if open_interests else 0,
        "Avg Leg Spread": float(np.mean(valid_spreads)) if valid_spreads else np.nan,
        "Avg IV": float(np.mean(ivs)) if ivs else 0.30,
    }


def strategy_signal_fit(strategy: str, stock_score: float) -> int:
    if strategy in ["Bull Put Credit Spread", "Cash Secured Put"]:
        if stock_score >= 65:
            return 15
        if stock_score >= 50:
            return 8
        if stock_score <= 35:
            return -10
        return 0

    if strategy == "Bear Call Credit Spread":
        if stock_score <= 35:
            return 15
        if stock_score <= 50:
            return 8
        if stock_score >= 65:
            return -10
        return 0

    if strategy == "Iron Condor":
        if 40 <= stock_score <= 60:
            return 15
        if 35 <= stock_score <= 65:
            return 8
        if stock_score <= 20 or stock_score >= 80:
            return -10
        return 0

    if strategy == "Covered Call":
        if 50 <= stock_score <= 75:
            return 15
        if 40 <= stock_score <= 85:
            return 8
        if stock_score < 30:
            return -8
        return 0

    return 0


def score_income_strategy(row) -> int:
    score = 0

    pop = safe_num(row.get("POP %"), 0)
    return_on_risk = safe_num(row.get("Return on Risk %"), 0)
    annualized_return = safe_num(row.get("Annualized Return %"), 0)
    min_volume = safe_num(row.get("Min Volume"), 0)
    min_oi = safe_num(row.get("Min Open Interest"), 0)
    avg_spread = safe_num(row.get("Avg Leg Spread"), 999)
    stock_score = safe_num(row.get("Stock Score"), 50)
    strategy = str(row.get("Strategy", ""))

    if pop >= 80:
        score += 35
    elif pop >= 70:
        score += 30
    elif pop >= 60:
        score += 22
    elif pop >= 50:
        score += 12
    else:
        score -= 10

    if return_on_risk >= 35:
        score += 25
    elif return_on_risk >= 25:
        score += 20
    elif return_on_risk >= 15:
        score += 14
    elif return_on_risk >= 8:
        score += 8

    if annualized_return >= 100:
        score += 10
    elif annualized_return >= 50:
        score += 7
    elif annualized_return >= 25:
        score += 4

    if min_volume >= 100 and min_oi >= 500:
        score += 10
    elif min_volume >= 25 and min_oi >= 100:
        score += 7
    elif min_oi >= 50:
        score += 4

    if avg_spread <= 0.10:
        score += 10
    elif avg_spread <= 0.30:
        score += 7
    elif avg_spread <= 0.60:
        score += 3
    else:
        score -= 8

    score += strategy_signal_fit(strategy, stock_score)

    return max(0, min(100, int(score)))


def strategy_note(strategy: str, stock_signal: str, pop: float, return_on_risk: float) -> str:
    if strategy == "Iron Condor":
        return f"Neutral income setup; stock signal is {stock_signal}."
    if strategy == "Covered Call":
        return f"Income against shares; assignment possible above short call."
    if strategy == "Cash Secured Put":
        return f"Income entry setup; assignment possible below short put."
    if "Bull Put" in strategy:
        return f"Bullish/neutral credit spread; POP {pop:.1f}%, ROR {return_on_risk:.1f}%."
    if "Bear Call" in strategy:
        return f"Bearish/neutral credit spread; POP {pop:.1f}%, ROR {return_on_risk:.1f}%."
    return f"Stock signal is {stock_signal}."


def make_credit_spread_row(symbol, price, expiration, dte, short_leg, long_leg, typ, stock_signal):
    short_strike = safe_num(short_leg.get("strike"), 0)
    long_strike = safe_num(long_leg.get("strike"), 0)

    if short_strike <= 0 or long_strike <= 0:
        return None

    if typ == "PUT":
        width = short_strike - long_strike
        strategy = "Bull Put Credit Spread"
    else:
        width = long_strike - short_strike
        strategy = "Bear Call Credit Spread"

    if width <= 0:
        return None

    credit = option_sell_price(short_leg) - option_buy_price(long_leg)
    if credit <= 0.05 or credit >= width or credit / width < 0.08:
        return None

    max_profit = credit * 100
    max_loss = (width - credit) * 100
    if max_loss <= 0:
        return None

    pop = chance_option_expires_otm(short_leg)
    return_on_risk = max_profit / max_loss * 100
    annualized_return = return_on_risk * 365 / max(dte, 1)

    if typ == "PUT":
        breakeven = short_strike - credit
        short_put = short_strike
        long_put = long_strike
        short_call = np.nan
        long_call = np.nan
        legs_text = f"Sell {short_strike:g}P / Buy {long_strike:g}P"
    else:
        breakeven = short_strike + credit
        short_put = np.nan
        long_put = np.nan
        short_call = short_strike
        long_call = long_strike
        legs_text = f"Sell {short_strike:g}C / Buy {long_strike:g}C"

    liquidity = leg_liquidity_stats([short_leg, long_leg])
    stock_score = safe_num(stock_signal.get("score"), 50)

    row = {
        "Ticker": symbol,
        "Strategy": strategy,
        "Expiration": expiration,
        "DTE": dte,
        "Price": price,
        "Signal": stock_signal.get("signal", "NEUTRAL"),
        "Stock Score": stock_score,
        "Short Strike": short_strike,
        "Long Strike": long_strike,
        "Short Put": short_put,
        "Long Put": long_put,
        "Short Call": short_call,
        "Long Call": long_call,
        "Credit": credit,
        "Width": width,
        "Max Profit": max_profit,
        "Max Loss": max_loss,
        "Breakeven": breakeven,
        "POP %": pop,
        "Return on Risk %": return_on_risk,
        "Annualized Return %": annualized_return,
        "Delta": abs(safe_num(short_leg.get("Delta"), 0)),
        "Legs": legs_text,
        "Short Contract": short_leg.get("contractSymbol", ""),
        "Long Contract": long_leg.get("contractSymbol", ""),
        **liquidity,
    }
    row["Trade Rank"] = score_income_strategy(row)
    row["Notes"] = strategy_note(strategy, row["Signal"], pop, return_on_risk)

    return row


def build_credit_spreads(symbol, price, expiration, dte, calls, puts, stock_signal):
    rows = []
    max_width = max(price * 0.10, 5)

    if puts is not None and not puts.empty:
        put_df = add_trade_prices(puts)
        short_puts = put_df[
            (put_df["strike"] < price) &
            (put_df["Sell Price"] > 0) &
            (put_df["Delta"].abs().between(0.10, 0.45, inclusive="both"))
        ].sort_values("strike", ascending=False)

        for _, short_leg in short_puts.head(14).iterrows():
            short_strike = safe_num(short_leg.get("strike"), 0)
            long_puts = put_df[
                (put_df["strike"] < short_strike) &
                (put_df["Buy Price"] > 0) &
                ((short_strike - put_df["strike"]) <= max_width)
            ].sort_values("strike", ascending=False)

            for _, long_leg in long_puts.head(4).iterrows():
                row = make_credit_spread_row(
                    symbol, price, expiration, dte, short_leg, long_leg, "PUT", stock_signal
                )
                if row:
                    rows.append(row)

    if calls is not None and not calls.empty:
        call_df = add_trade_prices(calls)
        short_calls = call_df[
            (call_df["strike"] > price) &
            (call_df["Sell Price"] > 0) &
            (call_df["Delta"].abs().between(0.10, 0.45, inclusive="both"))
        ].sort_values("strike", ascending=True)

        for _, short_leg in short_calls.head(14).iterrows():
            short_strike = safe_num(short_leg.get("strike"), 0)
            long_calls = call_df[
                (call_df["strike"] > short_strike) &
                (call_df["Buy Price"] > 0) &
                ((call_df["strike"] - short_strike) <= max_width)
            ].sort_values("strike", ascending=True)

            for _, long_leg in long_calls.head(4).iterrows():
                row = make_credit_spread_row(
                    symbol, price, expiration, dte, short_leg, long_leg, "CALL", stock_signal
                )
                if row:
                    rows.append(row)

    return rows


def build_covered_calls(symbol, price, expiration, dte, calls, stock_signal):
    if calls is None or calls.empty or price <= 0:
        return []

    rows = []
    call_df = add_trade_prices(calls)
    call_df = call_df[
        (call_df["strike"] > price) &
        (call_df["Sell Price"] > 0) &
        (call_df["Delta"].abs().between(0.10, 0.45, inclusive="both"))
    ].sort_values("Trade Rank", ascending=False)

    for _, short_leg in call_df.head(18).iterrows():
        short_strike = safe_num(short_leg.get("strike"), 0)
        credit = option_sell_price(short_leg)
        if short_strike <= price or credit <= 0:
            continue

        share_cost = price * 100
        max_profit = (short_strike - price + credit) * 100
        max_loss = max((price - credit) * 100, 0)
        income_yield = credit / price * 100
        annualized_return = income_yield * 365 / max(dte, 1)
        pop = chance_option_expires_otm(short_leg)
        liquidity = leg_liquidity_stats([short_leg])

        row = {
            "Ticker": symbol,
            "Strategy": "Covered Call",
            "Expiration": expiration,
            "DTE": dte,
            "Price": price,
            "Signal": stock_signal.get("signal", "NEUTRAL"),
            "Stock Score": safe_num(stock_signal.get("score"), 50),
            "Short Strike": short_strike,
            "Long Strike": np.nan,
            "Short Put": np.nan,
            "Long Put": np.nan,
            "Short Call": short_strike,
            "Long Call": np.nan,
            "Credit": credit,
            "Width": np.nan,
            "Max Profit": max_profit,
            "Max Loss": max_loss,
            "Breakeven": price - credit,
            "POP %": pop,
            "Return on Risk %": income_yield,
            "Annualized Return %": annualized_return,
            "Delta": abs(safe_num(short_leg.get("Delta"), 0)),
            "Legs": f"Own 100 shares / Sell {short_strike:g}C",
            "Short Contract": short_leg.get("contractSymbol", ""),
            "Long Contract": "",
            **liquidity,
        }
        row["Trade Rank"] = score_income_strategy(row)
        row["Notes"] = strategy_note("Covered Call", row["Signal"], pop, income_yield)
        rows.append(row)

    return rows


def build_cash_secured_puts(symbol, price, expiration, dte, puts, stock_signal):
    if puts is None or puts.empty:
        return []

    rows = []
    put_df = add_trade_prices(puts)
    put_df = put_df[
        (put_df["strike"] < price) &
        (put_df["Sell Price"] > 0) &
        (put_df["Delta"].abs().between(0.10, 0.40, inclusive="both"))
    ].sort_values("Trade Rank", ascending=False)

    for _, short_leg in put_df.head(18).iterrows():
        short_strike = safe_num(short_leg.get("strike"), 0)
        credit = option_sell_price(short_leg)
        if short_strike <= 0 or credit <= 0:
            continue

        max_profit = credit * 100
        max_loss = max((short_strike - credit) * 100, 0)
        income_yield = credit / short_strike * 100
        annualized_return = income_yield * 365 / max(dte, 1)
        pop = chance_option_expires_otm(short_leg)
        liquidity = leg_liquidity_stats([short_leg])

        row = {
            "Ticker": symbol,
            "Strategy": "Cash Secured Put",
            "Expiration": expiration,
            "DTE": dte,
            "Price": price,
            "Signal": stock_signal.get("signal", "NEUTRAL"),
            "Stock Score": safe_num(stock_signal.get("score"), 50),
            "Short Strike": short_strike,
            "Long Strike": np.nan,
            "Short Put": short_strike,
            "Long Put": np.nan,
            "Short Call": np.nan,
            "Long Call": np.nan,
            "Credit": credit,
            "Width": np.nan,
            "Max Profit": max_profit,
            "Max Loss": max_loss,
            "Breakeven": short_strike - credit,
            "POP %": pop,
            "Return on Risk %": income_yield,
            "Annualized Return %": annualized_return,
            "Delta": abs(safe_num(short_leg.get("Delta"), 0)),
            "Legs": f"Sell {short_strike:g}P cash secured",
            "Short Contract": short_leg.get("contractSymbol", ""),
            "Long Contract": "",
            **liquidity,
        }
        row["Trade Rank"] = score_income_strategy(row)
        row["Notes"] = strategy_note("Cash Secured Put", row["Signal"], pop, income_yield)
        rows.append(row)

    return rows


def build_iron_condors(symbol, price, expiration, dte, credit_spreads, stock_signal):
    if not credit_spreads:
        return []

    put_spreads = sorted(
        [r for r in credit_spreads if r["Strategy"] == "Bull Put Credit Spread"],
        key=lambda r: r["Trade Rank"],
        reverse=True,
    )[:10]
    call_spreads = sorted(
        [r for r in credit_spreads if r["Strategy"] == "Bear Call Credit Spread"],
        key=lambda r: r["Trade Rank"],
        reverse=True,
    )[:10]

    rows = []

    for put_row in put_spreads:
        for call_row in call_spreads:
            short_put = safe_num(put_row.get("Short Put"), 0)
            short_call = safe_num(call_row.get("Short Call"), 0)

            if short_put <= 0 or short_call <= 0 or short_put >= short_call:
                continue

            credit = safe_num(put_row.get("Credit"), 0) + safe_num(call_row.get("Credit"), 0)
            width = max(safe_num(put_row.get("Width"), 0), safe_num(call_row.get("Width"), 0))

            if credit <= 0.10 or width <= 0 or credit >= width:
                continue

            max_profit = credit * 100
            max_loss = (width - credit) * 100
            if max_loss <= 0:
                continue

            avg_iv = np.mean([safe_num(put_row.get("Avg IV"), 0.30), safe_num(call_row.get("Avg IV"), 0.30)])
            pop = probability_between_strikes(price, short_put, short_call, dte, avg_iv)
            if pop is None:
                pop = min(safe_num(put_row.get("POP %"), 0), safe_num(call_row.get("POP %"), 0))

            return_on_risk = max_profit / max_loss * 100
            annualized_return = return_on_risk * 365 / max(dte, 1)
            min_volume = min(safe_num(put_row.get("Min Volume"), 0), safe_num(call_row.get("Min Volume"), 0))
            min_oi = min(
                safe_num(put_row.get("Min Open Interest"), 0),
                safe_num(call_row.get("Min Open Interest"), 0),
            )
            avg_spread = np.nanmean([
                safe_num(put_row.get("Avg Leg Spread"), np.nan),
                safe_num(call_row.get("Avg Leg Spread"), np.nan),
            ])

            row = {
                "Ticker": symbol,
                "Strategy": "Iron Condor",
                "Expiration": expiration,
                "DTE": dte,
                "Price": price,
                "Signal": stock_signal.get("signal", "NEUTRAL"),
                "Stock Score": safe_num(stock_signal.get("score"), 50),
                "Short Strike": np.nan,
                "Long Strike": np.nan,
                "Short Put": short_put,
                "Long Put": safe_num(put_row.get("Long Put"), 0),
                "Short Call": short_call,
                "Long Call": safe_num(call_row.get("Long Call"), 0),
                "Credit": credit,
                "Width": width,
                "Max Profit": max_profit,
                "Max Loss": max_loss,
                "Breakeven": np.nan,
                "Lower Breakeven": short_put - credit,
                "Upper Breakeven": short_call + credit,
                "POP %": pop,
                "Return on Risk %": return_on_risk,
                "Annualized Return %": annualized_return,
                "Delta": np.nan,
                "Legs": (
                    f"Sell {short_put:g}P / Buy {safe_num(put_row.get('Long Put'), 0):g}P + "
                    f"Sell {short_call:g}C / Buy {safe_num(call_row.get('Long Call'), 0):g}C"
                ),
                "Short Contract": f"{put_row.get('Short Contract', '')} / {call_row.get('Short Contract', '')}",
                "Long Contract": f"{put_row.get('Long Contract', '')} / {call_row.get('Long Contract', '')}",
                "Min Volume": min_volume,
                "Min Open Interest": min_oi,
                "Avg Leg Spread": avg_spread,
                "Avg IV": avg_iv,
            }
            row["Trade Rank"] = score_income_strategy(row)
            row["Notes"] = strategy_note("Iron Condor", row["Signal"], pop, return_on_risk)
            rows.append(row)

    return rows


def format_strategy_table(df: pd.DataFrame):
    format_map = {
        "Price": "${:,.2f}",
        "Credit": "${:,.2f}",
        "Width": "{:.2f}",
        "Max Profit": "${:,.0f}",
        "Max Loss": "${:,.0f}",
        "Breakeven": "${:,.2f}",
        "Lower Breakeven": "${:,.2f}",
        "Upper Breakeven": "${:,.2f}",
        "POP %": "{:.1f}",
        "Return on Risk %": "{:.1f}",
        "Annualized Return %": "{:.1f}",
        "Delta": "{:.2f}",
        "Avg Leg Spread": "{:.2f}",
        "Avg IV": "{:.1%}",
        "Trade Rank": "{:.0f}",
    }
    format_map = {k: v for k, v in format_map.items() if k in df.columns}

    try:
        return df.style.format(format_map, na_rep="")
    except Exception:
        return df


# =========================
# JOURNAL
# =========================
def journal_columns():
    return [
        "Trade ID",
        "Status",
        "Entry Date/Time",
        "Exit Date/Time",
        "Ticker",
        "Trade Type",
        "Strike",
        "Expiration",
        "Entry Price",
        "Exit Price",
        "Contracts",
        "Profit/Loss",
        "Win/Loss",
        "Real POP %",
        "Chance ITM %",
        "Touch %",
        "Delta",
        "Trade Rank",
        "Notes",
    ]


def load_journal() -> pd.DataFrame:
    cols = journal_columns()
    try:
        df = pd.read_csv(JOURNAL_FILE)

        for col in cols:
            if col not in df.columns:
                df[col] = np.nan

        df["Trade ID"] = pd.to_numeric(df["Trade ID"], errors="coerce")

        if df["Trade ID"].isna().all() or df["Trade ID"].isna().any():
            df["Trade ID"] = range(1, len(df) + 1)

        # Prevent pandas dtype errors when writing text into blank journal columns.
        text_cols = [
            "Status",
            "Entry Date/Time",
            "Exit Date/Time",
            "Ticker",
            "Trade Type",
            "Expiration",
            "Win/Loss",
            "Notes",
        ]

        for text_col in text_cols:
            if text_col in df.columns:
                df[text_col] = df[text_col].astype("object")

        df = normalize_journal_dtypes(df)

        return df[cols]

    except Exception:
        return pd.DataFrame(columns=cols)


def save_journal(df: pd.DataFrame):
    df.to_csv(JOURNAL_FILE, index=False)


def normalize_journal_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps journal columns writable on Streamlit Cloud.
    Store most journal columns as object to avoid pandas dtype crashes.
    Convert to numeric only at calculation time.
    """
    df = df.copy()

    for col in journal_columns():
        if col not in df.columns:
            df[col] = ""

    # Keep every journal column object-friendly so strings/timestamps can be written safely.
    for col in df.columns:
        df[col] = df[col].astype("object")

    if "Trade ID" in df.columns:
        df["Trade ID"] = pd.to_numeric(df["Trade ID"], errors="coerce").astype("Int64")

    if "Contracts" in df.columns:
        df["Contracts"] = pd.to_numeric(df["Contracts"], errors="coerce").fillna(1).astype(int)

    return df


def next_trade_id(journal: pd.DataFrame) -> int:
    try:
        if journal.empty or "Trade ID" not in journal.columns:
            return 1
        ids = pd.to_numeric(journal["Trade ID"], errors="coerce").dropna()
        if ids.empty:
            return 1
        return int(ids.max()) + 1
    except Exception:
        return 1


def calc_trade_pnl(trade_type, entry_price, exit_price, contracts):
    try:
        trade_type = str(trade_type).upper().strip()
        entry_price = float(entry_price)
        exit_price = float(exit_price)
        contracts = int(contracts)

        multiplier = 100 if trade_type in ["CALL", "PUT", "SPREAD"] else 1
        return round((exit_price - entry_price) * multiplier * contracts, 2)

    except Exception:
        return 0.0


def trade_result_label(pnl):
    try:
        pnl = float(pnl)
        if pnl > 0:
            return "WIN"
        if pnl < 0:
            return "LOSS"
        return "BREAKEVEN"
    except Exception:
        return "BREAKEVEN"


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Settings")

    period = st.selectbox("Chart Period", ["1mo", "3mo", "6mo", "1y"], index=2)
    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m"], index=0)

    st.divider()
    st.subheader("Option Filters")

    delta_min = st.slider("Min Delta", 0.00, 1.00, 0.00, 0.05)
    delta_max = st.slider("Max Delta", 0.00, 1.00, 1.00, 0.05)
    pop_min = st.slider("Min POP", 0.00, 1.00, 0.00, 0.05)
    iv_min = st.slider("Min IV", 0.00, 2.00, 0.00, 0.05)
    iv_max = st.slider("Max IV", 0.00, 2.00, 2.00, 0.05)

    st.divider()
    st.caption("This tool is for decision support only, not financial advice.")


# =========================
# HEADER / TICKER SELECTION
# =========================
sp500_df = load_sp500_universe()

st.title("📈 TradeEdge Pro")

h1, h2 = st.columns(2)

with h1:
    selected_display = st.selectbox(
        "S&P 500 Company Search",
        sp500_df["Display"].tolist(),
        index=0,
    )

with h2:
    manual_input = st.text_input(
        "Manual Ticker or Company Name",
        placeholder="Example: AAPL, BAC, Bank of America, Tesla",
    )

ticker = resolve_ticker(manual_input, selected_display, sp500_df)
st.caption(f"Active ticker: **{ticker}**")


# =========================
# LOAD MAIN DATA BEFORE DASHBOARD
# =========================
raw_data = load_data(ticker, period, interval)

if raw_data.empty:
    st.error(f"No price data found for {ticker}. Try another ticker or a longer chart period.")
    st.stop()

try:
    data, signal_data = calculate_signal(raw_data)
except Exception as exc:
    st.error(f"Could not calculate signal data for {ticker}.")
    st.exception(exc)
    st.stop()

price = signal_data["price"]
price_change, price_change_pct = get_price_change(data)


# =========================
# TOP DASHBOARD
# =========================
m1, m2, m3, m4, m5, m6 = st.columns(6)

m1.metric("Price", fmt_money(price), delta=fmt_money(price_change) if price_change is not None else None)
m2.metric("Change %", fmt_pct(price_change_pct))
m3.metric("RSI", f"{signal_data['rsi']:.1f}")
m4.metric("Score", f"{signal_data['score']}/100")
m5.metric("Support", fmt_money(signal_data["support"]))
m6.metric("Resistance", fmt_money(signal_data["resistance"]))

signal = signal_data["signal"]

st.markdown(
    f"""
    <div class="signal-box" style="background:{signal_color(signal)};">
        <div class="signal-title">{signal}</div>
        <div class="signal-sub">Stock Trade Score: {signal_data['score']}/100 | VWAP: {fmt_money(signal_data['vwap'])}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


tabs = st.tabs(["📊 Chart", "🧾 Options", "📓 Journal", "🔍 Scanner", "📰 News", "🏢 Info", "❓ Guide", "🧪 Model Test"])


# =========================
# CHART TAB
# =========================
with tabs[0]:
    st.subheader("Chart with VWAP, Volume, RSI, Entry & Exit Levels")

    selected_strike = st.number_input("Highlight Strike", min_value=0.0, value=0.0, step=1.0)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        )
    )

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            yaxis="y2",
            opacity=0.25,
        )
    )

    fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="20 MA", mode="lines"))
    fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="50 MA", mode="lines"))
    fig.add_trace(go.Scatter(x=data.index, y=data["VWAP"], name="VWAP", mode="lines"))

    fig.add_hline(
        y=signal_data["support"],
        line_dash="dash",
        annotation_text=f"Support {fmt_money(signal_data['support'])}",
    )
    fig.add_hline(
        y=signal_data["resistance"],
        line_dash="dash",
        annotation_text=f"Resistance {fmt_money(signal_data['resistance'])}",
    )

    fig.add_hrect(
        y0=signal_data["entry_low"],
        y1=signal_data["entry_high"],
        opacity=0.18,
        annotation_text=f"Bullish Entry {fmt_money(signal_data['entry_low'])} - {fmt_money(signal_data['entry_high'])}",
        annotation_position="top left",
    )

    fig.add_hline(
        y=signal_data["bullish_exit"],
        line_dash="dot",
        annotation_text=f"Bullish Exit {fmt_money(signal_data['bullish_exit'])}",
    )
    fig.add_hline(
        y=signal_data["bearish_exit"],
        line_dash="dot",
        annotation_text=f"Bearish Exit {fmt_money(signal_data['bearish_exit'])}",
    )

    if selected_strike > 0:
        fig.add_hline(
            y=selected_strike,
            line_dash="solid",
            annotation_text=f"Strike {fmt_money(selected_strike)}",
        )

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        yaxis=dict(title="Price"),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
            rangemode="tozero",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.success(f"Bullish Entry: {fmt_money(signal_data['entry_low'])} - {fmt_money(signal_data['entry_high'])}")
    c2.success(f"Bullish Exit: {fmt_money(signal_data['bullish_exit'])}")
    c3.warning(f"Bearish Entry / Resistance: {fmt_money(signal_data['bearish_entry'])}")
    c4.error(f"Bearish Exit / Stop: {fmt_money(signal_data['bearish_exit'])}")

    st.subheader("RSI Panel")

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI"))
    rsi_fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought 70")
    rsi_fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold 30")
    rsi_fig.update_layout(
        height=260,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(rsi_fig, use_container_width=True)


# =========================
# OPTIONS TAB
# =========================
with tabs[1]:
    st.subheader("Options Chain with Auto Best Contract")

    expirations = get_options_expirations(ticker)

    if not expirations:
        st.error("No options available for this ticker.")
    else:
        exp = st.selectbox("Expiration", expirations)
        typ = st.radio("Type", ["CALL", "PUT"], horizontal=True)

        opt_df = load_option_chain(ticker, exp, typ)

        if opt_df.empty:
            st.warning("No contracts found.")
        else:
            opt_df, dte = prepare_options(opt_df, price, exp, typ)

            filtered = opt_df.copy()
            filtered = filtered[
                filtered["Delta"].abs().between(delta_min, delta_max, inclusive="both")
                | filtered["Delta"].isna()
            ]
            filtered = filtered[(filtered["POP"] >= pop_min) | filtered["POP"].isna()]
            filtered = filtered[filtered["IV"].between(iv_min, iv_max, inclusive="both")]
            # Apply quality gate first, then fall back if too restrictive.
            quality_filtered = filtered[filtered.apply(is_valid_trade, axis=1)]

            if not quality_filtered.empty:
                filtered = quality_filtered.sort_values("Trade Rank", ascending=False)
            else:
                filtered = filtered.sort_values("Trade Rank", ascending=False)
                st.warning("No contracts passed the strict quality gate. Showing best available ranked contracts.")

            if filtered.empty:
                st.warning("No contracts matched filters. Showing full chain.")
                filtered = opt_df.sort_values("Trade Rank", ascending=False)

            best = filtered.iloc[0]

            o1, o2, o3, o4 = st.columns(4)
            o1.metric("DTE", dte)
            o2.metric("Contracts", len(filtered))
            o3.metric("Trade Rank", f"{best['Trade Rank']:.0f}/100")
            o4.metric("Best Cost", fmt_money(best["Contract Cost"]))

            st.success(
                f"Best {typ}: Strike {fmt_money(best['strike'])} | "
                f"Last {fmt_money(best['lastPrice'])} | "
                f"Delta {safe_num(best['Delta'], 0):.2f} | "
                f"POP {fmt_pct(best['POP'])} | "
                f"Real POP {safe_num(best.get('Real POP %'), 0):.1f}% | "
                f"Touch {safe_num(best.get('Touch %'), 0):.1f}% | "
                f"Rank {best['Trade Rank']:.0f}/100"
            )

            display_cols = [
                "contractSymbol",
                "Trade Rank",
                "strike",
                "lastPrice",
                "Contract Cost",
                "bid",
                "ask",
                "Spread",
                "volume",
                "openInterest",
                "IV",
                "Delta",
                "Gamma",
                "Theta",
                "Vega",
                "Real POP %",
                "Chance ITM %",
                "Touch %",
                "POP",
                "Score",
            ]

            display_cols = [c for c in display_cols if c in filtered.columns]

            st.markdown("### 🔥 Top 3 Ranked Contracts")
            top3_cols = [
                "contractSymbol",
                "Trade Rank",
                "strike",
                "lastPrice",
                "Contract Cost",
                "Delta",
                "Real POP %",
                "Chance ITM %",
                "Touch %",
                "volume",
                "openInterest",
                "Spread",
                "Score",
            ]
            top3_cols = [c for c in top3_cols if c in filtered.columns]
            st.dataframe(filtered[top3_cols].head(3), use_container_width=True, height=180)

            st.markdown("### Full Options Chain")
            st.dataframe(filtered[display_cols], use_container_width=True, height=560)


# =========================
# JOURNAL TAB
# =========================
with tabs[2]:
    st.subheader("Trading Journal")

    journal = load_journal()

    st.markdown("### Log New Entry")

    with st.form("journal_entry_form"):
        a, b, c = st.columns(3)

        with a:
            j_ticker = st.text_input("Ticker", ticker)
            j_type = st.selectbox("Trade Type", ["CALL", "PUT", "SPREAD", "STOCK"])
            j_strike = st.number_input("Strike", min_value=0.0, step=0.5)

        with b:
            j_exp = st.date_input("Expiration")
            entry = st.number_input("Entry Price", min_value=0.0, step=0.01)
            contracts = st.number_input("Contracts/Shares", min_value=1, value=1, step=1)

        with c:
            j_real_pop = st.number_input("Real POP %", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            j_touch = st.number_input("Touch %", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            j_delta = st.number_input("Delta", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)

        j_rank = st.number_input("Trade Rank", min_value=0, max_value=100, value=0, step=1)
        j_notes = st.text_area("Entry Notes", placeholder="Why did you take this trade?")
        submit_entry = st.form_submit_button("Log Entry as OPEN")

        if submit_entry:
            new = pd.DataFrame(
                [
                    {
                        "Trade ID": next_trade_id(journal),
                        "Status": "OPEN",
                        "Entry Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Exit Date/Time": "",
                        "Ticker": j_ticker.upper().strip(),
                        "Trade Type": j_type,
                        "Strike": float(j_strike),
                        "Expiration": j_exp.strftime("%Y-%m-%d"),
                        "Entry Price": float(entry),
                        "Exit Price": "",
                        "Contracts": int(contracts),
                        "Profit/Loss": "",
                        "Win/Loss": "",
                        "Real POP %": float(j_real_pop),
                        "Chance ITM %": "",
                        "Touch %": float(j_touch),
                        "Delta": float(j_delta),
                        "Trade Rank": int(j_rank),
                        "Notes": j_notes,
                    }
                ]
            )

            journal = pd.concat([journal, new], ignore_index=True)
            journal = normalize_journal_dtypes(journal)
            save_journal(journal)
            st.success("Entry logged as OPEN. Refresh if it does not appear immediately.")

    st.divider()

    open_trades = journal[journal["Status"].astype(str).str.upper() == "OPEN"].copy()
    closed_trades = journal[journal["Status"].astype(str).str.upper() == "CLOSED"].copy()

    st.markdown("### Open Positions")

    if open_trades.empty:
        st.info("No open positions.")
    else:
        st.dataframe(format_journal_table(open_trades), use_container_width=True, height=260)

        st.markdown("### Close a Position")

        open_ids = open_trades["Trade ID"].astype(int).tolist()

        with st.form("close_trade_form"):
            close_id = st.selectbox("Select Open Trade ID", open_ids)
            exit_price = st.number_input("Exit Price", min_value=0.0, step=0.01)
            close_notes = st.text_area("Exit Notes", placeholder="Why did you exit?")
            submit_close = st.form_submit_button("Close Trade and Calculate P/L")

            if submit_close:
                idx_list = journal.index[pd.to_numeric(journal["Trade ID"], errors="coerce") == int(close_id)].tolist()

                if not idx_list:
                    st.error("Could not find that trade ID.")
                else:
                    idx = idx_list[0]
                    trade = journal.loc[idx]

                    pnl = calc_trade_pnl(
                        trade.get("Trade Type"),
                        trade.get("Entry Price"),
                        exit_price,
                        trade.get("Contracts"),
                    )

                    result = trade_result_label(pnl)

                    existing_notes = "" if pd.isna(trade.get("Notes")) else str(trade.get("Notes"))
                    combined_notes = existing_notes

                    if close_notes.strip():
                        combined_notes = (existing_notes + "\nExit: " + close_notes.strip()).strip()

                    # Normalize dtypes before assigning values.
                    journal = normalize_journal_dtypes(journal)

                    journal.at[idx, "Status"] = "CLOSED"
                    journal.at[idx, "Exit Date/Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    journal.at[idx, "Exit Price"] = float(exit_price)
                    journal.at[idx, "Profit/Loss"] = float(pnl)
                    journal.at[idx, "Win/Loss"] = str(result)
                    journal.at[idx, "Notes"] = str(combined_notes)

                    journal = normalize_journal_dtypes(journal)
                    save_journal(journal)

                    st.success(f"Trade closed: {result} | P/L: {fmt_money(pnl)}")

    st.divider()

    st.markdown("### Closed Trade Performance")

    if closed_trades.empty:
        st.info("No closed trades yet.")
    else:
        closed_trades["Profit/Loss"] = pd.to_numeric(closed_trades["Profit/Loss"], errors="coerce").fillna(0)

        total = len(closed_trades)
        wins = len(closed_trades[closed_trades["Win/Loss"] == "WIN"])
        losses = len(closed_trades[closed_trades["Win/Loss"] == "LOSS"])
        breakevens = len(closed_trades[closed_trades["Win/Loss"] == "BREAKEVEN"])
        win_rate = wins / total * 100 if total else 0
        total_pnl = closed_trades["Profit/Loss"].sum()

        j1, j2, j3, j4, j5, j6 = st.columns(6)
        j1.metric("Closed Trades", total)
        j2.metric("Wins", wins)
        j3.metric("Losses", losses)
        j4.metric("Breakeven", breakevens)
        j5.metric("Win Rate", f"{win_rate:.1f}%")
        with j6:
            pnl_metric_card("Total P/L", total_pnl)

        try:
            format_map = {
                "Strike": "{:.2f}",
                "Entry Price": "{:.2f}",
                "Exit Price": "{:.2f}",
                "Profit/Loss": "{:.2f}",
                "Real POP %": "{:.2f}",
                "Chance ITM %": "{:.2f}",
                "Touch %": "{:.2f}",
                "Delta": "{:.2f}",
                "Trade Rank": "{:.0f}",
                "Contracts": "{:.0f}",
            }
            format_map = {k: v for k, v in format_map.items() if k in closed_trades.columns}

            styled_closed = closed_trades.style.format(format_map).apply(
                lambda col: [
                    "color: #16a34a; font-weight: 700;" if pd.to_numeric(v, errors="coerce") > 0 else
                    "color: #dc2626; font-weight: 700;" if pd.to_numeric(v, errors="coerce") < 0 else
                    "color: #6b7280;"
                    for v in col
                ] if col.name == "Profit/Loss" else ["" for _ in col],
                axis=0,
            )
            st.dataframe(styled_closed, use_container_width=True, height=360)
        except Exception:
            st.dataframe(format_journal_table(closed_trades), use_container_width=True, height=360)

    st.divider()

    if not journal.empty:
        st.markdown("### Full Journal")
        st.dataframe(format_journal_table(journal), use_container_width=True, height=300)

        st.download_button(
            "Download Journal CSV",
            journal.to_csv(index=False),
            "trade_journal.csv",
            "text/csv",
        )

        if st.button("Clear Entire Journal"):
            save_journal(pd.DataFrame(columns=journal_columns()))
            st.warning("Journal cleared. Refresh app.")

# =========================
# SCANNER TAB
# =========================
with tabs[3]:
    st.subheader("Multi-Stock Strategy Scanner")

    scan_text = st.text_area("Tickers", SCAN_DEFAULT)
    scan_list = [x.strip().upper() for x in scan_text.replace("\n", ",").split(",") if x.strip()]

    scan_strategies = st.multiselect(
        "Strategies",
        STRATEGY_CHOICES,
        default=STRATEGY_CHOICES,
    )

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        scan_min_dte = st.number_input("Min DTE", min_value=1, max_value=365, value=14, step=1)
    with s2:
        scan_max_dte = st.number_input("Max DTE", min_value=1, max_value=365, value=60, step=1)
    with s3:
        scan_exp_limit = st.slider("Expirations per Ticker", 1, 6, 3)
    with s4:
        scan_min_rank = st.slider("Minimum Trade Rank", 0, 100, 55, 5)

    scan_top_per_ticker = st.slider("Top Trades per Ticker", 1, 10, 3)

    if st.button("Run Scanner"):
        if not scan_strategies:
            st.warning("Select at least one strategy to scan.")
            st.stop()

        if scan_max_dte < scan_min_dte:
            st.warning("Max DTE must be greater than or equal to Min DTE.")
            st.stop()

        rows = []
        skipped = []

        with st.spinner("Scanning tickers and option chains..."):
            for symbol in scan_list:
                try:
                    d = load_data(symbol, "3mo", "1d")
                    if d.empty:
                        skipped.append(f"{symbol}: no price data")
                        continue

                    _, sig = calculate_signal(d)
                    scan_price = safe_num(sig.get("price"), 0)
                    expirations = get_options_expirations(symbol)
                    selected_expirations = select_strategy_expirations(
                        expirations,
                        int(scan_min_dte),
                        int(scan_max_dte),
                        int(scan_exp_limit),
                    )

                    if scan_price <= 0 or not selected_expirations:
                        skipped.append(f"{symbol}: no usable expirations")
                        continue

                    need_calls = any(
                        strategy in scan_strategies
                        for strategy in ["Credit Spreads", "Iron Condors", "Covered Calls"]
                    )
                    need_puts = any(
                        strategy in scan_strategies
                        for strategy in ["Credit Spreads", "Iron Condors", "Cash Secured Puts"]
                    )

                    for exp, dte in selected_expirations:
                        calls = pd.DataFrame()
                        puts = pd.DataFrame()

                        if need_calls:
                            raw_calls = load_option_chain(symbol, exp, "CALL")
                            if not raw_calls.empty:
                                calls, _ = prepare_options(raw_calls, scan_price, exp, "CALL")

                        if need_puts:
                            raw_puts = load_option_chain(symbol, exp, "PUT")
                            if not raw_puts.empty:
                                puts, _ = prepare_options(raw_puts, scan_price, exp, "PUT")

                        credit_spreads = []
                        if "Credit Spreads" in scan_strategies or "Iron Condors" in scan_strategies:
                            credit_spreads = build_credit_spreads(
                                symbol, scan_price, exp, dte, calls, puts, sig
                            )

                        if "Credit Spreads" in scan_strategies:
                            rows.extend(credit_spreads)

                        if "Iron Condors" in scan_strategies:
                            rows.extend(
                                build_iron_condors(symbol, scan_price, exp, dte, credit_spreads, sig)
                            )

                        if "Covered Calls" in scan_strategies:
                            rows.extend(build_covered_calls(symbol, scan_price, exp, dte, calls, sig))

                        if "Cash Secured Puts" in scan_strategies:
                            rows.extend(build_cash_secured_puts(symbol, scan_price, exp, dte, puts, sig))

                except Exception as exc:
                    skipped.append(f"{symbol}: {exc}")

        if rows:
            scan_df = pd.DataFrame(rows)
            scan_df = scan_df[scan_df["Trade Rank"] >= scan_min_rank]
            scan_df = scan_df.sort_values(["Trade Rank", "POP %", "Annualized Return %"], ascending=False)

            if scan_df.empty:
                st.warning("No strategy candidates met the minimum rank. Lower the rank filter or widen DTE.")
            else:
                best = scan_df.iloc[0]
                st.success(
                    f"Best trade: {best['Ticker']} {best['Strategy']} | "
                    f"{best['Legs']} | Exp {best['Expiration']} | "
                    f"Rank {best['Trade Rank']:.0f}/100 | POP {best['POP %']:.1f}% | "
                    f"Credit {fmt_money(best['Credit'])}"
                )

                best_by_ticker = (
                    scan_df.groupby("Ticker", group_keys=False)
                    .head(int(scan_top_per_ticker))
                    .reset_index(drop=True)
                )

                display_cols = [
                    "Ticker",
                    "Strategy",
                    "Expiration",
                    "DTE",
                    "Trade Rank",
                    "Price",
                    "Signal",
                    "Stock Score",
                    "Legs",
                    "Credit",
                    "Width",
                    "Max Profit",
                    "Max Loss",
                    "POP %",
                    "Return on Risk %",
                    "Annualized Return %",
                    "Breakeven",
                    "Lower Breakeven",
                    "Upper Breakeven",
                    "Delta",
                    "Min Volume",
                    "Min Open Interest",
                    "Avg Leg Spread",
                    "Notes",
                ]
                display_cols = [c for c in display_cols if c in best_by_ticker.columns]

                st.markdown("### Best Ranked Trades")
                st.dataframe(
                    format_strategy_table(best_by_ticker[display_cols]),
                    use_container_width=True,
                    height=560,
                )

                csv_cols = [c for c in scan_df.columns if c not in ["Short Contract", "Long Contract"]]
                st.download_button(
                    "Download Strategy Scanner CSV",
                    scan_df[csv_cols].to_csv(index=False),
                    "strategy_scanner_results.csv",
                    "text/csv",
                )
        else:
            st.warning("Scanner returned no strategy candidates.")

        if skipped:
            with st.expander("Skipped tickers / notes"):
                st.write(pd.DataFrame({"Note": skipped}))


# =========================
# NEWS TAB
# =========================
with tabs[4]:
    st.subheader("News")

    news = load_news(ticker)

    if not news:
        st.warning("No news found.")
    else:
        for item in news[:10]:
            title = item.get("title") or item.get("content", {}).get("title", "No title")
            link = item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url", "")

            if link:
                st.markdown(f"### [{title}]({link})")
            else:
                st.write(title)


# =========================
# INFO TAB
# =========================
with tabs[5]:
    st.subheader("Company Info")

    info = load_info(ticker)

    company_name = info.get("longName", ticker)
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    market_cap = info.get("marketCap")
    summary = info.get("longBusinessSummary", "No company summary available.")

    i1, i2, i3 = st.columns(3)
    i1.metric("Company", company_name)
    i2.metric("Sector", sector)
    i3.metric("Industry", industry)

    if market_cap:
        st.metric("Market Cap", f"${market_cap:,.0f}")

    st.markdown("### Business Summary")
    st.write(summary)


# =========================
# GUIDE TAB
# =========================
with tabs[6]:
    st.subheader("How To Use")
    st.markdown(
        """
        1. Choose an S&P 500 company or type any ticker/company name manually.
        2. Review the score and signal at the top.
        3. Use the chart for VWAP, support, resistance, entry, exit, and volume.
        4. Use the options tab to compare contracts by liquidity, spread, delta, POP, and score.
        5. Track every trade in the journal.
        6. Download your journal CSV.

        **Important:** This is a decision-support tool, not financial advice.
        """
    )


# =========================
# MODEL TEST TAB (TEMP)
# =========================
with tabs[7]:
    st.subheader("POP & Probability of Touch Validation")

    test_price = 100
    test_days = 30
    test_iv = 0.30

    st.caption("Testing model with fixed inputs: Price=100, IV=30%, DTE=30")

    test_cases = pd.DataFrame([
        {"Type": "CALL", "Strike": 90, "Scenario": "Deep ITM Call"},
        {"Type": "CALL", "Strike": 100, "Scenario": "ATM Call"},
        {"Type": "CALL", "Strike": 110, "Scenario": "OTM Call"},
        {"Type": "PUT", "Strike": 110, "Scenario": "Deep ITM Put"},
        {"Type": "PUT", "Strike": 100, "Scenario": "ATM Put"},
        {"Type": "PUT", "Strike": 90, "Scenario": "OTM Put"},
    ])

    results = []

    for _, r in test_cases.iterrows():
        pop_result = calculate_real_pop(
            option_type=r["Type"],
            S=test_price,
            K=r["Strike"],
            premium=3,
            days=test_days,
            iv=test_iv,
        )

        touch = calculate_probability_touch(
            option_type=r["Type"],
            S=test_price,
            K=r["Strike"],
            days=test_days,
            iv=test_iv,
        )

        results.append({
            "Type": r["Type"],
            "Strike": r["Strike"],
            "Scenario": r["Scenario"],
            "Chance ITM %": round(pop_result[0], 2) if pop_result and pop_result[0] is not None else None,
            "Real POP %": round(pop_result[1], 2) if pop_result and pop_result[1] is not None else None,
            "Touch %": touch,
        })

    df_test = pd.DataFrame(results)
    st.dataframe(df_test, use_container_width=True)

    st.markdown("### ✅ What to Check")
    st.markdown(
        """
        - Touch % should be **highest**
        - Chance ITM % should be **middle**
        - Real POP % should be **lowest**

        #### Expected Behavior:
        - Deep ITM → very high probabilities
        - ATM → ~50% ITM, higher touch
        - OTM → lower ITM, moderate touch

        #### Rule:
        ```
        Touch % ≥ Chance ITM % ≥ Real POP %
        ```
        """
    )
