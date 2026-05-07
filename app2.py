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
    st.subheader("Multi-Stock Scanner")

    scan_text = st.text_area("Tickers", SCAN_DEFAULT)
    scan_list = [x.strip().upper() for x in scan_text.replace("\n", ",").split(",") if x.strip()]

    if st.button("Run Scanner"):
        rows = []

        with st.spinner("Scanning tickers..."):
            for symbol in scan_list:
                try:
                    d = load_data(symbol, "3mo", "1d")
                    if d.empty:
                        continue

                    _, sig = calculate_signal(d)

                    rows.append(
                        {
                            "Ticker": symbol,
                            "Price": round(sig["price"], 2),
                            "RSI": round(sig["rsi"], 1),
                            "VWAP": round(sig["vwap"], 2),
                            "Score": sig["score"],
                            "Signal": sig["signal"],
                            "Entry Low": round(sig["entry_low"], 2),
                            "Entry High": round(sig["entry_high"], 2),
                            "Exit": round(sig["bullish_exit"], 2),
                        }
                    )
                except Exception:
                    continue

        if rows:
            scan_df = pd.DataFrame(rows).sort_values("Score", ascending=False)
            st.dataframe(scan_df, use_container_width=True)
        else:
            st.warning("Scanner returned no results.")


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
