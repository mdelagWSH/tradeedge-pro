import os
import math
import json
import urllib.parse
import urllib.request
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange


st.set_page_config(page_title="Options Edge Terminal", layout="wide", initial_sidebar_state="expanded")

st.title("Options Edge Terminal")

st.warning(
    "Educational decision-support only. This does not guarantee winners. "
    "Use position sizing, stops, and risk discipline."
)


# ============================================================
# UI HELPERS
# ============================================================

def display_card(label, value, font_size=30):
    st.markdown(
        f"""
        <div style="
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 16px;
            min-height: 112px;
            background: white;
            overflow-wrap: anywhere;
            word-break: break-word;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        ">
            <div style="font-size: 14px; color: #6b7280; margin-bottom: 8px;">
                {label}
            </div>
            <div style="font-size: {font_size}px; font-weight: 700; line-height: 1.15;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TICKER RESOLVER
# ============================================================

COMMON_TICKERS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "netflix": "NFLX",
    "amd": "AMD",
    "spy": "SPY",
    "qqq": "QQQ",
    "palantir": "PLTR",
    "sofi": "SOFI",
    "robinhood": "HOOD",
    "coinbase": "COIN",
    "ford": "F",
    "disney": "DIS",
    "walmart": "WMT",
    "costco": "COST",
    "target": "TGT",
    "boeing": "BA",
    "coca cola": "KO",
    "coke": "KO",
    "pepsi": "PEP",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "bank of america": "BAC",
    "nike": "NKE",
    "starbucks": "SBUX",
    "mcdonalds": "MCD",
    "chipotle": "CMG",
    "paypal": "PYPL",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "intel": "INTC",
    "micron": "MU",
    "qualcomm": "QCOM",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "snowflake": "SNOW",
    "shopify": "SHOP",
    "draftkings": "DKNG",
}


@st.cache_data(ttl=3600)
def resolve_ticker(user_input):
    raw = str(user_input).strip()
    key = raw.lower()

    if not raw:
        return ""

    if key in COMMON_TICKERS:
        return COMMON_TICKERS[key]

    if raw.isalpha() and len(raw) <= 5:
        return raw.upper()

    try:
        query = urllib.parse.quote(raw)
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0"

        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())

        quotes = data.get("quotes", [])

        for q in quotes:
            symbol = q.get("symbol", "")
            quote_type = q.get("quoteType", "")
            exchange = q.get("exchange", "")

            if symbol and quote_type in ["EQUITY", "ETF"] and exchange in ["NMS", "NYQ", "ASE", "PCX"]:
                return symbol.upper()

    except Exception:
        pass

    return raw.upper()


# ============================================================
# MATH
# ============================================================

def dte(expiration):
    try:
        return max((datetime.strptime(expiration, "%Y-%m-%d").date() - date.today()).days, 0)
    except Exception:
        return 0


def bs_delta(S, K, T, r, sigma, option_type):
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return np.nan

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        if option_type == "call":
            return norm.cdf(d1)

        return norm.cdf(d1) - 1
    except Exception:
        return np.nan


def prob_itm(S, K, T, r, sigma, option_type):
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return np.nan

        d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        if option_type == "call":
            return norm.cdf(d2)

        return norm.cdf(-d2)
    except Exception:
        return np.nan


def prob_above_price(S, target_price, T, r, sigma):
    return prob_itm(S, target_price, T, r, sigma, "call")


def prob_below_price(S, target_price, T, r, sigma):
    return prob_itm(S, target_price, T, r, sigma, "put")


# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=60)
def get_price_data(ticker, period, interval):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.replace([np.inf, -np.inf], np.nan)
        return df.dropna()

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_expirations(ticker):
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []


@st.cache_data(ttl=60)
def get_chain(ticker, expiration):
    try:
        chain = yf.Ticker(ticker).option_chain(expiration)
        return chain.calls.copy(), chain.puts.copy()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=300)
def get_news(ticker):
    try:
        tk = yf.Ticker(ticker)
        news = tk.news or []
        rows = []

        for item in news[:10]:
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            link = item.get("link", "")
            published = item.get("providerPublishTime", None)
            published_text = ""

            if not title and isinstance(item.get("content"), dict):
                content = item.get("content", {})
                title = content.get("title", "")

                provider = content.get("provider", {})
                if isinstance(provider, dict):
                    publisher = provider.get("displayName", "")

                canonical = content.get("canonicalUrl", {})
                if isinstance(canonical, dict):
                    link = canonical.get("url", "")

                pub_date = content.get("pubDate", "")
                if pub_date:
                    try:
                        published_text = pd.to_datetime(pub_date).strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        published_text = ""

            if published:
                try:
                    published_text = datetime.fromtimestamp(published).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    published_text = ""

            if title:
                rows.append({
                    "Title": title,
                    "Publisher": publisher,
                    "Published": published_text,
                    "Link": link
                })

        return pd.DataFrame(rows, columns=["Title", "Publisher", "Published", "Link"])

    except Exception:
        return pd.DataFrame(columns=["Title", "Publisher", "Published", "Link"])


@st.cache_data(ttl=300)
def get_earnings_warning(ticker):
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        earnings_date = None

        if cal is None:
            return "Unknown", None, 0

        if isinstance(cal, dict):
            raw = cal.get("Earnings Date") or cal.get("EarningsDate")
            if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
                raw = raw[0]
            if raw is not None:
                earnings_date = pd.to_datetime(raw).date()

        elif isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                raw = cal.loc["Earnings Date"].values[0]
                if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
                    raw = raw[0]
                earnings_date = pd.to_datetime(raw).date()

            elif "Earnings Date" in cal.columns and not cal.empty:
                raw = cal["Earnings Date"].iloc[0]
                if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) > 0:
                    raw = raw[0]
                earnings_date = pd.to_datetime(raw).date()

        if earnings_date is None:
            return "Unknown", None, 0

        days = (earnings_date - date.today()).days

        if 0 <= days <= 3:
            return "High", earnings_date, days

        if 4 <= days <= 10:
            return "Medium", earnings_date, days

        if days > 10:
            return "Low", earnings_date, days

        return "Past/Unknown", earnings_date, days

    except Exception:
        return "Unknown", None, 0


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):
    try:
        data = df.copy()

        if data.empty or len(data) < 35:
            return pd.DataFrame()

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        data["RSI"] = RSIIndicator(close=close, window=14).rsi()

        macd = MACD(close=close)
        data["MACD"] = macd.macd()
        data["MACD_SIGNAL"] = macd.macd_signal()
        data["MACD_HIST"] = macd.macd_diff()

        data["EMA_9"] = data["Close"].ewm(span=9, adjust=False).mean()
        data["EMA_21"] = data["Close"].ewm(span=21, adjust=False).mean()
        data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()
        data["EMA_200"] = data["Close"].ewm(span=200, adjust=False).mean()

        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        data["ATR"] = atr.average_true_range()

        data["VOL_AVG"] = volume.rolling(20).mean()
        data["VOL_RATIO"] = volume / data["VOL_AVG"]

        data["RET"] = data["Close"].pct_change()
        data["REALIZED_VOL"] = data["RET"].rolling(20).std() * np.sqrt(252)

        data = data.replace([np.inf, -np.inf], np.nan)

        required_cols = [
            "Close", "Open", "High", "Low", "Volume",
            "RSI", "MACD", "MACD_SIGNAL", "MACD_HIST",
            "EMA_9", "EMA_21", "EMA_50", "EMA_200",
            "ATR", "VOL_RATIO"
        ]

        data = data.dropna(subset=required_cols)

        return data

    except Exception:
        return pd.DataFrame()


# ============================================================
# NEWS SENTIMENT
# ============================================================

POSITIVE_WORDS = [
    "beat", "beats", "upgrade", "upgraded", "raises", "raised", "strong",
    "growth", "surge", "record", "profit", "profits", "bullish", "outperform",
    "buy", "partnership", "contract", "approval", "launch", "expands",
    "higher", "positive", "momentum", "guidance raised"
]

NEGATIVE_WORDS = [
    "miss", "misses", "downgrade", "downgraded", "cuts", "cut", "weak",
    "decline", "falls", "fall", "drops", "drop", "lawsuit", "probe",
    "investigation", "sec", "bearish", "underperform", "sell", "layoffs",
    "recession", "warning", "guidance cut", "loss", "losses", "slumps",
    "recall", "delay", "delayed"
]


def score_news_sentiment(news_df):
    if news_df is None or news_df.empty:
        return 0, "No recent news"

    score = 0

    for title in news_df["Title"].fillna("").head(10):
        text = str(title).lower()

        pos_hits = [w for w in POSITIVE_WORDS if w in text]
        neg_hits = [w for w in NEGATIVE_WORDS if w in text]

        score += 10 * len(pos_hits)
        score -= 10 * len(neg_hits)

    score = int(max(-100, min(100, score)))

    if score >= 30:
        label = "Positive"
    elif score <= -30:
        label = "Negative"
    else:
        label = "Neutral"

    return score, label


# ============================================================
# MARKET FILTER
# ============================================================

def market_filter():
    try:
        spy = add_indicators(get_price_data("SPY", "6mo", "1d"))
        qqq = add_indicators(get_price_data("QQQ", "6mo", "1d"))
        vix = get_price_data("^VIX", "3mo", "1d")

        if spy.empty or qqq.empty:
            return {
                "bias": "Unknown",
                "call_ok": True,
                "put_ok": True,
                "vix": np.nan,
                "details": "Market data unavailable"
            }

        spy_last = spy.iloc[-1]
        qqq_last = qqq.iloc[-1]

        spy_bull = spy_last["Close"] > spy_last["EMA_21"] > spy_last["EMA_50"]
        qqq_bull = qqq_last["Close"] > qqq_last["EMA_21"] > qqq_last["EMA_50"]

        spy_bear = spy_last["Close"] < spy_last["EMA_21"] < spy_last["EMA_50"]
        qqq_bear = qqq_last["Close"] < qqq_last["EMA_21"] < qqq_last["EMA_50"]

        vix_value = np.nan
        if not vix.empty:
            vix_value = float(vix["Close"].iloc[-1])

        if spy_bull and qqq_bull:
            bias = "Bullish"
            call_ok = True
            put_ok = False
        elif spy_bear and qqq_bear:
            bias = "Bearish"
            call_ok = False
            put_ok = True
        else:
            bias = "Mixed"
            call_ok = True
            put_ok = True

        if not np.isnan(vix_value) and vix_value >= 25:
            bias = f"{bias} / High Volatility"

        return {
            "bias": bias,
            "call_ok": call_ok,
            "put_ok": put_ok,
            "vix": vix_value,
            "details": f"SPY bullish: {spy_bull}, QQQ bullish: {qqq_bull}"
        }

    except Exception:
        return {
            "bias": "Unknown",
            "call_ok": True,
            "put_ok": True,
            "vix": np.nan,
            "details": "Market filter failed"
        }


# ============================================================
# STOCK SETUP SCORING
# ============================================================

def support_resistance(df, lookback=60):
    recent = df.tail(min(lookback, len(df)))
    return float(recent["Low"].min()), float(recent["High"].max())


def score_stock_setup(df, news_score=0, earnings_risk="Unknown"):
    try:
        if df is None or df.empty or len(df) < 2:
            return None

        row = df.iloc[-1]
        prev = df.iloc[-2]

        price = float(row["Close"])
        support, resistance = support_resistance(df)
        atr = float(row["ATR"])
        rsi = float(row["RSI"])
        vol_ratio = float(row["VOL_RATIO"])

        call_score = 0
        put_score = 0
        call_reasons = []
        put_reasons = []

        if price > row["EMA_21"] > row["EMA_50"]:
            call_score += 25
            call_reasons.append("Bullish trend: price above EMA21 and EMA50")

        if price > row["EMA_200"]:
            call_score += 10
            call_reasons.append("Price above EMA200")

        if price < row["EMA_21"] < row["EMA_50"]:
            put_score += 25
            put_reasons.append("Bearish trend: price below EMA21 and EMA50")

        if price < row["EMA_200"]:
            put_score += 10
            put_reasons.append("Price below EMA200")

        if 30 <= rsi <= 45:
            call_score += 15
            call_reasons.append("RSI pullback zone")

        if rsi < 30:
            call_score += 10
            call_reasons.append("RSI oversold")

        if 55 <= rsi <= 70:
            put_score += 10
            put_reasons.append("RSI elevated")

        if rsi > 70:
            put_score += 15
            put_reasons.append("RSI overbought")

        bullish_cross = prev["MACD"] <= prev["MACD_SIGNAL"] and row["MACD"] > row["MACD_SIGNAL"]
        bearish_cross = prev["MACD"] >= prev["MACD_SIGNAL"] and row["MACD"] < row["MACD_SIGNAL"]

        if bullish_cross:
            call_score += 20
            call_reasons.append("Bullish MACD crossover")
        elif row["MACD"] > row["MACD_SIGNAL"]:
            call_score += 10
            call_reasons.append("MACD above signal")

        if bearish_cross:
            put_score += 20
            put_reasons.append("Bearish MACD crossover")
        elif row["MACD"] < row["MACD_SIGNAL"]:
            put_score += 10
            put_reasons.append("MACD below signal")

        if vol_ratio >= 1.5:
            call_score += 10
            put_score += 10
            call_reasons.append("Volume expansion")
            put_reasons.append("Volume expansion")

        if atr > 0:
            if abs(price - support) / atr <= 1.5:
                call_score += 15
                call_reasons.append("Near support")

            if abs(resistance - price) / atr <= 1.5:
                put_score += 15
                put_reasons.append("Near resistance")

        if news_score >= 30:
            call_score += 8
            call_reasons.append("Positive news sentiment")

        if news_score <= -30:
            put_score += 8
            put_reasons.append("Negative news sentiment")

        if earnings_risk == "High":
            call_score -= 15
            put_score -= 15

        if price > 0 and abs(row["EMA_21"] - row["EMA_50"]) / price < 0.005:
            call_score -= 8
            put_score -= 8

        call_score = int(max(0, min(100, call_score)))
        put_score = int(max(0, min(100, put_score)))

        if call_score >= 75 and call_score > put_score:
            signal = "STRONG CALL SETUP"
            preferred_side = "call"
            stock_score = call_score
        elif put_score >= 75 and put_score > call_score:
            signal = "STRONG PUT SETUP"
            preferred_side = "put"
            stock_score = put_score
        elif call_score >= 60 and call_score > put_score:
            signal = "CALL WATCHLIST"
            preferred_side = "call"
            stock_score = call_score
        elif put_score >= 60 and put_score > call_score:
            signal = "PUT WATCHLIST"
            preferred_side = "put"
            stock_score = put_score
        else:
            signal = "NO TRADE / WAIT"
            preferred_side = "call" if call_score >= put_score else "put"
            stock_score = max(call_score, put_score)

        if preferred_side == "call" and atr > 0:
            entry_low = price - 0.75 * atr
            entry_high = price + 0.25 * atr
            stop = price - 1.25 * atr
            target1 = price + 1.5 * atr
            target2 = price + 2.5 * atr
        elif preferred_side == "put" and atr > 0:
            entry_low = price - 0.25 * atr
            entry_high = price + 0.75 * atr
            stop = price + 1.25 * atr
            target1 = price - 1.5 * atr
            target2 = price - 2.5 * atr
        else:
            entry_low = np.nan
            entry_high = np.nan
            stop = np.nan
            target1 = np.nan
            target2 = np.nan

        return {
            "price": price,
            "support": support,
            "resistance": resistance,
            "atr": atr,
            "rsi": rsi,
            "vol_ratio": vol_ratio,
            "call_score": call_score,
            "put_score": put_score,
            "stock_score": stock_score,
            "preferred_side": preferred_side,
            "signal": signal,
            "entry_low": entry_low,
            "entry_high": entry_high,
            "stop": stop,
            "target1": target1,
            "target2": target2,
            "call_reasons": call_reasons,
            "put_reasons": put_reasons,
        }

    except Exception:
        return None


# ============================================================
# OPTION QUALITY
# ============================================================

def prepare_chain(chain):
    if chain is None or chain.empty:
        return pd.DataFrame()

    df = chain.copy()

    for col in [
        "bid", "ask", "lastPrice", "strike", "volume",
        "openInterest", "impliedVolatility", "contractSymbol"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["volume"] = df["volume"].fillna(0)
    df["openInterest"] = df["openInterest"].fillna(0)

    df = df[
        (df["bid"].fillna(0) > 0) &
        (df["ask"].fillna(0) > 0) &
        (df["ask"] >= df["bid"]) &
        (df["impliedVolatility"].fillna(0) > 0)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    df["spread_pct"] = np.where(df["mid"] > 0, df["spread"] / df["mid"] * 100, np.nan)
    df["cost_mid"] = df["mid"] * 100
    df["cost_ask"] = df["ask"] * 100

    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mid", "spread_pct", "cost_ask"])


def option_quality_score(row, min_delta, max_delta, min_volume, min_oi, max_spread_pct, max_cost):
    score = 0

    abs_delta = abs(row.get("delta", np.nan))

    if min_delta <= abs_delta <= max_delta:
        score += 25

    if row.get("spread_pct", 999) <= max_spread_pct:
        score += 25

    if row.get("volume", 0) >= min_volume:
        score += 20

    if row.get("openInterest", 0) >= min_oi:
        score += 20

    if row.get("cost_ask", 999999) <= max_cost:
        score += 10

    return int(max(0, min(100, score)))


def build_options_table(
    chain,
    option_type,
    stock_price,
    exp,
    stock_setup_score,
    min_delta,
    max_delta,
    min_volume,
    min_oi,
    max_spread_pct,
    max_cost
):
    try:
        df = prepare_chain(chain)

        if df.empty:
            return pd.DataFrame()

        T = max(dte(exp) / 365, 1 / 365)
        r = 0.045

        df["delta"] = df.apply(
            lambda x: bs_delta(stock_price, x["strike"], T, r, x["impliedVolatility"], option_type),
            axis=1
        )

        df["abs_delta"] = df["delta"].abs()

        df["prob_itm"] = df.apply(
            lambda x: prob_itm(stock_price, x["strike"], T, r, x["impliedVolatility"], option_type),
            axis=1
        )

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["delta", "prob_itm"])

        if df.empty:
            return pd.DataFrame()

        df["option_quality_score"] = df.apply(
            lambda row: option_quality_score(
                row, min_delta, max_delta, min_volume, min_oi, max_spread_pct, max_cost
            ),
            axis=1
        )

        df["final_trade_score"] = (
            0.60 * stock_setup_score +
            0.40 * df["option_quality_score"]
        ).round(0).astype(int)

        filtered = df[
            (df["abs_delta"].between(min_delta, max_delta)) &
            (df["spread_pct"].fillna(999) <= max_spread_pct) &
            (df["volume"] >= min_volume) &
            (df["openInterest"] >= min_oi) &
            (df["cost_ask"].fillna(999999) <= max_cost)
        ].copy()

        if filtered.empty:
            filtered = df.copy()

        cols = [
            "contractSymbol", "strike", "lastPrice", "bid", "ask", "mid",
            "cost_mid", "cost_ask", "spread_pct", "volume", "openInterest",
            "impliedVolatility", "delta", "prob_itm",
            "option_quality_score", "final_trade_score"
        ]

        return filtered[cols].sort_values("final_trade_score", ascending=False)

    except Exception:
        return pd.DataFrame()


def decision_label(stock_score, option_score, final_score, earnings_risk, market_conflict):
    if earnings_risk == "High":
        return "DO NOT TRADE", "High earnings risk"

    if market_conflict:
        return "WATCHLIST", "Market trend conflicts with trade direction"

    if stock_score < 60 and option_score >= 60:
        return "WATCHLIST", "Good contract, weak stock setup"

    if stock_score >= 70 and option_score < 60:
        return "FIND BETTER CONTRACT", "Good setup, weak option contract"

    if final_score >= 75:
        return "TRADE CANDIDATE", "Setup and contract quality align"

    if final_score >= 60:
        return "WATCHLIST", "Needs more confirmation"

    return "AVOID", "Low combined score"


# ============================================================
# SPREAD STRATEGIES
# ============================================================

def spread_liquidity_ok(a, b, min_volume=10, min_oi=25, max_spread_pct=35):
    try:
        avg_volume = (float(a["volume"]) + float(b["volume"])) / 2
        avg_oi = (float(a["openInterest"]) + float(b["openInterest"])) / 2
        avg_spread = (float(a["spread_pct"]) + float(b["spread_pct"])) / 2

        return avg_volume >= min_volume and avg_oi >= min_oi and avg_spread <= max_spread_pct
    except Exception:
        return False


def spread_liquidity_score(a, b):
    avg_volume = (float(a["volume"]) + float(b["volume"])) / 2
    avg_oi = (float(a["openInterest"]) + float(b["openInterest"])) / 2
    avg_spread = (float(a["spread_pct"]) + float(b["spread_pct"])) / 2

    volume_score = min(30, avg_volume / 5)
    oi_score = min(30, avg_oi / 20)
    spread_score = max(0, 40 - avg_spread)

    return min(100, volume_score + oi_score + spread_score)


def build_debit_call_spreads(calls, stock_price, exp, stock_score):
    df = prepare_chain(calls)
    if df.empty:
        return pd.DataFrame()

    rows = []
    df = df.sort_values("strike").reset_index(drop=True)
    T = max(dte(exp) / 365, 1 / 365)

    for i in range(len(df)):
        buy = df.iloc[i]

        for j in range(i + 1, min(i + 6, len(df))):
            sell = df.iloc[j]

            if not spread_liquidity_ok(buy, sell):
                continue

            width = sell["strike"] - buy["strike"]
            debit = buy["ask"] - sell["bid"]

            if width <= 0 or debit <= 0:
                continue

            max_profit = width - debit
            max_loss = debit
            breakeven = buy["strike"] + debit

            if max_profit <= 0 or max_loss <= 0:
                continue

            rr = max_profit / max_loss
            prob_est = prob_above_price(stock_price, breakeven, T, 0.045, buy["impliedVolatility"]) * 100
            liquidity = spread_liquidity_score(buy, sell)

            score = int(max(0, min(
                100,
                0.40 * stock_score +
                0.25 * min(100, rr * 25) +
                0.20 * liquidity +
                0.15 * prob_est
            )))

            rows.append({
                "Strategy": "Debit Call Spread",
                "Bias": "Bullish",
                "Buy Leg": buy["contractSymbol"],
                "Sell Leg": sell["contractSymbol"],
                "Buy Strike": buy["strike"],
                "Sell Strike": sell["strike"],
                "Width": width,
                "Debit/Credit": round(debit * 100, 2),
                "Max Profit": round(max_profit * 100, 2),
                "Max Loss": round(max_loss * 100, 2),
                "Breakeven": round(breakeven, 2),
                "Risk/Reward": round(rr, 2),
                "Probability Estimate %": round(prob_est, 1),
                "Spread Score": score
            })

    return pd.DataFrame(rows).sort_values("Spread Score", ascending=False) if rows else pd.DataFrame()


def build_debit_put_spreads(puts, stock_price, exp, stock_score):
    df = prepare_chain(puts)
    if df.empty:
        return pd.DataFrame()

    rows = []
    df = df.sort_values("strike", ascending=False).reset_index(drop=True)
    T = max(dte(exp) / 365, 1 / 365)

    for i in range(len(df)):
        buy = df.iloc[i]

        for j in range(i + 1, min(i + 6, len(df))):
            sell = df.iloc[j]

            if not spread_liquidity_ok(buy, sell):
                continue

            width = buy["strike"] - sell["strike"]
            debit = buy["ask"] - sell["bid"]

            if width <= 0 or debit <= 0:
                continue

            max_profit = width - debit
            max_loss = debit
            breakeven = buy["strike"] - debit

            if max_profit <= 0 or max_loss <= 0:
                continue

            rr = max_profit / max_loss
            prob_est = prob_below_price(stock_price, breakeven, T, 0.045, buy["impliedVolatility"]) * 100
            liquidity = spread_liquidity_score(buy, sell)

            score = int(max(0, min(
                100,
                0.40 * stock_score +
                0.25 * min(100, rr * 25) +
                0.20 * liquidity +
                0.15 * prob_est
            )))

            rows.append({
                "Strategy": "Debit Put Spread",
                "Bias": "Bearish",
                "Buy Leg": buy["contractSymbol"],
                "Sell Leg": sell["contractSymbol"],
                "Buy Strike": buy["strike"],
                "Sell Strike": sell["strike"],
                "Width": width,
                "Debit/Credit": round(debit * 100, 2),
                "Max Profit": round(max_profit * 100, 2),
                "Max Loss": round(max_loss * 100, 2),
                "Breakeven": round(breakeven, 2),
                "Risk/Reward": round(rr, 2),
                "Probability Estimate %": round(prob_est, 1),
                "Spread Score": score
            })

    return pd.DataFrame(rows).sort_values("Spread Score", ascending=False) if rows else pd.DataFrame()


def build_credit_put_spreads(puts, stock_price, exp, stock_score):
    df = prepare_chain(puts)
    if df.empty:
        return pd.DataFrame()

    rows = []
    df = df.sort_values("strike", ascending=False).reset_index(drop=True)
    T = max(dte(exp) / 365, 1 / 365)

    for i in range(len(df)):
        sell = df.iloc[i]

        for j in range(i + 1, min(i + 6, len(df))):
            buy = df.iloc[j]

            if not spread_liquidity_ok(sell, buy):
                continue

            width = sell["strike"] - buy["strike"]
            credit = sell["bid"] - buy["ask"]

            if width <= 0 or credit <= 0:
                continue

            max_profit = credit
            max_loss = width - credit
            breakeven = sell["strike"] - credit

            if max_profit <= 0 or max_loss <= 0:
                continue

            rr = max_profit / max_loss
            prob_est = prob_above_price(stock_price, breakeven, T, 0.045, sell["impliedVolatility"]) * 100
            liquidity = spread_liquidity_score(sell, buy)

            score = int(max(0, min(
                100,
                0.35 * stock_score +
                0.30 * prob_est +
                0.20 * liquidity +
                0.15 * min(100, rr * 100)
            )))

            rows.append({
                "Strategy": "Credit Put Spread",
                "Bias": "Bullish / Neutral",
                "Sell Leg": sell["contractSymbol"],
                "Buy Leg": buy["contractSymbol"],
                "Sell Strike": sell["strike"],
                "Buy Strike": buy["strike"],
                "Width": width,
                "Debit/Credit": round(credit * 100, 2),
                "Max Profit": round(max_profit * 100, 2),
                "Max Loss": round(max_loss * 100, 2),
                "Breakeven": round(breakeven, 2),
                "Risk/Reward": round(rr, 2),
                "Probability Estimate %": round(prob_est, 1),
                "Spread Score": score
            })

    return pd.DataFrame(rows).sort_values("Spread Score", ascending=False) if rows else pd.DataFrame()


def build_credit_call_spreads(calls, stock_price, exp, stock_score):
    df = prepare_chain(calls)
    if df.empty:
        return pd.DataFrame()

    rows = []
    df = df.sort_values("strike").reset_index(drop=True)
    T = max(dte(exp) / 365, 1 / 365)

    for i in range(len(df)):
        sell = df.iloc[i]

        for j in range(i + 1, min(i + 6, len(df))):
            buy = df.iloc[j]

            if not spread_liquidity_ok(sell, buy):
                continue

            width = buy["strike"] - sell["strike"]
            credit = sell["bid"] - buy["ask"]

            if width <= 0 or credit <= 0:
                continue

            max_profit = credit
            max_loss = width - credit
            breakeven = sell["strike"] + credit

            if max_profit <= 0 or max_loss <= 0:
                continue

            rr = max_profit / max_loss
            prob_est = prob_below_price(stock_price, breakeven, T, 0.045, sell["impliedVolatility"]) * 100
            liquidity = spread_liquidity_score(sell, buy)

            score = int(max(0, min(
                100,
                0.35 * stock_score +
                0.30 * prob_est +
                0.20 * liquidity +
                0.15 * min(100, rr * 100)
            )))

            rows.append({
                "Strategy": "Credit Call Spread",
                "Bias": "Bearish / Neutral",
                "Sell Leg": sell["contractSymbol"],
                "Buy Leg": buy["contractSymbol"],
                "Sell Strike": sell["strike"],
                "Buy Strike": buy["strike"],
                "Width": width,
                "Debit/Credit": round(credit * 100, 2),
                "Max Profit": round(max_profit * 100, 2),
                "Max Loss": round(max_loss * 100, 2),
                "Breakeven": round(breakeven, 2),
                "Risk/Reward": round(rr, 2),
                "Probability Estimate %": round(prob_est, 1),
                "Spread Score": score
            })

    return pd.DataFrame(rows).sort_values("Spread Score", ascending=False) if rows else pd.DataFrame()


def build_all_spreads(calls, puts, stock_price, exp, call_score, put_score):
    debit_calls = build_debit_call_spreads(calls, stock_price, exp, call_score)
    debit_puts = build_debit_put_spreads(puts, stock_price, exp, put_score)
    credit_puts = build_credit_put_spreads(puts, stock_price, exp, call_score)
    credit_calls = build_credit_call_spreads(calls, stock_price, exp, put_score)

    frames = [x for x in [debit_calls, debit_puts, credit_puts, credit_calls] if not x.empty]

    if not frames:
        return pd.DataFrame(), debit_calls, debit_puts, credit_puts, credit_calls

    all_spreads = pd.concat(frames, ignore_index=True)
    return all_spreads.sort_values("Spread Score", ascending=False), debit_calls, debit_puts, credit_puts, credit_calls


# ============================================================
# CHART
# ============================================================

def make_chart(df, setup, ticker):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.20, 0.22],
        subplot_titles=("Price Action", "Volume", "RSI / MACD")
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candles"
        ),
        row=1,
        col=1
    )

    for ema in ["EMA_9", "EMA_21", "EMA_50", "EMA_200"]:
        fig.add_trace(go.Scatter(x=df.index, y=df[ema], mode="lines", name=ema), row=1, col=1)

    fig.add_hline(y=setup["support"], line_dash="dash", annotation_text="Support", row=1, col=1)
    fig.add_hline(y=setup["resistance"], line_dash="dash", annotation_text="Resistance", row=1, col=1)

    if not np.isnan(setup["entry_low"]) and not np.isnan(setup["entry_high"]):
        fig.add_hrect(
            y0=setup["entry_low"],
            y1=setup["entry_high"],
            opacity=0.18,
            annotation_text="Entry Zone",
            row=1,
            col=1
        )

    if not np.isnan(setup["stop"]):
        fig.add_hline(y=setup["stop"], line_dash="dot", annotation_text="Stop", row=1, col=1)

    if not np.isnan(setup["target1"]):
        fig.add_hline(y=setup["target1"], line_dash="dot", annotation_text="Target 1", row=1, col=1)

    if not np.isnan(setup["target2"]):
        fig.add_hline(y=setup["target2"], line_dash="dot", annotation_text="Target 2", row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=3, col=1)

    fig.add_hline(y=70, line_dash="dash", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_HIST"], name="MACD Hist"), row=3, col=1)

    fig.update_layout(
        title=ticker.upper(),
        height=800,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h")
    )

    return fig


# ============================================================
# TRADE TRACKER
# ============================================================

TRADE_FILE = "trade_journal.csv"


def load_trades():
    if os.path.exists(TRADE_FILE):
        try:
            return pd.read_csv(TRADE_FILE)
        except Exception:
            pass

    return pd.DataFrame(columns=[
        "Date", "Ticker", "Trade Type", "Signal", "Contract",
        "Entry Price", "Exit Price", "Contracts",
        "Entry Value", "Exit Value", "P&L", "Return %",
        "Result", "Notes"
    ])


def save_trade(trade):
    df = load_trades()
    df = pd.concat([df, pd.DataFrame([trade])], ignore_index=True)
    df.to_csv(TRADE_FILE, index=False)


def calculate_trade_stats(df):
    if df.empty:
        return {
            "Total Trades": 0,
            "Win Rate": 0,
            "Total P&L": 0,
            "Average Win": 0,
            "Average Loss": 0,
            "Average Return": 0,
        }

    df = df.copy()
    df["P&L"] = pd.to_numeric(df["P&L"], errors="coerce").fillna(0)
    df["Return %"] = pd.to_numeric(df["Return %"], errors="coerce").fillna(0)

    total_trades = len(df)
    wins = df[df["P&L"] > 0]
    losses = df[df["P&L"] < 0]

    return {
        "Total Trades": total_trades,
        "Win Rate": len(wins) / total_trades * 100 if total_trades else 0,
        "Total P&L": df["P&L"].sum(),
        "Average Win": wins["P&L"].mean() if not wins.empty else 0,
        "Average Loss": losses["P&L"].mean() if not losses.empty else 0,
        "Average Return": df["Return %"].mean() if total_trades else 0
    }


# ============================================================
# SCANNER
# ============================================================

def run_scanner(inputs, period, interval):
    rows = []
    market = market_filter()

    for raw_input in inputs:
        ticker = resolve_ticker(raw_input)

        if not ticker:
            continue

        try:
            df = add_indicators(get_price_data(ticker, period, interval))

            if df.empty or len(df) < 2:
                continue

            news_df = get_news(ticker)
            news_score, news_label = score_news_sentiment(news_df)
            earnings_risk, _, _ = get_earnings_warning(ticker)

            setup = score_stock_setup(df, news_score, earnings_risk)

            if setup is None:
                continue

            adjusted = setup["stock_score"]

            if setup["preferred_side"] == "call" and not market["call_ok"]:
                adjusted -= 15

            if setup["preferred_side"] == "put" and not market["put_ok"]:
                adjusted -= 15

            adjusted = int(max(0, min(100, adjusted)))

            rows.append({
                "Input": raw_input,
                "Ticker": ticker,
                "Price": round(setup["price"], 2),
                "Decision": setup["signal"],
                "Preferred Side": setup["preferred_side"].upper(),
                "Stock Setup Score": setup["stock_score"],
                "Call Score": setup["call_score"],
                "Put Score": setup["put_score"],
                "News": news_label,
                "News Score": news_score,
                "Earnings Risk": earnings_risk,
                "Market Bias": market["bias"],
                "Adjusted Setup Score": adjusted,
                "RSI": round(setup["rsi"], 1),
                "Support": round(setup["support"], 2),
                "Resistance": round(setup["resistance"], 2),
            })

        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("Adjusted Setup Score", ascending=False)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Terminal Controls")

mode = st.sidebar.radio(
    "Mode",
    ["Single Ticker", "Multi-Stock Scanner", "Trade Tracker"]
)

ticker_input = st.sidebar.text_input("Ticker or Company Name", "Apple")
ticker = resolve_ticker(ticker_input)

if ticker:
    st.sidebar.caption(f"Resolved ticker: {ticker}")

period = st.sidebar.selectbox(
    "Chart Period",
    ["1mo", "3mo", "6mo", "1y", "2y"],
    index=2
)

interval = st.sidebar.selectbox(
    "Chart Interval",
    ["1d", "1h", "30m", "15m"],
    index=0
)

if interval in ["15m", "30m"] and period != "1mo":
    st.sidebar.warning("15m/30m data works best with 1mo. Switching period to 1mo.")
    period = "1mo"

elif interval == "1h" and period in ["1y", "2y"]:
    st.sidebar.warning("1h data works best with 6mo or less. Switching period to 6mo.")
    period = "6mo"

st.sidebar.markdown("---")
st.sidebar.subheader("Option Quality Filters")

min_delta = st.sidebar.slider("Min Delta", 0.05, 0.95, 0.30, 0.05)
max_delta = st.sidebar.slider("Max Delta", 0.05, 0.95, 0.60, 0.05)

if min_delta > max_delta:
    st.sidebar.error("Min Delta cannot be greater than Max Delta.")
    st.stop()

min_volume = st.sidebar.number_input("Min Option Volume", min_value=0, value=50, step=10)
min_oi = st.sidebar.number_input("Min Open Interest", min_value=0, value=100, step=25)
max_spread_pct = st.sidebar.slider("Max Bid/Ask Spread %", 1.0, 50.0, 15.0, 1.0)
max_cost = st.sidebar.number_input("Max Contract Cost", min_value=1, value=1500, step=50)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Rules")

risk_per_trade = st.sidebar.number_input("Max $ Risk Per Trade", min_value=10, value=250, step=25)
stop_loss_pct = st.sidebar.slider("Option Stop Loss %", 5, 100, 25, 5)
profit_target_pct = st.sidebar.slider("Option Profit Target %", 10, 200, 50, 5)


# ============================================================
# TRADE TRACKER MODE
# ============================================================

if mode == "Trade Tracker":
    st.subheader("Trade Tracker")

    st.info(
        "On Streamlit Cloud, CSV storage may reset when the app restarts. "
        "For permanent tracking, use Google Sheets or a database later."
    )

    with st.form("trade_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            trade_date = st.date_input("Trade Date", value=date.today())
            trade_ticker_input = st.text_input("Trade Ticker or Company Name", "Apple")
            trade_ticker = resolve_ticker(trade_ticker_input)

            trade_type = st.selectbox(
                "Trade Type",
                ["Long Call / Long Put", "Debit Spread", "Credit Spread"]
            )

            signal = st.selectbox(
                "Signal",
                [
                    "TRADE CANDIDATE",
                    "WATCHLIST",
                    "FIND BETTER CONTRACT",
                    "AVOID",
                    "MANUAL"
                ]
            )

        with col2:
            contract = st.text_input("Contract / Spread Legs", "")
            entry_price = st.number_input("Entry Price / Net Debit or Credit", min_value=0.00, value=1.00, step=0.01)
            exit_price = st.number_input("Exit Price / Closing Value", min_value=0.00, value=0.50, step=0.01)
            contracts = st.number_input("Number of Contracts", min_value=1, value=1, step=1)

        with col3:
            st.caption(f"Resolved trade ticker: {trade_ticker}")
            notes = st.text_area("Notes", "")

        submitted = st.form_submit_button("Save Trade")

        if submitted:
            entry_value = entry_price * 100 * contracts
            exit_value = exit_price * 100 * contracts

            if trade_type == "Credit Spread":
                pnl = entry_value - exit_value
                return_pct = (pnl / entry_value * 100) if entry_value > 0 else 0
            else:
                pnl = exit_value - entry_value
                return_pct = (pnl / entry_value * 100) if entry_value > 0 else 0

            result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"

            save_trade({
                "Date": trade_date,
                "Ticker": trade_ticker,
                "Trade Type": trade_type,
                "Signal": signal,
                "Contract": contract,
                "Entry Price": entry_price,
                "Exit Price": exit_price,
                "Contracts": contracts,
                "Entry Value": entry_value,
                "Exit Value": exit_value,
                "P&L": pnl,
                "Return %": return_pct,
                "Result": result,
                "Notes": notes
            })

            st.success(f"Trade saved: {result} | P&L: ${pnl:.2f} | Return: {return_pct:.1f}%")

    trades = load_trades()

    if trades.empty:
        st.info("No trades logged yet.")
    else:
        stats = calculate_trade_stats(trades)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades", stats["Total Trades"])
        c2.metric("Win Rate", f"{stats['Win Rate']:.1f}%")
        c3.metric("Total P&L", f"${stats['Total P&L']:.2f}")
        c4.metric("Avg Win", f"${stats['Average Win']:.2f}")
        c5.metric("Avg Loss", f"${stats['Average Loss']:.2f}")

        st.metric("Average Return", f"{stats['Average Return']:.1f}%")
        st.dataframe(trades, use_container_width=True, height=500)

        st.download_button(
            "Download Trade Journal CSV",
            trades.to_csv(index=False),
            file_name="trade_journal.csv",
            mime="text/csv"
        )

    st.stop()


# ============================================================
# SCANNER MODE
# ============================================================

if mode == "Multi-Stock Scanner":
    st.subheader("Multi-Stock Scanner")

    default_list = "Apple, Microsoft, Nvidia, Tesla, Amazon, Meta, Google, AMD, Netflix, SPY, QQQ"
    tickers_text = st.text_area("Tickers or Company Names", default_list, height=110)

    if st.button("Run Scanner"):
        inputs = [x.strip() for x in tickers_text.split(",") if x.strip()]

        with st.spinner("Scanning stocks..."):
            scan = run_scanner(inputs, period, interval)

        if scan.empty:
            st.warning("No scanner results found. Try 6mo/1d or 1y/1d.")
        else:
            st.dataframe(scan, use_container_width=True, height=560)

    st.stop()


# ============================================================
# SINGLE TICKER MODE
# ============================================================

if not ticker:
    st.warning("Enter a ticker or company name.")
    st.stop()

raw_df = get_price_data(ticker, period, interval)

if raw_df.empty:
    st.error(f"No price data returned for {ticker}. Try a different ticker or company name.")
    st.stop()

df = add_indicators(raw_df)

if df.empty or len(df) < 2:
    st.error("Not enough usable data after indicators. Try 1y/1d or another ticker.")
    st.stop()

market = market_filter()
news_df = get_news(ticker)
news_score, news_label = score_news_sentiment(news_df)
earnings_risk, earnings_date, earnings_days = get_earnings_warning(ticker)

setup = score_stock_setup(df, news_score, earnings_risk)

if setup is None:
    st.error("Not enough data to generate a setup.")
    st.stop()


# ============================================================
# DASHBOARD
# ============================================================

st.subheader("Trade Decision Dashboard")

top1, top2, top3, top4 = st.columns(4)
with top1:
    display_card("Input", ticker_input, font_size=26)
with top2:
    display_card("Resolved Ticker", ticker, font_size=32)
with top3:
    display_card("Price", f"${setup['price']:.2f}", font_size=32)
with top4:
    display_card("Preferred Side", setup["preferred_side"].upper(), font_size=32)

st.markdown("")

m1, m2, m3, m4 = st.columns(4)
with m1:
    display_card("Stock Setup Score", setup["stock_score"], font_size=32)
with m2:
    display_card("Call Setup Score", setup["call_score"], font_size=32)
with m3:
    display_card("Put Setup Score", setup["put_score"], font_size=32)
with m4:
    display_card("Setup", setup["signal"], font_size=24)

if setup["signal"] == "NO TRADE / WAIT":
    st.warning("Current stock setup is weak. A good option contract alone does not make this a trade.")
elif "STRONG" in setup["signal"]:
    st.success("Stock setup is strong. Now confirm option quality and risk.")
else:
    st.info("This is a watchlist setup. Wait for better confirmation or cleaner entry.")


# ============================================================
# MARKET / NEWS / EARNINGS
# ============================================================

st.subheader("Market, News, and Earnings Risk")

risk1, risk2, risk3, risk4 = st.columns(4)
with risk1:
    display_card("Market Bias", market["bias"], font_size=24)
with risk2:
    display_card("VIX", "-" if np.isnan(market["vix"]) else f"{market['vix']:.2f}", font_size=32)
with risk3:
    display_card("News Sentiment", news_label, font_size=24)
with risk4:
    display_card("Earnings Risk", earnings_risk, font_size=24)

if earnings_date:
    st.write(f"**Earnings Date:** {earnings_date} | **Days Away:** {earnings_days}")

if earnings_risk == "High":
    st.error("High earnings risk. Avoid unless intentionally trading earnings.")
elif earnings_risk == "Medium":
    st.warning("Medium earnings risk. Watch for IV crush.")


# ============================================================
# TECHNICAL METRICS
# ============================================================

st.subheader("Technical Levels")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("RSI", f"{setup['rsi']:.1f}")
c2.metric("Support", f"${setup['support']:.2f}")
c3.metric("Resistance", f"${setup['resistance']:.2f}")
c4.metric("ATR", f"${setup['atr']:.2f}")
c5.metric("Volume Ratio", f"{setup['vol_ratio']:.2f}x")

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("Entry Low", "-" if np.isnan(setup["entry_low"]) else f"${setup['entry_low']:.2f}")
c7.metric("Entry High", "-" if np.isnan(setup["entry_high"]) else f"${setup['entry_high']:.2f}")
c8.metric("Stop", "-" if np.isnan(setup["stop"]) else f"${setup['stop']:.2f}")
c9.metric("Target 1", "-" if np.isnan(setup["target1"]) else f"${setup['target1']:.2f}")
c10.metric("Target 2", "-" if np.isnan(setup["target2"]) else f"${setup['target2']:.2f}")

with st.expander("Stock Setup Reasons"):
    left, right = st.columns(2)

    with left:
        st.markdown("### Call Setup Reasons")
        if setup["call_reasons"]:
            for reason in setup["call_reasons"]:
                st.write(f"- {reason}")
        else:
            st.write("No strong call reasons.")

    with right:
        st.markdown("### Put Setup Reasons")
        if setup["put_reasons"]:
            for reason in setup["put_reasons"]:
                st.write(f"- {reason}")
        else:
            st.write("No strong put reasons.")

st.plotly_chart(make_chart(df, setup, ticker), use_container_width=True)


# ============================================================
# OPTIONS + FINAL DECISION
# ============================================================

st.subheader("Options Contract Analysis")

try:
    expirations = get_expirations(ticker)

    if not expirations:
        st.warning("No options found for this ticker.")
        st.stop()

    exp = st.selectbox("Expiration Date", expirations)

    side = st.radio(
        "Option Side",
        ["call", "put"],
        index=0 if setup["preferred_side"] == "call" else 1
    )

    calls, puts = get_chain(ticker, exp)
    chain = calls if side == "call" else puts

    option_df = build_options_table(
        chain=chain,
        option_type=side,
        stock_price=setup["price"],
        exp=exp,
        stock_setup_score=setup["stock_score"],
        min_delta=min_delta,
        max_delta=max_delta,
        min_volume=min_volume,
        min_oi=min_oi,
        max_spread_pct=max_spread_pct,
        max_cost=max_cost
    )

    if option_df.empty:
        st.warning("No usable option contracts found after filtering bad bid/ask data.")
    else:
        best = option_df.iloc[0]

        option_score = int(best["option_quality_score"])
        final_score = int(best["final_trade_score"])

        market_conflict = (
            (side == "call" and not market["call_ok"]) or
            (side == "put" and not market["put_ok"])
        )

        decision, decision_reason = decision_label(
            setup["stock_score"],
            option_score,
            final_score,
            earnings_risk,
            market_conflict
        )

        st.subheader("Final Trade Decision")

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            display_card("Stock Setup Score", setup["stock_score"], font_size=32)
        with d2:
            display_card("Option Quality Score", option_score, font_size=32)
        with d3:
            display_card("Final Trade Score", final_score, font_size=32)
        with d4:
            display_card("Decision", decision, font_size=22)

        if decision == "TRADE CANDIDATE":
            st.success(f"{decision}: {decision_reason}")
        elif decision in ["FIND BETTER CONTRACT", "WATCHLIST"]:
            st.warning(f"{decision}: {decision_reason}")
        else:
            st.error(f"{decision}: {decision_reason}")

        with st.expander("Clean Trade Checklist"):
            checklist = {
                "Stock setup score 60+": setup["stock_score"] >= 60,
                "Option quality score 60+": option_score >= 60,
                "Final trade score 65+": final_score >= 65,
                "Delta within range": min_delta <= abs(best["delta"]) <= max_delta,
                "Bid/ask spread acceptable": best["spread_pct"] <= max_spread_pct,
                "Volume acceptable": best["volume"] >= min_volume,
                "Open interest acceptable": best["openInterest"] >= min_oi,
                "Contract cost acceptable": best["cost_ask"] <= max_cost,
                "Earnings risk not high": earnings_risk != "High",
                "Market does not conflict": not market_conflict
            }

            for item, passed in checklist.items():
                st.write(("✅ " if passed else "❌ ") + item)

        display = option_df.copy()

        for col in ["lastPrice", "bid", "ask", "mid"]:
            display[col] = display[col].round(2)

        display["cost_mid"] = display["cost_mid"].round(0)
        display["cost_ask"] = display["cost_ask"].round(0)
        display["spread_pct"] = display["spread_pct"].round(1)
        display["impliedVolatility"] = (display["impliedVolatility"] * 100).round(1)
        display["delta"] = display["delta"].round(2)
        display["prob_itm"] = (display["prob_itm"] * 100).round(1)

        display = display.rename(columns={
            "contractSymbol": "Contract",
            "strike": "Strike",
            "lastPrice": "Last",
            "bid": "Bid",
            "ask": "Ask",
            "mid": "Mid",
            "cost_mid": "Cost Mid",
            "cost_ask": "Cost Ask",
            "spread_pct": "Spread %",
            "volume": "Volume",
            "openInterest": "Open Interest",
            "impliedVolatility": "IV %",
            "delta": "Delta",
            "prob_itm": "Prob ITM %",
            "option_quality_score": "Option Quality Score",
            "final_trade_score": "Final Trade Score"
        })

        st.dataframe(display, use_container_width=True, height=450)

        st.markdown("### Best Contract Summary")

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Strike", f"${best['strike']:.2f}")
        b2.metric("Ask Cost", f"${best['cost_ask']:.0f}")
        b3.metric("Delta", f"{best['delta']:.2f}")
        b4.metric("Prob ITM", f"{best['prob_itm'] * 100:.1f}%")
        b5.metric("Final Score", f"{final_score}")

        risk_per_contract = best["cost_ask"] * (stop_loss_pct / 100)
        max_contracts = int(risk_per_trade // max(risk_per_contract, 1))

        st.markdown("### Trade Plan")
        st.write(f"**Contract:** {best['contractSymbol']}")
        st.write(f"**Expiration:** {exp}")
        st.write(f"**Side:** {side.upper()}")
        st.write(f"**Ask cost per contract:** ${best['cost_ask']:.0f}")
        st.write(f"**Estimated risk per contract at {stop_loss_pct}% stop:** ${risk_per_contract:.0f}")
        st.write(f"**Max contracts based on ${risk_per_trade:.0f} risk:** {max_contracts}")
        st.write(f"**Profit target:** +{profit_target_pct}% option gain")
        st.write(f"**Stop loss:** -{stop_loss_pct}% option loss or break of stock technical stop")
        st.write(f"**Days to expiration:** {dte(exp)}")

    st.subheader("Options Spread Strategy Optimizer")

    with st.spinner("Building spread strategies..."):
        all_spreads, debit_calls, debit_puts, credit_puts, credit_calls = build_all_spreads(
            calls=calls,
            puts=puts,
            stock_price=setup["price"],
            exp=exp,
            call_score=setup["call_score"],
            put_score=setup["put_score"]
        )

    if all_spreads.empty:
        st.warning("No valid spread strategies found for this expiration.")
    else:
        best_spread = all_spreads.iloc[0]

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Best Strategy", best_spread["Strategy"])
        s2.metric("Spread Score", f"{best_spread['Spread Score']}")
        s3.metric("Max Profit", f"${best_spread['Max Profit']:.0f}")
        s4.metric("Max Loss", f"${best_spread['Max Loss']:.0f}")
        s5.metric("Breakeven", f"${best_spread['Breakeven']:.2f}")

        st.markdown("### Best Overall Spread")
        st.dataframe(pd.DataFrame([best_spread]), use_container_width=True)

        strategy_filter = st.selectbox(
            "View Spread Strategy",
            [
                "All Spreads",
                "Debit Call Spread",
                "Debit Put Spread",
                "Credit Put Spread",
                "Credit Call Spread"
            ]
        )

        if strategy_filter == "All Spreads":
            spread_display = all_spreads
        elif strategy_filter == "Debit Call Spread":
            spread_display = debit_calls
        elif strategy_filter == "Debit Put Spread":
            spread_display = debit_puts
        elif strategy_filter == "Credit Put Spread":
            spread_display = credit_puts
        else:
            spread_display = credit_calls

        if spread_display.empty:
            st.info("No spreads available for this strategy.")
        else:
            st.dataframe(spread_display.head(25), use_container_width=True, height=500)

        st.markdown("### Spread Strategy Notes")
        st.write("- **Debit Call Spread:** bullish, defined risk, cheaper than buying a call.")
        st.write("- **Debit Put Spread:** bearish, defined risk, cheaper than buying a put.")
        st.write("- **Credit Put Spread:** bullish/neutral, collects premium, wins if price stays above breakeven.")
        st.write("- **Credit Call Spread:** bearish/neutral, collects premium, wins if price stays below breakeven.")
        st.write("- Probability estimates are simplified and should be treated as risk estimates, not guarantees.")

except Exception as e:
    st.error(f"Options error: {e}")


# ============================================================
# NEWS
# ============================================================

st.subheader("Recent News")

if news_df.empty:
    st.info("No recent news found.")
else:
    for _, row in news_df.iterrows():
        title = row.get("Title", "")
        publisher = row.get("Publisher", "")
        published = row.get("Published", "")
        link = row.get("Link", "")

        if link:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #e5e7eb;
                    border-radius: 12px;
                    padding: 14px;
                    margin-bottom: 10px;
                    background: white;
                ">
                    <div style="font-size: 18px; font-weight: 700;">
                        <a href="{link}" target="_blank" style="text-decoration: none;">
                            {title}
                        </a>
                    </div>
                    <div style="font-size: 13px; color: #6b7280; margin-top: 6px;">
                        {publisher} | {published}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.write(f"**{title}**")
            st.caption(f"{publisher} | {published}")