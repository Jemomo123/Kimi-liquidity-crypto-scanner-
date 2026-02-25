import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import functools

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    COMPRESSION_THRESHOLD = 0.002
    SCAN_PAIRS = 40  # Reduced from 60 for Cloud safety
    RATE_LIMIT_DELAY = 0.05  # Slightly faster
    CACHE_TTL = 120  # 2 minutes
    
    DEFAULT_WATCHLIST = ["BTC", "ETH", "SOL", "DOGE", "PEPE", "WIF", "PUMP"]

# =============================================================================
# DATA CLASSES
# =============================================================================

class SignalType(Enum):
    COMPRESSION = "COMPRESSION"
    EXPANSION = "EXPANSION"
    PULLBACK = "PULLBACK"
    REVERSAL = "REVERSAL"

class CompressionType(Enum):
    NONE = "NONE"
    SQZ = "SQZ"
    CROSSOVER = "CROSSOVER"

class Direction(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class Signal:
    pair: str
    exchange: str
    signal_type: SignalType
    compression_type: CompressionType
    direction: Direction
    timeframe: str
    price: float
    spread_pct: float
    candle_type: Optional[str]
    rsi: float
    conviction: int
    conviction_tier: str
    timestamp: datetime

# =============================================================================
# CACHED API FUNCTIONS (Critical for Cloud)
# =============================================================================

@st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
def fetch_okx_klines(symbol: str, timeframe: str = "3m") -> pd.DataFrame:
    """Cached OKX data fetch"""
    try:
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": f"{symbol}-USDT-SWAP", "bar": timeframe, "limit": 130}
        response = requests.get(url, params=params, timeout=8)
        data = response.json()
        
        if data.get("code") != "0" or not data.get("data"):
            return pd.DataFrame()
        
        df = pd.DataFrame(data["data"], columns=[
            "timestamp", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "vol"]:
            df[col] = df[col].astype(float)
        return df.sort_values("timestamp").reset_index(drop=True)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
def fetch_gateio_klines(symbol: str, timeframe: str = "3m") -> pd.DataFrame:
    """Cached Gate.io data fetch"""
    try:
        url = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
        interval_map = {"3m": "3m", "5m": "5m", "15m": "15m", "1H": "1h", "4H": "4h"}
        params = {"contract": f"{symbol}_USDT", "interval": interval_map.get(timeframe, "3m"), "limit": 130}
        response = requests.get(url, params=params, timeout=8)
        data = response.json()
        
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=["t", "v", "c", "h", "l", "o"])
        df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "vol"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        for col in ["open", "high", "low", "close", "vol"]:
            df[col] = df[col].astype(float)
        return df.sort_values("timestamp").reset_index(drop=True)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)  # 5 min cache for volume data
def get_okx_top_pairs(limit: int = 25) -> List[str]:
    """Get top volume pairs - cached"""
    try:
        url = "https://www.okx.com/api/v5/market/tickers"
        params = {"instType": "SWAP", "limit": 50}
        response = requests.get(url, params=params, timeout=8)
        data = response.json()
        
        if data.get("code") != "0":
            return []
        
        tickers = [t for t in data["data"] if t["instId"].endswith("-USDT-SWAP")]
        tickers.sort(key=lambda x: float(x.get("volCcy24h", 0)), reverse=True)
        return [t["instId"].replace("-USDT-SWAP", "") for t in tickers[:limit]]
    except:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def get_gateio_top_pairs(limit: int = 15) -> List[str]:
    """Get top Gate.io pairs - cached"""
    try:
        url = "https://api.gateio.ws/api/v4/futures/usdt/contracts"
        response = requests.get(url, timeout=8)
        data = response.json()
        
        if not isinstance(data, list):
            return []
        
        contracts = [c for c in data if c.get("name", "").endswith("_USDT")]
        contracts.sort(key=lambda x: float(x.get("volume_24h_usd", 0) or 0), reverse=True)
        return [c["name"].replace("_USDT", "") for c in contracts[:limit]]
    except:
        return []

# =============================================================================
# TECHNICAL ANALYSIS
# =============================================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA and RSI"""
    if len(df) < 100:
        return df
    
    df = df.copy()
    df["sma20"] = df["close"].rolling(window=20, min_periods=20).mean()
    df["sma100"] = df["close"].rolling(window=100, min_periods=100).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    df["rsi14"] = 100 - (100 / (1 + rs))
    
    return df

def cluster_spread(row) -> float:
    """Calculate cluster spread %"""
    cluster = [row["close"], row["sma20"], row["sma100"]]
    return ((max(cluster) - min(cluster)) / row["close"]) * 100

def classify_candle(row, avg_body: float) -> Tuple[Optional[str], Optional[str]]:
    """Classify as elephant or tail"""
    body = abs(row["close"] - row["open"])
    total_range = row["high"] - row["low"]
    
    if total_range == 0:
        return None, None
    
    # Elephant: body > 1.5x avg and > 40% of range
    if body >= avg_body * 1.5 and body >= total_range * 0.4:
        direction = "long" if row["close"] > row["open"] else "short"
        return "elephant", direction
    
    # Tail: wick > 60% of range
    upper_wick = row["high"] - max(row["open"], row["close"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    
    if lower_wick >= total_range * 0.6:
        return "tail", "long"
    if upper_wick >= total_range * 0.6:
        return "tail", "short"
    
    return None, None

# =============================================================================
# SIGNAL DETECTION (Forward Only)
# =============================================================================

def detect_signals(df: pd.DataFrame) -> Optional[Signal]:
    """Detect all signal types - forward only"""
    if len(df) < 102:  # Need 100 for SMA + 2 for detection
        return None
    
    df = add_indicators(df)
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Skip if indicators not ready
    if pd.isna(curr["sma100"]):
        return None
    
    # Calculate metrics
    curr_spread = cluster_spread(curr)
    prev_spread = cluster_spread(prev)
    avg_body = df["close"].diff().abs().tail(20).mean()
    
    # Compression detection
    compression_type = CompressionType.NONE
    if curr_spread <= 0.20:
        # Check crossover
        prev_cross = (prev["sma20"] > prev["sma100"]) != (curr["sma20"] > curr["sma100"])
        sma_gap = abs(curr["sma20"] - curr["sma100"]) / curr["close"] * 100
        
        if prev_cross or sma_gap <= 0.05:
            compression_type = CompressionType.CROSSOVER
        else:
            # Count consecutive compressed (simplified - just check last 3)
            recent_spreads = [cluster_spread(df.iloc[-i]) for i in range(1, 4)]
            if all(s <= 0.20 for s in recent_spreads):
                compression_type = CompressionType.SQZ
            else:
                compression_type = CompressionType.NONE  # Building but not active
    
    # EXPANSION: Was compressed, now breaking out with momentum
    if prev_spread <= 0.20 and curr_spread > 0.20:
        candle_type, direction = classify_candle(curr, avg_body)
        if candle_type:
            return create_signal(SignalType.EXPANSION, compression_type, direction, 
                               candle_type, curr, curr_spread, df)
    
    # PULLBACK: Touching SMA20 and bouncing
    sma20 = curr["sma20"]
    candle_type, _ = classify_candle(curr, avg_body)
    
    if candle_type:
        # Long pullback
        if (curr["open"] > sma20 and curr["low"] <= sma20 * 1.002 and 
            curr["close"] > sma20 and curr["close"] > curr["open"]):
            return create_signal(SignalType.PULLBACK, compression_type, "long",
                               candle_type, curr, curr_spread, df)
        
        # Short pullback
        if (curr["open"] < sma20 and curr["high"] >= sma20 * 0.998 and
            curr["close"] < sma20 and curr["close"] < curr["open"]):
            return create_signal(SignalType.PULLBACK, compression_type, "short",
                               candle_type, curr, curr_spread, df)
    
    # REVERSAL: Stretched trend hitting SMA100
    trend_gap = abs(curr["sma20"] - curr["sma100"]) / curr["sma100"] * 100
    if trend_gap >= 1.2:
        if curr["sma20"] > curr["sma100"]:  # Uptrend
            if curr["low"] <= curr["sma100"] * 1.005:
                candle_type, _ = classify_candle(curr, avg_body)
                if candle_type:
                    return create_signal(SignalType.REVERSAL, CompressionType.NONE, "short",
                                       candle_type, curr, curr_spread, df)
        else:  # Downtrend
            if curr["high"] >= curr["sma100"] * 0.995:
                candle_type, _ = classify_candle(curr, avg_body)
                if candle_type:
                    return create_signal(SignalType.REVERSAL, CompressionType.NONE, "long",
                                       candle_type, curr, curr_spread, df)
    
    # COMPRESSION: Active but no breakout yet
    if compression_type in [CompressionType.SQZ, CompressionType.CROSSOVER]:
        return create_signal(SignalType.COMPRESSION, compression_type, "neutral",
                           None, curr, curr_spread, df)
    
    return None

def create_signal(sig_type, comp_type, direction, candle_type, curr, spread, df) -> Signal:
    """Create signal object with conviction scoring"""
    
    # Conviction scoring
    score = 0
    
    # Compression quality
    if spread <= 0.10: score += 25
    elif spread <= 0.15: score += 20
    elif spread <= 0.20: score += 15
    
    # Freshness (always fresh with forward detection)
    score += 25
    
    # Candle strength
    if candle_type == "elephant": score += 20
    elif candle_type == "tail": score += 15
    
    # Room ahead (simplified)
    recent_range = (df["high"].tail(20).max() - df["low"].tail(20).min()) / curr["close"] * 100
    if recent_range > 4: score += 15
    elif recent_range > 2: score += 8
    
    # RSI
    rsi = curr.get("rsi14", 50)
    if direction == "long" and rsi < 40: score += 10
    elif direction == "short" and rsi > 60: score += 10
    elif 40 <= rsi <= 60: score += 5
    
    # Tier
    if score >= 75: tier = "HIGH"
    elif score >= 50: tier = "MEDIUM"
    else: tier = "WATCH"
    
    return Signal(
        pair="",  # Filled by caller
        exchange="",  # Filled by caller
        signal_type=sig_type,
        compression_type=comp_type,
        direction=Direction(direction) if direction != "neutral" else Direction.NEUTRAL,
        timeframe="3m",
        price=curr["close"],
        spread_pct=spread,
        candle_type=candle_type,
        rsi=rsi,
        conviction=score,
        conviction_tier=tier,
        timestamp=datetime.now()
    )

# =============================================================================
# BTC ANALYSIS (Cached)
# =============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_btc_analysis():
    """Get BTC regime and liquidity - heavily cached"""
    try:
        # Fetch 1H data
        df = fetch_okx_klines("BTC", "1H")
        if len(df) < 100:
            return {}
        
        df = add_indicators(df)
        curr = df.iloc[-1]
        
        # Regime
        gap = abs(curr["sma20"] - curr["sma100"]) / curr["sma100"] * 100
        if gap >= 1.0:
            trend = "Trending Up" if curr["sma20"] > curr["sma100"] else "Trending Down"
        else:
            trend = "Ranging"
        
        spread = cluster_spread(curr)
        comp = "SQZ" if spread <= 0.15 else "CROSS" if spread <= 0.20 else "NONE"
        
        # Simple sweep detection (last 5 candles)
        sweep = "None"
        for i in range(-5, -1):
            if i >= -len(df):
                c = df.iloc[i]
                # Bear sweep: high breaks recent high, close below
                if c["high"] > df["high"].iloc[i-3:i].max() and c["close"] < c["high"]:
                    sweep = "Bear Sweep"
                    break
                # Bull sweep
                if c["low"] < df["low"].iloc[i-3:i].min() and c["close"] > c["low"]:
                    sweep = "Bull Sweep"
                    break
        
        return {
            "trend": trend,
            "compression": comp,
            "sweep": sweep,
            "price": curr["close"],
            "sma20": curr["sma20"],
            "in_zone": abs(curr["close"] - curr["sma20"]) / curr["sma20"] * 100 <= 0.38
        }
    except:
        return {}

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(page_title="Crypto Scanner", page_icon="🎯", layout="wide")
    
    # Custom styling
    st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { background-color: #ff4b4b; color: white; width: 100%; }
    .signal-card { 
        padding: 15px; 
        margin: 10px 0; 
        border-radius: 5px; 
        background: rgba(255,255,255,0.05);
        border-left: 4px solid;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 Crypto Expansion Edge Scanner")
    st.caption("Forward-only detection | Streamlit Cloud Optimized")
    
    # Sidebar
    st.sidebar.header("Settings")
    watchlist = st.sidebar.text_input("Watchlist", ", ".join(Config.DEFAULT_WATCHLIST))
    watchlist_symbols = [s.strip().upper() for s in watchlist.split(",") if s.strip()]
    
    max_pairs = st.sidebar.slider("Max Pairs", 20, 60, Config.SCAN_PAIRS)
    show_compression = st.sidebar.toggle("Show Compression", True)
    
    # BTC Context
    btc = get_btc_analysis()
    if btc:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("BTC Trend", btc.get("trend", "N/A"))
        col2.metric("Compression", btc.get("compression", "N/A"))
        col3.metric("Last Sweep", btc.get("sweep", "None"))
        col4.metric("In Zone", "Yes" if btc.get("in_zone") else "No")
    
    st.divider()
    
    # Scanning
    if st.button("🚀 SCAN NOW", type="primary") or st.session_state.get('auto_scan', False):
        
        # Build scan list
        with st.spinner("Loading pairs..."):
            okx_pairs = get_okx_top_pairs(max_pairs // 2)
            time.sleep(0.1)
            gate_pairs = get_gateio_top_pairs(max_pairs // 3)
        
        # Prioritize watchlist
        scan_list = []
        for sym in watchlist_symbols:
            scan_list.append((sym, "OKX", fetch_okx_klines))
        
        # Add volume pairs
        for sym in okx_pairs:
            if sym not in watchlist_symbols and len(scan_list) < max_pairs:
                scan_list.append((sym, "OKX", fetch_okx_klines))
        
        for sym in gate_pairs:
            if sym not in watchlist_symbols and len(scan_list) < max_pairs:
                scan_list.append((sym, "Gate.io", fetch_gateio_klines))
        
        # Scan
        progress = st.progress(0)
        signals = []
        
        for idx, (symbol, exchange, fetch_fn) in enumerate(scan_list):
            progress.progress((idx + 1) / len(scan_list))
            
            # Try 3m
            df = fetch_fn(symbol, "3m")
            if len(df) > 0:
                sig = detect_signals(df)
                if sig:
                    sig.pair = symbol
                    sig.exchange = exchange
                    sig.timeframe = "3m"
                    signals.append(sig)
            
            time.sleep(Config.RATE_LIMIT_DELAY)
            
            # Try 5m
            df = fetch_fn(symbol, "5m")
            if len(df) > 0:
                sig = detect_signals(df)
                if sig:
                    sig.pair = symbol
                    sig.exchange = exchange
                    sig.timeframe = "5m"
                    signals.append(sig)
            
            time.sleep(Config.RATE_LIMIT_DELAY)
        
        progress.empty()
        
        # Sort: Priority then conviction
        priority = {SignalType.REVERSAL: 0, SignalType.EXPANSION: 1, 
                   SignalType.PULLBACK: 2, SignalType.COMPRESSION: 3}
        signals.sort(key=lambda x: (priority.get(x.signal_type, 4), -x.conviction))
        
        # Filter
        if not show_compression:
            signals = [s for s in signals if s.signal_type != SignalType.COMPRESSION]
        
        # Display
        st.subheader(f"Signals Found: {len(signals)}")
        
        for sig in signals:
            # Color coding
            color = "#00c853" if sig.direction == Direction.LONG else "#ff1744" if sig.direction == Direction.SHORT else "#ffd600"
            emoji = "▲" if sig.direction == Direction.LONG else "▼" if sig.direction == Direction.SHORT else "◆"
            comp_badge = f"<span style='color:orange'>{sig.compression_type.value}</span>" if sig.compression_type != CompressionType.NONE else "<span style='color:gray'>NONE</span>"
            candle_emoji = "🐘" if sig.candle_type == "elephant" else "🐾" if sig.candle_type == "tail" else "—"
            
            st.markdown(f"""
            <div class="signal-card" style="border-left-color: {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <h4 style="margin: 0;">{emoji} {sig.signal_type.value} {sig.pair}</h4>
                        <small>{sig.timeframe} · {sig.exchange} · ${sig.price:.5f}</small>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        {comp_badge}<br>
                        <small>{sig.spread_pct:.2f}%</small>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        {candle_emoji} {sig.candle_type or '—'}<br>
                        <small>RSI: {sig.rsi:.0f}</small>
                    </div>
                    <div style="flex: 1; text-align: right;">
                        <span style="color: {'green' if sig.conviction_tier == 'HIGH' else 'orange' if sig.conviction_tier == 'MEDIUM' else 'gray'}; font-weight: bold;">
                            {sig.conviction_tier}
                        </span><br>
                        <small>{sig.conviction}/100</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if not signals:
            st.info("No active signals. Compression setups may be building.")

if __name__ == "__main__":
    main()
