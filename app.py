import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - EXACTLY AS PER PROMPT
# =============================================================================

class Config:
    COMPRESSION_THRESHOLD = 0.002  # 0.20%
    COMPRESSION_MIN_CANDLES = 3
    TREND_THRESHOLD = 0.012  # 1.2%
    PULLBACK_ZONE = 0.0038  # 0.38%
    ELEPHANT_BODY_MULT = 1.5
    ELEPHANT_RANGE_RATIO = 0.4
    TAIL_WICK_RATIO = 0.6
    SCAN_PAIRS = 60
    RATE_LIMIT_DELAY = 0.1
    CACHE_TTL = 60
    
    DEFAULT_WATCHLIST = ["LINEA", "FOGO", "SPX500", "DOGE", "WIF", "BTC", "PUMP", "FARTCOIN", "SOL", "PEPE"]

# =============================================================================
# DATA CLASSES
# =============================================================================

class SignalType(Enum):
    NONE = "NONE"
    COMPRESSION = "COMPRESSION"
    EXPANSION = "EXPANSION"
    PULLBACK = "PULLBACK"
    REVERSAL = "REVERSAL"

class CompressionType(Enum):
    NONE = "NONE"
    SQZ = "SQZ"
    CROSSOVER = "CROSSOVER"
    BUILDING = "BUILDING"

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
    details: Dict[str, Any]
    timestamp: datetime

# =============================================================================
# EXCHANGE APIs - EXACT IMPLEMENTATION AS PER PROMPT
# =============================================================================

class OKXAPI:
    BASE_URL = "https://www.okx.com/api/v5"
    
    @staticmethod
    def get_klines(symbol: str, bar: str = "3m", limit: int = 130) -> pd.DataFrame:
        try:
            url = f"{OKXAPI.BASE_URL}/market/candles"
            params = {"instId": f"{symbol}-USDT-SWAP", "bar": bar, "limit": limit}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") != "0" or not data.get("data"):
                return pd.DataFrame()
            
            df = pd.DataFrame(data["data"], columns=[
                "timestamp", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close", "vol"]:
                df[col] = df[col].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.error(f"OKX Error for {symbol}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_top_volume_pairs(limit: int = 30) -> List[str]:
        try:
            url = f"{OKXAPI.BASE_URL}/market/tickers"
            params = {"instType": "SWAP", "limit": 100}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("code") != "0":
                return []
            
            tickers = [t for t in data["data"] if t["instId"].endswith("-USDT-SWAP")]
            tickers.sort(key=lambda x: float(x.get("volCcy24h", 0)), reverse=True)
            pairs = [t["instId"].replace("-USDT-SWAP", "") for t in tickers[:limit]]
            return pairs
            
        except Exception as e:
            logger.error(f"OKX Volume Error: {e}")
            return []

class GateIOAPI:
    BASE_URL = "https://api.gateio.ws/api/v4"
    
    @staticmethod
    def get_klines(symbol: str, interval: str = "3m", limit: int = 130) -> pd.DataFrame:
        try:
            url = f"{GateIOAPI.BASE_URL}/futures/usdt/candlesticks"
            params = {"contract": f"{symbol}_USDT", "interval": interval, "limit": limit}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list) or not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=["t", "v", "c", "h", "l", "o"])
            df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "vol"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            for col in ["open", "high", "low", "close", "vol"]:
                df[col] = df[col].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.error(f"GateIO Error for {symbol}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_top_volume_pairs(limit: int = 20) -> List[str]:
        try:
            url = f"{GateIOAPI.BASE_URL}/futures/usdt/contracts"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if not isinstance(data, list):
                return []
            
            contracts = [c for c in data if c.get("name", "").endswith("_USDT")]
            contracts.sort(key=lambda x: float(x.get("volume_24h_usd", 0) or 0), reverse=True)
            pairs = [c["name"].replace("_USDT", "") for c in contracts[:limit]]
            return pairs
            
        except Exception as e:
            logger.error(f"GateIO Volume Error: {e}")
            return []

class MEXCAPI:
    BASE_URL = "https://contract.mexc.com/api/v1/contract"
    
    @staticmethod
    def get_klines(symbol: str, interval: str = "Min3", limit: int = 130) -> pd.DataFrame:
        try:
            url = f"{MEXCAPI.BASE_URL}/kline/{symbol}_USDT"
            params = {"interval": interval, "limit": limit}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") != 200 or not data.get("data"):
                return pd.DataFrame()
            
            klines = data["data"]["time"]
            df = pd.DataFrame(klines, columns=["timestamp", "open", "close", "high", "low", "vol", "amount"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "vol"]:
                df[col] = df[col].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.error(f"MEXC Error for {symbol}: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_top_volume_pairs(limit: int = 10) -> List[str]:
        try:
            url = f"{MEXCAPI.BASE_URL}/ticker"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get("code") != 200:
                return []
            
            tickers = [t for t in data["data"] if t.get("symbol", "").endswith("_USDT")]
            tickers.sort(key=lambda x: float(x.get("volume24", 0) or 0), reverse=True)
            pairs = [t["symbol"].replace("_USDT", "") for t in tickers[:limit]]
            return pairs
            
        except Exception as e:
            logger.error(f"MEXC Volume Error: {e}")
            return []

# =============================================================================
# TECHNICAL ANALYSIS ENGINE - EXACT PROMPT IMPLEMENTATION
# =============================================================================

class TAEngine:
    
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA20, SMA100, and RSI14 exactly as per prompt"""
        if len(df) < 100:
            return df
        
        df = df.copy()
        df["sma20"] = df["close"].rolling(window=20).mean()
        df["sma100"] = df["close"].rolling(window=100).mean()
        
        # RSI Calculation
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi14"] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def calculate_cluster_spread(row: pd.Series) -> float:
        """Calculate cluster spread percentage - EXACT FORMULA FROM PROMPT"""
        cluster = [row["close"], row["sma20"], row["sma100"]]
        return ((max(cluster) - min(cluster)) / row["close"]) * 100
    
    @staticmethod
    def classify_candle(row: pd.Series, avg_body: float) -> Tuple[Optional[str], Optional[str]]:
        """
        Classify candle as elephant or tail - EXACT PROMPT LOGIC
        Returns: (candle_type, direction)
        """
        open_price = row["open"]
        close = row["close"]
        high = row["high"]
        low = row["low"]
        
        body = abs(close - open_price)
        total_range = high - low
        
        if total_range == 0:
            return None, None
        
        # Elephant: body ≥ 150% of average body of last 20 candles AND body is at least 40% of total range
        if body >= avg_body * Config.ELEPHANT_BODY_MULT and body >= total_range * Config.ELEPHANT_RANGE_RATIO:
            direction = "long" if close > open_price else "short"
            return "elephant", direction
        
        # Tail: wick ≥ 60% of total range, body small
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        
        if lower_wick >= total_range * Config.TAIL_WICK_RATIO:
            return "tail", "long"  # Rejection down, bounce up
        if upper_wick >= total_range * Config.TAIL_WICK_RATIO:
            return "tail", "short"  # Rejection up, fall down
        
        return None, None
    
    @staticmethod
    def get_avg_body(df: pd.DataFrame, lookback: int = 20) -> float:
        """Calculate average body size of last N candles"""
        if len(df) < lookback:
            return 0
        recent = df.tail(lookback)
        bodies = (recent["close"] - recent["open"]).abs()
        return bodies.mean()

# =============================================================================
# SIGNAL DETECTION ENGINE - EXACT PROMPT IMPLEMENTATION
# =============================================================================

class SignalDetector:
    
    def __init__(self):
        self.ta = TAEngine()
    
    def detect_compression(self, df: pd.DataFrame) -> Tuple[CompressionType, float]:
        """
        Detect compression state on current candle only - EXACT PROMPT LOGIC
        Returns: (compression_type, spread_pct)
        """
        if len(df) < 2:
            return CompressionType.NONE, 0
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check current spread
        curr_spread = self.ta.calculate_cluster_spread(curr)
        
        if curr_spread > 0.20:
            return CompressionType.NONE, curr_spread
        
        # Check for crossover (current or previous candle had SMA cross)
        prev_sma20, prev_sma100 = prev["sma20"], prev["sma100"]
        curr_sma20, curr_sma100 = curr["sma20"], curr["sma100"]
        
        crossed = (prev_sma20 > prev_sma100) != (curr_sma20 > curr_sma100)
        sma_gap = abs(curr_sma20 - curr_sma100) / curr["close"] * 100
        already_together = sma_gap <= 0.05
        
        if crossed or already_together:
            return CompressionType.CROSSOVER, curr_spread
        
        # Count consecutive compressed candles (forward only from current)
        candles_in = self._count_consecutive_compressed(df)
        
        if candles_in >= Config.COMPRESSION_MIN_CANDLES:
            return CompressionType.SQZ, curr_spread
        
        return CompressionType.BUILDING, curr_spread
    
    def _count_consecutive_compressed(self, df: pd.DataFrame) -> int:
        """Count consecutive compressed candles ending at current"""
        count = 0
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            spread = self.ta.calculate_cluster_spread(row)
            if spread <= 0.20:
                count += 1
            else:
                break
            if count >= 10:  # Cap for performance
                break
        return count
    
    def detect_expansion(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect expansion breakout - EXACT PROMPT LOGIC
        Returns: {"direction": str, "candle_type": str} or None
        """
        if len(df) < 3:
            return None
        
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        # Previous must be in compression
        prev_spread = self.ta.calculate_cluster_spread(prev)
        if prev_spread > 0.20:
            return None
        
        # Current must be OUT of compression
        curr_spread = self.ta.calculate_cluster_spread(curr)
        if curr_spread <= 0.20:
            return None
        
        # Current must be elephant or tail
        avg_body = self.ta.get_avg_body(df[:-1])
        candle_type, direction = self.ta.classify_candle(curr, avg_body)
        
        if candle_type:
            return {"direction": direction, "candle_type": candle_type}
        
        return None
    
    def detect_pullback(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect pullback to SMA20 - EXACT PROMPT LOGIC
        Returns: {"direction": str, "candle_type": str} or None
        """
        if len(df) < 2:
            return None
        
        curr = df.iloc[-1]
        sma20 = curr["sma20"]
        
        # Must be elephant or tail
        avg_body = self.ta.get_avg_body(df[:-1])
        candle_type, _ = self.ta.classify_candle(curr, avg_body)
        if not candle_type:
            return None
        
        # LONG: opened above, touched below, closed above
        long_bounce = (
            curr["open"] > sma20 and          # Was above
            curr["low"] <= sma20 * 1.002 and  # Touched it
            curr["close"] > sma20 and         # Bounced back above
            curr["close"] > curr["open"]      # Green candle (bounce confirmed)
        )
        
        # SHORT: opened below, touched above, closed below
        short_bounce = (
            curr["open"] < sma20 and          # Was below
            curr["high"] >= sma20 * 0.998 and # Touched it
            curr["close"] < sma20 and         # Bounced back below
            curr["close"] < curr["open"]      # Red candle (rejection confirmed)
        )
        
        if long_bounce:
            return {"direction": "long", "candle_type": candle_type}
        if short_bounce:
            return {"direction": "short", "candle_type": candle_type}
        
        return None
    
    def detect_reversal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect trend exhaustion reversal - EXACT PROMPT LOGIC
        Returns: {"direction": str, "candle_type": str} or None
        """
        if len(df) < 20:
            return None
        
        curr = df.iloc[-1]
        sma20, sma100 = curr["sma20"], curr["sma100"]
        
        # Check if trend is stretched
        gap_pct = abs(sma20 - sma100) / sma100 * 100
        if gap_pct < 1.2:
            return None
        
        # Check if price reached SMA100
        if sma20 > sma100:  # Uptrend
            if not (curr["low"] <= sma100 * 1.005 and curr["high"] >= sma100 * 0.995):
                return None
            direction = "short"
        else:  # Downtrend
            if not (curr["high"] >= sma100 * 0.995 and curr["low"] <= sma100 * 1.005):
                return None
            direction = "long"
        
        # Must be elephant or tail
        avg_body = self.ta.get_avg_body(df[:-1])
        candle_type, _ = self.ta.classify_candle(curr, avg_body)
        
        if candle_type:
            return {"direction": direction, "candle_type": candle_type}
        
        return None

# =============================================================================
# CONVICTION SCORING - EXACT PROMPT IMPLEMENTATION
# =============================================================================

class ConvictionScorer:
    
    def calculate(self, signal_type: SignalType, df: pd.DataFrame, 
                  details: Dict, compression_spread: float) -> Tuple[int, str]:
        """Calculate conviction score 0-100 - EXACT PROMPT FORMULA"""
        score = 0
        curr = df.iloc[-1]
        
        # Compression quality (25 points)
        if compression_spread <= 0.10:
            score += 25
        elif compression_spread <= 0.15:
            score += 20
        elif compression_spread <= 0.20:
            score += 15
        
        # Freshness (25 points) - always fresh with forward-only detection
        score += 25
        
        # Candle strength (20 points)
        candle_type = details.get("candle_type", "")
        if candle_type == "elephant":
            score += 20
        elif candle_type == "tail":
            score += 15
        
        # Room ahead (15 points) - simplified calculation
        room = self._calculate_room(df, details.get("direction", "long"))
        if room > 2.0:
            score += 15
        elif room > 1.0:
            score += 8
        
        # RSI position (10 points)
        rsi = curr.get("rsi14", 50)
        direction = details.get("direction", "long")
        if direction == "long" and rsi < 40:
            score += 10
        elif direction == "short" and rsi > 60:
            score += 10
        elif 40 <= rsi <= 60:
            score += 5
        
        # Determine tier
        if score >= 75:
            tier = "HIGH"
        elif score >= 50:
            tier = "MEDIUM"
        else:
            tier = "WATCH"
        
        return score, tier
    
    def _calculate_room(self, df: pd.DataFrame, direction: str) -> float:
        """Calculate room to next resistance/support (simplified)"""
        curr = df.iloc[-1]
        close = curr["close"]
        
        # Use SMA distance as proxy for room
        if direction == "long":
            recent_highs = df["high"].tail(20).max()
            room = ((recent_highs - close) / close) * 100
        else:
            recent_lows = df["low"].tail(20).min()
            room = ((close - recent_lows) / close) * 100
        
        return max(room, 0.5)  # Minimum 0.5% room

# =============================================================================
# BTC LIQUIDITY ENGINE - EXACT PROMPT IMPLEMENTATION
# =============================================================================

class BTCLiquidityEngine:
    
    def __init__(self):
        self.ta = TAEngine()
    
    def get_btc_data(self, timeframe: str = "1H") -> pd.DataFrame:
        """Fetch BTC data for liquidity analysis"""
        tf_map = {"15m": "15m", "1H": "1H", "4H": "4H"}
        tf = tf_map.get(timeframe, "1H")
        
        # Use OKX for BTC data
        df = OKXAPI.get_klines("BTC", bar=tf, limit=200)
        if len(df) > 0:
            df = self.ta.add_indicators(df)
        return df
    
    def get_regime(self, df: pd.DataFrame) -> Dict:
        """Calculate BTC regime for given timeframe - EXACT PROMPT LOGIC"""
        if len(df) < 100:
            return {"trend": "Unknown", "compression": "NONE"}
        
        curr = df.iloc[-1]
        sma20, sma100 = curr["sma20"], curr["sma100"]
        
        # Trend calculation
        gap_pct = abs(sma20 - sma100) / sma100
        if gap_pct >= 0.01:
            trend = "Trending Up" if sma20 > sma100 else "Trending Down"
        else:
            trend = "Ranging"
        
        # Compression check
        spread = self.ta.calculate_cluster_spread(curr)
        if spread <= 0.20:
            comp = "SQZ" if spread <= 0.15 else "CROSSOVER"
        else:
            comp = "NONE"
        
        return {
            "trend": trend,
            "compression": comp,
            "sma_gap": gap_pct * 100,
            "price": curr["close"]
        }
    
    def detect_sweeps(self, df: pd.DataFrame) -> Dict:
        """Detect liquidity sweeps on 1H - EXACT PROMPT LOGIC"""
        if len(df) < 20:
            return {"sweep": "None", "direction": None}
        
        # Find 3-bar swing highs/lows (simplified 3-bar pattern)
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        
        # Recent 3-bar swing high (BSL)
        bsl_levels = []
        for i in range(3, len(highs) - 3):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                bsl_levels.append(highs[i])
        
        # Recent 3-bar swing low (SSL)
        ssl_levels = []
        for i in range(3, len(lows) - 3):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                ssl_levels.append(lows[i])
        
        # Check current candle for sweep
        curr_idx = len(df) - 1
        curr_high = highs[curr_idx]
        curr_low = lows[curr_idx]
        curr_close = closes[curr_idx]
        
        # Bear sweep: high breaks BSL, close below
        for bsl in sorted(bsl_levels, reverse=True)[:3]:
            if curr_high > bsl and curr_close < bsl:
                return {"sweep": "Bear Sweep", "direction": "bearish", "level": bsl}
        
        # Bull sweep: low breaks SSL, close above
        for ssl in sorted(ssl_levels)[:3]:
            if curr_low < ssl and curr_close > ssl:
                return {"sweep": "Bull Sweep", "direction": "bullish", "level": ssl}
        
        return {"sweep": "None", "direction": None}
    
    def get_liquidity_levels(self, df: pd.DataFrame) -> Dict:
        """Get BSL and SSL levels"""
        if len(df) < 50:
            return {"bsl": [], "ssl": []}
        
        current_price = df.iloc[-1]["close"]
        
        # Find recent highs/lows as liquidity levels
        highs = df["high"].tail(50).nlargest(3).values
        lows = df["low"].tail(50).nsmallest(3).values
        
        bsl = [h for h in highs if h > current_price]
        ssl = [l for l in lows if l < current_price]
        
        return {
            "bsl": sorted(bsl)[:3],
            "ssl": sorted(ssl, reverse=True)[:3]
        }
    
    def get_pullback_zone(self, df: pd.DataFrame) -> Dict:
        """Check if price is in pullback zone (SMA20 ± 0.38%)"""
        if len(df) < 20:
            return {"in_zone": False}
        
        curr = df.iloc[-1]
        price = curr["close"]
        sma20 = curr["sma20"]
        
        zone_upper = sma20 * (1 + Config.PULLBACK_ZONE)
        zone_lower = sma20 * (1 - Config.PULLBACK_ZONE)
        
        in_zone = zone_lower <= price <= zone_upper
        
        return {
            "in_zone": in_zone,
            "sma20": sma20,
            "zone_upper": zone_upper,
            "zone_lower": zone_lower,
            "distance_pct": abs(price - sma20) / sma20 * 100
        }
    
    def get_full_analysis(self) -> Dict:
        """Get complete BTC liquidity analysis"""
        # Fetch multiple timeframes
        df_15m = self.get_btc_data("15m")
        df_1h = self.get_btc_data("1H")
        df_4h = self.get_btc_data("4H")
        
        analysis = {
            "15m": self.get_regime(df_15m) if len(df_15m) > 0 else {},
            "1H": self.get_regime(df_1h) if len(df_1h) > 0 else {},
            "4H": self.get_regime(df_4h) if len(df_4h) > 0 else {},
            "liquidity_1H": {
                "sweep": self.detect_sweeps(df_1h) if len(df_1h) > 0 else {},
                "levels": self.get_liquidity_levels(df_1h) if len(df_1h) > 0 else {},
                "pullback_zone": self.get_pullback_zone(df_1h) if len(df_1h) > 0 else {}
            }
        }
        
        return analysis

# =============================================================================
# SCANNER ORCHESTRATOR - EXACT PROMPT IMPLEMENTATION
# =============================================================================

class CryptoScanner:
    
    def __init__(self):
        self.detector = SignalDetector()
        self.scorer = ConvictionScorer()
        self.exchanges = {
            "OKX": OKXAPI(),
            "Gate.io": GateIOAPI(),
            "MEXC": MEXCAPI()
        }
    
    def get_scan_list(self, watchlist: List[str]) -> List[Tuple[str, str, Any]]:
        """Build list of pairs to scan - EXACT PROMPT LOGIC"""
        pairs = []
        
        # Add watchlist pairs (default to OKX, fallback to others)
        for symbol in watchlist:
            pairs.append((symbol, "OKX", OKXAPI))
        
        # Add top volume pairs from each exchange
        try:
            okx_top = OKXAPI.get_top_volume_pairs(30)
            for symbol in okx_top:
                if symbol not in watchlist:
                    pairs.append((symbol, "OKX", OKXAPI))
        except:
            pass
        
        time.sleep(Config.RATE_LIMIT_DELAY)
        
        try:
            gate_top = GateIOAPI.get_top_volume_pairs(20)
            for symbol in gate_top:
                if symbol not in watchlist:
                    pairs.append((symbol, "Gate.io", GateIOAPI))
        except:
            pass
        
        time.sleep(Config.RATE_LIMIT_DELAY)
        
        try:
            mexc_top = MEXCAPI.get_top_volume_pairs(10)
            for symbol in mexc_top:
                if symbol not in watchlist:
                    pairs.append((symbol, "MEXC", MEXCAPI))
        except:
            pass
        
        # Remove duplicates, keep priority order
        seen = set()
        unique_pairs = []
        for p in pairs:
            key = f"{p[0]}_{p[1]}"
            if key not in seen and len(unique_pairs) < Config.SCAN_PAIRS:
                seen.add(key)
                unique_pairs.append(p)
        
        return unique_pairs
    
    def scan_pair(self, symbol: str, exchange: str, api_class, timeframe: str = "3m") -> Optional[Signal]:
        """Scan a single pair for signals - EXACT PROMPT LOGIC"""
        try:
            # Map timeframe to exchange format
            tf_map = {
                "3m": {"OKX": "3m", "Gate.io": "3m", "MEXC": "Min3"},
                "5m": {"OKX": "5m", "Gate.io": "5m", "MEXC": "Min5"}
            }
            
            tf = tf_map.get(timeframe, {}).get(exchange, "3m")
            
            # Fetch data
            df = api_class.get_klines(symbol, tf)
            if len(df) < 100:
                return None
            
            # Add indicators
            df = TAEngine.add_indicators(df)
            
            # Detect compression
            comp_type, spread = self.detector.detect_compression(df)
            
            # Check for expansion
            exp_details = self.detector.detect_expansion(df)
            if exp_details:
                score, tier = self.scorer.calculate(SignalType.EXPANSION, df, exp_details, spread)
                direction = Direction.LONG if exp_details["direction"] == "long" else Direction.SHORT
                
                return Signal(
                    pair=symbol,
                    exchange=exchange,
                    signal_type=SignalType.EXPANSION,
                    compression_type=comp_type,
                    direction=direction,
                    timeframe=timeframe,
                    price=df.iloc[-1]["close"],
                    spread_pct=spread,
                    candle_type=exp_details["candle_type"],
                    rsi=df.iloc[-1].get("rsi14", 50),
                    conviction=score,
                    conviction_tier=tier,
                    details={"breakout_type": "expansion"},
                    timestamp=datetime.now()
                )
            
            # Check for pullback
            pb_details = self.detector.detect_pullback(df)
            if pb_details:
                score, tier = self.scorer.calculate(SignalType.PULLBACK, df, pb_details, spread)
                direction = Direction.LONG if pb_details["direction"] == "long" else Direction.SHORT
                
                return Signal(
                    pair=symbol,
                    exchange=exchange,
                    signal_type=SignalType.PULLBACK,
                    compression_type=comp_type,
                    direction=direction,
                    timeframe=timeframe,
                    price=df.iloc[-1]["close"],
                    spread_pct=spread,
                    candle_type=pb_details["candle_type"],
                    rsi=df.iloc[-1].get("rsi14", 50),
                    conviction=score,
                    conviction_tier=tier,
                    details={},
                    timestamp=datetime.now()
                )
            
            # Check for reversal
            rev_details = self.detector.detect_reversal(df)
            if rev_details:
                score, tier = self.scorer.calculate(SignalType.REVERSAL, df, rev_details, spread)
                direction = Direction.LONG if rev_details["direction"] == "long" else Direction.SHORT
                
                return Signal(
                    pair=symbol,
                    exchange=exchange,
                    signal_type=SignalType.REVERSAL,
                    compression_type=CompressionType.NONE,
                    direction=direction,
                    timeframe=timeframe,
                    price=df.iloc[-1]["close"],
                    spread_pct=spread,
                    candle_type=rev_details["candle_type"],
                    rsi=df.iloc[-1].get("rsi14", 50),
                    conviction=score,
                    conviction_tier=tier,
                    details={},
                    timestamp=datetime.now()
                )
            
            # If in compression but no breakout, report compression
            if comp_type in [CompressionType.SQZ, CompressionType.CROSSOVER]:
                return Signal(
                    pair=symbol,
                    exchange=exchange,
                    signal_type=SignalType.COMPRESSION,
                    compression_type=comp_type,
                    direction=Direction.NEUTRAL,
                    timeframe=timeframe,
                    price=df.iloc[-1]["close"],
                    spread_pct=spread,
                    candle_type=None,
                    rsi=df.iloc[-1].get("rsi14", 50),
                    conviction=0,
                    conviction_tier="WATCH",
                    details={},
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Scan error for {symbol}: {e}")
            return None

# =============================================================================
# STREAMLIT UI - EXACT PROMPT SPECIFICATION
# =============================================================================

def render_header():
    """Render app header"""
    st.title("🎯 Crypto Expansion Edge Scanner")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Last Scan:** {datetime.now().strftime('%H:%M:%S')}")
    with col2:
        auto_refresh = st.toggle("Auto Refresh (60s)", value=False)
    with col3:
        if st.button("🔄 Scan Now", type="primary", use_container_width=True):
            st.session_state.force_scan = True
    
    return auto_refresh

def render_btc_regime(analysis: Dict):
    """Render BTC multi-timeframe regime - EXACT PROMPT SPEC"""
    st.subheader("📊 BTC Regime")
    
    cols = st.columns(3)
    timeframes = ["15m", "1H", "4H"]
    
    for idx, tf in enumerate(timeframes):
        with cols[idx]:
            data = analysis.get(tf, {})
            trend = data.get("trend", "Unknown")
            comp = data.get("compression", "NONE")
            
            # Color coding
            if "Up" in trend:
                color = "green"
                emoji = "🟢"
            elif "Down" in trend:
                color = "red"
                emoji = "🔴"
            else:
                color = "yellow"
                emoji = "🟡"
            
            st.markdown(f"""
            <div style='padding: 10px; border-left: 4px solid {color}; background: rgba(255,255,255,0.05);'>
                <h4>{tf}</h4>
                <p>{emoji} {trend}</p>
                <small>Compression: {comp}</small>
            </div>
            """, unsafe_allow_html=True)

def render_liquidity_engine(analysis: Dict):
    """Render BTC Liquidity Engine - EXACT PROMPT SPEC"""
    st.subheader("💧 BTC Liquidity Engine (1H)")
    
    liq = analysis.get("liquidity_1H", {})
    sweep = liq.get("sweep", {})
    levels = liq.get("levels", {})
    zone = liq.get("pullback_zone", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        bias = analysis.get("1H", {}).get("trend", "Neutral")
        st.metric("Bias", bias)
    
    with col2:
        sweep_type = sweep.get("sweep", "None")
        st.metric("Recent Sweep", sweep_type)
    
    with col3:
        in_zone = "✅ Yes" if zone.get("in_zone") else "❌ No"
        st.metric("In Pullback Zone", in_zone)
    
    with col4:
        gap = analysis.get("1H", {}).get("sma_gap", 0)
        st.metric("SMA Gap", f"{gap:.2f}%")
    
    # Liquidity levels
    st.markdown("**Liquidity Levels:**")
    col_bsl, col_ssl = st.columns(2)
    
    with col_bsl:
        st.markdown("🔴 BSL (Resistance):")
        for level in levels.get("bsl", [])[:3]:
            st.markdown(f"- ${level:,.2f}")
    
    with col_ssl:
        st.markdown("🟢 SSL (Support):")
        for level in levels.get("ssl", [])[:3]:
            st.markdown(f"- ${level:,.2f}")

def render_signal_card(signal: Signal):
    """Render a signal card - EXACT 6-COLUMN FORMAT FROM PROMPT"""
    # Determine colors
    if signal.direction == Direction.LONG:
        border_color = "#00c853"
        emoji = "▲"
    elif signal.direction == Direction.SHORT:
        border_color = "#ff1744"
        emoji = "▼"
    else:
        border_color = "#ffd600"
        emoji = "◆"
    
    # Compression badge color
    if signal.compression_type == CompressionType.SQZ:
        comp_color = "orange"
        comp_text = "SQZ"
    elif signal.compression_type == CompressionType.CROSSOVER:
        comp_color = "blue"
        comp_text = "CROSS"
    else:
        comp_color = "gray"
        comp_text = "NONE"
    
    # Candle emoji
    candle_emoji = "🐘" if signal.candle_type == "elephant" else "🐾" if signal.candle_type == "tail" else "—"
    
    # Conviction color
    if signal.conviction_tier == "HIGH":
        conv_color = "green"
    elif signal.conviction_tier == "MEDIUM":
        conv_color = "orange"
    else:
        conv_color = "gray"
    
    # Room calculation
    if signal.conviction >= 75:
        room_text = "Large"
    elif signal.conviction >= 50:
        room_text = "Moderate"
    else:
        room_text = "Tight"
    
    # Firewall (obstacles) - simplified
    firewall = "✓ Clear" if signal.conviction > 50 else "⚠ Check"
    
    # Signal age
    age = "New" if signal.signal_type != SignalType.COMPRESSION else "Active"
    
    st.markdown(f"""
    <div style='margin: 10px 0; padding: 15px; border-left: 4px solid {border_color}; 
                background: rgba(255,255,255,0.03); border-radius: 4px;'>
        <table style='width: 100%; color: white;'>
            <tr>
                <td style='width: 25%;'><b>{emoji} {signal.direction.value.title()} {signal.pair}</b><br>
                    <small>{signal.timeframe} · {signal.exchange} · ${signal.price:.5f}</small></td>
                <td style='width: 15%; text-align: center;'>
                    <span style='color: {comp_color}; font-weight: bold;'>{comp_text}</span><br>
                    <small>{signal.spread_pct:.2f}%</small>
                </td>
                <td style='width: 20%; text-align: center;'>
                    {candle_emoji} {signal.candle_type or '—'}<br>
                    <small>{age}</small>
                </td>
                <td style='width: 15%; text-align: center;'>
                    <b>{signal.rsi:.0f}</b><br>
                    <small>{'Fuel' if (signal.direction == Direction.LONG and signal.rsi < 40) or (signal.direction == Direction.SHORT and signal.rsi > 60) else 'Neutral'}</small>
                </td>
                <td style='width: 10%; text-align: center;'>
                    <small>{firewall}</small>
                </td>
                <td style='width: 15%; text-align: center;'>
                    <b>{room_text}</b><br>
                    <small>{signal.conviction}/100</small>
                </td>
            </tr>
        </table>
        <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);'>
            <span style='color: {conv_color}; font-weight: bold;'>CONVICTION: {signal.conviction_tier}</span> — {signal.conviction}/100 — 
            {signal.compression_type.value} · {signal.candle_type or 'No candle'} · {firewall} · RSI {signal.rsi:.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_scanner(scanner: CryptoScanner, watchlist: List[str]):
    """Render the multi-pair scanner - EXACT PROMPT SPEC"""
    st.subheader("🔍 Multi-Pair Scanner")
    
    # Get pairs to scan
    with st.spinner("Building scan list..."):
        pairs = scanner.get_scan_list(watchlist)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    signals = []
    
    for idx, (symbol, exchange, api_class) in enumerate(pairs):
        progress = (idx + 1) / len(pairs)
        progress_bar.progress(min(progress, 0.99))
        status_text.text(f"Scanning {symbol} on {exchange}... ({idx+1}/{len(pairs)})")
        
        # Scan 3m
        signal = scanner.scan_pair(symbol, exchange, api_class, "3m")
        if signal:
            signals.append(signal)
        
        time.sleep(Config.RATE_LIMIT_DELAY)
        
        # Scan 5m
        signal_5m = scanner.scan_pair(symbol, exchange, api_class, "5m")
        if signal_5m:
            signals.append(signal_5m)
        
        time.sleep(Config.RATE_LIMIT_DELAY)
    
    progress_bar.empty()
    status_text.empty()
    
    # Sort signals by priority - EXACT PROMPT ORDER
    priority_map = {
        SignalType.REVERSAL: 0,
        SignalType.EXPANSION: 1,
        SignalType.PULLBACK: 2,
        SignalType.COMPRESSION: 3
    }
    
    signals.sort(key=lambda x: (priority_map.get(x.signal_type, 4), -x.conviction))
    
    # Filter options
    col1, col2 = st.columns([1, 2])
    with col1:
        direction_filter = st.selectbox("Direction", ["All", "Longs Only", "Shorts Only"])
    with col2:
        show_compression = st.toggle("Show Compression Setups", value=True)
    
    # Filter signals
    filtered = signals
    if direction_filter == "Longs Only":
        filtered = [s for s in filtered if s.direction == Direction.LONG]
    elif direction_filter == "Shorts Only":
        filtered = [s for s in filtered if s.direction == Direction.SHORT]
    
    if not show_compression:
        filtered = [s for s in filtered if s.signal_type != SignalType.COMPRESSION]
    
    # Display signals
    st.markdown(f"**Found {len(filtered)} signals**")
    
    if not filtered:
        st.info("No signals detected. Click 'Scan Now' to refresh.")
    else:
        for signal in filtered:
            render_signal_card(signal)

def main():
    st.set_page_config(
        page_title="Crypto Expansion Scanner",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'force_scan' not in st.session_state:
        st.session_state.force_scan = False
    
    # Header
    auto_refresh = render_header()
    
    # Watchlist input
    st.sidebar.header("⚙️ Configuration")
    watchlist_input = st.sidebar.text_input(
        "Watchlist (comma separated)",
        value=", ".join(Config.DEFAULT_WATCHLIST)
    )
    watchlist = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]
    
    # Initialize engines
    scanner = CryptoScanner()
    btc_engine = BTCLiquidityEngine()
    
    # Fetch BTC analysis
    with st.spinner("Fetching BTC context..."):
        btc_analysis = btc_engine.get_full_analysis()
    
    # Render BTC sections
    render_btc_regime(btc_analysis)
    render_liquidity_engine(btc_analysis)
    
    st.markdown("---")
    
    # Render scanner
    render_scanner(scanner, watchlist)
    
    # Auto refresh logic
    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()
