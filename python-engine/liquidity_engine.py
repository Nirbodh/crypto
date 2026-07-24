# python-engine/liquidity_engine.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InstitutionalLiquidityEngine:
    """
    Institutional Liquidity & Advanced Candle Psychology Engine:
    - Daily Open, PDH (Previous Day High), PDL (Previous Day Low) Dynamics
    - Liquidity Sweeps (Buy-Side BSL & Sell-Side SSL Stop Hunts)
    - Advanced Candle Math (Wick Rejection %, Body Ratio, Displacement)
    - Volatility Energy Compression (Squeeze before Pump/Dump)
    - Dynamic Liquidity Target Probability Engine
    """

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculates Average True Range (ATR)"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().bfill()

    @classmethod
    def analyze_liquidity_and_candles(
        cls, 
        df_ltf: pd.DataFrame, 
        df_daily: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Main Analysis Pipeline for Liquidity & Candle Vectoring
        """
        if df_ltf is None or len(df_ltf) < 20:
            return {
                "liquidity_score": 50.0,
                "bias": "NEUTRAL",
                "key_liquidity_swept": False,
                "reasons": ["Insufficient OHLCV Data"]
            }

        df = df_ltf.copy()
        df['atr'] = cls.calculate_atr(df, period=14)
        avg_volume = df['volume'].rolling(window=20).mean()

        latest_close = float(df['close'].iloc[-1])
        latest_high = float(df['high'].iloc[-1])
        latest_low = float(df['low'].iloc[-1])
        latest_open = float(df['open'].iloc[-1])
        latest_volume = float(df['volume'].iloc[-1])
        
        atr_val = df['atr'].iloc[-1]
        latest_atr = float(atr_val) if not pd.isna(atr_val) and atr_val > 0 else 1.0

        # 1. DAILY OPEN & PREVIOUS DAY HIGH/LOW (PDH / PDL) CALCULATIONS
        if df_daily is not None and len(df_daily) >= 2:
            pdh = float(df_daily['high'].iloc[-2])
            pdl = float(df_daily['low'].iloc[-2])
            daily_open = float(df_daily['open'].iloc[-1])
        else:
            # Fallback for LTF candles assuming 24-period lookback
            lookback = min(len(df) - 1, 24)
            pdh = float(df['high'].iloc[-lookback-1:-1].max()) if lookback > 0 else latest_high
            pdl = float(df['low'].iloc[-lookback-1:-1].min()) if lookback > 0 else latest_low
            daily_open = float(df['open'].iloc[-lookback]) if len(df) >= lookback else float(df['open'].iloc[0])

        daily_open_distance_atr = (latest_close - daily_open) / latest_atr

        # 2. LIQUIDITY SWEEP DETECTION (BSL / SSL)
        bsl_sweep = (latest_high > pdh) and (latest_close < pdh)
        ssl_sweep = (latest_low < pdl) and (latest_close > pdl)

        bsl_depth_atr = ((latest_high - pdh) / latest_atr) if bsl_sweep else 0.0
        ssl_depth_atr = ((pdl - latest_low) / latest_atr) if ssl_sweep else 0.0

        # 3. ADVANCED CANDLE MATHEMATICS & REJECTION WICKS
        candle_range = max(latest_high - latest_low, 1e-8)
        body_size = abs(latest_close - latest_open)
        body_ratio = body_size / candle_range

        upper_wick = latest_high - max(latest_open, latest_close)
        lower_wick = min(latest_open, latest_close) - latest_low

        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

        # 4. INSTITUTIONAL DISPLACEMENT VECTOR DETECTOR
        mean_vol = avg_volume.iloc[-1] if not pd.isna(avg_volume.iloc[-1]) and avg_volume.iloc[-1] > 0 else 1.0
        vol_expansion_ratio = latest_volume / mean_vol
        range_expansion_ratio = candle_range / latest_atr

        is_displacement = (range_expansion_ratio >= 1.5) and (vol_expansion_ratio >= 2.0) and (body_ratio >= 0.65)
        displacement_type = "BULLISH_DISPLACEMENT" if (is_displacement and latest_close > latest_open) else \
                            ("BEARISH_DISPLACEMENT" if (is_displacement and latest_close < latest_open) else "NONE")

        is_compressed = range_expansion_ratio <= 0.35

        # 5. LIQUIDITY TARGET PREDICTION ENGINE
        dist_to_pdh_atr = abs(pdh - latest_close) / latest_atr
        dist_to_pdl_atr = abs(latest_close - pdl) / latest_atr

        prob_target_pdh = 1.0 / (dist_to_pdh_atr + 1.0)
        prob_target_pdl = 1.0 / (dist_to_pdl_atr + 1.0)

        target_destination = f"BUY_SIDE_LIQUIDITY (${pdh:.2f})" if prob_target_pdh > prob_target_pdl else f"SELL_SIDE_LIQUIDITY (${pdl:.2f})"

        # 6. UNIFIED LIQUIDITY SCORE CALCULATION
        score = 50.0
        reasons: List[str] = []

        if ssl_sweep:
            score += 25.0
            reasons.append(f"SSL Sweep Detected (Depth: {ssl_depth_atr:.2f} ATR) - Retail Shorts Trapped")
        elif bsl_sweep:
            score += 25.0
            reasons.append(f"BSL Sweep Detected (Depth: {bsl_depth_atr:.2f} ATR) - Retail Longs Trapped")

        if lower_wick_ratio >= 0.50 and latest_close > daily_open:
            score += 15.0
            reasons.append(f"Strong Lower Wick Rejection ({lower_wick_ratio*100:.1f}%) Above Daily Open")
        elif upper_wick_ratio >= 0.50 and latest_close < daily_open:
            score += 15.0
            reasons.append(f"Strong Upper Wick Rejection ({upper_wick_ratio*100:.1f}%) Below Daily Open")

        if displacement_type == "BULLISH_DISPLACEMENT":
            score += 20.0
            reasons.append("Bullish Institutional Displacement Vector Confirmed")
        elif displacement_type == "BEARISH_DISPLACEMENT":
            score += 20.0
            reasons.append("Bearish Institutional Displacement Vector Confirmed")

        if is_compressed:
            score += 10.0
            reasons.append("Volatility Energy Compression (Squeeze Warning)")

        score = float(np.clip(score, 0.0, 100.0))

        if ssl_sweep or displacement_type == "BULLISH_DISPLACEMENT" or (lower_wick_ratio > 0.5 and daily_open_distance_atr > 0):
            bias = "LONG"
        elif bsl_sweep or displacement_type == "BEARISH_DISPLACEMENT" or (upper_wick_ratio > 0.5 and daily_open_distance_atr < 0):
            bias = "SHORT"
        else:
            bias = "NEUTRAL"

        return {
            "liquidity_score": round(score, 2),
            "bias": bias,
            "key_liquidity_swept": bsl_sweep or ssl_sweep,
            "daily_metrics": {
                "daily_open": round(daily_open, 4),
                "pdh": round(pdh, 4),
                "pdl": round(pdl, 4),
                "daily_open_distance_atr": round(daily_open_distance_atr, 2)
            },
            "sweeps": {
                "bsl_sweep": bsl_sweep,
                "ssl_sweep": ssl_sweep,
                "bsl_depth_atr": round(bsl_depth_atr, 2),
                "ssl_depth_atr": round(ssl_depth_atr, 2)
            },
            "candle_math": {
                "body_ratio": round(body_ratio, 2),
                "upper_wick_ratio": round(upper_wick_ratio, 2),
                "lower_wick_ratio": round(lower_wick_ratio, 2),
                "displacement": displacement_type,
                "is_compressed": is_compressed
            },
            "liquidity_targets": {
                "target_destination": target_destination,
                "prob_pdh": round(prob_target_pdh, 3),
                "prob_pdl": round(prob_target_pdl, 3)
            },
            "reasons": reasons
        }

    @classmethod
    def analyze_liquidity(cls, df_ltf: pd.DataFrame, df_daily: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Direct alias method for main pipeline calls."""
        return cls.analyze_liquidity_and_candles(df_ltf, df_daily)


# Compatibility Alias
LiquidityEngine = InstitutionalLiquidityEngine


if __name__ == "__main__":
    import json
    
    print("==================================================")
    print("💧 TESTING INSTITUTIONAL LIQUIDITY ENGINE...")
    print("==================================================")
    
    # Generate Synthetic Test Candles
    dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq="15min")
    np.random.seed(42)
    
    close_prices = 100.0 + np.cumsum(np.random.randn(60) * 0.5)
    high_prices = close_prices + np.random.uniform(0.1, 1.5, 60)
    low_prices = close_prices - np.random.uniform(0.1, 1.5, 60)
    open_prices = low_prices + np.random.uniform(0.0, high_prices - low_prices)
    volumes = np.random.uniform(1000, 5000, 60)

    mock_df_ltf = pd.DataFrame({
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes
    }, index=dates)

    # Run Analysis
    result = InstitutionalLiquidityEngine.analyze_liquidity(mock_df_ltf)
    print("✅ Liquidity Engine Executed Successfully!")
    print("\n📊 Output Matrix:")
    print(json.dumps(result, indent=4))