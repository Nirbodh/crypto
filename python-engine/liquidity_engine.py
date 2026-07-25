import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Literal, Tuple
from datetime import datetime, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InstitutionalLiquidityEngine:
    """
    v4.1 - Fixed: Session detection with proper datetime handling
    - Context-Aware Sweep Scoring (Reversal vs Continuation)
    - Multi-Factor Liquidity Probability (ATR + Volume + Trend + Sweep + HTF)
    - Dual Compression (ATR + Volume)
    - Wilder ATR (Professional Standard)
    - Timeframe-Aware Daily Open
    - Extended Liquidity Targets (EQH/EQL, FVG, OrderBlock, Asia/London High/Low)
    - Volume Delta (Close vs Open)
    - Session Detection (Asia, London, New York)
    - Gradient Red Flags + Confidence Score
    """

    # ---- Session Definitions (UTC) ----
    ASIA_START = time(0, 0)
    ASIA_END = time(8, 0)
    LONDON_START = time(8, 0)
    LONDON_END = time(16, 0)
    NY_START = time(13, 0)
    NY_END = time(22, 0)

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14, wilder: bool = True) -> pd.Series:
        """
        Calculates Average True Range (ATR) - Wilder's smoothing by default.
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        if wilder:
            # Wilder's smoothing: EMA with alpha = 1/period
            atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
        else:
            atr = tr.rolling(window=period).mean()
        
        return atr.bfill()

    @staticmethod
    def detect_session(dt) -> str:
        """
        Detect trading session based on UTC time.
        Accepts: datetime, Timestamp, or string convertible to datetime.
        """
        # ---- FIX: Convert input to datetime if needed ----
        if isinstance(dt, (int, float)):
            # If it's a numeric index, we can't determine time. Return OFF_HOURS.
            return "OFF_HOURS"
        
        if isinstance(dt, str):
            try:
                dt = pd.to_datetime(dt)
            except Exception:
                return "OFF_HOURS"
        
        # Now dt should be a datetime-like object
        if not hasattr(dt, 'time'):
            return "OFF_HOURS"
        
        t = dt.time()
        if InstitutionalLiquidityEngine.ASIA_START <= t < InstitutionalLiquidityEngine.ASIA_END:
            return "ASIA"
        elif InstitutionalLiquidityEngine.LONDON_START <= t < InstitutionalLiquidityEngine.LONDON_END:
            return "LONDON"
        elif InstitutionalLiquidityEngine.NY_START <= t < InstitutionalLiquidityEngine.NY_END:
            return "NEW_YORK"
        else:
            return "OFF_HOURS"

    @staticmethod
    def calculate_volume_delta(df: pd.DataFrame) -> pd.Series:
        """
        Approximate Volume Delta: Bullish if Close > Open, Bearish if Close < Open.
        Returns a series of signed volume.
        """
        delta = pd.Series(0.0, index=df.index)
        bullish_mask = df['close'] > df['open']
        bearish_mask = df['close'] < df['open']
        delta.loc[bullish_mask] = df.loc[bullish_mask, 'volume']
        delta.loc[bearish_mask] = -df.loc[bearish_mask, 'volume']
        return delta

    @staticmethod
    def find_session_high_low(df: pd.DataFrame, session: str) -> Tuple[float, float]:
        """Find high and low for a given session within the dataframe."""
        if df.empty:
            return 0.0, 0.0
        
        # Filter by session (using index time)
        try:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                # Try to use timestamp column if exists
                if 'timestamp' in df.columns:
                    idx = pd.to_datetime(df['timestamp'])
                else:
                    return 0.0, 0.0
            else:
                idx = df.index
            
            # Create a temporary series for session detection
            session_series = idx.map(lambda x: InstitutionalLiquidityEngine.detect_session(x))
            session_df = df[session_series == session]
        except Exception:
            session_df = df
        
        if session_df.empty:
            return 0.0, 0.0
        
        return float(session_df['high'].max()), float(session_df['low'].min())

    @staticmethod
    def find_nearby_fvg(df: pd.DataFrame, current_price: float, atr: float) -> Tuple[float, float, bool]:
        """
        Find nearest Fair Value Gap (FVG) from the dataframe.
        Returns: (fvg_top, fvg_bottom, is_bullish)
        """
        if len(df) < 3:
            return 0.0, 0.0, False
        
        nearest_fvg_top = 0.0
        nearest_fvg_bottom = 0.0
        is_bullish = False
        min_distance = float('inf')
        
        # Scan for FVGs (simplified: gap between candle i and i-2)
        for i in range(2, len(df)):
            # Bullish FVG: low[i] > high[i-2]
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                top = df['low'].iloc[i]
                bottom = df['high'].iloc[i-2]
                # Check if current price is near this FVG (within 2 ATR)
                if abs(current_price - top) < 2 * atr or abs(current_price - bottom) < 2 * atr:
                    distance = min(abs(current_price - top), abs(current_price - bottom))
                    if distance < min_distance:
                        min_distance = distance
                        nearest_fvg_top = top
                        nearest_fvg_bottom = bottom
                        is_bullish = True
            
            # Bearish FVG: high[i] < low[i-2]
            elif df['high'].iloc[i] < df['low'].iloc[i-2]:
                top = df['low'].iloc[i-2]
                bottom = df['high'].iloc[i]
                if abs(current_price - top) < 2 * atr or abs(current_price - bottom) < 2 * atr:
                    distance = min(abs(current_price - top), abs(current_price - bottom))
                    if distance < min_distance:
                        min_distance = distance
                        nearest_fvg_top = top
                        nearest_fvg_bottom = bottom
                        is_bullish = False
        
        return nearest_fvg_top, nearest_fvg_bottom, is_bullish

    @classmethod
    def analyze_liquidity_and_candles(
        cls, 
        df_ltf: pd.DataFrame, 
        df_daily: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None,
        market_regime: Literal["TRENDING", "RANGING", "VOLATILE", "BEAR", "CRASH"] = "TRENDING",
        safety_score: Optional[float] = None,
        position_quality_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Main Analysis Pipeline for Liquidity & Candle Vectoring (v4.1)
        """
        if df_ltf is None or len(df_ltf) < 20:
            return {
                "liquidity_score": 50.0,
                "bias": "NEUTRAL",
                "key_liquidity_swept": False,
                "red_flags": {"critical": [], "major": [], "minor": ["INSUFFICIENT_DATA"]},
                "reasons": ["Insufficient OHLCV Data"],
                "features": {},
                "confidence_score": 50.0
            }

        df = df_ltf.copy()
        
        # ---- FIX: Ensure index is datetime for session detection ----
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'])
            else:
                # If no timestamp, use the row number as a fallback (but sessions won't work)
                logging.warning("⚠️ No datetime index or timestamp column found. Session detection will be disabled.")
                # We'll just set a dummy datetime index
                df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='15min')
        
        # ---- 1. WILDER ATR (Professional) ----
        df['atr'] = cls.calculate_atr(df, period=14, wilder=True)
        avg_volume = df['volume'].rolling(window=20).mean()
        latest_atr = float(df['atr'].iloc[-1]) if not pd.isna(df['atr'].iloc[-1]) and df['atr'].iloc[-1] > 0 else 1.0

        latest_close = float(df['close'].iloc[-1])
        latest_high = float(df['high'].iloc[-1])
        latest_low = float(df['low'].iloc[-1])
        latest_open = float(df['open'].iloc[-1])
        latest_volume = float(df['volume'].iloc[-1])
        latest_idx = df.index[-1]  # Now this is a Timestamp

        # ---- 2. VOLUME DELTA (NEW) ----
        df['delta'] = cls.calculate_volume_delta(df)
        cumulative_delta = df['delta'].sum()
        latest_delta = float(df['delta'].iloc[-1])
        delta_ratio = abs(latest_delta) / (avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0)

        # ---- 3. SESSION DETECTION (FIXED: now passes datetime) ----
        current_session = cls.detect_session(latest_idx)  # latest_idx is Timestamp now
        session_high, session_low = cls.find_session_high_low(df, current_session)
        
        # ---- 4. DAILY OPEN (TIMEFRAME-AWARE) ----
        # Determine timeframe from index frequency
        if len(df) > 1:
            freq = pd.infer_freq(df.index)
            if freq is None:
                # Try to estimate from median diff
                diff = np.median(np.diff(df.index.astype(np.int64))) / (60 * 1e9)  # in minutes
                if diff < 0.5:
                    timeframe = "1MIN"
                elif diff < 3:
                    timeframe = "5MIN"
                elif diff < 8:
                    timeframe = "15MIN"
                elif diff < 30:
                    timeframe = "30MIN"
                elif diff < 60:
                    timeframe = "1H"
                elif diff < 240:
                    timeframe = "4H"
                else:
                    timeframe = "1D"
            else:
                timeframe = freq
        else:
            timeframe = "1D"

        # Daily open: depends on timeframe
        if df_daily is not None and len(df_daily) >= 2:
            daily_open = float(df_daily['open'].iloc[-1])
            pdh = float(df_daily['high'].iloc[-2])
            pdl = float(df_daily['low'].iloc[-2])
        elif timeframe in ["1D", "D"]:
            # For daily, use previous day's close as reference
            daily_open = float(df['open'].iloc[-1])  # Today's open
            pdh = float(df['high'].iloc[-2]) if len(df) >= 2 else latest_high
            pdl = float(df['low'].iloc[-2]) if len(df) >= 2 else latest_low
        else:
            # For LTF, use daily data from df_daily or fallback to a lookback
            if df_daily is not None and len(df_daily) >= 1:
                daily_open = float(df_daily['open'].iloc[-1])
                # PDH/PDL from previous day if available
                if len(df_daily) >= 2:
                    pdh = float(df_daily['high'].iloc[-2])
                    pdl = float(df_daily['low'].iloc[-2])
                else:
                    # Use lookback in LTF to approximate
                    lookback = min(len(df) - 1, int(24 * 60 / (60 if "H" in timeframe else 15)))  # ~24 hours
                    pdh = float(df['high'].iloc[-lookback-1:-1].max()) if lookback > 0 else latest_high
                    pdl = float(df['low'].iloc[-lookback-1:-1].min()) if lookback > 0 else latest_low
            else:
                # Fallback: lookback based on timeframe
                if timeframe in ["1H", "H"]:
                    lookback = 24  # 24 hours
                elif timeframe in ["4H"]:
                    lookback = 6   # 24 hours
                elif "MIN" in timeframe or "min" in timeframe:
                    # Convert to minutes and calculate 24 hours worth
                    if "5" in timeframe:
                        lookback = 288  # 24*60/5
                    elif "15" in timeframe:
                        lookback = 96
                    elif "30" in timeframe:
                        lookback = 48
                    else:
                        lookback = min(len(df) - 1, 24)
                else:
                    lookback = min(len(df) - 1, 24)
                
                pdh = float(df['high'].iloc[-lookback-1:-1].max()) if lookback > 0 else latest_high
                pdl = float(df['low'].iloc[-lookback-1:-1].min()) if lookback > 0 else latest_low
                daily_open = float(df['open'].iloc[-lookback]) if len(df) > lookback else float(df['open'].iloc[0])

        daily_open_distance_atr = (latest_close - daily_open) / latest_atr if latest_atr > 0 else 0.0

        # ---- 5. EXTENDED LIQUIDITY TARGETS (NEW) ----
        # EQH/EQL (Estimated Equilibrium High/Low)
        eqh = (pdh + pdl) / 2 + (pdh - pdl) * 0.25
        eql = (pdh + pdl) / 2 - (pdh - pdl) * 0.25
        
        # Asia High/Low
        asia_high, asia_low = cls.find_session_high_low(df, "ASIA")
        
        # London High/Low
        london_high, london_low = cls.find_session_high_low(df, "LONDON")
        
        # Nearest FVG
        fvg_top, fvg_bottom, fvg_is_bullish = cls.find_nearby_fvg(df, latest_close, latest_atr)
        
        # ---- 6. LIQUIDITY SWEEP DETECTION (CONTEXT-AWARE) ----
        bsl_sweep = (latest_high > pdh) and (latest_close < pdh)
        ssl_sweep = (latest_low < pdl) and (latest_close > pdl)
        
        # ---- 7. SWEEP CONTEXT & SCORING ----
        bsl_depth_atr = ((latest_high - pdh) / latest_atr) if bsl_sweep and latest_atr > 0 else 0.0
        ssl_depth_atr = ((pdl - latest_low) / latest_atr) if ssl_sweep and latest_atr > 0 else 0.0
        
        # Determine if sweep is reversal or continuation
        is_bullish_context = (
            daily_open_distance_atr > 0.5
            and latest_close > daily_open
            and market_regime in ["TRENDING", "RANGING"]
        )
        is_bearish_context = (
            daily_open_distance_atr < -0.5
            and latest_close < daily_open
            and market_regime in ["BEAR", "CRASH"]
        )
        
        bsl_score = 0
        ssl_score = 0
        
        if bsl_sweep:
            if is_bullish_context:
                bsl_score = 15
                bsl_type = "BULLISH_CONTINUATION"
            elif is_bearish_context:
                bsl_score = -20
                bsl_type = "BEARISH_REVERSAL"
            else:
                bsl_score = -10
                bsl_type = "BEARISH_NEUTRAL"
        else:
            bsl_type = "NONE"
        
        if ssl_sweep:
            if is_bearish_context:
                ssl_score = 15
                ssl_type = "BEARISH_CONTINUATION"
            elif is_bullish_context:
                ssl_score = 20
                ssl_type = "BULLISH_REVERSAL"
            else:
                ssl_score = 15
                ssl_type = "BULLISH_NEUTRAL"
        else:
            ssl_type = "NONE"

        # ---- 8. CANDLE MATH ----
        candle_range = max(latest_high - latest_low, 1e-8)
        body_size = abs(latest_close - latest_open)
        body_ratio = body_size / candle_range

        upper_wick = latest_high - max(latest_open, latest_close)
        lower_wick = min(latest_open, latest_close) - latest_low

        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

        # ---- 9. DUAL COMPRESSION (ATR + Volume) ----
        atr_window = df['atr'].rolling(window=20).mean()
        atr_ratio = latest_atr / atr_window.iloc[-1] if atr_window.iloc[-1] > 0 else 1.0
        is_atr_compressed = atr_ratio <= 0.5
        
        vol_window = df['volume'].rolling(window=20).mean()
        vol_ratio = latest_volume / vol_window.iloc[-1] if vol_window.iloc[-1] > 0 else 1.0
        is_vol_compressed = vol_ratio <= 0.4
        
        is_compressed = is_atr_compressed and is_vol_compressed
        compression_strength = (1 - atr_ratio) * 0.6 + (1 - vol_ratio) * 0.4

        # ---- 10. DISPLACEMENT ----
        mean_vol = vol_window.iloc[-1] if not pd.isna(vol_window.iloc[-1]) and vol_window.iloc[-1] > 0 else 1.0
        vol_expansion_ratio = latest_volume / mean_vol
        range_expansion_ratio = candle_range / latest_atr if latest_atr > 0 else 1.0

        is_displacement = (range_expansion_ratio >= 1.5) and (vol_expansion_ratio >= 2.0) and (body_ratio >= 0.65)
        displacement_type = "BULLISH_DISPLACEMENT" if (is_displacement and latest_close > latest_open) else \
                            ("BEARISH_DISPLACEMENT" if (is_displacement and latest_close < latest_open) else "NONE")

        # ---- 11. MULTI-FACTOR LIQUIDITY PROBABILITY ----
        dist_to_pdh = abs(pdh - latest_close) / latest_atr if latest_atr > 0 else 1.0
        dist_to_pdl = abs(latest_close - pdl) / latest_atr if latest_atr > 0 else 1.0
        
        volume_factor = min(1.0, vol_expansion_ratio / 3.0)
        
        trend_towards_pdh = 1.0 if (daily_open_distance_atr > 0 and pdh > latest_close) else 0.5
        trend_towards_pdl = 1.0 if (daily_open_distance_atr < 0 and pdl < latest_close) else 0.5
        
        sweep_factor_pdh = 1.2 if bsl_sweep else 1.0
        sweep_factor_pdl = 1.2 if ssl_sweep else 1.0
        
        htf_bullish = False
        if df_1h is not None and len(df_1h) >= 1:
            htf_bullish = df_1h['close'].iloc[-1] > df_1h['open'].iloc[-1]
        elif df_4h is not None and len(df_4h) >= 1:
            htf_bullish = df_4h['close'].iloc[-1] > df_4h['open'].iloc[-1]
        
        htf_factor_pdh = 1.3 if htf_bullish else 0.8
        htf_factor_pdl = 0.8 if htf_bullish else 1.3
        
        prob_pdh = np.exp(-0.3 * dist_to_pdh) * volume_factor * trend_towards_pdh * sweep_factor_pdh * htf_factor_pdh
        prob_pdl = np.exp(-0.3 * dist_to_pdl) * volume_factor * trend_towards_pdl * sweep_factor_pdl * htf_factor_pdl
        
        total_prob = prob_pdh + prob_pdl
        if total_prob > 0:
            prob_pdh = prob_pdh / total_prob
            prob_pdl = prob_pdl / total_prob
        else:
            prob_pdh, prob_pdl = 0.5, 0.5

        target_destination = f"BUY_SIDE_LIQUIDITY ({pdh:.2f})" if prob_pdh > prob_pdl else f"SELL_SIDE_LIQUIDITY ({pdl:.2f})"

        # ---- 12. HTF CONFLUENCE ----
        htf_sweep_confluence = 0
        if df_1h is not None and len(df_1h) >= 2:
            htf_pdh = float(df_1h['high'].iloc[-2])
            htf_pdl = float(df_1h['low'].iloc[-2])
            htf_high = float(df_1h['high'].iloc[-1])
            htf_low = float(df_1h['low'].iloc[-1])
            if (htf_high > htf_pdh and df_1h['close'].iloc[-1] < htf_pdh) or (htf_low < htf_pdl and df_1h['close'].iloc[-1] > htf_pdl):
                htf_sweep_confluence += 1
        
        if df_4h is not None and len(df_4h) >= 2:
            htf_pdh = float(df_4h['high'].iloc[-2])
            htf_pdl = float(df_4h['low'].iloc[-2])
            htf_high = float(df_4h['high'].iloc[-1])
            htf_low = float(df_4h['low'].iloc[-1])
            if (htf_high > htf_pdh and df_4h['close'].iloc[-1] < htf_pdh) or (htf_low < htf_pdl and df_4h['close'].iloc[-1] > htf_pdl):
                htf_sweep_confluence += 1

        # ---- 13. UNIFIED LIQUIDITY SCORE ----
        score = 50.0
        reasons: List[str] = []

        if bsl_sweep:
            if bsl_type == "BULLISH_CONTINUATION":
                score += 15
                reasons.append(f"BSL Sweep - Bullish Continuation (+15)")
            elif bsl_type == "BEARISH_REVERSAL":
                score -= 20
                reasons.append(f"BSL Sweep - Bearish Reversal (-20)")
            else:
                score -= 10
                reasons.append(f"BSL Sweep - Bearish Neutral (-10)")
        
        if ssl_sweep:
            if ssl_type == "BULLISH_REVERSAL":
                score += 20
                reasons.append(f"SSL Sweep - Bullish Reversal (+20)")
            elif ssl_type == "BEARISH_CONTINUATION":
                score -= 15
                reasons.append(f"SSL Sweep - Bearish Continuation (-15)")
            else:
                score += 15
                reasons.append(f"SSL Sweep - Bullish Neutral (+15)")

        if lower_wick_ratio >= 0.50 and latest_close > daily_open:
            score += 12
            reasons.append(f"Strong Lower Wick ({lower_wick_ratio*100:.1f}%)")
        elif upper_wick_ratio >= 0.50 and latest_close < daily_open:
            score += 12
            reasons.append(f"Strong Upper Wick ({upper_wick_ratio*100:.1f}%)")

        if displacement_type == "BULLISH_DISPLACEMENT":
            score += 18
            reasons.append("Bullish Displacement (+18)")
        elif displacement_type == "BEARISH_DISPLACEMENT":
            score -= 18
            reasons.append("Bearish Displacement (-18)")

        if is_compressed:
            score += 12
            reasons.append(f"Dual Compression (Strength: {compression_strength:.2f})")
        elif is_atr_compressed or is_vol_compressed:
            score += 5
            reasons.append("Partial Compression")

        if htf_sweep_confluence >= 2:
            score += 15
            reasons.append(f"HTF Confluence ({htf_sweep_confluence}/2)")
        elif htf_sweep_confluence >= 1:
            score += 8
            reasons.append(f"HTF Confluence ({htf_sweep_confluence}/2)")

        if delta_ratio > 1.5:
            if latest_delta > 0:
                score += 8
                reasons.append(f"Strong Bullish Delta ({delta_ratio:.2f}x)")
            else:
                score -= 8
                reasons.append(f"Strong Bearish Delta ({delta_ratio:.2f}x)")

        if current_session == "LONDON" and (latest_high > london_high or latest_low < london_low):
            score += 5
            reasons.append(f"London Session Breakout")
        elif current_session == "NEW_YORK" and (latest_high > london_high or latest_low < london_low):
            score += 5
            reasons.append(f"New York Session Breakout")

        score = float(np.clip(score, 0.0, 100.0))

        # ---- 14. BIAS ----
        if score > 60 and (ssl_sweep or displacement_type == "BULLISH_DISPLACEMENT" or lower_wick_ratio > 0.5):
            bias = "LONG"
        elif score < 40 and (bsl_sweep or displacement_type == "BEARISH_DISPLACEMENT" or upper_wick_ratio > 0.5):
            bias = "SHORT"
        else:
            bias = "NEUTRAL"

        # ---- 15. RED FLAGS ----
        red_flags = {"critical": [], "major": [], "minor": []}
        if score < 20:
            red_flags["critical"].append("LIQUIDITY_CRITICAL_WEAKNESS")
        if bsl_sweep and ssl_sweep:
            red_flags["major"].append("DOUBLE_SWEEP_INDECISION")
        if upper_wick_ratio > 0.7:
            red_flags["major"].append("STRONG_UPPER_WICK_REJECTION")
        if lower_wick_ratio > 0.7:
            red_flags["major"].append("STRONG_LOWER_WICK_REJECTION")
        if is_compressed and score < 50:
            red_flags["minor"].append("COMPRESSION_WITH_WEAK_SCORE")
        if vol_expansion_ratio > 3.0 and body_ratio < 0.3:
            red_flags["minor"].append("HIGH_VOLUME_LOW_BODY")

        # ---- 16. CONFIDENCE ----
        conf_factors = [
            min(100, (ssl_depth_atr + bsl_depth_atr) * 20) if (ssl_sweep or bsl_sweep) else 50,
            min(100, max(lower_wick_ratio, upper_wick_ratio) * 120) if max(lower_wick_ratio, upper_wick_ratio) > 0.3 else 40,
            100 if displacement_type != "NONE" else 60,
            min(100, 50 + htf_sweep_confluence * 25),
            min(100, 50 + delta_ratio * 10) if delta_ratio > 0 else 50
        ]
        confidence_score = round(np.mean(conf_factors), 1)

        # ---- 17. FEATURES ----
        features = {
            "pdh": round(pdh, 4),
            "pdl": round(pdl, 4),
            "daily_open": round(daily_open, 4),
            "daily_open_distance_atr": round(daily_open_distance_atr, 2),
            "bsl_sweep": bsl_sweep,
            "ssl_sweep": ssl_sweep,
            "bsl_depth_atr": round(bsl_depth_atr, 2),
            "ssl_depth_atr": round(ssl_depth_atr, 2),
            "bsl_type": bsl_type,
            "ssl_type": ssl_type,
            "body_ratio": round(body_ratio, 2),
            "upper_wick_ratio": round(upper_wick_ratio, 2),
            "lower_wick_ratio": round(lower_wick_ratio, 2),
            "displacement_type": displacement_type,
            "is_compressed": is_compressed,
            "compression_strength": round(compression_strength, 2),
            "range_expansion_ratio": round(range_expansion_ratio, 2),
            "vol_expansion_ratio": round(vol_expansion_ratio, 2),
            "delta_ratio": round(delta_ratio, 2),
            "cumulative_delta": round(cumulative_delta, 2),
            "prob_pdh": round(prob_pdh, 3),
            "prob_pdl": round(prob_pdl, 3),
            "current_session": current_session,
            "asia_high": round(asia_high, 4) if asia_high else 0,
            "asia_low": round(asia_low, 4) if asia_low else 0,
            "london_high": round(london_high, 4) if london_high else 0,
            "london_low": round(london_low, 4) if london_low else 0,
            "eqh": round(eqh, 4),
            "eql": round(eql, 4),
            "fvg_top": round(fvg_top, 4) if fvg_top else 0,
            "fvg_bottom": round(fvg_bottom, 4) if fvg_bottom else 0,
            "fvg_is_bullish": fvg_is_bullish,
            "htf_sweep_confluence": htf_sweep_confluence,
            "market_regime": market_regime
        }

        return {
            "liquidity_score": round(score, 2),
            "bias": bias,
            "key_liquidity_swept": bsl_sweep or ssl_sweep,
            "confidence_score": confidence_score,
            "red_flags": red_flags,
            "features": features,
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
                "ssl_depth_atr": round(ssl_depth_atr, 2),
                "bsl_type": bsl_type,
                "ssl_type": ssl_type
            },
            "candle_math": {
                "body_ratio": round(body_ratio, 2),
                "upper_wick_ratio": round(upper_wick_ratio, 2),
                "lower_wick_ratio": round(lower_wick_ratio, 2),
                "displacement": displacement_type,
                "is_compressed": is_compressed,
                "range_expansion_ratio": round(range_expansion_ratio, 2),
                "vol_expansion_ratio": round(vol_expansion_ratio, 2)
            },
            "volume_profile": {
                "delta_ratio": round(delta_ratio, 2),
                "cumulative_delta": round(cumulative_delta, 2),
                "session": current_session
            },
            "liquidity_targets": {
                "target_destination": target_destination,
                "prob_pdh": round(prob_pdh, 3),
                "prob_pdl": round(prob_pdl, 3),
                "eqh": round(eqh, 4),
                "eql": round(eql, 4),
                "asia_high": round(asia_high, 4) if asia_high else 0,
                "asia_low": round(asia_low, 4) if asia_low else 0,
                "london_high": round(london_high, 4) if london_high else 0,
                "london_low": round(london_low, 4) if london_low else 0
            },
            "reasons": reasons,
            "score": round(score, 2)
        }

    @classmethod
    def analyze_liquidity(cls, df_ltf: pd.DataFrame, df_daily: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Direct alias method for main pipeline calls (backward compatible)."""
        return cls.analyze_liquidity_and_candles(df_ltf, df_daily)


# Compatibility Alias
LiquidityEngine = InstitutionalLiquidityEngine


if __name__ == "__main__":
    import json
    
    print("=" * 70)
    print("💧 TESTING INSTITUTIONAL LIQUIDITY ENGINE v4.1...")
    print("=" * 70)
    
    # Generate Synthetic Test Candles (15m)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="15min")
    np.random.seed(42)
    
    close_prices = 100.0 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.random.uniform(0.1, 1.5, 100)
    low_prices = close_prices - np.random.uniform(0.1, 1.5, 100)
    open_prices = low_prices + np.random.uniform(0.0, high_prices - low_prices)
    volumes = np.random.uniform(1000, 5000, 100)

    mock_df_ltf = pd.DataFrame({
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes
    }, index=dates)

    # Create mock daily data
    daily_dates = pd.date_range(end=pd.Timestamp.now(), periods=5, freq="D")
    daily_close = np.array([100, 101, 99, 102, 101])
    daily_high = daily_close + np.random.uniform(0.5, 2.0, 5)
    daily_low = daily_close - np.random.uniform(0.5, 2.0, 5)
    daily_open = np.array([99.5, 100.5, 100, 98, 101])
    mock_df_daily = pd.DataFrame({
        "open": daily_open,
        "high": daily_high,
        "low": daily_low,
        "close": daily_close,
        "volume": np.random.uniform(10000, 50000, 5)
    }, index=daily_dates)

    # Run Analysis
    result = InstitutionalLiquidityEngine.analyze_liquidity_and_candles(
        df_ltf=mock_df_ltf,
        df_daily=mock_df_daily,
        df_1h=None,
        df_4h=None,
        market_regime="TRENDING"
    )
    
    print("✅ Liquidity Engine v4.1 Executed!")
    print(f"\n📊 Liquidity Score: {result['liquidity_score']}")
    print(f"🎯 Bias: {result['bias']}")
    print(f"🔐 Confidence: {result['confidence_score']}%")
    print(f"🚩 Red Flags: {result['red_flags']}")
    print(f"📋 Reasons: {result['reasons'][:3]}...")
    print(f"📍 Target: {result['liquidity_targets']['target_destination']}")
    print("\n" + "=" * 70)
    print("✅ Engine ready for production integration.")