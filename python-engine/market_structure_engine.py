# python-engine/market_structure_engine.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

class MarketStructureEngine:
    """
    Advanced Institutional Market Structure Engine:
    - Equal Highs / Equal Lows (EQH / EQL)
    - Structural Breaks: BOS & CHOCH
    - Multi-Timeframe Trend Matrix
    """

    @staticmethod
    def detect_equal_highs_lows(df: pd.DataFrame, tolerance: float = 0.0015) -> Dict[str, Any]:
        highs = df['high'].values
        lows = df['low'].values
        
        eqh_list, eql_list = [], []

        for i in range(len(highs) - 20, len(highs) - 1):
            for j in range(i + 3, len(highs)):
                if abs(highs[i] - highs[j]) / highs[i] <= tolerance:
                    eqh_list.append(float((highs[i] + highs[j]) / 2.0))
                if abs(lows[i] - lows[j]) / lows[i] <= tolerance:
                    eql_list.append(float((lows[i] + lows[j]) / 2.0))

        has_eqh = len(eqh_list) > 0
        has_eql = len(eql_list) > 0

        return {
            "has_eqh": has_eqh,
            "has_eql": has_eql,
            "nearest_eqh": eqh_list[-1] if has_eqh else None,
            "nearest_eql": eql_list[-1] if has_eql else None
        }

    @classmethod
    def analyze_structure(cls, df_ltf: pd.DataFrame, htf_bias: str = "BULLISH") -> Dict[str, Any]:
        if df_ltf is None or len(df_ltf) < 50:
            return {"structure_score": 50.0, "bias": "NEUTRAL", "reasons": ["Insufficient Data"]}

        df = df_ltf.copy()
        current_close = float(df['close'].iloc[-1])

        recent_max = float(df['high'].iloc[-20:-1].max())
        recent_min = float(df['low'].iloc[-20:-1].min())
        prev_max = float(df['high'].iloc[-40:-20].max()) if len(df) >= 40 else recent_max
        prev_min = float(df['low'].iloc[-40:-20].min()) if len(df) >= 40 else recent_min

        bullish_bos = current_close > recent_max
        bearish_bos = current_close < recent_min

        bullish_choch = (df['close'].iloc[-2] <= prev_min) and (current_close > recent_max)
        bearish_choch = (df['close'].iloc[-2] >= prev_max) and (current_close < recent_min)

        eq_data = cls.detect_equal_highs_lows(df)

        score = 50.0
        reasons: List[str] = []

        if bullish_choch:
            score += 25.0
            reasons.append("Bullish CHOCH Detected")
        elif bearish_choch:
            score -= 25.0
            reasons.append("Bearish CHOCH Detected")

        if bullish_bos:
            score += 15.0
            reasons.append("Bullish BOS Confirmed")
        elif bearish_bos:
            score -= 15.0
            reasons.append("Bearish BOS Confirmed")

        if eq_data["has_eqh"] and current_close < eq_data["nearest_eqh"]:
            score += 10.0
            reasons.append(f"EQH Magnet Above (${eq_data['nearest_eqh']:.2f})")

        if eq_data["has_eql"] and current_close > eq_data["nearest_eql"]:
            score -= 10.0
            reasons.append(f"EQL Magnet Below (${eq_data['nearest_eql']:.2f})")

        if (htf_bias == "BULLISH" and (bullish_bos or bullish_choch)) or \
           (htf_bias == "BEARISH" and (bearish_bos or bearish_choch)):
            score += 10.0
            reasons.append(f"HTF Confluence Matched ({htf_bias})")

        score = float(np.clip(score, 0.0, 100.0))
        bias = "BULLISH" if score >= 60.0 else ("BEARISH" if score <= 40.0 else "NEUTRAL")

        return {
            "structure_score": round(score, 2),
            "bias": bias,
            "bullish_choch": bullish_choch,
            "bearish_choch": bearish_choch,
            "bullish_bos": bullish_bos,
            "bearish_bos": bearish_bos,
            "equal_highs_lows": eq_data,
            "reasons": reasons
        }