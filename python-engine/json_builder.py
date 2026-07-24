import json
import pandas as pd
import numpy as np
from typing import Dict, Any

class ScalperJSONBuilder:
    @staticmethod
    def build_scalper_json(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """
        Calculates Institutional Confluence Score, maps market conditions,
        scenarios, and risks, and outputs a complete Scalper JSON.
        """
        if df.empty or len(df) < 5:
            return json.dumps({"error": "Insufficient data to build JSON"}, indent=2)
        
        # Extract latest closed candle
        latest = df.iloc[-1].to_dict()
        prev = df.iloc[-2].to_dict()

        # ==========================================
        # 1. SCORING ENGINE (-100 to +100)
        # ==========================================
        score = 0
        confluences = []

        # Market Structure (+25 / -25)
        structure = latest.get('Structure', 'NEUTRAL')
        if structure == 'BULLISH':
            score += 25
            confluences.append("Bullish Market Structure (BOS/HH-HL)")
        elif structure == 'BEARISH':
            score -= 25
            confluences.append("Bearish Market Structure (BOS/LH-LL)")

        # Liquidity Sweep (+20 / -20)
        if latest.get('Liquidity_Sweep_Bullish', 0) == 1:
            score += 20
            confluences.append("Bullish Liquidity Sweep & Recovery")
        elif latest.get('Liquidity_Sweep_Bearish', 0) == 1:
            score -= 20
            confluences.append("Bearish Liquidity Sweep & Rejection")

        # FVG Zone Alignment (+15 / -15)
        fvg_type = latest.get('FVG_Type', 'NONE')
        if fvg_type == 'BULLISH':
            score += 15
            confluences.append("Trading into Bullish Fair Value Gap (FVG)")
        elif fvg_type == 'BEARISH':
            score -= 15
            confluences.append("Trading into Bearish Fair Value Gap (FVG)")

        # Volume Profile & Volume Spike (+15 / -15)
        vol_mult = latest.get('Volume_Multiple', 1.0)
        rvol = latest.get('Relative_Volume', 1.0)
        if vol_mult > 1.5 or rvol > 1.5:
            if latest.get('close', 0) > latest.get('open', 0):
                score += 15
                confluences.append("High Volume Institutional Buying")
            else:
                score -= 15
                confluences.append("High Volume Institutional Selling")

        # Premium / Discount Pricing (+10 / -10)
        prem_disc = latest.get('Premium_Discount', 0.0)
        if prem_disc < -0.5:
            score += 10
            confluences.append("Deep Discount Zone (Optimal Trade Entry - OTE)")
        elif prem_disc > 0.5:
            score -= 10
            confluences.append("Premium Zone (Overpriced / Short Potential)")

        # Session & Kill Zone Alignment (+15 / -15)
        if latest.get('Kill_Zone', 0) == 1:
            if score > 0:
                score += 15
                confluences.append("Session Kill-Zone Momentum (London/NY Open)")
            elif score < 0:
                score -= 15
                confluences.append("Session Kill-Zone Distribution")

        # Clamp Score between -100 and +100
        score = max(-100, min(100, score))

        # Determine Bias and Confidence Level
        if score >= 45:
            bias = "BULLISH"
            confidence = "HIGH" if score >= 75 else "MEDIUM"
        elif score <= -45:
            bias = "BEARISH"
            confidence = "HIGH" if score <= -75 else "MEDIUM"
        else:
            bias = "NEUTRAL"
            confidence = "LOW"

        # ==========================================
        # 2. KEY LEVELS & LIQUIDITY MAP
        # ==========================================
        current_price = float(latest.get('close', 0.0))
        atr = float(latest.get('ATR', current_price * 0.01))

        # Dynamic Key Levels
        support_1 = float(latest.get('OrderBlock_Lower', current_price - atr)) if not np.isnan(latest.get('OrderBlock_Lower', np.nan)) else current_price - atr
        support_2 = float(df['low'].tail(20).min())
        resistance_1 = float(latest.get('OrderBlock_Upper', current_price + atr)) if not np.isnan(latest.get('OrderBlock_Upper', np.nan)) else current_price + atr
        resistance_2 = float(df['high'].tail(20).max())

        # ==========================================
        # 3. SCENARIO ANALYSIS & RISKS
        # ==========================================
        bullish_scenario = {
            "trigger": f"Hold above Support 1 ({support_1:.2f}) and break local swing high with volume surge.",
            "target_1": resistance_1,
            "target_2": resistance_2,
            "invalidation": support_2
        }

        bearish_scenario = {
            "trigger": f"Rejection at Resistance 1 ({resistance_1:.2f}) or loss of Support 1 ({support_1:.2f}).",
            "target_1": support_1,
            "target_2": support_2,
            "invalidation": resistance_2
        }

        risk_factors = []
        if latest.get('ADX', 20) < 20:
            risk_factors.append("Low Trend Strength (ADX < 20) - High risk of sideways whipsaw")
        if latest.get('Kill_Zone', 0) == 0:
            risk_factors.append("Off-Session Trading - Low volume and poor liquidity conditions")
        if abs(score) < 45:
            risk_factors.append("Conflicting Signals - No clear Institutional Confluence")

        # ==========================================
        # 4. JSON OUTPUT STRUCTURE
        # ==========================================
        scalper_payload: Dict[str, Any] = {
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": str(latest.get('timestamp', '')),
                "analyst_mode": "Quantitative Institutional Research"
            },
            "market_condition": {
                "current_price": current_price,
                "overall_bias": bias,
                "confidence_level": confidence,
                "institutional_score": score,
                "session": latest.get('Session', 'UNKNOWN'),
                "kill_zone_active": bool(latest.get('Kill_Zone', 0)),
                "volatility_atr": atr,
                "trend_adx": float(latest.get('ADX', 20.0))
            },
            "confluences_detected": confluences,
            "liquidity_map": {
                "equal_highs_detected": bool(latest.get('Equal_High', 0)),
                "equal_lows_detected": bool(latest.get('Equal_Low', 0)),
                "fvg_status": fvg_type,
                "premium_discount_ratio": round(float(prem_disc), 2),
                "poc_price_level": float(latest.get('Fair_Value', current_price))
            },
            "key_levels": {
                "support_1": round(support_1, 2),
                "support_2": round(support_2, 2),
                "resistance_1": round(resistance_1, 2),
                "resistance_2": round(resistance_2, 2)
            },
            "scenario_analysis": {
                "bullish_path": bullish_scenario,
                "bearish_path": bearish_scenario,
                "primary_recommendation": "LONG" if bias == "BULLISH" else ("SHORT" if bias == "BEARISH" else "NO_TRADE")
            },
            "risk_management": {
                "key_risk_factors": risk_factors,
                "recommended_max_risk": "1% per trade",
                "disclaimer": "Automated quantitative analysis. No absolute certainty guaranteed."
            }
        }

        return json.dumps(scalper_payload, indent=2)