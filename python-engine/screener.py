# python-engine/screener.py

import pandas as pd
import numpy as np
import logging
from indicators import TechnicalIndicators
from smc_engine import SMCEngine
from liquidity_engine import InstitutionalLiquidityEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TechnicalScreener:
    """
    Multi-Timeframe Technical & Institutional Screener (v2.0 SMC Integrated - Pro Edition).
    Combines MTF Technicals, Institutional Liquidity Sweeps, and SMC Engine with fail-safe signatures.
    """

    def __init__(self):
        self.smc_engine = SMCEngine(fvg_threshold_pct=0.002)

    def _score_timeframe(self, df: pd.DataFrame, tf_weight: float = 1.0, **kwargs) -> dict:
        """
        Evaluates technical setup for a single timeframe (Base Score 0 to 100).
        Includes tf_weight and **kwargs to support multi-timeframe engines safely.
        """
        if df is None or len(df) < 30:
            return {
                "score": 50.0, 
                "signals": ["Insufficient Data"], 
                "reasons": ["Insufficient Data (<30 candles)"],
                "raw_metrics": {"rsi": 50.0, "vol_mult": 1.0, "adx": 0.0, "macd_bullish": False}
            }

        df_calc = TechnicalIndicators.calculate_indicators(df)
        if df_calc is None or df_calc.empty:
            return {
                "score": 50.0, 
                "signals": ["Calculation Error"], 
                "reasons": ["Indicator Calculation Failed"],
                "raw_metrics": {"rsi": 50.0, "vol_mult": 1.0, "adx": 0.0, "macd_bullish": False}
            }

        last = df_calc.iloc[-1]
        prev = df_calc.iloc[-2]
        signals = []
        reasons = []
        score = 50.0

        close = float(last.get('close', 0.0))
        ema20 = float(last.get('EMA_20', close))
        ema50 = float(last.get('EMA_50', close))

        # --- EMA Trend Matrix ---
        if close > ema20 > ema50:
            score += 15
            signals.append("Bullish EMA Alignment (Close > EMA20 > EMA50)")
        elif close < ema20 < ema50:
            score -= 15
            reasons.append("Bearish EMA Alignment (Close < EMA20 < EMA50)")
        else:
            reasons.append("Choppy / Mixed EMA Trend")

        # --- RSI Momentum Guard ---
        rsi = float(last.get('RSI', 50.0))
        if 50.0 <= rsi <= 68.0:
            score += 15
            signals.append(f"Bullish RSI Momentum ({rsi:.1f})")
        elif 32.0 <= rsi < 50.0:
            score -= 10
            reasons.append(f"Weak RSI Momentum ({rsi:.1f})")
        elif rsi > 70.0:
            score += 5
            signals.append(f"Overbought RSI ({rsi:.1f})")
            reasons.append(f"Overbought Risk RSI ({rsi:.1f})")
        elif rsi < 30.0:
            score += 5
            signals.append(f"Potential Oversold Bounce RSI ({rsi:.1f})")
            reasons.append(f"Oversold Condition RSI ({rsi:.1f})")

        # --- MACD Crossover Engine ---
        macd = float(last.get('MACD', 0.0))
        macd_sig = float(last.get('MACD_Signal', 0.0))
        prev_macd = float(prev.get('MACD', 0.0))
        prev_sig = float(prev.get('MACD_Signal', 0.0))
        macd_bullish = macd > macd_sig

        if prev_macd <= prev_sig and macd > macd_sig:
            score += 15
            signals.append("Bullish MACD Cross")
        elif macd > macd_sig:
            score += 8
            signals.append("MACD Above Signal")
        elif prev_macd >= prev_sig and macd < macd_sig:
            score -= 15
            reasons.append("Bearish MACD Cross")
        else:
            reasons.append("MACD Below Signal")

        # --- Institutional Volume Expansion ---
        vol_mult = float(last.get('Volume_Multiple', 1.0))
        if vol_mult >= 1.5:
            score += 10
            signals.append(f"Volume Expansion ({vol_mult:.2f}x)")
        else:
            reasons.append(f"Weak Volume ({vol_mult:.2f}x < 1.5x)")

        # --- ADX Trend Strength Filter ---
        adx = float(last.get('ADX', 0.0))
        if adx >= 25.0:
            score += 5
            signals.append(f"Strong Trend ADX ({adx:.1f})")
        else:
            reasons.append(f"Weak Trend / Ranging ADX ({adx:.1f})")

        # Apply weight scaling if passed externally (Default is 1.0)
        final_tf_score = max(0.0, min(100.0, score * tf_weight))

        return {
            "score": round(final_tf_score, 2),
            "signals": signals,
            "reasons": reasons,
            "raw_metrics": {
                "rsi": rsi,
                "vol_mult": vol_mult,
                "adx": adx,
                "macd_bullish": macd_bullish
            }
        }

    def run_screener(
        self,
        df_5m: pd.DataFrame = None,
        df_15m: pd.DataFrame = None,
        df_30m: pd.DataFrame = None,
        df_1h: pd.DataFrame = None,
        df_4h: pd.DataFrame = None,
        df_daily: pd.DataFrame = None,
        symbol: str = "BTC/USDT"
    ) -> dict:
        try:
            timeframes = [
                ("4h", df_4h, 0.30),
                ("1h", df_1h, 0.25),
                ("30m", df_30m, 0.20),
                ("15m", df_15m, 0.15),
                ("5m", df_5m, 0.10)
            ]

            weighted_score = 0.0
            total_weight = 0.0
            breakdown = {}
            all_signals = {}
            all_reasons = {}
            raw_metrics_map = {}

            for tf_name, df, weight in timeframes:
                if df is not None and not df.empty:
                    res = self._score_timeframe(df, tf_weight=1.0)
                    weighted_score += res['score'] * weight
                    total_weight += weight
                    breakdown[f"score_{tf_name}"] = res['score']
                    all_signals[tf_name] = res['signals']
                    all_reasons[tf_name] = res['reasons']
                    raw_metrics_map[tf_name] = res['raw_metrics']
                else:
                    breakdown[f"score_{tf_name}"] = 50.0
                    all_signals[tf_name] = ["Data Unavailable"]
                    all_reasons[tf_name] = ["Data Unavailable"]
                    raw_metrics_map[tf_name] = {"rsi": 50.0, "vol_mult": 1.0, "adx": 0.0, "macd_bullish": False}

            base_score = (weighted_score / total_weight) if total_weight > 0 else 50.0
            adjusted_score = base_score
            penalties = []
            bonuses = []

            score_4h = breakdown.get("score_4h", 50.0)
            score_1h = breakdown.get("score_1h", 50.0)
            vol_5m = raw_metrics_map.get("5m", {}).get("vol_mult", 1.0)
            adx_1h = raw_metrics_map.get("1h", {}).get("adx", 0.0)
            rsi_5m = raw_metrics_map.get("5m", {}).get("rsi", 50.0)
            macd_bull_5m = raw_metrics_map.get("5m", {}).get("macd_bullish", False)

            # --- Rule 1: HTF Bearish Guardrail ---
            if score_4h < 45.0:
                adjusted_score -= 15.0
                penalties.append("HTF Bearish Guardrail Penalty (-15)")

            # --- Rule 2: Volume Confirmation Gate ---
            if vol_5m < 1.5:
                adjusted_score -= 10.0
                penalties.append("Low Volume Penalty (-10)")
            elif vol_5m >= 2.0:
                adjusted_score += 10.0
                bonuses.append("Smart Money Volume Spike Bonus (+10)")

            # --- Rule 3: ADX Regime Filter ---
            if adx_1h < 20.0:
                adjusted_score -= 5.0
                penalties.append("Choppy Market ADX Penalty (-5)")
            elif adx_1h >= 25.0:
                adjusted_score += 5.0
                bonuses.append("Strong Trend ADX Bonus (+5)")

            # --- SMC & Institutional Liquidity Integration ---
            eval_df = df_15m if df_15m is not None else (df_5m if df_5m is not None else df_1h)
            smc_data = self.smc_engine.calculate_smc_score(eval_df, symbol=symbol)
            liquidity_data = InstitutionalLiquidityEngine.analyze_liquidity(eval_df, df_daily)

            # SMC Overlay Bonuses / Penalties
            smc_score = smc_data.get("smc_score", 50.0)
            if smc_score >= 65.0:
                adjusted_score += 10.0
                bonuses.append(f"SMC Bullish Structure Bonus (+10) [Score: {smc_score}]")
            elif smc_score <= 35.0:
                adjusted_score -= 10.0
                penalties.append(f"SMC Bearish Structure Penalty (-10) [Score: {smc_score}]")

            # Check if chasing in Premium Zone
            is_discount = smc_data.get("details", {}).get("zone", {}).get("is_discount", True)
            if not is_discount:
                adjusted_score -= 15.0
                penalties.append("SMC Guardrail: Chasing in Premium Zone (-15)")

            final_technical_score = max(0.0, min(100.0, adjusted_score))

            # --- Mode Classification (Triple Engine Logic) ---
            mode = "NO_SETUP"
            if score_4h >= 55.0 and score_1h >= 55.0 and vol_5m >= 1.5 and is_discount:
                mode = "MODE_A_TREND_FOLLOWING"
            elif (smc_data.get("choch_confirmed", False) or liquidity_data.get("key_liquidity_swept", False)) and is_discount:
                mode = "MODE_C_SMC_SWEEP_REVERSAL"
            elif score_4h < 50.0 and vol_5m >= 2.0 and (35.0 <= rsi_5m <= 55.0 or macd_bull_5m):
                mode = "MODE_B_EARLY_REVERSAL"

            return {
                "symbol": symbol,
                "technical_score": round(final_technical_score, 2),
                "base_score": round(base_score, 2),
                "setup_mode": mode,
                "penalties": penalties,
                "bonuses": bonuses,
                "breakdown": breakdown,
                "smc_metrics": smc_data,
                "liquidity_metrics": liquidity_data,
                "signals": all_signals,
                "reasons": all_reasons
            }

        except Exception as e:
            logging.error(f"⚠️ Screener Execution Error: {e}")
            return {
                "technical_score": 50.0,
                "base_score": 50.0,
                "setup_mode": "ERROR",
                "penalties": [],
                "bonuses": [],
                "breakdown": {},
                "smc_metrics": {},
                "liquidity_metrics": {},
                "signals": {},
                "reasons": {"error": str(e)}
            }