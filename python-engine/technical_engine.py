# python-engine/technical_engine.py

import pandas as pd
import logging
from typing import Dict, Any, Optional
from screener import TechnicalScreener

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TechnicalEngine:
    """
    Institutional Pro-Level Technical Analysis Engine wrapper around TechnicalScreener.
    Supports Single-Timeframe & Multi-Timeframe Technical Fusion.
    """
    def __init__(self):
        self.screener = TechnicalScreener()

    def analyze(self, df_15m: pd.DataFrame, df_1h: Optional[pd.DataFrame] = None, df_4h: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Institutional Grade Multi-TF Technical Scoring Engine.
        """
        if df_15m is None or df_15m.empty:
            return {
                "technical_score": 50.0,
                "momentum_bullish": False,
                "signals": [],
                "red_flags": ["NO_15M_TECHNICAL_DATA"]
            }

        try:
            # 1. Core 15m Analysis
            res_15m = self.screener._score_timeframe(df_15m, tf_weight=1.0)
            score_15m = res_15m.get("score", 50.0)
            signals = res_15m.get("signals", [])

            # 2. Multi-Timeframe Weighted Aggregation (If HTF data provided)
            total_weight = 1.0
            weighted_score = score_15m

            if df_1h is not None and not df_1h.empty:
                res_1h = self.screener._score_timeframe(df_1h, tf_weight=1.0)
                score_1h = res_1h.get("score", 50.0)
                weighted_score += (score_1h * 1.2)  # Higher weight for 1H
                total_weight += 1.2
                signals.extend([f"1H_{s}" for s in res_1h.get("signals", [])])

            if df_4h is not None and not df_4h.empty:
                res_4h = self.screener._score_timeframe(df_4h, tf_weight=1.0)
                score_4h = res_4h.get("score", 50.0)
                weighted_score += (score_4h * 1.5)  # Highest weight for 4H
                total_weight += 1.5
                signals.extend([f"4H_{s}" for s in res_4h.get("signals", [])])

            final_tech_score = round(weighted_score / total_weight, 2)

            # 3. Institutional Risk Guard & Red Flags
            red_flags = []
            if final_tech_score < 42.0:
                red_flags.append("BEARISH_TECHNICAL_MOMENTUM")
            if "RSI_OVERBOUGHT" in signals and final_tech_score < 60.0:
                red_flags.append("OVERBOUGHT_DIVERGENCE_RISK")

            return {
                "technical_score": final_tech_score,
                "momentum_bullish": final_tech_score >= 55.0,
                "signals": list(set(signals)),
                "rsi_15m": res_15m.get("rsi", 50.0),
                "red_flags": red_flags
            }

        except Exception as e:
            logging.error(f"⚠️ Exception in TechnicalEngine Analysis: {e}")
            return {
                "technical_score": 50.0,
                "momentum_bullish": False,
                "signals": [],
                "red_flags": [f"ENGINE_CRASH: {str(e)}"]
            }