# python-engine/score_fusion_engine.py

from typing import Dict, Any, List


class InstitutionalScoreFusionEngine:
    
    @staticmethod
    def fuse_scores(
        symbol: str,
        tech_score: float,
        smc_score: float,
        liquidity_score: float,
        mtf_score: float = 50.0,
        derivatives_score: float = 50.0,
        fundamental_score: float = 50.0,
        sentiment_score: float = 50.0,
        session_score: float = 50.0,          # সেশন বুস্ট স্কোর
        fvg_mitigation_score: float = 50.0,   # FVG মিটিগেশন স্কোর
        estimated_win_rate: float = 0.50,
        rr_ratio: float = 2.0,
        btc_regime_bullish: bool = True,
        market_volatility_high: bool = False,
        red_flags: List[str] = None
    ) -> Dict[str, Any]:
        
        red_flags = red_flags or []
        rejection_reasons = []

        # -------------------------------------------------------------
        # 1. Dynamic Weighting Logic (Market Regime Based Adaptation)
        # -------------------------------------------------------------
        if btc_regime_bullish:
            weights = {
                "tech": 0.20,
                "smc": 0.15,
                "liquidity": 0.15,
                "mtf": 0.15,
                "derivatives": 0.10,
                "fundamental": 0.10,
                "sentiment": 0.05,
                "session": 0.05,
                "fvg_mitigation": 0.05
            }
        else:
            weights = {
                "tech": 0.15,
                "smc": 0.15,
                "liquidity": 0.20,
                "mtf": 0.10,
                "derivatives": 0.15,
                "fundamental": 0.10,
                "sentiment": 0.05,
                "session": 0.05,
                "fvg_mitigation": 0.05
            }

        unified_score = round(
            (tech_score * weights["tech"]) +
            (smc_score * weights["smc"]) +
            (liquidity_score * weights["liquidity"]) +
            (mtf_score * weights["mtf"]) +
            (derivatives_score * weights["derivatives"]) +
            (fundamental_score * weights["fundamental"]) +
            (sentiment_score * weights["sentiment"]) +
            (session_score * weights["session"]) +
            (fvg_mitigation_score * weights["fvg_mitigation"]),
            2
        )

        # 2. Mathematical Expected Value Calculation: EV = (Win_Rate * RR) - (Loss_Rate * 1)
        loss_rate = 1.0 - estimated_win_rate
        ev_r = round((estimated_win_rate * rr_ratio) - (loss_rate * 1.0), 2)

        # EV Classification
        if ev_r < 1.2:
            ev_tier = "REJECT"
        elif 1.2 <= ev_r < 1.5:
            ev_tier = "WATCHLIST"
        elif 1.5 <= ev_r <= 2.0:
            ev_tier = "TRADE_CANDIDATE"
        else:
            ev_tier = "HIGH_CONVICTION"

        # 3. Dynamic Threshold & Quant Hard Risk Gatekeeper Rules
        is_passed = True

        if not btc_regime_bullish:
            is_passed = False
            rejection_reasons.append("BTC Market Regime Bearish (Hard Guard)")

        if ev_r < 1.2:
            is_passed = False
            rejection_reasons.append(f"Low Expected Value (EV: {ev_r}R < 1.2R threshold)")

        required_score = 75.0 if not market_volatility_high else 78.0
        if unified_score < required_score:
            is_passed = False
            rejection_reasons.append(f"Unified Score Low ({unified_score}/100 < {required_score} threshold)")

        if len(red_flags) > 0:
            is_passed = False
            rejection_reasons.append(f"Red Flags Detected: {', '.join(red_flags)}")

        return {
            "symbol": symbol,
            "unified_score": unified_score,
            "ev_r": ev_r,
            "ev_tier": ev_tier,
            "is_passed": is_passed,
            "rejection_reasons": rejection_reasons,
            "score_breakdown": {
                "technical": tech_score,
                "smc": smc_score,
                "liquidity": liquidity_score,
                "mtf": mtf_score,
                "derivatives": derivatives_score,
                "fundamental": fundamental_score,
                "sentiment": sentiment_score,
                "session": session_score,
                "fvg_mitigation": fvg_mitigation_score
            },
            "applied_weights": weights,
            "red_flags": red_flags
        }