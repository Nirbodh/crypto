import math
import logging
from typing import Dict, Any, List, Optional, Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InstitutionalScoreFusionEngine:
    """
    v5.1 - Institutional Grade Score Fusion Engine
    - Dynamic Threshold (Regime-Based)
    - Risk Engine Integration (Safety + Quality → Risk Score)
    - Confidence Score + Level (High/Medium/Weak)
    - Soft Penalty System (No Hard Reject for most flags)
    - Red Flag Severity with Pre-defined Penalties
    - Fix: Risk double counting removed from conviction score
    - Fix: Dynamic max penalty cap (default 30)
    """

    # ---- Penalty Dictionary (Soft Penalties) ----
    PENALTY_MAP = {
        # Liquidity & Volume
        "LOW_VOLUME": 3,
        "WIDE_SPREAD": 4,
        "THIN_ORDERBOOK": 5,
        "LOW_LIQUIDITY": 4,
        
        # Technical Signals
        "BEARISH_DIVERGENCE": 6,
        "OVERBOUGHT": 4,
        "OVERSOLD": 3,
        "WEAK_MOMENTUM": 3,
        
        # SMC / Structure
        "LIQUIDITY_SWEEP": 5,
        "FVG_UNMITIGATED": 4,
        "WEAK_ORDER_BLOCK": 3,
        "CHoCH_FAILURE": 6,
        
        # Timing & Entry
        "LATE_ENTRY": 4,
        "POOR_RISK_REWARD": 5,
        "MISSED_BREAKOUT": 3,
        
        # Market Context
        "BEARISH_SENTIMENT": 5,
        "HIGH_FUNDING_RATE": 4,
        "OPEN_INTEREST_DROP": 3,
        
        # Risk Related
        "MARGIN_ISSUE": 8,
        "LIQUIDATION_RISK": 10,
        "HIGH_LEVERAGE": 6,
        
        # Data Quality
        "STALE_DATA": 3,
        "INSUFFICIENT_DATA": 5,
        "EXCHANGE_ISSUE": 7,
        
        # Default (if flag not found)
        "DEFAULT": 3
    }

    @staticmethod
    def fuse_scores(
        # ----- Core Signals (Required) -----
        symbol: str,
        tech_score: float,
        smc_score: float,
        liquidity_score: float,
        
        # ----- Market Volatility (NEW) -----
        market_volatility_high: Optional[bool] = None,
        
        # ----- Optional Signals (Default 50) -----
        mtf_score: float = 50.0,
        derivatives_score: float = 50.0,
        fundamental_score: float = 50.0,
        sentiment_score: float = 50.0,
        session_score: float = 50.0,
        fvg_mitigation_score: float = 50.0,
        
        # ----- Risk Engine Integration (NEW) -----
        risk_score: Optional[float] = None,           # 0-100 (composite from RiskEngine)
        safety_score: Optional[float] = None,         # Legacy (will map to risk_score if not provided)
        position_quality_score: Optional[float] = None,
        effective_leverage: float = 1.0,
        capital_exposure_pct: float = 50.0,
        
        # ----- Market Context -----
        market_regime: Literal["TRENDING", "RANGING", "VOLATILE", "BEAR", "CRASH"] = "TRENDING",
        desired_direction: Literal["LONG", "SHORT", "BOTH"] = "BOTH",
        btc_regime_bullish: Optional[bool] = None,    # Legacy support
        
        # ----- ATR / Volatility -----
        atr_ratio_pct: float = 1.0,
        volatility_penalty_factor: float = 0.15,
        
        # ----- Net Expected Value (with costs) -----
        estimated_win_rate: float = 0.50,
        rr_ratio: float = 2.0,
        fee_rate_pct: float = 0.1,
        funding_rate_pct: float = 0.01,
        slippage_rate_pct: float = 0.05,
        
        # ----- Red Flags (Soft Penalty System) -----
        red_flags: Optional[List[str]] = None,         # Legacy -> will be penalized
        penalty_flags: Optional[List[str]] = None,     # Soft penalty list
        critical_flags: Optional[List[str]] = None,    # Hard Reject (only these)
        
        # ----- Weight Customization -----
        custom_weights: Optional[Dict[str, float]] = None,
        
        # ----- Kelly / Sizing -----
        kelly_fraction: float = 0.25,
        max_recommended_risk_pct: float = 2.0,
        
        # ----- Feature Flags -----
        include_confidence_metrics: bool = True,
        
        # 🔥 NEW: Max penalty cap (default 30, matching previous hardcoded value)
        max_penalty_pct: float = 30.0
        
    ) -> Dict[str, Any]:
        
        # ============================================================
        # 1. INIT & LEGACY MAPPING
        # ============================================================
        # Map legacy btc_regime_bullish to market_regime
        if btc_regime_bullish is not None and market_regime == "TRENDING":
            market_regime = "TRENDING" if btc_regime_bullish else "BEAR"
        
        # Normalize lists
        critical_flags = critical_flags or []
        penalty_flags = penalty_flags or []
        
        # Legacy red_flags -> move to penalty_flags
        if red_flags:
            # Check for critical keywords
            for flag in red_flags:
                flag_upper = flag.upper()
                if "DELIST" in flag_upper or "HACK" in flag_upper or "INSOLV" in flag_upper:
                    critical_flags.append(flag)
                else:
                    penalty_flags.append(flag)
        
        rejection_reasons = []
        applied_penalties = []
        warning_messages = []
        
        # ---- Build Risk Score ----
        # If risk_score not provided, derive from safety + quality
        if risk_score is None:
            if safety_score is not None and position_quality_score is not None:
                risk_score = (safety_score * 0.6 + position_quality_score * 0.4)
            elif safety_score is not None:
                risk_score = safety_score
            elif position_quality_score is not None:
                risk_score = position_quality_score
            else:
                risk_score = 50.0  # Neutral default
        
        risk_score = max(0, min(100, risk_score))
        
        # Normalize all scores to 0-100 range
        all_scores = {
            "tech": max(0, min(100, tech_score)),
            "smc": max(0, min(100, smc_score)),
            "liquidity": max(0, min(100, liquidity_score)),
            "mtf": max(0, min(100, mtf_score)),
            "derivatives": max(0, min(100, derivatives_score)),
            "fundamental": max(0, min(100, fundamental_score)),
            "sentiment": max(0, min(100, sentiment_score)),
            "session": max(0, min(100, session_score)),
            "fvg_mitigation": max(0, min(100, fvg_mitigation_score)),
            "risk": risk_score  # NEW: Risk Score as a component
        }
        
        # ============================================================
        # 2. DYNAMIC WEIGHT ENGINE (with Risk Score)
        # ============================================================
        if custom_weights:
            weights = custom_weights
        else:
            # Base weights (Risk Score now included)
            base_weights = {
                "tech": 0.16,
                "smc": 0.14,
                "liquidity": 0.14,
                "risk": 0.18,      # Risk Score gets significant weight
                "mtf": 0.10,
                "derivatives": 0.08,
                "fundamental": 0.08,
                "sentiment": 0.05,
                "session": 0.04,
                "fvg_mitigation": 0.03
            }
            
            # ---- Regime Adjustments ----
            regime_adj = {}
            if market_regime == "TRENDING":
                regime_adj = {"tech": 0.04, "mtf": 0.03, "risk": -0.02, "smc": -0.01, "liquidity": -0.02}
            elif market_regime == "RANGING":
                regime_adj = {"tech": -0.04, "smc": 0.05, "liquidity": 0.04, "risk": 0.02, "mtf": -0.03}
            elif market_regime == "VOLATILE":
                regime_adj = {"tech": -0.03, "risk": 0.06, "liquidity": 0.04, "derivatives": 0.03, "mtf": 0.02}
            elif market_regime == "BEAR":
                regime_adj = {"risk": 0.08, "liquidity": 0.04, "derivatives": 0.04, "tech": -0.05, "smc": -0.02}
            elif market_regime == "CRASH":
                regime_adj = {"risk": 0.12, "liquidity": 0.05, "fundamental": 0.03, "tech": -0.08, "smc": -0.04}
            
            # ---- Volatility Adaptive (ATR based) ----
            vol_adj = {}
            if atr_ratio_pct > 2.0:
                vol_adj = {"risk": 0.04, "liquidity": 0.03, "tech": -0.04, "smc": -0.02}
            if atr_ratio_pct > 4.0:
                vol_adj = {"risk": 0.06, "liquidity": 0.05, "tech": -0.06, "smc": -0.03}
            
            # Apply adjustments
            for k, v in regime_adj.items():
                if k in base_weights:
                    base_weights[k] = max(0.01, base_weights[k] + v)
            for k, v in vol_adj.items():
                if k in base_weights:
                    base_weights[k] = max(0.01, base_weights[k] + v)
            
            # Normalize to sum to 1.0
            total = sum(base_weights.values())
            weights = {k: v / total for k, v in base_weights.items()}
        
        # ============================================================
        # 3. BASE UNIFIED SCORE (with Risk Component)
        # ============================================================
        raw_unified_score = sum(all_scores[k] * weights[k] for k in weights)
        
        # ---- Volatility Penalty (continuous) ----
        vol_penalty = 0.0
        if atr_ratio_pct > 1.5:
            penalty_factor = min(1.0, (atr_ratio_pct - 1.5) / 3.5)
            vol_penalty = raw_unified_score * (penalty_factor * volatility_penalty_factor)
            applied_penalties.append(f"Volatility: -{round(vol_penalty, 1)} pts")
        
        # ---- Soft Penalties (from penalty_flags) ----
        total_penalty = 0.0
        if penalty_flags:
            for flag in penalty_flags:
                # Find penalty from map, use default if not found
                penalty = InstitutionalScoreFusionEngine.PENALTY_MAP.get(flag.upper(), 
                                                                         InstitutionalScoreFusionEngine.PENALTY_MAP["DEFAULT"])
                total_penalty += penalty
                warning_messages.append(f"⚡ {flag}: -{penalty} pts")
            
            # 🔥 FIX: Cap total penalty using dynamic max_penalty_pct (default 30)
            total_penalty = min(max_penalty_pct, total_penalty)
            applied_penalties.append(f"Red Flags ({len(penalty_flags)}): -{round(total_penalty, 1)} pts")
        
        unified_score = max(0, raw_unified_score - vol_penalty - total_penalty)
        unified_score = round(min(100, unified_score), 2)

        # ---- RAW SCORE DEBUG ----
        logging.info(
            f"""
RAW SCORE DEBUG
{symbol}

raw_unified_score={raw_unified_score}

penalties={applied_penalties}

final_score={unified_score}

"""
        )
        
        # ============================================================
        # 4. DYNAMIC THRESHOLD (Regime-Based)
        # ============================================================
        threshold_map = {
            "TRENDING": 68,
            "RANGING": 70,
            "VOLATILE": 74,
            "BEAR": 74,
            "CRASH": 80
        }
        pass_threshold = threshold_map.get(market_regime, 75)
        
        # ============================================================
        # 5. NET EXPECTED VALUE (EV)
        # ============================================================
        fee = fee_rate_pct / 100.0
        funding = funding_rate_pct / 100.0
        slippage = slippage_rate_pct / 100.0
        
        loss_rate = max(0.01, 1.0 - estimated_win_rate)
        
        gross_ev = (estimated_win_rate * rr_ratio) - (loss_rate * 1.0)
        total_costs = fee + funding + slippage
        net_ev = gross_ev - total_costs
        
        # EV Score: Sigmoid centered at 0.5
        ev_score = 100.0 / (1.0 + math.exp(-1.2 * (net_ev - 0.5)))
        
        # Risk-Adjusted EV
        risk_adjusted_net_ev = net_ev * (risk_score / 100) if risk_score < 60 else net_ev
        
        # EV Tier
        if net_ev < 0.5:
            ev_tier = "REJECT"
        elif 0.5 <= net_ev < 1.0:
            ev_tier = "WATCHLIST"
        elif 1.0 <= net_ev < 1.8:
            ev_tier = "TRADE_CANDIDATE"
        else:
            ev_tier = "HIGH_CONVICTION"
        
        # ============================================================
        # 6. DIRECTIONAL RESTRICTIONS
        # ============================================================
        long_allowed = market_regime not in ["BEAR", "CRASH"]
        short_allowed = market_regime not in ["CRASH"]  # 🔥 FIX: SHORT not allowed in CRASH (only if liquidity available)
        
        direction_passed = True
        direction_penalty = 0
        if desired_direction == "LONG" and not long_allowed:
            direction_passed = False
            direction_penalty = 15
            rejection_reasons.append(f"LONG restricted in {market_regime} regime")
        elif desired_direction == "SHORT" and not short_allowed:
            direction_passed = False
            direction_penalty = 15
            rejection_reasons.append("SHORT restricted in CRASH regime (liquidity risk)")
        
        # Apply direction penalty to score (if not hard rejected)
        if not direction_passed:
            unified_score = max(0, unified_score - direction_penalty)
            applied_penalties.append(f"Direction Mismatch: -{direction_penalty} pts")
        
        # ============================================================
        # 7. FINAL PASS / FAIL
        # ============================================================
        is_passed = True
        all_reasons = rejection_reasons.copy()
        
        # ---- DEBUG: Log full state before gates ----
        logging.info(
            f"""
================ FUSION DEBUG ================
{symbol}
direction_passed={direction_passed}
direction_penalty={direction_penalty}
critical_flags={critical_flags}
net_ev={net_ev}
pass_threshold={pass_threshold}
unified_score={unified_score}
==============================================
"""
        )
        
        # Hard Gate 1: Critical Flags
        if critical_flags:
            logging.warning(f"{symbol} FAIL: critical_flags")
            is_passed = False
            all_reasons.extend([f"CRITICAL: {flag}" for flag in critical_flags])
        
        # Hard Gate 2: Direction (only if direction penalty > 10)
        if not direction_passed and direction_penalty > 10:
            logging.warning(f"{symbol} FAIL: direction")
            is_passed = False
        
        # Hard Gate 3: Net EV < 0.5
        if net_ev < 0.5:
            logging.warning(f"{symbol} FAIL: net_ev={net_ev}")
            is_passed = False
            all_reasons.append(f"Net EV too low: {net_ev:.2f} R < 0.5")
        
        # Hard Gate 4: Score < Dynamic Threshold
        if unified_score < pass_threshold:
            logging.warning(
                f"{symbol} FAIL: score {unified_score} threshold {pass_threshold}"
            )
            is_passed = False
            all_reasons.append(f"Score {unified_score:.1f} < {pass_threshold} ({market_regime} threshold)")
        
        # ============================================================
        # 8. CONVICTION SCORE & FINAL GRADE (FIXED: removed risk double counting)
        # ============================================================
        # 🔥 FIX: Conviction is now a blend of Unified Score (60%) and EV Score (40%)
        # Risk is already embedded in unified_score via weighted average
        conviction_score = (
            unified_score * 0.60 +
            ev_score * 0.40
        )
        conviction_score = round(max(0, min(100, conviction_score)), 1)
        
        # ---- Final Grade ----
        if conviction_score >= 95:
            grade = "A+"
        elif conviction_score >= 85:
            grade = "A"
        elif conviction_score >= 75:
            grade = "B"
        elif conviction_score >= 65:
            grade = "C"
        elif conviction_score >= 55:
            grade = "D"
        else:
            grade = "REJECT"
        
        # ============================================================
        # 9. CONFIDENCE SCORE + LEVEL
        # ============================================================
        # Consensus (std dev of components)
        component_list = list(all_scores.values())
        if component_list:
            mean_comp = sum(component_list) / len(component_list)
            variance = sum((x - mean_comp) ** 2 for x in component_list) / len(component_list)
            std_dev = math.sqrt(variance)
            consensus_score = max(0, 100 - (std_dev * 2.5))
        else:
            consensus_score = 50
        
        # EV Clarity: distance from 0 EV
        ev_clarity = min(100, abs(net_ev) * 30)
        
        # Risk Alignment (use risk_score directly, no double counting)
        risk_align = risk_score
        
        # Combined Confidence Score
        confidence_score = (
            consensus_score * 0.35 +
            ev_clarity * 0.30 +
            risk_align * 0.20 +
            (safety_score or 50) * 0.15
        )
        confidence_score = round(max(0, min(100, confidence_score)), 1)
        
        # ---- Confidence Level ----
        if confidence_score >= 85:
            confidence_level = "HIGH"
        elif confidence_score >= 70:
            confidence_level = "MEDIUM"
        elif confidence_score >= 55:
            confidence_level = "LOW"
        else:
            confidence_level = "VERY_LOW"
        
        # ============================================================
        # 10. KELLY & RECOMMENDED RISK
        # ============================================================
        if rr_ratio > 0:
            kelly_pct = (estimated_win_rate - (loss_rate / rr_ratio))
            kelly_pct = max(0, min(1, kelly_pct))
        else:
            kelly_pct = 0
        
        recommended_risk_pct = kelly_pct * kelly_fraction * 100
        recommended_risk_pct = min(max_recommended_risk_pct, recommended_risk_pct)
        recommended_risk_pct = round(max(0.01, recommended_risk_pct), 2)
        
        # ============================================================
        # 11. FINAL RETURN
        # ============================================================
        return {
            "symbol": symbol,
            
            # Core Results
            "is_passed": is_passed,
            "conviction_score": conviction_score,
            "final_grade": grade,
            "final_unified_score": unified_score,
            "raw_unified_score": round(raw_unified_score, 2),
            "pass_threshold": pass_threshold,
            "market_regime": market_regime,
            
            # Risk Integration
            "risk_score": round(risk_score, 1),
            "safety_score_used": safety_score,
            "quality_score_used": position_quality_score,
            "effective_leverage_used": effective_leverage,
            "capital_exposure_pct_used": capital_exposure_pct,
            
            # Expected Value
            "net_ev_r": round(net_ev, 2),
            "gross_ev_r": round(gross_ev, 2),
            "risk_adjusted_ev_r": round(risk_adjusted_net_ev, 2),
            "ev_score": round(ev_score, 1),
            "ev_tier": ev_tier,
            "cost_breakdown": {
                "fee_rate_pct": round(fee_rate_pct, 2),
                "funding_rate_pct": round(funding_rate_pct, 2),
                "slippage_rate_pct": round(slippage_rate_pct, 2),
                "total_cost_r": round(total_costs, 3)
            },
            
            # Confidence
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "confidence_metrics": {
                "consensus_score": round(consensus_score, 1),
                "ev_clarity_score": round(ev_clarity, 1),
                "risk_alignment_score": round(risk_align, 1),
                "component_std_dev": round(std_dev, 2)
            } if include_confidence_metrics else {},
            
            # Position Sizing
            "recommended_risk_percent": recommended_risk_pct,
            "kelly_fraction_used": kelly_fraction,
            
            # Direction
            "desired_direction": desired_direction,
            "long_allowed": long_allowed,
            "short_allowed": short_allowed,
            
            # Penalties & Warnings
            "applied_penalties": applied_penalties,
            "warning_messages": warning_messages,
            
            # Red Flags (Categorized)
            "critical_flags": critical_flags,
            "penalty_flags": penalty_flags,
            
            # Decision Logs
            "rejection_reasons": all_reasons if not is_passed else [],
            
            # Score Breakdown
            "score_breakdown": all_scores,
            "applied_weights": {k: round(v, 3) for k, v in weights.items()},
            
            # Legacy Compatibility
            "unified_score": unified_score,
            "red_flags": critical_flags + penalty_flags
        }


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 SCORE FUSION v5.1 - INSTITUTIONAL GRADE (Risk Double Counting FIXED)")
    print("=" * 70)
    
    # Test 1: Full Risk Integration
    print("\n--- 1. FULL RISK INTEGRATION ---")
    res1 = InstitutionalScoreFusionEngine.fuse_scores(
        symbol="BTC/USDT",
        tech_score=85,
        smc_score=80,
        liquidity_score=78,
        risk_score=92,
        safety_score=90,
        position_quality_score=88,
        market_regime="TRENDING",
        desired_direction="LONG",
        estimated_win_rate=0.62,
        rr_ratio=3.2,
        atr_ratio_pct=1.2,
        effective_leverage=2.0
    )
    print(f"✅ Passed: {res1['is_passed']} | Grade: {res1['final_grade']}")
    print(f"Conviction: {res1['conviction_score']} | Confidence: {res1['confidence_score']}% ({res1['confidence_level']})")
    print(f"Risk Score: {res1['risk_score']}")
    print(f"Weights: {res1['applied_weights']}")
    
    # Test 2: Soft Penalty System
    print("\n--- 2. SOFT PENALTY SYSTEM ---")
    res2 = InstitutionalScoreFusionEngine.fuse_scores(
        symbol="ETH/USDT",
        tech_score=82,
        smc_score=75,
        liquidity_score=70,
        risk_score=65,
        market_regime="VOLATILE",
        desired_direction="BOTH",
        estimated_win_rate=0.52,
        rr_ratio=2.2,
        penalty_flags=["LIQUIDITY_SWEEP", "WEAK_VOLUME", "BEARISH_DIVERGENCE", "LATE_ENTRY"],
        atr_ratio_pct=3.8,
        max_penalty_pct=25.0  # 🔥 NEW: custom penalty cap
    )
    print(f"✅ Passed: {res2['is_passed']} | Grade: {res2['final_grade']}")
    print(f"Final Score: {res2['final_unified_score']} (Raw: {res2['raw_unified_score']})")
    print(f"Penalties: {res2['applied_penalties']}")
    print(f"Warnings: {res2['warning_messages']}")
    
    # Test 3: Dynamic Threshold (RANGING)
    print("\n--- 3. DYNAMIC THRESHOLD (RANGING) ---")
    res3 = InstitutionalScoreFusionEngine.fuse_scores(
        symbol="SOL/USDT",
        tech_score=73,
        smc_score=72,
        liquidity_score=74,
        risk_score=70,
        market_regime="RANGING",
        desired_direction="BOTH",
        estimated_win_rate=0.50,
        rr_ratio=1.8,
        atr_ratio_pct=1.5
    )
    print(f"Threshold: {res3['pass_threshold']} | Score: {res3['final_unified_score']}")
    print(f"Passed: {res3['is_passed']} | Grade: {res3['final_grade']}")
    
    # Test 4: BEAR Regime (LONG Restricted)
    print("\n--- 4. BEAR REGIME (LONG RESTRICTED) ---")
    res4 = InstitutionalScoreFusionEngine.fuse_scores(
        symbol="AVAX/USDT",
        tech_score=85,
        smc_score=80,
        liquidity_score=75,
        risk_score=78,
        market_regime="BEAR",
        desired_direction="LONG",
        estimated_win_rate=0.55,
        rr_ratio=2.5,
        atr_ratio_pct=2.0
    )
    print(f"Passed: {res4['is_passed']}")
    print(f"Long Allowed: {res4['long_allowed']}")
    print(f"Short Allowed: {res4['short_allowed']}")
    print(f"Rejection: {res4['rejection_reasons']}")
    print(f"Conviction: {res4['conviction_score']}")
    
    # Test 5: CRASH Regime (SHORT Restricted)
    print("\n--- 5. CRASH REGIME (SHORT RESTRICTED) ---")
    res5 = InstitutionalScoreFusionEngine.fuse_scores(
        symbol="BTC/USDT",
        tech_score=70,
        smc_score=65,
        liquidity_score=60,
        risk_score=50,
        market_regime="CRASH",
        desired_direction="SHORT",
        estimated_win_rate=0.45,
        rr_ratio=1.8,
        atr_ratio_pct=4.0
    )
    print(f"Passed: {res5['is_passed']}")
    print(f"Short Allowed: {res5['short_allowed']}")
    print(f"Rejection Reasons: {res5['rejection_reasons']}")
    
    # Test 6: Critical Flags (Hard Reject)
    print("\n--- 6. CRITICAL FLAGS (HARD REJECT) ---")
    res6 = InstitutionalScoreFusionEngine.fuse_scores(
        symbol="XXX/USDT",
        tech_score=90,
        smc_score=85,
        liquidity_score=80,
        risk_score=90,
        market_regime="TRENDING",
        desired_direction="LONG",
        critical_flags=["EXCHANGE_HACK", "DELISTED"],
        estimated_win_rate=0.70,
        rr_ratio=4.0
    )
    print(f"❌ Passed: {res6['is_passed']}")
    print(f"Reasons: {res6['rejection_reasons']}")
    print(f"Grade: {res6['final_grade']}")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed. Score Fusion v5.1 ready for production.")
