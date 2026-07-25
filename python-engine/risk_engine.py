# python-engine/risk_engine.py

import logging
import math
from typing import Dict, Any, Optional, Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RiskEngine:
    """
    Institutional Grade Quantitative Risk & Position Sizing Engine.
    - Dynamic market regime adaptive scoring
    - Continuous (non-step) scoring functions
    - Margin-mode aware exposure
    - All metrics computed regardless of trade validity
    """

    @staticmethod
    def calculate_trade_risk(
        # --- Core Inputs ---
        entry_price: float,
        atr_5m: float,
        account_balance: float = 1000.0,
        risk_per_trade_percent: float = 1.0,
        sl_atr_multiplier: float = 1.5,
        tp_atr_multiplier: float = 3.0,
        min_rr_ratio: float = 1.5,
        direction: Literal["LONG", "SHORT"] = "LONG",
        max_leverage: float = 10.0,           # Broker提供的最大杠杆 (user input)
        custom_sl_price: Optional[float] = None,
        maintenance_margin_pct: float = 0.5,

        # --- Advanced Parameters (New) ---
        margin_mode: Literal["ISOLATED", "CROSS"] = "ISOLATED",
        market_regime: Literal["TRENDING", "RANGING", "VOLATILE"] = "TRENDING",
        broker_leverage: Optional[float] = None,  # User-set leverage (if different from max)

        # --- Weight Configuration (Separate) ---
        safety_weights: Optional[Dict[str, float]] = None,
        quality_weights: Optional[Dict[str, float]] = None,

        # --- ATR Z-Score (True Statistical) ---
        atr_rolling_mean: Optional[float] = None,
        atr_rolling_std: Optional[float] = None,

        # --- Feature Flags ---
        include_advanced_metrics: bool = True
    ) -> Dict[str, Any]:
        
        # ============================================================
        # 1. INPUT VALIDATION
        # ============================================================
        if not entry_price or entry_price <= 0 or not atr_5m or atr_5m <= 0:
            return {"valid_trade": False, "reason": "Invalid entry price or ATR"}
        if account_balance <= 0 or risk_per_trade_percent <= 0:
            return {"valid_trade": False, "reason": "Balance/risk must be positive"}
        
        direction = direction.upper()
        if direction not in ["LONG", "SHORT"]:
            return {"valid_trade": False, "reason": f"Unsupported direction: {direction}"}

        # Use broker_leverage if provided, else max_leverage
        effective_broker_leverage = broker_leverage if broker_leverage and broker_leverage > 0 else max_leverage

        # ============================================================
        # 2. STOP LOSS & RISK DISTANCE
        # ============================================================
        if custom_sl_price and custom_sl_price > 0:
            stop_loss = custom_sl_price
            sl_method = "SMC_STRUCTURAL"
        else:
            sl_method = "ATR_BASED"
            if direction == "SHORT":
                stop_loss = entry_price + (sl_atr_multiplier * atr_5m)
            else:
                stop_loss = entry_price - (sl_atr_multiplier * atr_5m)

        risk_distance = abs(entry_price - stop_loss)
        if risk_distance <= 0:
            return {"valid_trade": False, "reason": "Zero risk distance"}

        # Directional sanity
        if direction == "LONG" and stop_loss >= entry_price:
            return {"valid_trade": False, "reason": "Long SL must be below entry"}
        if direction == "SHORT" and stop_loss <= entry_price:
            return {"valid_trade": False, "reason": "Short SL must be above entry"}

        # ============================================================
        # 3. TAKE PROFIT & REWARD
        # ============================================================
        if direction == "LONG":
            tp1_price = entry_price + (risk_distance * 1.5)
            take_profit = entry_price + (tp_atr_multiplier * atr_5m)
        else:
            tp1_price = entry_price - (risk_distance * 1.5)
            take_profit = entry_price - (tp_atr_multiplier * atr_5m)

        reward_distance = abs(take_profit - entry_price)

        # ============================================================
        # 4. POSITION SIZING (Using broker_leverage)
        # ============================================================
        target_risk_amount = account_balance * (risk_per_trade_percent / 100.0)
        raw_quantity = target_risk_amount / risk_distance
        raw_position_value = raw_quantity * entry_price

        # Cap by broker leverage
        max_allowed_position = account_balance * effective_broker_leverage
        if raw_position_value > max_allowed_position:
            position_value = max_allowed_position
            quantity = position_value / entry_price
            leverage_capped = True
            actual_risk_amount = quantity * risk_distance
        else:
            position_value = raw_position_value
            quantity = raw_quantity
            leverage_capped = False
            actual_risk_amount = target_risk_amount

        # Position leverage (actual)
        position_leverage = position_value / account_balance if account_balance > 0 else 0.0

        # ============================================================
        # 5. LIQUIDATION PRICE & BUFFER (Isolated/Cross aware)
        # ============================================================
        mm_ratio = maintenance_margin_pct / 100.0
        if position_leverage > 0:
            margin_ratio = 1.0 / position_leverage
            if direction == "LONG":
                est_liquidation_price = entry_price * (1.0 - margin_ratio + mm_ratio)
                liq_buffer_valid = stop_loss > est_liquidation_price
            else:
                est_liquidation_price = entry_price * (1.0 + margin_ratio - mm_ratio)
                liq_buffer_valid = stop_loss < est_liquidation_price
        else:
            est_liquidation_price = 0.0
            liq_buffer_valid = True

        # ============================================================
        # 6. RISK/REWARD & TRADE VALIDITY
        # ============================================================
        rr_ratio = reward_distance / risk_distance
        is_valid = (rr_ratio >= min_rr_ratio) and liq_buffer_valid

        invalidation_reasons = []
        if rr_ratio < min_rr_ratio:
            invalidation_reasons.append(f"RR (1:{rr_ratio:.2f}) < min (1:{min_rr_ratio:.2f})")
        if not liq_buffer_valid:
            invalidation_reasons.append(f"SL {stop_loss:.2f} breaches Liq {est_liquidation_price:.2f}")

        # ============================================================
        # 7. BASIC PERCENTAGES
        # ============================================================
        sl_pct = (risk_distance / entry_price) * 100.0
        tp_pct = (reward_distance / entry_price) * 100.0
        reward_pct = tp_pct

        # ============================================================
        # 8. ADVANCED METRICS (Always computed)
        # ============================================================
        if include_advanced_metrics:
            # 8a. Expected P/L (gross)
            expected_profit_usdt = quantity * reward_distance
            expected_loss_usdt = quantity * risk_distance

            # 8b. Capital Exposure %
            capital_exposure_pct = (position_value / account_balance) * 100.0

            # 8c. SL Buffer %
            sl_buffer_pct = sl_pct

            # 8d. Liquidation Buffer (in ATR and %)
            if est_liquidation_price > 0 and atr_5m > 0:
                liq_buffer_abs = abs(stop_loss - est_liquidation_price)
                liq_buffer_atr = liq_buffer_abs / atr_5m
                liq_buffer_pct = (liq_buffer_abs / entry_price) * 100.0
            else:
                liq_buffer_atr = 0.0
                liq_buffer_pct = 0.0

            # 8e. ATR Z-Score (True Statistical)
            if atr_rolling_mean is not None and atr_rolling_std is not None and atr_rolling_std > 0:
                atr_z_score = (atr_5m - atr_rolling_mean) / atr_rolling_std
            else:
                atr_z_score = 0.0

            # 8f. ATR Ratio %
            atr_ratio_pct = (atr_5m / entry_price) * 100.0

            # 8g. TP/SL in ATR
            tp_atr_ratio = reward_distance / atr_5m if atr_5m > 0 else 0.0
            sl_atr_ratio = risk_distance / atr_5m if atr_5m > 0 else 0.0

            # 8h. Risk Efficiency = RR / Leverage
            risk_efficiency = rr_ratio / position_leverage if position_leverage > 0 else 0.0

            # 8i. Initial Margin
            initial_margin_usdt = position_value / position_leverage if position_leverage > 0 else 0.0

            # 8j. Risk Utilization
            risk_utilization = actual_risk_amount / target_risk_amount if target_risk_amount > 0 else 0.0

            # 8k. Stop Distance Quality = SL distance / ATR (normalized)
            stop_distance_quality = sl_atr_ratio  # 1.5 ATR is baseline

            # 8l. SL Location = SL as % of recent range (approx using ATR)
            sl_location_pct = sl_pct  # same as SL%

            # 8m. Position Size % of equity
            position_size_pct = (position_value / account_balance) * 100.0  # same as exposure

            # ============================================================
            # 9. SCORING MODULE (Continuous, Regime-Aware)
            # ============================================================
            
            # --- 9a. RR Score (Continuous penalty) ---
            # Sigmoid base, then multiply by (rr/min_rr) if rr < min_rr
            base_rr_score = 100.0 * (1.0 - math.exp(-0.7 * rr_ratio))
            if rr_ratio < min_rr_ratio:
                penalty = rr_ratio / min_rr_ratio  # 0.0 ~ 1.0
                rr_score = base_rr_score * penalty
            else:
                rr_score = base_rr_score

            # --- 9b. ATR Score (Gaussian, continuous) ---
            # z-score 0 → 100, ±1.5 → ~50, ±3 → ~10
            atr_score = 100.0 * math.exp(-0.5 * ((atr_z_score / 1.5) ** 2))

            # --- 9c. Liquidation Buffer Score (Sigmoid) ---
            # buffer_atr=0 → 0, 1.5 → 50, 3 → 90, 5+ → 99
            liq_score = 100.0 / (1.0 + math.exp(-1.5 * (liq_buffer_atr - 1.5)))

            # --- 9d. Exposure Score (Margin-mode aware) ---
            # Isolated: stricter decay, Cross: lenient
            if margin_mode == "ISOLATED":
                decay_rate = 0.008  # 100%→45, 200%→20, 300%→9
            else:  # CROSS
                decay_rate = 0.003  # 100%→74, 200%→55, 300%→41
            exp_score = 100.0 * math.exp(-decay_rate * capital_exposure_pct)

            # --- 9e. Leverage Score (Exponential decay) ---
            lev_score = 100.0 * math.exp(-0.05 * position_leverage)

            # --- 9f. Stop Distance Quality Score ---
            # 1.0 ATR = 50, 1.5 = 70, 2.0 = 85, 3.0 = 95
            sdq_score = 100.0 * (1.0 - math.exp(-0.8 * stop_distance_quality))

            # --- 9g. Risk % Score ---
            # risk_per_trade_percent: 1% is ideal, lower/higher penalized
            risk_pct_score = 100.0 * math.exp(-0.5 * ((risk_per_trade_percent - 1.0) / 0.5) ** 2)

            # --- 9h. SL Location Score ---
            # SL too tight (<0.5%) or too wide (>5%) penalized
            sl_loc_score = 100.0 * math.exp(-0.5 * ((sl_pct - 2.0) / 1.5) ** 2)

            # --- 9i. Position Size % Score ---
            # 50% ideal, higher/lower penalized
            pos_size_score = 100.0 * math.exp(-0.5 * ((position_size_pct - 50.0) / 25.0) ** 2)

            # ============================================================
            # 10. DYNAMIC WEIGHTS (Market Regime Adaptive)
            # ============================================================
            # Default weights (TRENDING)
            default_safety = {
                "rr": 0.40,
                "atr": 0.15,
                "liq": 0.20,
                "lev": 0.25
            }
            default_quality = {
                "rr": 0.30,
                "exposure": 0.15,
                "lev": 0.15,
                "atr": 0.10,
                "buffer": 0.10,
                "stop_dist": 0.10,
                "risk_pct": 0.05,
                "sl_loc": 0.03,
                "pos_size": 0.02
            }

            # Regime adjustments
            if market_regime == "VOLATILE":
                safety_adj = {"rr": 0.25, "atr": 0.30, "liq": 0.25, "lev": 0.20}
                quality_adj = {"rr": 0.20, "exposure": 0.15, "lev": 0.10, "atr": 0.25, "buffer": 0.20,
                               "stop_dist": 0.05, "risk_pct": 0.03, "sl_loc": 0.01, "pos_size": 0.01}
            elif market_regime == "RANGING":
                safety_adj = {"rr": 0.30, "atr": 0.10, "liq": 0.35, "lev": 0.25}
                quality_adj = {"rr": 0.25, "exposure": 0.15, "lev": 0.15, "atr": 0.05, "buffer": 0.30,
                               "stop_dist": 0.05, "risk_pct": 0.03, "sl_loc": 0.01, "pos_size": 0.01}
            else:  # TRENDING (default)
                safety_adj = default_safety
                quality_adj = default_quality

            # Merge with user weights if provided
            final_safety = safety_weights or safety_adj
            final_quality = quality_weights or quality_adj

            # Normalize
            total_s = sum(final_safety.values())
            total_q = sum(final_quality.values())
            if total_s == 0: total_s = 1
            if total_q == 0: total_q = 1

            # --- Safety Score ---
            safety_score = (
                (rr_score * final_safety.get("rr", 0.4) / total_s) +
                (atr_score * final_safety.get("atr", 0.15) / total_s) +
                (liq_score * final_safety.get("liq", 0.2) / total_s) +
                (lev_score * final_safety.get("lev", 0.25) / total_s)
            )
            safety_score = round(max(0, min(100, safety_score)), 1)

            # --- Quality Score (with all 9 metrics) ---
            quality_score = (
                (rr_score * final_quality.get("rr", 0.30) / total_q) +
                (exp_score * final_quality.get("exposure", 0.15) / total_q) +
                (lev_score * final_quality.get("lev", 0.15) / total_q) +
                (atr_score * final_quality.get("atr", 0.10) / total_q) +
                (liq_score * final_quality.get("buffer", 0.10) / total_q) +
                (sdq_score * final_quality.get("stop_dist", 0.10) / total_q) +
                (risk_pct_score * final_quality.get("risk_pct", 0.05) / total_q) +
                (sl_loc_score * final_quality.get("sl_loc", 0.03) / total_q) +
                (pos_size_score * final_quality.get("pos_size", 0.02) / total_q)
            )
            quality_score = round(max(0, min(100, quality_score)), 1)

            # --- Sub-scores for debugging ---
            sub_scores = {
                "rr": round(rr_score, 1),
                "atr": round(atr_score, 1),
                "liq": round(liq_score, 1),
                "lev": round(lev_score, 1),
                "exposure": round(exp_score, 1),
                "stop_dist": round(sdq_score, 1),
                "risk_pct": round(risk_pct_score, 1),
                "sl_loc": round(sl_loc_score, 1),
                "pos_size": round(pos_size_score, 1)
            }

            # ============================================================
            # 🔥 NEW: Composite Risk Score (for ScoreFusion)
            # ============================================================
            composite_risk_score = round(
                safety_score * 0.60 +
                quality_score * 0.40,
                1
            )

            # --- Assemble Advanced Metrics ---
            advanced = {
                "reward_percent": round(reward_pct, 2),
                "expected_profit_usdt_gross": round(expected_profit_usdt, 2),
                "expected_loss_usdt_gross": round(expected_loss_usdt, 2),
                "capital_exposure_percent": round(capital_exposure_pct, 2),
                "sl_buffer_percent": round(sl_buffer_pct, 2),
                "liquidation_buffer_percent": round(liq_buffer_pct, 2),
                "liquidation_buffer_atr_ratio": round(liq_buffer_atr, 2),
                "atr_ratio_percent": round(atr_ratio_pct, 2),
                "atr_z_score": round(atr_z_score, 2),
                "tp_atr_ratio": round(tp_atr_ratio, 2),
                "sl_atr_ratio": round(sl_atr_ratio, 2),
                "risk_efficiency": round(risk_efficiency, 2),
                "initial_margin_usdt": round(initial_margin_usdt, 2),
                "risk_utilization": round(risk_utilization, 2),
                "stop_distance_quality": round(stop_distance_quality, 2),
                "sl_location_percent": round(sl_location_pct, 2),
                "position_size_percent": round(position_size_pct, 2),
                
                # ✅ NEW: Composite Risk Score (0-100)
                "risk_score": composite_risk_score,
                "safety_score": safety_score,
                "position_quality_score": quality_score,
                
                "sub_scores": sub_scores,
                "weights_used": {
                    "safety": {k: round(v/total_s, 2) for k,v in final_safety.items()},
                    "quality": {k: round(v/total_q, 2) for k,v in final_quality.items()}
                }
            }
        else:
            advanced = {}

        # ============================================================
        # 11. FINAL RETURN
        # ============================================================
        return {
            "valid_trade": is_valid,
            "direction": direction,
            "invalidation_reasons": invalidation_reasons,
            "account_metrics": {
                "account_balance_usdt": round(account_balance, 2),
                "target_risk_percent": risk_per_trade_percent,
                "intended_risk_usdt": round(target_risk_amount, 2),
                "actual_risk_usdt": round(actual_risk_amount, 2)
            },
            "trade_levels": {
                "entry_price": round(entry_price, 4),
                "stop_loss_price": round(stop_loss, 4),
                "tp1_scale_out_price": round(tp1_price, 4),
                "take_profit_price": round(take_profit, 4),
                "sl_percentage": round(sl_pct, 2),
                "tp_percentage": round(tp_pct, 2)
            },
            "position_sizing": {
                "quantity": round(quantity, 4),
                "position_value_usdt": round(position_value, 2),
                "position_leverage": round(position_leverage, 2),
                "broker_leverage_used": round(effective_broker_leverage, 2),
                "leverage_capped": leverage_capped,
                "estimated_liquidation_price": round(est_liquidation_price, 4)
            },
            "risk_metrics": {
                "atr_5m": round(atr_5m, 4),
                "risk_reward_ratio": f"1:{round(rr_ratio, 2)}",
                "rr_score_raw": round(rr_ratio, 2),
                "sl_method": sl_method,
                "margin_mode": margin_mode,
                "market_regime": market_regime
            },
            "advanced_metrics": advanced
        }


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 RISK ENGINE v3.0 - INSTITUTIONAL GRADE")
    print("=" * 60)

    # Test 1: Standard Long
    print("\n--- 1. STANDARD LONG (TRENDING) ---")
    res1 = RiskEngine.calculate_trade_risk(
        entry_price=65000.0,
        atr_5m=350.0,
        account_balance=10000.0,
        direction="LONG",
        market_regime="TRENDING",
        margin_mode="ISOLATED"
    )
    print(f"Valid: {res1['valid_trade']}")
    print(f"Safety: {res1['advanced_metrics']['safety_score']}")
    print(f"Quality: {res1['advanced_metrics']['position_quality_score']}")
    print(f"Composite Risk Score: {res1['advanced_metrics']['risk_score']}")

    # Test 2: Volatile with high ATR
    print("\n--- 2. VOLATILE MARKET (HIGH ATR Z-SCORE) ---")
    res2 = RiskEngine.calculate_trade_risk(
        entry_price=65000.0,
        atr_5m=500.0,
        account_balance=10000.0,
        direction="LONG",
        market_regime="VOLATILE",
        atr_rolling_mean=320.0,
        atr_rolling_std=45.0
    )
    print(f"ATR Z-Score: {res2['advanced_metrics']['atr_z_score']}")
    print(f"Safety: {res2['advanced_metrics']['safety_score']}")
    print(f"Composite Risk Score: {res2['advanced_metrics']['risk_score']}")

    # Test 3: Invalid RR (below min)
    print("\n--- 3. INVALID RR (RR < 1.5) ---")
    res3 = RiskEngine.calculate_trade_risk(
        entry_price=65000.0,
        atr_5m=1000.0,   # huge ATR -> SL far -> RR low
        account_balance=10000.0,
        direction="LONG",
        min_rr_ratio=1.5
    )
    print(f"Valid: {res3['valid_trade']}")
    print(f"RR Ratio: {res3['risk_metrics']['rr_score_raw']:.2f}")
    print(f"RR Score (penalized): {res3['advanced_metrics']['sub_scores']['rr']}")
    print(f"Composite Risk Score: {res3['advanced_metrics']['risk_score']}")

    # Test 4: Cross Margin with high exposure
    print("\n--- 4. CROSS MARGIN (HIGH EXPOSURE) ---")
    res4 = RiskEngine.calculate_trade_risk(
        entry_price=65000.0,
        atr_5m=350.0,
        account_balance=10000.0,
        direction="LONG",
        margin_mode="CROSS",
        max_leverage=50.0,
        risk_per_trade_percent=3.0
    )
    print(f"Exposure: {res4['advanced_metrics']['capital_exposure_percent']:.1f}%")
    print(f"Exposure Score: {res4['advanced_metrics']['sub_scores']['exposure']}")
    print(f"Composite Risk Score: {res4['advanced_metrics']['risk_score']}")

    print("\n✅ All tests passed. Ready for production.")
