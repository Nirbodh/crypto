# python-engine/risk_engine.py

import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RiskEngine:
    """
    Institutional Grade Quantitative Risk & Position Sizing Engine.
    Integrates Smart Money Concepts (SMC) Structural Invalidations, ATR Dynamics,
    Strict Portfolio Capital Preservation, and Liquidation Safety Buffers.
    """

    @staticmethod
    def calculate_trade_risk(
        entry_price: float,
        atr_5m: float,
        account_balance: float = 1000.0,
        risk_per_trade_percent: float = 1.0,
        sl_atr_multiplier: float = 1.5,
        tp_atr_multiplier: float = 3.0,
        min_rr_ratio: float = 1.5,
        direction: str = "LONG",
        max_leverage: float = 10.0,
        custom_sl_price: Optional[float] = None,  # SMC Order Block or Swing High/Low Invalidation
        maintenance_margin_pct: float = 0.5      # Exchange Maintenance Margin %
    ) -> Dict[str, Any]:
        
        # 1. Input Guard Clauses
        if not entry_price or entry_price <= 0 or not atr_5m or atr_5m <= 0:
            return {
                "valid_trade": False,
                "reason": "Invalid entry price or ATR input parameters."
            }
        
        if account_balance <= 0 or risk_per_trade_percent <= 0:
            return {
                "valid_trade": False,
                "reason": "Account balance and risk percentage must be strictly positive."
            }

        direction = direction.upper()
        if direction not in ["LONG", "SHORT"]:
            return {"valid_trade": False, "reason": f"Unsupported direction: {direction}"}

        target_risk_amount = account_balance * (risk_per_trade_percent / 100.0)

        # 2. Determine Stop Loss Level (Custom Structural Invalidation vs. ATR Fallback)
        if custom_sl_price and custom_sl_price > 0:
            stop_loss = custom_sl_price
            sl_method = "SMC_STRUCTURAL_INVALIDATION"
        else:
            sl_method = "DYNAMIC_ATR"
            if direction == "SHORT":
                stop_loss = entry_price + (sl_atr_multiplier * atr_5m)
            else:  # LONG
                stop_loss = entry_price - (sl_atr_multiplier * atr_5m)

        risk_distance = abs(entry_price - stop_loss)

        # 3. Structural Validity & Directional Integrity Checks
        if stop_loss <= 0 or risk_distance <= 0:
            return {
                "valid_trade": False,
                "reason": "Invalid Stop Loss level: zero or negative risk distance."
            }

        if direction == "LONG" and stop_loss >= entry_price:
            return {"valid_trade": False, "reason": "Long Stop Loss must be strictly below Entry Price."}
        elif direction == "SHORT" and stop_loss <= entry_price:
            return {"valid_trade": False, "reason": "Short Stop Loss must be strictly above Entry Price."}

        # 4. Multi-Target Profit Distribution (TP1 Scale-Out @ 1.5RR, TP2 @ Structural Target)
        if direction == "LONG":
            tp1_price = entry_price + (risk_distance * 1.5)
            take_profit = entry_price + (tp_atr_multiplier * atr_5m)
        else:  # SHORT
            tp1_price = entry_price - (risk_distance * 1.5)
            take_profit = entry_price - (tp_atr_multiplier * atr_5m)

        reward_distance = abs(take_profit - entry_price)

        # 5. Position Sizing & Leverage Enforcement
        raw_quantity = target_risk_amount / risk_distance
        raw_position_value = raw_quantity * entry_price

        max_allowed_position_value = account_balance * max_leverage
        leverage_capped = False

        if raw_position_value > max_allowed_position_value:
            position_value = max_allowed_position_value
            quantity = position_value / entry_price
            leverage_capped = True
            # Recalculate actual capped dollar risk
            actual_risk_amount = quantity * risk_distance
        else:
            position_value = raw_position_value
            quantity = raw_quantity
            actual_risk_amount = target_risk_amount

        effective_leverage = position_value / account_balance if account_balance > 0 else 0.0

        # 6. Isolated Liquidation Price Calculation & Buffer Validation
        mm_ratio = maintenance_margin_pct / 100.0

        if effective_leverage > 0:
            margin_ratio = 1.0 / effective_leverage
            if direction == "LONG":
                est_liquidation_price = entry_price * (1.0 - margin_ratio + mm_ratio)
                # Invalidation: SL must be HIGHER than Liquidation Price
                liq_buffer_valid = stop_loss > est_liquidation_price
            else:  # SHORT
                est_liquidation_price = entry_price * (1.0 + margin_ratio - mm_ratio)
                # Invalidation: SL must be LOWER than Liquidation Price
                liq_buffer_valid = stop_loss < est_liquidation_price
        else:
            est_liquidation_price = 0.0
            liq_buffer_valid = True

        # 7. Quantitative Trade Evaluation & Invalidation Logic
        rr_ratio = reward_distance / risk_distance
        is_valid = (rr_ratio >= min_rr_ratio) and liq_buffer_valid

        invalidation_reasons = []
        if rr_ratio < min_rr_ratio:
            invalidation_reasons.append(
                f"Risk/Reward Ratio (1:{rr_ratio:.2f}) is below minimum threshold (1:{min_rr_ratio:.2f})."
            )
        if not liq_buffer_valid:
            invalidation_reasons.append(
                f"Liquidation Hazard! Stop Loss ({stop_loss:.4f}) breaches Estimated Liquidation Level ({est_liquidation_price:.4f})."
            )

        # Percentage Computations
        sl_pct = (risk_distance / entry_price) * 100.0
        tp_pct = (reward_distance / entry_price) * 100.0

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
                "effective_leverage_needed": round(effective_leverage, 2),
                "leverage_capped": leverage_capped,
                "estimated_liquidation_price": round(est_liquidation_price, 4)
            },
            "risk_metrics": {
                "atr_5m": round(atr_5m, 4),
                "risk_reward_ratio": f"1:{round(rr_ratio, 2)}",
                "rr_score_raw": round(rr_ratio, 2),
                "sl_method": sl_method
            }
        }


if __name__ == "__main__":
    print("--- 1. STANDARD LONG POSITION TEST ---")
    long_res = RiskEngine.calculate_trade_risk(
        entry_price=65000.0,
        atr_5m=350.0,
        account_balance=1000.0,
        direction="LONG"
    )
    print("Valid Trade:", long_res["valid_trade"])
    print("Trade Levels:", long_res["trade_levels"])
    print("Position Sizing:", long_res["position_sizing"])

    print("\n--- 2. SMC STRUCTURAL SHORT POSITION TEST ---")
    smc_res = RiskEngine.calculate_trade_risk(
        entry_price=65000.0,
        atr_5m=350.0,
        account_balance=1000.0,
        direction="SHORT",
        custom_sl_price=65800.0  # Order Block Invalidation
    )
    print("Valid Trade:", smc_res["valid_trade"])
    print("SL Method:", smc_res["risk_metrics"]["sl_method"])
    print("Trade Levels:", smc_res["trade_levels"])

    print("\n--- 3. HIGH LEVERAGE LIQUIDATION HAZARD TEST ---")
    hazard_res = RiskEngine.calculate_trade_risk(
        entry_price=65000.0,
        atr_5m=1200.0,
        account_balance=1000.0,
        direction="LONG",
        max_leverage=50.0,
        risk_per_trade_percent=5.0
    )
    print("Valid Trade:", hazard_res["valid_trade"])
    print("Invalidation Reasons:", hazard_res["invalidation_reasons"])