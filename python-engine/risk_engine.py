# python-engine/risk_engine.py

class RiskEngine:

    @staticmethod
    def calculate_trade_risk(entry_price, atr_5m, account_balance=1000.0, risk_per_trade_percent=1.0, 
                             sl_atr_multiplier=1.5, tp_atr_multiplier=3.0, min_rr_ratio=1.5):
        """
        Calculates Trade Levels, Position Sizing, and Risk/Reward parameters.
        """
        if not entry_price or not atr_5m or atr_5m <= 0:
            return {
                "valid_trade": False,
                "reason": "Invalid entry price or ATR data"
            }

        risk_amount = account_balance * (risk_per_trade_percent / 100.0)

        stop_loss = entry_price - (sl_atr_multiplier * atr_5m)
        take_profit = entry_price + (tp_atr_multiplier * atr_5m)

        risk_distance = abs(entry_price - stop_loss)
        reward_distance = abs(take_profit - entry_price)

        # 🛑 Guard: Check if risk distance is zero or negative
        if stop_loss <= 0 or risk_distance <= 0:
            return {
                "valid_trade": False,
                "reason": "Invalid risk distance"
            }

        quantity = risk_amount / risk_distance
        position_value = quantity * entry_price
        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0

        is_valid = rr_ratio >= min_rr_ratio

        return {
            "valid_trade": is_valid,
            "account_metrics": {
                "account_balance_usdt": round(account_balance, 2),
                "risk_percent": risk_per_trade_percent,
                "max_risk_usdt": round(risk_amount, 2)
            },
            "trade_levels": {
                "entry_price": round(entry_price, 6),
                "stop_loss_price": round(stop_loss, 6),
                "take_profit_price": round(take_profit, 6),
                "sl_percentage": round(((entry_price - stop_loss) / entry_price) * 100, 2),
                "tp_percentage": round(((take_profit - entry_price) / entry_price) * 100, 2)
            },
            "position_sizing": {
                "quantity": round(quantity, 4),
                "position_value_usdt": round(position_value, 2),
                "effective_leverage_needed": round(position_value / account_balance, 2) if account_balance > 0 else 0
            },
            "risk_metrics": {
                "atr_5m": round(atr_5m, 6),
                "risk_reward_ratio": f"1:{round(rr_ratio, 2)}",
                "rr_score_raw": round(rr_ratio, 2)
            }
        }