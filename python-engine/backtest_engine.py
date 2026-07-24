import pandas as pd
import numpy as np
from typing import List, Dict, Any

class QuantBacktestEngine:
    """
    Institutional Quantitative Backtest Engine:
    - Evaluates historical trade signals (TP/SL outcomes)
    - Supports Long & Short Position Calculations
    - Computes Sharpe Ratio, Profit Factor, Win Rate, and Max Drawdown with Division Guards
    """
    def __init__(self, initial_capital: float = 10000.0, risk_per_trade_pct: float = 0.02, risk_free_rate: float = 0.02):
        self.initial_capital = float(initial_capital)
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.risk_free_rate = risk_free_rate
        self.trade_history = []

    def run_historical_backtest(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        trades_count = len(signals)
        if trades_count == 0:
            return {"error": "No historical signals provided."}

        current_capital = self.initial_capital
        wins, losses = 0, 0
        total_profit, total_loss = 0.0, 0.0
        equity_curve = [current_capital]
        returns_list = []

        for sig in signals:
            risk_amount = current_capital * self.risk_per_trade_pct
            entry = float(sig.get('entry_price', 0))
            sl = float(sig.get('sl_price', 0))
            tp = float(sig.get('tp_price', 0))
            direction = sig.get('direction', 'LONG').upper()

            if entry <= 0 or sl <= 0 or tp <= 0:
                continue

            # Calculate Risk to Reward ratio dynamically based on direction
            if direction == 'LONG':
                sl_dist = abs(entry - sl)
                tp_dist = abs(tp - entry)
            else: # SHORT
                sl_dist = abs(sl - entry)
                tp_dist = abs(entry - tp)

            if sl_dist == 0:
                continue

            rr_ratio = tp_dist / sl_dist
            outcome = sig.get('actual_outcome', 'SL').upper()

            if outcome == 'TP':
                wins += 1
                pnl = risk_amount * rr_ratio
                total_profit += pnl
            else:
                losses += 1
                pnl = -risk_amount
                total_loss += abs(pnl)

            prev_cap = current_capital
            current_capital += pnl
            equity_curve.append(current_capital)
            returns_list.append((current_capital - prev_cap) / prev_cap if prev_cap > 0 else 0.0)

        actual_trades_count = wins + losses
        if actual_trades_count == 0:
            return {"error": "No valid trades executed."}

        win_rate = (wins / actual_trades_count) * 100.0

        # ✅ FIX: Profit Factor - total_loss ০ হলে অথবা ইনফিনিটি হ্যান্ডেল করতে সেফটি গার্ড
        profit_factor = 999.0
        if total_loss > 0:
            profit_factor = round(total_profit / total_loss, 2)
        elif total_profit > 0:
            profit_factor = 999.0  # Infinite profit factor when there are zero losses

        # Max Drawdown
        equity_series = pd.Series(equity_curve)
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_drawdown = round(drawdown.min() * 100.0, 2) if not drawdown.empty else 0.0

        # ✅ FIX: Sharpe Ratio - স্ট্যান্ডার্ড ডেভিয়েশন ০ বা রিটার্ন সিরিজ খালি হলে ক্র্যাশ রোধ করা
        returns_series = pd.Series(returns_list)
        sharpe_ratio = 0.0
        if not returns_series.empty and returns_series.std() > 0:
            sharpe_ratio = round((returns_series.mean() - (self.risk_free_rate / 365)) / returns_series.std() * np.sqrt(365), 2)

        net_roi = ((current_capital - self.initial_capital) / self.initial_capital) * 100.0

        return {
            "total_trades": actual_trades_count,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "starting_capital": self.initial_capital,
            "ending_capital": round(current_capital, 2),
            "net_roi_pct": round(net_roi, 2)
        }