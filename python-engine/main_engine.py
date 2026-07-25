# python-engine/main_engine.py

import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from exchange_manager import ExchangeManager
from coin_universe import CoinUniverseEngine
from data_fetcher import DataFetcher
from technical_engine import TechnicalEngine
from smc_engine import SMCEngine
from liquidity_engine import InstitutionalLiquidityEngine
from derivatives_engine import InstitutionalDerivativesEngine
from fundamental_engine import MultiSourceFundamentalEngine
from score_fusion_engine import InstitutionalScoreFusionEngine
from mtf_engine import MultiTimeframeEngine
from ai_engine import AIDebateEngine
from telegram_notifier import TelegramNotifier
from database_engine import DatabaseEngine
from indicators import TechnicalIndicators, SessionIndicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

COOLDOWN_HOURS = 4

class QuantTradingOrchestrator:
    def __init__(self):
        logging.info("⚙️ Initializing Institutional Quant Engine Orchestrator (Final Production Edition)...")
        self.ex_mgr = ExchangeManager()
        self.universe_engine = CoinUniverseEngine(self.ex_mgr)
        self.fetcher = DataFetcher()
        self.tech_engine = TechnicalEngine()
        self.smc_engine = SMCEngine()
        self.liq_engine = InstitutionalLiquidityEngine()
        self.deriv_engine = InstitutionalDerivativesEngine()
        self.fund_engine = MultiSourceFundamentalEngine()
        self.ai_engine = AIDebateEngine()
        self.telegram = TelegramNotifier()
        self.db = DatabaseEngine()

    def is_duplicate_signal(self, symbol: str) -> bool:
        return self.db.is_duplicate_signal(symbol, cooldown_hours=COOLDOWN_HOURS)

    def _process_single_asset(self, symbol: str, btc_context: dict) -> Dict[str, Any]:
        try:
            if self.is_duplicate_signal(symbol):
                return {"symbol": symbol, "status": "SKIPPED_COOLDOWN"}

            # 1. Fetch Multi-Timeframe OHLCV Data
            df_daily = self.fetcher.get_ohlcv(symbol, timeframe="1d", limit=100)
            df_4h    = self.fetcher.get_ohlcv(symbol, timeframe="4h", limit=100)
            df_1h    = self.fetcher.get_ohlcv(symbol, timeframe="1h", limit=100)
            df_30m   = self.fetcher.get_ohlcv(symbol, timeframe="30m", limit=100)
            df_15m   = self.fetcher.get_ohlcv(symbol, timeframe="15m", limit=100)

            if any(df is None or df.empty for df in [df_daily, df_4h, df_1h, df_30m, df_15m]):
                return {"symbol": symbol, "status": "SKIPPED_DATA_MISSING"}

            current_price = float(df_15m['close'].iloc[-1])

            # ================================================================
            # 2. Unified Indicator Calculation (CVD, Session, FVG Mitigation)
            # ================================================================
            df_15m_calc = TechnicalIndicators.calculate_all(df_15m.copy())
            
            # CVD Score (0-100) from indicators
            cvd_score = float(df_15m_calc['CVD_Score'].iloc[-1]) if 'CVD_Score' in df_15m_calc.columns else 50.0
            
            # Session Score (0-100) using SessionIndicators
            session_score = SessionIndicators.get_session_score(df_15m_calc)
            
            # FVG Mitigation Score (0-100) from indicators
            fvg_mitigation_score = float(df_15m_calc['FVG_Mitigation_Score'].iloc[-1]) if 'FVG_Mitigation_Score' in df_15m_calc.columns else 50.0

            # ================================================================
            # 3. Layer Calculations (Technical, SMC, Liquidity, MTF, Derivatives, Fundamental)
            # ================================================================
            tech_res = self.tech_engine.analyze(df_15m)
            daily_liq_res = self.liq_engine.analyze_liquidity(df_daily)
            h4_smc_res    = self.smc_engine.analyze(df_4h)
            h1_smc_res    = self.smc_engine.analyze(df_1h)
            m30_smc_res   = self.smc_engine.analyze(df_30m)

            # Check if FVG exists in 1H (from SMC engine)
            fvgs = h1_smc_res.get("fvgs", [])
            fvg_present = len(fvgs) > 0

            mtf_res = MultiTimeframeEngine.evaluate_mtf_alignment(
                daily_data={"bias": daily_liq_res.get("bias", "NEUTRAL"), "liquidity_swept": daily_liq_res.get("key_liquidity_swept", False)},
                h4_data={"choch_confirmed": h4_smc_res.get("choch_confirmed", False), "structure": h4_smc_res.get("market_structure", "NEUTRAL")},
                h1_data={"fvg_present": fvg_present, "ob_mitigated": h1_smc_res.get("ob_mitigated", False)},
                m30_data={"structure": m30_smc_res.get("market_structure", "NEUTRAL"), "choch_confirmed": m30_smc_res.get("choch_confirmed", False)},
                m15_data={"entry_signal": tech_res.get("momentum_bullish", False), "entry_type": "LONG"}
            )

            deriv_res = self.deriv_engine.analyze_derivatives(symbol)
            deriv_res["cvd_score"] = cvd_score  # Inject CVD score
            
            fund_res = self.fund_engine.fetch_fundamental_data(symbol)
            fear_greed = fund_res.get("fear_and_greed_index", 50.0)
            sentiment_score = float(fear_greed) if isinstance(fear_greed, (int, float)) else 50.0

            # ================================================================
            # 4. Dynamic SL, TP & Position Sizing
            # ================================================================
            atr_val = tech_res.get("atr", current_price * 0.02)
            swing_low = daily_liq_res.get("swing_low", current_price * 0.98)
            order_block_low = h1_smc_res.get("order_block_price", current_price * 0.985)
            
            sl_price = round(min(current_price - (atr_val * 1.5), swing_low, order_block_low), 4)
            if sl_price >= current_price:
                sl_price = round(current_price * 0.98, 4)

            risk_amount_per_unit = current_price - sl_price
            tp_price = round(current_price + (risk_amount_per_unit * 2.5), 4)

            account_balance = 10_000.0
            risk_percentage = 0.01
            allowed_risk_capital = account_balance * risk_percentage
            position_qty_by_risk = allowed_risk_capital / risk_amount_per_unit if risk_amount_per_unit > 0 else 500.0 / current_price
            
            pos_val = round(position_qty_by_risk * current_price, 2)
            pos_val = max(100.0, min(pos_val, 5000.0))
            qty = round(pos_val / current_price, 4)

            # ================================================================
            # 5. Unified Score Fusion Execution (with all new params)
            # ================================================================
            
            # ---- FIX: Properly handle red_flags from tech_res (dict) and fund_res (list) ----
            tech_flags = tech_res.get("red_flags", {})
            if isinstance(tech_flags, dict):
                tech_flags = (
                    tech_flags.get("critical", [])
                    + tech_flags.get("major", [])
                    + tech_flags.get("minor", [])
                )
            else:
                # If it's already a list, use it; otherwise fallback to empty list
                if not isinstance(tech_flags, list):
                    tech_flags = []
            
            fund_flags = fund_res.get("red_flags", [])
            if not isinstance(fund_flags, list):
                fund_flags = []
            
            red_flags = tech_flags + fund_flags
            # ----------------------------------------------------------------

            fusion_res = InstitutionalScoreFusionEngine.fuse_scores(
                symbol=symbol,
                tech_score=tech_res.get("technical_score", 50.0),
                smc_score=h1_smc_res.get("smc_score", 50.0),
                liquidity_score=daily_liq_res.get("liquidity_score", 50.0),
                mtf_score=mtf_res.get("mtf_score", 50.0),
                derivatives_score=deriv_res.get("derivatives_score", 50.0),
                fundamental_score=fund_res.get("fundamental_score", 50.0),
                sentiment_score=sentiment_score,
                session_score=session_score,                               # Session score
                fvg_mitigation_score=fvg_mitigation_score,                 # FVG mitigation score
                estimated_win_rate=0.65,
                rr_ratio=2.5,
                btc_regime_bullish=btc_context["btc_regime_bullish"],
                market_volatility_high=btc_context["market_volatility_high"],
                red_flags=red_flags
            )

            return {
                "symbol": symbol,
                "status": "SUCCESS",
                "current_price": current_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "qty": qty,
                "pos_val": pos_val,
                "atr_val": atr_val,
                "tech_res": tech_res,
                "h1_smc_res": h1_smc_res,
                "m30_smc_res": m30_smc_res,
                "daily_liq_res": daily_liq_res,
                "mtf_res": mtf_res,
                "deriv_res": deriv_res,
                "fund_res": fund_res,
                "fusion_res": fusion_res
            }

        except Exception as e:
            logging.exception(f"⚠️ Error processing asset {symbol} in worker thread")
            return {"symbol": symbol, "status": "ERROR", "message": str(e)}

    def scan_and_execute(self, max_universe_size: int = 15):
        target_coins = self.universe_engine.build_tradable_universe(
            min_volume_usdt=2_000_000,
            min_exchanges=2
        )[:max_universe_size]

        logging.info(f"🔎 Starting Parallel Multi-Layer Scan Cycle for {len(target_coins)} Assets...")

        df_btc_daily = self.fetcher.get_ohlcv("BTC/USDT", timeframe="1d", limit=100)
        btc_regime_bullish = True
        market_volatility_high = False

        if df_btc_daily is not None and not df_btc_daily.empty:
            btc_tech = self.tech_engine.analyze(df_btc_daily)
            btc_close = float(df_btc_daily['close'].iloc[-1])
            btc_ema_20 = float(df_btc_daily['close'].ewm(span=20, adjust=False).mean().iloc[-1])
            btc_rsi = btc_tech.get("rsi", 50.0)
            btc_regime_bullish = (btc_close > btc_ema_20) and (btc_rsi > 45.0)
            
            btc_atr = btc_tech.get("atr", btc_close * 0.02)
            market_volatility_high = (btc_atr / btc_close) > 0.035

        btc_context = {
            "btc_regime_bullish": btc_regime_bullish,
            "market_volatility_high": market_volatility_high
        }

        scanned_results = []

        max_workers = min(10, len(target_coins)) if target_coins else 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._process_single_asset, symbol, btc_context): symbol
                for symbol in target_coins
            }

            for future in as_completed(future_to_symbol):
                res = future.result()
                if res.get("status") == "SUCCESS":
                    scanned_results.append(res)

        scanned_results.sort(key=lambda x: x["fusion_res"]["unified_score"], reverse=True)

        logging.info(f"🏆 Parallel Scoring Complete. Evaluating {len(scanned_results)} assets for Gatekeeper & AI Debate...")

        for asset in scanned_results:
            symbol = asset["symbol"]
            fusion_res = asset["fusion_res"]

            if not fusion_res['is_passed']:
                continue

            try:
                payload = {
                    "symbol": symbol,
                    "price": asset["current_price"],
                    "gatekeeper_passed": fusion_res['is_passed'],
                    "unified_score": fusion_res['unified_score'],
                    "ev_r": fusion_res['ev_r'],
                    "technical": asset["tech_res"],
                    "smc": asset["h1_smc_res"],
                    "derivatives": asset["deriv_res"],
                    "market_microstructure": {
                        "cvd_score": asset["deriv_res"].get("cvd_score", 50.0),
                        "atr": asset["atr_val"]
                    }
                }

                ai_res = self.ai_engine.run_debate(payload)
                ai_confidence = int(ai_res.get('confidence', 0))
                ai_decision = ai_res.get('final_decision', 'HOLD')

                if (
                    fusion_res['unified_score'] >= (78.0 if market_volatility_high else 72.0) and
                    ai_confidence >= 70 and
                    ai_decision in ["EXECUTE_LONG", "EXECUTE_SHORT"]
                ):
                    logging.info(f"🚀 HIGH CONVICTION SIGNAL! Dispatching Alert for {symbol}...")
                    self.telegram.send_trade_signal(
                        symbol=symbol,
                        decision=ai_decision,
                        confidence=ai_confidence,
                        score=fusion_res['unified_score'],
                        ev_r=fusion_res['ev_r'],
                        entry=asset["current_price"],
                        sl=asset["sl_price"],
                        tp=asset["tp_price"],
                        summary=ai_res.get('summary', 'High conviction trade detected.')
                    )
                    self.db.save_trade_signal(payload=payload, ai_verdict=ai_res)

            except Exception:
                logging.exception(f"⚠️ Error processing asset {symbol} in worker thread")
                return {
                    "symbol": symbol,
                    "status": "ERROR"
                }

    def run_forever(self, scan_limit: int = 15, poll_interval_seconds: int = 300):
        while True:
            try:
                self.scan_and_execute(max_universe_size=scan_limit)
                time.sleep(poll_interval_seconds)
            except KeyboardInterrupt:
                break
            except Exception as e:
                time.sleep(15)

if __name__ == "__main__":
    orchestrator = QuantTradingOrchestrator()
    orchestrator.run_forever(scan_limit=15, poll_interval_seconds=300)
