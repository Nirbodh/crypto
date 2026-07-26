# python-engine/main_engine.py

import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from risk_engine import RiskEngine

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

            # ============================================================
            # 2. Unified Indicator Calculation (CVD, Session, FVG Mitigation)
            # ============================================================
            df_15m_calc = TechnicalIndicators.calculate_all(df_15m.copy())
            
            cvd_score = float(df_15m_calc['CVD_Score'].iloc[-1]) if 'CVD_Score' in df_15m_calc.columns else 50.0
            session_score = SessionIndicators.get_session_score(df_15m_calc)
            fvg_mitigation_score = float(df_15m_calc['FVG_Mitigation_Score'].iloc[-1]) if 'FVG_Mitigation_Score' in df_15m_calc.columns else 50.0

            # ============================================================
            # 3. Layer Calculations
            # ============================================================
            tech_res = self.tech_engine.analyze(df_15m)
            daily_liq_res = self.liq_engine.analyze_liquidity(df_daily)
            h4_smc_res    = self.smc_engine.analyze(df_4h)
            h1_smc_res    = self.smc_engine.analyze(df_1h)
            m30_smc_res   = self.smc_engine.analyze(df_30m)

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
            deriv_res["cvd_score"] = cvd_score
            
            fund_res = self.fund_engine.fetch_fundamental_data(symbol)
            fear_greed = fund_res.get("fear_and_greed_index", 50.0)
            sentiment_score = float(fear_greed) if isinstance(fear_greed, (int, float)) else 50.0

            # ============================================================
            # 4. Dynamic SL, TP & Position Sizing
            # ============================================================
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

            # ============================================================
            # 5. RISK ENGINE INTEGRATION
            # ============================================================
            # Determine direction (respecting market regime)
            h1_bias = h1_smc_res.get("bias", "NEUTRAL")
            tech_bullish = tech_res.get("momentum_bullish", False)
            market_regime = btc_context.get("market_regime", "TRENDING")
            
            # Preferred direction based on SMC/Technical
            if h1_bias == "BULLISH" or tech_bullish:
                preferred_direction = "LONG"
            elif h1_bias == "BEARISH" or (not tech_bullish and h1_bias != "NEUTRAL"):
                preferred_direction = "SHORT"
            else:
                preferred_direction = "LONG"

            # Adjust direction based on market regime
            if market_regime in ["BEAR", "CRASH"]:
                if preferred_direction == "LONG":
                    # Try SHORT if SMC bias is bearish or neutral with bearish tech
                    if h1_bias == "BEARISH" or (not tech_bullish and h1_bias != "BULLISH"):
                        direction = "SHORT"
                        logging.debug(f"🔄 {symbol}: Market regime {market_regime} - overriding LONG to SHORT")
                    else:
                        logging.info(f"⛔ {symbol}: Market regime {market_regime} and bias {h1_bias} - no safe direction")
                        return {"symbol": symbol, "status": "RISK_REJECTED", "reason": "No safe direction in BEAR/CRASH"}
                else:
                    direction = preferred_direction
            else:
                direction = preferred_direction

            # Calculate rolling ATR mean & std (for ATR Z-score)
            df_atr = df_15m.copy()
            atr_series = df_atr['high'] - df_atr['low']
            atr_series = atr_series.rolling(14).mean().fillna(atr_val)
            atr_rolling_mean = float(atr_series.iloc[-20:].mean()) if len(atr_series) >= 20 else atr_val
            atr_rolling_std = float(atr_series.iloc[-20:].std()) if len(atr_series) >= 20 else atr_val * 0.1

            risk_res = RiskEngine.calculate_trade_risk(
                entry_price=current_price,
                atr_5m=atr_val,
                custom_sl_price=sl_price,
                direction=direction,
                account_balance=account_balance,
                market_regime=market_regime,
                atr_rolling_mean=atr_rolling_mean,
                atr_rolling_std=atr_rolling_std
            )
            logging.debug(f"📊 RiskEngine result for {symbol}: {risk_res}")

            # Early reject if risk is invalid
            if not risk_res.get("valid_trade", False):
                invalidation_reasons = risk_res.get("invalidation_reasons", [])
                reason_str = ", ".join(invalidation_reasons) if invalidation_reasons else "Unknown reason"
                logging.info(f"⛔ RiskEngine rejected {symbol} - {reason_str} (RR: {risk_res.get('risk_metrics', {}).get('risk_reward_ratio', 'N/A')})")
                return {"symbol": symbol, "status": "RISK_REJECTED"}

            # ============================================================
            # 6. Unified Score Fusion Execution
            # ============================================================
            tech_flags = tech_res.get("red_flags", {})
            if isinstance(tech_flags, dict):
                tech_flags = (
                    tech_flags.get("critical", [])
                    + tech_flags.get("major", [])
                    + tech_flags.get("minor", [])
                )
            else:
                if not isinstance(tech_flags, list):
                    tech_flags = []
            
            fund_flags = fund_res.get("red_flags", [])
            if not isinstance(fund_flags, list):
                fund_flags = []
            
            red_flags = tech_flags + fund_flags

            adv = risk_res.get("advanced_metrics", {})
            risk_score = adv.get("risk_score", 50.0)
            safety_score = adv.get("safety_score", 50.0)
            position_quality_score = adv.get("position_quality_score", 50.0)

            rr_ratio_raw = risk_res.get("risk_metrics", {}).get("rr_score_raw", 2.0)
            effective_rr = max(1.0, float(rr_ratio_raw))

            # Dynamic Estimated Win Rate
            tech_score_val = tech_res.get("technical_score", 50.0)
            smc_score_val = h1_smc_res.get("smc_score", 50.0)
            mtf_score_val = mtf_res.get("mtf_score", 50.0)
            liq_score_val = daily_liq_res.get("liquidity_score", 50.0)
            
            composite_win_rate = (
                (tech_score_val / 100.0) * 0.40 +
                (smc_score_val / 100.0) * 0.30 +
                (mtf_score_val / 100.0) * 0.20 +
                (liq_score_val / 100.0) * 0.10
            )
            estimated_win_rate = 0.30 + (composite_win_rate * 0.45)
            estimated_win_rate = round(max(0.30, min(0.75, estimated_win_rate)), 3)

            fusion_res = InstitutionalScoreFusionEngine.fuse_scores(
                symbol=symbol,
                tech_score=tech_score_val,
                smc_score=smc_score_val,
                liquidity_score=liq_score_val,
                mtf_score=mtf_score_val,
                derivatives_score=deriv_res.get("derivatives_score", 50.0),
                fundamental_score=fund_res.get("fundamental_score", 50.0),
                sentiment_score=sentiment_score,
                session_score=session_score,
                fvg_mitigation_score=fvg_mitigation_score,
                risk_score=risk_score,
                safety_score=safety_score,
                position_quality_score=position_quality_score,
                effective_leverage=risk_res.get("effective_leverage", 1.0),
                capital_exposure_pct=risk_res.get("position_size_percent", 50.0),
                estimated_win_rate=estimated_win_rate,
                rr_ratio=effective_rr,
                btc_regime_bullish=btc_context["btc_regime_bullish"],
                market_volatility_high=btc_context["market_volatility_high"],
                red_flags=red_flags
            )

            fusion_res["risk_engine"] = risk_res

            # ---- FIX 1: Extract correct scores ----
            unified_score = fusion_res.get("final_unified_score", fusion_res.get("unified_score", 0.0))
            ev_r = fusion_res.get("net_ev_r", 0.0)

            logging.info(
                f"📊 {symbol} Component Scores: "
                f"Tech={tech_score_val:.1f}, "
                f"SMC={smc_score_val:.1f}, "
                f"Liq={liq_score_val:.1f}, "
                f"MTF={mtf_score_val:.1f}, "
                f"Deriv={deriv_res.get('derivatives_score', 50):.1f}, "
                f"Fund={fund_res.get('fundamental_score', 50):.1f}, "
                f"Risk={risk_score:.1f}, "
                f"WinRate={estimated_win_rate:.1%}, "
                f"RR={effective_rr:.2f}, "
                f"Final={unified_score:.1f}"
            )

            # ---- FIX 2: Return with unified_score and ev_r ----
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
                "risk_res": risk_res,
                "fusion_res": fusion_res,
                "unified_score": unified_score,  # ✅ NEW
                "ev_r": ev_r                     # ✅ NEW
            }

        except Exception as e:
            logging.exception(f"⚠️ Error processing asset {symbol} in worker thread")
            return {"symbol": symbol, "status": "ERROR", "message": str(e)}

    def scan_and_execute(self, max_universe_size: int = 30):
        target_coins = self.universe_engine.build_tradable_universe(
            min_volume_usdt=2_000_000,
            min_exchanges=2
        )[:max_universe_size]

        logging.info(f"🔎 Starting Parallel Multi-Layer Scan Cycle for {len(target_coins)} Assets...")

        df_btc_daily = self.fetcher.get_ohlcv("BTC/USDT", timeframe="1d", limit=100)
        btc_regime_bullish = True
        market_volatility_high = False
        market_regime = "TRENDING"

        if df_btc_daily is not None and not df_btc_daily.empty:
            btc_tech = self.tech_engine.analyze(df_btc_daily)
            btc_close = float(df_btc_daily['close'].iloc[-1])
            btc_ema_20 = float(df_btc_daily['close'].ewm(span=20, adjust=False).mean().iloc[-1])
            btc_rsi = btc_tech.get("rsi", 50.0)
            btc_regime_bullish = (btc_close > btc_ema_20) and (btc_rsi > 45.0)
            
            btc_atr = btc_tech.get("atr", btc_close * 0.02)
            market_volatility_high = (btc_atr / btc_close) > 0.035
            if btc_regime_bullish:
                market_regime = "TRENDING" if not market_volatility_high else "VOLATILE"
            else:
                market_regime = "BEAR" if not market_volatility_high else "CRASH"

        btc_context = {
            "btc_regime_bullish": btc_regime_bullish,
            "market_volatility_high": market_volatility_high,
            "market_regime": market_regime
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
                elif res.get("status") == "RISK_REJECTED":
                    reason = res.get("reason", "RR/risk criteria not met")
                    logging.debug(f"⛔ {res['symbol']} rejected by RiskEngine: {reason}")

        scanned_results.sort(key=lambda x: x.get("unified_score", 0), reverse=True)

        logging.info(f"🏆 Parallel Scoring Complete. Evaluating {len(scanned_results)} assets for Gatekeeper & AI Debate...")
        logging.info("📋 Step 1: Before Gatekeeper - Starting asset evaluation loop...")

        # ---- FIX 3: Lower threshold ----
        # Use 68 as base, adjust for volatility
        base_threshold = 55.0
        threshold = 60.0 if market_volatility_high else base_threshold

        for asset in scanned_results:
            symbol = asset["symbol"]
            fusion_res = asset["fusion_res"]
            unified_score = asset.get("unified_score", 0.0)
            logging.info(
                f"🔍 Checking {symbol} | Score={unified_score:.1f} | Passed={fusion_res.get('is_passed')}"
            )
            
            # ---- FIX 4: Use unified_score for gatekeeping ----
            if unified_score < threshold:
                logging.debug(f"⛔ Gatekeeper rejected {symbol} (Score: {unified_score:.1f} < {threshold:.1f})")
                continue
            
            # Also check is_passed flag (but if score is high enough, we can allow)
            if not fusion_res.get('is_passed', False) and unified_score < 70:
                logging.debug(f"⛔ Gatekeeper rejected {symbol} due to is_passed=False and score {unified_score:.1f} < 70")
                continue
            
            logging.info(f"✅ Step 2: After Gatekeeper - {symbol} PASSED (Score: {unified_score:.1f} >= {threshold:.1f})")

            try:
                payload = {
                    "symbol": symbol,
                    "price": asset["current_price"],
                    "gatekeeper_passed": True,
                    "unified_score": unified_score,
                    "ev_r": asset.get("ev_r", 0.0),
                    "technical": asset["tech_res"],
                    "smc": asset["h1_smc_res"],
                    "derivatives": asset["deriv_res"],
                    "risk": asset["risk_res"],
                    "market_microstructure": {
                        "cvd_score": asset["deriv_res"].get("cvd_score", 50.0),
                        "atr": asset["atr_val"]
                    }
                }

                logging.info(f"🧠 Step 3: Before Gemini AI Debate for {symbol}...")
                ai_res = self.ai_engine.run_debate(payload)
                ai_confidence = int(ai_res.get('confidence', 0))
                ai_decision = ai_res.get('final_decision', 'HOLD')
                logging.info(f"✅ Step 4: After Gemini AI Debate for {symbol} - Decision: {ai_decision}, Confidence: {ai_confidence}%")

                # ---- FIX 5: Final execution threshold ----
                exec_threshold = 72.0 if market_volatility_high else 68.0
                if (
                    unified_score >= exec_threshold and
                    ai_confidence >= 70 and
                    ai_decision in ["EXECUTE_LONG", "EXECUTE_SHORT"]
                ):
                    logging.info(f"🚀 HIGH CONVICTION SIGNAL for {symbol}! Dispatching...")
                    
                    logging.info(f"💾 Step 5: Before Database Save for {symbol}...")
                    self.db.save_trade_signal(payload=payload, ai_verdict=ai_res)
                    logging.info(f"✅ Database record saved for {symbol}.")

                    logging.info(f"📤 Step 6: Before Telegram Alert for {symbol}...")
                    self.telegram.send_trade_signal(
                        symbol=symbol,
                        decision=ai_decision,
                        confidence=ai_confidence,
                        score=unified_score,
                        ev_r=asset.get("ev_r", 0.0),
                        entry=asset["current_price"],
                        sl=asset["sl_price"],
                        tp=asset["tp_price"],
                        summary=ai_res.get('summary', 'High conviction trade detected.')
                    )
                    logging.info(f"✅ Telegram alert sent for {symbol}.")
                else:
                    logging.info(f"⏳ {symbol} did not meet final execution criteria. Score: {unified_score:.1f} >= {exec_threshold:.1f}? {unified_score >= exec_threshold}, Conf: {ai_confidence}")

                logging.info(f"✅ Step 7: Finished processing {symbol}")

            except Exception as e:
                logging.exception(f"⚠️ Error processing asset {symbol} in main loop")
                continue

    def run_forever(self, scan_limit: int = 30, poll_interval_seconds: int = 300):
        while True:
            try:
                self.scan_and_execute(max_universe_size=scan_limit)
                time.sleep(poll_interval_seconds)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.error(f"🔥 Critical error in main loop: {e}")
                time.sleep(15)

if __name__ == "__main__":
    orchestrator = QuantTradingOrchestrator()
    orchestrator.run_forever(scan_limit=30, poll_interval_seconds=300)
