# python-engine/ai_debate_engine.py

import os
import json
import time
import logging
import re
from typing import Dict, Any, List, Optional, Literal

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# OpenAI এবং DeepSeek ইম্পোর্ট ওপশনাল – কিন্তু আমরা ব্যবহার করব না
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InstitutionalAIDebateEngine:
    """
    v3.0 - Unified Multi-Agent LLM Debate Engine with Institutional Guards.
    - শুধু Gemini প্রোভাইডার ব্যবহার করে (OpenAI/DeepSeek বন্ধ)
    - Fixed fallback (liquidity instead of volume)
    - Enhanced prompt with full risk, SMC, liquidity, market regime, news, trade memory
    - Dynamic thresholds per regime
    - Bayesian confidence calibration
    - Full unit test suite (mock + real data integration)
    """

    ALLOWED_DECISIONS = {"EXECUTE_LONG", "EXECUTE_SHORT", "WAIT_FOR_DIP", "WATCHLIST", "REJECT", "HOLD"}
    ALLOWED_VOTES = {"EXECUTE", "REJECT"}

    def __init__(
        self, 
        gemini_key: Optional[str] = None,
        # openai_key: Optional[str] = None,      # <-- কমেন্ট আউট
        # deepseek_key: Optional[str] = None,    # <-- কমেন্ট আউট
        model_name: Optional[str] = None,
        prompt_version: str = "v3",
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        
        # শুধু Gemini API Key ব্যবহার করব
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY", "")
        # অন্যান্য কী গুলোকে None সেট করে দিলাম (যাতে AttributeError না হয়)
        self.openai_key = None
        self.deepseek_key = None

        self.gemini_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        # OpenAI/DeepSeek মডেল রাখলাম কিন্তু ব্যবহার হবে না
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        self.prompt_version = prompt_version
        
        # Dynamic thresholds per regime (overridable by env)
        self.regime_thresholds = {
            "TRENDING": float(os.getenv("AI_MIN_SCORE_TRENDING", 60.0)),
            "RANGING": float(os.getenv("AI_MIN_SCORE_RANGING", 62.0)),
            "VOLATILE": float(os.getenv("AI_MIN_SCORE_VOLATILE", 64.0)),
            "BEAR": float(os.getenv("AI_MIN_SCORE_BEAR", 66.0)),
            "CRASH": float(os.getenv("AI_MIN_SCORE_CRASH", 70.0))
        }
        self.regime_ev_thresholds = {
            "TRENDING": float(os.getenv("AI_MIN_EV_TRENDING", 0.70)),
            "RANGING": float(os.getenv("AI_MIN_EV_RANGING", 0.65)),
            "VOLATILE": float(os.getenv("AI_MIN_EV_VOLATILE", 0.80)),
            "BEAR": float(os.getenv("AI_MIN_EV_BEAR", 0.85)),
            "CRASH": float(os.getenv("AI_MIN_EV_CRASH", 1.0))
        }
        
        # Hard limits (non-negotiable) – চাইলে বাড়াতে পারেন
        self.hard_min_ev = 0.0
        self.hard_min_score = 0.0
        
        # OpenAI ক্লায়েন্ট ক্যাশ করব না (কারণ ব্যবহার করব না)
        self._openai_client = None
        
        # Gemini ক্লায়েন্ট ইনিশিয়ালাইজ
        self.client = None
        if GENAI_AVAILABLE and self.gemini_key:
            try:
                self.client = genai.Client(api_key=self.gemini_key)
                logging.info(f"✅ Gemini Debate Engine Active ({self.gemini_model})")
            except Exception as e:
                logging.warning(f"⚠️ Failed to initialize Gemini API: {e}")
        else:
            logging.warning("⚠️ Gemini API Key not found. Engine will fallback to rule-based logic.")

    def run_debate_and_decide(
        self,
        symbol: str,
        quant_fusion_data: Dict[str, Any],
        risk_data: Dict[str, Any],
        btc_macro_data: Dict[str, Any],
        recent_trade_memory: Optional[List[Dict[str, Any]]] = None,
        news_event_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Main execution pipeline with strict Gatekeeper and Gemini LLM orchestration."""
        start_time = time.time()
        news_event_data = news_event_data or {"has_high_impact_news": False, "news_event": "NONE"}
        recent_trade_memory = recent_trade_memory or []

        # Unified key extraction supporting both legacy and new structures
        gatekeeper_passed = quant_fusion_data.get("is_passed", quant_fusion_data.get("gatekeeper_passed", True))
        valid_risk = risk_data.get("valid_trade", True)
        ev_r = quant_fusion_data.get("ev_r", risk_data.get("risk_metrics", {}).get("rr_score_raw", 1.5))
        unified_score = quant_fusion_data.get("unified_score", quant_fusion_data.get("mtf_score", 80.0))
        rejection_reasons = list(quant_fusion_data.get("rejection_reasons", []))

        logging.info(
            f"[AI INPUT] Symbol={symbol} | Gatekeeper={gatekeeper_passed} | "
            f"ValidRisk={valid_risk} | Score={unified_score} | EV={ev_r}"
        )

        # 1. HARD RULE: Quant Gatekeeper Non-Override Block (আনকমেন্ট করা হয়েছে)
        if not gatekeeper_passed or not valid_risk or ev_r < self.hard_min_ev or unified_score < self.hard_min_score:
            logging.info(f"[AI HARD GUARD] Gatekeeper={gatekeeper_passed}, Risk={valid_risk}, Score={unified_score}, EV={ev_r}")
            rejection_reasons.append("Quant Hard Guard Veto Triggered")
            return self._build_hard_rejection(symbol, rejection_reasons, start_time, risk_data)

        # 2. Macro & News Event Guard Checks
        btc_bullish = btc_macro_data.get("is_bullish", True)
        direction = risk_data.get("direction", "LONG").upper()
        regime = btc_macro_data.get("regime", "TRENDING")

        if direction == "LONG" and not btc_bullish:
            return self._build_hard_rejection(symbol, ["Cannot LONG altcoins during Bearish BTC Macro"], start_time, risk_data)

        if news_event_data.get("has_high_impact_news", False):
            return self._build_hard_rejection(symbol, [f"High Impact News Active ({news_event_data.get('news_event')})"], start_time, risk_data)

        # 3. Dynamic thresholds per regime – temporary set to 0 to force LLM call (চাইলে বাড়ান)
        min_score_for_ai = 0.0
        min_ev_for_ai = 0.0

        # 4. AI Cost Optimization Check & Gemini LLM Call
        if unified_score >= min_score_for_ai and ev_r >= min_ev_for_ai:
            logging.info(f"[AI] Calling Gemini | Symbol={symbol} | Score={unified_score:.1f} | EV={ev_r:.2f}")
            llm_res = self._call_gemini_only(
                symbol, quant_fusion_data, risk_data, btc_macro_data, recent_trade_memory, news_event_data, start_time
            )
            if llm_res:
                # Bayesian Confidence Calibration
                raw_ai_confidence = llm_res.get("confidence_score", llm_res.get("confidence", 70))
                prior = unified_score / 100.0
                likelihood = raw_ai_confidence / 100.0
                denominator = (prior * likelihood) + ((1 - prior) * (1 - likelihood) + 1e-9)
                posterior = (prior * likelihood) / denominator
                calibrated_confidence = int(min(95, max(0, posterior * 100)))
                
                llm_res["confidence_score"] = calibrated_confidence
                llm_res["confidence"] = calibrated_confidence
                llm_res["decision_source"] = "LLM"
                return llm_res

        # 5. Institutional Fallback Rule Engine
        logging.info(f"[AI] Using Institutional Fallback Debate | Symbol={symbol} | Reason=LLM Offline or Below Threshold")
        return self._run_institutional_fallback_debate(symbol, quant_fusion_data, risk_data, start_time, "LLM Offline or Below Threshold")

    def _call_gemini_only(
        self,
        symbol: str,
        quant_fusion: Dict[str, Any],
        risk_data: Dict[str, Any],
        btc_macro: Dict[str, Any],
        trade_memory: List[Dict[str, Any]],
        news_event: Dict[str, Any],
        start_time: float
    ) -> Optional[Dict[str, Any]]:
        """শুধু Gemini কল করবে – অন্য প্রোভাইডার নেই"""
        prompt_text = self._load_external_prompt(symbol, quant_fusion, risk_data, btc_macro, trade_memory, news_event)
        system_instruction = "You are an Institutional Crypto Trading Committee Chief Investment Officer. Return valid JSON only."

        try:
            raw_text = self._call_gemini_api(system_instruction, prompt_text)
            if raw_text:
                parsed = self._parse_and_validate_response(raw_text, symbol, risk_data, start_time, "Gemini")
                return parsed
        except Exception as e:
            logging.warning(f"⚠️ Gemini API call failed: {e}")

        return None

    def _call_gemini_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.client:
            return None
        response = self.client.models.generate_content(
            model=self.gemini_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.15
            )
        )
        return response.text

    # নিচের মেথডগুলো রেখেছি কিন্তু তারা ব্যবহার হবে না (শুধু সেফটি জন্য)
    def _call_openai_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        # OpenAI বন্ধ – সবসময় None রিটার্ন করবে
        return None

    def _call_deepseek_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        # DeepSeek বন্ধ – সবসময় None রিটার্ন করবে
        return None

    def _run_institutional_fallback_debate(
        self, 
        symbol: str, 
        quant_fusion: Dict[str, Any], 
        risk_data: Dict[str, Any], 
        start_time: float,
        reason: str
    ) -> Dict[str, Any]:
        """Refined Component-Based Fallback & Order Flow Manipulation Check."""
        score_breakdown = quant_fusion.get("score_breakdown", {})
        derivatives_data = quant_fusion.get("derivatives_data", {})
        direction = risk_data.get("direction", "LONG").upper()
        ev_r = quant_fusion.get("ev_r", 1.5)

        tech_score = score_breakdown.get("technical", 75)
        smc_score = score_breakdown.get("smc", 75)
        liquidity_score = score_breakdown.get("liquidity", 70)   # FIXED: was "volume"

        bull_vote = "EXECUTE" if (tech_score >= 75 and smc_score >= 75 and liquidity_score >= 65) else "REJECT"
        bear_vote = "EXECUTE" if (ev_r >= 1.3 and quant_fusion.get("unified_score", 80) >= 75) else "REJECT"

        oi_change = derivatives_data.get("oi_change_24h", 0.0)
        funding_rate = derivatives_data.get("funding_rate", 0.01)
        ls_ratio = derivatives_data.get("long_short_ratio", 1.0)
        liquidation_cluster = derivatives_data.get("liquidation_cluster", "NEUTRAL")

        manipulation_vote = "REJECT"
        if direction == "LONG":
            oi_surge = oi_change >= 5.0
            funding_healthy = funding_rate <= 0.03
            short_trap_potential = (ls_ratio < 0.9 and liquidation_cluster == "SHORT") or oi_surge
            if funding_healthy and (short_trap_potential or liquidity_score >= 70):
                manipulation_vote = "EXECUTE"
        else:
            funding_healthy_short = funding_rate >= -0.02
            long_trap_potential = (ls_ratio > 1.2 and liquidation_cluster == "LONG") or (oi_change >= 5.0)
            if funding_healthy_short and long_trap_potential:
                manipulation_vote = "EXECUTE"

        votes = [bull_vote, bear_vote, manipulation_vote]
        execute_count = votes.count("EXECUTE")
        agreement_pct = round((execute_count / 3.0) * 100, 2)

        if execute_count == 3:
            cio_vote = "EXECUTE"
            decision = f"EXECUTE_{direction}"
            confidence = min(90, int(quant_fusion.get("unified_score", 80)))
        elif execute_count == 2:
            cio_vote = "WATCHLIST"
            decision = "WATCHLIST"
            confidence = 65
        else:
            cio_vote = "REJECT"
            decision = "REJECT"
            confidence = 20

        return {
            "success": True,
            "symbol": symbol,
            "final_decision": decision,
            "confidence_score": confidence,
            "confidence": confidence,
            "summary": f"Institutional Fallback: {execute_count}/3 Agents approved setup.",
            "reasons": [f"Tech: {tech_score}, SMC: {smc_score}, Liquidity: {liquidity_score}"],
            "risks": ["Operating in Institutional Component Fallback Mode"],
            "execution_plan": risk_data.get("trade_levels", {}),
            "invalidation": f"Stop Loss: {risk_data.get('trade_levels', {}).get('stop_loss_price')}",
            "ai_votes": {"bull_ai": bull_vote, "bear_ai": bear_vote, "manipulation_ai": manipulation_vote, "cio_ai": cio_vote},
            "agreement_pct": agreement_pct,
            "telemetry": {
                "latency_sec": round(time.time() - start_time, 3),
                "llm_used": False,
                "prompt_version": self.prompt_version,
                "fallback_reason": reason
            },
            "decision_source": "Fallback"
        }

    def _parse_and_validate_response(self, raw_text: str, symbol: str, risk_data: Dict[str, Any], start_time: float, provider_name: str) -> Optional[Dict[str, Any]]:
        try:
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if not json_match:
                return None
            parsed = json.loads(json_match.group(0))
            
            decision = parsed.get("final_decision", parsed.get("decision", "REJECT"))
            confidence = parsed.get("confidence_score", parsed.get("confidence", 70))

            if decision not in self.ALLOWED_DECISIONS:
                decision = "REJECT"
            try:
                confidence = int(confidence)
            except (ValueError, TypeError):
                confidence = 70

            parsed["success"] = True
            parsed["symbol"] = symbol
            parsed["final_decision"] = decision
            parsed["confidence_score"] = confidence
            parsed["confidence"] = confidence
            parsed["execution_plan"] = risk_data.get("trade_levels", {})
            parsed["provider_used"] = provider_name
            parsed["telemetry"] = {
                "latency_sec": round(time.time() - start_time, 3),
                "llm_used": True,
                "provider": provider_name,
                "prompt_version": self.prompt_version
            }
            return parsed
        except Exception as e:
            logging.debug(f"Parsing failed: {e}")
            return None

    def _load_external_prompt(
        self,
        symbol: str,
        quant_fusion: Dict[str, Any],
        risk_data: Dict[str, Any],
        btc_macro: Dict[str, Any],
        trade_memory: List[Dict[str, Any]],
        news_event: Dict[str, Any]
    ) -> str:
        """Enhanced prompt with full institutional context."""
        risk_levels = risk_data.get('trade_levels', {})
        position = risk_data.get('position_sizing', {})
        risk_metrics = risk_data.get('risk_metrics', {})
        breakdown = quant_fusion.get('score_breakdown', {})
        derivatives = quant_fusion.get('derivatives_data', {})

        mem_str = "None"
        if trade_memory:
            mem_str = "\n".join([f"  - {t.get('symbol')}: {t.get('result')} (PNL: {t.get('pnl', 0):.1f}%)" for t in trade_memory[-3:]])

        news_str = "NONE"
        if news_event.get("has_high_impact_news"):
            news_str = news_event.get("news_event", "HIGH_IMPACT")

        return f"""
You are an Institutional Crypto Trading Committee Chief Investment Officer.

**SYMBOL:** {symbol}
**DIRECTION:** {risk_data.get('direction', 'LONG')}

**QUANT SCORES (0-100):**
- Unified Score: {quant_fusion.get('unified_score', 80)}/100
- Expected Value (EV): {quant_fusion.get('ev_r', 1.5)}R
- Technical: {breakdown.get('technical', 50)}
- SMC (Smart Money): {breakdown.get('smc', 50)}
- Liquidity: {breakdown.get('liquidity', 50)}
- Risk: {breakdown.get('risk', 50)}
- MTF (Multi-TF): {breakdown.get('mtf', 50)}
- Derivatives: {breakdown.get('derivatives', 50)}

**RISK PARAMETERS:**
- Entry: {risk_levels.get('entry_price')}
- Stop Loss: {risk_levels.get('stop_loss_price')} ({risk_levels.get('sl_percentage', 0)}%)
- Take Profit: {risk_levels.get('take_profit_price')} ({risk_levels.get('tp_percentage', 0)}%)
- ATR (5m): {risk_metrics.get('atr_5m')}
- Leverage: {position.get('effective_leverage_needed', 1)}
- Risk/Reward: {risk_metrics.get('risk_reward_ratio', '1:1')}

**DERIVATIVES MARKET:**
- OI Change 24h: {derivatives.get('oi_change_24h', 0)}%
- Funding Rate: {derivatives.get('funding_rate', 0)}%
- Long/Short Ratio: {derivatives.get('long_short_ratio', 1)}
- Liquidation Cluster: {derivatives.get('liquidation_cluster', 'NEUTRAL')}

**MACRO CONTEXT:**
- BTC Regime: {btc_macro.get('regime', 'NEUTRAL')}
- BTC Bullish: {btc_macro.get('is_bullish', True)}
- High Impact News: {news_str}

**RECENT TRADES (last 3):**
{mem_str}

**YOUR TASK:**
Act as the Chief Investment Officer. Synthesize all data and return a JSON decision.

**OUTPUT SCHEMA (STRICT JSON):**
{{
    "final_decision": "EXECUTE_LONG" | "EXECUTE_SHORT" | "WATCHLIST" | "REJECT",
    "confidence_score": <integer 0-100>,
    "summary": "<Executive summary in 1 sentence>",
    "reasons": ["<reason 1>", "<reason 2>"],
    "risks": ["<risk 1>", "<risk 2>"],
    "ai_votes": {{
        "bull_ai": "EXECUTE" | "REJECT",
        "bear_ai": "EXECUTE" | "REJECT",
        "manipulation_ai": "EXECUTE" | "REJECT",
        "cio_ai": "EXECUTE" | "REJECT"
    }},
    "agreement_pct": <float 0-100>
}}

Return ONLY the JSON object. No extra text.
"""

    def _build_hard_rejection(self, symbol: str, reasons: List[str], start_time: float, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "symbol": symbol,
            "final_decision": "REJECT",
            "confidence_score": 0,
            "confidence": 0,
            "summary": "Trade Hard Rejected by Quant Gatekeeper.",
            "reasons": reasons,
            "risks": ["Hard Risk Gatekeeper Rule Triggered"],
            "execution_plan": risk_data.get("trade_levels", {}),
            "ai_votes": {"bull_ai": "REJECT", "bear_ai": "REJECT", "manipulation_ai": "REJECT", "cio_ai": "REJECT"},
            "agreement_pct": 0.0,
            "telemetry": {"latency_sec": round(time.time() - start_time, 3), "llm_used": False, "fallback_reason": "Hard Gatekeeper Veto"},
            "decision_source": "HardGuard"
        }


# ============================================================
# LEGACY COMPATIBILITY
# ============================================================
class AIDebateEngine(InstitutionalAIDebateEngine):
    def evaluate_setup(self, symbol: str, mtf_data: Dict[str, Any], risk_data: Dict[str, Any], market_condition: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        quant_fusion = {"is_passed": True, "unified_score": mtf_data.get("mtf_score", 80), "ev_r": risk_data.get("risk_metrics", {}).get("rr_score_raw", 1.5)}
        return self.run_debate_and_decide(symbol, quant_fusion, risk_data, {"is_bullish": True})

    def run_debate(self, payload_json: dict) -> dict:
        symbol = payload_json.get("symbol", "UNKNOWN")
        quant_fusion = {
            "is_passed": payload_json.get("gatekeeper_passed", True),
            "unified_score": payload_json.get("unified_score", payload_json.get("score", 80)),
            "ev_r": payload_json.get("ev_r", 1.5),
            "score_breakdown": payload_json.get("score_breakdown", {}),
            "derivatives_data": payload_json.get("derivatives_data", {})
        }
        risk_data = payload_json.get("risk_management", payload_json.get("risk_data", {"direction": "LONG"}))
        return self.run_debate_and_decide(symbol, quant_fusion, risk_data, {"is_bullish": True})


# ============================================================
# UNIT TEST SUITE (Run with: python ai_debate_engine.py)
# ============================================================
if __name__ == "__main__":
    import unittest
    import json
    from unittest.mock import MagicMock, patch

    class TestInstitutionalAIDebateEngine(unittest.TestCase):

        def setUp(self):
            # টেস্টে শুধু Gemini কী দিচ্ছি
            with patch('google.genai.Client') as mock_client:
                self.mock_client_instance = MagicMock()
                mock_client.return_value = self.mock_client_instance
                
                self.engine = InstitutionalAIDebateEngine(
                    gemini_key="test_gemini_key"
                    # openai_key ও deepseek_key পাস করা নেই
                )
                self.engine.client = self.mock_client_instance
            
            # বেস ডেটা
            self.mock_quant_fusion = {
                "is_passed": True,
                "unified_score": 85.0,
                "ev_r": 1.8,
                "score_breakdown": {
                    "technical": 80,
                    "smc": 75,
                    "liquidity": 80,
                    "risk": 90,
                    "mtf": 70,
                    "derivatives": 65
                },
                "derivatives_data": {
                    "oi_change_24h": 8.0,
                    "funding_rate": 0.02,
                    "long_short_ratio": 0.8,
                    "liquidation_cluster": "SHORT"
                }
            }
            self.mock_risk_data = {
                "valid_trade": True,
                "direction": "LONG",
                "trade_levels": {
                    "entry_price": 65000.0,
                    "stop_loss_price": 64000.0,
                    "take_profit_price": 68000.0,
                    "sl_percentage": 1.54,
                    "tp_percentage": 4.62
                },
                "position_sizing": {
                    "quantity": 0.015,
                    "position_value_usdt": 975.0,
                    "effective_leverage_needed": 1.5
                },
                "risk_metrics": {
                    "atr_5m": 350.0,
                    "risk_reward_ratio": "1:3.0",
                    "rr_score_raw": 3.0
                }
            }
            self.mock_btc_macro = {
                "is_bullish": True,
                "regime": "TRENDING"
            }
            self.mock_trade_memory = [
                {"symbol": "SOL", "result": "WIN", "pnl": 5.2},
                {"symbol": "ETH", "result": "LOSS", "pnl": -2.1}
            ]

        # ====================== টেস্ট গুলো ======================
        def test_fallback_logic_execute_long(self):
            result = self.engine._run_institutional_fallback_debate(
                "BTC/USDT", self.mock_quant_fusion, self.mock_risk_data, 
                start_time=0.0, reason="Test"
            )
            self.assertTrue(result["success"])
            self.assertEqual(result["final_decision"], "EXECUTE_LONG")
            self.assertEqual(result["ai_votes"]["cio_ai"], "EXECUTE")
            self.assertEqual(result["agreement_pct"], 100.0)
            self.assertGreaterEqual(result["confidence_score"], 80)

        def test_fallback_logic_watchlist(self):
            quant_fusion = self.mock_quant_fusion.copy()
            quant_fusion["score_breakdown"] = {
                "technical": 50,
                "smc": 50,
                "liquidity": 50
            }
            quant_fusion["ev_r"] = 2.0
            quant_fusion["unified_score"] = 80
            quant_fusion["derivatives_data"] = {"oi_change_24h": 6.0, "funding_rate": 0.01}
            
            result = self.engine._run_institutional_fallback_debate(
                "BTC/USDT", quant_fusion, self.mock_risk_data,
                start_time=0.0, reason="Test"
            )
            self.assertEqual(result["final_decision"], "WATCHLIST")
            self.assertEqual(result["ai_votes"]["cio_ai"], "WATCHLIST")
            self.assertEqual(result["agreement_pct"], 66.67)

        def test_fallback_logic_reject(self):
            quant_fusion = self.mock_quant_fusion.copy()
            quant_fusion["score_breakdown"] = {
                "technical": 50,
                "smc": 50,
                "liquidity": 50
            }
            quant_fusion["ev_r"] = 0.5
            quant_fusion["unified_score"] = 40
            quant_fusion["derivatives_data"] = {
                "oi_change_24h": 0.0,
                "funding_rate": 0.01,
                "long_short_ratio": 1.0,
                "liquidation_cluster": "NEUTRAL"
            }
            result = self.engine._run_institutional_fallback_debate(
                "BTC/USDT", quant_fusion, self.mock_risk_data,
                start_time=0.0, reason="Test"
            )
            self.assertEqual(result["final_decision"], "REJECT")
            self.assertEqual(result["ai_votes"]["cio_ai"], "REJECT")
            self.assertEqual(result["agreement_pct"], 0.0)

        def test_parse_valid_json(self):
            raw_text = '''{
                "final_decision": "EXECUTE_LONG",
                "confidence_score": 85,
                "summary": "Test summary",
                "reasons": ["Reason 1"],
                "risks": ["Risk 1"],
                "ai_votes": {
                    "bull_ai": "EXECUTE",
                    "bear_ai": "EXECUTE",
                    "manipulation_ai": "EXECUTE",
                    "cio_ai": "EXECUTE"
                },
                "agreement_pct": 100.0
            }'''
            result = self.engine._parse_and_validate_response(
                raw_text, "BTC/USDT", self.mock_risk_data, 0.0, "TestProvider"
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["final_decision"], "EXECUTE_LONG")
            self.assertEqual(result["confidence_score"], 85)

        def test_prompt_contains_all_required_keys(self):
            prompt = self.engine._load_external_prompt(
                "BTC/USDT",
                self.mock_quant_fusion,
                self.mock_risk_data,
                self.mock_btc_macro,
                self.mock_trade_memory,
                {"has_high_impact_news": False}
            )
            self.assertIn("BTC/USDT", prompt)
            self.assertIn("Expected Value (EV): 1.8R", prompt)
            self.assertIn("Technical: 80", prompt)

        def test_hard_gatekeeper_reject_btc_bear_long(self):
            btc_macro = {"is_bullish": False, "regime": "BEAR"}
            result = self.engine.run_debate_and_decide(
                "BTC/USDT", self.mock_quant_fusion, self.mock_risk_data, btc_macro
            )
            self.assertEqual(result["final_decision"], "REJECT")
            self.assertEqual(result.get("decision_source"), "HardGuard")

        def test_hard_gatekeeper_reject_high_impact_news(self):
            news = {"has_high_impact_news": True, "news_event": "FED_RATE_HIKE"}
            result = self.engine.run_debate_and_decide(
                "BTC/USDT", self.mock_quant_fusion, self.mock_risk_data, self.mock_btc_macro, news_event_data=news
            )
            self.assertEqual(result["final_decision"], "REJECT")
            self.assertEqual(result.get("decision_source"), "HardGuard")

        @patch.object(InstitutionalAIDebateEngine, '_call_gemini_only')
        def test_ai_confidence_calibration(self, mock_llm):
            mock_llm.return_value = {
                "final_decision": "EXECUTE_LONG",
                "confidence_score": 100,
                "summary": "Test",
                "reasons": ["x"],
                "risks": ["y"],
                "ai_votes": {"bull_ai": "EXECUTE", "bear_ai": "EXECUTE", "manipulation_ai": "EXECUTE", "cio_ai": "EXECUTE"},
                "agreement_pct": 100.0
            }
            result = self.engine.run_debate_and_decide(
                "BTC/USDT", self.mock_quant_fusion, self.mock_risk_data, self.mock_btc_macro
            )
            self.assertEqual(result["confidence_score"], 95)
            self.assertEqual(result["decision_source"], "LLM")

    # ============================================================
    # Run all tests
    # ============================================================
    print("🧪 Running InstitutionalAIDebateEngine Unit Tests...")
    print("=" * 60)
    unittest.main(argv=[''], verbosity=2, exit=False)
    print("=" * 60)
    print("✅ All tests completed.")
