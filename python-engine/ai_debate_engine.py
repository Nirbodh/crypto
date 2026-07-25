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

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InstitutionalAIDebateEngine:
    """
    v3.0 - Unified Multi-Agent LLM Debate Engine with Multi-Provider Failover & Institutional Guards.
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
        openai_key: Optional[str] = None,
        deepseek_key: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt_version: str = "v3",
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        
        # API Keys & Models for Multi-Provider Failover
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY", "")
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY", "")
        self.deepseek_key = deepseek_key or os.getenv("DEEPSEEK_API_KEY", "")

        self.gemini_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        self.prompt_version = prompt_version
        
        # Dynamic thresholds per regime (overridable by env)
        self.regime_thresholds = {
            "TRENDING": float(os.getenv("AI_MIN_SCORE_TRENDING", 78.0)),
            "RANGING": float(os.getenv("AI_MIN_SCORE_RANGING", 75.0)),
            "VOLATILE": float(os.getenv("AI_MIN_SCORE_VOLATILE", 82.0)),
            "BEAR": float(os.getenv("AI_MIN_SCORE_BEAR", 80.0)),
            "CRASH": float(os.getenv("AI_MIN_SCORE_CRASH", 85.0))
        }
        self.regime_ev_thresholds = {
            "TRENDING": float(os.getenv("AI_MIN_EV_TRENDING", 1.3)),
            "RANGING": float(os.getenv("AI_MIN_EV_RANGING", 1.2)),
            "VOLATILE": float(os.getenv("AI_MIN_EV_VOLATILE", 1.5)),
            "BEAR": float(os.getenv("AI_MIN_EV_BEAR", 1.4)),
            "CRASH": float(os.getenv("AI_MIN_EV_CRASH", 1.6))
        }
        
        # Hard limits (non-negotiable)
        self.hard_min_ev = float(os.getenv("HARD_MIN_EV", 1.2))
        self.hard_min_score = float(os.getenv("HARD_MIN_SCORE", 75.0))
        
        # Cache OpenAI client if available
        self._openai_client = None
        if OPENAI_AVAILABLE and self.openai_key:
            try:
                self._openai_client = OpenAI(api_key=self.openai_key)
            except Exception:
                pass
        
        self.client = None
        if GENAI_AVAILABLE and self.gemini_key:
            try:
                self.client = genai.Client(api_key=self.gemini_key)
                logging.info(f"✅ Gemini Debate Engine Active ({self.gemini_model})")
            except Exception as e:
                logging.warning(f"⚠️ Failed to initialize Gemini API: {e}")
        else:
            logging.warning("⚠️ Operating with Fallback or Multi-Provider Support Mode.")

    def run_debate_and_decide(
        self,
        symbol: str,
        quant_fusion_data: Dict[str, Any],
        risk_data: Dict[str, Any],
        btc_macro_data: Dict[str, Any],
        recent_trade_memory: Optional[List[Dict[str, Any]]] = None,
        news_event_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Main execution pipeline with strict Gatekeeper and Multi-Provider LLM orchestration."""
        start_time = time.time()
        news_event_data = news_event_data or {"has_high_impact_news": False, "news_event": "NONE"}
        recent_trade_memory = recent_trade_memory or []

        # Unified key extraction supporting both legacy and new structures
        gatekeeper_passed = quant_fusion_data.get("is_passed", quant_fusion_data.get("gatekeeper_passed", True))
        valid_risk = risk_data.get("valid_trade", True)
        ev_r = quant_fusion_data.get("ev_r", risk_data.get("risk_metrics", {}).get("rr_score_raw", 1.5))
        unified_score = quant_fusion_data.get("unified_score", quant_fusion_data.get("mtf_score", 80.0))
        rejection_reasons = list(quant_fusion_data.get("rejection_reasons", []))

        # 1. HARD RULE: Quant Gatekeeper Non-Override Block
        if not gatekeeper_passed or not valid_risk or ev_r < self.hard_min_ev or unified_score < self.hard_min_score:
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

        # 3. Dynamic thresholds per regime
        min_score_for_ai = self.regime_thresholds.get(regime, 78.0)
        min_ev_for_ai = self.regime_ev_thresholds.get(regime, 1.3)

        # 4. AI Cost Optimization Check & Multi-Provider LLM Call
        if unified_score >= min_score_for_ai and ev_r >= min_ev_for_ai:
            llm_res = self._call_multi_provider_llm(
                symbol, quant_fusion_data, risk_data, btc_macro_data, recent_trade_memory, news_event_data, start_time
            )
            if llm_res:
                # Bayesian Confidence Calibration
                raw_ai_confidence = llm_res.get("confidence_score", llm_res.get("confidence", 70))
                # Prior = quant score (0-1), Likelihood = AI confidence (0-1)
                prior = unified_score / 100.0
                likelihood = raw_ai_confidence / 100.0
                # Avoid division by zero
                denominator = (prior * likelihood) + ((1 - prior) * (1 - likelihood) + 1e-9)
                posterior = (prior * likelihood) / denominator
                calibrated_confidence = int(min(95, max(0, posterior * 100)))
                
                llm_res["confidence_score"] = calibrated_confidence
                llm_res["confidence"] = calibrated_confidence
                return llm_res

        # 5. Institutional Fallback Rule Engine
        logging.info(f"⚡ Executing Component-Based Fallback Debate for {symbol}")
        return self._run_institutional_fallback_debate(symbol, quant_fusion_data, risk_data, start_time, "LLM Offline or Below Threshold")

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
            }
        }

    def _call_multi_provider_llm(
        self,
        symbol: str,
        quant_fusion: Dict[str, Any],
        risk_data: Dict[str, Any],
        btc_macro: Dict[str, Any],
        trade_memory: List[Dict[str, Any]],
        news_event: Dict[str, Any],
        start_time: float
    ) -> Optional[Dict[str, Any]]:
        """Orchestrates multi-provider failover: Gemini -> OpenAI -> DeepSeek."""
        prompt_text = self._load_external_prompt(symbol, quant_fusion, risk_data, btc_macro, trade_memory, news_event)
        system_instruction = "You are an Institutional Crypto Trading Committee Chief Investment Officer. Return valid JSON only."

        providers = [
            ("Gemini", lambda: self._call_gemini_api(system_instruction, prompt_text)),
            ("OpenAI", lambda: self._call_openai_api(system_instruction, prompt_text)),
            ("DeepSeek", lambda: self._call_deepseek_api(system_instruction, prompt_text))
        ]

        for name, provider_func in providers:
            try:
                raw_text = provider_func()
                if raw_text:
                    parsed = self._parse_and_validate_response(raw_text, symbol, risk_data, start_time, name)
                    if parsed:
                        return parsed
            except Exception as e:
                logging.warning(f"⚠️ Provider [{name}] failed: {e}")

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

    def _call_openai_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not OPENAI_AVAILABLE or not self.openai_key:
            return None
        # Use cached client if available
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.openai_key)
        response = self._openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.15,
            max_tokens=1000
        )
        return response.choices[0].message.content

    def _call_deepseek_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.deepseek_key:
            return None
        headers = {"Authorization": f"Bearer {self.deepseek_key}", "Content-Type": "application/json"}
        data = {
            "model": self.deepseek_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.15,
            "max_tokens": 1000
        }
        res = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data, timeout=20)
        if res.status_code == 200:
            return res.json()['choices'][0]['message']['content']
        return None

    def _parse_and_validate_response(self, raw_text: str, symbol: str, risk_data: Dict[str, Any], start_time: float, provider_name: str) -> Optional[Dict[str, Any]]:
        try:
            # Extract JSON from markdown or plain text
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if not json_match:
                return None
            parsed = json.loads(json_match.group(0))
            
            decision = parsed.get("final_decision", parsed.get("decision", "REJECT"))
            confidence = parsed.get("confidence_score", parsed.get("confidence", 70))

            # Validate and default
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

        # Format trade memory (last 3 trades)
        mem_str = "None"
        if trade_memory:
            mem_str = "\n".join([f"  - {t.get('symbol')}: {t.get('result')} (PNL: {t.get('pnl', 0):.1f}%)" for t in trade_memory[-3:]])

        # News
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
            "telemetry": {"latency_sec": round(time.time() - start_time, 3), "llm_used": False, "fallback_reason": "Hard Gatekeeper Veto"}
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
            """Initialize engine with test keys and mock external clients."""
            # Mock the Gemini client to prevent real API calls
            with patch('google.genai.Client') as mock_client:
                self.mock_client_instance = MagicMock()
                mock_client.return_value = self.mock_client_instance
                
                self.engine = InstitutionalAIDebateEngine(
                    gemini_key="test_gemini_key",
                    openai_key="test_openai_key",
                    deepseek_key="test_deepseek_key"
                )
                # Replace the actual client with mock
                self.engine.client = self.mock_client_instance
            
            # Base test data
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

        # ============================================================
        # 1. FALLBACK LOGIC TESTS
        # ============================================================
        
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
            # Ensure manipulation vote is REJECT
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

        def test_fallback_logic_missing_liquidity_key(self):
            quant_fusion = self.mock_quant_fusion.copy()
            quant_fusion["score_breakdown"] = {
                "technical": 80,
                "smc": 80
                # 'liquidity' key missing
            }
            result = self.engine._run_institutional_fallback_debate(
                "BTC/USDT", quant_fusion, self.mock_risk_data,
                start_time=0.0, reason="Test"
            )
            # Should use default 70 for liquidity, so bull_vote = EXECUTE
            self.assertEqual(result["final_decision"], "EXECUTE_LONG")

        def test_fallback_logic_short_direction(self):
            risk_data = self.mock_risk_data.copy()
            risk_data["direction"] = "SHORT"
            result = self.engine._run_institutional_fallback_debate(
                "BTC/USDT", self.mock_quant_fusion, risk_data,
                start_time=0.0, reason="Test"
            )
            self.assertEqual(result["final_decision"], "EXECUTE_SHORT")

        def test_fallback_logic_manipulation_long_trap(self):
            quant_fusion = self.mock_quant_fusion.copy()
            quant_fusion["derivatives_data"] = {
                "oi_change_24h": 6.0,
                "funding_rate": 0.01,
                "long_short_ratio": 1.5,
                "liquidation_cluster": "LONG"
            }
            risk_data = self.mock_risk_data.copy()
            risk_data["direction"] = "SHORT"
            result = self.engine._run_institutional_fallback_debate(
                "BTC/USDT", quant_fusion, risk_data,
                start_time=0.0, reason="Test"
            )
            self.assertEqual(result["ai_votes"]["manipulation_ai"], "EXECUTE")
            self.assertEqual(result["final_decision"], "EXECUTE_SHORT")

        # ============================================================
        # 2. LLM PARSING TESTS
        # ============================================================
        
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
            self.assertEqual(result["provider_used"], "TestProvider")

        def test_parse_json_with_markdown(self):
            raw_text = '''```json
            {
                "final_decision": "WATCHLIST",
                "confidence_score": 70,
                "summary": "Test",
                "reasons": ["x"],
                "risks": ["y"],
                "ai_votes": {"bull_ai": "EXECUTE", "bear_ai": "REJECT", "manipulation_ai": "EXECUTE", "cio_ai": "WATCHLIST"},
                "agreement_pct": 66.0
            }
            ```'''
            result = self.engine._parse_and_validate_response(
                raw_text, "BTC/USDT", self.mock_risk_data, 0.0, "TestProvider"
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["final_decision"], "WATCHLIST")

        def test_parse_json_with_extra_text(self):
            raw_text = '''Here is the analysis:
            {
                "final_decision": "REJECT",
                "confidence_score": 20,
                "summary": "Too risky",
                "reasons": ["Risk is high"],
                "risks": ["Liquidation"],
                "ai_votes": {"bull_ai": "REJECT", "bear_ai": "REJECT", "manipulation_ai": "REJECT", "cio_ai": "REJECT"},
                "agreement_pct": 0.0
            }
            Final decision.'''
            result = self.engine._parse_and_validate_response(
                raw_text, "BTC/USDT", self.mock_risk_data, 0.0, "TestProvider"
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["final_decision"], "REJECT")

        def test_parse_malformed_json(self):
            raw_text = '''{
                "final_decision": "EXECUTE_LONG",
                "confidence_score": 85,
                "summary": "Test",
                "reasons": ["x"],
                "risks": ["y"],
                "ai_votes": {"bull_ai": "EXECUTE", "bear_ai": "EXECUTE", "manipulation_ai": "EXECUTE", "cio_ai": "EXECUTE"},
                "agreement_pct": 100.0
            '''
            result = self.engine._parse_and_validate_response(
                raw_text, "BTC/USDT", self.mock_risk_data, 0.0, "TestProvider"
            )
            self.assertIsNone(result)

        def test_parse_invalid_decision(self):
            raw_text = '''{
                "final_decision": "BUY",
                "confidence_score": 90,
                "summary": "Test",
                "reasons": ["x"],
                "risks": ["y"],
                "ai_votes": {"bull_ai": "EXECUTE", "bear_ai": "EXECUTE", "manipulation_ai": "EXECUTE", "cio_ai": "EXECUTE"},
                "agreement_pct": 100.0
            }'''
            result = self.engine._parse_and_validate_response(
                raw_text, "BTC/USDT", self.mock_risk_data, 0.0, "TestProvider"
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["final_decision"], "REJECT")

        def test_parse_missing_fields(self):
            raw_text = '''{
                "confidence_score": 80,
                "summary": "Test",
                "reasons": ["x"],
                "risks": ["y"]
            }'''
            result = self.engine._parse_and_validate_response(
                raw_text, "BTC/USDT", self.mock_risk_data, 0.0, "TestProvider"
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["final_decision"], "REJECT")
            self.assertEqual(result["confidence_score"], 80)

        def test_parse_confidence_as_string(self):
            raw_text = '''{
                "final_decision": "EXECUTE_LONG",
                "confidence_score": "85",
                "summary": "Test",
                "reasons": ["x"],
                "risks": ["y"],
                "ai_votes": {"bull_ai": "EXECUTE", "bear_ai": "EXECUTE", "manipulation_ai": "EXECUTE", "cio_ai": "EXECUTE"},
                "agreement_pct": 100.0
            }'''
            result = self.engine._parse_and_validate_response(
                raw_text, "BTC/USDT", self.mock_risk_data, 0.0, "TestProvider"
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["confidence_score"], 85)

        # ============================================================
        # 3. PROMPT GENERATION TESTS (FIXED)
        # ============================================================
        
        def test_prompt_contains_all_required_keys(self):
            prompt = self.engine._load_external_prompt(
                "BTC/USDT",
                self.mock_quant_fusion,
                self.mock_risk_data,
                self.mock_btc_macro,
                self.mock_trade_memory,
                {"has_high_impact_news": False}
            )
            # ফিক্স: প্রম্পটে Expected Value (EV): 1.8R আছে
            self.assertIn("BTC/USDT", prompt)
            self.assertIn("**DIRECTION:** LONG", prompt)
            self.assertIn("Unified Score: 85.0/100", prompt)
            self.assertIn("Expected Value (EV): 1.8R", prompt)   # <-- FIXED
            self.assertIn("Technical: 80", prompt)
            self.assertIn("SMC (Smart Money): 75", prompt)
            self.assertIn("Liquidity: 80", prompt)
            self.assertIn("Risk: 90", prompt)
            self.assertIn("Entry: 65000.0", prompt)
            self.assertIn("Stop Loss: 64000.0", prompt)
            self.assertIn("Take Profit: 68000.0", prompt)
            self.assertIn("ATR (5m): 350.0", prompt)
            self.assertIn("Leverage: 1.5", prompt)
            self.assertIn("Risk/Reward: 1:3.0", prompt)
            self.assertIn("BTC Regime: TRENDING", prompt)
            self.assertIn("BTC Bullish: True", prompt)
            self.assertIn("SOL", prompt)
            self.assertIn("ETH", prompt)
            self.assertIn("OUTPUT SCHEMA", prompt)

        def test_prompt_handles_missing_data(self):
            empty_quant = {"score_breakdown": {}, "derivatives_data": {}}
            empty_risk = {}
            prompt = self.engine._load_external_prompt(
                "TEST/USDT",
                empty_quant,
                empty_risk,
                {"is_bullish": False, "regime": "BEAR"},
                [],
                {"has_high_impact_news": False}
            )
            # ফিক্স: Expected Value (EV): 1.5R
            self.assertIn("TEST/USDT", prompt)
            self.assertIn("**DIRECTION:** LONG", prompt)
            self.assertIn("Expected Value (EV): 1.5R", prompt)   # <-- FIXED
            self.assertIn("Technical: 50", prompt)
            self.assertIn("SMC (Smart Money): 50", prompt)
            self.assertIn("Liquidity: 50", prompt)
            self.assertIn("Risk: 50", prompt)
            self.assertIn("Leverage: 1", prompt)
            self.assertIn("BTC Regime: BEAR", prompt)

        def test_prompt_handles_short_direction(self):
            risk_data = self.mock_risk_data.copy()
            risk_data["direction"] = "SHORT"
            prompt = self.engine._load_external_prompt(
                "BTC/USDT",
                self.mock_quant_fusion,
                risk_data,
                self.mock_btc_macro,
                [],
                {"has_high_impact_news": False}
            )
            self.assertIn("**DIRECTION:** SHORT", prompt)

        # ============================================================
        # 4. INTEGRATION: GATEKEEPER & HARD REJECTION
        # ============================================================
        
        def test_hard_gatekeeper_reject_low_score(self):
            quant_fusion = self.mock_quant_fusion.copy()
            quant_fusion["unified_score"] = 70.0
            quant_fusion["is_passed"] = True
            result = self.engine.run_debate_and_decide(
                "BTC/USDT", quant_fusion, self.mock_risk_data, self.mock_btc_macro
            )
            self.assertEqual(result["final_decision"], "REJECT")
            self.assertIn("Quant Hard Guard Veto Triggered", str(result["reasons"]))

        def test_hard_gatekeeper_reject_low_ev(self):
            quant_fusion = self.mock_quant_fusion.copy()
            quant_fusion["ev_r"] = 1.0
            quant_fusion["is_passed"] = True
            result = self.engine.run_debate_and_decide(
                "BTC/USDT", quant_fusion, self.mock_risk_data, self.mock_btc_macro
            )
            self.assertEqual(result["final_decision"], "REJECT")
            self.assertIn("Quant Hard Guard Veto Triggered", str(result["reasons"]))

        def test_hard_gatekeeper_reject_btc_bear_long(self):
            btc_macro = {"is_bullish": False, "regime": "BEAR"}
            result = self.engine.run_debate_and_decide(
                "BTC/USDT", self.mock_quant_fusion, self.mock_risk_data, btc_macro
            )
            self.assertEqual(result["final_decision"], "REJECT")
            self.assertIn("Cannot LONG altcoins during Bearish BTC Macro", str(result["reasons"]))

        def test_hard_gatekeeper_reject_high_impact_news(self):
            news = {"has_high_impact_news": True, "news_event": "FED_RATE_HIKE"}
            result = self.engine.run_debate_and_decide(
                "BTC/USDT", self.mock_quant_fusion, self.mock_risk_data, self.mock_btc_macro, news_event_data=news
            )
            self.assertEqual(result["final_decision"], "REJECT")
            self.assertIn("High Impact News Active", str(result["reasons"]))

        @patch.object(InstitutionalAIDebateEngine, '_call_multi_provider_llm')
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
            # Posterior should be 1.0 => capped to 95
            self.assertEqual(result["confidence_score"], 95)
            self.assertEqual(result["confidence"], 95)

        # ============================================================
        # 5. REAL DATA INTEGRATION TEST (optional, skipped if imports fail)
        # ============================================================
        @unittest.skipIf(True, "Skipping real data test to avoid external dependencies in CI")
        def test_with_real_current_data(self):
            """
            REAL DATA TEST: Fetches actual OHLCV from exchange and runs the full pipeline.
            This is an integration test (slower, requires internet).
            """
            try:
                from data_fetcher import DataFetcher
                from technical_engine import TechnicalEngine
                from risk_engine import RiskEngine
            except ImportError as e:
                self.skipTest(f"Skipping real data test due to import error: {e}")

            try:
                fetcher = DataFetcher()
                symbol = "BTC/USDT"
                df_15m = fetcher.fetch_ohlcv(symbol, timeframe="15m", limit=50)
                if df_15m is None or df_15m.empty:
                    self.skipTest("Could not fetch real data (no internet or exchange issue)")
                
                tech_engine = TechnicalEngine()
                tech_result = tech_engine.analyze(df_15m, market_regime="TRENDING")
                
                risk_result = RiskEngine.calculate_trade_risk(
                    entry_price=df_15m['close'].iloc[-1],
                    atr_5m=tech_result.get('features', {}).get('atr_ratio_15m', 0.01) * df_15m['close'].iloc[-1] or 100,
                    account_balance=10000.0,
                    direction="LONG"
                )
                
                quant_fusion = {
                    "is_passed": True,
                    "unified_score": tech_result["technical_score"],
                    "ev_r": risk_result.get("risk_metrics", {}).get("rr_score_raw", 1.5),
                    "score_breakdown": {
                        "technical": tech_result["technical_score"],
                        "smc": 75.0,
                        "liquidity": 80.0,
                        "risk": risk_result.get("advanced_metrics", {}).get("safety_score", 70.0)
                    }
                }
                
                result = self.engine.run_debate_and_decide(
                    symbol=symbol,
                    quant_fusion_data=quant_fusion,
                    risk_data=risk_result,
                    btc_macro_data={"is_bullish": True, "regime": "TRENDING"}
                )
                
                self.assertTrue(result["success"])
                self.assertIn(result["final_decision"], self.engine.ALLOWED_DECISIONS)
                
                print("\n" + "="*60)
                print(f"🧪 REAL DATA TEST RESULT for {symbol}:")
                print(f"   Decision: {result['final_decision']}")
                print(f"   Confidence: {result['confidence_score']}%")
                print(f"   Summary: {result.get('summary', 'N/A')}")
                print("="*60)
                
                with open(f"test_output_{symbol.replace('/', '_')}.json", "w") as f:
                    json.dump(result, f, indent=4)
                print(f"✅ Result saved to test_output_{symbol.replace('/', '_')}.json")
                
            except Exception as e:
                self.fail(f"Real data test crashed: {e}")

    # ============================================================
    # Run all tests
    # ============================================================
    print("🧪 Running InstitutionalAIDebateEngine Unit Tests...")
    print("=" * 60)
    unittest.main(argv=[''], verbosity=2, exit=False)
    print("=" * 60)
    print("✅ All tests completed.")