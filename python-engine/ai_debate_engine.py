import os
import json
import time
import logging
import re
from typing import Dict, Any, List, Optional

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
    Unified Multi-Agent LLM Debate Engine with Multi-Provider Failover & Institutional Guards.
    Consolidates ai_debate_engine.py and legacy ai_engine.py logic into a single robust pipeline.
    """

    ALLOWED_DECISIONS = {"EXECUTE_LONG", "EXECUTE_SHORT", "WAIT_FOR_DIP", "WATCHLIST", "REJECT", "HOLD"}
    ALLOWED_VOTES = {"EXECUTE", "REJECT"}

    def __init__(
        self, 
        gemini_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        deepseek_key: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt_version: str = "v2",
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
        
        self.min_score_for_ai = float(self.config.get("AI_MIN_SCORE", os.getenv("AI_MIN_SCORE", 78.0)))
        self.min_ev_for_ai = float(self.config.get("AI_MIN_EV", os.getenv("AI_MIN_EV", 1.3)))
        self.hard_min_ev = float(self.config.get("HARD_MIN_EV", os.getenv("HARD_MIN_EV", 1.2)))
        self.hard_min_score = float(self.config.get("HARD_MIN_SCORE", os.getenv("HARD_MIN_SCORE", 75.0)))
        
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

        if direction == "LONG" and not btc_bullish:
            return self._build_hard_rejection(symbol, ["Cannot LONG altcoins during Bearish BTC Macro"], start_time, risk_data)

        if news_event_data.get("has_high_impact_news", False):
            return self._build_hard_rejection(symbol, [f"High Impact News Active ({news_event_data.get('news_event')})"], start_time, risk_data)

        # 3. AI Cost Optimization Check & Multi-Provider LLM Call
        if unified_score >= self.min_score_for_ai and ev_r >= self.min_ev_for_ai:
            llm_res = self._call_multi_provider_llm(
                symbol, quant_fusion_data, risk_data, btc_macro_data, recent_trade_memory, start_time
            )
            if llm_res:
                # Conservative Calibrated Confidence (Max 95% Cap)
                raw_ai_confidence = llm_res.get("confidence_score", llm_res.get("confidence", 70))
                quant_factor = min(1.0, unified_score / 100.0)
                calibrated_confidence = min(95, int(raw_ai_confidence * quant_factor))
                
                llm_res["confidence_score"] = calibrated_confidence
                llm_res["confidence"] = calibrated_confidence
                return llm_res

        # 4. Institutional Fallback Rule Engine
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
        volume_score = score_breakdown.get("volume", 70)

        bull_vote = "EXECUTE" if (tech_score >= 75 and smc_score >= 75 and volume_score >= 65) else "REJECT"
        bear_vote = "EXECUTE" if (ev_r >= self.min_ev_for_ai and quant_fusion.get("unified_score", 80) >= 75) else "REJECT"

        oi_change = derivatives_data.get("oi_change_24h", 0.0)
        funding_rate = derivatives_data.get("funding_rate", 0.01)
        ls_ratio = derivatives_data.get("long_short_ratio", 1.0)
        liquidation_cluster = derivatives_data.get("liquidation_cluster", "NEUTRAL")

        manipulation_vote = "REJECT"
        if direction == "LONG":
            oi_surge = oi_change >= 5.0
            funding_healthy = funding_rate <= 0.03
            short_trap_potential = (ls_ratio < 0.9 and liquidation_cluster == "SHORT") or oi_surge
            if funding_healthy and (short_trap_potential or volume_score >= 70):
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
            "reasons": [f"Tech: {tech_score}, SMC: {smc_score}, Vol: {volume_score}"],
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
        start_time: float
    ) -> Optional[Dict[str, Any]]:
        """Orchestrates multi-provider failover: Gemini -> OpenAI -> DeepSeek."""
        prompt_text = self._load_external_prompt(symbol, quant_fusion, risk_data, btc_macro, trade_memory)
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
        client = OpenAI(api_key=self.openai_key)
        response = client.chat.completions.create(
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
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if not json_match:
                return None
            parsed = json.loads(json_match.group(0))
            
            decision = parsed.get("final_decision", parsed.get("decision", "REJECT"))
            confidence = parsed.get("confidence_score", parsed.get("confidence", 70))

            if decision not in self.ALLOWED_DECISIONS:
                decision = "REJECT"

            parsed["success"] = True
            parsed["symbol"] = symbol
            parsed["final_decision"] = decision
            parsed["confidence_score"] = int(confidence)
            parsed["confidence"] = int(confidence)
            parsed["execution_plan"] = risk_data.get("trade_levels", {})
            parsed["provider_used"] = provider_name
            parsed["telemetry"] = {
                "latency_sec": round(time.time() - start_time, 3),
                "llm_used": True,
                "provider": provider_name,
                "prompt_version": self.prompt_version
            }
            return parsed
        except Exception:
            return None

    def _load_external_prompt(self, symbol: str, quant_fusion: Dict[str, Any], risk_data: Dict[str, Any], btc_macro: Dict[str, Any], trade_memory: List[Dict[str, Any]]) -> str:
        return f"""
        You are an Institutional Crypto Trading Committee.
        Symbol: {symbol} | Direction: {risk_data.get('direction', 'LONG')}
        Unified Score: {quant_fusion.get('unified_score', 80)}/100 | EV: {quant_fusion.get('ev_r', 1.5)}R
        Breakdown: {json.dumps(quant_fusion.get('score_breakdown', {}))}
        Derivatives: {json.dumps(quant_fusion.get('derivatives_data', {}))}

        OUTPUT SCHEMA STRICT JSON:
        {{
            "final_decision": "EXECUTE_LONG" | "EXECUTE_SHORT" | "WATCHLIST" | "REJECT",
            "confidence_score": <integer 0-100>,
            "summary": "<Executive Summary>",
            "reasons": ["<reason 1>"],
            "risks": ["<risk 1>"],
            "ai_votes": {{"bull_ai": "EXECUTE"|"REJECT", "bear_ai": "EXECUTE"|"REJECT", "manipulation_ai": "EXECUTE"|"REJECT", "cio_ai": "EXECUTE"|"REJECT"}},
            "agreement_pct": <float>
        }}
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
            "agreement_pct": 100.0,
            "telemetry": {"latency_sec": round(time.time() - start_time, 3), "llm_used": False, "fallback_reason": "Hard Gatekeeper Veto"}
        }

# Legacy Compatibility Alias
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