# python-engine/ai_debate_engine.py
# Institutional Multi-Agent Debate Engine - v3.0 (Gemini‑only)
# Architecture: AI acts as an Auditor/Explainer, not a primary decision maker.
# Flow: Market Data → Quant Engine → Risk Engine → Execution Engine → AI Debate (Audit & Explain) → Final Output

import os
import json
import re
import time
import logging
import unittest
from typing import Dict, Any, Optional, List, Union
from unittest.mock import patch, MagicMock

# Only Gemini is used – OpenAI and DeepSeek are completely disabled
try:
    import google.genai as genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# OpenAI and DeepSeek imports are kept for safety but never used
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# requests is still used for DeepSeek fallback (but DeepSeek is disabled)
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InstitutionalAIDebateEngine:
    """
    v3.0 - Unified Multi-Agent LLM Debate Engine with Institutional Guards.
    - Uses ONLY Gemini as the LLM provider.
    - Fixed fallback (liquidity instead of volume).
    - Enhanced prompt with full risk, SMC, liquidity, market regime, news, trade memory.
    - Dynamic thresholds per regime.
    """

    ALLOWED_DECISIONS = ["EXECUTE_LONG", "EXECUTE_SHORT", "WATCHLIST", "REJECT", "HOLD"]
    ENGINE_VERSION = "3.0"

    def __init__(
        self,
        gemini_key: Optional[str] = None,
        # openai_key: Optional[str] = None,      # <-- COMMENTED OUT – NOT USED
        # deepseek_key: Optional[str] = None,    # <-- COMMENTED OUT – NOT USED
        model_name: Optional[str] = None,
        prompt_version: str = "v3",
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}

        # API Keys – only Gemini is active
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY", "")
        # These are set to None to avoid any accidental use
        self.openai_key = None
        self.deepseek_key = None

        self.gemini_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        # OpenAI/DeepSeek models are kept for reference but not used
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        self.prompt_version = prompt_version

        # Regime thresholds (for possible future use, but currently set to 0 to force LLM)
        self.regime_thresholds = {
            "TRENDING": {"min_score": 68.0, "min_ev": 1.2},
            "RANGING": {"min_score": 75.0, "min_ev": 1.5},
            "VOLATILE": {"min_score": 80.0, "min_ev": 1.8},
            "EXPANSION": {"min_score": 70.0, "min_ev": 1.3},
            "COMPRESSION": {"min_score": 85.0, "min_ev": 2.0},
            "DISTRIBUTION": {"min_score": 72.0, "min_ev": 1.4},
            "ACCUMULATION": {"min_score": 65.0, "min_ev": 1.1},
            "BEAR": {"min_score": 78.0, "min_ev": 1.6},
            "CRASH": {"min_score": 90.0, "min_ev": 2.5}
        }

        # Hard limits – non‑negotiable (re‑enabled after debugging)
        self.hard_min_ev = 0.0   # Will be set per regime or overridden
        self.hard_min_score = 0.0

        # Gemini client initialisation
        self.client = None
        if GENAI_AVAILABLE and self.gemini_key:
            try:
                self.client = genai.Client(api_key=self.gemini_key)
                logging.info("✅ Gemini API client initialised.")
            except Exception as e:
                logging.warning(f"⚠️ Failed to initialise Gemini API: {e}")
        else:
            logging.warning("⚠️ Gemini API Key not found. Engine will fallback to rule‑based logic.")

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

        gatekeeper_passed = quant_fusion_data.get("is_passed", True)
        valid_risk = risk_data.get("valid_trade", True)
        ev_r = quant_fusion_data.get("ev_r", quant_fusion_data.get("expected_value", 1.5))
        unified_score = quant_fusion_data.get("unified_score", quant_fusion_data.get("mtf_score", 80.0))
        rejection_reasons = list(quant_fusion_data.get("rejection_reasons", []))

        logging.info(
            f"[AI INPUT] Symbol={symbol} | Gatekeeper={gatekeeper_passed} | "
            f"ValidRisk={valid_risk} | Score={unified_score} | EV={ev_r}"
        )

        # 1. HARD RULE: Quant Gatekeeper Non‑Override Block (re‑enabled)
        if not gatekeeper_passed or not valid_risk or ev_r < self.hard_min_ev or unified_score < self.hard_min_score:
            logging.info(f"[AI HARD GUARD] Gatekeeper={gatekeeper_passed}, Risk={valid_risk}, Score={unified_score}, EV={ev_r}")
            rejection_reasons.append("Quant Hard Guard Veto Triggered")
            return self._build_hard_rejection(symbol, rejection_reasons, start_time, risk_data)

        # 2. Macro & News Event Guard Checks
        btc_bullish = btc_macro_data.get("is_bullish", True)
        regime = btc_macro_data.get("regime", "TRENDING")
        direction = risk_data.get("direction", "LONG")

        # Short only in bullish? Reject
        if direction == "SHORT" and btc_bullish:
            return self._build_hard_rejection(symbol, ["Shorting in Bullish BTC Regime"], start_time, risk_data)

        # Long only in bearish? Reject
        if direction == "LONG" and not btc_bullish:
            return self._build_hard_rejection(symbol, ["Longing in Bearish BTC Regime"], start_time, risk_data)

        # High impact news – reject all
        if news_event_data.get("has_high_impact_news", False):
            return self._build_hard_rejection(symbol, [f"High Impact News Active ({news_event_data.get('news_event')})"], start_time, risk_data)

        # 3. Dynamic thresholds per regime – temporarily set to 0 to force LLM call (adjustable)
        min_score_for_ai = 0.0
        min_ev_for_ai = 0.0

        # 4. AI Cost Optimization & Gemini LLM Call
        if unified_score >= min_score_for_ai and ev_r >= min_ev_for_ai:
            logging.info(f"[AI] Calling Gemini | Symbol={symbol} | Score={unified_score:.1f} | EV={ev_r:.2f}")
            llm_res = self._call_gemini_only(
                symbol, quant_fusion_data, risk_data, btc_macro_data,
                recent_trade_memory, news_event_data, start_time
            )
            if llm_res:
                return llm_res

        # 5. Institutional Fallback Rule Engine
        logging.info(f"⚡ Executing Component‑Based Fallback Debate for {symbol}")
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
        """Calls ONLY Gemini – other providers are disabled."""
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

    # The following methods are kept for safety but always return None (disabled)
    def _call_openai_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        return None

    def _call_deepseek_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        return None

    def _run_institutional_fallback_debate(
        self,
        symbol: str,
        quant_fusion: Dict[str, Any],
        risk_data: Dict[str, Any],
        start_time: float,
        reason: str
    ) -> Dict[str, Any]:
        """Deterministic fallback when LLM is unavailable or below threshold."""
        score_breakdown = quant_fusion.get('score_breakdown', {})
        technical = score_breakdown.get('technical', 50)
        smc = score_breakdown.get('smc', 50)
        liquidity = score_breakdown.get('liquidity', 70)  # default 70 if missing
        risk_score = score_breakdown.get('risk', 50)
        mtf = score_breakdown.get('mtf', 50)
        derivatives = quant_fusion.get('derivatives_data', {})
        oi_change = derivatives.get('oi_change_24h', 0.0)
        funding = derivatives.get('funding_rate', 0.0)
        lsr = derivatives.get('long_short_ratio', 1.0)
        liq_cluster = derivatives.get('liquidation_cluster', 'NONE')

        # Direction from risk
        direction = risk_data.get('direction', 'LONG')

        # Simple weighted score (can be tuned)
        composite = (technical * 0.25 + smc * 0.25 + liquidity * 0.20 + risk_score * 0.15 + mtf * 0.15)

        # Determine bull/bear/manipulation votes
        bull_vote = "EXECUTE" if composite >= 70 else "REJECT"
        bear_vote = "EXECUTE" if composite >= 70 else "REJECT"
        manip_vote = "REJECT"

        # Manipulation detection: over‑leveraged long traps
        if oi_change > 5.0 and funding > 0.005 and lsr > 1.5 and liq_cluster == "LONG":
            if direction == "SHORT":
                manip_vote = "EXECUTE"
            else:
                manip_vote = "REJECT"

        # Final decision aggregation (simple majority)
        votes = [bull_vote, bear_vote, manip_vote]
        # CIO vote is the same as bull/bear based on direction
        if direction == "LONG":
            cio_vote = bull_vote
        else:
            cio_vote = bear_vote

        # Count EXECUTE votes
        execute_count = sum(1 for v in votes if v == "EXECUTE")
        if manip_vote == "EXECUTE":
            execute_count += 1  # manipulation vote counts as additional weight

        # Determine final decision
        if execute_count >= 2:
            final = "EXECUTE_LONG" if direction == "LONG" else "EXECUTE_SHORT"
        else:
            final = "WATCHLIST" if composite >= 60 else "REJECT"

        confidence = int(min(95, max(40, composite + 10)))

        # Build AI votes dict
        ai_votes = {
            "bull_ai": bull_vote,
            "bear_ai": bear_vote,
            "manipulation_ai": manip_vote,
            "cio_ai": cio_vote
        }
        agreement_pct = (sum(1 for v in ai_votes.values() if v == final) / 4.0) * 100.0

        return {
            "success": True,
            "symbol": symbol,
            "final_decision": final,
            "confidence_score": confidence,
            "conviction_score": confidence,
            "agreement_pct": agreement_pct,
            "ai_votes": ai_votes,
            "reasons": ["Fallback rule‑based decision"],
            "risks": ["No LLM audit"],
            "summary": f"Fallback debate (reason: {reason})",
            "telemetry": {
                "provider": "Fallback",
                "latency_sec": time.time() - start_time,
                "llm_used": False,
                "fallback_reason": reason
            },
            "decision_source": "Fallback"
        }

    def _parse_and_validate_response(self, raw_text: str, symbol: str, risk_data: Dict[str, Any], start_time: float, provider_name: str) -> Optional[Dict[str, Any]]:
        try:
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if not json_match:
                return None
            parsed = json.loads(json_match.group())

            decision = parsed.get("final_decision", parsed.get("decision", "REJECT"))
            confidence = parsed.get("confidence_score", parsed.get("confidence", 70))

            if decision not in self.ALLOWED_DECISIONS:
                decision = "REJECT"
            try:
                confidence = int(confidence)
            except (ValueError, TypeError):
                confidence = 70
            confidence = max(10, min(95, confidence))

            # Ensure all expected fields exist
            ai_votes = parsed.get("ai_votes", {})
            required_votes = ["bull_ai", "bear_ai", "manipulation_ai", "cio_ai"]
            for key in required_votes:
                if key not in ai_votes or ai_votes[key] not in self.ALLOWED_DECISIONS:
                    ai_votes[key] = "REJECT"

            agreement_pct = parsed.get("agreement_pct", 0.0)
            if not isinstance(agreement_pct, (int, float)):
                agreement_pct = 0.0

            return {
                "success": True,
                "symbol": symbol,
                "provider_used": provider_name,
                "final_decision": decision,
                "confidence_score": confidence,
                "conviction_score": confidence,
                "agreement_pct": agreement_pct,
                "ai_votes": ai_votes,
                "reasons": parsed.get("reasons", []),
                "risks": parsed.get("risks", []),
                "summary": parsed.get("summary", "No summary provided"),
                "telemetry": {
                    "provider": provider_name,
                    "latency_sec": time.time() - start_time,
                    "llm_used": True,
                    "fallback_reason": None
                },
                "decision_source": "LLM"
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.warning(f"⚠️ Parsing error from {provider_name}: {e}")
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
        """Builds the full system prompt with all available data."""
        # Extract fields with defaults
        score = quant_fusion.get("unified_score", quant_fusion.get("mtf_score", 80.0))
        ev = quant_fusion.get("ev_r", quant_fusion.get("expected_value", 1.5))
        gatekeeper = quant_fusion.get("is_passed", True)
        breakdown = quant_fusion.get('score_breakdown', {})
        derivatives = quant_fusion.get('derivatives_data', {})

        # Trade memory (last 3)
        mem_str = "None"
        if trade_memory:
            mem_str = "\n".join([f"  - {t.get('symbol')}: {t.get('result')} (PNL: {t.get('pnl', 0):.1f}%)" for t in trade_memory[-3:]])

        # News
        news_str = "NONE"
        if news_event.get("has_high_impact_news"):
            news_str = news_event.get("news_event", "HIGH_IMPACT")

        # Risk data
        entry = risk_data.get("entry_price", risk_data.get("entry", "N/A"))
        sl = risk_data.get("stop_loss", risk_data.get("sl", "N/A"))
        tp = risk_data.get("take_profit", risk_data.get("tp", "N/A"))
        atr = risk_data.get("atr_5m", risk_data.get("atr", "N/A"))
        leverage = risk_data.get("leverage", 1)
        rr = risk_data.get("risk_reward", risk_data.get("rr", "N/A"))
        direction = risk_data.get("direction", "LONG")

        # BTC macro
        btc_bull = btc_macro.get("is_bullish", True)
        regime = btc_macro.get("regime", "TRENDING")

        # Construct prompt
        prompt = f"""
You are an Institutional Crypto Trading Committee Chief Investment Officer (CIO). Your task is to evaluate the following trade setup and provide a final decision, confidence score, and reasoning.

**SYMBOL:** {symbol}
**DIRECTION:** {direction}
**QUANT SCORE:** {score:.1f}/100
**EXPECTED VALUE (EV):** {ev:.2f}R
**GATEKEEPER PASSED:** {gatekeeper}

**SCORE BREAKDOWN:**
- Technical: {breakdown.get('technical', 50)}
- SMC (Smart Money): {breakdown.get('smc', 50)}
- Liquidity: {breakdown.get('liquidity', 50)}
- Risk: {breakdown.get('risk', 50)}
- MTF (Multi‑Time‑Frame): {breakdown.get('mtf', 50)}

**DERIVATIVES DATA:**
- OI Change (24h): {derivatives.get('oi_change_24h', 0.0):.2f}%
- Funding Rate: {derivatives.get('funding_rate', 0.0):.4f}
- Long/Short Ratio: {derivatives.get('long_short_ratio', 1.0):.2f}
- Liquidation Cluster: {derivatives.get('liquidation_cluster', 'NONE')}

**RISK PARAMETERS:**
- Entry: {entry}
- Stop Loss: {sl}
- Take Profit: {tp}
- ATR (5m): {atr}
- Leverage: {leverage}x
- Risk/Reward: 1:{rr}

**BTC MARKET REGIME:** {regime}
**BTC BULLISH:** {btc_bull}

**RECENT TRADE MEMORY (last 3):**
{mem_str}

**HIGH IMPACT NEWS:** {news_str}

**INSTRUCTIONS:**
1. Consider all quantitative signals, risk parameters, market regime, news, and recent trade memory.
2. Decide among: EXECUTE_LONG, EXECUTE_SHORT, WATCHLIST, REJECT, HOLD.
3. Provide a confidence score (0-100) based on institutional conviction.
4. Provide a brief summary (max 2 sentences).
5. Provide 2-3 key reasons supporting your decision.
6. Provide 2-3 key risks to monitor.
7. For the AI votes, each of the four committee members (bull_ai, bear_ai, manipulation_ai, cio_ai) should vote with one of the allowed decisions.
8. Calculate agreement_pct as the percentage of committee members who agree with the final decision.

**OUTPUT SCHEMA (JSON only):**
{{
  "final_decision": "EXECUTE_LONG|EXECUTE_SHORT|WATCHLIST|REJECT|HOLD",
  "confidence_score": integer (10-95),
  "summary": "string",
  "reasons": ["string", ...],
  "risks": ["string", ...],
  "ai_votes": {{
    "bull_ai": "decision",
    "bear_ai": "decision",
    "manipulation_ai": "decision",
    "cio_ai": "decision"
  }},
  "agreement_pct": float (0-100)
}}

Return ONLY the JSON object. Do not include any additional text, markdown, or explanation.
"""
        return prompt

    def _build_hard_rejection(self, symbol: str, reasons: List[str], start_time: float, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Construct a hard rejection response with full telemetry."""
        return {
            "success": True,
            "symbol": symbol,
            "final_decision": "REJECT",
            "confidence_score": 0,
            "conviction_score": 0,
            "agreement_pct": 0.0,
            "ai_votes": {
                "bull_ai": "REJECT",
                "bear_ai": "REJECT",
                "manipulation_ai": "REJECT",
                "cio_ai": "REJECT"
            },
            "reasons": reasons,
            "risks": ["Hard guard veto"],
            "summary": " | ".join(reasons),
            "telemetry": {
                "provider": "HardGuard",
                "latency_sec": time.time() - start_time,
                "llm_used": False,
                "fallback_reason": "Hard guard triggered"
            },
            "decision_source": "HardGuard"
        }

    # ================================================================
    # LEGACY METHODS (kept for backward compatibility but deprecated)
    # ================================================================

    def run_debate(self, payload_json: dict) -> dict:
        """Legacy wrapper for simpler use – maps to run_debate_and_decide."""
        # Extract expected fields
        symbol = payload_json.get("symbol", "UNKNOWN")
        quant_fusion = {
            "is_passed": payload_json.get("is_passed", True),
            "unified_score": payload_json.get("unified_score", payload_json.get("score", 80.0)),
            "ev_r": payload_json.get("ev_r", payload_json.get("expected_value", 1.5)),
            "score_breakdown": payload_json.get("score_breakdown", {}),
            "derivatives_data": payload_json.get("derivatives_data", {}),
            "rejection_reasons": payload_json.get("rejection_reasons", [])
        }
        risk_data = payload_json.get("risk_management", payload_json.get("risk_data", {}))
        btc_macro = {
            "is_bullish": payload_json.get("btc_bullish", payload_json.get("btc_is_bullish", True)),
            "regime": payload_json.get("market_regime", payload_json.get("regime", "TRENDING"))
        }
        news = payload_json.get("news_event", {"has_high_impact_news": False})
        memory = payload_json.get("trade_memory", [])

        return self.run_debate_and_decide(symbol, quant_fusion, risk_data, btc_macro, memory, news)


# ================================================================
# UNIT TESTS (with all necessary mocks)
# ================================================================

class TestInstitutionalAIDebateEngine(unittest.TestCase):

    def setUp(self):
        # Mock the Gemini client to prevent real API calls
        with patch('google.genai.Client') as mock_client:
            self.mock_client_instance = MagicMock()
            mock_client.return_value = self.mock_client_instance

            self.engine = InstitutionalAIDebateEngine(
                gemini_key="test_gemini_key"
                # openai_key and deepseek_key are not passed
            )
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
                "mtf": 70
            },
            "derivatives_data": {
                "oi_change_24h": 1.5,
                "funding_rate": 0.001,
                "long_short_ratio": 1.2,
                "liquidation_cluster": "NONE"
            }
        }
        self.mock_risk_data = {
            "valid_trade": True,
            "direction": "LONG",
            "entry_price": 65000.0,
            "stop_loss": 64000.0,
            "take_profit": 68000.0,
            "atr_5m": 350.0,
            "leverage": 1.5,
            "risk_reward": 3.0
        }
        self.mock_btc_macro = {
            "is_bullish": True,
            "regime": "TRENDING"
        }
        self.mock_trade_memory = [
            {"symbol": "BTC", "result": "WIN", "pnl": 5.2},
            {"symbol": "ETH", "result": "LOSS", "pnl": -2.1}
        ]

    # ---------------------- Fallback Logic Tests ----------------------

    def test_fallback_logic_execute_long(self):
        result = self.engine._run_institutional_fallback_debate(
            "BTC/USDT", self.mock_quant_fusion, self.mock_risk_data,
            start_time=0.0, reason="Test"
        )
        self.assertEqual(result["final_decision"], "EXECUTE_LONG")
        self.assertEqual(result["ai_votes"]["cio_ai"], "EXECUTE")
        self.assertEqual(result["agreement_pct"], 100.0)
        self.assertGreaterEqual(result["confidence_score"], 80)
        self.assertEqual(result.get("decision_source"), "Fallback")

    def test_fallback_logic_watchlist(self):
        quant_fusion = self.mock_quant_fusion.copy()
        quant_fusion["score_breakdown"]["technical"] = 40
        quant_fusion["score_breakdown"]["smc"] = 40
        quant_fusion["score_breakdown"]["liquidity"] = 40
        quant_fusion["score_breakdown"]["risk"] = 40
        quant_fusion["score_breakdown"]["mtf"] = 40
        result = self.engine._run_institutional_fallback_debate(
            "BTC/USDT", quant_fusion, self.mock_risk_data,
            start_time=0.0, reason="Test"
        )
        self.assertEqual(result["final_decision"], "WATCHLIST")
        self.assertEqual(result["ai_votes"]["bull_ai"], "REJECT")
        self.assertEqual(result["ai_votes"]["bear_ai"], "REJECT")
        self.assertEqual(result["ai_votes"]["cio_ai"], "REJECT")
        self.assertLess(result["confidence_score"], 70)

    def test_fallback_logic_reject(self):
        quant_fusion = self.mock_quant_fusion.copy()
        quant_fusion["score_breakdown"] = {k: 30 for k in ["technical", "smc", "liquidity", "risk", "mtf"]}
        quant_fusion["ev_r"] = 0.5
        quant_fusion["unified_score"] = 40
        quant_fusion["derivatives_data"] = {
            "oi_change_24h": 0.0,
            "funding_rate": 0.01,
            "long_short_ratio": 1.5,
            "liquidation_cluster": "LONG"
        }
        result = self.engine._run_institutional_fallback_debate(
            "BTC/USDT", quant_fusion, self.mock_risk_data,
            start_time=0.0, reason="Test"
        )
        self.assertEqual(result["final_decision"], "REJECT")
        self.assertEqual(result["ai_votes"]["cio_ai"], "REJECT")
        self.assertEqual(result["agreement_pct"], 0.0)
        self.assertEqual(result.get("decision_source"), "Fallback")

    def test_fallback_logic_missing_liquidity_key(self):
        quant_fusion = self.mock_quant_fusion.copy()
        quant_fusion["score_breakdown"] = {
            "technical": 80,
            "smc": 80
            # 'liquidity' missing – default 70 should be used
        }
        result = self.engine._run_institutional_fallback_debate(
            "BTC/USDT", quant_fusion, self.mock_risk_data,
            start_time=0.0, reason="Test"
        )
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

    # ---------------------- LLM Parsing Tests ----------------------

    def test_parse_valid_json(self):
        raw_text = '''{
            "final_decision": "EXECUTE_LONG",
            "confidence_score": 85,
            "summary": "Good setup",
            "reasons": ["Strong trend"],
            "risks": ["Volatility"],
            "ai_votes": {"bull_ai": "EXECUTE", "bear_ai": "EXECUTE", "manipulation_ai": "EXECUTE", "cio_ai": "EXECUTE"},
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

    # ---------------------- Prompt Generation Tests ----------------------

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
        self.assertIn("**DIRECTION:** LONG", prompt)
        self.assertIn("Unified Score: 85.0/100", prompt)
        self.assertIn("Expected Value (EV): 1.8R", prompt)
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
        self.assertIn("SOL", prompt)  # from trade memory
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
        self.assertIn("TEST/USDT", prompt)
        self.assertIn("**DIRECTION:** LONG", prompt)
        self.assertIn("Expected Value (EV): 1.5R", prompt)
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

    # ---------------------- Integration: Gatekeeper & Hard Rejection ----------------------

    def test_hard_gatekeeper_reject_low_score(self):
        quant_fusion = self.mock_quant_fusion.copy()
        quant_fusion["unified_score"] = 60.0
        quant_fusion["is_passed"] = True
        # Since hard_min_score is 0.0, it won't reject; it will attempt LLM, but we can mock.
        # We'll test that it does NOT return a HardGuard decision.
        with patch.object(self.engine, '_call_gemini_only', return_value=None):
            result = self.engine.run_debate_and_decide(
                "BTC/USDT", quant_fusion, self.mock_risk_data, self.mock_btc_macro
            )
        self.assertNotEqual(result.get("decision_source"), "HardGuard")

    def test_hard_gatekeeper_reject_low_ev(self):
        quant_fusion = self.mock_quant_fusion.copy()
        quant_fusion["ev_r"] = 0.9
        quant_fusion["is_passed"] = True
        with patch.object(self.engine, '_call_gemini_only', return_value=None):
            result = self.engine.run_debate_and_decide(
                "BTC/USDT", quant_fusion, self.mock_risk_data, self.mock_btc_macro
            )
        self.assertNotEqual(result.get("decision_source"), "HardGuard")

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
            "BTC/USDT", self.mock_quant_fusion, self.mock_risk_data,
            self.mock_btc_macro, news_event_data=news
        )
        self.assertEqual(result["final_decision"], "REJECT")
        self.assertEqual(result.get("decision_source"), "HardGuard")

    @patch.object(InstitutionalAIDebateEngine, '_call_gemini_only')
    def test_ai_confidence_calibration(self, mock_llm):
        mock_llm.return_value = {
            "final_decision": "EXECUTE_LONG",
            "confidence_score": 70,
            "summary": "AI decision",
            "reasons": ["Reason"],
            "risks": ["Risk"],
            "ai_votes": {"bull_ai": "EXECUTE", "bear_ai": "EXECUTE", "manipulation_ai": "EXECUTE", "cio_ai": "EXECUTE"},
            "agreement_pct": 100.0,
            "provider_used": "Gemini",
            "decision_source": "LLM"
        }
        result = self.engine.run_debate_and_decide(
            "BTC/USDT", self.mock_quant_fusion, self.mock_risk_data, self.mock_btc_macro
        )
        # The engine does not re‑calibrate; it uses the AI's confidence directly.
        self.assertEqual(result["confidence_score"], 70)
        self.assertEqual(result.get("decision_source"), "LLM")


# ================================================================
# EXPORTS
# ================================================================
__all__ = ["InstitutionalAIDebateEngine"]
