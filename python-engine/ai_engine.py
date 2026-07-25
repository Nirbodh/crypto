# python-engine/ai_engine.py
# Institutional Wrapper for AIDebateEngine - v2.0
# Architecture: AI acts as an Auditor/Explainer, not a primary decision maker.
# Flow: Market Data → Quant Engine → Risk Engine → Execution Engine → AI Debate (Audit & Explain) → Final Output

import os
import json
import time
import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Union

from ai_debate_engine import InstitutionalAIDebateEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ================================================================
# 1. DECISION ENUM (Type Safety)
# ================================================================
class Decision(Enum):
    EXECUTE_LONG = "EXECUTE_LONG"
    EXECUTE_SHORT = "EXECUTE_SHORT"
    WATCHLIST = "WATCHLIST"
    REJECT = "REJECT"
    HOLD = "HOLD"


# ================================================================
# 2. MAIN WRAPPER CLASS
# ================================================================
class AIDebateEngine:
    """
    Legacy compatibility wrapper with enhanced validation, calibration, and observability.
    Now acts as an institutional-grade auditor/explainer rather than a primary decision maker.
    """

    ENGINE_VERSION = "2.4"
    PROMPT_VERSION = "v3"
    QUANT_VERSION = "4.1"
    RISK_VERSION = "3.0"

    def __init__(
        self,
        gemini_key: str = None,
        openai_key: str = None,
        deepseek_key: str = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self._engine = InstitutionalAIDebateEngine(
            gemini_key=gemini_key,
            openai_key=openai_key,
            deepseek_key=deepseek_key,
            prompt_version=self.PROMPT_VERSION,
            config=config or {}
        )
        # Legacy attributes (kept for compatibility)
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.deepseek_key = deepseek_key or os.getenv("DEEPSEEK_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        # Provider metrics store
        self._provider_metrics = {}

    # ================================================================
    # 3. VALIDATION & SAFE GET
    # ================================================================

    @staticmethod
    def _safe_get(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
        """Safely retrieve a value using multiple possible keys."""
        for key in keys:
            if key in data:
                return data[key]
        return default

    def _validate_payload(self, payload: Dict[str, Any]) -> None:
        """Ensure all required fields are present."""
        required = [
            "symbol",
            "unified_score",
            "ev_r"
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            # Try alternative names
            alt_map = {
                "unified_score": ["score", "mtf_score", "unified_score"],
                "ev_r": ["ev_r", "expected_value", "rr_score_raw"]
            }
            still_missing = []
            for key in missing:
                if key in alt_map:
                    if not any(alt in payload for alt in alt_map[key]):
                        still_missing.append(key)
                else:
                    still_missing.append(key)
            if still_missing:
                raise ValueError(f"Missing required field(s): {still_missing}")

    # ================================================================
    # 4. PAYLOAD MAPPING & NORMALIZATION
    # ================================================================

    def _map_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and map legacy payload to new engine format with validation."""
        # Validate first
        self._validate_payload(payload)

        # Core fields with fallback
        symbol = self._safe_get(payload, ["symbol"], "UNKNOWN")

        # Quant fusion
        unified_score = self._safe_get(
            payload,
            ["unified_score", "score", "mtf_score"],
            80.0
        )
        ev_r = self._safe_get(
            payload,
            ["ev_r", "expected_value", "rr_score_raw"],
            1.5
        )
        is_passed = self._safe_get(
            payload,
            ["is_passed", "gatekeeper_passed"],
            True
        )

        # Score breakdown normalization
        breakdown = payload.get("score_breakdown", {})
        # Map 'volume' to 'liquidity' if liquidity missing
        if "volume" in breakdown and "liquidity" not in breakdown:
            breakdown["liquidity"] = breakdown["volume"]

        # Normalize key names to match what new engine expects
        normalized_breakdown = {
            "technical": breakdown.get("technical", 50),
            "smc": breakdown.get("smc", 50),
            "liquidity": breakdown.get("liquidity", 50),
            "risk": breakdown.get("risk", 50),
            "mtf": breakdown.get("mtf", 50),
            "derivatives": breakdown.get("derivatives", 50),
            "volume": breakdown.get("volume", 50),  # keep for reference
        }

        quant_fusion = {
            "is_passed": is_passed,
            "unified_score": float(unified_score),
            "ev_r": float(ev_r),
            "score_breakdown": normalized_breakdown,
            "derivatives_data": payload.get("derivatives_data", {}),
            "rejection_reasons": payload.get("rejection_reasons", [])
        }

        # Risk data
        risk_data = payload.get("risk_management", payload.get("risk_data", {}))
        if not risk_data:
            risk_data = {
                "direction": self._safe_get(payload, ["direction"], "LONG"),
                "valid_trade": self._safe_get(payload, ["valid_trade"], True),
                "trade_levels": payload.get("trade_levels", {}),
                "position_sizing": payload.get("position_sizing", {}),
                "risk_metrics": payload.get("risk_metrics", {})
            }

        # Macro / regime
        btc_macro = {
            "is_bullish": self._safe_get(
                payload,
                ["btc_bullish", "btc_is_bullish"],
                True
            ),
            "regime": self._safe_get(
                payload,
                ["market_regime", "regime"],
                "TRENDING"
            )
        }

        # News / memory
        news_event = payload.get("news_event", {"has_high_impact_news": False})
        trade_memory = payload.get("trade_memory", [])

        return {
            "symbol": symbol,
            "quant_fusion": quant_fusion,
            "risk_data": risk_data,
            "btc_macro": btc_macro,
            "news_event": news_event,
            "trade_memory": trade_memory
        }

    # ================================================================
    # 5. CALIBRATED CONFIDENCE (Institutional Grade)
    # ================================================================

    @staticmethod
    def _calculate_calibrated_confidence(
        quant_score: float,
        ev_r: float,
        agreement_pct: float,
        regime: str,
        base_confidence: int
    ) -> int:
        """
        Institutional confidence calibration:
        40% Quant Score, 20% EV, 20% Agreement, 20% Market Regime.
        """
        # Normalize quant score (0-100)
        q_score = max(0, min(100, quant_score)) / 100.0

        # Normalize EV (capped at 3.0 for scaling)
        ev_score = min(1.0, ev_r / 3.0)

        # Agreement (already 0-100)
        agree_score = max(0, min(100, agreement_pct)) / 100.0

        # Regime factor
        regime_boost = {
            "TRENDING": 1.0,
            "RANGING": 0.85,
            "VOLATILE": 0.80,
            "EXPANSION": 1.1,
            "COMPRESSION": 0.70,
            "DISTRIBUTION": 0.90,
            "ACCUMULATION": 1.05,
            "BEAR": 0.75,
            "CRASH": 0.60
        }.get(regime.upper(), 1.0)

        # Weighted score (0-1)
        weighted = (
            q_score * 0.40 +
            ev_score * 0.20 +
            agree_score * 0.20 +
            regime_boost * 0.20
        )

        # Scale to percentage and blend with base confidence
        calibrated = int(weighted * 100)
        # Blend with base confidence (so we don't ignore AI's own confidence)
        final = int(0.6 * calibrated + 0.4 * base_confidence)
        return max(10, min(95, final))

    # ================================================================
    # 6. TELEMETRY & METADATA
    # ================================================================

    def _build_telemetry(
        self,
        provider_name: str,
        start_time: float,
        llm_used: bool,
        fallback_reason: Optional[str] = None,
        token_input: Optional[int] = None,
        token_output: Optional[int] = None
    ) -> Dict[str, Any]:
        """Rich telemetry for production debugging."""
        return {
            "provider": provider_name or "Unknown",
            "latency_ms": int((time.time() - start_time) * 1000),
            "fallback_used": not llm_used,
            "fallback_reason": fallback_reason,
            "token_input": token_input or 0,
            "token_output": token_output or 0,
            "prompt_version": self.PROMPT_VERSION,
            "engine_version": self.ENGINE_VERSION,
            "quant_version": self.QUANT_VERSION,
            "risk_version": self.RISK_VERSION,
            "timestamp": time.time()
        }

    def _build_metadata(self) -> Dict[str, Any]:
        """Engine metadata for versioning."""
        return {
            "engine": "InstitutionalAI",
            "version": self.ENGINE_VERSION,
            "prompt": self.PROMPT_VERSION,
            "quant": self.QUANT_VERSION,
            "risk": self.RISK_VERSION,
            "execution": "v2.0"
        }

    # ================================================================
    # 7. PROVIDER METRICS STORE
    # ================================================================

    def _record_provider_metrics(self, provider: str, latency_ms: int, retry_count: int = 0) -> None:
        """Keep track of provider performance."""
        if provider not in self._provider_metrics:
            self._provider_metrics[provider] = {"calls": 0, "total_latency": 0, "avg_latency": 0, "retries": 0}
        stats = self._provider_metrics[provider]
        stats["calls"] += 1
        stats["total_latency"] += latency_ms
        stats["avg_latency"] = stats["total_latency"] / stats["calls"]
        stats["retries"] += retry_count

    # ================================================================
    # 8. MAIN METHOD: run_debate
    # ================================================================

    def run_debate(self, payload_json: dict) -> dict:
        """
        Enhanced legacy run_debate with institutional-grade validation, calibration,
        and rich output schema.
        """
        start_time = time.time()
        provider_used = "Fallback"
        llm_used = False
        token_input = 0
        token_output = 0
        retry_count = 0

        try:
            # ---- 1. Normalize payload ----
            mapped = self._map_payload(payload_json)

            # ---- 2. Call new engine ----
            result = self._engine.run_debate_and_decide(
                symbol=mapped["symbol"],
                quant_fusion_data=mapped["quant_fusion"],
                risk_data=mapped["risk_data"],
                btc_macro_data=mapped["btc_macro"],
                recent_trade_memory=mapped["trade_memory"],
                news_event_data=mapped["news_event"]
            )

            # ---- 3. Extract metrics ----
            provider_used = result.get("provider_used", "Fallback")
            llm_used = result.get("telemetry", {}).get("llm_used", False)
            fallback_reason = result.get("telemetry", {}).get("fallback_reason")
            latency_ms = result.get("telemetry", {}).get("latency_sec", 0) * 1000
            # Track provider performance
            self._record_provider_metrics(provider_used, int(latency_ms), 0)

            # ---- 4. Calibrated Confidence ----
            quant_score = mapped["quant_fusion"].get("unified_score", 80.0)
            ev_r = mapped["quant_fusion"].get("ev_r", 1.5)
            agreement = result.get("agreement_pct", 100.0)
            regime = mapped["btc_macro"].get("regime", "TRENDING")
            base_confidence = result.get("confidence_score", 70)

            calibrated_conf = self._calculate_calibrated_confidence(
                quant_score=quant_score,
                ev_r=ev_r,
                agreement_pct=agreement,
                regime=regime,
                base_confidence=base_confidence
            )

            # ---- 5. Position Quality Grade ----
            def grade_from_score(score: int) -> str:
                if score >= 95:
                    return "A+"
                elif score >= 90:
                    return "A"
                elif score >= 85:
                    return "B+"
                elif score >= 80:
                    return "B"
                elif score >= 75:
                    return "C"
                else:
                    return "D"

            grade = grade_from_score(calibrated_conf)

            # ---- 6. Build final result with strong schema ----
            final_result = {
                # Core
                "success": result.get("success", True),
                "symbol": result.get("symbol", mapped["symbol"]),
                "provider_used": provider_used,
                "final_decision": result.get("final_decision", Decision.REJECT.value),
                "confidence_score": calibrated_conf,
                "conviction_score": result.get("conviction_score", calibrated_conf),
                "agreement_pct": result.get("agreement_pct", 0.0),
                "final_grade": grade,
                "execution_plan": result.get("execution_plan", {}),
                "ai_votes": result.get("ai_votes", {}),
                "telemetry": self._build_telemetry(
                    provider_name=provider_used,
                    start_time=start_time,
                    llm_used=llm_used,
                    fallback_reason=fallback_reason,
                    token_input=token_input,
                    token_output=token_output
                ),
                "reasons": result.get("reasons", []),
                "risks": result.get("risks", []),
                "summary": result.get("summary", ""),
                "timestamp": time.time(),
                "engine_version": self.ENGINE_VERSION,
                # Extra: debate history
                "debate_history": {
                    "bull": result.get("ai_votes", {}).get("bull_ai", "REJECT"),
                    "bear": result.get("ai_votes", {}).get("bear_ai", "REJECT"),
                    "manipulation": result.get("ai_votes", {}).get("manipulation_ai", "REJECT"),
                    "cio": result.get("ai_votes", {}).get("cio_ai", "REJECT")
                },
                "market_regime": {
                    "type": mapped["btc_macro"].get("regime", "UNKNOWN"),
                    "strength": min(100, int(quant_score * 0.8 + 20)),
                    "volatility": "HIGH" if mapped["btc_macro"].get("regime") in ["VOLATILE", "EXPANSION", "BEAR", "CRASH"] else "MEDIUM",
                    "trend_quality": "HIGH" if mapped["btc_macro"].get("regime") in ["TRENDING", "ACCUMULATION"] else "LOW"
                },
                "metadata": self._build_metadata()
            }

            # If new engine returned a hard rejection, ensure we capture that
            if not result.get("success", True) and "error" in result:
                final_result["error"] = result["error"]
                final_result["summary"] = result.get("error", "Unknown error")
                final_result["final_decision"] = Decision.REJECT.value
                final_result["confidence_score"] = 0

            return final_result

        except Exception as e:
            logging.error(f"❌ AIDebateEngine wrapper crashed: {e}")
            return {
                "success": False,
                "symbol": payload_json.get("symbol", "UNKNOWN"),
                "final_decision": Decision.REJECT.value,
                "confidence_score": 0,
                "conviction_score": 0,
                "agreement_pct": 0.0,
                "final_grade": "REJECT",
                "summary": f"Wrapper error: {str(e)}",
                "reasons": ["Engine wrapper crashed"],
                "risks": ["System error"],
                "timestamp": time.time(),
                "engine_version": self.ENGINE_VERSION,
                "error": str(e)
            }

    # ================================================================
    # 9. LEGACY METHODS (Deprecated, kept for compatibility)
    # ================================================================

    def _build_system_prompt(self) -> str:
        logging.warning("_build_system_prompt is deprecated. Use run_debate.")
        return "You are an Institutional Crypto Trading Committee."

    def _build_user_prompt(self, payload_json: dict) -> str:
        logging.warning("_build_user_prompt is deprecated. Use run_debate.")
        return "Legacy method - use run_debate."

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        logging.warning("_call_gemini is deprecated. Use run_debate.")
        return None

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        logging.warning("_call_openai is deprecated. Use run_debate.")
        return None

    def _call_deepseek(self, system_prompt: str, user_prompt: str) -> str:
        logging.warning("_call_deepseek is deprecated. Use run_debate.")
        return None

    def _parse_ai_response(self, text: str, symbol: str, default_score: int = 70) -> dict:
        logging.warning("_parse_ai_response is deprecated. Use run_debate.")
        return {"final_decision": Decision.HOLD.value, "confidence": default_score, "summary": "Parsing disabled."}


# ================================================================
# 10. EXPORTS
# ================================================================
__all__ = ["AIDebateEngine", "InstitutionalAIDebateEngine", "Decision"]