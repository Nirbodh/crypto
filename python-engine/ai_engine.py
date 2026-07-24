# python-engine/ai_engine.py

import time
import json
import os
import re
import requests
from google import genai
from google.genai import types
from openai import OpenAI

class AIDebateEngine:
    def __init__(self, gemini_key: str = None, openai_key: str = None, deepseek_key: str = None):
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.deepseek_key = deepseek_key or os.getenv("DEEPSEEK_API_KEY")

        self.gemini_model = "gemini-2.5-flash"
        self.openai_model = "gpt-4o-mini"
        self.deepseek_model = "deepseek-chat"

    def _build_system_prompt(self) -> str:
        return (
            "You are the Chief Investment Officer (CIO) leading a 4-Agent Quantitative Crypto Trading Committee:\n\n"
            "1. [BULLISH SCALPER]: Looks for high-probability momentum, HTF alignment, and volume spikes.\n"
            "2. [RISK AUDITOR]: Looks for traps, weak RR, overbought conditions, and reasons NOT to trade.\n"
            "3. [MANIPULATION DETECTOR]: Looks for whale dumps, fake breakouts, funding traps, and long squeeze risks.\n"
            "4. [CHIEF ARBITER]: Weighs arguments, enforces risk compliance, and issues the FINAL DECISION.\n\n"
            "CRITICAL DIRECTIVES:\n"
            "- Base decisions STRICTLY on the provided quantitative JSON payload.\n"
            "- If Quant Risk Gatekeeper failed (is_passed: false), you MUST REJECT the trade regardless of technical scores.\n"
            "- Outputs MUST be structured, crisp, and objective."
        )

    def _build_user_prompt(self, payload_json: dict) -> str:
        formatted_json = json.dumps(payload_json, indent=2)
        price = payload_json.get('price', 'N/A')
        risk_mgmt = payload_json.get('risk_management', {})
        trade_levels = risk_mgmt.get('trade_levels', {})
        pos_sizing = risk_mgmt.get('position_sizing', {})

        sl_price = trade_levels.get('stop_loss_price', 'N/A')
        tp_price = trade_levels.get('take_profit_price', 'N/A')
        qty = pos_sizing.get('quantity', 'N/A')
        val_usdt = pos_sizing.get('position_value_usdt', 'N/A')

        return (
            f"Analyze this trade candidate from our Quantitative Engine:\n\n"
            f"```json\n{formatted_json}\n```\n\n"
            f"Perform the 4-Agent Debate and provide output strictly in the following format:\n\n"
            f"---\n"
            f"### 🐂 1. BULLISH SCALPER PERSPECTIVE\n"
            f"- **Key Strengths:** [List 2 main bullish drivers]\n"
            f"- **Target Thesis:** [Why this entry can succeed]\n\n"
            f"### 🐻 2. RISK AUDITOR PERSPECTIVE\n"
            f"- **Key Risks:** [List 2 main failure points or weak RR issues]\n"
            f"- **Counter Thesis:** [Why this setup might fail]\n\n"
            f"### 🕵️ 3. MANIPULATION DETECTOR PERSPECTIVE\n"
            f"- **Liquidity Traps:** [Check for fakeouts, liquidity sweeps, or funding traps]\n"
            f"- **Whale Warning:** [Long/Short squeeze risk or OI divergences]\n\n"
            f"### ⚖️ 4. CHIEF ARBITER FINAL CONSENSUS\n"
            f"- **Final Action:** [MUST BE ONE OF: \"EXECUTE_LONG\" | \"WAIT_FOR_DIP\" | \"REJECT\"]\n"
            f"- **Confidence Score:** [0-100 as integer]\n"
            f"- **Execution Plan:**\n"
            f"  - **Entry Price:** ${price}\n"
            f"  - **Stop-Loss:** ${sl_price}\n"
            f"  - **Take-Profit:** ${tp_price}\n"
            f"  - **Position Size:** {qty} tokens (${val_usdt} USDT)\n"
            f"- **Arbitration Reason:** [2-3 sentences explaining final decision]\n"
            f"---\n\n"
            f"IMPORTANT: Provide raw JSON after '---': "
            f'{{"final_decision": "...", "confidence": 0, "summary": "..."}}'
        )

    # -------------------------------------------------------------
    # 🤖 Individual Provider Call Handlers (With Failover Protection)
    # -------------------------------------------------------------
    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        if not self.gemini_key:
            return None
        try:
            client = genai.Client(api_key=self.gemini_key)
            response = client.models.generate_content(
                model=self.gemini_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                )
            )
            return response.text
        except Exception as e:
            print(f"⚠️ [Gemini Bypass - Quota/Key Error]: {e}")
            return None

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        if not self.openai_key:
            return None
        try:
            client = OpenAI(api_key=self.openai_key)
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ [OpenAI Bypass - Quota/Key Error]: {e}")
            return None

    def _call_deepseek(self, system_prompt: str, user_prompt: str) -> str:
        if not self.deepseek_key:
            return None
        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.deepseek_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1000
            }
            res = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data, timeout=20)
            if res.status_code == 200:
                return res.json()['choices'][0]['message']['content']
            else:
                print(f"⚠️ [DeepSeek Bypass - API Status {res.status_code}]: {res.text}")
                return None
        except Exception as e:
            print(f"⚠️ [DeepSeek Bypass - Quota/Key Error]: {e}")
            return None

    # -------------------------------------------------------------
    # 🥊 Multi-Provider Fallback Orchestrator
    # -------------------------------------------------------------
    def run_debate(self, payload_json: dict) -> dict:
        symbol = payload_json.get("symbol", "UNKNOWN")
        gatekeeper_passed = payload_json.get("gatekeeper_passed", True)

        # 🛑 HARD RISK GUARD: If Quant Gatekeeper REJECTS, AI is completely bypassed!
        if not gatekeeper_passed:
            reasons = payload_json.get("rejection_reasons", ["Failed Quant Risk Gatekeeper Rules"])
            summary_msg = f"Trade Rejected by Quant Gatekeeper: {', '.join(reasons)}"
            return {
                "success": True,
                "symbol": symbol,
                "analysis": f"❌ **QUANT GATEKEEPER HARD BLOCK**\n\nTrade candidate failed quantitative risk checks. Reasons: {reasons}. AI Execution cancelled.",
                "final_decision": "REJECT",
                "confidence": 0,
                "summary": summary_msg
            }

        user_prompt = self._build_user_prompt(payload_json)
        system_instruction = self._build_system_prompt()

        # Fallback Sequence: Gemini -> OpenAI -> DeepSeek
        providers = [
            ("Gemini", self._call_gemini),
            ("OpenAI", self._call_openai),
            ("DeepSeek", self._call_deepseek)
        ]

        raw_text = None
        active_provider = None

        for name, provider_func in providers:
            print(f"🤖 Requesting AI Debate from [{name}]...")
            raw_text = provider_func(system_instruction, user_prompt)
            if raw_text:
                active_provider = name
                print(f"✅ Successfully executed AI Debate via [{name}].")
                break

        # If all LLM APIs fail or run out of quota
        if not raw_text:
            return {
                "success": False,
                "symbol": symbol,
                "error": "All AI Provider APIs (Gemini, OpenAI, DeepSeek) were bypassed or exhausted.",
                "final_decision": "HOLD",
                "confidence": 0,
                "summary": "All AI APIs skipped due to missing keys or quota limits. Defaulting to HOLD."
            }

        # Parse the successfully retrieved response
        parsed = self._parse_ai_response(raw_text, symbol)

        return {
            "success": True,
            "symbol": symbol,
            "provider_used": active_provider,
            "analysis": raw_text,
            "final_decision": parsed["final_decision"],
            "confidence": parsed["confidence"],
            "summary": parsed["summary"]
        }

    def _parse_ai_response(self, text: str, symbol: str, default_score: int = 70) -> dict:
        try:
            # Enhanced Multi-line JSON block regex search
            json_match = re.search(r'\{[\s\S]*?"final_decision"[\s\S]*?\}', text)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return {
                    "final_decision": parsed.get("final_decision", "HOLD"),
                    "confidence": int(parsed.get("confidence", default_score)),
                    "summary": parsed.get("summary", "AI analysis complete.")
                }
        except Exception:
            pass

        # Fallback text parsing if JSON block is missing or malformed
        decision = "HOLD"
        if "EXECUTE_LONG" in text:
            decision = "EXECUTE_LONG"
        elif "WAIT_FOR_DIP" in text:
            decision = "WAIT_FOR_DIP"
        elif "REJECT" in text:
            decision = "REJECT"

        conf_match = re.search(r'Confidence Score.*?(\d+)', text, re.IGNORECASE)
        confidence = int(conf_match.group(1)) if conf_match else default_score
        
        summary_match = re.search(r'Arbitration Reason:\s*(.*?)(?:\n---|$)', text, re.DOTALL | re.IGNORECASE)
        summary = summary_match.group(1).strip() if summary_match else text[-200:].strip()

        return {
            "final_decision": decision,
            "confidence": confidence,
            "summary": summary[:300]
        }