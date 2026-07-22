# python-engine/ai_engine.py
import time
import json
import os
import re
from google import genai
from google.genai import types

class AIDebateEngine:
    def __init__(self, api_key: str = None):
        """
        Gemini Client Initialization using google-genai SDK
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("⚠️ Gemini API Key not found! Set GEMINI_API_KEY environment variable.")
            
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"

    def _build_system_prompt(self) -> str:
        return (
            "You are the Chief Investment Officer (CIO) leading a 3-Agent Quantitative Crypto Trading Committee:\n\n"
            "1. [BULLISH SCALPER]: Looks for high-probability momentum, HTF alignment, and volume spikes.\n"
            "2. [RISK AUDITOR]: Looks for traps, weak RR, overbought conditions, and reasons NOT to trade.\n"
            "3. [CHIEF ARBITER]: Weighs arguments, enforces risk compliance, and issues the FINAL DECISION.\n\n"
            "CRITICAL DIRECTIVES:\n"
            "- Base decisions STRICTLY on the provided quantitative JSON payload.\n"
            "- Enforce Stop-Loss & Take-Profit levels from the Risk Engine.\n"
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
            f"Perform the 3-Agent Debate and provide output strictly in the following format:\n\n"
            f"---\n"
            f"### 🐂 1. BULLISH SCALPER PERSPECTIVE\n"
            f"- **Key Strengths:** [List 2 main bullish drivers from HTF/LTF/Volume]\n"
            f"- **Target Thesis:** [Why this entry can succeed]\n\n"
            f"### 🐻 2. RISK AUDITOR PERSPECTIVE\n"
            f"- **Key Risks:** [List 2 main failure points, volume concerns, or market risks]\n"
            f"- **Counter Thesis:** [Why this setup might fail or trap buyers]\n\n"
            f"### ⚖️ 3. CHIEF ARBITER FINAL CONSENSUS\n"
            f"- **Final Action:** [MUST BE ONE OF: \"EXECUTE_LONG\" | \"WAIT_FOR_DIP\" | \"REJECT\"]\n"
            f"- **Confidence Score:** [0-100 as integer, e.g., 85]\n"
            f"- **Execution Plan:**\n"
            f"  - **Entry Price:** ${price}\n"
            f"  - **Stop-Loss:** ${sl_price}\n"
            f"  - **Take-Profit:** ${tp_price}\n"
            f"  - **Position Size:** {qty} tokens (${val_usdt} USDT)\n"
            f"- **Arbitration Reason:** [2-3 sentences explaining the final decision]\n"
            f"---\n\n"
            f"IMPORTANT: At the end of your response, provide a valid JSON object with the following keys: "
            f"final_decision (string: EXECUTE_LONG or WAIT_FOR_DIP or REJECT), "
            f"confidence (integer 0-100), "
            f"summary (short string). "
            f"Do NOT wrap JSON in markdown code fence. Just output the raw JSON after the '---' section."
        )

    def _parse_ai_response(self, text: str, symbol: str, default_score: int = 70) -> dict:
        """
        পার্স করে JSON বের করার চেষ্টা করে; না পেলে রেজেক্স দিয়ে এক্সট্র্যাক্ট করে।
        """
        # 1. Try to extract JSON from the text
        try:
            # Find JSON-like content between { and }
            json_match = re.search(r'\{[^{}]*"final_decision"[^{}]*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return {
                    "final_decision": parsed.get("final_decision", "HOLD"),
                    "confidence": int(parsed.get("confidence", default_score)),
                    "summary": parsed.get("summary", "AI analysis complete.")
                }
        except:
            pass

        # 2. Fallback: regex extraction from plain text
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

    def run_debate(self, payload_json: dict) -> dict:
        """
        AI Multi-Agent debate using Gemini with Retry Mechanism.
        Returns dict with success, symbol, analysis, plus parsed final_decision, confidence, summary.
        """
        max_retries = 3
        retry_delay = 2
        symbol = payload_json.get("symbol", "UNKNOWN")

        for attempt in range(1, max_retries + 1):
            try:
                user_prompt = self._build_user_prompt(payload_json)
                system_instruction = self._build_system_prompt()

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                    )
                )
                raw_text = response.text

                # Parse the response to extract decision, confidence, summary
                parsed = self._parse_ai_response(raw_text, symbol)

                return {
                    "success": True,
                    "symbol": symbol,
                    "analysis": raw_text,
                    "final_decision": parsed["final_decision"],
                    "confidence": parsed["confidence"],
                    "summary": parsed["summary"]
                }

            except Exception as e:
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                    continue
                # Final attempt failed
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": str(e),
                    "final_decision": "HOLD",
                    "confidence": 50,
                    "summary": "AI Engine error. Defaulting to HOLD."
                }

if __name__ == "__main__":
    print("✅ AIDebateEngine module loaded successfully!")