# python-engine/telegram_notifier.py

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()


class TelegramNotifier:

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")


    def send_trade_alert(self, payload: dict, ai_verdict: dict):

        if not self.bot_token or not self.chat_id:
            print("⚠️ Telegram credentials missing in .env!")
            return


        symbol = payload.get("symbol", "N/A")

        unified_score = (
            payload.get("score", 0)
            or payload.get("unified_score", 0)
        )


        # ==============================
        # Scores
        # ==============================

        tech = payload.get("technical_score", 0)

        if not tech:
            tech = payload.get("technical", {}).get(
                "technical_score",
                0
            )

        deriv = payload.get("derivatives", {}).get(
            "derivatives_score",
            0
        )

        fund = payload.get("fundamental", {}).get(
            "fundamental_score",
            0
        )

        sent = payload.get("sentiment", {}).get(
            "sentiment_score",
            0
        )


        # ==============================
        # Risk Management FIX
        # ==============================

        risk = payload.get(
            "risk_management",
            {}
        )

        trade_levels = risk.get(
            "trade_levels",
            {}
        )


        entry = (
            risk.get("entry_price")
            or trade_levels.get("entry_price")
            or trade_levels.get("entry")
            or payload.get("price")
            or 0.0
        )


        sl = (
            risk.get("stop_loss")
            or trade_levels.get("stop_loss_price")
            or trade_levels.get("stop_loss")
            or 0.0
        )


        tp = (
            risk.get("take_profit_1")
            or trade_levels.get("take_profit_price")
            or trade_levels.get("take_profit")
            or 0.0
        )


        # ==============================
        # AI Response Parser
        # ==============================

        analysis_text = ai_verdict.get(
            "analysis",
            ""
        )


        decision = "HOLD"
        confidence = 0
        summary = "Analysis complete."


        if "EXECUTE_LONG" in analysis_text:
            decision = "LONG"

        elif "WAIT_FOR_DIP" in analysis_text:
            decision = "HOLD"

        elif "REJECT" in analysis_text:
            decision = "HOLD"



        conf_match = re.search(
            r"Confidence Score.*?(\d+)",
            analysis_text,
            re.IGNORECASE | re.DOTALL
        )

        if conf_match:
            confidence = int(
                conf_match.group(1)
            )



        reason_match = re.search(
            r"Arbitration Reason:\s*(.*?)(?:\n---|$)",
            analysis_text,
            re.DOTALL | re.IGNORECASE
        )


        if reason_match:
            summary = reason_match.group(1).strip()

        else:
            summary = analysis_text[-200:].strip()



        # ==============================
        # Telegram Message
        # ==============================

        emoji_map = {
            "LONG": "🟢 LONG",
            "SHORT": "🔴 SHORT",
            "HOLD": "🟡 HOLD"
        }


        decision_str = emoji_map.get(
            decision,
            "🟡 HOLD"
        )


        message = (
            f"🚀 *QUANT CRYPTO AI - TRADE ALERT* 🚀\n\n"
            f"📌 *Pair:* {symbol}\n"
            f"🎯 *Unified Score:* {unified_score:.2f}/100\n"
            f"🤖 *AI Decision:* {decision_str} "
            f"(Confidence: {confidence}%)\n\n"

            f"📊 *Scores Breakdown:*\n"
            f" • Technical: {tech:.1f}/100\n"
            f" • Derivatives: {deriv:.1f}/100\n"
            f" • Fundamental: {fund:.1f}/100\n"
            f" • Sentiment: {sent:.1f}/100\n\n"

            f"💡 *Execution Plan:*\n"
            f" • Entry: {entry:.4f}\n"
            f" • Stop Loss: {sl:.4f}\n"
            f" • Take Profit: {tp:.4f}\n\n"

            f"📝 *AI Summary:* {summary}"
        )


        # ==============================
        # Send Telegram
        # ==============================

        url = (
            f"https://api.telegram.org/"
            f"bot{self.bot_token}/sendMessage"
        )


        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }


        try:

            res = requests.post(
                url,
                json=data,
                timeout=10
            )

            if res.status_code == 200:
                print(
                    f"✅ Telegram Alert Delivered for {symbol}"
                )

            else:
                print(
                    f"❌ Telegram API Error: {res.text}"
                )


        except Exception as e:

            print(
                f"❌ Failed to send Telegram alert: {e}"
            )