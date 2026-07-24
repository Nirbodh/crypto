# python-engine/telegram_notifier.py

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TelegramNotifier:
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initializes Telegram Notifier.
        Fallback to .env variables if explicit parameters are not provided.
        """
        raw_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        raw_chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

        self.bot_token = raw_token.strip() if raw_token else None
        self.chat_id = raw_chat_id.strip() if raw_chat_id else None

        if not self.bot_token or not self.chat_id:
            logging.warning("⚠️ Telegram Credentials missing in .env or arguments. Telegram alerts will be skipped.")

    def send_message(self, text: str) -> bool:
        """
        Generic method to send raw HTML/text messages to Telegram.
        """
        if not self.bot_token or not self.chat_id:
            logging.warning("⚠️ Telegram Credentials missing. Skipping message transmission.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }

        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                logging.info("📲 Telegram Message successfully sent!")
                return True
            else:
                logging.error(f"❌ Telegram API Error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logging.error(f"❌ Telegram Notification Exception: {e}")
            return False

    def send_institutional_signal(self, signal_data: dict) -> bool:
        """
        Legacy/Detailed institutional signal formatter.
        """
        symbol = signal_data.get("symbol", "UNKNOWN")
        bias = signal_data.get("bias", "NEUTRAL")
        score = signal_data.get("score", 0.0)
        ev_r = signal_data.get("ev_r", 0.0)
        entry = signal_data.get("entry", 0.0)
        sl = signal_data.get("sl", 0.0)
        tp = signal_data.get("tp", 0.0)
        reasons = signal_data.get("reasons", [])

        reasons_formatted = "\n".join([f"• {r}" for r in reasons]) if reasons else "• High probability setup verified."

        message = f"""🚀 <b>INSTITUTIONAL QUANT SIGNAL</b> 🚀
----------------------------------
📌 <b>Symbol:</b> {symbol}
🎯 <b>Bias / Action:</b> {bias}
🔥 <b>Unified Score:</b> {score:.1f}/100 | <b>EV:</b> {ev_r}R

📐 <b>EXECUTION LEVELS:</b>
• <b>Entry:</b> ${entry}
• <b>Stop Loss:</b> ${sl}
• <b>Take Profit:</b> ${tp}

🤖 <b>KEY CATALYSTS & REASONING:</b>
{reasons_formatted}

----------------------------------
⚡ <i>Automated Quant AI Pipeline</i>
"""
        return self.send_message(message)

    def send_trade_signal(
        self,
        symbol: str,
        decision: str,
        confidence: int,
        score: float,
        ev_r: float,
        entry: float,
        sl: float,
        tp: float,
        summary: str
    ) -> bool:
        """
        Adapter method specifically matching the parameters called in main_engine.py
        """
        signal_data = {
            "symbol": symbol,
            "bias": f"{decision} (Confidence: {confidence}%)",
            "score": score,
            "ev_r": ev_r,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "reasons": [summary]
        }

        return self.send_institutional_signal(signal_data)


if __name__ == "__main__":
    print("==================================================")
    print("📡 TESTING TELEGRAM NOTIFIER ADAPTER...")
    print("==================================================")

    notifier = TelegramNotifier()

    # Direct Test Call matching main_engine signature
    success = notifier.send_trade_signal(
        symbol="BTC/USDT",
        decision="EXECUTE_LONG",
        confidence=85,
        score=84.5,
        ev_r=2.5,
        entry=64825.50,
        sl=63500.00,
        tp=67500.00,
        summary="Confluence of 4H Order Block Sweep and Bullish FVG."
    )

    if not success:
        print("\n💡 NOTE: To test actual Telegram message delivery, set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file.")