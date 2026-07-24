# python-engine/notifier.py

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TelegramNotifier:
    def __init__(self, bot_token: str = None, chat_id: str = None):
        raw_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        raw_chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        
        self.bot_token = raw_token.strip() if raw_token else None
        self.chat_id = raw_chat_id.strip() if raw_chat_id else None

    def send_message(self, text: str) -> bool:
        """
        Generic method to send raw HTML/text messages to Telegram.
        """
        if not self.bot_token or not self.chat_id:
            logging.warning("⚠️ Telegram Credentials missing. Skipping Telegram Alert.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
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
            logging.error(f"❌ Telegram Notification Failed: {e}")
            return False

    def send_trade_signal(self, symbol: str, exchange: str, ai_output: str, parsed_plan: dict) -> bool:
        """
        Sends formatted Trade Signal to Telegram Channel / Bot using robust HTML formatting.
        """
        if not self.bot_token or not self.chat_id:
            logging.warning("⚠️ Telegram Credentials missing. Skipping Telegram Alert.")
            return False

        # Safe values extraction
        action = parsed_plan.get('action', 'EXECUTE_LONG')
        confidence = parsed_plan.get('confidence', '80')
        entry = parsed_plan.get('entry_price', 'N/A')
        sl = parsed_plan.get('stop_loss', 'N/A')
        tp = parsed_plan.get('take_profit', 'N/A')
        qty = parsed_plan.get('position_qty', 'N/A')
        usdt = parsed_plan.get('position_usdt', 'N/A')
        reason = parsed_plan.get('reason', 'High probability setup verified.')

        # HTML Formatting
        message = f"""🚀 <b>NEW QUANT AI TRADE SIGNAL</b> 🚀
----------------------------------
📌 <b>Symbol:</b> {symbol} ({exchange.upper()})
🎯 <b>Action:</b> {action}
🔥 <b>Confidence:</b> {confidence}%

📐 <b>EXECUTION PLAN:</b>
• <b>Entry:</b> ${entry}
• <b>Stop Loss:</b> ${sl}
• <b>Take Profit:</b> ${tp}
• <b>Position Size:</b> {qty} tokens (~${usdt} USDT)

🤖 <b>AI ARBITRATION SUMMARY:</b>
{reason}

----------------------------------
⚡ <i>Automated Quant AI Pipeline</i>
"""
        return self.send_message(message)


if __name__ == "__main__":
    print("==================================================")
    print("📡 TESTING TELEGRAM NOTIFIER MODULE...")
    print("==================================================")
    
    notifier = TelegramNotifier()
    
    # Mock Data for Signal Test
    mock_plan = {
        "action": "EXECUTE_LONG",
        "confidence": 88,
        "entry_price": 64825.05,
        "stop_loss": 64200.00,
        "take_profit": 66100.00,
        "position_qty": 0.015,
        "position_usdt": 972.37,
        "reason": "Confluence of 4H Order Block + 1H Bullish FVG Sweep."
    }

    print("🔄 Attempting test message transmission...")
    success = notifier.send_trade_signal(
        symbol="BTC/USDT", 
        exchange="Binance", 
        ai_output="", 
        parsed_plan=mock_plan
    )
    
    if not success:
        print("💡 NOTE: If you haven't added TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env file, this warning is expected.")