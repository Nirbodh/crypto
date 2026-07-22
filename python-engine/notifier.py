# python-engine/notifier.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    def __init__(self, bot_token: str = None, chat_id: str = None):
        raw_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        raw_chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        
        self.bot_token = raw_token.strip() if raw_token else None
        self.chat_id = raw_chat_id.strip() if raw_chat_id else None

    def send_trade_signal(self, symbol: str, exchange: str, ai_output: str, parsed_plan: dict):
        """
        Sends formatted Trade Signal to Telegram Channel / Bot using robust HTML formatting.
        """
        if not self.bot_token or not self.chat_id:
            print("⚠️ Telegram Credentials missing. Skipping Telegram Alert.")
            return False

        # HTML Formatting
        message = f"""🚀 <b>NEW QUANT AI TRADE SIGNAL</b> 🚀
----------------------------------
📌 <b>Symbol:</b> {symbol} ({exchange.upper()})
🎯 <b>Action:</b> {parsed_plan.get('action', 'EXECUTE_LONG')}
🔥 <b>Confidence:</b> {parsed_plan.get('confidence', '80')}%

📐 <b>EXECUTION PLAN:</b>
• <b>Entry:</b> ${parsed_plan.get('entry_price')}
• <b>Stop Loss:</b> ${parsed_plan.get('stop_loss')}
• <b>Take Profit:</b> ${parsed_plan.get('take_profit')}
• <b>Position Size:</b> {parsed_plan.get('position_qty')} tokens (~${parsed_plan.get('position_usdt')} USDT)

🤖 <b>AI ARBITRATION SUMMARY:</b>
{parsed_plan.get('reason', 'High probability setup verified.')}

----------------------------------
⚡ <i>Automated Quant AI Pipeline</i>
"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }

        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                print(f"📲 Telegram Alert successfully sent for {symbol}!")
                return True
            else:
                print(f"❌ Telegram API Error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Telegram Notification Failed: {e}")
            return False