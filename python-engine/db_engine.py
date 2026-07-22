# python-engine/db_engine.py
import sqlite3
import json
import os
from datetime import datetime

class DatabaseEngine:
    def __init__(self, db_path="quant_trading.db"):
        """
        Initializes the Database Engine. Defaulting to SQLite for zero-config simplicity.
        """
        self.db_path = db_path
        self._init_sqlite_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_sqlite_db(self):
        """
        Creates necessary tables for market scans and AI debate decisions.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # 1. Market Scans Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            score INTEGER,
            pass BOOLEAN,
            price REAL,
            atr_5m REAL,
            raw_payload TEXT
        )
        """)

        # 2. AI Trade Decisions Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_trade_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            final_action TEXT NOT NULL,
            confidence_score INTEGER,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            position_size_tokens REAL,
            position_value_usdt REAL,
            ai_debate_summary TEXT,
            status TEXT DEFAULT 'PENDING'
        )
        """)

        conn.commit()
        conn.close()

    def save_scan_result(self, payload: dict):
        """
        Saves raw screener scan data into the database.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            symbol = payload.get("symbol")
            exchange = payload.get("exchange", "BINANCE")
            score = payload.get("score", 0)
            passed = payload.get("pass", False)
            price = payload.get("price", 0.0)
            atr_5m = payload.get("atr_5m", 0.0)
            raw_payload = json.dumps(payload)

            cursor.execute("""
                INSERT INTO market_scans (symbol, exchange, score, pass, price, atr_5m, raw_payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, exchange, score, passed, price, atr_5m, raw_payload))

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"❌ DB Save Scan Error: {e}")

    def save_ai_decision(self, symbol: str, exchange: str, decision_data: dict, ai_text: str):
        """
        Saves AI Decision Engine consensus & execution plan.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            action = decision_data.get("action", "REJECT")
            confidence = decision_data.get("confidence", 0)
            entry = decision_data.get("entry_price", 0.0)
            sl = decision_data.get("stop_loss", 0.0)
            tp = decision_data.get("take_profit", 0.0)
            qty = decision_data.get("position_qty", 0.0)
            val_usdt = decision_data.get("position_usdt", 0.0)

            cursor.execute("""
                INSERT INTO ai_trade_decisions 
                (symbol, exchange, final_action, confidence_score, entry_price, stop_loss, take_profit, position_size_tokens, position_value_usdt, ai_debate_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, exchange, action, confidence, entry, sl, tp, qty, val_usdt, ai_text))

            conn.commit()
            conn.close()
            print(f"💾 [DB SUCCESS] AI Decision saved for {symbol} ({action})")
        except Exception as e:
            print(f"❌ DB Save AI Decision Error: {e}")

    def fetch_recent_decisions(self, limit=5):
        """
        Retrieves recent AI decisions from history.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, symbol, final_action, confidence_score, entry_price FROM ai_trade_decisions ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows