# python-engine/db_engine.py

import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DatabaseEngine:
    """
    Production Quantitative Database Engine:
    Manages SQLite storage for Market Scans, AI Debate Decisions, Trade Performance,
    Strategy Analytics, and AI Feedback Loops.
    """
    def __init__(self, db_path: str = "quant_trading.db"):
        self.db_path = db_path
        self._init_sqlite_db()

    @contextmanager
    def _get_connection(self):
        """Context Manager ensuring connections and cursors are safely closed."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row  # Enables column-access by name
        
        # Enable Write-Ahead Logging (WAL) for multi-threaded safety
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_sqlite_db(self):
        """Creates necessary tables for market scans, AI debate decisions, and performance analytics."""
        try:
            with self._get_connection() as conn:
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

                # 3. Trade Results Table (Production Upgrade #10)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id INTEGER,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    pnl_usdt REAL NOT NULL,
                    pnl_percent REAL NOT NULL,
                    r_multiple REAL NOT NULL,
                    exit_reason TEXT NOT NULL,
                    opened_at DATETIME,
                    closed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (decision_id) REFERENCES ai_trade_decisions (id)
                )
                """)

                # 4. Performance Metrics Table (Production Upgrade #10)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    net_profit_usdt REAL
                )
                """)

                # 5. AI Feedback Loop Table (Production Upgrade #10)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_feedback_loop (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER,
                    predicted_confidence INTEGER,
                    actual_outcome TEXT,
                    lesson_learned TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trade_results (id)
                )
                """)

                conn.commit()
        except Exception as e:
            logging.error(f"❌ DB Initialization Error: {e}")

    def save_scan_result(self, payload: Dict[str, Any]):
        """Saves raw screener scan data into the database."""
        try:
            symbol = payload.get("symbol", "UNKNOWN")
            exchange = payload.get("exchange", "BINANCE")
            score = payload.get("score", 0)
            passed = payload.get("pass", False)
            price = payload.get("price", 0.0)
            atr_5m = payload.get("atr_5m", 0.0)
            raw_payload = json.dumps(payload)

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO market_scans (symbol, exchange, score, pass, price, atr_5m, raw_payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (symbol, exchange, score, passed, price, atr_5m, raw_payload))
                conn.commit()
        except Exception as e:
            logging.error(f"❌ DB Save Scan Error: {e}")

    def save_ai_decision(self, symbol: str, exchange: str, decision_data: Dict[str, Any], ai_text: str) -> Optional[int]:
        """Saves AI Decision Engine consensus & execution plan. Returns inserted row ID."""
        try:
            action = decision_data.get("final_decision") or decision_data.get("action") or "REJECT"
            confidence = decision_data.get("confidence") or decision_data.get("confidence_score") or 0
            entry = decision_data.get("entry_price") or decision_data.get("entry") or 0.0
            sl = decision_data.get("stop_loss") or decision_data.get("sl") or 0.0
            tp = decision_data.get("take_profit") or decision_data.get("tp") or 0.0
            qty = decision_data.get("position_qty") or decision_data.get("quantity") or 0.0
            val_usdt = decision_data.get("position_usdt") or decision_data.get("position_value_usdt") or 0.0

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO ai_trade_decisions 
                    (symbol, exchange, final_action, confidence_score, entry_price, stop_loss, take_profit, position_size_tokens, position_value_usdt, ai_debate_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, exchange, str(action), int(confidence), float(entry), float(sl), float(tp), float(qty), float(val_usdt), str(ai_text)))
                conn.commit()
                inserted_id = cursor.lastrowid

            logging.info(f"💾 [DB SUCCESS] AI Decision saved for {symbol} ({action})")
            return inserted_id
        except Exception as e:
            logging.error(f"❌ DB Save AI Decision Error: {e}")
            return None

    def fetch_recent_decisions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves recent AI decisions from history as dictionary list."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, symbol, exchange, final_action, confidence_score, entry_price, stop_loss, take_profit, status 
                    FROM ai_trade_decisions 
                    ORDER BY id DESC 
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logging.error(f"❌ DB Fetch Decisions Error: {e}")
            return []

    def record_trade_result(self, trade_data: Dict[str, Any]) -> Optional[int]:
        """Records completed trade execution result for performance tracking."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trade_results
                    (decision_id, symbol, exchange, direction, entry_price, exit_price, pnl_usdt, pnl_percent, r_multiple, exit_reason, opened_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data.get("decision_id"),
                    trade_data.get("symbol"),
                    trade_data.get("exchange", "BINANCE"),
                    trade_data.get("direction", "LONG"),
                    trade_data.get("entry_price"),
                    trade_data.get("exit_price"),
                    trade_data.get("pnl_usdt"),
                    trade_data.get("pnl_percent"),
                    trade_data.get("r_multiple"),
                    trade_data.get("exit_reason", "TP_HIT"),
                    trade_data.get("opened_at")
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logging.error(f"❌ DB Record Trade Result Error: {e}")
            return None


if __name__ == "__main__":
    test_db_filename = "test_quant.db"
    db = DatabaseEngine(test_db_filename)
    
    # Save test AI Decision
    dec_id = db.save_ai_decision(
        symbol="BTC/USDT",
        exchange="BINANCE",
        decision_data={
            "action": "EXECUTE_LONG",
            "confidence": 85,
            "entry_price": 65000.0,
            "stop_loss": 63700.0,
            "take_profit": 68250.0,
            "position_qty": 0.015,
            "position_usdt": 1000.0
        },
        ai_text="Strong SMC confluence with institutional liquidity sweep."
    )
    
    decisions = db.fetch_recent_decisions(1)
    print("Recent Decisions:", decisions)

    # Clean up test database securely
    if os.path.exists(test_db_filename):
        try:
            os.remove(test_db_filename)
            # Remove WAL files if generated
            if os.path.exists(f"{test_db_filename}-wal"):
                os.remove(f"{test_db_filename}-wal")
            if os.path.exists(f"{test_db_filename}-shm"):
                os.remove(f"{test_db_filename}-shm")
            print("Cleanup successful without PermissionError.")
        except Exception as err:
            print(f"Cleanup Note: {err}")