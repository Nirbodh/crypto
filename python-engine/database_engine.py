# python-engine/database_engine.py

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta

class DatabaseEngine:
    def __init__(self):
        # Read PostgreSQL credentials from environment variables
        self.db_host = os.getenv("POSTGRES_HOST", "localhost")
        self.db_port = os.getenv("POSTGRES_PORT", "5432")
        self.db_name = os.getenv("POSTGRES_DB", "quant_db")
        self.db_user = os.getenv("POSTGRES_USER", "postgres")
        self.db_pass = os.getenv("POSTGRES_PASSWORD", "postgres123")
        
        self.init_db()

    def get_connection(self):
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_pass
        )

    def init_db(self):
        """Creates necessary tables if they don't exist."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 1. Trade Signals Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_signals (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    unified_score FLOAT NOT NULL,
                    decision VARCHAR(20) NOT NULL,
                    payload JSONB,
                    ai_verdict JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Duplicate Signal Cooldown Cache Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_cooldowns (
                    symbol VARCHAR(20) PRIMARY KEY,
                    last_sent_at TIMESTAMP NOT NULL
                );
            """)

            conn.commit()
            cursor.close()
            conn.close()
            print("✅ PostgreSQL Database Initialized Successfully.")
        except Exception as e:
            print(f"⚠️ Database Initialization Error: {e}")

    def save_trade_signal(self, payload: dict, ai_verdict: dict):
        """Saves executed trade signal to DB."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            symbol = payload.get("symbol")
            score = payload.get("unified_score", 0.0)
            decision = ai_verdict.get("final_decision", "HOLD")

            import json
            cursor.execute("""
                INSERT INTO trade_signals (symbol, unified_score, decision, payload, ai_verdict)
                VALUES (%s, %s, %s, %s, %s);
            """, (symbol, score, decision, json.dumps(payload), json.dumps(ai_verdict)))

            # Update cooldown cache
            cursor.execute("""
                INSERT INTO signal_cooldowns (symbol, last_sent_at)
                VALUES (%s, %s)
                ON CONFLICT (symbol) DO UPDATE SET last_sent_at = EXCLUDED.last_sent_at;
            """, (symbol, datetime.now()))

            conn.commit()
            cursor.close()
            conn.close()
            print(f"💾 Trade Signal saved to Database for {symbol}")
        except Exception as e:
            print(f"❌ Failed to save signal to Database: {e}")

    def is_duplicate_signal(self, symbol: str, cooldown_hours: int = 4) -> bool:
        """Checks if a signal was sent within the cooldown period directly from DB."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT last_sent_at FROM signal_cooldowns WHERE symbol = %s;
            """, (symbol,))
            
            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if row:
                last_sent = row['last_sent_at']
                if datetime.now() - last_sent < timedelta(hours=cooldown_hours):
                    return True
            return False
        except Exception as e:
            print(f"⚠️ Cooldown check error: {e}")
            return False