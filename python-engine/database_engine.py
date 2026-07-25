import os
import json
from datetime import datetime, timezone, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor


class DatabaseEngine:
    """
    Production-ready PostgreSQL database engine with:
    - DATABASE_URL support (Railway, Render, Neon, Supabase)
    - Individual POSTGRES_* variables for local development
    - SSL mode configurable via POSTGRES_SSLMODE
    - Automatic connection & cursor cleanup via context managers
    - Timezone-aware UTC timestamps
    - Duplicate signal cooldown cache
    - Health check method
    """

    def __init__(self):
        # Primary: DATABASE_URL (used by Railway, Render, Neon, Supabase)
        self.database_url = os.getenv("DATABASE_URL")

        # Fallback: individual PostgreSQL variables (local Docker / dev)
        self.db_host = os.getenv("POSTGRES_HOST", "localhost")
        self.db_port = os.getenv("POSTGRES_PORT", "5432")
        self.db_name = os.getenv("POSTGRES_DB", "quant_db")
        self.db_user = os.getenv("POSTGRES_USER", "postgres")
        self.db_pass = os.getenv("POSTGRES_PASSWORD", "postgres123")

        # SSL mode: default "require" for production, can be overridden
        self.ssl_mode = os.getenv("POSTGRES_SSLMODE", "require")

        self.init_db()

    def get_connection(self):
        """
        Returns a PostgreSQL connection object.
        Uses DATABASE_URL if available, otherwise individual params.
        SSL is enabled by default for managed providers.
        """
        if self.database_url:
            return psycopg2.connect(
                self.database_url,
                sslmode=self.ssl_mode
            )

        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_pass,
        )

    def init_db(self):
        """Create required tables if they do not exist."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # 1. Trade signals table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS trade_signals (
                            id SERIAL PRIMARY KEY,
                            symbol VARCHAR(50) NOT NULL,
                            unified_score DOUBLE PRECISION NOT NULL,
                            decision VARCHAR(20) NOT NULL,
                            payload JSONB,
                            ai_verdict JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)

                    # 2. Cooldown cache table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS signal_cooldowns (
                            symbol VARCHAR(50) PRIMARY KEY,
                            last_sent_at TIMESTAMP NOT NULL
                        );
                    """)

                    conn.commit()

            print("✅ PostgreSQL Database Initialized Successfully.")

        except Exception as e:
            print(f"⚠️ Database Initialization Error: {e}")

    def save_trade_signal(self, payload: dict, ai_verdict: dict):
        """Save executed trade signal to database and update cooldown cache."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    symbol = payload.get("symbol")
                    score = payload.get("unified_score", 0.0)
                    decision = ai_verdict.get("final_decision", "HOLD")

                    cur.execute("""
                        INSERT INTO trade_signals
                        (symbol, unified_score, decision, payload, ai_verdict)
                        VALUES (%s, %s, %s, %s, %s);
                    """, (
                        symbol,
                        score,
                        decision,
                        json.dumps(payload),
                        json.dumps(ai_verdict)
                    ))

                    # Update cooldown cache
                    cur.execute("""
                        INSERT INTO signal_cooldowns (symbol, last_sent_at)
                        VALUES (%s, %s)
                        ON CONFLICT (symbol) DO UPDATE
                        SET last_sent_at = EXCLUDED.last_sent_at;
                    """, (
                        symbol,
                        datetime.now(timezone.utc)
                    ))

                    conn.commit()

            print(f"💾 Trade Signal saved to PostgreSQL ({symbol})")

        except Exception as e:
            print(f"❌ Failed to save signal: {e}")

    def is_duplicate_signal(self, symbol: str, cooldown_hours: int = 4) -> bool:
        """
        Returns True if a signal for this symbol was sent within the cooldown period.
        Uses timezone-aware UTC comparison.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT last_sent_at
                        FROM signal_cooldowns
                        WHERE symbol = %s;
                    """, (symbol,))

                    row = cur.fetchone()

            if not row:
                return False

            last_sent = row["last_sent_at"]

            # last_sent should be timezone-aware; if not, make it aware
            if last_sent.tzinfo is None:
                last_sent = last_sent.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)

            if now - last_sent < timedelta(hours=cooldown_hours):
                return True

            return False

        except Exception as e:
            print(f"⚠️ Cooldown check error: {e}")
            return False

    def health_check(self) -> bool:
        """
        Perform a simple database health check.
        Returns True if the database is reachable and responsive.
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
            return True
        except Exception as e:
            print(f"❌ Database Health Check Failed: {e}")
            return False


# ============================================
# QUICK TEST (if run directly)
# ============================================
if __name__ == "__main__":
    db = DatabaseEngine()

    if db.health_check():
        print("✅ Database connection is healthy.")
    else:
        print("❌ Database connection failed.")

    # Test duplicate check
    symbol = "BTC/USDT"
    is_dup = db.is_duplicate_signal(symbol, cooldown_hours=1)
    print(f"Duplicate for {symbol}: {is_dup}")

    # Test saving a dummy signal
    test_payload = {
        "symbol": symbol,
        "unified_score": 85.0,
        "decision": "BUY"
    }
    test_ai = {"final_decision": "EXECUTE_LONG", "confidence": 78}
    db.save_trade_signal(test_payload, test_ai)
