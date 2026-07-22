# python-engine/scheduler.py

import time
import schedule
from datetime import datetime
from main_engine import run_quant_pipeline

SCAN_INTERVAL_MINUTES = 5  # Runs every 5 minutes for crypto 5m candles


def scheduled_job():
    print(f"\n⏰ [SCHEDULER] Triggering Quant Pipeline Job at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        run_quant_pipeline()
    except Exception as e:
        print(f"❌ [SCHEDULER ERROR] Pipeline execution failed: {e}")


def start_scheduler():
    print("=" * 65)
    print("🤖 QUANT CRYPTO AI - AUTONOMOUS 24/7 SCHEDULER STARTED")
    print(f"⏱️ Scan Frequency: Every {SCAN_INTERVAL_MINUTES} Minutes", flush=True)
    print("=" * 65, flush=True)
    

    # Initial Run
    scheduled_job()

    # Schedule recurring job
    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(scheduled_job)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    start_scheduler()