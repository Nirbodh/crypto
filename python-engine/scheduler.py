# python-engine/scheduler.py

import time
import schedule
from datetime import datetime
from main_engine import QuantTradingOrchestrator  # <-- ক্লাস ইমপোর্ট, ফাংশন না

SCAN_INTERVAL_MINUTES = 10


def scheduled_job():
    print(f"\n⏰ [SCHEDULER] Triggering Quant Pipeline Job at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        # 🔥 সরাসরি অর্কেস্ট্রেটর তৈরি করে স্ক্যান চালানো
        orchestrator = QuantTradingOrchestrator()
        orchestrator.scan_and_execute(max_universe_size=300)
    except Exception as e:
        print(f"❌ [SCHEDULER ERROR] Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


def start_scheduler():
    print("=" * 65)
    print("🤖 QUANT CRYPTO AI - AUTONOMOUS 24/7 SCHEDULER STARTED")
    print(f"⏱️ Scan Frequency: Every {SCAN_INTERVAL_MINUTES} Minutes", flush=True)
    print("=" * 65, flush=True)

    # Initial Run
    scheduled_job()

    # Schedule recurring job every 10 minutes
    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(scheduled_job)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    start_scheduler()
