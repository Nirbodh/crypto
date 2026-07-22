# python-engine/main_engine.py

import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from data_fetcher import DataFetcher
from screener import TechnicalScreener
from fundamental_engine import MultiSourceFundamentalEngine
from derivatives_engine import DerivativesEngineV2
from sentiment_engine import SentimentEngine
from score_fusion_engine import ScoreFusionEngine
from risk_engine import RiskEngine
from ai_engine import AIDebateEngine
from database_engine import DatabaseEngine
from telegram_notifier import TelegramNotifier


# Global In-Memory Cache for Duplicate Signal Protection
# Note: For full production, this will be migrated to PostgreSQL / Redis
SENT_SIGNALS_CACHE = {}
COOLDOWN_HOURS = 4  # Cooldown period per symbol in hours


def is_duplicate_signal(symbol: str) -> bool:
    """Check if an alert was sent for this symbol within the cooldown window."""
    now = datetime.now()
    if symbol in SENT_SIGNALS_CACHE:
        last_sent = SENT_SIGNALS_CACHE[symbol]
        if now - last_sent < timedelta(hours=COOLDOWN_HOURS):
            return True
    return False


def run_quant_pipeline():
    print("\n" + "="*65)
    print(f"🚀 QUANT CRYPTO AI - PIPELINE SCAN [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print("="*65)

    print("🔑 API Connection Status:")
    print(f" - Gemini AI:       {'✅ Loaded' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
    print(f" - CoinMarketCap:   {'✅ Loaded' if os.getenv('COINMARKETCAP_API_KEY') else '❌ Missing'}")
    print(f" - CoinGlass:       {'✅ Loaded' if os.getenv('COINGLASS_API_KEY') else '❌ Missing'}")
    print(f" - CoinGecko:       {'✅ Loaded' if os.getenv('COINGECKO_API_KEY') else '❌ Missing'}")
    print("="*65)

    # 1. Initialize Engines
    fetcher = DataFetcher()
    screener = TechnicalScreener()
    fund_engine = MultiSourceFundamentalEngine()
    deriv_engine = DerivativesEngineV2()
    sent_engine = SentimentEngine()
    fusion_engine = ScoreFusionEngine()

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    ai_engine = AIDebateEngine(api_key=gemini_key) if gemini_key else None

    db_engine = DatabaseEngine()
    notifier = TelegramNotifier()

    # Dynamic Top Volume Symbol Selection (Fixed Method Name Issue)
    try:
        symbols = fetcher.get_top_volume_symbols(exchange_name="binance", limit=20)
        if not symbols:
            symbols = ["ETH/USDT", "SOL/USDT", "BTC/USDT", "BNB/USDT"]
    except Exception as e:
        print(f"⚠️ Top volume fetch warning: {e}. Falling back to default list.")
        symbols = ["ETH/USDT", "SOL/USDT", "BTC/USDT", "BNB/USDT"]

    print(f"📋 Scanning Top Market Pairs (5m, 15m, 30m, 1h, 4h): {symbols}\n")

    # Fetch Global Market Sentiment
    sent_data = sent_engine.fetch_sentiment_score()
    print(
        f"🌍 Market Fear & Greed Index: "
        f"{sent_data.get('fear_and_greed_index')} "
        f"({sent_data.get('classification')})"
    )

    # 2. Main Scan Loop
    for symbol in symbols:
        print(f"\n🔍 [ANALYZING]: {symbol}")

        # Duplicate Signal Check
        if is_duplicate_signal(symbol):
            print(f"⌛ SKIPPED {symbol} - Already alerted within last {COOLDOWN_HOURS} hours.")
            continue

        # Fetch 5 Timeframe Candlestick Data: 5m, 15m, 30m, 1h, 4h
        df_5m  = fetcher.fetch_ohlcv(symbol, exchange_name="binance", timeframe="5m", limit=100)
        df_15m = fetcher.fetch_ohlcv(symbol, exchange_name="binance", timeframe="15m", limit=100)
        df_30m = fetcher.fetch_ohlcv(symbol, exchange_name="binance", timeframe="30m", limit=100)
        df_1h  = fetcher.fetch_ohlcv(symbol, exchange_name="binance", timeframe="1h", limit=100)
        df_4h  = fetcher.fetch_ohlcv(symbol, exchange_name="binance", timeframe="4h", limit=100)

        if df_5m is None or df_15m is None or df_30m is None or df_1h is None or df_4h is None:
            print(f"❌ Missing candle data for {symbol}. Skipping.")
            continue

        # -------------------------------------------------------------
        # Gatekeeper 1: Technical Score Filter
        # -------------------------------------------------------------
        tech_result = screener.run_screener(
            df_5m=df_5m,
            df_15m=df_15m,
            df_30m=df_30m,
            df_1h=df_1h,
            df_4h=df_4h
        )
        tech_result["symbol"] = symbol
        tech_score = tech_result.get("technical_score", 0)

        print(f"📊 Technical Score (5 TF): {tech_score}/100")

        if tech_score < 60:
            print(f"⏩ SKIPPED {symbol} - Weak Technical Setup (< 60)")
            continue

        # Fundamentals & Derivatives Fetching
        print(f"⚙️ Running Fundamental & Derivatives Engines for {symbol}...")
        fund_result = fund_engine.fetch_fundamental_data(symbol)
        deriv_result = deriv_engine.fetch_derivatives_data(symbol)

        # Score Fusion
        fusion_result = fusion_engine.fuse_scores(
            tech_data=tech_result,
            fund_data=fund_result,
            deriv_data=deriv_result,
            sent_data=sent_data
        )

        unified_score = fusion_result.get("unified_score", 0)
        print(f"🎯 UNIFIED QUANT SCORE: {unified_score:.2f}/100")
        print(f"Technical   : {tech_result.get('technical_score')}")
        print(f"Fundamental : {fund_result.get('fundamental_score')}")
        print(f"Derivatives : {deriv_result.get('derivatives_score')}")
        print(f"Sentiment   : {sent_data.get('sentiment_score')}")
        print(f"Unified     : {fusion_result.get('unified_score')}")

        # -------------------------------------------------------------
        # Gatekeeper 2: Unified Quant Score Filter
        # -------------------------------------------------------------
        if not fusion_result.get("pass_to_ai_debate", False):
            print(
		f"🛑 REJECTED {symbol} - " 
		f"Fusion Gate Failed ({unified_score:.2f})"
	    )
            continue

        # -------------------------------------------------------------
        # Gatekeeper 3: AI Gate Filter (Score >= 80)
        # -------------------------------------------------------------
        if unified_score >= 80:
            print(f"🔥 HIGH CONVICTION SETUP ({unified_score:.2f}) -> ROUTING TO GEMINI AI DEBATE ENGINE: {symbol}")
        else:
            print(f"⚠️ MODERATE SETUP ({unified_score:.2f}) -> SKIPPING AI DEBATE TO REDUCE COST/NOISE")
            continue

        # Risk Calculation Engine (Fixed Parameter Mismatch)
        entry_price = float(df_5m['close'].iloc[-1])
        atr_5m = (
            float(df_5m['atr'].iloc[-1])
            if 'atr' in df_5m.columns
            else entry_price * 0.01
        )

        risk_analysis = RiskEngine.calculate_trade_risk(
            entry_price=entry_price,
            atr_5m=atr_5m,
            account_balance=10000.0,
            risk_per_trade_percent=1.5
        )

        ai_payload = {
            "symbol": symbol,
            "unified_score": unified_score,
            "technical": tech_result,
            "fundamental": fund_result,
            "derivatives": deriv_result,
            "sentiment": sent_data,
            "fusion_breakdown": fusion_result.get("breakdown", {}),
            "red_flags": fusion_result.get("all_red_flags", []),
            "green_flags": fusion_result.get("all_green_flags", []),
            "risk_management": risk_analysis
        }

        # -------------------------------------------------------------
        # Gatekeeper 4: AI Decision & Execution Filter
        # -------------------------------------------------------------
        if ai_engine:
            print(f"🤖 Audit by Gemini AI Engine for {symbol}...")
            ai_verdict = ai_engine.run_debate(ai_payload)
            
            decision = ai_verdict.get("final_decision", "HOLD")
            
            if decision in ["EXECUTE_LONG", "EXECUTE_SHORT", "LONG", "SHORT"]:
                print(f"🚀 STRICT EXECUTION CONFIRMED FOR {symbol} ({decision})")
                
                # Save to Database
                db_engine.save_trade_signal(ai_payload, ai_verdict)
                
                # Send Alert
                print(f"📱 Sending Telegram Alert for {symbol}...")
                notifier.send_trade_alert(ai_payload, ai_verdict)
                
                # Record to Cache
                SENT_SIGNALS_CACHE[symbol] = datetime.now()
            else:
                print(f"⏸️ REJECTED BY AI - Verdict: '{decision}' for {symbol} (Actionable Signal Not Met)")

        time.sleep(1)


if __name__ == "__main__":
    run_quant_pipeline()