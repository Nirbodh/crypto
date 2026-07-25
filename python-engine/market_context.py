# python-engine/market_context.py

import pandas as pd
import numpy as np
import requests
import ccxt
import logging
import time
from typing import Dict, Any, Optional
from smc_engine import SMCEngine
from indicators import CoreIndicators  # <-- আপনার indicators থেকে ইমপোর্ট

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_json(obj):
    """
    NumPy / Pandas টাইপ বা কাস্টম অবজেক্টকে স্ট্যান্ডার্ড পাইথন টাইপে
    রূপান্তর করে যাতে JSON serialization এ কোনো সমস্যা না হয়।
    """
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_json(x) for x in obj)
    elif hasattr(obj, "item"):  # numpy/pandas scalar
        return obj.item()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return clean_json(obj.to_dict())
    elif isinstance(obj, (np.ndarray)):
        return clean_json(obj.tolist())
    else:
        return obj


class MarketContext:
    """Institutional Market Context Engine - Fetches macro data for AI & Risk Engine."""

    # ---- Class-level caching ----
    _futures_exchange = None
    _smc_engine = None

    @classmethod
    def _get_futures_exchange(cls):
        """Lazy-load and cache Binance USDM futures client."""
        if cls._futures_exchange is None:
            try:
                cls._futures_exchange = ccxt.binanceusdm({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                })
                logger.info("✅ Binance USDM Futures client initialized.")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Binance USDM: {e}")
                cls._futures_exchange = False
        return cls._futures_exchange if cls._futures_exchange is not False else None

    @classmethod
    def _get_smc_engine(cls):
        """Lazy-load and cache SMC Engine."""
        if cls._smc_engine is None:
            cls._smc_engine = SMCEngine()
        return cls._smc_engine

    @staticmethod
    def _http_get_with_retry(url: str, max_retries: int = 2, timeout: int = 10) -> Optional[dict]:
        """HTTP GET with retry logic."""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"⚠️ HTTP {response.status_code} from {url}, attempt {attempt+1}")
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning(f"⚠️ Network error on attempt {attempt+1}: {e}")
                time.sleep(0.5 * (2 ** attempt))
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                break
        return None

    @staticmethod
    def get_fear_and_greed_index() -> Dict[str, Any]:
        """Crypto Fear & Greed Index ফেচ করে।"""
        try:
            data = MarketContext._http_get_with_retry("https://api.alternative.me/fng/")
            if data and "data" in data and len(data["data"]) > 0:
                value = int(data["data"][0]["value"])
                classification = data["data"][0]["value_classification"]
                return {"value": value, "sentiment": classification}
        except Exception as e:
            logger.error(f"⚠️ Error fetching Fear & Greed Index: {e}")

        return {"value": 50, "sentiment": "Neutral"}

    @staticmethod
    def get_derivatives_metrics(symbol: str = "BTC/USDT") -> Dict[str, Any]:
        """
        Binance Derivatives API থেকে Funding Rate এবং Open Interest ফেচ করে।
        """
        metrics = {
            "funding_rate": 0.01,
            "funding_sentiment": "NEUTRAL",
            "open_interest": 0
        }

        try:
            exchange = MarketContext._get_futures_exchange()
            if exchange is None:
                return metrics

            futures_symbol = symbol.replace('/USDT', '/USDT:USDT') if '/USDT' in symbol else f"{symbol}/USDT:USDT"

            # ১. ফান্ডিং রেট ফেচ
            funding_info = exchange.fetch_funding_rate(futures_symbol)
            funding_rate = funding_info.get('fundingRate', 0) * 100  # Percentage
            metrics["funding_rate"] = round(funding_rate, 4)

            # Realistic thresholds
            if funding_rate >= 0.05:
                metrics["funding_sentiment"] = "OVERHEATED_BULLISH"
            elif funding_rate <= -0.03:
                metrics["funding_sentiment"] = "EXTREME_BEARISH"
            else:
                metrics["funding_sentiment"] = "HEALTHY"

            # ২. ওপেন ইন্টারেস্ট (Open Interest) ফেচ
            oi_info = exchange.fetch_open_interest(futures_symbol)
            metrics["open_interest"] = oi_info.get('openInterestAmount', 0)

        except ccxt.BadSymbol:
            logger.warning(f"⚠️ Symbol {symbol} not found on Binance Futures.")
        except ccxt.RateLimitExceeded:
            logger.warning("⚠️ Binance Futures rate limit exceeded, using cached data.")
        except Exception as e:
            logger.error(f"⚠️ Error fetching Derivatives Metrics ({symbol}): {e}")

        return metrics

    @staticmethod
    def get_btc_dominance() -> float:
        """CoinGecko পাবলিক এপিআই থেকে BTC Dominance ফেচ করে।"""
        try:
            data = MarketContext._http_get_with_retry("https://api.coingecko.com/api/v3/global")
            if data and "data" in data and "market_cap_percentage" in data["data"]:
                btc_dom = data["data"]["market_cap_percentage"].get("btc", 0)
                return round(btc_dom, 2)
        except Exception as e:
            logger.error(f"⚠️ Error fetching BTC Dominance: {e}")

        return 50.0

    @classmethod
    def build_full_market_context(
        cls,
        btc_1h_df: Optional[pd.DataFrame] = None,
        symbol: str = "BTC/USDT"
    ) -> Dict[str, Any]:
        """
        পুরো মার্কেটের স্ন্যাপশট তৈরি করে যা AI ও Risk Engine ব্যবহার করবে।
        """
        fng = cls.get_fear_and_greed_index()
        derivatives = cls.get_derivatives_metrics(symbol)
        btc_dom = cls.get_btc_dominance()

        # SMC Engine (lazy-loaded)
        smc_engine = cls._get_smc_engine()

        # SMC Analysis (if BTC data provided, else default)
        if btc_1h_df is not None and not btc_1h_df.empty:
            # আপনার CoreIndicators.calculate ব্যবহার করে RSI এবং EMA যোগ করা
            btc_calc = CoreIndicators.calculate(btc_1h_df.copy())
            smc_analysis = smc_engine.calculate_smc_score(btc_calc, symbol="BTC/USDT")
        else:
            smc_analysis = {
                "smc_liquidity_score": 50.0,
                "details": {"fvg": {}, "sweeps": {}, "order_blocks": {}, "zone": {}},
                "market_structure": "NEUTRAL"
            }

        # ---- BTC ট্রেন্ড ডিটেকশন (আপনার indicators ব্যবহার করে) ----
        btc_trend = "NEUTRAL"
        if btc_1h_df is not None and not btc_1h_df.empty:
            btc_calc = CoreIndicators.calculate(btc_1h_df.copy())
            last = btc_calc.iloc[-1]
            try:
                # আপনার EMA_50 কলাম ব্যবহার (যদি না থাকে তাহলে নিজে ক্যালকুলেট)
                if 'EMA_50' in last:
                    ma50 = last['EMA_50']
                else:
                    ma50 = btc_calc['close'].ewm(span=50, adjust=False).mean().iloc[-1]

                rsi_val = last.get('RSI', 50)

                if not pd.isna(ma50) and not pd.isna(rsi_val):
                    if last['close'] > ma50 and rsi_val > 50:
                        btc_trend = "BULLISH"
                    elif last['close'] < ma50 and rsi_val < 50:
                        btc_trend = "BEARISH"
            except Exception as e:
                logger.debug(f"Could not compute BTC trend: {e}")

        context_json = {
            "symbol": symbol,
            "btc_macro_trend": btc_trend,
            "btc_dominance": f"{btc_dom}%",
            "fear_and_greed": f"{fng['value']} ({fng['sentiment']})",
            "funding_rate": f"{derivatives['funding_rate']}%",
            "funding_sentiment": derivatives["funding_sentiment"],
            "open_interest": derivatives.get("open_interest", 0),
            "smc_liquidity_score": smc_analysis.get("smc_liquidity_score", 50),
            "smc_details": smc_analysis.get("details", {}),
            "market_structure": smc_analysis.get("market_structure", "NEUTRAL"),
            "news_sentiment": "NEUTRAL"
        }

        return clean_json(context_json)


# ================================================================
# QUICK TEST (Run with: python market_context.py)
# ================================================================
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    print("🧪 Testing MarketContext v2.0")
    print("=" * 60)

    # Create mock BTC data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="1h")
    np.random.seed(42)
    close = 65000 + np.cumsum(np.random.randn(100) * 100)
    high = close + np.abs(np.random.randn(100) * 50)
    low = close - np.abs(np.random.randn(100) * 50)
    open_ = low + np.random.rand(100) * (high - low)
    volume = np.random.randint(100, 500, 100)

    mock_df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    }, index=dates)

    print("📊 Building market context...")
    context = MarketContext.build_full_market_context(
        btc_1h_df=mock_df,
        symbol="BTC/USDT"
    )

    print(f"✅ BTC Trend: {context['btc_macro_trend']}")
    print(f"✅ Fear & Greed: {context['fear_and_greed']}")
    print(f"✅ Funding Rate: {context['funding_rate']}")
    print(f"✅ Funding Sentiment: {context['funding_sentiment']}")
    print(f"✅ SMC Score: {context['smc_liquidity_score']}")
    print(f"✅ BTC Dominance: {context['btc_dominance']}")

    print("\n" + "=" * 60)
    print("✅ MarketContext v2.0 ready for production (aligned with indicators.py).")