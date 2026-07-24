# python-engine/sentiment_engine.py

import requests
import time
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SentimentEngine:
    """
    Institutional Grade Market Sentiment Engine.
    Processes Crypto Fear & Greed Index with Contrarian Market Dynamics, 
    Connection Pooling, Robust Fallbacks, and Smart Money Divergence Detection.
    """

    def __init__(self, cache_ttl_seconds: int = 1800, request_timeout: int = 5):
        self.api_url = "https://api.alternative.me/fng/"
        self.cache_ttl = cache_ttl_seconds  # Default: 30 minutes
        self.request_timeout = request_timeout
        self._cached_data: Dict[str, Any] = {}
        self._last_fetch_time: float = 0.0
        
        # Use Requests Session for Connection Pooling & Latency Optimization
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "InstitutionalCryptoEngine/2.0",
            "Accept": "application/json"
        })

    def fetch_sentiment_score(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetches Fear & Greed Index, calculates institutional contrarian score, 
        and flags sentiment divergence risk for the trading pipeline.
        """
        current_time = time.time()

        # 1. Return cached response if valid and not forcing refresh
        if not force_refresh and self._cached_data and (current_time - self._last_fetch_time < self.cache_ttl):
            return {**self._cached_data, "cached": True, "age_seconds": int(current_time - self._last_fetch_time)}

        try:
            response = self.session.get(self.api_url, timeout=self.request_timeout)
            if response.status_code == 200:
                data = response.json()
                raw_fng = data.get('data', [{}])[0]

                fng_val = int(raw_fng.get('value', 50))
                raw_classification = raw_fng.get('value_classification', 'Neutral')

                # 2. Institutional Analysis & Contrarian Score Calculation
                sentiment_metrics = self._derive_institutional_metrics(fng_val, raw_classification)

                self._cached_data = {
                    "sentiment_score": fng_val,                  # Raw Direct Score (0 to 100)
                    "fear_and_greed_index": fng_val,
                    "classification": raw_classification,
                    "contrarian_bias": sentiment_metrics["contrarian_bias"], # BULLISH / BEARISH / NEUTRAL
                    "regime": sentiment_metrics["regime"],
                    "risk_flag": sentiment_metrics["risk_flag"],
                    "status": "SUCCESS",
                    "timestamp": int(current_time)
                }
                self._last_fetch_time = current_time
                logging.info(f"📊 Market Sentiment Updated: {fng_val} ({raw_classification}) | Regime: {sentiment_metrics['regime']}")
                
                return {**self._cached_data, "cached": False, "age_seconds": 0}

        except requests.exceptions.Timeout:
            logging.warning("⚠️ Sentiment API Connection Timed Out. Activating Fallback.")
        except Exception as e:
            logging.error(f"⚠️ Sentiment API Unexpected Error: {e}")

        # 3. Fail-Safe Fallback Logic
        if self._cached_data:
            logging.info("ℹ️ Serving stale sentiment cache due to API network error.")
            return {**self._cached_data, "cached": True, "status": "STALE_CACHE_FALLBACK"}

        # Complete Fallback (Neutral Default)
        return {
            "sentiment_score": 50,
            "fear_and_greed_index": 50,
            "classification": "Neutral",
            "contrarian_bias": "NEUTRAL",
            "regime": "BALANCED_MARKET",
            "risk_flag": "NONE",
            "status": "DEFAULT_FALLBACK",
            "cached": False,
            "timestamp": int(current_time)
        }

    def _derive_institutional_metrics(self, fng_val: int, classification: str) -> Dict[str, str]:
        """
        Derives smart money metrics based on market extremes.
        - Extreme Fear (0-25) -> High Risk-Reward for Accumulation (Contrarian Bullish)
        - Extreme Greed (75-100) -> High Risk of Distribution / Long Squeeze (Contrarian Bearish)
        """
        if fng_val <= 20:
            return {
                "contrarian_bias": "BULLISH",
                "regime": "EXTREME_FEAR_ACCUMULATION",
                "risk_flag": "CAPITULATION_REBOUND_ZONE"
            }
        elif fng_val <= 40:
            return {
                "contrarian_bias": "SLIGHTLY_BULLISH",
                "regime": "FEAR_DISCOUNT_ZONE",
                "risk_flag": "NONE"
            }
        elif fng_val <= 60:
            return {
                "contrarian_bias": "NEUTRAL",
                "regime": "BALANCED_NEUTRAL",
                "risk_flag": "NONE"
            }
        elif fng_val <= 80:
            return {
                "contrarian_bias": "SLIGHTLY_BEARISH",
                "regime": "GREED_EXPANSION",
                "risk_flag": "OVERBOUGHT_WARNING"
            }
        else: # > 80
            return {
                "contrarian_bias": "BEARISH",
                "regime": "EXTREME_GREED_DISTRIBUTION",
                "risk_flag": "LONG_SQUEEZE_CORRECTION_RISK"
            }


if __name__ == "__main__":
    # Quick Standalone Test
    engine = SentimentEngine()
    sentiment = engine.fetch_sentiment_score()
    print("Updated Sentiment Engine Output:\n", sentiment)