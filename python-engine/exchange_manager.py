import ccxt
import time
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ExchangeManager:
    """
    Production Exchange Aggregator connecting to 10 Primary Exchanges.
    Supports Spot & Perpetual Derivatives via CCXT with IP-Ban & Rate-Limit Shield.
    """
    SUPPORTED_EXCHANGES = [
        "bybit", "mexc", "okx", "gate", "bitget",
        "kucoin", "binance", "coinbase", "kraken", "htx"
    ]

    def __init__(self):
        self.exchanges: Dict[str, Any] = {}
        self.banned_exchanges: Dict[str, float] = {}  # Tracks IP-banned exchange timestamps
        self._initialize_exchanges()

    def _initialize_exchanges(self):
        logging.info("🔌 Initializing Multi-Exchange Connections with Rate-Limit Shield...")
        for ex_id in self.SUPPORTED_EXCHANGES:
            try:
                exchange_class = getattr(ccxt, ex_id)
                options = {
                    'enableRateLimit': True,
                    'timeout': 15000,
                }

                if ex_id in ["binance", "bybit", "okx", "bitget", "mexc", "gate"]:
                    options['options'] = {'defaultType': 'future'}

                instance = exchange_class(options)
                self.exchanges[ex_id] = instance
                logging.info(f"    ✅ Connected: {ex_id.upper()}")
            except Exception as e:
                logging.warning(f"    ⚠️ Could not initialize exchange {ex_id}: {e}")

    def _is_banned(self, exchange_id: str) -> bool:
        """Check if an exchange is temporarily IP-banned."""
        if exchange_id in self.banned_exchanges:
            ban_time = self.banned_exchanges[exchange_id]
            if time.time() - ban_time < 600:
                return True
            else:
                del self.banned_exchanges[exchange_id]
        return False

    def fetch_tickers_from_exchange(self, exchange_id: str) -> Dict[str, Any]:
        """Fetches all 24h market tickers with auto retry and IP ban guard."""
        if self._is_banned(exchange_id):
            logging.warning(f"🚫 Skipping {exchange_id.upper()} due to recent IP Ban / Rate Limit.")
            return {}

        ex = self.exchanges.get(exchange_id)
        if not ex:
            return {}

        for attempt in range(2):
            try:
                if not ex.markets:
                    ex.load_markets()
                tickers = ex.fetch_tickers()
                if not tickers or not isinstance(tickers, dict):
                    return {}
                return tickers
            except (ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
                logging.error(f"🛑 Rate Limit / IP Ban hit on {exchange_id.upper()}: {e}")
                self.banned_exchanges[exchange_id] = time.time()
                break
            except Exception as e:
                logging.warning(f"⚠️ Attempt {attempt + 1} failed for {exchange_id.upper()}: {e}")
                time.sleep(1.0)
        return {}

    def fetch_ohlcv_from_exchange(
        self, 
        symbol: str, 
        timeframe: str = "1h", 
        limit: int = 100, 
        exchange_id: str = "binance"
    ) -> Optional[List[List[Any]]]:
        """Fetches raw OHLCV candlestick data with IP Ban fallbacks."""
        if self._is_banned(exchange_id):
            return None

        ex = self.exchanges.get(exchange_id)
        if not ex:
            return None

        try:
            if not ex.markets:
                ex.load_markets()
            
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            time.sleep(0.1)
            
            if not ohlcv or not isinstance(ohlcv, list):
                return None
            return ohlcv
        except (ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
            logging.error(f"🛑 [418/429] IP Banned or Rate Limited on {exchange_id.upper()}: {e}")
            self.banned_exchanges[exchange_id] = time.time()
            return None
        except Exception as e:
            logging.debug(f"⚠️ Error fetching OHLCV for {symbol} on {exchange_id}: {e}")
            return None

    def fetch_ohlcv_df(
        self, 
        symbol: str, 
        timeframe: str = "1h", 
        limit: int = 100, 
        exchange_id: str = "binance"
    ) -> Optional[pd.DataFrame]:
        """Convenience method: Returns formatted pandas DataFrame."""
        raw_ohlcv = self.fetch_ohlcv_from_exchange(symbol, timeframe, limit, exchange_id)
        if not raw_ohlcv:
            return None

        df = pd.DataFrame(raw_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

if __name__ == "__main__":
    mgr = ExchangeManager()
    binance_tickers = mgr.fetch_tickers_from_exchange("binance")
    print(f"Total Tickers fetched from Binance Futures: {len(binance_tickers)}")