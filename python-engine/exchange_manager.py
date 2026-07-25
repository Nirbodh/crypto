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
        "bybit", "mexc",  "bitget",
        "kucoin", "binance", "coinbase", "kraken", 
    ]

    # Exchanges that support perpetual futures (USDT-margined)
    PERPETUAL_EXCHANGES = {
        "binance", "bybit", "bitget", "mexc", 
    }

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

                # Perpetual/swap exchanges – set correct defaultType
                if ex_id == "binance":
                    options['options'] = {'defaultType': 'future'}
                elif ex_id in self.PERPETUAL_EXCHANGES:
                    options['options'] = {'defaultType': 'swap'}

                instance = exchange_class(options)
                self.exchanges[ex_id] = instance
                logging.info(f"    ✅ Connected: {ex_id.upper()}")
            except Exception as e:
                logging.warning(f"    ⚠️ Could not initialize exchange {ex_id}: {e}")

    # ----------------------------------------------------------------------
    # HELPER: Safe market loader
    # ----------------------------------------------------------------------
    def _ensure_markets_loaded(self, ex, exchange_id: str) -> bool:
        """
        Safely load exchange markets. Returns True if markets are available.
        """
        try:
            if ex.markets:
                return True

            markets = ex.load_markets()
            if not markets:
                logging.warning(f"⚠️ Empty market response from {exchange_id.upper()}")
                return False
            return True

        except Exception as e:
            logging.warning(f"⚠️ Market loading failed for {exchange_id.upper()}: {e}")
            return False

    # ----------------------------------------------------------------------
    # HELPER: Symbol normalizer (only for perpetual exchanges)
    # ----------------------------------------------------------------------
    def _normalize_symbol(self, symbol: str, exchange_id: str) -> str:
        """
        Convert a simple symbol (e.g., "BTC") to exchange-specific format.
        Only modifies symbols for perpetual exchanges (Binance, Bybit, etc.).
        For spot exchanges, returns the symbol unchanged.
        """
        if exchange_id.lower() not in self.PERPETUAL_EXCHANGES:
            return symbol

        # Already in unified format (e.g., BTC/USDT:USDT)
        if ":USDT" in symbol:
            return symbol

        # If no slash, assume base asset only → BTC/USDT:USDT
        if "/" not in symbol:
            return f"{symbol}/USDT:USDT"

        # If slash exists, extract base and quote
        base, quote = symbol.split("/")
        if quote == "USDT":
            return f"{base}/USDT:USDT"

        # If quote is not USDT, keep as is (e.g., BTC/USDC) – avoid invalid conversion
        return symbol

    # ----------------------------------------------------------------------
    # Existing methods with all fixes
    # ----------------------------------------------------------------------
    def get_exchange(self, exchange_id: str):
        """
        Returns the exchange instance for the given exchange ID.
        """
        return self.exchanges.get(exchange_id)

    def _is_banned(self, exchange_id: str) -> bool:
        """Check if an exchange is temporarily IP-banned."""
        if exchange_id in self.banned_exchanges:
            ban_time = self.banned_exchanges[exchange_id]
            if time.time() - ban_time < 600:   # 10 minutes cooldown
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

        if not self._ensure_markets_loaded(ex, exchange_id):
            return {}

        for attempt in range(2):
            try:
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

        if not self._ensure_markets_loaded(ex, exchange_id):
            return None

        # Normalize symbol only for perpetual exchanges
        symbol = self._normalize_symbol(symbol, exchange_id)

        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            time.sleep(0.1)   # extra safety
            if not ohlcv or not isinstance(ohlcv, list):
                return None
            return ohlcv
        except (ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
            logging.error(f"🛑 [418/429] IP Banned or Rate Limited on {exchange_id.upper()}: {e}")
            self.banned_exchanges[exchange_id] = time.time()
            return None
        except Exception as e:
            logging.warning(f"⚠️ Error fetching OHLCV for {symbol} on {exchange_id}: {e}")
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

    # Test tickers from Binance (futures)
    binance_tickers = mgr.fetch_tickers_from_exchange("binance")
    print(f"✅ Total Tickers fetched from Binance Futures: {len(binance_tickers)}")

    # Test OHLCV for BTC perpetual
    df_btc = mgr.fetch_ohlcv_df("BTC", "15m", 200, "binance")
    if df_btc is not None:
        print("\n📊 BTC 15m OHLCV (first 5 rows):")
        print(df_btc.head())
    else:
        print("❌ Failed to fetch BTC OHLCV")

    # Test MEXC (which previously caused NoneType error)
    mexc_tickers = mgr.fetch_tickers_from_exchange("mexc")
    print(f"✅ Total Tickers fetched from MEXC Futures: {len(mexc_tickers)}")

    # Test SNDK – but check if pair exists first
    binance = mgr.get_exchange("binance")
    if binance and "SNDK/USDT:USDT" in binance.markets:
        df_sndk = mgr.fetch_ohlcv_df("SNDK", "1h", 100, "binance")
        if df_sndk is not None:
            print("\n📊 SNDK 1h OHLCV (first 5 rows):")
            print(df_sndk.head())
        else:
            print("❌ Failed to fetch SNDK OHLCV (pair may exist but data unavailable)")
    else:
        print("ℹ️ SNDK/USDT:USDT not found on Binance – skipping test")
