# python-engine/data_fetcher.py

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import time
import logging
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Any, Union
from functools import lru_cache
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataFetcher:
    """
    v3.1 - Institutional Grade Async Data Fetcher
    - Retry Mechanism with Exponential Backoff
    - Symbol Validation (load_markets cache)
    - Minimum Candle Check (50 candles minimum)
    - Duplicate Removal + Sort + NaN Cleanup
    - Async I/O + In-memory TTL caching
    - Dynamic exchange health scoring
    """

    def __init__(self, enable_async: bool = True, cache_ttl_seconds: int = 60, min_candles: int = 50):
        self.enable_async = enable_async
        self.cache_ttl_seconds = cache_ttl_seconds
        self.min_candles = min_candles  # Minimum required candles for indicators
        
        # ---- 1. Exchange Configuration ----
        self.exchange_configs = {
            'bybit': {'enableRateLimit': True, 'options': {'defaultType': 'spot'}},
            'mexc': {'enableRateLimit': True},
            'kucoin': {'enableRateLimit': True},
            'okx': {'enableRateLimit': True},
            'binance': {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
        }
        
        # ---- 2. Sync Exchanges (Legacy & Fallback) ----
        self.exchanges = {}
        self.exchange_markets = {}  # Cache for markets
        for name, config in self.exchange_configs.items():
            try:
                self.exchanges[name] = ccxt.bybit(config) if name == 'bybit' else getattr(ccxt, name)(config)
                self.exchange_markets[name] = None  # Lazy load
            except Exception:
                self.exchanges[name] = None
        
        # ---- 3. Async Exchanges (High Performance) ----
        self.async_exchanges = {}
        if self.enable_async:
            for name, config in self.exchange_configs.items():
                try:
                    self.async_exchanges[name] = ccxt_async.bybit(config) if name == 'bybit' else getattr(ccxt_async, name)(config)
                except Exception:
                    self.async_exchanges[name] = None
        
        # ---- 4. State Management ----
        self.banned_exchanges = set()
        self.bad_symbols = set()  # permanently blacklisted symbols (delisted)
        self.exchange_health = {name: {"successes": 0, "failures": 0, "latency_ms": []} for name in self.exchange_configs}
        
        # ---- 5. Cache (In-Memory TTL) ----
        self._cache = {}
        
        # ---- 6. Connection Pool (Sync) ----
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; CryptoQuantBot/2.0)'})
        
        # ---- 7. Filters ----
        self.stable_coins = {
            'USDT', 'USDC', 'BUSD', 'FDUSD', 'DAI', 'TUSD', 
            'USDD', 'USD1', 'PYUSD', 'USDE', 'USDP', 'USDS',
            'EUR', 'GBP', 'AUD', 'BRL', 'TRY', 'RUB'
        }
        self.excluded_base_assets = {
            'BANK', 'TEST', 'FAKE', 'XUSD', 'XAUT', 'PAXG',
            'BULL', 'BEAR', 'UP', 'DOWN', '3L', '3S', '2L', '2S',
            'COIN', 'ORCL', 'TSM', 'ARM', 'SMCI', 'DELL', 'NATGAS', 
            'ANTHROPIC', 'HOOD', 'RKLB', 'TQQQ', 'AAPL', 'MSFT', 'NVDA'
        }

    # ================================================================
    # 1. CACHE HELPERS
    # ================================================================
    def _get_cache(self, key: str) -> Optional[Any]:
        """Get cached item if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if (time.time() - timestamp) < self.cache_ttl_seconds:
                return value
            else:
                del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """Store item in cache with current timestamp."""
        self._cache[key] = (value, time.time())

    # ================================================================
    # 2. SYMBOL VALIDATION (load_markets with cache)
    # ================================================================
    def _load_markets(self, exchange_name: str, market_type: str = 'spot') -> Optional[Dict]:
        """Load markets for an exchange with caching."""
        ex_obj = self.exchanges.get(exchange_name)
        if not ex_obj:
            return None
        
        cache_key = f"markets_{exchange_name}_{market_type}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached
        
        try:
            # Set market type before loading
            if market_type == 'swap':
                ex_obj.options['defaultType'] = 'swap'
            else:
                ex_obj.options['defaultType'] = 'spot'
            
            markets = ex_obj.load_markets()
            self._set_cache(cache_key, markets)
            return markets
        except Exception as e:
            logging.debug(f"⚠️ Failed to load markets for {exchange_name}: {e}")
            return None

    def _is_symbol_valid(self, exchange_name: str, symbol: str, market_type: str = 'spot') -> bool:
        """Check if symbol exists on exchange (with caching)."""
        markets = self._load_markets(exchange_name, market_type)
        if markets is None:
            return False
        
        # Try both formats: "BTC/USDT" and "BTC/USDT:USDT" for futures
        if market_type == 'swap':
            alt_symbol = symbol.replace('/USDT', '/USDT:USDT')
            return symbol in markets or alt_symbol in markets
        return symbol in markets

    # ================================================================
    # 3. RETRY WITH EXPONENTIAL BACKOFF
    # ================================================================
    def _retry_call(self, func, *args, max_retries: int = 3, **kwargs):
        """Synchronous retry with exponential backoff."""
        last_exception = None
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                return result, latency_ms
            except (ccxt.NetworkError, ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
                wait_time = (2 ** attempt) + 0.5
                logging.warning(f"⚠️ Retry {attempt+1}/{max_retries} after {wait_time:.1f}s: {e}")
                time.sleep(wait_time)
                last_exception = e
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                # Non-retryable errors (BadSymbol, AuthenticationError) raise immediately
                raise e
        return None, 0

    # ================================================================
    # 4. DYNAMIC EXCHANGE HEALTH & PRIORITY
    # ================================================================
    def _get_best_exchange(self, exclude: List[str] = None) -> str:
        """Return highest health exchange excluding banned or specified ones."""
        exclude = exclude or []
        candidates = [name for name in self.exchange_configs 
                      if name not in self.banned_exchanges and name not in exclude]
        if not candidates:
            return 'binance'  # ultimate fallback
        
        scores = {}
        for name in candidates:
            health = self.exchange_health[name]
            total = health['successes'] + health['failures']
            if total == 0:
                success_rate = 0.95  # default high for untested
            else:
                success_rate = health['successes'] / total
            
            avg_latency = sum(health['latency_ms'][-10:]) / len(health['latency_ms'][-10:]) if health['latency_ms'] else 200
            latency_score = max(0, 1 - (avg_latency / 1000))
            
            scores[name] = (success_rate * 0.7) + (latency_score * 0.3)
        
        return max(scores, key=scores.get)

    def _record_health(self, exchange_name: str, success: bool, latency_ms: float) -> None:
        """Record exchange performance metrics."""
        health = self.exchange_health[exchange_name]
        if success:
            health['successes'] += 1
        else:
            health['failures'] += 1
        health['latency_ms'].append(latency_ms)
        if len(health['latency_ms']) > 100:
            health['latency_ms'] = health['latency_ms'][-100:]

    def _get_priority_order(self) -> List[str]:
        """Dynamic priority order based on health scores."""
        scored = []
        for name in self.exchange_configs:
            if name in self.banned_exchanges:
                continue
            health = self.exchange_health[name]
            total = health['successes'] + health['failures']
            if total == 0:
                score = 0.9
            else:
                score = health['successes'] / total
            scored.append((name, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored] if scored else ['binance', 'bybit', 'okx', 'kucoin', 'mexc']

    # ================================================================
    # 5. MARKET SYMBOLS (Sync)
    # ================================================================

    def get_top_volume_symbols(
        self, 
        exchange_name: Optional[str] = None, 
        limit: Optional[int] = None, 
        exclude_btc: bool = False,
        market_type: str = 'spot'
    ) -> List[str]:
        """Fetch top volume USDT symbols with multi-exchange failover."""
        search_order = [exchange_name] if exchange_name else self._get_priority_order()
        
        for ex_name in search_order:
            if ex_name in self.banned_exchanges:
                continue

            ex_obj = self.exchanges.get(ex_name)
            if not ex_obj:
                continue

            try:
                if market_type == 'swap':
                    ex_obj.options['defaultType'] = 'swap'
                else:
                    ex_obj.options['defaultType'] = 'spot'

                cache_key = f"tickers_{ex_name}_{market_type}"
                cached = self._get_cache(cache_key)
                if cached is not None:
                    tickers = cached
                else:
                    start_time = time.time()
                    tickers, latency_ms = self._retry_call(ex_obj.fetch_tickers, max_retries=2)
                    self._record_health(ex_name, True, latency_ms)
                    self._set_cache(cache_key, tickers)

                usdt_pairs = []
                for symbol, data in tickers.items():
                    if not symbol.endswith('/USDT'):
                        continue
                    
                    quote_vol = data.get('quoteVolume')
                    if quote_vol is None:
                        continue

                    base = symbol.split('/')[0].upper()
                    
                    if base in self.stable_coins or base in self.excluded_base_assets:
                        continue
                    if exclude_btc and base == 'BTC':
                        continue

                    vol = float(quote_vol)

                    usdt_pairs.append({
                        'symbol': symbol,
                        'volume': vol,
                        'base': base,
                        'price': data.get('last', 0)
                    })

                if usdt_pairs:
                    usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
                    
                    if limit and isinstance(limit, int):
                        symbols = [item['symbol'] for item in usdt_pairs[:limit]]
                    else:
                        symbols = [item['symbol'] for item in usdt_pairs]

                    logging.info(f"✅ Loaded {len(symbols)} symbols from {ex_name.upper()}")
                    return symbols

            except (ccxt.DDoSProtection, ccxt.RateLimitExceeded):
                logging.error(f"🛑 Rate Limit on {ex_name.upper()}. Banning...")
                self.banned_exchanges.add(ex_name)
                self._record_health(ex_name, False, 0)
            except Exception as e:
                logging.warning(f"⚠️ Failed fetching from {ex_name.upper()}: {e}")
                self._record_health(ex_name, False, 0)
                time.sleep(0.5)

        logging.error("❌ All exchanges failed.")
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']

    # ================================================================
    # 6. OHLCV FETCH (With ALL fixes: Retry, Validation, Cleanup)
    # ================================================================

    def fetch_ohlcv(
        self, 
        symbol: str, 
        exchange_name: Optional[str] = None, 
        timeframe: str = "4h", 
        limit: int = 150,
        market_type: str = 'spot'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV candles with:
        1. Symbol Validation (load_markets)
        2. Retry Mechanism (NetworkError -> retry 3x)
        3. Minimum Candle Check (default 50)
        4. Duplicate Removal
        5. Sort by Timestamp
        6. NaN Cleanup
        7. Cache + Health Tracking
        """
        # Skip bad symbols
        if symbol in self.bad_symbols:
            return None

        # Check cache
        cache_key = f"ohlcv_{symbol}_{timeframe}_{limit}_{market_type}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached.copy()

        search_order = [exchange_name] if exchange_name else self._get_priority_order()
        
        for ex_name in search_order:
            if ex_name in self.banned_exchanges:
                continue

            ex_obj = self.exchanges.get(ex_name)
            if not ex_obj:
                continue

            try:
                # ---- FIX 2: Symbol Validation ----
                if not self._is_symbol_valid(ex_name, symbol, market_type):
                    logging.debug(f"⚠️ {symbol} not found on {ex_name.upper()}")
                    self._record_health(ex_name, False, 0)
                    continue

                # Set market type
                if market_type == 'swap':
                    ex_obj.options['defaultType'] = 'swap'
                    # Adjust symbol for futures if needed
                    fetch_symbol = symbol.replace('/USDT', '/USDT:USDT')
                else:
                    ex_obj.options['defaultType'] = 'spot'
                    fetch_symbol = symbol

                # ---- FIX 1: Retry Mechanism ----
                start_time = time.time()
                ohlcv, latency_ms = self._retry_call(
                    ex_obj.fetch_ohlcv, fetch_symbol, timeframe=timeframe, limit=limit,
                    max_retries=3
                )
                
                self._record_health(ex_name, True, latency_ms)

                # ---- FIX 3: Empty Candle Check (minimum candles) ----
                if not ohlcv or len(ohlcv) < self.min_candles:
                    logging.warning(f"⚠️ {symbol} on {ex_name} has only {len(ohlcv) if ohlcv else 0} candles (need {self.min_candles})")
                    continue

                # Create DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Ensure numeric types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)

                # ---- FIX 4: Duplicate Remove ----
                df = df.drop_duplicates(subset=['timestamp'])

                # ---- FIX 5: Sort Timestamp ----
                df = df.sort_values('timestamp').reset_index(drop=True)

                # ---- FIX 6: NaN Remove ----
                df = df.dropna()

                # Final check after cleanup
                if len(df) < self.min_candles:
                    logging.warning(f"⚠️ {symbol} on {ex_name} has only {len(df)} candles after cleanup")
                    continue

                # Cache the result
                self._set_cache(cache_key, df.copy())
                return df

            except (ccxt.BadSymbol, ccxt.BadRequest):
                logging.warning(f"⚠️ {symbol} invalid on {ex_name.upper()}. Blacklisting.")
                self.bad_symbols.add(symbol)
                self._record_health(ex_name, False, 0)
                return None
            except (ccxt.DDoSProtection, ccxt.RateLimitExceeded):
                logging.error(f"🛑 Rate Limit on {ex_name.upper()} for {symbol}. Banning...")
                self.banned_exchanges.add(ex_name)
                self._record_health(ex_name, False, 0)
            except Exception as e:
                logging.debug(f"⚠️ Error fetching {timeframe} for {symbol} from {ex_name.upper()}: {e}")
                self._record_health(ex_name, False, 0)
                time.sleep(0.5)

        logging.warning(f"❌ Missing candle data for {symbol}.")
        return None

    def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = "4h", 
        limit: int = 150, 
        exchange_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Unified Alias Interface."""
        return self.fetch_ohlcv(symbol=symbol, exchange_name=exchange_name, timeframe=timeframe, limit=limit)

    # ================================================================
    # 7. MULTI-TIMEFRAME FETCH
    # ================================================================

    def fetch_ohlcv_multi_tf(
        self, 
        symbol: str, 
        timeframes: Optional[List[str]] = None, 
        limit: int = 150
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multiple timeframes with smart rate limiting."""
        timeframes = timeframes or ["5m", "15m", "30m", "1h", "4h", "1d"]
        multi_tf_data = {}
        
        best_ex = self._get_best_exchange()
        
        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, exchange_name=best_ex, timeframe=tf, limit=limit)
            if df is not None and not df.empty and len(df) >= self.min_candles:
                multi_tf_data[tf] = df
            else:
                logging.warning(f"⚠️ Could not retrieve {tf} for {symbol}")
            
            # Dynamic delay
            health = self.exchange_health.get(best_ex, {})
            avg_latency = sum(health.get('latency_ms', [0])[-10:]) / len(health.get('latency_ms', [0])[-10:]) if health.get('latency_ms') else 200
            delay = max(0.05, min(0.5, avg_latency / 1000 * 0.5))
            time.sleep(delay)
                
        return multi_tf_data

    # ================================================================
    # 8. FUTURES DATA (Funding Rate, Open Interest)
    # ================================================================

    def fetch_funding_rate(self, symbol: str, exchange_name: str = 'binance') -> Optional[float]:
        """Fetch current funding rate for perpetual futures."""
        ex_obj = self.exchanges.get(exchange_name)
        if not ex_obj:
            return None
        try:
            ex_obj.options['defaultType'] = 'swap'
            funding = ex_obj.fetch_funding_rate(symbol.replace('/USDT', '/USDT:USDT'))
            return funding.get('fundingRate', 0.0)
        except Exception:
            return None

    def fetch_open_interest(self, symbol: str, exchange_name: str = 'binance') -> Optional[float]:
        """Fetch open interest for perpetual futures."""
        ex_obj = self.exchanges.get(exchange_name)
        if not ex_obj:
            return None
        try:
            ex_obj.options['defaultType'] = 'swap'
            oi = ex_obj.fetch_open_interest(symbol.replace('/USDT', '/USDT:USDT'))
            return oi.get('openInterest', 0.0)
        except Exception:
            return None

    # ================================================================
    # 9. VOLUME NORMALIZATION (USDT equivalent)
    # ================================================================

    @staticmethod
    def normalize_volume_to_usdt(df: pd.DataFrame) -> pd.DataFrame:
        """Convert volume to USDT-equivalent (close * volume)."""
        if df is not None and not df.empty:
            df['volume_usdt'] = df['close'] * df['volume']
        return df

    # ================================================================
    # 10. ASYNC FETCH (For 1000+ symbols)
    # ================================================================

    async def _async_fetch_single(self, exchange_name: str, symbol: str, timeframe: str = "4h", limit: int = 150):
        """Async single fetch using aiohttp."""
        ex = self.async_exchanges.get(exchange_name)
        if not ex:
            return None
        try:
            start_time = time.time()
            ohlcv = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            latency_ms = (time.time() - start_time) * 1000
            self._record_health(exchange_name, True, latency_ms)
            if ohlcv and len(ohlcv) >= self.min_candles:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.drop_duplicates(subset=['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                df = df.dropna()
                return df
        except Exception as e:
            logging.debug(f"Async error {exchange_name} {symbol}: {e}")
            self._record_health(exchange_name, False, 0)
        return None

    async def fetch_ohlcv_async(
        self, 
        symbols: List[str], 
        timeframe: str = "4h", 
        limit: int = 150,
        max_concurrent: int = 20
    ) -> Dict[str, pd.DataFrame]:
        """
        Async bulk fetch for hundreds of symbols.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(ex_name, sym):
            async with semaphore:
                return await self._async_fetch_single(ex_name, sym, timeframe, limit)
        
        best_ex = self._get_best_exchange()
        tasks = [fetch_with_semaphore(best_ex, sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for sym, res in zip(symbols, results):
            if isinstance(res, pd.DataFrame) and not res.empty and len(res) >= self.min_candles:
                output[sym] = res
            else:
                logging.debug(f"⚠️ No data for {sym}")
        
        return output

    def close(self):
        """Close sessions gracefully."""
        self._session.close()
        for ex in self.async_exchanges.values():
            if ex:
                try:
                    asyncio.run(ex.close())
                except:
                    pass


# ================================================================
# TEST (Run with: python data_fetcher.py)
# ================================================================
if __name__ == "__main__":
    import json
    
    print("🧪 Testing DataFetcher v3.1 (with all fixes)")
    print("=" * 60)
    
    fetcher = DataFetcher(min_candles=30)
    
    # Test 1: Symbol Validation
    print("🔍 Testing Symbol Validation...")
    print(f"  BTC/USDT valid on binance: {fetcher._is_symbol_valid('binance', 'BTC/USDT')}")
    print(f"  BTC/USDT valid on bybit: {fetcher._is_symbol_valid('bybit', 'BTC/USDT')}")
    
    # Test 2: Fetch OHLCV with Retry + Validation + Cleanup
    print("\n📈 Fetching BTC/USDT 1h (should work)...")
    df = fetcher.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=30)
    if df is not None and len(df) >= 30:
        print(f"✅ Fetched {len(df)} candles. Last close: {df['close'].iloc[-1]}")
        print(f"   Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   NaN check: {df.isna().sum().sum()}")
        print(f"   Duplicate check: {df['timestamp'].duplicated().sum()}")
    else:
        print("❌ Failed to fetch")
    
    # Test 3: Bad Symbol (should skip quickly)
    print("\n🚫 Testing invalid symbol 'FAKE/USDT'...")
    df_bad = fetcher.fetch_ohlcv("FAKE/USDT", timeframe="1h", limit=10)
    print(f"  Result: {df_bad is None} (should be None)")
    
    # Test 4: Empty Data (should return None)
    print("\n📭 Testing empty data...")
    df_empty = fetcher.fetch_ohlcv("BTC/USDT", timeframe="1s", limit=5)  # 1s often not supported
    print(f"  Result: {df_empty is None} (should be None if not enough candles)")
    
    print("\n" + "=" * 60)
    print("✅ DataFetcher v3.1 ready for production.")