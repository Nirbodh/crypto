# python-engine/data_fetcher.py

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ================================================================
# DATA STRUCTURES (new – non‑breaking)
# ================================================================

@dataclass
class OHLCVResult:
    success: bool
    data: Optional[pd.DataFrame] = None
    quality: float = 0.0            # 0‑100
    exchange: Optional[str] = None
    latency_ms: float = 0.0
    reason: Optional[str] = None    # failure reason

# ================================================================
# CACHE BACKEND ABSTRACTION (future‑proof)
# ================================================================

class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        pass


class MemoryCache(CacheBackend):
    def __init__(self):
        self._store = {}

    def get(self, key):
        if key in self._store:
            value, timestamp = self._store[key]
            if time.time() < timestamp:
                return value
            else:
                del self._store[key]
        return None

    def set(self, key, value, ttl_seconds):
        self._store[key] = (value, time.time() + ttl_seconds)

    def delete(self, key):
        self._store.pop(key, None)


# Placeholder for Redis (swap when ready)
class RedisCache(CacheBackend):
    def __init__(self, connection_params=None):
        raise NotImplementedError("RedisCache is reserved for future use")
    def get(self, key): pass
    def set(self, key, value, ttl_seconds): pass
    def delete(self, key): pass


# ================================================================
# MAIN DATA FETCHER (v4.0)
# ================================================================

class DataFetcher:
    """
    v4.0 – Institutional Grade Data Fetcher
    - Dynamic min_candles per indicator
    - Exchange cooldown instead of permanent ban
    - Smart exchange fallback (top‑2 healthy)
    - Cache abstraction (Memory / Redis ready)
    - Timestamp gap detection + quality score
    - Structured OHLCVResult for future engines
    - Parallel async multi‑timeframe
    - 100% backward‑compatible: get_ohlcv() still returns pd.DataFrame
    """

    def __init__(self,
                 enable_async: bool = True,
                 cache_ttl_seconds: int = 60,
                 default_min_candles: int = 50,
                 cache_backend: Optional[CacheBackend] = None):
        self.enable_async = enable_async
        self.cache_ttl_seconds = cache_ttl_seconds
        self.default_min_candles = default_min_candles

        # Cache (abstraction)
        self.cache = cache_backend or MemoryCache()

        # Exchange configs
        self.exchange_configs = {
            'bybit': {'enableRateLimit': True, 'options': {'defaultType': 'spot'}},
            'mexc': {'enableRateLimit': True},
            'kucoin': {'enableRateLimit': True},
            'okx': {'enableRateLimit': True},
            'binance': {'enableRateLimit': True, 'options': {'defaultType': 'spot'}}
        }

        # Sync exchanges (for main get_ohlcv)
        self.exchanges = {}
        for name, config in self.exchange_configs.items():
            try:
                self.exchanges[name] = ccxt.bybit(config) if name == 'bybit' else getattr(ccxt, name)(config)
            except Exception:
                self.exchanges[name] = None

        # Async exchanges
        self.async_exchanges = {}
        if self.enable_async:
            for name, config in self.exchange_configs.items():
                try:
                    self.async_exchanges[name] = ccxt_async.bybit(config) if name == 'bybit' else getattr(ccxt_async, name)(config)
                except Exception:
                    self.async_exchanges[name] = None

        # State
        self.bad_symbols = set()                  # permanently invalid symbols
        self.banned_until = {}                    # exchange -> cooldown end timestamp
        self.exchange_health = {
            name: {"successes": 0, "failures": 0, "latency_ms": []}
            for name in self.exchange_configs
        }

        # Filters (for get_top_volume_symbols)
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
    # HEALTH & PRIORITY (P2, P3)
    # ================================================================

    def _record_health(self, exchange_name: str, success: bool, latency_ms: float):
        health = self.exchange_health[exchange_name]
        if success:
            health['successes'] += 1
        else:
            health['failures'] += 1
        health['latency_ms'].append(latency_ms)
        if len(health['latency_ms']) > 100:
            health['latency_ms'] = health['latency_ms'][-100:]

    def _ban_exchange_temp(self, exchange_name: str, duration_seconds: int = 300):
        """Cooldown instead of permanent ban (P3)."""
        self.banned_until[exchange_name] = time.time() + duration_seconds
        logging.warning(f"🚫 {exchange_name.upper()} rate limited, cooldown {duration_seconds}s")

    def _is_exchange_banned(self, exchange_name: str) -> bool:
        if exchange_name in self.banned_until:
            if time.time() < self.banned_until[exchange_name]:
                return True
            else:
                del self.banned_until[exchange_name]  # cooldown expired
        return False

    def _get_priority_order(self, max_fallback: int = 2) -> List[str]:
        """Top N healthy exchanges by success rate (P2)."""
        scored = []
        for name in self.exchange_configs:
            if self._is_exchange_banned(name):
                continue
            health = self.exchange_health[name]
            total = health['successes'] + health['failures']
            if total == 0:
                score = 0.95
            else:
                score = health['successes'] / total
            scored.append((name, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored[:max_fallback]]

    # ================================================================
    # CACHING (P9: exchange‑specific cache key)
    # ================================================================

    def _cache_key(self, exchange: str, symbol: str, timeframe: str, limit: int, market_type: str) -> str:
        return f"{exchange}_{symbol}_{timeframe}_{limit}_{market_type}"

    # ================================================================
    # SYMBOL VALIDATION
    # ================================================================

    def _load_markets(self, exchange_name: str, market_type: str = 'spot') -> Optional[Dict]:
        ex_obj = self.exchanges.get(exchange_name)
        if not ex_obj:
            return None
        cache_key = f"markets_{exchange_name}_{market_type}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            if market_type == 'swap':
                ex_obj.options['defaultType'] = 'swap'
            else:
                ex_obj.options['defaultType'] = 'spot'
            markets = ex_obj.load_markets()
            self.cache.set(cache_key, markets, self.cache_ttl_seconds * 5)
            return markets
        except Exception as e:
            logging.debug(f"⚠️ Failed to load markets for {exchange_name}: {e}")
            return None

    def _is_symbol_valid(self, exchange_name: str, symbol: str, market_type: str = 'spot') -> bool:
        markets = self._load_markets(exchange_name, market_type)
        if markets is None:
            return False
        if market_type == 'swap':
            alt_symbol = symbol.replace('/USDT', '/USDT:USDT')
            return symbol in markets or alt_symbol in markets
        return symbol in markets

    # ================================================================
    # RETRY MECHANISM
    # ================================================================

    def _retry_call(self, func, *args, max_retries: int = 3, **kwargs):
        last_exception = None
        for attempt in range(max_retries):
            try:
                start = time.time()
                result = func(*args, **kwargs)
                latency = (time.time() - start) * 1000
                return result, latency
            except (ccxt.NetworkError, ccxt.DDoSProtection, ccxt.RateLimitExceeded) as e:
                wait = (2 ** attempt) + 0.5
                logging.warning(f"⚠️ Retry {attempt+1}/{max_retries} after {wait:.1f}s: {e}")
                time.sleep(wait)
                last_exception = e
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                raise e
        return None, 0

    # ================================================================
    # QUALITY & GAP DETECTION (P5, P6)
    # ================================================================

    def _compute_quality(self, df: pd.DataFrame, expected_limit: int, timeframe: str,
                         exchange: str, latency_ms: float) -> float:
        """Score 0‑100 based on completeness, gaps, exchange reliability."""
        score = 100.0
        # Penalty for missing candles
        if len(df) < expected_limit:
            ratio = len(df) / expected_limit
            score -= 30 * (1 - ratio)
        # Gap detection (P5)
        if len(df) > 1:
            freq = pd.Timedelta(timeframe).total_seconds()
            time_diffs = df['timestamp'].diff().dropna()
            expected_diff = pd.Timedelta(seconds=freq)
            gaps = time_diffs[time_diffs > expected_diff * 1.5]
            if not gaps.empty:
                penalty = min(20, 5 * len(gaps))
                score -= penalty
                logging.warning(f"⚠️ {exchange} {timeframe}: "
                                f"found {len(gaps)} timestamp gaps. Largest: {gaps.max()}")
        # Exchange reliability bonus
        health = self.exchange_health[exchange]
        total = health['successes'] + health['failures']
        if total > 0:
            success_rate = health['successes'] / total
            score += 5 * success_rate
        # Latency penalty
        if latency_ms > 500:
            score -= min(5, (latency_ms - 500) / 100)
        return max(0, min(100, score))

    # ================================================================
    # INTERNAL CORE FETCH (returns OHLCVResult)
    # ================================================================

    def _fetch_ohlcv_internal(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 150,
        exchange_name: Optional[str] = None,
        market_type: str = 'spot',
        min_candles: Optional[int] = None
    ) -> OHLCVResult:
        """All the logic, returns structured result."""
        min_req = min_candles if min_candles is not None else self.default_min_candles

        # Permanently blacklisted symbol
        if symbol in self.bad_symbols:
            return OHLCVResult(False, reason="permanently_blacklisted")

        # Determine search order (top 2 healthy exchanges)
        search_order = [exchange_name] if exchange_name else self._get_priority_order(max_fallback=2)

        for ex_name in search_order:
            if self._is_exchange_banned(ex_name):
                continue
            ex_obj = self.exchanges.get(ex_name)
            if not ex_obj:
                continue

            # 1) Exchange‑specific cache check
            cache_key = self._cache_key(ex_name, symbol, timeframe, limit, market_type)
            cached = self.cache.get(cache_key)
            if cached is not None:
                # Cached data – assign quality manually for cached hits
                return OHLCVResult(True, cached.copy(), quality=95.0, exchange=ex_name)

            # 2) Symbol validation
            if not self._is_symbol_valid(ex_name, symbol, market_type):
                logging.debug(f"⚠️ {symbol} invalid on {ex_name.upper()}")
                self._record_health(ex_name, False, 0)
                continue

            try:
                # Set market type
                if market_type == 'swap':
                    ex_obj.options['defaultType'] = 'swap'
                    fetch_symbol = symbol.replace('/USDT', '/USDT:USDT')
                else:
                    ex_obj.options['defaultType'] = 'spot'
                    fetch_symbol = symbol

                # 3) Fetch with retry
                ohlcv_raw, latency = self._retry_call(
                    ex_obj.fetch_ohlcv, fetch_symbol, timeframe=timeframe, limit=limit,
                    max_retries=3
                )
                self._record_health(ex_name, True, latency)

                # 4) Minimum candle check (P1)
                if not ohlcv_raw or len(ohlcv_raw) < min_req:
                    logging.warning(
                        f"🔍 {symbol} | Exchange={ex_name.upper()} | TF={timeframe} | "
                        f"Candles={len(ohlcv_raw) if ohlcv_raw else 0} | Required={min_req} | "
                        f"Reason=InsufficientData"
                    )
                    continue

                # 5) Build DataFrame + clean
                df = pd.DataFrame(ohlcv_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                df = df.drop_duplicates(subset=['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                df = df.dropna()

                if len(df) < min_req:
                    logging.warning(
                        f"🔍 {symbol} | Exchange={ex_name.upper()} | TF={timeframe} | "
                        f"Candles={len(df)} after cleanup | Required={min_req} | "
                        f"Reason=InsufficientAfterCleanup"
                    )
                    continue

                # 6) Quality score + cache
                quality = self._compute_quality(df, limit, timeframe, ex_name, latency)
                self.cache.set(cache_key, df.copy(), self.cache_ttl_seconds)

                return OHLCVResult(True, df, quality=quality, exchange=ex_name, latency_ms=latency)

            except (ccxt.BadSymbol, ccxt.BadRequest):
                logging.warning(f"🚫 {symbol} permanently invalid on {ex_name.upper()}")
                self.bad_symbols.add(symbol)
                self._record_health(ex_name, False, 0)
                return OHLCVResult(False, reason="bad_symbol", exchange=ex_name)
            except (ccxt.DDoSProtection, ccxt.RateLimitExceeded):
                self._ban_exchange_temp(ex_name, 300)  # P3
                self._record_health(ex_name, False, 0)
                logging.error(f"🛑 {ex_name.upper()} rate limited, cooldown 300s")
            except Exception as e:
                self._record_health(ex_name, False, 0)
                logging.warning(f"❌ Fetch error {symbol} {timeframe} from {ex_name.upper()}: {e}")

        return OHLCVResult(False, reason="all_exchanges_failed")

    # ================================================================
    # PUBLIC API – 100% BACKWARD COMPATIBLE
    # ================================================================

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 150,
        exchange_name: Optional[str] = None,
        market_type: str = 'spot',
        min_candles: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Standard method – returns pd.DataFrame or None.
        (Used by all existing engines: main, technical, smc, liquidity…)
        """
        result = self._fetch_ohlcv_internal(symbol, timeframe, limit,
                                            exchange_name, market_type, min_candles)
        if not result.success or result.data is None:
            return None
        # Attach metadata as DataFrame.attrs (non‑destructive)
        result.data.attrs['quality'] = result.quality
        result.data.attrs['exchange'] = result.exchange
        result.data.attrs['latency_ms'] = result.latency_ms
        return result.data

    def fetch_ohlcv_result(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 150,
        exchange_name: Optional[str] = None,
        market_type: str = 'spot',
        min_candles: Optional[int] = None
    ) -> OHLCVResult:
        """
        New method – returns full OHLCVResult with quality, exchange, etc.
        For gradual migration of advanced engines.
        """
        return self._fetch_ohlcv_internal(symbol, timeframe, limit,
                                          exchange_name, market_type, min_candles)

    def get_ohlcv(self, symbol, timeframe="4h", limit=150, exchange_name=None) -> Optional[pd.DataFrame]:
        """Legacy alias – exactly as before, returns DataFrame or None."""
        return self.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit,
                                exchange_name=exchange_name, market_type='spot')

    # ================================================================
    # MULTI‑TIMEFRAME (SYNC) – uses fetch_ohlcv internally
    # ================================================================

    def fetch_ohlcv_multi_tf(self, symbol, timeframes=None, limit=150) -> Dict[str, Optional[pd.DataFrame]]:
        """Returns dict of DataFrame (or None) per timeframe."""
        timeframes = timeframes or ["5m", "15m", "30m", "1h", "4h", "1d"]
        results = {}
        best_ex = self._get_priority_order(max_fallback=1)
        best_ex = best_ex[0] if best_ex else 'binance'
        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, timeframe=tf, limit=limit, exchange_name=best_ex)
            results[tf] = df
            time.sleep(0.1)  # gentle rate limit
        return results

    # ================================================================
    # ASYNC PARALLEL MULTI‑TF (P7) – returns DataFrame dict for now
    # ================================================================

    async def async_fetch_ohlcv_multi_tf(self, symbol: str, timeframes: List[str] = None,
                                         limit: int = 150, min_candles: Optional[int] = None) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch all timeframes in parallel for one symbol, returns DataFrame dict."""
        timeframes = timeframes or ["5m", "15m", "30m", "1h", "4h", "1d"]
        best_ex = self._get_priority_order(max_fallback=1)
        best_ex = best_ex[0] if best_ex else 'binance'

        async def fetch_one(tf):
            # Use the internal async method if available, else fallback to sync in thread
            # For simplicity we reuse the sync internal (since we have async exchanges)
            ex = self.async_exchanges.get(best_ex)
            if not ex:
                return None
            try:
                # Use our own async logic
                start = time.time()
                ohlcv = await ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
                latency = (time.time() - start) * 1000
                if not ohlcv or len(ohlcv) < (min_candles or self.default_min_candles):
                    return None
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True).dropna()
                # Cache
                cache_key = self._cache_key(best_ex, symbol, tf, limit, 'spot')
                self.cache.set(cache_key, df.copy(), self.cache_ttl_seconds)
                return df
            except Exception as e:
                logging.warning(f"Async fetch error {symbol} {tf}: {e}")
                return None

        tasks = [fetch_one(tf) for tf in timeframes]
        results_list = await asyncio.gather(*tasks)
        return dict(zip(timeframes, results_list))

    # ================================================================
    # TOP VOLUME SYMBOLS (mostly unchanged)
    # ================================================================

    def get_top_volume_symbols(self, exchange_name=None, limit=None, exclude_btc=False, market_type='spot') -> List[str]:
        """Fetch top volume USDT symbols (with fallback)."""
        search_order = [exchange_name] if exchange_name else self._get_priority_order(max_fallback=2)
        for ex_name in search_order:
            if self._is_exchange_banned(ex_name):
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
                cached = self.cache.get(cache_key)
                if cached is not None:
                    tickers = cached
                else:
                    tickers, latency = self._retry_call(ex_obj.fetch_tickers, max_retries=2)
                    self._record_health(ex_name, True, latency)
                    self.cache.set(cache_key, tickers, self.cache_ttl_seconds)

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
                    usdt_pairs.append({
                        'symbol': symbol,
                        'volume': float(quote_vol),
                        'base': base,
                        'price': data.get('last', 0)
                    })
                if usdt_pairs:
                    usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
                    symbols = [item['symbol'] for item in usdt_pairs[:limit]] if limit else [item['symbol'] for item in usdt_pairs]
                    logging.info(f"✅ Loaded {len(symbols)} symbols from {ex_name.upper()}")
                    return symbols
            except (ccxt.DDoSProtection, ccxt.RateLimitExceeded):
                self._ban_exchange_temp(ex_name, 300)
                logging.error(f"🛑 Rate limit on {ex_name.upper()}, cooldown 300s")
            except Exception as e:
                logging.warning(f"⚠️ Failed fetching from {ex_name.upper()}: {e}")
                self._record_health(ex_name, False, 0)
        logging.error("❌ All exchanges failed for top symbols.")
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']

    # ================================================================
    # FUTURES HELPERS
    # ================================================================

    def fetch_funding_rate(self, symbol: str, exchange_name: str = 'binance') -> Optional[float]:
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
    # UTILITY
    # ================================================================

    @staticmethod
    def normalize_volume_to_usdt(df: pd.DataFrame) -> pd.DataFrame:
        if df is not None and not df.empty:
            df['volume_usdt'] = df['close'] * df['volume']
        return df

    def close(self):
        for ex in self.async_exchanges.values():
            if ex:
                try:
                    asyncio.run(ex.close())
                except:
                    pass

# ================================================================
# QUICK TEST
# ================================================================
if __name__ == "__main__":
    print("🧪 Testing DataFetcher v4.0 (backward‑compatible)")
    fetcher = DataFetcher(min_candles=30)

    # Old‑style call (returns DataFrame or None)
    df = fetcher.get_ohlcv("BTC/USDT", "1h", limit=30)
    if df is not None:
        print(f"✅ Standard get_ohlcv: {len(df)} candles, last close {df['close'].iloc[-1]}")
        # New attributes available (optional)
        print(f"   Data attrs: exchange={df.attrs.get('exchange')}, quality={df.attrs.get('quality')}")

    # New structured call (returns OHLCVResult)
    res = fetcher.fetch_ohlcv_result("BTC/USDT", "1h", limit=30, min_candles=14)
    if res.success:
        print(f"✅ Structured result: exchange={res.exchange}, quality={res.quality:.1f}")
    else:
        print(f"❌ Failed: {res.reason}")

    # Invalid symbol test
    df_bad = fetcher.get_ohlcv("FAKE/USDT", "1h", limit=10)
    print(f"❌ Bad symbol returns: {df_bad is None} (expected True)")

    # Cooldown test (rate limit simulation)
    # Not executed but mechanism is in place.
