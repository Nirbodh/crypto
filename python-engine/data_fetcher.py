import ccxt
import pandas as pd
import time
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataFetcher:
    def __init__(self):
        # Exchange Priority Order: Bybit -> MEXC -> KuCoin -> OKX -> Binance (Fallback)
        self.exchanges = {
            'bybit': ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'spot'}}),
            'mexc': ccxt.mexc({'enableRateLimit': True}),
            'kucoin': ccxt.kucoin({'enableRateLimit': True}),
            'okx': ccxt.okx({'enableRateLimit': True}),
            'binance': ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        }
        
        self.banned_exchanges = set()

        self.stable_coins = {
            'USDT', 'USDC', 'BUSD', 'FDUSD', 'DAI', 'TUSD', 
            'USDD', 'USD1', 'PYUSD', 'USDE', 'USDP', 'USDS',
            'EUR', 'GBP', 'AUD', 'BRL', 'TRY', 'RUB'
        }
        self.excluded_base_assets = {
            'BANK', 'TEST', 'FAKE', 'XUSD', 'XAUT', 'PAXG',
            'BULL', 'BEAR', 'UP', 'DOWN', '3L', '3S', '2L', '2S',
            # Stocks & Synthetics Filter
            'COIN', 'ORCL', 'TSM', 'ARM', 'SMCI', 'DELL', 'NATGAS', 
            'ANTHROPIC', 'HOOD', 'RKLB', 'TQQQ', 'AAPL', 'MSFT', 'NVDA'
        }

    def get_top_volume_symbols(
        self, 
        exchange_name: Optional[str] = None, 
        limit: Optional[int] = None, 
        exclude_btc: bool = False
    ) -> List[str]:
        """Fetch top volume USDT symbols with multi-exchange failover."""
        search_order = [exchange_name] if exchange_name else ['bybit', 'mexc', 'kucoin', 'okx', 'binance']
        
        for ex_name in search_order:
            if ex_name in self.banned_exchanges:
                continue

            ex_obj = self.exchanges.get(ex_name)
            if not ex_obj:
                continue

            try:
                logging.info(f"🔄 Fetching Market Symbols from {ex_name.upper()}...")
                tickers = ex_obj.fetch_tickers()
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
                        'volume': float(quote_vol)
                    })

                if usdt_pairs:
                    usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
                    
                    if limit is not None and isinstance(limit, int):
                        symbols = [item['symbol'] for item in usdt_pairs[:limit]]
                    else:
                        symbols = [item['symbol'] for item in usdt_pairs]

                    logging.info(f"✅ Successfully loaded {len(symbols)} symbols from {ex_name.upper()}")
                    return symbols

            except (ccxt.DDoSProtection, ccxt.RateLimitExceeded):
                logging.error(f"🛑 IP Ban detected on {ex_name.upper()}. Switching exchange...")
                self.banned_exchanges.add(ex_name)
            except Exception as e:
                logging.warning(f"⚠️ Failed fetching symbols from {ex_name.upper()}: {e}. Trying next exchange...")
                time.sleep(0.5)

        logging.error("❌ All exchanges failed to retrieve market symbols.")
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']

    def fetch_ohlcv(
        self, 
        symbol: str, 
        exchange_name: Optional[str] = None, 
        timeframe: str = "4h", 
        limit: int = 150
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV candles with automatic Exchange Rotation on error."""
        search_order = [exchange_name] if exchange_name else ['bybit', 'mexc', 'kucoin', 'okx', 'binance']
        
        for ex_name in search_order:
            if ex_name in self.banned_exchanges:
                continue

            ex_obj = self.exchanges.get(ex_name)
            if not ex_obj:
                continue

            try:
                ohlcv = ex_obj.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                
                if ohlcv and len(ohlcv) > 0:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    return df

            except (ccxt.DDoSProtection, ccxt.RateLimitExceeded):
                logging.error(f"🛑 Rate limit/IP Ban hit on {ex_name.upper()}. Blacklisting for this session...")
                self.banned_exchanges.add(ex_name)
            except Exception as e:
                logging.debug(f"⚠️ Error fetching {timeframe} for {symbol} from {ex_name.upper()}: {e}")
                time.sleep(0.3)

        logging.warning(f"❌ Missing candle data for {symbol} across ALL queried exchanges.")
        return None

    def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = "4h", 
        limit: int = 150, 
        exchange_name: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Unified Alias Interface required by Main Orchestrator Engine.
        Maps directly to fetch_ohlcv.
        """
        return self.fetch_ohlcv(symbol=symbol, exchange_name=exchange_name, timeframe=timeframe, limit=limit)

    def fetch_ohlcv_multi_tf(
        self, 
        symbol: str, 
        timeframes: Optional[List[str]] = None, 
        limit: int = 150
    ) -> Dict[str, pd.DataFrame]:
        """Helper method with rate-limit protection between timeframe queries."""
        timeframes = timeframes or ["5m", "15m", "30m", "1h", "4h", "1d"]
        multi_tf_data = {}
        
        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            if df is not None and not df.empty:
                multi_tf_data[tf] = df
            else:
                logging.warning(f"⚠️ Could not retrieve {tf} timeframe data for {symbol}")
            
            # Smart delay to prevent rate limit spikes
            time.sleep(0.15)
                
        return multi_tf_data