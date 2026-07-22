# python-engine/data_fetcher.py
import ccxt
import pandas as pd

class DataFetcher:
    def __init__(self):
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'bybit': ccxt.bybit({'enableRateLimit': True}),
            'mexc': ccxt.mexc({'enableRateLimit': True}),
            'kucoin': ccxt.kucoin({'enableRateLimit': True})
        }
        
        # সম্প্রসারিত স্টেবলকয়েন + বাদ দেয়ার তালিকা
        self.stable_coins = [
            'USDT', 'USDC', 'BUSD', 'FDUSD', 'DAI', 'TUSD', 
            'USDD', 'USD1', 'PYUSD', 'USDE', 'USDP', 'USDS'
        ]
        self.excluded_base_assets = {
            'BANK', 'TEST', 'FAKE', 'XUSD'  # প্রয়োজনমতো বাড়ান
        }

    def get_top_volume_symbols(self, exchange_name='binance', limit=10, exclude_btc=True):
        try:
            ex_obj = self.exchanges.get(exchange_name)
            if not ex_obj:
                return []

            tickers = ex_obj.fetch_tickers()
            usdt_pairs = []

            for symbol, data in tickers.items():
                if not symbol.endswith('/USDT'):
                    continue
                if data.get('quoteVolume') is None:
                    continue

                base = symbol.split('/')[0]
                
                if base in self.stable_coins:
                    continue
                if base in self.excluded_base_assets:
                    continue
                if exclude_btc and base == 'BTC':
                    continue

                usdt_pairs.append({
                    'symbol': symbol,
                    'volume': float(data['quoteVolume'])
                })

            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            return [item['symbol'] for item in usdt_pairs[:limit]]

        except Exception as e:
            print(f"❌ Error fetching symbols from {exchange_name}: {e}")
            return []

    def fetch_ohlcv(self, symbol, exchange_name, timeframe="4h", limit=150):
        try:
            ex_obj = self.exchanges.get(exchange_name)
            if not ex_obj:
                return None

            ohlcv = ex_obj.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            print(f"❌ Error fetching {timeframe} for {symbol} from {exchange_name}: {e}")
            return None