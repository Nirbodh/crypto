# python-engine/coin_universe.py

import os
import logging
import requests
import re
import time  # ✅ Added for rate limiting
from typing import List, Dict, Set, Optional
from dotenv import load_dotenv
from exchange_manager import ExchangeManager

load_dotenv()

# Institutional Stables & Leveraged Tokens Filter Sets
STABLES = {
    "USDC", "FDUSD", "USDD", "TUSD", "BUSD", "USDP", "DAI", "EURC", "PYUSD", "USTC"
}

# Common Commodities & Non-Crypto assets that might appear in CMC but are not crypto perpetuals
COMMODITIES = {
    "XAU", "XAG", "CL", "NG", "GC", "SI", "PL", "PA", "HG", "XPT", "XPD"
}

# High-risk/meme/leveraged suffixes
LEVERAGED_SUFFIXES = ("UP", "DOWN", "3L", "3S", "5L", "5S", "BULL", "BEAR", "1000X")

# Exchange Trust Weightings for Institutional Scoring
EXCHANGE_WEIGHTS = {
    "binance": 5,
    "bybit": 5,
    "bitget": 4,
    "mexc": 3,
    "kucoin": 4,
    "coinbase": 5
}

class CoinUniverseEngine:
    """
    Institutional Layer 0 Universe Engine:
    - Filters strictly for Active USDT Perpetual Linear Swaps
    - Excludes Stables, Leveraged Tokens, ETFs, and Commodities via Regex & Metadata
    - Robust Multi-Exchange Volume Extraction & Weighted Scoring
    """
    def __init__(self, exchange_mgr: ExchangeManager, cmc_api_key: str = "", coingecko_api_key: str = ""):
        self.ex_mgr = exchange_mgr
        self.cmc_api_key = cmc_api_key or os.getenv("COINMARKETCAP_API_KEY", "")
        self.coingecko_api_key = coingecko_api_key or os.getenv("COINGECKO_API_KEY", "")

    def fetch_cmc_top_rankings(self, limit: int = 500) -> Set[str]:
        """Fetch Top Ranked Market Cap Coins from CoinMarketCap API (Top 500)."""
        if not self.cmc_api_key:
            logging.warning("⚠️ CoinMarketCap API key missing. Skipping CMC Rank Filter.")
            return set()
            
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        headers = {'X-CMC_PRO_API_KEY': self.cmc_api_key}
        params = {'start': '1', 'limit': str(limit), 'convert': 'USD'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                valid_symbols = {coin['symbol'].upper() for coin in data.get('data', [])}
                logging.info(f"📊 Fetched {len(valid_symbols)} top ranked symbols from CoinMarketCap.")
                return valid_symbols
            else:
                logging.warning(f"⚠️ CMC API returned status {response.status_code}")
        except Exception as e:
            logging.error(f"⚠️ CoinMarketCap API Request Error: {e}")
        return set()

    def _is_etf_or_synthetic(self, base_coin: str) -> bool:
        """Regex-based ETF and synthetic/stock token filter."""
        # Match SPY, QQQ, TQQQ, etc., or anything ending with ETF
        etf_patterns = [
            r"^(SPY|QQQ|TQQQ|SOXL|SOXS|NQ|ES|AAPL|TSLA|NVDA|MSFT|AMZN|META|GOOGL|COIN|MSTR)\d*$", 
            r".*ETF$",
            r".*/USD$",  # If it explicitly looks like fiat pair base
            r".*[0-9]+[L|S]$",  # Leveraged tokens like BTC3L, ETH5S
        ]
        for pattern in etf_patterns:
            if re.match(pattern, base_coin, re.IGNORECASE):
                return True
        return False

    def build_tradable_universe(
        self, 
        max_universe_size: int = 50, 
        min_volume_usdt: float = 2_000_000, 
        min_exchange_weight: int = 6,
        **kwargs
    ) -> List[str]:
        """
        Builds institutional-grade tradable universe applying strict derivative metadata checks.
        Returns FULL SYMBOLS like "BTC/USDT:USDT" for direct exchange use.
        """
        logging.info(f"🌐 Building Institutional Universe (Min Vol: ${min_volume_usdt:,.0f})...")
        
        cmc_ranks = self.fetch_cmc_top_rankings(limit=500)
        
        symbol_weights: Dict[str, int] = {}
        symbol_max_volumes: Dict[str, float] = {}
        symbol_has_derivatives: Dict[str, Dict[str, bool]] = {}

        for ex_id in self.ex_mgr.SUPPORTED_EXCHANGES:
            exchange_instance = self.ex_mgr.get_exchange(ex_id)
            if not exchange_instance:
                continue

            # Robust Market Loading Check
            try:
                if not exchange_instance.markets:
                    exchange_instance.load_markets()
                # CRITICAL FIX: Check if markets is actually a dict, else skip
                if not exchange_instance.markets or not isinstance(exchange_instance.markets, dict):
                    logging.warning(f"⚠️ Skipping {ex_id}: markets is None or not a dict.")
                    continue
            except Exception as e:
                logging.warning(f"⚠️ Failed to load markets for {ex_id}: {e}")
                continue

            # ✅ Rate limit: sleep between exchange requests to avoid 429 Too Many Requests
            time.sleep(0.5)

            tickers = self.ex_mgr.fetch_tickers_from_exchange(ex_id)
            if not tickers:
                logging.warning(f"⚠️ No tickers fetched from {ex_id}")
                continue

            ex_weight = EXCHANGE_WEIGHTS.get(ex_id, 2)

            for raw_symbol, ticker in tickers.items():
                market = exchange_instance.markets.get(raw_symbol)
                if not market:
                    continue

                # 1. Strict Institutional Metadata Filters for Derivatives
                if not market.get("active", False):
                    continue
                if not market.get("swap", False):  # Must be Swap / Perpetual
                    continue
                if not market.get("linear", False): # Must be Linear Contract
                    continue
                if market.get("quote") != "USDT":
                    continue

                clean_symbol = market.get("base", "").upper()
                if not clean_symbol:
                    continue

                # 2. Exclude Stables, Commodities, Leveraged Tokens & ETFs
                if clean_symbol in STABLES:
                    continue
                if clean_symbol in COMMODITIES:
                    continue
                if clean_symbol.endswith(LEVERAGED_SUFFIXES):
                    continue
                if self._is_etf_or_synthetic(clean_symbol):
                    continue

                # 3. Robust Multi-Field Volume Extraction (with safe float conversion)
                last_price = ticker.get('last')
                if last_price is None:
                    last_price = ticker.get('close')
                try:
                    last_price = float(last_price) if last_price is not None else 0.0
                except (ValueError, TypeError):
                    last_price = 0.0

                vol_usdt = (
                    ticker.get("quoteVolume") or
                    ticker.get("turnover24h") or
                    (ticker.get("baseVolume", 0.0) * last_price) if last_price else 0.0
                )
                
                if vol_usdt is None:
                    vol_usdt = 0.0
                try:
                    vol_usdt = float(vol_usdt)
                except (ValueError, TypeError):
                    vol_usdt = 0.0

                if vol_usdt < min_volume_usdt:
                    continue

                # 4. Track Metrics & Capabilities
                has_funding = market.get("info", {}).get("fundingRate") is not None or "fundingRate" in market
                has_oi = "openInterest" in ticker or True

                if clean_symbol not in symbol_has_derivatives:
                    symbol_has_derivatives[clean_symbol] = {"funding": has_funding, "oi": has_oi}

                symbol_weights[clean_symbol] = symbol_weights.get(clean_symbol, 0) + ex_weight
                symbol_max_volumes[clean_symbol] = max(symbol_max_volumes.get(clean_symbol, 0.0), vol_usdt)

        # 5. Final Filtering & Ranking using Weight and CMC Verification
        universe = []
        for sym, weight in symbol_weights.items():
            # Must meet exchange weight threshold OR exist in top CMC rank
            if weight >= min_exchange_weight or (cmc_ranks and sym in cmc_ranks):
                score = symbol_max_volumes.get(sym, 0.0)
                universe.append((sym, score))

        universe.sort(key=lambda x: x[1], reverse=True)
        
        # ============================================================
        # 🔥 CRITICAL FIX: Convert to FULL Exchange Format
        # This solves the "Missing candle data for BTC" error
        # ============================================================
        final_list = [f"{sym}/USDT:USDT" for sym, _ in universe[:max_universe_size]]

        logging.info(f"✅ Institutional Universe Ready: {len(final_list)} Verified USDT Perpetual Assets Selected.")
        if final_list:
            logging.info(f"📋 Sample Universe: {final_list[:5]}")
        return final_list
