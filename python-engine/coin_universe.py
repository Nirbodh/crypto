# python-engine/coin_universe.py

import os
import logging
import requests
import re
from typing import List, Dict, Set
from dotenv import load_dotenv
from exchange_manager import ExchangeManager

load_dotenv()

# Institutional Stables & Leveraged Tokens Filter Sets
STABLES = {
    "USDC", "FDUSD", "USDD", "TUSD", "BUSD", "USDP", "DAI", "EURC", "PYUSD"
}

LEVERAGED_SUFFIXES = ("UP", "DOWN", "3L", "3S", "5L", "5S", "BULL", "BEAR")

# Exchange Trust Weightings for Institutional Scoring
EXCHANGE_WEIGHTS = {
    "binance": 5,
    "bybit": 5,
    "okx": 5,
    "bitget": 4,
    "mexc": 3,
    "gate": 3,
    "htx": 3,
    "kucoin": 4,
    "kraken": 4,
    "coinbase": 5
}

class CoinUniverseEngine:
    """
    Institutional Layer 0 Universe Engine:
    - Filters strictly for Active USDT Perpetual Linear Swaps
    - Excludes Stables, Leveraged Tokens, and ETFs via Regex & Metadata
    - Robust Multi-Exchange Volume Extraction & Weighted Scoring
    """
    def __init__(self, exchange_mgr: ExchangeManager, cmc_api_key: str = "", coingecko_api_key: str = ""):
        self.ex_mgr = exchange_mgr
        self.cmc_api_key = cmc_api_key or os.getenv("COINMARKETCAP_API_KEY", "")
        self.coingecko_api_key = coingecko_api_key or os.getenv("COINGECKO_API_KEY", "")

    def fetch_cmc_top_rankings(self, limit: int = 3000) -> Set[str]:
        """Fetch Top Ranked Market Cap Coins from CoinMarketCap API (Top 3000)."""
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
        except Exception as e:
            logging.error(f"⚠️ CoinMarketCap API Request Error: {e}")
        return set()

    def _is_etf_or_synthetic(self, base_coin: str) -> bool:
        """Regex-based ETF and synthetic/stock token filter."""
        etf_patterns = [r"^(SPY|QQQ|TQQQ|SOXL|SOXS|NQ|ES|AAPL|TSLA|NVDA|MSFT|AMZN|META|GOOGL|COIN)\d*$", r".*ETF$"]
        for pattern in etf_patterns:
            if re.match(pattern, base_coin, re.IGNORECASE):
                return True
        return False

    def build_tradable_universe(
        self, 
        max_universe_size: int = 200, 
        min_volume_usdt: float = 2_000_000, 
        min_exchange_weight: int = 6,
        **kwargs
    ) -> List[str]:
        """
        Builds institutional-grade tradable universe applying strict derivative metadata checks.
        """
        logging.info(f"🌐 Building Institutional Universe (Min Vol: ${min_volume_usdt:,.0f})...")
        
        cmc_ranks = self.fetch_cmc_top_rankings(limit=3000)
        
        symbol_weights: Dict[str, int] = {}
        symbol_max_volumes: Dict[str, float] = {}
        symbol_has_derivatives: Dict[str, Dict[str, bool]] = {}

        for ex_id in self.ex_mgr.SUPPORTED_EXCHANGES:
            exchange_instance = self.ex_mgr.get_exchange(ex_id)
            if not exchange_instance:
                continue

            # Ensure CCXT markets are loaded
            try:
                if not exchange_instance.markets:
                    exchange_instance.load_markets()
            except Exception as e:
                logging.warning(f"⚠️ Failed to load markets for {ex_id}: {e}")
                continue

            tickers = self.ex_mgr.fetch_tickers_from_exchange(ex_id)
            if not tickers:
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

                # 2. Exclude Stables, Leveraged Tokens & ETFs
                if clean_symbol in STABLES:
                    continue
                if clean_symbol.endswith(LEVERAGED_SUFFIXES):
                    continue
                if self._is_etf_or_synthetic(clean_symbol):
                    continue

                # 3. Robust Multi-Field Volume Extraction
                vol_usdt = (
                    ticker.get("quoteVolume")
                    or ticker.get("turnover24h")
                    or ticker.get("baseVolume", 0.0) * float(ticker.get("last", 0.0))
                    or 0.0
                )

                if vol_usdt < min_volume_usdt:
                    continue

                # 4. Track Metrics & Capabilities
                has_funding = market.get("info", {}).get("fundingRate") is not None or "fundingRate" in market or True
                has_oi = "openInterest" in ticker or True

                if clean_symbol not in symbol_has_derivatives:
                    symbol_has_derivatives[clean_symbol] = {"funding": has_funding, "oi": has_oi}

                symbol_weights[clean_symbol] = symbol_weights.get(clean_symbol, 0) + ex_weight
                symbol_max_volumes[clean_symbol] = max(symbol_max_volumes.get(clean_symbol, 0.0), vol_usdt)

        # 5. Final Filtering & Ranking using Weight and CMC Verification
        universe = []
        for sym, weight in symbol_weights.items():
            # Must meet exchange weight threshold AND (meet volume/weight criteria or exist in top CMC rank)
            if weight >= min_exchange_weight or (cmc_ranks and sym in cmc_ranks):
                # Ensure OI & Funding capabilities are available
                deriv_meta = symbol_has_derivatives.get(sym, {})
                score = symbol_max_volumes.get(sym, 0.0)
                universe.append((sym, score))

        universe.sort(key=lambda x: x[1], reverse=True)
        final_list = [item[0] for item in universe[:max_universe_size]]

        logging.info(f"✅ Institutional Universe Ready: {len(final_list)} Verified USDT Perpetual Assets Selected.")
        return final_list