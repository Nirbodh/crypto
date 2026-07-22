# python-engine/fundamental_engine.py

import os
import requests
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class MultiSourceFundamentalEngine:
    def __init__(self):
        self.cmc_api_key = os.getenv("COINMARKETCAP_API_KEY", "")
        self.cg_api_key = os.getenv("COINGECKO_API_KEY", "")
        self.cc_api_key = os.getenv("COINCOMPARE_API_KEY", "")
        
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.cmc_base = "https://pro-api.coinmarketcap.com/v1"
        self.cryptocompare_base = "https://min-api.cryptocompare.com/data"
        self.coinbase_base = "https://api.exchange.coinbase.com"

    def _extract_base_symbol(self, symbol: str) -> str:
        return symbol.split('/')[0].upper()

    def fetch_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        base_symbol = self._extract_base_symbol(symbol)
        
        cg_data = self._fetch_coingecko_data(base_symbol)
        cmc_data = self._fetch_cmc_data(base_symbol)
        cc_data = self._fetch_cryptocompare_data(base_symbol)
        cb_data = self._fetch_coinbase_data(base_symbol)

        # Fallback priority logic for critical metrics
        rank = cmc_data.get("rank") or cg_data.get("rank") or 999
        mcap = cmc_data.get("mcap") or cg_data.get("mcap") or 0.0
        fdv = cg_data.get("fdv") or cmc_data.get("fdv") or mcap
        ath_change = cg_data.get("ath_change_pct", 0.0)

        score, red_flags, green_flags = self._calculate_multi_source_score(
            rank=rank,
            mcap=mcap,
            fdv=fdv,
            ath_change=ath_change,
            is_coinbase_listed=cb_data.get("is_listed", False)
        )

        mcap_fdv_ratio = round(mcap / fdv, 2) if (fdv > 0 and mcap > 0) else 1.0

        return {
            "symbol": symbol,
            "fundamental_score": score,
            "sources_synced": {
                "coingecko": cg_data.get("status") == "OK",
                "coinmarketcap": cmc_data.get("status") == "OK",
                "cryptocompare": cc_data.get("status") == "OK",
                "coinbase": cb_data.get("status") == "OK"
            },
            "metrics": {
                "rank": rank,
                "market_cap_usd": mcap,
                "fdv_usd": fdv,
                "mcap_fdv_ratio": mcap_fdv_ratio,
                "ath_change_pct": round(ath_change, 2),
                "coinbase_listed": cb_data.get("is_listed", False)
            },
            "red_flags": red_flags,
            "green_flags": green_flags
        }

    def _fetch_coingecko_data(self, symbol: str) -> dict:
        try:
            url = f"{self.coingecko_base}/coins/markets?vs_currency=usd&symbols={symbol.lower()}"
            headers = {}
            if self.cg_api_key:
                headers["x-cg-demo-api-key"] = self.cg_api_key
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200 and len(res.json()) > 0:
                data = res.json()[0]
                return {
                    "status": "OK",
                    "rank": data.get("market_cap_rank"),
                    "mcap": float(data.get("market_cap") or 0),
                    "fdv": float(data.get("fully_diluted_valuation") or 0),
                    "ath_change_pct": float(data.get("ath_change_percentage") or 0)
                }
        except Exception:
            pass
        return {"status": "FAILED"}

    def _fetch_cmc_data(self, symbol: str) -> dict:
        if not self.cmc_api_key:
            return {"status": "NO_API_KEY"}
        try:
            url = f"{self.cmc_base}/cryptocurrency/quotes/latest?symbol={symbol}"
            headers = {"X-CMC_PRO_API_KEY": self.cmc_api_key}
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                res_data = res.json().get("data", {})
                if symbol in res_data:
                    # Fix: CMC quotes endpoint returns a Dict for symbol, NOT a List
                    coin_info = res_data[symbol]
                    if isinstance(coin_info, list):  # Just in case array is returned for duplicates
                        coin_info = coin_info[0]
                        
                    quote = coin_info["quote"]["USD"]
                    return {
                        "status": "OK",
                        "rank": coin_info.get("cmc_rank"),
                        "mcap": float(quote.get("market_cap") or 0),
                        "fdv": float(quote.get("fully_diluted_market_cap") or quote.get("market_cap") or 0)
                    }
        except Exception:
            pass
        return {"status": "FAILED"}

    def _fetch_cryptocompare_data(self, symbol: str) -> dict:
        try:
            url = f"{self.cryptocompare_base}/pricemultifull?fsyms={symbol}&tsyms=USD"
            headers = {}
            if self.cc_api_key:
                headers["authorization"] = f"Apikey {self.cc_api_key}"
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                raw = res.json().get("RAW", {}).get(symbol, {}).get("USD", {})
                return {
                    "status": "OK",
                    "price": float(raw.get("PRICE", 0)),
                    "volume_24h": float(raw.get("VOLUME24HOURTO", 0))
                }
        except Exception:
            pass
        return {"status": "FAILED"}

    def _fetch_coinbase_data(self, symbol: str) -> dict:
        try:
            # Exchange API gives direct product pair check faster and cleanly
            url = f"{self.coinbase_base}/products/{symbol}-USD"
            res = requests.get(url, timeout=3)
            if res.status_code == 200:
                return {"status": "OK", "is_listed": True}
        except Exception:
            pass
        return {"status": "FAILED", "is_listed": False}

    def _calculate_multi_source_score(self, rank: int, mcap: float, fdv: float, ath_change: float, is_coinbase_listed: bool):
        score = 50
        red_flags = []
        green_flags = []

        # 1. Rank Score
        if rank <= 20:
            score += 25
            green_flags.append(f"Top Tier Asset (Rank #{rank})")
        elif rank <= 100:
            score += 15
            green_flags.append(f"Established Asset (Rank #{rank})")
        elif rank > 300:
            score -= 15
            red_flags.append(f"Low Market Cap Rank (#{rank}) - Liquidity Risk")

        # 2. Tokenomics (MCAP vs FDV Unlocking Risk)
        if fdv > 0 and mcap > 0:
            mcap_fdv_ratio = mcap / fdv
            if mcap_fdv_ratio < 0.3:
                score -= 20
                red_flags.append(f"High Dilution Risk (Only {mcap_fdv_ratio*100:.1f}% Unlocked)")
            elif mcap_fdv_ratio >= 0.75:
                score += 10
                green_flags.append("High Circulating Supply Unlocked (>75%)")

        # 3. Exchange Quality Check
        if is_coinbase_listed:
            score += 10
            green_flags.append("Coinbase Tier-1 Listing Verified")

        # 4. ATH Drawdown Check
        if ath_change < -90.0:
            score -= 10
            red_flags.append(f"Heavy Drawdown from ATH ({ath_change:.1f}%)")

        score = max(0, min(100, score))
        return score, red_flags, green_flags