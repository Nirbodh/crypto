# python-engine/derivatives_engine.py

import os
import logging
import requests
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class InstitutionalDerivativesEngine:
    """
    Derivatives Intelligence Engine consuming CoinGlass & Futures Data.
    Analyzes Open Interest (OI), Funding Rates, Long/Short Ratios & Short/Long Squeeze Potential.
    """
    def __init__(self, coinglass_api_key: str = ""):
        self.api_key = coinglass_api_key or os.getenv("COINGLASS_API_KEY", "")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "coinglassSecret": self.api_key,
                "CG-API-KEY": self.api_key,
                "accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            })

    def fetch_coinglass_live_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches real-time derivatives statistics from CoinGlass API.
        Safe fallback to neutral structure on failure or missing API key.
        """
        clean_symbol = symbol.split("/")[0].upper()  # E.g., 'BTC' from 'BTC/USDT'
        
        default_response = {
            "oi_change_pct": 0.0,
            "funding_rate": 0.01,
            "long_short_ratio": 1.0,
            "squeeze_potential": "NEUTRAL"
        }

        if not self.api_key:
            logging.warning(f"⚠️ CoinGlass API Key missing! Returning neutral derivatives structure for {symbol}.")
            return default_response

        url = f"https://open-api.coinglass.com/public/v2/indicator/open_interest?symbol={clean_symbol}"

        try:
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                res_json = response.json()
                data = res_json.get("data", [])
                
                if data and isinstance(data, list):
                    first_item = data[0]
                    
                    oi_change = float(first_item.get("h24Change", 0.0) or 0.0)
                    funding = float(first_item.get("avgFundingRate", 0.01) or 0.01)
                    ls_ratio = float(first_item.get("longShortRatio", 1.0) or 1.0)

                    # Dynamic Short or Long Squeeze Risk Detection
                    squeeze = "NEUTRAL"
                    if oi_change > 5.0 and funding < 0.0 and ls_ratio < 0.85:
                        squeeze = "POTENTIAL_SHORT_SQUEEZE"
                    elif oi_change > 5.0 and funding > 0.03 and ls_ratio > 1.5:
                        squeeze = "POTENTIAL_LONG_LIQUIDATION"

                    return {
                        "oi_change_pct": round(oi_change, 2),
                        "funding_rate": round(funding, 4),
                        "long_short_ratio": round(ls_ratio, 2),
                        "squeeze_potential": squeeze
                    }
            else:
                logging.warning(f"⚠️ CoinGlass API HTTP {response.status_code} for {symbol}")

        except Exception as e:
            logging.error(f"⚠️ CoinGlass Request Error for {symbol}: {e}")

        return default_response

    def process_derivatives(
        self, 
        oi_change_pct: float, 
        funding_rate: float, 
        long_short_ratio: float,
        squeeze_potential: str = "NEUTRAL"
    ) -> Dict[str, Any]:
        """
        Calculates Derivatives Score (0 - 100) based on institutional positioning.
        Capped strictly at 90.0 maximum to preserve realistic quantitative variance.
        """
        score = 50.0

        # 1. Open Interest Expansion Evaluation
        if oi_change_pct > 5.0:
            score += 15.0  # Liquidity building up
        elif oi_change_pct < -5.0:
            score -= 15.0  # Liquidity flush / Unwinding

        # 2. Funding Rate Adjustments (Leverage Sentiment)
        if -0.01 <= funding_rate <= 0.015:
            score += 10.0  # Healthy & Sustainable Leverage
        elif funding_rate > 0.04 or funding_rate < -0.03:
            score -= 20.0  # Overheated Market / Liquidation Risk

        # 3. Long/Short Ratio Adjustments
        if 0.8 <= long_short_ratio <= 1.2:
            score += 10.0  # Equilibrium / Balanced Positioning

        # 4. Squeeze Dynamics Adjustment
        if squeeze_potential == "POTENTIAL_SHORT_SQUEEZE":
            score += 10.0  # Bullish Squeeze Energy
        elif squeeze_potential == "POTENTIAL_LONG_LIQUIDATION":
            score -= 15.0  # Bearish Cascade Threat

        # Cap strictly between 0.0 and 90.0
        final_score = min(90.0, max(0.0, score))

        return {
            "derivatives_score": round(final_score, 2),
            "oi_change_pct": round(oi_change_pct, 2),
            "funding_rate": round(funding_rate, 4),
            "long_short_ratio": round(long_short_ratio, 2),
            "squeeze_potential": squeeze_potential
        }

    def analyze_derivatives(self, symbol: str) -> Dict[str, Any]:
        """
        Convenience Wrapper: Fetches live data and computes score seamlessly.
        """
        live_data = self.fetch_coinglass_live_data(symbol)
        return self.process_derivatives(
            oi_change_pct=live_data.get("oi_change_pct", 0.0),
            funding_rate=live_data.get("funding_rate", 0.01),
            long_short_ratio=live_data.get("long_short_ratio", 1.0),
            squeeze_potential=live_data.get("squeeze_potential", "NEUTRAL")
        )


if __name__ == "__main__":
    # Standalone Test
    engine = InstitutionalDerivativesEngine()
    result = engine.process_derivatives(
        oi_change_pct=10.0, 
        funding_rate=0.005, 
        long_short_ratio=0.9, 
        squeeze_potential="POTENTIAL_SHORT_SQUEEZE"
    )
    print("--- Standalone Test Result ---")
    print("Processed Derivatives Result:", result)