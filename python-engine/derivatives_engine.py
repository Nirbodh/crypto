# python-engine/derivatives_engine.py

import os
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class DerivativesEngineV2:
    def __init__(self):
        self.coinglass_key = os.getenv("COINGLASS_API_KEY", "")
        self.binance_fapi = "https://fapi.binance.com"
        self.binance_spot = "https://api.binance.com"
        self.cg_headers = {"coinglassSecret": self.coinglass_key} if self.coinglass_key else {}

    def _format_symbol(self, symbol: str) -> str:
        return symbol.replace('/', '').upper()

    def fetch_derivatives_data(self, symbol: str) -> dict:
        formatted_symbol = self._format_symbol(symbol)
        base_symbol = symbol.split('/')[0].upper()

        try:
            oi_metrics = self._get_oi_trend_and_matrix(formatted_symbol)
            funding_metrics = self._get_funding_history_and_slope(formatted_symbol)
            ls_metrics = self._get_long_short_ratio(formatted_symbol)
            cvd_metrics = self._get_cvd_metrics(formatted_symbol)
            basis_metrics = self._get_spot_futures_basis(formatted_symbol)
            coinglass_metrics = self._fetch_coinglass_aggregates(base_symbol)

            score, raw_points, red_flags, green_flags = self._calculate_production_score(
                oi_metrics, funding_metrics, ls_metrics, cvd_metrics, basis_metrics, coinglass_metrics
            )

            return {
                "symbol": symbol,
                "derivatives_score": score,
                "raw_score_points": raw_points,
                "oi_matrix_signal": oi_metrics.get("oi_price_matrix", "NEUTRAL"),
                "oi_change_1h_pct": oi_metrics.get("oi_change_1h_pct", 0.0),
                "current_funding_pct": funding_metrics.get("current_funding_pct", 0.0),
                "funding_slope": funding_metrics.get("funding_slope", "FLAT"),
                "long_short_ratio": ls_metrics.get("ls_ratio", 1.0),
                "cvd_bias": cvd_metrics.get("cvd_bias", "NEUTRAL"),
                "basis_premium_pct": basis_metrics.get("basis_pct", 0.0),
                "coinglass_status": coinglass_metrics.get("status", "NOT_CONFIGURED"),
                "red_flags": red_flags,
                "green_flags": green_flags,
                "smart_money_verdict": "INSTITUTIONAL_ACCUMULATION" if score >= 70 else "NEUTRAL_OR_RISKY"
            }

        except Exception as e:
            return {
                "symbol": symbol,
                "derivatives_score": 50,
                "raw_score_points": 0,
                "error": str(e),
                "red_flags": [f"Derivatives Error: {str(e)}"],
                "green_flags": []
            }

    def _fetch_coinglass_aggregates(self, base_symbol: str) -> dict:
        if not self.coinglass_key:
            return {"status": "NO_API_KEY"}
        try:
            url = f"https://open-api.coinglass.com/public/v2/open_interest?symbol={base_symbol}"
            res = requests.get(url, headers=self.cg_headers, timeout=5)
            if res.status_code == 200 and res.json().get("code") == "0":
                data = res.json().get("data", [])
                total_oi = sum(float(x.get("openInterest", 0)) for x in data if isinstance(x, dict))
                return {"status": "OK", "aggregate_oi_usd": total_oi}
        except Exception:
            pass
        return {"status": "FAILED"}

    def _get_oi_trend_and_matrix(self, symbol: str) -> dict:
        try:
            url = f"{self.binance_fapi}/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=12"
            res = requests.get(url, timeout=5)
            if res.status_code != 200 or not res.json():
                return {"current_oi": 0, "oi_change_1h_pct": 0, "oi_price_matrix": "NEUTRAL"}

            data = res.json()
            current_oi = float(data[-1]["sumOpenInterestValue"])
            oi_1h_ago = float(data[0]["sumOpenInterestValue"])
            oi_change_1h_pct = ((current_oi - oi_1h_ago) / oi_1h_ago) * 100 if oi_1h_ago > 0 else 0.0

            matrix_signal = "NEUTRAL"
            if oi_change_1h_pct > 1.5:
                matrix_signal = "BULLISH_INFLOW"
            elif oi_change_1h_pct < -1.5:
                matrix_signal = "BEARISH_OUTFLOW"

            return {
                "current_oi": current_oi,
                "oi_change_1h_pct": round(oi_change_1h_pct, 2),
                "oi_price_matrix": matrix_signal
            }
        except Exception:
            return {"current_oi": 0, "oi_change_1h_pct": 0, "oi_price_matrix": "NEUTRAL"}

    def _get_funding_history_and_slope(self, symbol: str) -> dict:
        try:
            url = f"{self.binance_fapi}/fapi/v1/fundingRate?symbol={symbol}&limit=8"
            res = requests.get(url, timeout=5)
            if res.status_code != 200 or not res.json():
                return {"current_funding_pct": 0.01, "funding_slope": "FLAT"}

            data = res.json()
            rates = [float(x["fundingRate"]) * 100 for x in data]
            current_funding = rates[-1]

            # Calculate funding rate trend (Slope)
            x = np.arange(len(rates))
            slope = np.polyfit(x, rates, 1)[0]

            slope_status = "FLAT"
            if slope > 0.001:
                slope_status = "RISING"
            elif slope < -0.001:
                slope_status = "FALLING"

            return {
                "current_funding_pct": round(current_funding, 4),
                "funding_slope": slope_status
            }
        except Exception:
            return {"current_funding_pct": 0.01, "funding_slope": "FLAT"}

    def _get_long_short_ratio(self, symbol: str) -> dict:
        try:
            url = f"{self.binance_fapi}/futures/data/topLongShortAccountRatio?symbol={symbol}&period=5m&limit=1"
            res = requests.get(url, timeout=5)
            if res.status_code == 200 and isinstance(res.json(), list) and len(res.json()) > 0:
                return {"ls_ratio": round(float(res.json()[0].get("longShortRatio", 1.0)), 2)}
        except Exception:
            pass
        return {"ls_ratio": 1.0}

    def _get_cvd_metrics(self, symbol: str) -> dict:
        try:
            url = f"{self.binance_fapi}/fapi/v1/trades?symbol={symbol}&limit=300"
            res = requests.get(url, timeout=5)
            if res.status_code != 200 or not res.json():
                return {"cvd_bias": "NEUTRAL"}

            trades = res.json()
            buy_vol = sum(float(t["qty"]) for t in trades if not t["isBuyerMaker"])
            sell_vol = sum(float(t["qty"]) for t in trades if t["isBuyerMaker"])
            total_vol = buy_vol + sell_vol

            if total_vol == 0:
                return {"cvd_bias": "NEUTRAL"}

            net_cvd = buy_vol - sell_vol
            if net_cvd > total_vol * 0.1:
                bias = "AGGRESSIVE_BUYING"
            elif net_cvd < -total_vol * 0.1:
                bias = "AGGRESSIVE_SELLING"
            else:
                bias = "BALANCED"

            return {"cvd_bias": bias}
        except Exception:
            return {"cvd_bias": "NEUTRAL"}

    def _get_spot_futures_basis(self, symbol: str) -> dict:
        try:
            f_price = float(requests.get(f"{self.binance_fapi}/fapi/v1/ticker/price?symbol={symbol}", timeout=3).json()["price"])
            s_price = float(requests.get(f"{self.binance_spot}/api/v3/ticker/price?symbol={symbol}", timeout=3).json()["price"])
            basis_pct = ((f_price - s_price) / s_price) * 100
            return {"basis_pct": round(basis_pct, 4)}
        except Exception:
            return {"basis_pct": 0.0}

    def _calculate_production_score(self, oi, funding, ls, cvd, basis, cg):
        raw_points = 0
        red_flags, green_flags = [], []

        # 1. Open Interest (OI)
        oi_matrix = oi.get("oi_price_matrix")
        if oi_matrix == "BULLISH_INFLOW":
            raw_points += 15
            green_flags.append("Institutional OI Capital Inflow")
        elif oi_matrix == "BEARISH_OUTFLOW":
            raw_points -= 10
            red_flags.append("Capital Outflow / Liquidation Risk")

        # 2. Cumulative Volume Delta (CVD)
        cvd_bias = cvd.get("cvd_bias")
        if cvd_bias == "AGGRESSIVE_BUYING":
            raw_points += 15
            green_flags.append("Aggressive Market Buy Delta (CVD)")
        elif cvd_bias == "AGGRESSIVE_SELLING":
            raw_points -= 15
            red_flags.append("Aggressive Market Selling Pressure")

        # 3. Long / Short Ratio Logic
        ls_ratio = ls.get("ls_ratio", 1.0)
        if 1.1 <= ls_ratio <= 1.6:
            raw_points += 15
            green_flags.append(f"Healthy Long/Short Ratio ({ls_ratio})")
        elif ls_ratio > 2.0:
            raw_points -= 10
            red_flags.append(f"Overcrowded Longs Ratio ({ls_ratio})")
        elif ls_ratio < 0.8:
            raw_points -= 10
            red_flags.append(f"Heavy Short Bias ({ls_ratio})")

        # 4. Funding Rate & Slope
        funding_pct = funding.get("current_funding_pct", 0.01)
        slope = funding.get("funding_slope", "FLAT")
        if 0.001 <= funding_pct <= 0.03 and slope != "RISING":
            raw_points += 10
            green_flags.append("Healthy Funding Rate")
        elif funding_pct > 0.05:
            raw_points -= 15
            red_flags.append(f"Overheated Funding Rate ({funding_pct}%)")

        # 5. CoinGlass Verification
        if cg.get("status") == "OK":
            raw_points += 5
            green_flags.append("CoinGlass Cross-Exchange Multi-OI Synced")

        # Base Score is 50 (Neutral)
        final_score = int(max(0, min(100, 50 + raw_points)))

        return final_score, raw_points, red_flags, green_flags