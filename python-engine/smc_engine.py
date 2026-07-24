import pandas as pd
import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SMCEngine:
    def __init__(self, fvg_threshold_pct=0.002):
        """
        fvg_threshold_pct: FVG চিহ্নিত করার জন্য ন্যূনতম গ্যাপ পার্সেন্টেজ (default 0.2%)
        """
        self.fvg_threshold_pct = fvg_threshold_pct

    def detect_fair_value_gaps(self, df: pd.DataFrame) -> dict:
        """
        Bullish এবং Bearish Fair Value Gaps (FVG) চিহ্নিত করে।
        """
        if df is None or len(df) < 3:
            return {"bullish_fvg": False, "bearish_fvg": False, "gap_size": 0.0}

        latest_idx = df.index[-1]
        prev2_idx = df.index[-3]

        # Bullish FVG: Candle 1 High < Candle 3 Low
        bullish_gap = df.loc[latest_idx, 'low'] - df.loc[prev2_idx, 'high']
        is_bullish_fvg = bullish_gap > (df.loc[latest_idx, 'close'] * self.fvg_threshold_pct)

        # Bearish FVG: Candle 1 Low > Candle 3 High
        bearish_gap = df.loc[prev2_idx, 'low'] - df.loc[latest_idx, 'high']
        is_bearish_fvg = bearish_gap > (df.loc[latest_idx, 'close'] * self.fvg_threshold_pct)

        return {
            "bullish_fvg": bool(is_bullish_fvg),
            "bearish_fvg": bool(is_bearish_fvg),
            "gap_size": round(float(max(bullish_gap, bearish_gap, 0)), 4)
        }

    def detect_liquidity_sweeps(self, df: pd.DataFrame, lookback=20) -> dict:
        """
        Previous Day High (PDH) / Low (PDL) বা Swing High/Low Liquidity Sweep চিহ্নিত করে।
        """
        if df is None or len(df) < lookback + 1:
            return {"bullish_sweep": False, "bearish_sweep": False, "key_high_swept": 0.0, "key_low_swept": 0.0}

        current_candle = df.iloc[-1]
        past_candles = df.iloc[-(lookback + 1):-1]

        highest_high = past_candles['high'].max()
        lowest_low = past_candles['low'].min()

        # Bullish Sweep (Sell-side liquidity sweep): Low wicks below lowest low, but closes above it
        bullish_sweep = (current_candle['low'] < lowest_low) and (current_candle['close'] > lowest_low)

        # Bearish Sweep (Buy-side liquidity sweep): High wicks above highest high, but closes below it
        bearish_sweep = (current_candle['high'] > highest_high) and (current_candle['close'] < highest_high)

        return {
            "bullish_sweep": bool(bullish_sweep),
            "bearish_sweep": bool(bearish_sweep),
            "key_high_swept": round(float(highest_high), 4),
            "key_low_swept": round(float(lowest_low), 4)
        }

    def detect_order_blocks(self, df: pd.DataFrame, lookback=10) -> dict:
        """
        Order Block (Bullish / Bearish OB) রেঞ্জ চিহ্নিত করে।
        """
        if df is None or len(df) < lookback:
            return {"bullish_ob": False, "bearish_ob": False}

        recent = df.iloc[-lookback:]
        current_close = df.iloc[-1]['close']
        
        # Bullish OB: Strong upward expansion after a down candle
        down_candles = recent[recent['close'] < recent['open']]
        bullish_ob = False
        if not down_candles.empty:
            last_down_candle = down_candles.iloc[-1]
            if current_close > last_down_candle['high'] and df.iloc[-1]['volume'] > df['volume'].tail(20).mean():
                bullish_ob = True

        # Bearish OB: Strong downward expansion after an up candle
        up_candles = recent[recent['close'] > recent['open']]
        bearish_ob = False
        if not up_candles.empty:
            last_up_candle = up_candles.iloc[-1]
            if current_close < last_up_candle['low'] and df.iloc[-1]['volume'] > df['volume'].tail(20).mean():
                bearish_ob = True

        return {
            "bullish_ob": bool(bullish_ob),
            "bearish_ob": bool(bearish_ob)
        }

    def calculate_premium_discount_zone(self, df: pd.DataFrame, lookback=50) -> dict:
        """
        Calculates Macro Range Equilibrium (50%), Premium Zone (>50%), and Discount Zone (<50%).
        """
        if df is None or len(df) < lookback:
            return {"zone": "UNKNOWN", "is_discount": False, "equilibrium": 0.0}

        recent_range = df.iloc[-lookback:]
        high_price = recent_range['high'].max()
        low_price = recent_range['low'].min()
        current_close = df['close'].iloc[-1]

        range_span = high_price - low_price
        if range_span <= 0:
            return {"zone": "UNKNOWN", "is_discount": False, "equilibrium": current_close}

        equilibrium = low_price + (range_span * 0.50)
        is_discount = current_close < equilibrium

        return {
            "zone": "DISCOUNT" if is_discount else "PREMIUM",
            "is_discount": bool(is_discount),
            "equilibrium": round(float(equilibrium), 4),
            "current_price": round(float(current_close), 4),
            "range_high": round(float(high_price), 4),
            "range_low": round(float(low_price), 4)
        }

    def calculate_smc_score(self, df: pd.DataFrame, symbol: str = "BTC/USDT") -> dict:
        """
        SMC & Liquidity Score (0-100) হিসাব করে score_fusion_engine-এর জন্য প্রস্তুত করে।
        """
        if df is None or df.empty or len(df) < 20:
            return {
                "symbol": symbol,
                "smc_score": 50.0,
                "smc_liquidity_score": 50.0,
                "choch_confirmed": False,
                "fvg_present": False,
                "ob_mitigated": False,
                "market_structure": "NEUTRAL",
                "green_flags": [],
                "red_flags": ["INSUFFICIENT_SMC_DATA"],
                "fvgs": []  # ফাঁকা লিস্ট
            }

        fvg_res = self.detect_fair_value_gaps(df)
        sweep_res = self.detect_liquidity_sweeps(df)
        ob_res = self.detect_order_blocks(df)
        zone_res = self.calculate_premium_discount_zone(df)

        base_score = 50.0
        green_flags = []
        red_flags = []

        # Discount vs Premium Zone Impact
        if zone_res["is_discount"]:
            base_score += 10.0
            green_flags.append("PRICE_IN_DISCOUNT_ZONE")
        else:
            base_score -= 10.0
            red_flags.append("PRICE_IN_PREMIUM_ZONE_NO_LONG")

        # Bullish Boosters
        if sweep_res["bullish_sweep"]:
            base_score += 20.0
            green_flags.append("LIQUIDITY_SWEEP_BULLISH")
        if fvg_res["bullish_fvg"]:
            base_score += 15.0
            green_flags.append("BULLISH_FVG_SUPPORT")
        if ob_res["bullish_ob"]:
            base_score += 10.0
            green_flags.append("BULLISH_ORDER_BLOCK")

        # Bearish Factors
        if sweep_res["bearish_sweep"]:
            base_score -= 20.0
            red_flags.append("LIQUIDITY_SWEEP_BEARISH")
        if fvg_res["bearish_fvg"]:
            base_score -= 15.0
            red_flags.append("BEARISH_FVG_RESISTANCE")
        if ob_res["bearish_ob"]:
            base_score -= 10.0
            red_flags.append("BEARISH_ORDER_BLOCK")

        final_score = round(max(0.0, min(100.0, base_score)), 2)

        # FVG লিস্ট তৈরি করা (শুধু মক ডেটা, বাস্তবে ডিটেইলস যোগ করা যেতে পারে)
        fvgs = []
        if fvg_res["bullish_fvg"] or fvg_res["bearish_fvg"]:
            # ডামি FVG লিস্ট, প্রয়োজনমতো আরও বিস্তারিত যোগ করা যায়
            fvgs.append({
                "type": "BULLISH" if fvg_res["bullish_fvg"] else "BEARISH",
                "top": float(df['high'].iloc[-1]),
                "bottom": float(df['low'].iloc[-1])
            })

        return {
            "symbol": symbol,
            "smc_score": final_score,
            "smc_liquidity_score": final_score,
            "choch_confirmed": sweep_res["bullish_sweep"] or sweep_res["bearish_sweep"],
            "fvg_present": fvg_res["bullish_fvg"] or fvg_res["bearish_fvg"],
            "ob_mitigated": ob_res["bullish_ob"] or ob_res["bearish_ob"],
            "market_structure": "BULLISH" if final_score > 60 else "BEARISH" if final_score < 40 else "NEUTRAL",
            "details": {
                "fvg": fvg_res,
                "sweeps": sweep_res,
                "order_blocks": ob_res,
                "zone": zone_res
            },
            "green_flags": green_flags,
            "red_flags": red_flags,
            "fvgs": fvgs   # <-- নতুন কী, fvgs লিস্ট যোগ করা হয়েছে
        }

    def analyze(self, df: pd.DataFrame, symbol: str = "BTC/USDT") -> dict:
        """Wrapper method for pipeline uniform call."""
        return self.calculate_smc_score(df, symbol=symbol)

    @staticmethod
    def check_fvg_mitigation(df: pd.DataFrame, fvgs: list) -> bool:
        """
        Checks if current price has mitigated any existing Fair Value Gap (FVG).
        """
        if df is None or df.empty or not fvgs:
            return False
        
        current_low = float(df['low'].iloc[-1])
        current_high = float(df['high'].iloc[-1])

        for fvg in fvgs:
            fvg_low = fvg.get('bottom', 0)
            fvg_high = fvg.get('top', 0)
            
            # যদি বর্তমান ক্যান্ডেলের প্রাইস FVG জোনের ভেতরে প্রবেশ করে (Mitigation)
            if (current_low <= fvg_high) and (current_high >= fvg_low):
                return True
                
        return False

    @classmethod
    def evaluate_smc(cls, df: pd.DataFrame, symbol: str = "BTC/USDT") -> dict:
        """Static / Class level wrapper for pipeline integration."""
        engine = cls()
        return engine.calculate_smc_score(df, symbol=symbol)


if __name__ == "__main__":
    print("==================================================")
    print("🧠 TESTING SMART MONEY CONCEPTS (SMC) ENGINE...")
    print("==================================================")
    
    # Mock Data Generation for Testing
    dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq="15min")
    np.random.seed(101)
    
    close_prices = 64000.0 + np.cumsum(np.random.randn(60) * 100)
    high_prices = close_prices + np.random.uniform(50, 300, 60)
    low_prices = close_prices - np.random.uniform(50, 300, 60)
    open_prices = low_prices + np.random.uniform(0.0, high_prices - low_prices)
    volumes = np.random.uniform(10, 100, 60)

    mock_df = pd.DataFrame({
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes
    }, index=dates)

    smc = SMCEngine()
    result = smc.calculate_smc_score(mock_df, symbol="BTC/USDT")
    
    print("✅ SMC Engine Executed Successfully!")
    print("\n📊 Generated Output Matrix:")
    print(json.dumps(result, indent=4))