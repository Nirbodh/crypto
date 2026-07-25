# python-engine/smc_engine.py

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

    def _detect_swings(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        সুইং হাই এবং সুইং লো চিহ্নিত করে (SMC-র ভিত্তি)।
        """
        df_copy = df.copy()
        # রোলিং উইন্ডোতে সেন্টার করে ম্যাক্স/মিন খুঁজি
        df_copy['swing_high'] = df_copy['high'].where(
            df_copy['high'] == df_copy['high'].rolling(window * 2 + 1, center=True, min_periods=1).max()
        )
        df_copy['swing_low'] = df_copy['low'].where(
            df_copy['low'] == df_copy['low'].rolling(window * 2 + 1, center=True, min_periods=1).min()
        )
        return df_copy

    def detect_fair_value_gaps(self, df: pd.DataFrame) -> dict:
        """
        সম্পূর্ণ ডেটাফ্রেম স্ক্যান করে সব Fair Value Gaps (FVG) চিহ্নিত করে।
        সর্বশেষ ৫টি FVG রিটার্ন করে।
        """
        if df is None or len(df) < 3:
            return {"bullish_fvg": False, "bearish_fvg": False, "fvgs_list": [], "gap_size": 0.0}

        fvgs = []
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        for i in range(2, len(df)):
            # Bullish FVG: বর্তমান ক্যান্ডেলের Low > দুই ক্যান্ডেল আগের High
            if low[i] > high[i - 2]:
                gap_size = low[i] - high[i - 2]
                if gap_size > (close[i] * self.fvg_threshold_pct):
                    fvgs.append({
                        "type": "BULLISH",
                        "top": round(float(low[i]), 4),
                        "bottom": round(float(high[i - 2]), 4),
                        "index": i,
                        "gap_size": round(float(gap_size), 4)
                    })
            # Bearish FVG: বর্তমান ক্যান্ডেলের High < দুই ক্যান্ডেল আগের Low
            elif high[i] < low[i - 2]:
                gap_size = low[i - 2] - high[i]
                if gap_size > (close[i] * self.fvg_threshold_pct):
                    fvgs.append({
                        "type": "BEARISH",
                        "top": round(float(low[i - 2]), 4),
                        "bottom": round(float(high[i]), 4),
                        "index": i,
                        "gap_size": round(float(gap_size), 4)
                    })

        # সর্বশেষ ৫টি FVG ধরে রাখি (পারফরম্যান্সের জন্য)
        latest_fvgs = fvgs[-5:] if fvgs else []
        bullish_present = any(f['type'] == 'BULLISH' for f in latest_fvgs)
        bearish_present = any(f['type'] == 'BEARISH' for f in latest_fvgs)
        max_gap = max([f['gap_size'] for f in latest_fvgs]) if latest_fvgs else 0.0

        return {
            "bullish_fvg": bool(bullish_present),
            "bearish_fvg": bool(bearish_present),
            "fvgs_list": latest_fvgs,
            "gap_size": round(float(max_gap), 4)
        }

    def detect_choch(self, df: pd.DataFrame) -> dict:
        """
        সুইং হাই/লো ব্যবহার করে Change of Character (CHoCH) চিহ্নিত করে।
        """
        if df is None or len(df) < 20:
            return {"confirmed": False, "direction": "NEUTRAL", "prev_swing_high": 0.0, "prev_swing_low": 0.0}

        # সুইং পয়েন্টগুলো বের করি
        swing_highs = df[df['swing_high'].notna()]
        swing_lows = df[df['swing_low'].notna()]

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"confirmed": False, "direction": "NEUTRAL", "prev_swing_high": 0.0, "prev_swing_low": 0.0}

        last_high = swing_highs.iloc[-1]['high']
        prev_high = swing_highs.iloc[-2]['high']
        last_low = swing_lows.iloc[-1]['low']
        prev_low = swing_lows.iloc[-2]['low']
        current_close = df.iloc[-1]['close']

        bullish_choch = False
        bearish_choch = False

        # বুলিশ CHoCH: প্রাইস পূর্ববর্তী উচ্চ সুইং ভেঙে উপরে গেছে (ডাউনট্রেন্ড ব্রেক)
        if current_close > prev_high and last_low >= prev_low:
            bullish_choch = True
        # বিয়ারিশ CHoCH: প্রাইস পূর্ববর্তী নিম্ন সুইং ভেঙে নিচে গেছে (আপট্রেন্ড ব্রেক)
        elif current_close < prev_low and last_high <= prev_high:
            bearish_choch = True

        return {
            "confirmed": bool(bullish_choch or bearish_choch),
            "direction": "BULLISH" if bullish_choch else "BEARISH" if bearish_choch else "NEUTRAL",
            "prev_swing_high": round(float(prev_high), 4),
            "prev_swing_low": round(float(prev_low), 4)
        }

    def detect_liquidity_sweeps(self, df: pd.DataFrame, lookback: int = 20) -> dict:
        """
        Previous Day High (PDH) / Low (PDL) বা Swing High/Low Liquidity Sweep চিহ্নিত করে।
        """
        if df is None or len(df) < lookback + 1:
            return {"bullish_sweep": False, "bearish_sweep": False, "key_high_swept": 0.0, "key_low_swept": 0.0}

        current_candle = df.iloc[-1]
        past_candles = df.iloc[-(lookback + 1):-1]

        highest_high = past_candles['high'].max()
        lowest_low = past_candles['low'].min()

        # Bullish Sweep: লো উইক নিচে গিয়ে ক্লোজ উপরে
        bullish_sweep = (current_candle['low'] < lowest_low) and (current_candle['close'] > lowest_low)
        # Bearish Sweep: হাই উইক উপরে গিয়ে ক্লোজ নিচে
        bearish_sweep = (current_candle['high'] > highest_high) and (current_candle['close'] < highest_high)

        return {
            "bullish_sweep": bool(bullish_sweep),
            "bearish_sweep": bool(bearish_sweep),
            "key_high_swept": round(float(highest_high), 4),
            "key_low_swept": round(float(lowest_low), 4)
        }

    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 10) -> dict:
        """
        Order Block (Bullish / Bearish OB) রেঞ্জ চিহ্নিত করে এবং OB প্রাইস রিটার্ন করে।
        """
        if df is None or len(df) < lookback:
            return {"bullish_ob": False, "bearish_ob": False, "order_block_price": 0.0}

        recent = df.iloc[-lookback:]
        current_close = df.iloc[-1]['close']
        order_block_price = 0.0
        
        # Bullish OB: ডাউন ক্যান্ডেলের পর শক্তিশালী আপ মুভ
        down_candles = recent[recent['close'] < recent['open']]
        bullish_ob = False
        if not down_candles.empty:
            last_down_candle = down_candles.iloc[-1]
            order_block_price = float(last_down_candle['low'])  # OB সাপোর্ট
            if current_close > last_down_candle['high'] and df.iloc[-1]['volume'] > df['volume'].tail(20).mean():
                bullish_ob = True

        # Bearish OB: আপ ক্যান্ডেলের পর শক্তিশালী ডাউন মুভ
        up_candles = recent[recent['close'] > recent['open']]
        bearish_ob = False
        if not up_candles.empty:
            last_up_candle = up_candles.iloc[-1]
            order_block_price = float(last_up_candle['high'])  # OB রেজিস্ট্যান্স
            if current_close < last_up_candle['low'] and df.iloc[-1]['volume'] > df['volume'].tail(20).mean():
                bearish_ob = True

        return {
            "bullish_ob": bool(bullish_ob),
            "bearish_ob": bool(bearish_ob),
            "order_block_price": order_block_price
        }

    def calculate_premium_discount_zone(self, df: pd.DataFrame, lookback: int = 50) -> dict:
        """
        ম্যাক্রো রেঞ্জের ইকুইলিব্রিয়াম (৫০%), প্রিমিয়াম জোন (>৫০%), ডিসকাউন্ট জোন (<৫০%) ক্যালকুলেট করে।
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
        এখানে CHoCH (সুইং-ভিত্তিক), FVG (সম্পূর্ণ স্ক্যান), OB মিটিগেশন সব সঠিকভাবে যুক্ত।
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
                "fvgs": []
            }

        # ১. সুইং ডিটেক্ট করে CHoCH বের করি
        df_with_swings = self._detect_swings(df)
        choch_res = self.detect_choch(df_with_swings)

        # ২. FVG (পুরো ডেটাফ্রেম স্ক্যান)
        fvg_res = self.detect_fair_value_gaps(df)

        # ৩. লিকুইডিটি সুইপ
        sweep_res = self.detect_liquidity_sweeps(df)

        # ৪. অর্ডার ব্লক
        ob_res = self.detect_order_blocks(df)

        # ৫. প্রিমিয়াম/ডিসকাউন্ট জোন
        zone_res = self.calculate_premium_discount_zone(df)

        base_score = 50.0
        green_flags = []
        red_flags = []

        # --- CHoCH ইমপ্যাক্ট (সবচেয়ে গুরুত্বপূর্ণ) ---
        if choch_res["confirmed"]:
            if choch_res["direction"] == "BULLISH":
                base_score += 25.0
                green_flags.append("CHoCH_BULLISH_CONFIRMED")
            elif choch_res["direction"] == "BEARISH":
                base_score -= 25.0
                red_flags.append("CHoCH_BEARISH_CONFIRMED")

        # --- লিকুইডিটি সুইপ ---
        if sweep_res["bullish_sweep"]:
            base_score += 15.0
            green_flags.append("LIQUIDITY_SWEEP_BULLISH")
        if sweep_res["bearish_sweep"]:
            base_score -= 15.0
            red_flags.append("LIQUIDITY_SWEEP_BEARISH")

        # --- FVG ---
        if fvg_res["bullish_fvg"]:
            base_score += 10.0
            green_flags.append("BULLISH_FVG_SUPPORT")
        if fvg_res["bearish_fvg"]:
            base_score -= 10.0
            red_flags.append("BEARISH_FVG_RESISTANCE")

        # --- অর্ডার ব্লক ---
        if ob_res["bullish_ob"]:
            base_score += 10.0
            green_flags.append("BULLISH_ORDER_BLOCK")
        if ob_res["bearish_ob"]:
            base_score -= 10.0
            red_flags.append("BEARISH_ORDER_BLOCK")

        # --- জোন (ডিসকাউন্ট/প্রিমিয়াম) ---
        if zone_res["is_discount"]:
            base_score += 10.0
            green_flags.append("PRICE_IN_DISCOUNT_ZONE")
        else:
            base_score -= 10.0
            red_flags.append("PRICE_IN_PREMIUM_ZONE")

        # --- OB মিটিগেশন চেক (CHoCH + OB রিট্রেসমেন্ট) ---
        current_close = float(df['close'].iloc[-1])
        ob_price = ob_res.get("order_block_price", 0.0)
        ob_mitigated = False

        if choch_res["confirmed"] and ob_price > 0:
            # ১.৫% ব্যান্ডের মধ্যে প্রাইস OB জোনে ফিরে এসেছে কিনা
            if choch_res["direction"] == "BULLISH" and (ob_price * 0.985) <= current_close <= (ob_price * 1.015):
                ob_mitigated = True
                base_score += 15.0
                green_flags.append("OB_MITIGATED_BULLISH_RETRACEMENT")
            elif choch_res["direction"] == "BEARISH" and (ob_price * 0.985) <= current_close <= (ob_price * 1.015):
                ob_mitigated = True
                base_score -= 15.0
                red_flags.append("OB_MITIGATED_BEARISH_RETRACEMENT")

        final_score = round(max(0.0, min(100.0, base_score)), 2)

        # --- FVG লিস্ট (ডিটেইল সহ) ---
        fvgs = fvg_res.get("fvgs_list", [])

        return {
            "symbol": symbol,
            "smc_score": final_score,
            "smc_liquidity_score": final_score,
            "choch_confirmed": bool(choch_res["confirmed"]),
            "fvg_present": bool(fvg_res["bullish_fvg"] or fvg_res["bearish_fvg"]),
            "ob_mitigated": bool(ob_mitigated),
            "market_structure": "BULLISH" if final_score > 55 else "BEARISH" if final_score < 45 else "NEUTRAL",
            "details": {
                "choch": choch_res,
                "fvg": fvg_res,
                "sweeps": sweep_res,
                "order_blocks": ob_res,
                "zone": zone_res
            },
            "green_flags": green_flags,
            "red_flags": red_flags,
            "fvgs": fvgs  # বিস্তারিত FVG লিস্ট
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
    print("🧠 TESTING UPGRADED SMART MONEY CONCEPTS (SMC) ENGINE...")
    print("==================================================")
    
    # টেস্টিং-এর জন্য এমন ডেটা বানানো হয়েছে যাতে FVG ও CHoCH ক্লিয়ারলি দেখা যায়
    dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq="15min")
    np.random.seed(42)
    
    # ইচ্ছাকৃতভাবে FVG তৈরি করার জন্য প্রাইস স্পাইক যোগ করা হয়েছে
    close_prices = 64000.0 + np.cumsum(np.random.randn(60) * 50)
    # কৃত্রিম FVG তৈরি (Index 30-এ বড় গ্যাপ)
    close_prices[30] = close_prices[28] + 400  # স্পাইক আপ
    close_prices[31] = close_prices[30] - 100
    close_prices[32] = close_prices[31] + 50
    
    high_prices = close_prices + np.random.uniform(20, 150, 60)
    low_prices = close_prices - np.random.uniform(20, 150, 60)
    open_prices = low_prices + np.random.uniform(0.0, high_prices - low_prices)
    volumes = np.random.uniform(10, 100, 60)

    mock_df = pd.DataFrame({
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volumes
    }, index=dates)

    smc = SMCEngine(fvg_threshold_pct=0.001)  # ০.১% থ্রেশহোল্ড (টেস্টের জন্য)
    result = smc.calculate_smc_score(mock_df, symbol="BTC/USDT")
    
    print("✅ SMC Engine Executed Successfully!")
    print("\n📊 Generated Output Matrix:")
    print(json.dumps(result, indent=4))
    
    print("\n🔍 FVG List Details:")
    if result.get("fvgs"):
        for idx, fvg in enumerate(result["fvgs"]):
            print(f"  FVG #{idx+1}: {fvg['type']} | Top: {fvg['top']} | Bottom: {fvg['bottom']} | Gap: {fvg['gap_size']}")
    else:
        print("  No FVG detected in this sample.")