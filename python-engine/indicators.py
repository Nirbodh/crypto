# python-engine/indicators.py
import pandas as pd
import pandas_ta as ta

class TechnicalIndicators:
    @staticmethod
    def calculate_indicators(df):
        """প্রদত্ত ডাটাফ্রেমের ওপর স্ক্যাল্পিং এবং ট্রেন্ড ইন্ডিকেটরস হিসাব করে"""
        if df is None or df.empty or len(df) < 30:  # স্ক্যাল্পিং ডাটার জন্য ৩০+ ক্যান্ডেল যথেষ্ট
            return None
            
        df = df.copy()
        
        # ১. RSI (14)
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # ২. MACD (12, 26, 9)
        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd_df is not None:
            df['MACD'] = macd_df['MACD_12_26_9']
            df['MACD_Signal'] = macd_df['MACDs_12_26_9']
        
        # ৩. EMA 20 এবং EMA 50
        df['EMA_20'] = ta.ema(df['close'], length=20)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        
        # ৪. ADX (14)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None:
            df['ADX'] = adx_df['ADX_14']
            
        # ৫. ATR (14) - ডায়নামিক SL/TP এর জন্য
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # ৬. Volume Multiple (সর্বশেষ ৫ ক্যান্ডেলের মুভিং অ্যাভারেজ ভলিউম বনাম কারেন্ট ভলিউম)
        df['Vol_MA5'] = df['volume'].rolling(window=5).mean()
        df['Volume_Multiple'] = df['volume'] / df['Vol_MA5']
        
        return df

    @classmethod
    def build_scalper_json(cls, symbol, df_4h, df_1h, df_15m, df_5m):
        """
        4H, 1H, 15M এবং 5M ডাটাফ্রেম সিঙ্ক করে আপনার স্ক্যাল্পিং ইঞ্জিনের জন্য 
        একটি ইউনিফাইড স্ট্যান্ডার্ড JSON/Dict তৈরি করে।
        """
        p_4h = cls.calculate_indicators(df_4h)
        p_1h = cls.calculate_indicators(df_1h)
        p_15m = cls.calculate_indicators(df_15m)
        p_5m = cls.calculate_indicators(df_5m)
        
        if any(x is None for x in [p_4h, p_1h, p_15m, p_5m]):
            return None
            
        # সর্বশেষ ক্যান্ডেলের ডাটা পয়েন্ট (Current State)
        c_4h = p_4h.iloc[-1]
        c_1h = p_1h.iloc[-1]
        c_15m = p_15m.iloc[-1]
        c_5m = p_5m.iloc[-1]
        
        # ইউনিফাইড স্ক্যাল্পিং ম্যাট্রিক্স
        unified_data = {
            "symbol": symbol,
            "price": {
                "close_5m": float(c_5m['close']),
                "atr_5m": float(c_5m['ATR']),
                "atr_15m": float(c_15m['ATR'])
            },
            "trends": {
                # Higher Timeframe: 4H এবং 1H ট্রেন্ড বুলিশ ফিল্টার
                "bullish_4h": bool(c_4h['EMA_20'] > c_4h['EMA_50'] and c_4h['MACD'] > c_4h['MACD_Signal']),
                "bullish_1h": bool(c_1h['EMA_20'] > c_1h['EMA_50'] and c_1h['MACD'] > c_1h['MACD_Signal']),
                
                # Execution Timeframe: 15M ট্রেন্ড বুলিশ কনফার্মেশন (EMA Cross + RSI > 50)
                "bullish_15m": bool(c_15m['EMA_20'] > c_15m['EMA_50'] and c_15m['RSI'] > 50),
                
                # Trigger Timeframe: 5M ব্রেকআউট ট্রিগার (Close > EMA 20 + RSI > 55)
                "trigger_5m": bool(c_5m['close'] > c_5m['EMA_20'] and c_5m['RSI'] > 55)
            },
            "volume": {
                "multiple_5m": round(float(c_5m['Volume_Multiple']), 2)
            },
            "market_adx": {
                "adx_15m": round(float(c_15m['ADX']), 2)
            }
        }
        
        return unified_data