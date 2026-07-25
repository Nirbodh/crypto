import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from typing import Dict, Optional, List, Tuple, Any

# ============================================================
# SECTION 1: CORE INDICATORS (FULLY FIXED)
# ============================================================
class CoreIndicators:
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. RSI (ঠিক আছে)
        df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # 2. MACD (🔥 FIX: macd_df ভেরিয়েবল সরিয়ে সোজা অ্যাসাইন করা হলো)
        macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()   # (ঐচ্ছিক)
        
        # 3. EMAs (🔥 FIX: pandas_ta.ema -> EMAIndicator)
        for period in [9, 20, 50, 100, 200]:
            df[f'EMA_{period}'] = EMAIndicator(close=df['close'], window=period).ema_indicator()
        
        # 4. ATR (🔥 FIX: pandas_ta.atr -> AverageTrueRange)
        df['ATR'] = AverageTrueRange(
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            window=14
        ).average_true_range()
        
        # 5. ADX (🔥 FIX: pandas_ta.adx -> ADXIndicator)
        adx_ind = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['ADX'] = adx_ind.adx()
        
        # 6. OBV (🔥 FIX: pandas_ta.obv -> OnBalanceVolumeIndicator)
        obv_ind = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['OBV'] = obv_ind.on_balance_volume()
        
        if df['OBV'] is not None:
            df['OBV_MA3'] = df['OBV'].rolling(3).mean()
            df['OBV_Rising'] = (df['OBV'] > df['OBV_MA3']).astype(int)
        else:
            df['OBV_Rising'] = 0
            
        return df


# ============================================================
# SECTION 2: VOLUME INDICATORS (Institutional Grade)
# ============================================================
class VolumeIndicators:
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Volume Multiples
        df['Vol_MA20'] = df['volume'].rolling(20).mean()
        df['Volume_Multiple'] = np.where(df['Vol_MA20'] > 0, df['volume'] / df['Vol_MA20'], 1.0)
        
        df['Vol_MA50'] = df['volume'].rolling(50).mean()
        df['Relative_Volume'] = np.where(df['Vol_MA50'] > 0, df['volume'] / df['Vol_MA50'], 1.0)
        
        # VWAP
        cum_vol = df['volume'].cumsum()
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['VWAP'] = np.where(cum_vol > 0, (df['volume'] * typical_price).cumsum() / cum_vol, df['close'])
        
        # Volume Profile (POC, VAH, VAL)
        df['POC'] = 0
        df['VAH'] = 0
        df['VAL'] = 0
        if len(df) >= 30:
            price_bins = np.linspace(df['low'].min(), df['high'].max(), 30)
            df['Price_Level'] = pd.cut(df['close'], bins=price_bins, labels=False)
            vol_by_level = df.groupby('Price_Level', observed=False)['volume'].sum()
            
            if not vol_by_level.empty and vol_by_level.sum() > 0:
                poc_level = vol_by_level.idxmax()
                df['POC'] = np.where(df['Price_Level'] == poc_level, 1, 0)
                
                sorted_levels = vol_by_level.sort_values(ascending=False)
                cum_v = 0
                vah_level = None
                for level, v in sorted_levels.items():
                    cum_v += v
                    if cum_v >= 0.7 * vol_by_level.sum():
                        vah_level = level
                        break
                val_level = sorted_levels.index[-1] if vah_level is not None else None
                if vah_level is not None and val_level is not None:
                    df['VAH'] = np.where(df['Price_Level'] == vah_level, 1, 0)
                    df['VAL'] = np.where(df['Price_Level'] == val_level, 1, 0)

        # Delta Volume
        df['Delta'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        return df


# ============================================================
# SECTION 3: SMART MONEY CONCEPTS (SMC) - ADVANCED
# ============================================================
class SmartMoneyIndicators:
    
    @staticmethod
    def detect_swing_points(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.DataFrame:
        df = df.copy()
        df['Swing_High'] = 0
        df['Swing_Low'] = 0
        df['Swing_High_Price'] = np.nan
        df['Swing_Low_Price'] = np.nan
        
        if len(df) < left + right + 1:
            return df
            
        high_roll_max = df['high'].rolling(window=left+right+1, center=True).max()
        low_roll_min = df['low'].rolling(window=left+right+1, center=True).min()
        
        df.loc[df['high'] == high_roll_max, 'Swing_High'] = 1
        df.loc[df['low'] == low_roll_min, 'Swing_Low'] = 1
        df.loc[df['Swing_High'] == 1, 'Swing_High_Price'] = df['high']
        df.loc[df['Swing_Low'] == 1, 'Swing_Low_Price'] = df['low']
        return df

    @staticmethod
    def detect_liquidity_sweep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Liquidity_Sweep_Bullish'] = 0
        df['Liquidity_Sweep_Bearish'] = 0
        df['Sweep_Displacement'] = 0.0
        df['Sweep_Recovery'] = 0.0
        
        df['Last_Swing_Low'] = df['Swing_Low_Price'].ffill()
        df['Last_Swing_High'] = df['Swing_High_Price'].ffill()
        
        bullish_sweep = (df['low'] < df['Last_Swing_Low']) & (df['close'] > df['Last_Swing_Low']) & (df['Volume_Multiple'] > 1.2)
        bearish_sweep = (df['high'] > df['Last_Swing_High']) & (df['close'] < df['Last_Swing_High']) & (df['Volume_Multiple'] > 1.2)
        
        atr = df['ATR'].replace(0, np.nan)
        df.loc[bullish_sweep, 'Liquidity_Sweep_Bullish'] = 1
        df.loc[bullish_sweep, 'Sweep_Displacement'] = (df.loc[bullish_sweep, 'Last_Swing_Low'] - df.loc[bullish_sweep, 'low']) / atr
        
        df.loc[bearish_sweep, 'Liquidity_Sweep_Bearish'] = 1
        df.loc[bearish_sweep, 'Sweep_Displacement'] = (df.loc[bearish_sweep, 'high'] - df.loc[bearish_sweep, 'Last_Swing_High']) / atr
        return df

    @staticmethod
    def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['FVG_Type'] = 'NONE'
        df['FVG_Upper'] = np.nan
        df['FVG_Lower'] = np.nan
        df['FVG_Gap_Size'] = 0.0
        df['FVG_Displacement'] = 0.0
        df['FVG_Mitigation_Pct'] = 0.0
        
        high_lag2 = df['high'].shift(2)
        low_lag2 = df['low'].shift(2)
        atr_val = df['ATR'].replace(0, np.nan)
        
        bullish_fvg = df['low'] > high_lag2
        bearish_fvg = df['high'] < low_lag2
        
        df.loc[bullish_fvg, 'FVG_Type'] = 'BULLISH'
        df.loc[bullish_fvg, 'FVG_Upper'] = df.loc[bullish_fvg, 'low']
        df.loc[bullish_fvg, 'FVG_Lower'] = high_lag2[bullish_fvg]
        df.loc[bullish_fvg, 'FVG_Gap_Size'] = (df.loc[bullish_fvg, 'low'] - high_lag2[bullish_fvg]) / atr_val
        
        df.loc[bearish_fvg, 'FVG_Type'] = 'BEARISH'
        df.loc[bearish_fvg, 'FVG_Upper'] = low_lag2[bearish_fvg]
        df.loc[bearish_fvg, 'FVG_Lower'] = df.loc[bearish_fvg, 'high']
        df.loc[bearish_fvg, 'FVG_Gap_Size'] = (low_lag2[bearish_fvg] - df.loc[bearish_fvg, 'high']) / atr_val
        
        candle_range = (df['high'] - df['low']).replace(0, np.nan)
        df['FVG_Displacement'] = abs(df['close'] - df['open']) / candle_range
        return df

    @staticmethod
    def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['OrderBlock_Upper'] = np.nan
        df['OrderBlock_Lower'] = np.nan
        df['OrderBlock_Type'] = 'NONE'
        df['Breaker_Block'] = 0
        df['IFVG'] = 0
        
        is_disp = (abs(df['close'] - df['open']) / df['ATR']) > 1.2
        
        for i in range(2, len(df)):
            if is_disp.iloc[i]:
                # Bullish OB (Previous bearish candle before strong displacement up)
                if df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i-1] < df['open'].iloc[i-1]:
                    df.iloc[i, df.columns.get_loc('OrderBlock_Upper')] = df['open'].iloc[i-1]
                    df.iloc[i, df.columns.get_loc('OrderBlock_Lower')] = df['close'].iloc[i-1]
                    df.iloc[i, df.columns.get_loc('OrderBlock_Type')] = 'BULLISH'
                # Bearish OB
                elif df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i-1] > df['open'].iloc[i-1]:
                    df.iloc[i, df.columns.get_loc('OrderBlock_Upper')] = df['close'].iloc[i-1]
                    df.iloc[i, df.columns.get_loc('OrderBlock_Lower')] = df['open'].iloc[i-1]
                    df.iloc[i, df.columns.get_loc('OrderBlock_Type')] = 'BEARISH'
                    
            # IFVG Logic (Mitigated and flipped FVG)
            if df['FVG_Type'].iloc[i-1] == 'BULLISH' and df['close'].iloc[i] < df['FVG_Lower'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('IFVG')] = 1
            elif df['FVG_Type'].iloc[i-1] == 'BEARISH' and df['close'].iloc[i] > df['FVG_Upper'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('IFVG')] = 1
                
        return df

    @staticmethod
    def detect_market_structure(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Structure'] = 'NEUTRAL'
        df['BOS'] = 0
        df['CHOCH'] = 0
        df['MSS'] = 0
        
        swings = df[(df['Swing_High'] == 1) | (df['Swing_Low'] == 1)].index
        if len(swings) < 2:
            return df
            
        last_high = None
        last_low = None
        last_dir = 'NEUTRAL'
        
        for idx in swings:
            if df.loc[idx, 'Swing_High'] == 1:
                curr_high = df.loc[idx, 'high']
                if last_high and curr_high > last_high:
                    df.loc[idx, 'BOS'] = 1
                    df.loc[idx, 'Structure'] = 'BULLISH'
                    if last_dir == 'BEARISH':
                        df.loc[idx, 'CHOCH'] = 1
                        if df.loc[idx, 'Volume_Multiple'] > 1.5:
                            df.loc[idx, 'MSS'] = 1
                    last_dir = 'BULLISH'
                last_high = curr_high
            elif df.loc[idx, 'Swing_Low'] == 1:
                curr_low = df.loc[idx, 'low']
                if last_low and curr_low < last_low:
                    df.loc[idx, 'BOS'] = 1
                    df.loc[idx, 'Structure'] = 'BEARISH'
                    if last_dir == 'BULLISH':
                        df.loc[idx, 'CHOCH'] = 1
                        if df.loc[idx, 'Volume_Multiple'] > 1.5:
                            df.loc[idx, 'MSS'] = 1
                    last_dir = 'BEARISH'
                last_low = curr_low
                
        df['Structure'] = df['Structure'].replace('NEUTRAL', method='ffill')
        return df

    @staticmethod
    def detect_premium_discount(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        df = df.copy()
        range_high = df['high'].rolling(period).max()
        range_low = df['low'].rolling(period).min()
        mid = (range_high + range_low) / 2
        
        df['Fair_Value'] = mid
        df['Premium_Discount'] = np.where(df['close'] > mid, 'PREMIUM', 'DISCOUNT')
        df['OTE_Zone'] = np.where(
            (df['close'] <= range_high - 0.62 * (range_high - range_low)) & 
            (df['close'] >= range_high - 0.79 * (range_high - range_low)), 1, 0
        )
        return df


# ============================================================
# SECTION 4: SESSIONS & HTF BIAS
# ============================================================
class SessionIndicators:
    @staticmethod
    def detect_session(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Session'] = 'MAIN'
        df['Kill_Zone'] = 0
        
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            hours = df['timestamp'].dt.hour
            minutes = df['timestamp'].dt.minute
            
            df.loc[(hours >= 8) & (hours < 16), 'Session'] = 'LONDON'
            df.loc[(hours >= 13) & (hours < 22), 'Session'] = 'NY'
            df.loc[hours < 8, 'Session'] = 'ASIA'
            
            london_kz = ((hours == 8) & (minutes >= 30)) | ((hours == 9) & (minutes < 30))
            ny_kz = ((hours == 14) & (minutes >= 30)) | ((hours == 15) & (minutes < 30))
            df.loc[london_kz | ny_kz, 'Kill_Zone'] = 1
            
        return df


# ============================================================
# SECTION 5: SEPARATE DYNAMIC SCORING ENGINE (Scalable)
# ============================================================
class ScoringEngine:
    """
    Modular and Weighted Scoring Engine.
    Converts individual signals into a Normalized Score (-100 to +100).
    """
    
    # Category Weights (Total = 1.0 or 100%)
    WEIGHTS = {
        'trend_structure': 0.25,
        'liquidity': 0.20,
        'smc_arrays': 0.20,
        'volume_momentum': 0.15,
        'ema_alignment': 0.10,
        'session_context': 0.10
    }

    @classmethod
    def evaluate(cls, row: pd.Series, htf_bias: str = 'NEUTRAL', smt_divergence: bool = False) -> Dict[str, Any]:
        
        # 1. Trend Structure Score (-100 to +100)
        struct_score = 0
        if row.get('Structure') == 'BULLISH': struct_score += 50
        elif row.get('Structure') == 'BEARISH': struct_score -= 50
        if row.get('CHOCH') == 1 or row.get('MSS') == 1:
            struct_score += 50 if row.get('Structure') == 'BULLISH' else -50
        
        # 2. Liquidity Sweeps
        liq_score = 0
        if row.get('Liquidity_Sweep_Bullish') == 1:
            disp = row.get('Sweep_Displacement', 1.0)
            liq_score = min(100, int(40 * disp))
        elif row.get('Liquidity_Sweep_Bearish') == 1:
            disp = row.get('Sweep_Displacement', 1.0)
            liq_score = -min(100, int(40 * disp))

        # 3. SMC Arrays (FVG, OB, IFVG, BPR)
        smc_score = 0
        fvg_type = row.get('FVG_Type', 'NONE')
        if fvg_type == 'BULLISH':
            disp = row.get('FVG_Displacement', 0.5)
            smc_score += 40 if disp > 0.6 else 20
        elif fvg_type == 'BEARISH':
            disp = row.get('FVG_Displacement', 0.5)
            smc_score -= 40 if disp > 0.6 else 20
            
        ob_type = row.get('OrderBlock_Type', 'NONE')
        if ob_type == 'BULLISH': smc_score += 30
        elif ob_type == 'BEARISH': smc_score -= 30
        
        if row.get('IFVG') == 1: smc_score += 20 if row.get('close') > row.get('open') else -20
        if row.get('OTE_Zone') == 1: smc_score += 10 if row.get('Premium_Discount') == 'DISCOUNT' else -10

        # 4. Volume & Momentum (Relative Volume, Delta, RSI Penalties, ADX)
        vol_score = 0
        rvol = row.get('Relative_Volume', 1.0)
        if rvol > 1.5: vol_score += 30 if row.get('Delta', 0) > 0 else -30
        
        adx = row.get('ADX', 20)
        adx_bonus = 10 if adx > 25 else (20 if adx > 35 else 0)
        vol_score += adx_bonus if struct_score >= 0 else -adx_bonus
        
        # RSI Overbought/Oversold Penalties
        rsi = row.get('RSI', 50)
        if rsi > 70: vol_score -= 25  # Penalty for Longs
        elif rsi < 30: vol_score += 25  # Penalty for Shorts

        # 5. EMA Alignment Score
        ema_score = 0
        c = row.get('close', 0)
        e9, e20, e50 = row.get('EMA_9', 0), row.get('EMA_20', 0), row.get('EMA_50', 0)
        if c > e9 > e20 > e50: ema_score = 100
        elif c < e9 < e20 < e50: ema_score = -100

        # 6. Session & HTF Alignment
        session_score = 0
        if row.get('Kill_Zone') == 1: session_score += 30
        if htf_bias == 'BULLISH': session_score += 50
        elif htf_bias == 'BEARISH': session_score -= 50
        if smt_divergence: session_score += 20

        # Calculate Weighted Final Normalized Score (-100 to +100)
        final_score = (
            (struct_score * cls.WEIGHTS['trend_structure']) +
            (liq_score * cls.WEIGHTS['liquidity']) +
            (smc_score * cls.WEIGHTS['smc_arrays']) +
            (vol_score * cls.WEIGHTS['volume_momentum']) +
            (ema_score * cls.WEIGHTS['ema_alignment']) +
            (session_score * cls.WEIGHTS['session_context'])
        )
        
        normalized_score = float(np.clip(final_score, -100.0, 100.0))
        return {
            "confluence_score": round(normalized_score, 2),
            "breakdown": {
                "structure": struct_score,
                "liquidity": liq_score,
                "smc": smc_score,
                "momentum": vol_score,
                "ema": ema_score
            }
        }


# ============================================================
# SECTION 6: INSTITUTIONAL JSON & AI VECTOR BUILDER
# ============================================================
class InstitutionalJSONBuilder:
    
    @staticmethod
    def build_output(
        df: pd.DataFrame, 
        symbol: str, 
        timeframe: str,
        htf_bias: str = 'BULLISH',
        macro_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        
        last = df.iloc[-1]
        
        # Calculate Dynamic Scoring via ScoringEngine
        scoring_res = ScoringEngine.evaluate(last, htf_bias=htf_bias)
        score = scoring_res['confluence_score']
        
        # 5-Tier Recommendation System
        if score >= 65: rec = "STRONG BUY"
        elif 25 <= score < 65: rec = "BUY"
        elif -25 < score < 25: rec = "WAIT"
        elif -65 < score <= -25: rec = "SELL"
        else: rec = "STRONG SELL"
        
        # Dynamic Confidence Score (%)
        confidence_pct = min(99.0, round(abs(score) * 0.85 + (last.get('ADX', 20) * 0.3), 1))

        # Comprehensive Risk Factors
        risk_factors = []
        if last.get('RSI', 50) > 75: risk_factors.append("RSI_OVERBOUGHT_PENALTY")
        if last.get('RSI', 50) < 25: risk_factors.append("RSI_OVERSOLD_PENALTY")
        if last.get('ADX', 20) < 20: risk_factors.append("LOW_TREND_STRENGTH_ADX")
        if last.get('Kill_Zone', 0) == 0: risk_factors.append("OUTSIDE_KILL_ZONE")
        
        # External Macro/Derivatives Risks
        if macro_data:
            if macro_data.get('funding_rate', 0) > 0.03: risk_factors.append("HIGH_POSITIVE_FUNDING_RATE")
            if macro_data.get('is_weekend', False): risk_factors.append("WEEKEND_LOW_LIQUIDITY")
            if macro_data.get('high_impact_news', False): risk_factors.append("HIGH_IMPACT_NEWS_EVENT")

        # Explicit AI Feature Vector (Optimized for LLM Context Windows)
        ai_feature_vector = {
            "symbol": symbol,
            "tf": timeframe,
            "trend_structure": str(last.get('Structure', 'NEUTRAL')),
            "htf_bias": htf_bias,
            "adx": float(round(last.get('ADX', 0), 1)),
            "rsi": float(round(last.get('RSI', 0), 1)),
            "relative_volume": float(round(last.get('Relative_Volume', 1.0), 2)),
            "fvg_status": str(last.get('FVG_Type', 'NONE')),
            "order_block": str(last.get('OrderBlock_Type', 'NONE')),
            "liquidity_swept": bool(last.get('Liquidity_Sweep_Bullish') or last.get('Liquidity_Sweep_Bearish')),
            "pricing_zone": str(last.get('Premium_Discount', 'NEUTRAL')),
            "session": str(last.get('Session', 'MAIN')),
            "kill_zone_active": bool(last.get('Kill_Zone', 0) == 1)
        }

        # Construct Master Institutional JSON
        output = {
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": str(last.get('timestamp', pd.Timestamp.now())),
                "engine_version": "v3.0-Production-AI-Ready"
            },
            "ai_feature_vector": ai_feature_vector,
            "confluence_score": score,
            "confidence": f"{confidence_pct}%",
            "recommendation": rec,
            "scoring_breakdown": scoring_res['breakdown'],
            "market_condition": {
                "market_structure": last.get('Structure', 'NEUTRAL'),
                "htf_bias": htf_bias,
                "session": last.get('Session', 'MAIN'),
                "in_kill_zone": bool(last.get('Kill_Zone', 0) == 1),
                "premium_discount": last.get('Premium_Discount', 'NEUTRAL')
            },
            "liquidity_map": {
                "swing_high": float(last.get('Last_Swing_High', 0.0)),
                "swing_low": float(last.get('Last_Swing_Low', 0.0)),
                "poc_vwap": float(last.get('VWAP', 0.0)),
                "fvg_upper": None if np.isnan(last.get('FVG_Upper', np.nan)) else float(last.get('FVG_Upper')),
                "fvg_lower": None if np.isnan(last.get('FVG_Lower', np.nan)) else float(last.get('FVG_Lower')),
                "order_block_zone": [
                    None if np.isnan(last.get('OrderBlock_Lower', np.nan)) else float(last.get('OrderBlock_Lower')),
                    None if np.isnan(last.get('OrderBlock_Upper', np.nan)) else float(last.get('OrderBlock_Upper'))
                ]
            },
            "scenario_analysis": {
                "bullish_invalidation": float(last.get('Last_Swing_Low', 0.0)),
                "bearish_invalidation": float(last.get('Last_Swing_High', 0.0)),
                "target_liquidity": float(last.get('Last_Swing_High', 0.0)) if score >= 0 else float(last.get('Last_Swing_Low', 0.0))
            },
            "risk_factors": risk_factors
        }
        
        return output


# ============================================================
# SECTION 7: PIPELINE EXECUTION ENGINE
# ============================================================
class InstitutionalTradingEngine:
    """Master Engine executing Data Pipeline -> Indicators -> SMC -> Scoring -> AI JSON"""
    
    @staticmethod
    def process_data(df: pd.DataFrame, symbol: str = "BTCUSDT", timeframe: str = "15m") -> Dict[str, Any]:
        # 1. Calculate Technicals & Volume
        df = CoreIndicators.calculate(df)
        df = VolumeIndicators.calculate(df)
        
        # 2. Calculate SMC Concepts
        df = SmartMoneyIndicators.detect_swing_points(df)
        df = SmartMoneyIndicators.detect_liquidity_sweep(df)
        df = SmartMoneyIndicators.detect_fvg(df)
        df = SmartMoneyIndicators.detect_order_blocks(df)
        df = SmartMoneyIndicators.detect_market_structure(df)
        df = SmartMoneyIndicators.detect_premium_discount(df)
        
        # 3. Calculate Sessions
        df = SessionIndicators.detect_session(df)
        
        # Dummy Macro/Derivatives data simulation
        macro_context = {
            "funding_rate": 0.01,
            "is_weekend": False,
            "high_impact_news": False
        }
        
        # 4. Build Institutional Output JSON
        final_json = InstitutionalJSONBuilder.build_output(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            htf_bias="BULLISH",
            macro_data=macro_context
        )
        
        return final_json


# ============================================================
# SECTION 8: DEMO RUNNER
# ============================================================
if __name__ == "__main__":
    import json
    
    # Simulate Real Market Data (100 Candles)
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='15min')
    price_changes = np.random.randn(100) * 15
    close_prices = 65000 + np.cumsum(price_changes)
    
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices - np.random.rand(100)*10,
        'high': close_prices + np.random.rand(100)*20,
        'low': close_prices - np.random.rand(100)*20,
        'close': close_prices,
        'volume': np.random.randint(100, 1000, size=100)
    })
    
    # Run Complete Engine Pipeline
    result = InstitutionalTradingEngine.process_data(sample_df, symbol="BTCUSDT", timeframe="15m")
    
    # Print Beautiful JSON Output
    print(json.dumps(result, indent=2))