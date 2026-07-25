import pandas as pd
import numpy as np
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from typing import Dict, Optional, List, Tuple

# ============================================================
# SECTION 1: CORE INDICATORS (FULLY FIXED)
# ============================================================
class CoreIndicators:
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        # ---------- RSI ----------
        df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()

        # ---------- MACD ----------
        macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

        # ---------- EMAs (replaced ta.ema) ----------
        for period in [9, 20, 50, 100, 200]:
            df[f'EMA_{period}'] = EMAIndicator(close=df['close'], window=period).ema_indicator()

        # ---------- ATR (replaced ta.atr) ----------
        df['ATR'] = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        ).average_true_range()

        # ---------- ADX (replaced ta.adx) ----------
        adx_ind = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['ADX'] = adx_ind.adx()

        # ---------- OBV (replaced ta.obv) ----------
        obv_ind = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['OBV'] = obv_ind.on_balance_volume()
        if df['OBV'] is not None:
            df['OBV_MA3'] = df['OBV'].rolling(3).mean()
            df['OBV_Rising'] = (df['OBV'] > df['OBV_MA3']).astype(int)
        else:
            df['OBV_Rising'] = 0

        return df


# ============================================================
# SECTION 2: VOLUME INDICATORS
# ============================================================
class VolumeIndicators:
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        df['Vol_MA20'] = df['volume'].rolling(20).mean()
        df['Volume_Multiple'] = np.where(df['Vol_MA20'] > 0, df['volume'] / df['Vol_MA20'], 1.0)

        cum_vol = df['volume'].cumsum()
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['VWAP'] = np.where(cum_vol > 0, (df['volume'] * typical_price).cumsum() / cum_vol, df['close'])

        df['Vol_MA50'] = df['volume'].rolling(50).mean()
        df['Relative_Volume'] = np.where(df['Vol_MA50'] > 0, df['volume'] / df['Vol_MA50'], 1.0)

        # Volume Profile (POC, VAH, VAL)
        df['POC'] = 0
        df['VAH'] = 0
        df['VAL'] = 0

        if len(df) >= 30:
            price_min, price_max = df['low'].min(), df['high'].max()
            if price_max > price_min:
                price_bins = np.linspace(price_min, price_max, 30)
                df['Price_Level'] = pd.cut(df['close'], bins=price_bins, labels=False, include_lowest=True)
                vol_by_level = df.groupby('Price_Level', observed=False)['volume'].sum()

                if not vol_by_level.empty and vol_by_level.sum() > 0:
                    poc_level = vol_by_level.idxmax()
                    df['POC'] = np.where(df['Price_Level'] == poc_level, 1, 0)

                    sorted_levels = vol_by_level.sort_values(ascending=False)
                    cum_vol_target = 0.7 * vol_by_level.sum()
                    running_vol = 0
                    value_area_levels = []

                    for level, vol in sorted_levels.items():
                        running_vol += vol
                        value_area_levels.append(level)
                        if running_vol >= cum_vol_target:
                            break

                    if value_area_levels:
                        vah_level = max(value_area_levels)
                        val_level = min(value_area_levels)
                        df['VAH'] = np.where(df['Price_Level'] == vah_level, 1, 0)
                        df['VAL'] = np.where(df['Price_Level'] == val_level, 1, 0)

                df.drop(columns=['Price_Level'], errors='ignore', inplace=True)

        df['Delta'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        return df


# ============================================================
# SECTION 3: SMART MONEY CONCEPTS (SMC)
# ============================================================
class SmartMoneyIndicators:

    @staticmethod
    def detect_swing_points(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.DataFrame:
        df['Swing_High'] = 0
        df['Swing_Low'] = 0
        df['Swing_High_Price'] = np.nan
        df['Swing_Low_Price'] = np.nan

        if len(df) < left + right + 1:
            return df

        high_vals = df['high'].values
        low_vals = df['low'].values

        for i in range(left, len(df) - right):
            window_highs = high_vals[i-left : i+right+1]
            if high_vals[i] == np.max(window_highs):
                df.iloc[i, df.columns.get_loc('Swing_High')] = 1
                df.iloc[i, df.columns.get_loc('Swing_High_Price')] = high_vals[i]

            window_lows = low_vals[i-left : i+right+1]
            if low_vals[i] == np.min(window_lows):
                df.iloc[i, df.columns.get_loc('Swing_Low')] = 1
                df.iloc[i, df.columns.get_loc('Swing_Low_Price')] = low_vals[i]

        return df

    @staticmethod
    def detect_equal_highs_lows(df: pd.DataFrame, tolerance: float = 0.001) -> pd.DataFrame:
        df['Equal_High'] = 0
        df['Equal_Low'] = 0
        if len(df) < 3:
            return df

        highs = df['high'].values
        lows = df['low'].values

        for i in range(2, len(df)):
            if highs[i] > 0 and abs(highs[i] - highs[i-1]) / highs[i] <= tolerance:
                df.iloc[i, df.columns.get_loc('Equal_High')] = 1
            if lows[i] > 0 and abs(lows[i] - lows[i-1]) / lows[i] <= tolerance:
                df.iloc[i, df.columns.get_loc('Equal_Low')] = 1
        return df

    @staticmethod
    def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        df['Liquidity_Sweep_Bullish'] = 0
        df['Liquidity_Sweep_Bearish'] = 0
        df['Sweep_Displacement'] = 0.0
        df['Sweep_Recovery'] = 0.0

        if len(df) < lookback + 2:
            return df

        prev_swing_low = df['Swing_Low_Price'].ffill().shift(1).rolling(lookback).min()
        prev_swing_high = df['Swing_High_Price'].ffill().shift(1).rolling(lookback).max()

        vol_spike = df['Volume_Multiple'] > 1.5
        displacement = df['Relative_Volume'] > 1.5

        bullish_sweep = (df['low'] < prev_swing_low) & (df['close'] > prev_swing_low) & (vol_spike | displacement)
        bearish_sweep = (df['high'] > prev_swing_high) & (df['close'] < prev_swing_high) & (vol_spike | displacement)

        atr = df['ATR'].replace(0, np.nan)

        df.loc[bullish_sweep, 'Liquidity_Sweep_Bullish'] = 1
        df.loc[bullish_sweep, 'Sweep_Displacement'] = (df.loc[bullish_sweep, 'low'] - prev_swing_low[bullish_sweep]) / atr[bullish_sweep]
        df.loc[bullish_sweep, 'Sweep_Recovery'] = (df.loc[bullish_sweep, 'close'] - prev_swing_low[bullish_sweep]) / atr[bullish_sweep]

        df.loc[bearish_sweep, 'Liquidity_Sweep_Bearish'] = 1
        df.loc[bearish_sweep, 'Sweep_Displacement'] = (df.loc[bearish_sweep, 'high'] - prev_swing_high[bearish_sweep]) / atr[bearish_sweep]
        df.loc[bearish_sweep, 'Sweep_Recovery'] = (df.loc[bearish_sweep, 'close'] - prev_swing_high[bearish_sweep]) / atr[bearish_sweep]

        return df

    @staticmethod
    def detect_fvg(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        df['FVG_Type'] = 'NONE'
        df['FVG_Upper'] = np.nan
        df['FVG_Lower'] = np.nan
        df['FVG_Gap_Size'] = 0.0
        df['FVG_Displacement'] = 0.0
        df['FVG_Mitigation_Pct'] = 0.0
        df['FVG_Invalidated'] = 0

        if len(df) < 3:
            return df

        high_lag2 = df['high'].shift(2)
        low_lag2 = df['low'].shift(2)
        atr_val = df['ATR'].replace(0, np.nan)

        bullish_fvg = df['low'] > high_lag2
        bearish_fvg = df['high'] < low_lag2

        df.loc[bullish_fvg, 'FVG_Type'] = 'BULLISH'
        df.loc[bullish_fvg, 'FVG_Upper'] = df.loc[bullish_fvg, 'low']
        df.loc[bullish_fvg, 'FVG_Lower'] = high_lag2[bullish_fvg]
        df.loc[bullish_fvg, 'FVG_Gap_Size'] = (df.loc[bullish_fvg, 'low'] - high_lag2[bullish_fvg]) / atr_val[bullish_fvg]

        candle_range = (df['high'] - df['low']).replace(0, np.nan)
        df.loc[bullish_fvg, 'FVG_Displacement'] = abs(df.loc[bullish_fvg, 'close'] - df.loc[bullish_fvg, 'open']) / candle_range[bullish_fvg]

        df.loc[bearish_fvg, 'FVG_Type'] = 'BEARISH'
        df.loc[bearish_fvg, 'FVG_Upper'] = low_lag2[bearish_fvg]
        df.loc[bearish_fvg, 'FVG_Lower'] = df.loc[bearish_fvg, 'high']
        df.loc[bearish_fvg, 'FVG_Gap_Size'] = (low_lag2[bearish_fvg] - df.loc[bearish_fvg, 'high']) / atr_val[bearish_fvg]
        df.loc[bearish_fvg, 'FVG_Displacement'] = abs(df.loc[bearish_fvg, 'close'] - df.loc[bearish_fvg, 'open']) / candle_range[bearish_fvg]

        return df

    @staticmethod
    def detect_premium_discount(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        df['Premium_Discount'] = 0.0
        df['Fair_Value'] = np.nan
        df['OTE_Upper'] = np.nan
        df['OTE_Lower'] = np.nan

        if len(df) < period:
            return df

        rolling_high = df['high'].rolling(period).max()
        rolling_low = df['low'].rolling(period).min()
        mid = (rolling_high + rolling_low) / 2

        df['Fair_Value'] = mid
        range_val = (rolling_high - rolling_low).replace(0, np.nan)

        df['Premium_Discount'] = np.where(
            df['close'] > mid,
            (df['close'] - mid) / (rolling_high - mid + 1e-8),
            -(mid - df['close']) / (mid - rolling_low + 1e-8)
        )
        df['Premium_Discount'] = df['Premium_Discount'].clip(-1, 1)

        df['OTE_Upper'] = rolling_high - 0.62 * range_val
        df['OTE_Lower'] = rolling_high - 0.79 * range_val
        return df

    @staticmethod
    def detect_market_structure(df: pd.DataFrame) -> pd.DataFrame:
        df['Structure'] = 'NEUTRAL'
        df['BOS'] = 0
        df['CHOCH'] = 0
        df['MSS'] = 0
        df['HH'] = 0
        df['HL'] = 0
        df['LH'] = 0
        df['LL'] = 0

        swing_indices = df[(df['Swing_High'] == 1) | (df['Swing_Low'] == 1)].index
        if len(swing_indices) < 2:
            return df

        last_high, last_low = None, None
        last_structure = 'NEUTRAL'

        for idx in swing_indices:
            is_high = df.loc[idx, 'Swing_High'] == 1
            is_low = df.loc[idx, 'Swing_Low'] == 1

            if is_high:
                curr_high = df.loc[idx, 'high']
                if last_high is not None:
                    if curr_high > last_high:
                        df.loc[idx, 'HH'] = 1
                        df.loc[idx, 'BOS'] = 1
                        df.loc[idx, 'Structure'] = 'BULLISH'
                        if last_structure == 'BEARISH':
                            df.loc[idx, 'CHOCH'] = 1
                            if df.loc[idx, 'Volume_Multiple'] > 1.5:
                                df.loc[idx, 'MSS'] = 1
                        last_structure = 'BULLISH'
                    else:
                        df.loc[idx, 'LH'] = 1
                last_high = curr_high

            if is_low:
                curr_low = df.loc[idx, 'low']
                if last_low is not None:
                    if curr_low < last_low:
                        df.loc[idx, 'LL'] = 1
                        df.loc[idx, 'BOS'] = 1
                        df.loc[idx, 'Structure'] = 'BEARISH'
                        if last_structure == 'BULLISH':
                            df.loc[idx, 'CHOCH'] = 1
                            if df.loc[idx, 'Volume_Multiple'] > 1.5:
                                df.loc[idx, 'MSS'] = 1
                        last_structure = 'BEARISH'
                    else:
                        df.loc[idx, 'HL'] = 1
                last_low = curr_low

        df['Structure'] = df['Structure'].replace('NEUTRAL', np.nan).ffill().fillna('NEUTRAL')
        return df

    @staticmethod
    def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
        df['OrderBlock_Upper'] = np.nan
        df['OrderBlock_Lower'] = np.nan
        df['OrderBlock_Type'] = 'NONE'
        df['Breaker_Block'] = 0
        df['Mitigation_Block'] = 0

        if len(df) < 10:
            return df

        for i in range(1, len(df)):
            if df['Swing_Low'].iloc[i] == 1 and df['Volume_Multiple'].iloc[i] > 1.2:
                df.iloc[i, df.columns.get_loc('OrderBlock_Upper')] = df['open'].iloc[i-1]
                df.iloc[i, df.columns.get_loc('OrderBlock_Lower')] = df['close'].iloc[i-1]
                df.iloc[i, df.columns.get_loc('OrderBlock_Type')] = 'BULLISH'

            elif df['Swing_High'].iloc[i] == 1 and df['Volume_Multiple'].iloc[i] > 1.2:
                df.iloc[i, df.columns.get_loc('OrderBlock_Upper')] = df['close'].iloc[i-1]
                df.iloc[i, df.columns.get_loc('OrderBlock_Lower')] = df['open'].iloc[i-1]
                df.iloc[i, df.columns.get_loc('OrderBlock_Type')] = 'BEARISH'

        return df

    @staticmethod
    def detect_displacement(df: pd.DataFrame) -> pd.DataFrame:
        df['Displacement'] = 0
        df['Displacement_Strength'] = 0.0

        if len(df) < 20:
            return df

        body = abs(df['close'] - df['open'])
        avg_body = body.rolling(20).mean()
        atr = df['ATR'].replace(0, np.nan)

        mask = (body > avg_body * 1.5) & (df['Volume_Multiple'] > 1.5)
        df.loc[mask, 'Displacement'] = 1
        df.loc[mask, 'Displacement_Strength'] = (body[mask] / atr[mask]).clip(upper=10.0)
        return df


# ============================================================
# SECTION 4: SESSION INDICATORS
# ============================================================
class SessionIndicators:
    @staticmethod
    def detect_session(df: pd.DataFrame) -> pd.DataFrame:
        df['Session'] = 'NONE'
        df['Kill_Zone'] = 0

        if 'timestamp' not in df.columns:
            return df

        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        hours = df['timestamp'].dt.hour
        minutes = df['timestamp'].dt.minute

        df.loc[hours < 8, 'Session'] = 'ASIA'
        df.loc[(hours >= 8) & (hours < 16), 'Session'] = 'LONDON'
        df.loc[(hours >= 13) & (hours < 22), 'Session'] = 'NY'

        london_kz = ((hours == 8) & (minutes >= 30)) | ((hours == 9) & (minutes < 30))
        ny_kz = ((hours == 14) & (minutes >= 30)) | ((hours == 15) & (minutes < 30))

        df.loc[london_kz | ny_kz, 'Kill_Zone'] = 1
        return df

    @staticmethod
    def get_session_score(df: pd.DataFrame) -> float:
        """
        Calculates session-based score from the last row of the DataFrame.
        Returns a score from 0 to 100, where higher means better session context.
        """
        if df is None or df.empty:
            return 50.0

        # Ensure session columns exist
        if 'Session' not in df.columns or 'Kill_Zone' not in df.columns:
            df = SessionIndicators.detect_session(df)

        if df.empty:
            return 50.0

        last = df.iloc[-1]
        score = 50.0  # neutral base

        # Kill Zone bonus (London 8:30-9:30 or NY 14:30-15:30)
        if last.get('Kill_Zone', 0) == 1:
            score += 30

        # London or NY session bonus (more liquidity)
        if last.get('Session', '') in ['LONDON', 'NY']:
            score += 20

        # Cap at 100
        return min(score, 100.0)

# ============================================================
# SECTION 5: UNIFIED WRAPPER CLASS
# ============================================================
class TechnicalIndicators:
    @classmethod
    def calculate_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        df = CoreIndicators.calculate(df)
        df = VolumeIndicators.calculate(df)
        df = SmartMoneyIndicators.detect_swing_points(df)
        df = SmartMoneyIndicators.detect_equal_highs_lows(df)
        df = SmartMoneyIndicators.detect_liquidity_sweep(df)
        df = SmartMoneyIndicators.detect_fvg(df)
        df = SmartMoneyIndicators.detect_premium_discount(df)
        df = SmartMoneyIndicators.detect_market_structure(df)
        df = SmartMoneyIndicators.detect_order_blocks(df)
        df = SmartMoneyIndicators.detect_displacement(df)
        df = SessionIndicators.detect_session(df)

        return df

    @classmethod
    def calculate_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Alias method to guarantee compatibility with TechnicalScreener."""
        return cls.calculate_all(df)

    @classmethod
    def build_unified_json(cls, symbol: str, df_4h: pd.DataFrame, df_1h: pd.DataFrame) -> Optional[Dict]:
        if df_4h is None or df_1h is None or df_4h.empty or df_1h.empty:
            return None

        df_4h_calc = cls.calculate_all(df_4h.copy())
        df_1h_calc = cls.calculate_all(df_1h.copy())

        row_4h = df_4h_calc.iloc[-1]
        row_1h = df_1h_calc.iloc[-1]

        def clean_val(val, default=0.0):
            if pd.isna(val) or np.isinf(val):
                return default
            return round(float(val), 4)

        return {
            "symbol": symbol,
            "current_price": clean_val(row_1h['close']),
            "timeframe_4h": {
                "rsi": clean_val(row_4h.get('RSI', 50)),
                "macd": clean_val(row_4h.get('MACD', 0)),
                "macd_signal": clean_val(row_4h.get('MACD_Signal', 0)),
                "adx": clean_val(row_4h.get('ADX', 20)),
                "atr": clean_val(row_4h.get('ATR', 0)),
                "structure": str(row_4h.get('Structure', 'NEUTRAL')),
                "fvg_type": str(row_4h.get('FVG_Type', 'NONE')),
                "ob_type": str(row_4h.get('OrderBlock_Type', 'NONE')),
                "premium_discount": clean_val(row_4h.get('Premium_Discount', 0))
            },
            "timeframe_1h": {
                "rsi": clean_val(row_1h.get('RSI', 50)),
                "macd": clean_val(row_1h.get('MACD', 0)),
                "macd_signal": clean_val(row_1h.get('MACD_Signal', 0)),
                "adx": clean_val(row_1h.get('ADX', 20)),
                "atr": clean_val(row_1h.get('ATR', 0)),
                "structure": str(row_1h.get('Structure', 'NEUTRAL')),
                "fvg_type": str(row_1h.get('FVG_Type', 'NONE')),
                "ob_type": str(row_1h.get('OrderBlock_Type', 'NONE')),
                "session": str(row_1h.get('Session', 'NONE')),
                "kill_zone": int(row_1h.get('Kill_Zone', 0)),
                "volume_multiple": clean_val(row_1h.get('Volume_Multiple', 1.0))
            }
        }

    @classmethod
    def build_scalper_json(
        cls,
        symbol: str,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_30m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_5m: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Builds a simplified JSON for scalping with only essential fields.
        Now includes 30m timeframe data.
        """
        if any(df is None or df.empty for df in [df_4h, df_1h, df_30m, df_15m, df_5m]):
            return None

        df_4h_calc = cls.calculate_all(df_4h.copy())
        df_1h_calc = cls.calculate_all(df_1h.copy())
        df_30m_calc = cls.calculate_all(df_30m.copy())
        df_15m_calc = cls.calculate_all(df_15m.copy())
        df_5m_calc = cls.calculate_all(df_5m.copy())

        row_4h = df_4h_calc.iloc[-1]
        row_1h = df_1h_calc.iloc[-1]
        row_30m = df_30m_calc.iloc[-1]
        row_15m = df_15m_calc.iloc[-1]
        row_5m = df_5m_calc.iloc[-1]

        def clean_val(val, default=0.0):
            if pd.isna(val) or np.isinf(val):
                return default
            return round(float(val), 4)

        return {
            "symbol": symbol,
            "price": clean_val(row_5m["close"]),
            "timeframes": {
                "4h": {
                    "close": clean_val(row_4h["close"]),
                    "rsi": clean_val(row_4h.get("RSI")),
                    "ema20": clean_val(row_4h.get("EMA_20")),
                    "ema50": clean_val(row_4h.get("EMA_50"))
                },
                "1h": {
                    "close": clean_val(row_1h["close"]),
                    "rsi": clean_val(row_1h.get("RSI")),
                    "macd": clean_val(row_1h.get("MACD"))
                },
                "30m": {
                    "close": clean_val(row_30m["close"]),
                    "rsi": clean_val(row_30m.get("RSI")),
                    "volume": clean_val(row_30m.get("volume"))
                },
                "15m": {
                    "close": clean_val(row_15m["close"]),
                    "rsi": clean_val(row_15m.get("RSI")),
                    "volume": clean_val(row_15m.get("volume"))
                },
                "5m": {
                    "close": clean_val(row_5m["close"]),
                    "rsi": clean_val(row_5m.get("RSI")),
                    "atr": clean_val(row_5m.get("ATR"))
                }
            }
        }
