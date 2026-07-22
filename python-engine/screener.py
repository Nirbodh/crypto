# python-engine/screener.py

import pandas as pd
import numpy as np


class TechnicalScreener:
    """
    Multi-Timeframe Technical Screener for Crypto Quant Pipeline.
    Evaluates 5m, 15m, 30m, 1h, and 4h timeframes to assign a 0-100 score.
    """

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates essential technical indicators if not already present."""
        if df is None or df.empty or len(df) < 30:
            return df

        df = df.copy()

        # 1. Exponential Moving Averages (EMA)
        if 'ema_20' not in df.columns:
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        if 'ema_50' not in df.columns:
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        if 'ema_200' not in df.columns:
            df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # 2. Relative Strength Index (RSI - 14)
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))

        # 3. MACD (12, 26, 9)
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

        # 4. Average True Range (ATR - 14)
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14).mean()

        # 5. Volume Moving Average (MA 20)
        if 'vol_ma20' not in df.columns:
            df['vol_ma20'] = df['volume'].rolling(window=20).mean()

        return df

    def _score_timeframe(self, df: pd.DataFrame, tf_weight: float) -> dict:
        """Evaluates technical setup for a single timeframe (Scores 0 to 100)."""
        if df is None or len(df) < 20:
            return {"score": 50, "signals": []}

        df = self._calculate_indicators(df)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        score = 50.0  # Base neutral score

        # A. Trend Alignment (EMA 20 vs EMA 50)
        close = last['close']
        ema20 = last.get('ema_20', close)
        ema50 = last.get('ema_50', close)

        if close > ema20 > ema50:
            score += 15
            signals.append("Bullish EMA Alignment (Close > EMA20 > EMA50)")
        elif close < ema20 < ema50:
            score -= 15
            signals.append("Bearish EMA Alignment (Close < EMA20 < EMA50)")

        # B. RSI Momentum (Refined logic)
        rsi = last.get('rsi', 50)
        if 50 <= rsi <= 68:
            score += 15
            signals.append(f"Bullish RSI Momentum ({rsi:.1f})")
        elif 32 <= rsi < 50:
            score -= 10
            signals.append(f"Bearish RSI Momentum ({rsi:.1f})")
        elif rsi > 70:
            score += 5
            signals.append(f"Overbought RSI ({rsi:.1f})")
        elif rsi < 30:
            score += 5  # Reduced to +5 for cautious oversold evaluation
            signals.append(f"Potential Oversold Bounce RSI ({rsi:.1f})")

        # C. MACD Crossover / Histogram
        macd = last.get('macd', 0)
        macd_sig = last.get('macd_signal', 0)
        prev_macd = prev.get('macd', 0)
        prev_sig = prev.get('macd_signal', 0)

        if prev_macd <= prev_sig and macd > macd_sig:
            score += 15
            signals.append("Bullish MACD Cross")
        elif macd > macd_sig:
            score += 8
            signals.append("MACD Above Signal")
        elif prev_macd >= prev_sig and macd < macd_sig:
            score -= 15
            signals.append("Bearish MACD Cross")

        # D. Volume Expansion
        vol = last.get('volume', 0)
        vol_ma = last.get('vol_ma20', 1)
        if vol_ma > 0 and (vol / vol_ma) >= 1.5:
            score += 10
            signals.append(f"Volume Spike ({vol / vol_ma:.1f}x)")

        # Bound score between 0 and 100
        final_tf_score = max(0.0, min(100.0, score))
        return {"score": final_tf_score, "signals": signals}

    def run_screener(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_30m: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame
    ) -> dict:
        """
        Calculates Multi-Timeframe Technical Score using dynamic weighted averages:
        4H: 30%, 1H: 25%, 30M: 20%, 15M: 15%, 5M: 10%
        """
        try:
            res_4h = self._score_timeframe(df_4h, tf_weight=0.30)
            res_1h = self._score_timeframe(df_1h, tf_weight=0.25)
            res_30m = self._score_timeframe(df_30m, tf_weight=0.20)
            res_15m = self._score_timeframe(df_15m, tf_weight=0.15)
            res_5m = self._score_timeframe(df_5m, tf_weight=0.10)

            # Weighted Scoring Engine
            total_score = (
                (res_4h['score'] * 0.30) +
                (res_1h['score'] * 0.25) +
                (res_30m['score'] * 0.20) +
                (res_15m['score'] * 0.15) +
                (res_5m['score'] * 0.10)
            )

            all_signals = {
                "4h": res_4h['signals'],
                "1h": res_1h['signals'],
                "30m": res_30m['signals'],
                "15m": res_15m['signals'],
                "5m": res_5m['signals']
            }

            return {
                "technical_score": round(total_score, 2),
                "breakdown": {
                    "score_4h": res_4h['score'],
                    "score_1h": res_1h['score'],
                    "score_30m": res_30m['score'],
                    "score_15m": res_15m['score'],
                    "score_5m": res_5m['score']
                },
                "signals": all_signals
            }

        except Exception as e:
            print(f"⚠️ Screener Execution Error: {e}")
            return {
                "technical_score": 50.0,
                "breakdown": {},
                "signals": [f"Error: {str(e)}"]
            }