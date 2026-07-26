# python-engine/json_builder.py

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Literal

# Import institutional engines
try:
    from technical_engine import TechnicalEngine
    from liquidity_engine import InstitutionalLiquidityEngine
    from risk_engine import RiskEngine
    from score_fusion_engine import InstitutionalScoreFusionEngine
    from smc_engine import SMCEngine
    ENGINES_AVAILABLE = True
except ImportError:
    ENGINES_AVAILABLE = False


class JSONBuilder:
    """
    v3.0 - Institutional JSON Builder (formerly ScalperJSONBuilder)
    - Configurable scoring weights
    - Score breakdown: trend, smc, volume, liquidity, risk
    - Probability-based output (bull/bear/range)
    - Detailed trade plan (entry zones, multiple TPs)
    - Institutional signals (OI, Funding, Order Flow)
    - AI-friendly flattened summary
    - Market regime classification
    - Entry quality grading
    - Execution timing recommendations
    """

    # ============================================================
    # CONFIGURABLE WEIGHTS
    # ============================================================
    WEIGHTS = {
        "structure": 25,
        "liquidity_sweep": 20,
        "fvg": 15,
        "volume": 15,
        "premium_discount": 10,
        "killzone": 15,
    }

    # Risk Penalties
    RISK_PENALTIES = {
        "weak_trend": 10,      # ADX < 20
        "no_killzone": 8,      # Off-session
        "high_atr": 7,         # ATR > 3%
        "funding_extreme": 12, # Funding > 0.05% or < -0.05%
        "low_volume": 6,       # RVOL < 0.5
        "wide_spread": 5,      # Spread > 0.1%
        "high_leverage": 10,   # Leverage > 5x
    }

    # ============================================================
    # MARKET REGIME MAP
    # ============================================================
    REGIME_MAP = {
        "TRENDING": {"bias": "BULLISH" if "bull" else "BEARISH", "confidence_boost": 1.2},
        "RANGING": {"bias": "NEUTRAL", "confidence_boost": 0.8},
        "VOLATILE": {"bias": "NEUTRAL", "confidence_boost": 0.7},
        "EXPANSION": {"bias": "BULLISH", "confidence_boost": 1.3},
        "COMPRESSION": {"bias": "NEUTRAL", "confidence_boost": 0.6},
        "DISTRIBUTION": {"bias": "BEARISH", "confidence_boost": 0.9},
        "ACCUMULATION": {"bias": "BULLISH", "confidence_boost": 1.1},
    }

    @classmethod
    def build_json(
        cls,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        account_balance: float = 1000.0,
        market_regime: Literal["TRENDING", "RANGING", "VOLATILE", "EXPANSION", "COMPRESSION", "DISTRIBUTION", "ACCUMULATION"] = "TRENDING",
        # Optional pre-computed engine results
        tech_result: Optional[Dict[str, Any]] = None,
        liq_result: Optional[Dict[str, Any]] = None,
        risk_result: Optional[Dict[str, Any]] = None,
        score_result: Optional[Dict[str, Any]] = None,
        smc_result: Optional[Dict[str, Any]] = None,
        # Custom weights (optional)
        custom_weights: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Builds a complete Institutional JSON payload with all quantitative features.
        """
        if df.empty or len(df) < 20:
            return json.dumps({"error": "Insufficient data (minimum 20 candles required)"}, indent=2)

        # ---- 1. Clean Data ----
        df_clean = df.copy()
        df_clean = df_clean.replace([np.nan, np.inf, -np.inf], None)

        if not ENGINES_AVAILABLE:
            return cls._build_fallback_json(df_clean, symbol, timeframe)

        # ---- 2. Run Engines if not provided ----
        if tech_result is None:
            tech_engine = TechnicalEngine()
            atr_ratio = (df_clean['close'].pct_change().std() * 100) / 2.0
            tech_result = tech_engine.analyze(
                df_15m=df_clean,
                market_regime=market_regime,
                atr_ratio_15m=atr_ratio
            )

        if liq_result is None:
            liq_result = InstitutionalLiquidityEngine.analyze_liquidity_and_candles(
                df_ltf=df_clean,
                market_regime=market_regime
            )

        if smc_result is None:
            smc_engine = SMCEngine()
            smc_result = smc_engine.calculate_smc_score(df_clean, symbol=symbol)

        if risk_result is None:
            risk_engine = RiskEngine()
            latest_close = float(df_clean['close'].iloc[-1])
            atr_val = float(tech_result.get('features', {}).get('atr_ratio_15m', 0.01) * latest_close or 100)
            risk_result = risk_engine.calculate_trade_risk(
                entry_price=latest_close,
                atr_5m=atr_val,
                account_balance=account_balance,
                direction="LONG",
                market_regime=market_regime,
                include_advanced_metrics=True
            )

        if score_result is None:
            score_result = InstitutionalScoreFusionEngine.fuse_scores(
                symbol=symbol,
                tech_score=tech_result.get("technical_score", 50),
                smc_score=smc_result.get("smc_score", 50),
                liquidity_score=liq_result.get("liquidity_score", 50),
                safety_score=risk_result.get("advanced_metrics", {}).get("safety_score", 50),
                position_quality_score=risk_result.get("advanced_metrics", {}).get("position_quality_score", 50),
                effective_leverage=risk_result.get("position_sizing", {}).get("effective_leverage_needed", 1.0),
                market_regime=market_regime,
                estimated_win_rate=0.55,
                rr_ratio=risk_result.get("risk_metrics", {}).get("rr_score_raw", 2.0)
            )

        # ---- 3. Extract Core Data ----
        latest_close = float(df_clean['close'].iloc[-1])
        latest_high = float(df_clean['high'].iloc[-1])
        latest_low = float(df_clean['low'].iloc[-1])
        latest_open = float(df_clean['open'].iloc[-1])
        latest_volume = float(df_clean['volume'].iloc[-1])

        features = tech_result.get("features", {})
        adx = features.get("adx_15m", 25.0)
        rsi = features.get("rsi_15m", 50.0)
        cmf = features.get("cmf_15m", 0.0)

        # ---- 4. Check Missing Indicators ----
        required_indicators = ["ATR", "ADX", "Volume_Multiple", "Premium_Discount"]
        missing_indicators = []
        for ind in required_indicators:
            if ind not in df_clean.columns and ind not in features:
                missing_indicators.append(ind)

        # ---- 5. Configurable Scoring (using WEIGHTS) ----
        weights = custom_weights or cls.WEIGHTS
        score_breakdown = {}

        # 5a. Market Structure Score
        market_bias = tech_result.get("market_bias", "NEUTRAL")
        structure_score = weights["structure"] if market_bias in ["BULLISH", "BEARISH"] else 0
        score_breakdown["structure"] = structure_score if market_bias == "BULLISH" else -structure_score

        # 5b. Liquidity Sweep Score
        sweeps = liq_result.get("sweeps", {})
        liquidity_score = 0
        if sweeps.get("ssl_sweep"):
            liquidity_score = weights["liquidity_sweep"]
        elif sweeps.get("bsl_sweep"):
            liquidity_score = -weights["liquidity_sweep"]
        score_breakdown["liquidity_sweep"] = liquidity_score

        # 5c. FVG Score
        fvg_bullish = smc_result.get("fvg_present", False) and smc_result.get("market_structure") == "BULLISH"
        fvg_bearish = smc_result.get("fvg_present", False) and smc_result.get("market_structure") == "BEARISH"
        fvg_score = weights["fvg"] if fvg_bullish else (-weights["fvg"] if fvg_bearish else 0)
        score_breakdown["fvg"] = fvg_score

        # 5d. Volume Score
        vol_mult = features.get("vol_expansion_ratio", 1.0)
        volume_score = 0
        if vol_mult > 1.5:
            volume_score = weights["volume"] if latest_close > latest_open else -weights["volume"]
        score_breakdown["volume"] = volume_score

        # 5e. Premium/Discount Score
        prem_disc = liq_result.get("features", {}).get("price_vs_poc_atr", 0.0)
        premium_score = weights["premium_discount"] if prem_disc < -0.5 else (-weights["premium_discount"] if prem_disc > 0.5 else 0)
        score_breakdown["premium_discount"] = premium_score

        # 5f. Killzone Score
        killzone_active = liq_result.get("features", {}).get("kill_zone_active", False)
        killzone_score = weights["killzone"] if killzone_active else 0
        score_breakdown["killzone"] = killzone_score

        # ---- 6. Risk Penalties ----
        risk_penalties = {}
        total_risk_penalty = 0

        if adx < 20:
            risk_penalties["weak_trend"] = cls.RISK_PENALTIES["weak_trend"]
            total_risk_penalty += cls.RISK_PENALTIES["weak_trend"]

        if not killzone_active:
            risk_penalties["no_killzone"] = cls.RISK_PENALTIES["no_killzone"]
            total_risk_penalty += cls.RISK_PENALTIES["no_killzone"]

        atr_pct = features.get("atr_ratio_15m", 0.01) * 100
        if atr_pct > 3.0:
            risk_penalties["high_atr"] = cls.RISK_PENALTIES["high_atr"]
            total_risk_penalty += cls.RISK_PENALTIES["high_atr"]

        funding_rate = smc_result.get("details", {}).get("funding_rate", 0.0)
        if abs(funding_rate) > 0.0005:
            risk_penalties["funding_extreme"] = cls.RISK_PENALTIES["funding_extreme"]
            total_risk_penalty += cls.RISK_PENALTIES["funding_extreme"]

        rvol = features.get("vol_expansion_ratio", 1.0)
        if rvol < 0.5:
            risk_penalties["low_volume"] = cls.RISK_PENALTIES["low_volume"]
            total_risk_penalty += cls.RISK_PENALTIES["low_volume"]

        effective_leverage = risk_result.get("position_sizing", {}).get("effective_leverage_needed", 1.0)
        if effective_leverage > 5:
            risk_penalties["high_leverage"] = cls.RISK_PENALTIES["high_leverage"]
            total_risk_penalty += cls.RISK_PENALTIES["high_leverage"]

        # ---- 7. Calculate Total Score (0-100 normalized) ----
        raw_score = sum(score_breakdown.values()) - total_risk_penalty
        institutional_score = max(0, min(100, (raw_score + 100) / 2))

        # ---- 8. Trend Score, SMC Score, Risk Score (separate) ----
        trend_score = (score_breakdown.get("structure", 0) + score_breakdown.get("killzone", 0)) / 2 + 50
        trend_score = max(0, min(100, trend_score))

        smc_score_value = (score_breakdown.get("fvg", 0) + score_breakdown.get("liquidity_sweep", 0)) / 2 + 50
        smc_score_value = max(0, min(100, smc_score_value))

        volume_score_value = score_breakdown.get("volume", 0) + 50
        volume_score_value = max(0, min(100, volume_score_value))

        risk_score_value = 100 - total_risk_penalty
        risk_score_value = max(0, min(100, risk_score_value))

        # ---- 9. Probability Calculation ----
        bull_prob = max(0, min(100, institutional_score * 0.7 + (100 - total_risk_penalty) * 0.3))
        bear_prob = max(0, min(100, 100 - institutional_score * 0.7 + total_risk_penalty * 0.3))
        range_prob = max(0, 100 - bull_prob - bear_prob)

        # ---- 10. Market Regime ----
        regime_info = cls.REGIME_MAP.get(market_regime, {"bias": "NEUTRAL", "confidence_boost": 1.0})

        # ---- 11. Confidence Calculation ----
        base_confidence = (institutional_score / 100) * 90
        confidence = min(
            95,
            int(
                base_confidence
                * (0.8 + (adx / 100) * 0.3)
                * (0.9 + (rvol / 3) * 0.2)
                * regime_info["confidence_boost"]
            )
        )
        confidence = max(10, min(95, confidence))

        # ---- 12. Entry Quality ----
        if confidence >= 85 and institutional_score >= 75:
            entry_grade = "A"
        elif confidence >= 70 and institutional_score >= 60:
            entry_grade = "B"
        elif confidence >= 55 and institutional_score >= 50:
            entry_grade = "C"
        else:
            entry_grade = "D"

        # ---- 13. Trade Plan ----
        atr_val = features.get("atr_ratio_15m", 0.01) * latest_close or 100
        entry_buffer = atr_val * 0.15

        trade_plan = {
            "entry_type": "LIMIT",
            "entry_zone": [
                round(latest_close - entry_buffer, 4),
                round(latest_close + entry_buffer, 4)
            ],
            "stop_loss": round(latest_close - (atr_val * 1.5), 4) if market_bias == "BULLISH" else round(latest_close + (atr_val * 1.5), 4),
            "tp1": round(latest_close + (atr_val * 1.5), 4) if market_bias == "BULLISH" else round(latest_close - (atr_val * 1.5), 4),
            "tp2": round(latest_close + (atr_val * 2.5), 4) if market_bias == "BULLISH" else round(latest_close - (atr_val * 2.5), 4),
            "tp3": round(latest_close + (atr_val * 4.0), 4) if market_bias == "BULLISH" else round(latest_close - (atr_val * 4.0), 4),
            "partial_exit": [40, 30, 30],
            "move_sl_to_be_after_tp1": True
        }

        # ---- 14. Execution Timing ----
        execution = {
            "execute_now": confidence >= 75 and institutional_score >= 65,
            "wait_retest": confidence >= 60 and institutional_score >= 55 and confidence < 75,
            "confirmation_needed": confidence < 60,
            "best_session": "LONDON" if killzone_active else "NY",
            "urgency": "HIGH" if confidence >= 80 else "MEDIUM" if confidence >= 60 else "LOW"
        }

        # ---- 15. Institutional Signals ----
        oi_change = smc_result.get("details", {}).get("oi_change_24h", 0.0)
        institutional_signals = {
            "smart_money_bias": market_bias,
            "orderflow": "BUY" if cmf > 0.05 else "SELL" if cmf < -0.05 else "NEUTRAL",
            "whale_activity": "HIGH" if features.get("vol_expansion_ratio", 1.0) > 2.0 else "MEDIUM" if features.get("vol_expansion_ratio", 1.0) > 1.5 else "LOW",
            "funding": "BULLISH" if funding_rate < -0.01 else "BEARISH" if funding_rate > 0.02 else "NEUTRAL",
            "oi_trend": "RISING" if oi_change > 5 else "FALLING" if oi_change < -5 else "FLAT",
            "liquidity_grab": sweeps.get("bsl_sweep") or sweeps.get("ssl_sweep"),
            "fvg_count": len(smc_result.get("fvgs", [])),
            "order_block_price": smc_result.get("order_block_price", 0.0),
            "breaker_block": smc_result.get("ob_mitigated", False)
        }

        # ---- 16. AI-Friendly Flat Summary ----
        quant_summary = {
            "institutional_score": round(institutional_score, 1),
            "market_bias": market_bias,
            "confidence": confidence,
            "risk_level": "LOW" if total_risk_penalty < 15 else "MEDIUM" if total_risk_penalty < 30 else "HIGH",
            "ev_r": risk_result.get("risk_metrics", {}).get("rr_score_raw", 2.0),
            "trade_direction": "LONG" if market_bias == "BULLISH" else "SHORT" if market_bias == "BEARISH" else "NEUTRAL",
            "score_breakdown": {
                "trend": round(trend_score, 1),
                "smc": round(smc_score_value, 1),
                "volume": round(volume_score_value, 1),
                "risk": round(risk_score_value, 1)
            }
        }

        # ---- 17. Risk Management (for AI Debate Engine) ----
        risk_management = {
            "valid_trade": confidence >= 55 and institutional_score >= 50,
            "direction": quant_summary["trade_direction"],
            "trade_levels": {
                "entry_price": round(latest_close, 4),
                "stop_loss_price": trade_plan["stop_loss"],
                "take_profit_price": trade_plan["tp1"],
                "sl_percentage": round((abs(latest_close - trade_plan["stop_loss"]) / latest_close) * 100, 2),
                "tp_percentage": round((abs(trade_plan["tp1"] - latest_close) / latest_close) * 100, 2)
            },
            "position_sizing": {
                "quantity": risk_result.get("position_sizing", {}).get("quantity", 0.0),
                "position_value_usdt": risk_result.get("position_sizing", {}).get("position_value_usdt", 0.0),
                "effective_leverage_needed": risk_result.get("position_sizing", {}).get("effective_leverage_needed", 1.0)
            },
            "risk_metrics": {
                "atr_5m": atr_val,
                "risk_reward_ratio": f"1:{risk_result.get('risk_metrics', {}).get('rr_score_raw', 2.0)}",
                "rr_score_raw": risk_result.get("risk_metrics", {}).get("rr_score_raw", 2.0)
            }
        }

        # ---- 18. Assemble Final JSON ----
        scalper_payload: Dict[str, Any] = {
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": str(df_clean.index[-1]) if hasattr(df_clean.index, 'str') else str(df_clean.iloc[-1].get('timestamp', '')),
                "analyst_mode": "Institutional Quant Research v3.0"
            },
            "market_condition": {
                "current_price": round(latest_close, 4),
                "overall_bias": market_bias,
                "confidence": confidence,
                "confidence_level": "HIGH" if confidence >= 80 else "MEDIUM" if confidence >= 60 else "LOW",
                "institutional_score": round(institutional_score, 1),
                "market_regime": market_regime,
                "regime_bias": regime_info["bias"],
                "adx": round(adx, 1),
                "rsi": round(rsi, 1),
                "cmf": round(cmf, 3),
                "atr": round(atr_val, 4)
            },
            "probabilities": {
                "bull": round(bull_prob, 1),
                "bear": round(bear_prob, 1),
                "range": round(range_prob, 1)
            },
            "score_breakdown": {
                "trend": round(trend_score, 1),
                "smc": round(smc_score_value, 1),
                "volume": round(volume_score_value, 1),
                "risk": round(risk_score_value, 1),
                "total": round(institutional_score, 1)
            },
            "risk_penalties": risk_penalties,
            "missing_indicators": missing_indicators,
            "confluences_detected": cls._build_confluences(
                tech_result, liq_result, smc_result, institutional_signals
            ),
            "liquidity_map": {
                "bsl_sweep": sweeps.get("bsl_sweep", False),
                "ssl_sweep": sweeps.get("ssl_sweep", False),
                "liquidity_grab": sweeps.get("bsl_sweep") or sweeps.get("ssl_sweep"),
                "fvg_present": smc_result.get("fvg_present", False),
                "fvg_count": len(smc_result.get("fvgs", [])),
                "order_block_price": smc_result.get("order_block_price", 0.0),
                "breakers": smc_result.get("ob_mitigated", False),
                "premium_discount_ratio": round(prem_disc, 2)
            },
            "key_levels": {
                "support_1": round(latest_close - atr_val, 4),
                "support_2": round(float(df_clean['low'].tail(20).min()), 4),
                "resistance_1": round(latest_close + atr_val, 4),
                "resistance_2": round(float(df_clean['high'].tail(20).max()), 4)
            },
            "scenario_analysis": {
                "bullish": {
                    "probability": round(bull_prob, 1),
                    "trigger": f"Hold above Support 1 ({round(latest_close - atr_val, 4):.2f}) with volume surge",
                    "targets": [trade_plan["tp1"], trade_plan["tp2"], trade_plan["tp3"]],
                    "invalidation": round(latest_close - atr_val * 2, 4)
                },
                "bearish": {
                    "probability": round(bear_prob, 1),
                    "trigger": f"Rejection at Resistance 1 ({round(latest_close + atr_val, 4):.2f})",
                    "targets": [trade_plan["tp1"], trade_plan["tp2"], trade_plan["tp3"]],
                    "invalidation": round(latest_close + atr_val * 2, 4)
                }
            },
            "trade_plan": trade_plan,
            "execution": execution,
            "entry_quality": {
                "grade": entry_grade,
                "score": round(institutional_score, 1)
            },
            "institutional_signals": institutional_signals,
            "quant_summary": quant_summary,
            "risk_management": risk_management  # <-- ADDED for AI debate engine compatibility
        }

        # Clean NaN values
        def clean_nan(obj):
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(v) for v in obj]
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            elif pd.isna(obj):
                return None
            else:
                return obj

        clean_payload = clean_nan(scalper_payload)
        return json.dumps(clean_payload, indent=2)

    @classmethod
    def _build_confluences(
        cls,
        tech_result: Dict[str, Any],
        liq_result: Dict[str, Any],
        smc_result: Dict[str, Any],
        signals: Dict[str, Any]
    ) -> List[str]:
        """Builds a list of confluence signals."""
        confluences = []
        features = tech_result.get("features", {})

        if tech_result.get("market_bias") in ["BULLISH", "BEARISH"]:
            confluences.append(f"Market Bias: {tech_result['market_bias']}")

        if features.get("trend_alignment") in ["STRONG_BULLISH", "STRONG_BEARISH"]:
            confluences.append(f"Strong Trend Alignment: {features['trend_alignment']}")

        if liq_result.get("sweeps", {}).get("ssl_sweep"):
            confluences.append("Sell-side Liquidity Sweep (SSL)")
        if liq_result.get("sweeps", {}).get("bsl_sweep"):
            confluences.append("Buy-side Liquidity Sweep (BSL)")

        if smc_result.get("fvg_present"):
            confluences.append(f"FVG Present: {len(smc_result.get('fvgs', []))} zones")

        if smc_result.get("ob_mitigated"):
            confluences.append("Order Block Mitigated")

        if signals.get("orderflow") in ["BUY", "SELL"]:
            confluences.append(f"Orderflow: {signals['orderflow']}")

        if signals.get("whale_activity") == "HIGH":
            confluences.append("High Whale Activity Detected")

        if features.get("rsi_15m", 50) > 55:
            confluences.append("Bullish RSI Momentum")
        elif features.get("rsi_15m", 50) < 45:
            confluences.append("Bearish RSI Momentum")

        return confluences[:8]  # Limit to 8 confluences

    @staticmethod
    def _build_fallback_json(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """Fallback JSON when institutional engines are not available."""
        latest = df.iloc[-1]
        return json.dumps({
            "metadata": {"symbol": symbol, "timeframe": timeframe, "analyst_mode": "FALLBACK"},
            "market_condition": {"current_price": float(latest['close']), "overall_bias": "NEUTRAL"},
            "error": "Institutional engines not available. Install technical_engine, liquidity_engine, risk_engine, score_fusion_engine."
        }, indent=2)


# ============================================================
# UNIT TEST (Run with: python json_builder.py)
# ============================================================
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="15min")
    close = 65000 + np.cumsum(np.random.randn(100) * 50)
    high = close + np.abs(np.random.randn(100) * 30)
    low = close - np.abs(np.random.randn(100) * 30)
    open_ = low + np.random.rand(100) * (high - low)
    volume = np.random.randint(10, 100, 100)

    mock_df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume
    }, index=dates)

    print("🧪 Testing JSONBuilder v3.0...")
    print("=" * 70)

    result = JSONBuilder.build_json(
        df=mock_df,
        symbol="BTC/USDT",
        timeframe="15m",
        account_balance=10000.0,
        market_regime="TRENDING"
    )

    parsed = json.loads(result)
    print(f"✅ Institutional Score: {parsed['market_condition']['institutional_score']}")
    print(f"✅ Confidence: {parsed['market_condition']['confidence']}%")
    print(f"✅ Entry Grade: {parsed['entry_quality']['grade']}")
    print(f"✅ Probabilities: Bull={parsed['probabilities']['bull']}%, Bear={parsed['probabilities']['bear']}%, Range={parsed['probabilities']['range']}%")
    print(f"✅ Score Breakdown: {json.dumps(parsed['score_breakdown'], indent=2)}")
    print(f"✅ Risk Management (trade_levels): {json.dumps(parsed.get('risk_management', {}).get('trade_levels', {}), indent=2)}")
    print("=" * 70)
    print("✅ JSON built successfully. Full output saved to json_output.json")

    with open("json_output.json", "w") as f:
        f.write(result)
