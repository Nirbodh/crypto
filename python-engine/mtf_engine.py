from typing import Dict, Any, List, Optional

class MultiTimeframeEngine:
    """
    Institutional Multi-Timeframe (MTF) Alignment Matrix:
    - Daily: Macro Trend, Liquidity Sweeps & Key Levels
    - 4H: Market Structure Shift (MSS) & CHOCH Confirmation
    - 1H: Fair Value Gap (FVG) & Order Block (OB) Mitigation
    - 30M: Intermediate Structure Confirmation (NEW)
    - 15M/5M: Micro Trigger & Optimal Trade Entry (OTE)
    """

    @staticmethod
    def evaluate_mtf_alignment(
        daily_data: Optional[dict] = None,
        h4_data: Optional[dict] = None,
        h1_data: Optional[dict] = None,
        m30_data: Optional[dict] = None,
        m15_data: Optional[dict] = None,
        **kwargs  # extra args for future expansion
    ) -> Dict[str, Any]:
        
        daily_data = daily_data or {}
        h4_data = h4_data or {}
        h1_data = h1_data or {}
        m30_data = m30_data or {}
        m15_data = m15_data or {}

        score = 0
        confluences: List[str] = []
        warnings: List[str] = []
        critical_conflicts: List[str] = []

        # -------------------------------------------------------------
        # 1. Daily Bias (HTF Macro Trend & Key Liquidity Sweeps)
        # -------------------------------------------------------------
        daily_bias = str(daily_data.get("bias", daily_data.get("structure", "NEUTRAL"))).upper()
        daily_sweep = (
            daily_data.get("liquidity_swept", False) or 
            daily_data.get("key_liquidity_swept", False) or
            daily_data.get("bsl_swept", False) or
            daily_data.get("ssl_swept", False) or
            daily_data.get("Liquidity_Sweep_Bullish", 0) == 1 or
            daily_data.get("Liquidity_Sweep_Bearish", 0) == 1
        )
        
        if daily_sweep:
            score += 30
            confluences.append("Daily Key Liquidity Swept (BSL/SSL)")
            
        if daily_bias in ["BULLISH", "BEARISH"]:
            score += 20
            confluences.append(f"Daily Macro Trend {daily_bias.capitalize()}")

        # -------------------------------------------------------------
        # 2. 4H Structure (Market Structure Shift / CHOCH)
        # -------------------------------------------------------------
        h4_choch = (
            h4_data.get("choch_confirmed", False) or 
            h4_data.get("mss_confirmed", False) or 
            h4_data.get("CHOCH", 0) == 1 or 
            h4_data.get("MSS", 0) == 1
        )
        h4_structure = str(h4_data.get("structure", h4_data.get("bias", "NEUTRAL"))).upper()

        if h4_choch:
            score += 25
            confluences.append("4H CHOCH/MSS Confirmed (Market Structure Shift)")
        elif h4_structure in ["BULLISH", "BEARISH"]:
            score += 15
            confluences.append(f"4H Structure {h4_structure.capitalize()}")

        # -------------------------------------------------------------
        # 3. 1H Zone (Mitigation & Institutional Interest Zone)
        # -------------------------------------------------------------
        h1_fvg = (
            h1_data.get("fvg_present", False) or 
            h1_data.get("fvg_active", False) or 
            h1_data.get("fvg_type", "NONE") in ["BULLISH", "BEARISH"]
        )
        h1_ob = (
            h1_data.get("ob_mitigated", False) or 
            h1_data.get("ob_present", False) or 
            h1_data.get("in_discount_zone", False) or
            h1_data.get("ob_type", "NONE") in ["BULLISH", "BEARISH"]
        )

        if h1_fvg or h1_ob:
            score += 15
            confluences.append("1H Unmitigated FVG / Order Block Zone Active")

        # -------------------------------------------------------------
        # 4. 30M Intermediate Confirmation (NEW)
        # -------------------------------------------------------------
        m30_structure = str(m30_data.get("structure", m30_data.get("bias", "NEUTRAL"))).upper()
        m30_choch = (
            m30_data.get("choch_confirmed", False) or
            m30_data.get("mss_confirmed", False)
        )
        m30_entry = (
            m30_data.get("entry_signal", False) or
            m30_data.get("trigger_confirmed", False)
        )

        if m30_structure in ["BULLISH", "BEARISH"]:
            # Add bonus if aligns with daily/4H
            if (daily_bias == "BULLISH" and m30_structure == "BULLISH") or \
               (daily_bias == "BEARISH" and m30_structure == "BEARISH"):
                score += 12
                confluences.append(f"30M Structure aligns with Daily ({m30_structure})")
            else:
                score += 6
                confluences.append(f"30M Structure {m30_structure.capitalize()} (neutral alignment)")

        if m30_choch:
            score += 8
            confluences.append("30M CHOCH/MSS Confirmed")

        if m30_entry:
            score += 5
            confluences.append("30M Entry Signal Active")

        # -------------------------------------------------------------
        # 5. 15M Entry Setup (Micro Trigger & Execution Zone)
        # -------------------------------------------------------------
        m15_entry = (
            m15_data.get("entry_signal", False) or 
            m15_data.get("trigger_confirmed", False) or
            m15_data.get("Displacement", 0) == 1
        )
        m15_entry_type = str(m15_data.get("entry_type", m15_data.get("direction", "NEUTRAL"))).upper()

        if m15_entry:
            score += 10
            confluences.append("15M Micro Entry Signal Triggered")
        else:
            warnings.append("15M Micro Entry not yet fully confirmed")

        # -------------------------------------------------------------
        # 6. Multi-Timeframe Structural Conflict Checks
        # -------------------------------------------------------------
        # Critical: Taking LONG against Daily Bearish or SHORT against Daily Bullish
        if daily_bias == "BEARISH" and m15_entry_type in ["LONG", "BULLISH"]:
            score -= 30
            msg = "⚠️ Critical Conflict: Taking LONG into Daily Bearish Macro Bias"
            warnings.append(msg)
            critical_conflicts.append(msg)
        elif daily_bias == "BULLISH" and m15_entry_type in ["SHORT", "BEARISH"]:
            score -= 30
            msg = "⚠️ Critical Conflict: Taking SHORT into Daily Bullish Macro Bias"
            warnings.append(msg)
            critical_conflicts.append(msg)

        # Intermediate: 4H vs Daily conflict
        if daily_bias == "BULLISH" and h4_structure == "BEARISH" and not h4_choch:
            score -= 15
            warnings.append("⚠️ Intermediate Conflict: 4H Market Structure is Bearish against Daily Bullish")
        elif daily_bias == "BEARISH" and h4_structure == "BULLISH" and not h4_choch:
            score -= 15
            warnings.append("⚠️ Intermediate Conflict: 4H Market Structure is Bullish against Daily Bearish")

        # New: 30M vs 4H conflict (if both are opposite)
        if h4_structure in ["BULLISH", "BEARISH"] and m30_structure in ["BULLISH", "BEARISH"]:
            if h4_structure != m30_structure:
                score -= 8
                warnings.append(f"⚠️ Minor Conflict: 30M Structure ({m30_structure}) differs from 4H ({h4_structure})")

        final_score = max(0, min(100, score))
        is_aligned = (final_score >= 70) and (len(critical_conflicts) == 0)

        # Alignment Strength
        if final_score >= 85:
            alignment_strength = "STRONG"
        elif final_score >= 70:
            alignment_strength = "MEDIUM"
        else:
            alignment_strength = "WEAK"

        return {
            "mtf_score": final_score,
            "is_aligned": is_aligned,
            "alignment_strength": alignment_strength,
            "confluences": confluences,
            "warnings": warnings,
            "critical_conflicts": critical_conflicts,
            "daily_bias": daily_bias,
            "h4_structure": h4_structure,
            "m30_structure": m30_structure,
            "m15_entry_type": m15_entry_type
        }

    @classmethod
    def evaluate_mtf(
        cls,
        daily_data: Optional[dict] = None,
        h4_data: Optional[dict] = None,
        h1_data: Optional[dict] = None,
        m30_data: Optional[dict] = None,
        m15_data: Optional[dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Alias method for seamless integration across pipeline engines.
        Now supports all timeframes (including 30M).
        """
        return cls.evaluate_mtf_alignment(
            daily_data=daily_data,
            h4_data=h4_data,
            h1_data=h1_data,
            m30_data=m30_data,
            m15_data=m15_data,
            **kwargs
        )


if __name__ == "__main__":
    d_data = {"bias": "BULLISH", "liquidity_swept": True}
    h4_d = {"structure": "BULLISH", "choch_confirmed": True}
    h1_d = {"fvg_present": True, "ob_present": True}
    m30_d = {"structure": "BULLISH", "choch_confirmed": True, "entry_signal": True}
    m15_d = {"entry_signal": True, "entry_type": "LONG"}

    result = MultiTimeframeEngine.evaluate_mtf(d_data, h4_d, h1_d, m30_d, m15_d)
    print("MTF Score:", result["mtf_score"])
    print("Is Aligned:", result["is_aligned"])
    print("Strength:", result["alignment_strength"])
    print("Confluences:", result["confluences"])
    print("Warnings:", result["warnings"])
    print("Critical Conflicts:", result["critical_conflicts"])