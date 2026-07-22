# python-engine/score_fusion_engine.py

class ScoreFusionEngine:
    def __init__(self):
        # Weight adjustment: More emphasis on Technical for entry timing
        self.weights = {
            "technical": 0.45,
            "derivatives": 0.25,
            "fundamental": 0.15,
            "sentiment": 0.15
        }

    def fuse_scores(self, tech_data: dict, fund_data: dict, deriv_data: dict, sent_data: dict) -> dict:
        tech_score = float(tech_data.get("technical_score", tech_data.get("score", 50)))
        deriv_score = float(deriv_data.get("derivatives_score", 50))
        fund_score = float(fund_data.get("fundamental_score", 50))
        sent_score = float(sent_data.get("sentiment_score", 50))

        unified_score = (
            (tech_score * self.weights["technical"]) +
            (deriv_score * self.weights["derivatives"]) +
            (fund_score * self.weights["fundamental"]) +
            (sent_score * self.weights["sentiment"])
        )

        unified_score = round(max(0, min(100, unified_score)), 2)

        all_red_flags = tech_data.get("red_flags", []) + fund_data.get("red_flags", []) + deriv_data.get("red_flags", []) + sent_data.get("red_flags", [])
        all_green_flags = tech_data.get("green_flags", []) + fund_data.get("green_flags", []) + deriv_data.get("green_flags", []) + sent_data.get("green_flags", [])

        # GATEKEEPER RULE: Threshold lowered to 75 & allowing up to 2 red flags
        pass_threshold = unified_score >= 75.0 and len(all_red_flags) <= 2

        return {
            "symbol": tech_data.get("symbol", "UNKNOWN"),
            "unified_score": unified_score,
            "pass_to_ai_debate": pass_threshold,
            "breakdown": {
                "technical": tech_score,
                "derivatives": deriv_score,
                "fundamental": fund_score,
                "sentiment": sent_score
            },
            "all_red_flags": all_red_flags,
            "all_green_flags": all_green_flags,
            "fusion_verdict": "PASS_TO_AI_DEBATE" if pass_threshold else "REJECTED_BY_QUANT_FUSION"
        }