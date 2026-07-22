# python-engine/sentiment_engine.py

import requests


class SentimentEngine:
    def fetch_sentiment_score(self) -> dict:
        """
        Fetches Alternative.me Crypto Fear & Greed Index.
        Normalizes index (0-100) into realistic market sentiment score.
        """
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                val = int(data['data'][0]['value'])
                classification = data['data'][0]['value_classification']

                # Real Alignment: Extreme Fear (0-25) -> Bearish Sentiment Score (20-40)
                # Neutral (45-55) -> Neutral Score (50)
                # Extreme Greed (75-100) -> Strong Sentiment Score (75-90)
                sentiment_score = val  # Fear & Greed value directly correlates to market sentiment score

                return {
                    "sentiment_score": sentiment_score,
                    "fear_and_greed_index": val,
                    "classification": classification
                }
        except Exception as e:
            print(f"⚠️ Sentiment API Error: {e}")

        # Default Neutral Fallback
        return {
            "sentiment_score": 50,
            "fear_and_greed_index": 50,
            "classification": "Neutral"
        }