# python-engine/market_context.py
import requests
import ccxt

class MarketContext:
    
    @staticmethod
    def get_fear_and_greed_index():
        """Crypto Fear & Greed Index ফেচ করে।"""
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=5)
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                value = int(data["data"][0]["value"])
                classification = data["data"][0]["value_classification"]
                return {"value": value, "sentiment": classification}
        except Exception as e:
            print(f"⚠️ Error fetching Fear & Greed Index: {e}")
            
        return {"value": 50, "sentiment": "Neutral"}

    @staticmethod
    def get_derivatives_metrics(symbol="BTC/USDT"):
        """Binance Derivatives এপিআই থেকে Funding Rate এবং Open Interest ফেচ করে।"""
        metrics = {
            "funding_rate": 0.01,
            "funding_sentiment": "NEUTRAL",
            "open_interest_trend": "STABLE"
        }
        try:
            binance_futures = ccxt.binanceusdm({'enableRateLimit': True})
            
            # ১. ফান্ডিং রেট ফেচ
            funding_info = binance_futures.fetch_funding_rate(symbol)
            funding_rate = funding_info.get('fundingRate', 0) * 100
            metrics["funding_rate"] = round(funding_rate, 4)
            
            if funding_rate > 0.03:
                metrics["funding_sentiment"] = "OVERHEATED_BULLISH"
            elif funding_rate < -0.01:
                metrics["funding_sentiment"] = "EXTREME_BEARISH"
            else:
                metrics["funding_sentiment"] = "HEALTHY"
                
            # ২. ওপেন ইন্টারেস্ট (Open Interest) ফেচ
            oi_info = binance_futures.fetch_open_interest(symbol)
            metrics["open_interest"] = oi_info.get('openInterestAmount', 0)
            
        except Exception as e:
            print(f"⚠️ Error fetching Derivatives Metrics ({symbol}): {e}")
            
        return metrics

    @staticmethod
    def get_btc_dominance():
        """CoinGecko পাবলিক এপিআই থেকে BTC Dominance ফেচ করে।"""
        try:
            url = "https://api.coingecko.com/api/v3/global"
            response = requests.get(url, timeout=5)
            data = response.json()
            if "data" in data and "market_cap_percentage" in data["data"]:
                btc_dom = data["data"]["market_cap_percentage"].get("btc", 0)
                return round(btc_dom, 2)
        except Exception as e:
            print(f"⚠️ Error fetching BTC Dominance: {e}")
            
        return 50.0

    @classmethod
    def build_full_market_context(cls, btc_1h_df=None):
        """পুরো মার্কেটের স্ন্যাপশট তৈরি করে যা AI ও Risk Engine ব্যবহার করবে।"""
        fng = cls.get_fear_and_greed_index()
        derivatives = cls.get_derivatives_metrics("BTC/USDT")
        btc_dom = cls.get_btc_dominance()
        
        # বিটিসি ক্যান্ডেল দিয়ে সিম্পল ম্যাক্রো ট্রেন্ড বের করা
        btc_trend = "NEUTRAL"
        if btc_1h_df is not None and not btc_1h_df.empty:
            last = btc_1h_df.iloc[-1]
            # আপনার indicators.py অনুযায়ী সঠিক কলামের নাম ব্যবহার করা হয়েছে (EMA_50, RSI)
            if 'EMA_50' in last and 'RSI' in last:
                if last['close'] > last['EMA_50'] and last['RSI'] > 50:
                    btc_trend = "BULLISH"
                elif last['close'] < last['EMA_50'] and last['RSI'] < 50:
                    btc_trend = "BEARISH"

        context_json = {
            "btc_macro_trend": btc_trend,
            "btc_dominance": f"{btc_dom}%",
            "fear_and_greed": f"{fng['value']} ({fng['sentiment']})",
            "funding_rate": f"{derivatives['funding_rate']}%",
            "funding_sentiment": derivatives["funding_sentiment"],
            "news_sentiment": "NEUTRAL"
        }
        
        return context_json