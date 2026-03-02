"""
Task 5 - AI Trading Agent for SecureFinAI Contest
Agentic Trading with LLM + Technical Indicators + NOFX Data + Sentiment + ML Predictions

Author: Alexei (alxy)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import requests


class MLClient:
    """Client for ML prediction service"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict(self, symbol, model="GRU", market_type="crypto"):
        """Get ML prediction for symbol"""
        try:
            # Convert symbol format
            if market_type == "crypto":
                symbol_query = f"{symbol}/USD"
            else:
                symbol_query = symbol
            
            url = f"{self.base_url}/api/predict"
            params = {
                "company": symbol_query,
                "model_type": model,
                "market_type": market_type
            }
            
            resp = requests.post(url, params=params, timeout=60)
            data = resp.json()
            
            summary = data.get("prediction_summary", {})
            confidence = summary.get("confidence", 0)
            change = summary.get("next_change", "0%")
            
            # Parse change to float
            change_str = str(change).replace("+", "").replace("−", "-").replace("%", "").strip()
            try:
                change_val = float(change_str)
            except:
                change_val = 0
            
            return {
                "change": change_val,
                "confidence": confidence,
                "prediction": data
            }
            
        except Exception as e:
            print(f"   ⚠️ ML error: {e}")
            return None


class AlpacaTrader:
    """Client for Alpaca Paper Trading"""
    
    def __init__(self, api_key="PKS7VQVXN4KGDCOEO656XK7QYY", secret_key="HJW9fPTNcjDavkBAWiBXETp5ouMTd16S3vqWhydieZ36"):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            
            self.trading_client = TradingClient(api_key, secret_key, paper=True)
            self.OrderSide = OrderSide
            self.TimeInForce = TimeInForce
            self.connected = True
            print("✅ Alpaca Trading connected (paper)")
            
            # Get account info
            account = self.trading_client.get_account()
            print(f"   Account: ${float(account.cash):.2f}")
            
        except Exception as e:
            print(f"❌ Alpaca connection failed: {e}")
            self.connected = False
    
    def place_order(self, symbol, side, qty=1):
        """Place a market order"""
        if not self.connected:
            return None
        
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            
            # For crypto, symbol format is like BTC/USD
            crypto_symbol = f"{symbol}/USD"
            
            order = MarketOrderRequest(
                symbol=crypto_symbol,
                qty=qty,
                side=OrderSide.BUY if side == "Buy" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            
            result = self.trading_client.submit_order(order)
            print(f"   ✅ {side} {qty} {symbol}")
            return result
            
        except Exception as e:
            print(f"   ❌ Order failed: {e}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        if not self.connected:
            return []
        
        try:
            positions = self.trading_client.get_all_positions()
            return [p.symbol for p in positions]
        except:
            return []
    
    def close_all(self):
        """Close all positions (for end-of-day risk management)"""
        if not self.connected:
            return
        
        try:
            positions = self.trading_client.get_all_positions()
            closed = 0
            
            for p in positions:
                try:
                    # Use close_position API (handles fractional qty)
                    self.trading_client.close_position(p.symbol)
                    closed += 1
                    print(f"   🔴 Closed: {p.symbol}")
                except Exception as e:
                    print(f"   ⚠️ {p.symbol}: {e}")
            
            if closed > 0:
                print(f"   ✅ Closed {closed} positions (EOD)")
            return closed
            
        except Exception as e:
            print(f"   ⚠️ Close all error: {e}")
            return 0
    
    def manage_positions(self):
        """Manage open positions: take profit, break-even, stop loss"""
        if not self.connected:
            return []
        
        TP = 0.02   # +2% take profit
        SL = -0.02  # -2% stop loss
        
        try:
            positions = self.trading_client.get_all_positions()
            actions = []
            
            for p in positions:
                entry_price = float(p.avg_entry_price)
                current_price = float(p.current_price)
                pnl_pct = (current_price - entry_price) / entry_price
                
                symbol = p.symbol.replace("USD", "")
                
                # Decision
                if pnl_pct >= TP:
                    action = "TAKE_PROFIT"
                    reason = f"+{pnl_pct:.1%}"
                elif pnl_pct <= SL:
                    action = "STOP_LOSS"
                    reason = f"{pnl_pct:.1%}"
                elif abs(pnl_pct) < 0.005:  # Within 0.5% of entry = break-even
                    action = "BREAK_EVEN"
                    reason = f"{pnl_pct:+.1%}"
                else:
                    action = "HOLD"
                    reason = f"{pnl_pct:+.1%}"
                
                # Execute close if needed
                if action in ["TAKE_PROFIT", "STOP_LOSS", "BREAK_EVEN"]:
                    try:
                        self.trading_client.close_position(p.symbol)
                        print(f"   📊 {symbol}: {action} {reason}")
                        actions.append(action)
                    except Exception as e:
                        print(f"   ⚠️ {symbol}: {e}")
                else:
                    print(f"   ⏸️ {symbol}: {action} {reason}")
                    
            return actions
            
        except Exception as e:
            print(f"   ⚠️ Manage error: {e}")
            return []


class NewsClient:
    """Client for crypto news sentiment"""
    
    def __init__(self):
        self.api_url = "https://free-crypto-news-api.cloudpub.ru/api/analyze"
        self.cache = {}
    
    def get_sentiment(self):
        """Fetch news sentiment"""
        import urllib.request
        import ssl
        
        # Check cache (5 min TTL)
        cache_key = "news_sentiment"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < 300:
                return cached_data
        
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(self.api_url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            
            with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            # Count sentiments from articles
            articles = data.get('articles', [])
            bullish = sum(1 for a in articles if a.get('sentiment') == 'bullish')
            bearish = sum(1 for a in articles if a.get('sentiment') == 'bearish')
            neutral = sum(1 for a in articles if a.get('sentiment') == 'neutral')
            
            # Calculate score: (bullish - bearish) / total
            total = bullish + bearish + neutral
            score = (bullish - bearish) / total if total > 0 else 0
            
            result = {
                'bullish': bullish,
                'bearish': bearish,
                'neutral': neutral,
                'score': score,
                'articles': articles[:5]  # Top 5 articles
            }
            
            # Cache result
            self.cache[cache_key] = (datetime.now(), result)
            
            return result
            
        except Exception as e:
            print(f"News API error: {e}")
            return {'bullish': 0, 'bearish': 0, 'neutral': 0, 'score': 0, 'articles': []}


class NOFXClient:
    """Client for NOFX API data"""
    
    def __init__(self, api_key="cm_568c67eae410d912c54c"):
        self.api_key = api_key
        self.base_url = "https://nofxos.ai/api"
        self.cache = {}
    
    def get_coin_data(self, symbol):
        """Get comprehensive coin data from NOFX"""
        cache_key = f"coin_{symbol}"
        
        # Check cache (5 min TTL)
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < 300:
                return cached_data
        
        # Fetch from API
        try:
            import urllib.request
            import ssl
            
            url = f"{self.base_url}/coin/{symbol}USDT?auth={self.api_key}"
            
            # Create SSL context that doesn't verify (for compatibility)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            coin_data = data.get('data', {})
            
            # Extract key metrics
            result = {
                'symbol': symbol,
                'price': coin_data.get('price', 0),
                'price_change_24h': coin_data.get('price_change_24h', 0) * 100,
                'ai_score': coin_data.get('ai_score'),
                
                # OI from Binance
                'oi_binance': coin_data.get('oi', {}).get('binance', {}),
            }
            
            # Calculate derived metrics
            oi_data = result['oi_binance']
            result['oi_delta_1h'] = oi_data.get('delta', {}).get('1h', {}).get('oi_delta_percent', 0) * 100
            result['oi_delta_24h'] = oi_data.get('delta', {}).get('24h', {}).get('oi_delta_percent', 0) * 100
            
            # Net position (positive = net long, negative = net short)
            result['net_long'] = oi_data.get('net_long', 0)
            result['net_short'] = oi_data.get('net_short', 0)
            result['net_position'] = result['net_long'] + result['net_short']
            
            # Long-short ratio
            if result['net_short'] != 0:
                result['long_short_ratio'] = abs(result['net_long'] / result['net_short'])
            else:
                result['long_short_ratio'] = 1.0
            
            # Cache result
            self.cache[cache_key] = (datetime.now(), result)
            
            return result
            
        except Exception as e:
            print(f"NOFX API error for {symbol}: {e}")
            return None
    
    def get_fund_flow(self, coin, duration="1h"):
        """Get fund flow for a coin"""
        try:
            import urllib.request
            import ssl
            
            url = f"{self.base_url}/netflow/low-ranking?type=institution&trade=future&duration={duration}&auth={self.api_key}"
            
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            # Find the coin
            for item in data.get('data', {}).get('netflows', []):
                if coin in item['symbol']:
                    return item['amount'] / 1e6  # Convert to millions
            
            return 0
            
        except Exception as e:
            return 0
    
    def get_funding_rate(self, symbol):
        """Get funding rate for a coin"""
        try:
            import urllib.request
            import ssl
            
            url = f"{self.base_url}/funding-rate?symbol={symbol}USDT&auth={self.api_key}"
            
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            return data.get('data', {}).get('funding_rate', 0)
            
        except Exception as e:
            return 0
    
    def get_oi_cap_rank(self, symbol):
        """Get OI market cap rank for a coin"""
        try:
            import urllib.request
            import ssl
            
            url = f"{self.base_url}/oi-cap/ranking?auth={self.api_key}&limit=30"
            
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                data = json.loads(response.read().decode())
            
            # Find the coin
            for p in data.get('data', {}).get('rankings', []):
                if p.get('symbol') == symbol:
                    return {
                        'rank': p.get('rank', 0),
                        'oi_value': p.get('oi_value', 0),
                        'net_long': p.get('net_long', 0)
                    }
            
            return {'rank': 0, 'oi_value': 0, 'net_long': 0}
            
        except Exception as e:
            return {'rank': 0, 'oi_value': 0, 'net_long': 0}


class SecureFinAIAgent:
    """
    AI Trading Agent for Task 5: Agentic Trading
    
    Uses:
    - NOFX data (fund flow, OI, net position)
    - Technical indicators (RSI, MACD, EMA)
    - Sentiment analysis
    - LLM for final decision
    """
    
    # 20 Alpaca coins to track
    ALPACA_COINS = [
        "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "UNI",
        "LTC", "BCH", "AAVE", "YFI", "CRV", "XTZ", "GRT", "BAT", "SUSHI", "SHIB"
    ]
    
    def __init__(self, use_llm=False, use_nofx=True, execute_trades=True):
        self.use_llm = use_llm
        self.use_nofx = use_nofx
        self.position = "Hold"
        
        # Initialize NOFX client
        if use_nofx:
            self.nofx = NOFXClient()
            print("✅ NOFX client initialized")
        else:
            self.nofx = None
        
        # Initialize Alpaca Trading
        if execute_trades:
            self.alpaca = AlpacaTrader()
        else:
            self.alpaca = None
        
        # Initialize ML Predictions
        self.ml = MLClient()
        print("✅ ML client initialized")
        
        # Initialize News client
        self.news = NewsClient()
        print("✅ News client initialized")
        
        # Load LLM if enabled
        if use_llm:
            try:
                from openai import OpenAI
                self.llm = OpenAI(
                    api_key="your-key",
                    base_url="https://api.minimax.chat/v1"
                )
                print("✅ LLM initialized")
            except Exception as e:
                print(f"⚠️ LLM not available: {e}")
                self.use_llm = False
    
    def get_nofx_score(self, symbol):
        """Get NOFX score for a symbol (-1 to 1)"""
        if not self.nofx:
            return 0, {}
        
        data = self.nofx.get_coin_data(symbol)
        if not data:
            return 0, {}
        
        score = 0
        
        # 1. Fund flow (institution) - weight 3
        flow = self.nofx.get_fund_flow(symbol)
        if flow < -50:  # >$50M outflow
            score -= 3
        elif flow < -10:  # >$10M outflow
            score -= 2
        elif flow < -1:  # >$1M outflow
            score -= 1
        elif flow > 1:
            score += 1
        
        # 2. OI change - weight 2
        oi_delta = data.get('oi_delta_1h', 0)
        if oi_delta < -1:
            score -= 2
        elif oi_delta < -0.5:
            score -= 1
        elif oi_delta > 1:
            score += 2
        elif oi_delta > 0.5:
            score += 1
        
        # 3. Net position - weight 2
        net_pos = data.get('net_position', 0)
        if net_pos < -1_000_000:
            score -= 2
        elif net_pos < -100_000:
            score -= 1
        elif net_pos > 1_000_000:
            score += 2
        elif net_pos > 100_000:
            score += 1
        
        # 4. AI Score - weight 1
        ai_score = data.get('ai_score')
        if ai_score:
            if ai_score > 80:
                score += 1
            elif ai_score < 30:
                score -= 1
        
        # 5. Price change - weight 1
        price_change = data.get('price_change_24h', 0)
        if price_change > 5:
            score += 1
        elif price_change < -5:
            score -= 1
        
        # 6. Funding rate - weight 1 (negative = bears paying, bullish signal)
        funding_rate = self.nofx.get_funding_rate(symbol)
        if funding_rate < -0.001:  # Very negative - long pays short
            score += 1
        elif funding_rate > 0.001:  # Very positive - short pays long
            score -= 1
        
        # 7. Long-short ratio - weight 1
        ls_ratio = data.get('long_short_ratio', 1)
        if ls_ratio > 1.3:  # Crowded long
            score -= 1  # Potential reversal
        elif ls_ratio < 0.7:  # Crowded short
            score += 1  # Potential reversal
        
        # 8. OI market cap rank - weight 1 (top coins = more liquid, safer)
        oi_cap = self.nofx.get_oi_cap_rank(symbol)
        oi_rank = oi_cap.get('rank', 0)
        if oi_rank > 0 and oi_rank <= 10:
            score += 1  # Top 10 coins - higher liquidity
        elif oi_rank > 20:
            score -= 1  # Lower liquidity coins
        
        # Normalize to -1 to 1 (max possible score is 13)
        normalized_score = max(-1, min(1, score / 7))
        
        # Return score + all metrics for display
        metrics = {
            'fund_flow': flow,
            'oi_delta_1h': oi_delta,
            'net_position': net_pos,
            'price_change_24h': price_change,
            'funding_rate': funding_rate,
            'long_short_ratio': ls_ratio,
            'ai_score': ai_score,
            'oi_cap_rank': oi_rank
        }
        
        return normalized_score, metrics
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        if df is None or len(df) < 20:
            return {}
        
        close = df['Close'].values
        volume = df['Volume'].values if 'Volume' in df.columns else None
        
        # RSI
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(rs) > 0 else 50
        
        # EMA
        ema20 = pd.Series(close).ewm(span=20).mean().iloc[-1]
        ema50 = pd.Series(close).ewm(span=50).mean().iloc[-1]
        
        # MACD
        close_series = pd.Series(close)
        ema12 = close_series.ewm(span=12).mean()
        ema26 = close_series.ewm(span=26).mean()
        macd_line = ema12 - ema26
        macd = macd_line.iloc[-1]
        signal = macd_line.ewm(span=9).mean().iloc[-1]
        
        # Price momentum
        price_change = (close[-1] - close[0]) / close[0] * 100 if len(close) > 1 else 0
        
        return {
            "rsi": rsi,
            "ema20": ema20,
            "ema50": ema50,
            "macd": macd,
            "macd_signal": signal,
            "macd_histogram": macd - signal,
            "price_change_5d": price_change,
            "close": close[-1]
        }
    
    def generate_signal(self, indicators, nofx_score=0, sentiment_score=0, nofx_data=None, nofx_metrics=None, ml_prediction=None):
        """Generate trading signal requiring CONSENSUS from all sources"""
        
        if not indicators:
            return "Hold", "Insufficient data"
        
        rsi = indicators.get("rsi", 50)
        macd_hist = indicators.get("macd_histogram", 0)
        
        # === 1. TECHNICAL ANALYSIS ===
        ta_buy = 0
        ta_sell = 0
        
        if rsi < 35:
            ta_buy += 1
        elif rsi > 65:
            ta_sell += 1
        
        if macd_hist > 0:
            ta_buy += 1
        elif macd_hist < 0:
            ta_sell += 1
        
        if indicators.get("ema20", 0) > indicators.get("ema50", 0):
            ta_buy += 1
        else:
            ta_sell += 1
        
        # === 2. NOFX SCORE ===
        nofx_buy = 1 if nofx_score > 0.1 else 0
        nofx_sell = 1 if nofx_score < -0.1 else 0
        
        # === Enhanced NOFX: check L/S ratio and Flow ===
        if nofx_data and nofx_metrics:
            ls_ratio = nofx_data.get("long_short_ratio", 1.0)
            flow = nofx_metrics.get("fund_flow", 0)
            
            # Strong L/S signal (1.3 = 30% more long than short)
            if ls_ratio > 1.3:
                nofx_buy = 1
                nofx_sell = 0
            elif ls_ratio < 0.7:
                nofx_buy = 0
                nofx_sell = 1
            
            # Strong flow override (+/- $10M)
            if flow > 10_000_000:
                nofx_buy = 1
            elif flow < -10_000_000:
                nofx_sell = 1
        
        # === 3. SENTIMENT ===
        sent_buy = 1 if sentiment_score > 0.1 else 0
        sent_sell = 1 if sentiment_score < -0.1 else 0
        
        # === 4. ML PREDICTION ===
        ml_buy = 0
        ml_sell = 0
        if ml_prediction:
            ml_change = ml_prediction.get("change", 0)
            ml_conf = ml_prediction.get("confidence", 0)
            
            # Only trust ML if confidence > 70%
            if ml_conf > 0.7:
                if ml_change > 1:  # >1% predicted increase
                    ml_buy = 1
                elif ml_change < -1:  # <-1% predicted decrease
                    ml_sell = 1
        
        # === CONSENSUS: All 4 must agree ===
        # With ML: require 3/4 sources agree
        if ml_prediction:
            if ta_buy >= 2 and nofx_buy and sent_buy and ml_buy:
                return "Buy", f"CONSENSUS BUY: TA={ta_buy}/3, NOFX, Sent, ML"
            elif ta_sell >= 2 and nofx_sell and sent_sell and ml_sell:
                return "Sell", f"CONSENSUS SELL: TA={ta_sell}/3, NOFX, Sent, ML"
        else:
            # Without ML: original 3-source consensus
            if ta_buy >= 2 and nofx_buy and sent_buy:
                return "Buy", f"CONSENSUS BUY: TA={ta_buy}/3, NOFX={nofx_score:.2f}, Sent={sentiment_score:.2f}"
            elif ta_sell >= 2 and nofx_sell and sent_sell:
                return "Sell", f"CONSENSUS SELL: TA={ta_sell}/3, NOFX={nofx_score:.2f}, Sent={sentiment_score:.2f}"
        
        return "Hold", f"RSI={rsi:.0f}, NOFX={nofx_score:.2f}, Sent={sentiment_score:.2f}"
    
    def decide(self, market_data, symbol="BTC", news=None, sentiment_score=None):
        """
        Main interface for the contest.
        
        Args:
            market_data: DataFrame with OHLCV
            symbol: Trading symbol (e.g., "BTC", "ETH")
            news: Optional list of news items
            sentiment_score: Optional sentiment (-1 to 1). If None, fetches from news API.
            
        Returns:
            dict: {"action": "Buy/Hold/Sell", "rationale": "..."}
        """
        # Get news sentiment if not provided
        if sentiment_score is None:
            try:
                news_data = self.news.get_sentiment()
                sentiment_score = news_data.get('score', 0)
                news = news_data.get('articles', [])
                print(f"   News: score={sentiment_score:.2f}, B={news_data.get('bullish', 0)}, N={news_data.get('neutral', 0)}, Be={news_data.get('bearish', 0)}")
            except Exception as e:
                print(f"   News: error - {e}")
                sentiment_score = 0
        else:
            sentiment_score = sentiment_score or 0
        
        # Get NOFX score and metrics
        nofx_score = 0
        nofx_data = None
        nofx_metrics = {}
        if self.nofx and symbol in self.ALPACA_COINS:
            try:
                nofx_score, nofx_metrics = self.get_nofx_score(symbol)
                nofx_data = self.nofx.get_coin_data(symbol)
                print(f"   NOFX: score={nofx_score:.2f}, Flow={nofx_metrics.get('fund_flow', 0):.1f}M, OI={nofx_metrics.get('oi_delta_1h', 0):.2f}%, L/S={nofx_metrics.get('long_short_ratio', 1):.2f}")
            except Exception as e:
                print(f"   NOFX: error - {e}")
        
        # Calculate indicators
        indicators = self.calculate_indicators(market_data)
        
        # Get ML prediction
        ml_prediction = None
        try:
            # Determine market type
            market_type = "crypto" if symbol in self.ALPACA_COINS else "stock"
            ml_prediction = self.ml.predict(symbol, model="GRU", market_type=market_type)
            if ml_prediction:
                print(f"   🤖 ML: {ml_prediction.get('change', 0):+.1f}% (conf: {ml_prediction.get('confidence', 0)*100:.0f}%)")
        except Exception as e:
            print(f"   🤖 ML: unavailable - {e}")
        
        # Generate signal
        action, reason = self.generate_signal(indicators, nofx_score, sentiment_score, nofx_data, nofx_metrics, ml_prediction)
        
        # Use LLM for refined decision if available
        if self.use_llm and news:
            action = self.llm_decision(action, reason, indicators, news)
        
        # Format rationale (max 50 words)
        rationale = self.format_rationale(action, indicators, nofx_score, sentiment_score)
        
        self.position = action
        
        # Execute order if Alpaca is connected
        order_result = None
        MAX_POSITIONS = 5  # Max open positions
        
        if self.alpaca and self.alpaca.connected and action in ["Buy", "Sell"]:
            # Check current position count
            if action == "Buy":
                positions = self.alpaca.get_positions()
                if len(positions) >= MAX_POSITIONS:
                    print(f"   ⏭️ Max positions reached ({MAX_POSITIONS}) - SKIP {symbol}")
                    return {
                        "action": "Hold",
                        "rationale": f"Max positions ({MAX_POSITIONS}) reached",
                        "order_executed": False
                    }
            # Check current positions
            positions = self.alpaca.get_positions()
            crypto_symbol = f"{symbol}/USD"
            
            # For SELL: only sell if we have the position
            if action == "Sell":
                if crypto_symbol in positions:
                    # Close entire position
                    try:
                        pos = self.alpaca.trading_client.get_position(crypto_symbol)
                        qty = abs(float(pos.qty))
                        print(f"   📝 Closing FULL position: {symbol} ({qty} units)")
                        order_result = self.alpaca.place_order(symbol, "Sell", qty=int(qty))
                    except Exception as e:
                        print(f"   ⚠️ Could not close: {e}")
                else:
                    print(f"   ⏭️ No position to sell for {symbol}")
                action = "Hold"  # Mark as handled
            
            # For BUY: position sizing based on risk management
            elif action == "Buy":
                price = indicators.get("close", 0)
                qty = 0  # Initialize
                
                # Risk management: adaptive position sizing
                # BTC: max 1 coin, ETH: 1-2 coins, Mid: 2%, Cheap: 5%
                if price > 10000:  # BTC class
                    # Only buy 1 if we can afford it
                    qty = 1
                    print(f"   📝 Buy {symbol} qty={qty} (~${price:.0f}, BTC tier)")
                elif price > 500:  # ETH class
                    PORTFOLIO_PCT = 0.02  # 2%
                elif price > 100:
                    PORTFOLIO_PCT = 0.02  # 2%
                else:
                    PORTFOLIO_PCT = 0.05  # 5%
                
                MIN_ORDER = 10  # Minimum $10 order
                
                if price > 0 and qty == 0:  # Calculate qty unless BTC tier
                    # Get portfolio value
                    portfolio_value = 100000  # Default
                    if self.alpaca and self.alpaca.connected:
                        try:
                            account = self.alpaca.trading_client.get_account()
                            portfolio_value = float(account.portfolio_value)
                        except:
                            pass
                    
                    # Calculate position size
                    trade_value = portfolio_value * PORTFOLIO_PCT
                    trade_value = max(trade_value, MIN_ORDER)  # At least $10
                    
                    qty = int(trade_value / price)  # Floor to integer
                    
                    if qty < 1:
                        print(f"   ⏭️ {symbol} price ${price:.0f} too high: ${trade_value:.0f} / ${price:.0f} = 0 qty")
                    else:
                        actual_value = qty * price
                        if actual_value >= MIN_ORDER:
                            print(f"   📝 Buy {symbol} qty={qty} (~${actual_value:.0f}, {PORTFOLIO_PCT*100:.0f}%)")
                            order_result = self.alpaca.place_order(symbol, "Buy", qty=qty)
                        else:
                            print(f"   ⏭️ {symbol} order ${actual_value:.2f} < ${MIN_ORDER} min")
                elif price > 0 and qty > 0:
                    # BTC tier - execute
                    if price >= MIN_ORDER:
                        order_result = self.alpaca.place_order(symbol, "Buy", qty=qty)
                    else:
                        print(f"   ⏭️ {symbol} price ${price:.2f} < ${MIN_ORDER} min")
                else:
                    print(f"   ⚠️ No price data for {symbol}")
        
        return {
            "action": action,
            "rationale": rationale,
            "nofx_score": nofx_score,
            "nofx_data": nofx_data,
            "nofx_metrics": nofx_metrics,
            "sentiment_score": sentiment_score,
            "order_executed": order_result is not None
        }
    
    def llm_decision(self, base_action, reason, indicators, news):
        """Use LLM to refine decision"""
        try:
            prompt = f"""
You are a trading expert. Based on:
- Technical: RSI={indicators.get('rsi', 50):.0f}, MACD={indicators.get('macd_histogram', 0):.2f}
- NOFX sentiment: {reason}
- News: {news[:3] if news else 'None'}

Decide: Buy, Hold, or Sell (only one word).
"""
            response = self.llm.chat.completions.create(
                model="MiniMax-M2.5",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            llm_action = response.choices[0].message.content.strip()
            if llm_action in ["Buy", "Hold", "Sell"]:
                return llm_action
        except Exception as e:
            print(f"LLM error: {e}")
        
        return base_action
    
    def format_rationale(self, action, indicators, nofx_score, sentiment):
        """Format rationale (max 50 words)"""
        rsi = indicators.get("rsi", 50)
        macd = "bullish" if indicators.get("macd_histogram", 0) > 0 else "bearish"
        
        # NOFX sentiment
        if nofx_score > 0.3:
            nofx_sent = "NOFX bullish"
        elif nofx_score < -0.3:
            nofx_sent = "NOFX bearish"
        else:
            nofx_sent = "NOFX neutral"
        
        # Sentiment
        if sentiment > 0.3:
            sent = "bullish"
        elif sentiment < -0.3:
            sent = "bearish"
        else:
            sent = "neutral"
        
        parts = [
            f"RSI {rsi:.0f}",
            f"MACD {macd}",
            nofx_sent,
            f"sentiment {sent}"
        ]
        
        rationale = f"{action} - " + ", ".join(parts)
        
        # Trim to 50 words
        words = rationale.split()
        if len(words) > 50:
            rationale = " ".join(words[:50])
        
        return rationale


if __name__ == "__main__":
    # Test with NOFX
    agent = SecureFinAIAgent(use_llm=False, use_nofx=True)
    
    # Mock market data
    import pandas as pd
    import numpy as np
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    mock_data = pd.DataFrame({
        'Open': np.random.uniform(68000, 70000, 30),
        'High': np.random.uniform(69000, 71000, 30),
        'Low': np.random.uniform(67000, 69000, 30),
        'Close': np.random.uniform(68000, 70000, 30),
        'Volume': np.random.uniform(1000, 5000, 30)
    }, index=dates)
    
    print("\n=== BTC Decision ===")
    result = agent.decide(mock_data, symbol="BTC", sentiment_score=0)
    print(f"Action: {result['action']}")
    print(f"Rationale: {result['rationale']}")
    print(f"NOFX Score: {result['nofx_score']}")
    
    print("\n=== ETH Decision ===")
    result = agent.decide(mock_data, symbol="ETH", sentiment_score=0)
    print(f"Action: {result['action']}")
    print(f"Rationale: {result['rationale']}")
    print(f"NOFX Score: {result['nofx_score']}")
