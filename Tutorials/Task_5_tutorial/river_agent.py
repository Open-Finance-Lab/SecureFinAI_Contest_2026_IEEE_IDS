"""
Task 5 - AI Trading Agent with River ML
Agentic Trading with Online Learning + Technical Indicators + Sentiment

Uses River for online/streaming machine learning with concept drift detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from river import linear_model, preprocessing, metrics, drift, optim, losses
from collections import deque


class RiverMLPredictor:
    """
    Online ML predictor using River for streaming price prediction.
    
    Features:
    - Online learning (no retraining needed)
    - Concept drift detection
    - Rolling metrics tracking
    """
    
    def __init__(self, learning_rate=0.01, window_size=100):
        self.window_size = window_size
        
        # Feature pipeline: StandardScaler + Linear Regression with SGD
        self.model = preprocessing.StandardScaler() | linear_model.LinearRegression(
            optimizer=optim.SGD(lr=learning_rate),
            loss=losses.Squared()
        )
        
        # Classification model for direction prediction (up/down)
        self.classifier = preprocessing.StandardScaler() | linear_model.LogisticRegression(
            optimizer=optim.SGD(lr=learning_rate)
        )
        
        # Drift detector (ADWIN - Adaptive Windowing)
        self.drift_detector = drift.ADWIN()
        
        # Metrics (online updates)
        self.regression_metric = metrics.RMSE()
        self.classification_metric = metrics.Accuracy()
        
        # Data buffer for feature computation
        self.data_buffer = deque(maxlen=window_size)
        
        # Training counter
        self.n_samples = 0
        self.drift_detected = False
        
    def compute_features(self, df):
        """
        Compute features from market data.
        
        Features:
        - Price returns (1d, 3d, 5d, 10d)
        - Volatility (rolling std)
        - Volume change
        - Price momentum
        - RSI
        - MACD
        """
        if df is None or len(df) < 30:
            return None
        
        close = df['Close'].values
        volume = df['Volume'].values if 'Volume' in df.columns else np.ones(len(close))
        
        # Price returns
        returns_1d = (close[-1] - close[-2]) / close[-2] if len(close) >= 2 else 0
        returns_3d = (close[-1] - close[-4]) / close[-4] if len(close) >= 4 else 0
        returns_5d = (close[-1] - close[-6]) / close[-6] if len(close) >= 6 else 0
        returns_10d = (close[-1] - close[-11]) / close[-11] if len(close) >= 11 else 0
        
        # Volatility (5-day rolling std)
        volatility = np.std(close[-5:]) if len(close) >= 5 else 0
        
        # Volume change
        volume_change = (volume[-1] - volume[-2]) / volume[-2] if len(volume) >= 2 and volume[-2] > 0 else 0
        
        # Momentum
        momentum = (close[-1] - close[0]) / close[0] if len(close) > 1 else 0
        
        # RSI (14-day)
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(rs) > 0 and len(rs.dropna()) > 0 else 50
        rsi_normalized = (rsi - 50) / 50  # Normalize to ~[-1, 1]
        
        # MACD
        close_series = pd.Series(close)
        ema12 = close_series.ewm(span=12).mean()
        ema26 = close_series.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        macd_histogram = (macd_line.iloc[-1] - signal_line.iloc[-1]) / close[-1] if len(macd_line) > 0 else 0
        
        # Price position (current vs min/max in window)
        price_position = (close[-1] - min(close)) / (max(close) - min(close)) if max(close) != min(close) else 0.5
        
        return {
            'returns_1d': returns_1d,
            'returns_3d': returns_3d,
            'returns_5d': returns_5d,
            'returns_10d': returns_10d,
            'volatility': volatility,
            'volume_change': volume_change,
            'momentum': momentum,
            'rsi': rsi_normalized,
            'macd': macd_histogram,
            'price_position': price_position
        }
    
    def predict(self, df):
        """
        Predict next price change and direction.
        
        Returns:
            dict: {
                'price_change': predicted % change,
                'direction': 'up'/'down',
                'confidence': 0-1,
                'regression_error': current RMSE,
                'classification_accuracy': current accuracy
            }
        """
        features = self.compute_features(df)
        
        if features is None or self.n_samples < 10:
            # Not enough data for prediction
            return {
                'price_change': 0,
                'direction': 'hold',
                'confidence': 0,
                'regression_error': 0,
                'classification_accuracy': 0,
                'reason': 'insufficient_data'
            }
        
        # Regression prediction (price change %)
        price_change_pred = self.model.predict_one(features)
        
        # Classification prediction (direction)
        direction_pred = self.classifier.predict_one(features)
        direction_proba = self.classifier.predict_proba_one(features)
        
        # Confidence from probability
        confidence = max(direction_proba.get(True, 0.5), direction_proba.get(False, 0.5))
        
        return {
            'price_change': price_change_pred * 100,  # Convert to %
            'direction': 'up' if direction_pred else 'down',
            'confidence': confidence,
            'regression_error': self.regression_metric.get(),
            'classification_accuracy': self.classification_metric.get(),
            'reason': 'prediction'
        }
    
    def learn(self, df, actual_return=None):
        """
        Update model with new data point (online learning).
        
        Args:
            df: Market data DataFrame
            actual_return: Actual return (if known, for supervised learning)
        """
        features = self.compute_features(df)
        
        if features is None:
            return
        
        # Compute target: next day return (if we have enough data)
        if actual_return is None and len(self.data_buffer) > 0:
            # Use stored actual return
            actual_return = self.data_buffer[-1].get('actual_return', 0)
        
        if actual_return is not None:
            # Update regression model
            self.model.learn_one(features, actual_return)
            
            # Update classification model (direction)
            direction = actual_return > 0
            self.classifier.learn_one(features, direction)
            
            # Update metrics
            pred = self.model.predict_one(features)
            self.regression_metric.update(actual_return, pred)
            
            class_pred = self.classifier.predict_one(features)
            self.classification_metric.update(direction, class_pred)
            
            # Drift detection on residuals
            residual = abs(actual_return - pred)
            drift_result = self.drift_detector.update(residual)
            
            if drift_result:
                self.drift_detected = True
                print(f"   ⚠️ Concept drift detected! (RMSE: {self.regression_metric.get():.4f})")
                # Optional: Reset model or adjust learning rate
                # self.model = preprocessing.StandardScaler() | linear_model.LinearRegression(...)
        
        # Store data point
        close = df['Close'].values
        if len(self.data_buffer) > 0:
            prev_close = self.data_buffer[-1]['close']
            current_return = (close[-1] - prev_close) / prev_close if prev_close > 0 else 0
        else:
            current_return = 0
        
        self.data_buffer.append({
            'features': features,
            'close': close[-1],
            'actual_return': current_return
        })
        
        self.n_samples += 1
    
    def get_status(self):
        """Get model status and metrics."""
        return {
            'n_samples': self.n_samples,
            'rmse': self.regression_metric.get(),
            'accuracy': self.classification_metric.get(),
            'drift_detected': self.drift_detected,
            'buffer_size': len(self.data_buffer)
        }


class RiverTradingAgent:
    """
    Trading Agent with River ML for Task 5: Agentic Trading
    
    Combines:
    - River online ML for price prediction
    - Technical indicators
    - Sentiment analysis
    - Risk management
    """
    
    def __init__(self, use_river=True, learning_rate=0.01):
        self.use_river = use_river
        
        # Initialize River ML predictor
        if use_river:
            self.ml_predictor = RiverMLPredictor(learning_rate=learning_rate)
            print("✅ River ML initialized")
        else:
            self.ml_predictor = None
        
        # Risk parameters
        self.max_position_size = 1.0  # Max 100% of portfolio
        self.stop_loss = -0.02  # -2%
        self.take_profit = 0.03  # +3%
        
        # Current position
        self.position = None  # None, 'long', or 'short'
        self.entry_price = 0
        
    def calculate_indicators(self, df):
        """Calculate technical indicators for decision making."""
        if df is None or len(df) < 20:
            return {}
        
        close = df['Close'].values
        volume = df['Volume'].values if 'Volume' in df.columns else None
        
        # RSI
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(rs) > 0 and len(rs.dropna()) > 0 else 50
        
        # EMA
        ema20 = pd.Series(close).ewm(span=20).mean().iloc[-1]
        ema50 = pd.Series(close).ewm(span=50).mean().iloc[-1]
        
        # MACD
        close_series = pd.Series(close)
        ema12 = close_series.ewm(span=12).mean()
        ema26 = close_series.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line.iloc[-1] - signal.iloc[-1]
        
        # Bollinger Bands
        sma20 = pd.Series(close).rolling(window=20).mean()
        std20 = pd.Series(close).rolling(window=20).std()
        bb_upper = sma20.iloc[-1] + 2 * std20.iloc[-1]
        bb_lower = sma20.iloc[-1] - 2 * std20.iloc[-1]
        bb_position = (close[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        # Volatility
        volatility = std20.iloc[-1] / sma20.iloc[-1] if sma20.iloc[-1] > 0 else 0
        
        return {
            'rsi': rsi,
            'ema20': ema20,
            'ema50': ema50,
            'macd_histogram': macd_histogram,
            'bb_position': bb_position,
            'volatility': volatility,
            'close': close[-1]
        }
    
    def generate_signal(self, indicators, ml_prediction, sentiment_score=0):
        """
        Generate trading signal based on ML + Technical + Sentiment.
        
        Consensus approach:
        - ML prediction (River)
        - Technical analysis
        - Sentiment
        """
        if not indicators:
            return "Hold", "Insufficient data"
        
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_histogram', 0)
        bb_position = indicators.get('bb_position', 0.5)
        
        # === 1. ML Signal (River) ===
        ml_signal = 0
        ml_confidence = 0
        if ml_prediction and ml_prediction.get('reason') == 'prediction':
            ml_confidence = ml_prediction.get('confidence', 0)
            if ml_prediction.get('direction') == 'up' and ml_confidence > 0.6:
                ml_signal = 1
            elif ml_prediction.get('direction') == 'down' and ml_confidence > 0.6:
                ml_signal = -1
        
        # === 2. Technical Signal ===
        ta_signal = 0
        ta_reasons = []
        
        # RSI signals
        if rsi < 35:
            ta_signal += 1
            ta_reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 65:
            ta_signal -= 1
            ta_reasons.append(f"RSI overbought ({rsi:.0f})")
        
        # MACD signals
        if macd_hist > 0:
            ta_signal += 0.5
            ta_reasons.append("MACD bullish")
        else:
            ta_signal -= 0.5
            ta_reasons.append("MACD bearish")
        
        # Bollinger Bands signals
        if bb_position < 0.2:
            ta_signal += 1
            ta_reasons.append("BB lower band")
        elif bb_position > 0.8:
            ta_signal -= 1
            ta_reasons.append("BB upper band")
        
        # === 3. Sentiment Signal ===
        sentiment_signal = 0
        if sentiment_score > 0.2:
            sentiment_signal = 1
        elif sentiment_score < -0.2:
            sentiment_signal = -1
        
        # === Consensus ===
        total_signal = ml_signal * ml_confidence + ta_signal * 0.5 + sentiment_signal * 0.3
        
        # Generate action
        if total_signal > 0.8:
            action = "Buy"
            rationale = f"BUY: ML={ml_prediction.get('direction', 'N/A')}({ml_confidence:.2f}), TA={'+'.join(ta_reasons[:2])}, Sent={sentiment_score:.2f}"
        elif total_signal < -0.8:
            action = "Sell"
            rationale = f"SELL: ML={ml_prediction.get('direction', 'N/A')}({ml_confidence:.2f}), TA={'+'.join(ta_reasons[:2])}, Sent={sentiment_score:.2f}"
        else:
            action = "Hold"
            rationale = f"HOLD: RSI={rsi:.0f}, ML_conf={ml_confidence:.2f}, Sent={sentiment_score:.2f}"
        
        # Truncate rationale to 50 words
        rationale = ' '.join(rationale.split()[:50])
        
        return action, rationale
    
    def decide(self, market_data, symbol="BTC", sentiment_score=0):
        """
        Main interface for the contest.
        
        Args:
            market_data: DataFrame with OHLCV
            symbol: Trading symbol
            sentiment_score: Optional sentiment (-1 to 1)
        
        Returns:
            dict: {"action": "Buy/Hold/Sell", "rationale": "..."}
        """
        # 1. Get ML prediction from River
        ml_prediction = None
        if self.use_river and self.ml_predictor:
            ml_prediction = self.ml_predictor.predict(market_data)
            print(f"   🤖 River ML: {ml_prediction.get('direction', 'N/A')} (conf={ml_prediction.get('confidence', 0):.2f}, change={ml_prediction.get('price_change', 0):+.2f}%)")
        
        # 2. Calculate technical indicators
        indicators = self.calculate_indicators(market_data)
        if indicators:
            print(f"   📊 RSI={indicators.get('rsi', 0):.0f}, MACD={indicators.get('macd_histogram', 0):+.4f}, BB={indicators.get('bb_position', 0):.2f}")
        
        # 3. Generate signal
        action, rationale = self.generate_signal(indicators, ml_prediction, sentiment_score)
        
        # 4. Update model with new data (online learning)
        if self.use_river and self.ml_predictor:
            self.ml_predictor.learn(market_data)
            
            # Log status periodically
            if self.ml_predictor.n_samples % 10 == 0:
                status = self.ml_predictor.get_status()
                print(f"   📈 Model: {status['n_samples']} samples, RMSE={status['rmse']:.4f}, Acc={status['accuracy']:.2%}")
        
        return {
            "action": action,
            "rationale": rationale,
            "ml_prediction": ml_prediction,
            "indicators": indicators,
            "sentiment_score": sentiment_score
        }
    
    def get_river_status(self):
        """Get River ML model status."""
        if self.ml_predictor:
            return self.ml_predictor.get_status()
        return None


if __name__ == "__main__":
    # Demo usage
    import yfinance as yf
    
    print("="*60)
    print("River ML Trading Agent - Demo")
    print("="*60)
    
    # Load sample data
    print("\n📥 Loading ETH data...")
    eth = yf.download("ETH-USD", start="2024-01-01", end="2024-12-31", progress=False)
    print(f"   Loaded {len(eth)} days")
    
    # Initialize agent
    agent = RiverTradingAgent(use_river=True)
    
    # Simulate trading on historical data
    print("\n🚀 Running simulation...")
    
    window = 60  # Need 60 days for all indicators
    decisions = []
    
    for i in range(window, len(eth) - 5):
        # Get data up to day i
        data = eth.iloc[:i+1].copy()
        
        # Make decision
        decision = agent.decide(data, symbol="ETH")
        
        # Check actual return (next day)
        actual_return = (eth.iloc[i+1]['Close'] - eth.iloc[i]['Close']) / eth.iloc[i]['Close']
        
        decisions.append({
            'date': eth.index[i],
            'action': decision['action'],
            'actual_return': actual_return,
            'ml_confidence': decision['ml_prediction'].get('confidence', 0) if decision['ml_prediction'] else 0
        })
        
        # Print every 50 days
        if i % 50 == 0:
            print(f"\n   Day {i}: {decision['action']} - {decision['rationale'][:60]}")
    
    # Summary
    print("\n" + "="*60)
    print("Simulation Summary")
    print("="*60)
    
    df_decisions = pd.DataFrame(decisions)
    
    # Calculate performance
    buy_returns = df_decisions[df_decisions['action'] == 'Buy']['actual_return']
    sell_returns = df_decisions[df_decisions['action'] == 'Sell']['actual_return']
    
    print(f"Total decisions: {len(df_decisions)}")
    print(f"Buy signals: {len(buy_returns)} (avg return: {buy_returns.mean():.2%})")
    print(f"Sell signals: {len(sell_returns)} (avg return: {sell_returns.mean():.2%})")
    
    # River ML status
    status = agent.get_river_status()
    print(f"\nRiver ML Status:")
    print(f"   Samples: {status['n_samples']}")
    print(f"   RMSE: {status['rmse']:.4f}")
    print(f"   Accuracy: {status['accuracy']:.2%}")
    print(f"   Drift detected: {status['drift_detected']}")
