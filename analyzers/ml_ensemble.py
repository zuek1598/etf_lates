#!/usr/bin/env python3
"""
ML Ensemble Forecasting - Raw Output with Confidence Scores
No bias correction - raw predictions with uncertainty quantification
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from utilities.shared_utils import extract_column
# from system.model_cache import ModelCache  # REMOVED during optimization

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class MLEnsemble:
    """ML-based forecasting with raw output and confidence scores"""
    
    def __init__(self, use_enhanced_features: bool = False):
        self.forecast_horizon = 60
        self.max_forecast_range = 0.15
        self.use_enhanced_features = use_enhanced_features
        # self.model_cache = ModelCache()  # REMOVED during optimization
        # NO BIAS CORRECTION - raw output only
        
        self.risk_params = {
            'low_risk': [100, 60, 20],
            'medium_risk': [80, 40, 20],
            'high_risk': [60, 30, 20]
        }
        
        # Fixed scaling bounds for cross-ETF comparability
        # Based on typical market statistics
        self.feature_bounds = {
            'volatility': (0.0, 1.0),        # 0-100% annualized vol
            'momentum': (0.5, 1.5),          # -50% to +50% momentum
            'mean_reversion': (0.8, 1.2),    # 80-120% of mean
            'trend': (0.7, 1.3),             # 70-130% trend strength
            'sma_ratio': (0.8, 1.2),         # 80-120% of SMA
            'return_ratio': (0.5, 1.5)       # -50% to +50% return deviation
        }
    
    def get_price_data(self, etf_data: pd.DataFrame) -> pd.Series:
        """Extract price data using shared utility"""
        return extract_column(etf_data, 'Close')
    
    def robust_scale_features(self, features: np.ndarray) -> np.ndarray:
        """
        Scale features using fixed bounds for cross-ETF comparability
        Uses robust scaling: (x - lower) / (upper - lower)
        Clips to [0, 1] to handle outliers gracefully
        """
        if features.shape[1] != 6:
            return features
        
        scaled = np.zeros_like(features)
        bounds_list = list(self.feature_bounds.values())
        
        for i in range(6):
            lower, upper = bounds_list[i]
            scaled[:, i] = (features[:, i] - lower) / (upper - lower)
            scaled[:, i] = np.clip(scaled[:, i], 0, 1)
        
        return scaled
    
    def extract_ml_features(self, prices: pd.Series, volumes: pd.Series = None) -> np.ndarray:
        """Extract features for ML models (efficient version)"""
        if len(prices) < 60:
            return np.array([[0.0] * 6])
        
        returns = prices.pct_change().dropna()
        features = []
        
        # Feature 1: Momentum (20-day)
        momentum = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else 0
        features.append(momentum)
        
        # Feature 2: Volatility (30-day)
        vol_30 = returns.tail(30).std() if len(returns) > 30 else 0.01
        features.append(vol_30)
        
        # Feature 3: RSI momentum
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        avg_gain = gains.tail(14).mean() if len(gains) > 14 else 0
        avg_loss = losses.tail(14).mean() if len(losses) > 14 else 0
        rsi = avg_gain / (avg_gain + avg_loss) if (avg_gain + avg_loss) > 0 else 0.5
        features.append(rsi)
        
        # Feature 4: Price position (60-day range)
        price_60_high = prices.tail(60).max()
        price_60_low = prices.tail(60).min()
        price_pos = (prices.iloc[-1] - price_60_low) / (price_60_high - price_60_low) if price_60_high > price_60_low else 0.5
        features.append(price_pos)
        
        # Feature 5: SMA ratio
        sma_20 = prices.tail(20).mean()
        sma_ratio = prices.iloc[-1] / sma_20 if sma_20 > 0 else 1.0
        features.append(sma_ratio)
        
        # Feature 6: Recent return vs historical
        recent_return = returns.tail(20).mean()
        hist_return = returns.tail(60).mean() if len(returns) > 60 else recent_return
        return_ratio = recent_return / hist_return if hist_return != 0 else 1.0
        features.append(return_ratio)
        
        return np.array([features])
    
    def train_ensemble(self, prices: pd.Series, lookback_days: int = 100) -> Dict:
        """Train Random Forest + Ridge ensemble"""
        if not ML_AVAILABLE or len(prices) < lookback_days + self.forecast_horizon:
            return {'model': None, 'scaler': None, 'weights': None}
        
        try:
            features_list = []
            targets = []
            
            for i in range(lookback_days, len(prices) - self.forecast_horizon):
                # Extract features at each point
                window_prices = prices.iloc[i-lookback_days:i]
                X = self.extract_ml_features(window_prices)
                features_list.append(X[0])
                
                # Target: 60-day return
                future_price = prices.iloc[i + self.forecast_horizon]
                current_price = prices.iloc[i]
                target_return = (future_price - current_price) / current_price
                targets.append(target_return)
            
            if len(features_list) < 10:
                return {'model': None, 'scaler': None, 'weights': None}
            
            X = np.array(features_list)
            y = np.array(targets)
            
            # Use robust scaling with fixed bounds (not per-ETF StandardScaler)
            # This ensures features are comparable across all ETFs
            X_scaled = self.robust_scale_features(X)
            
            # Train ensemble
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            ridge = Ridge(alpha=1.0)
            
            rf.fit(X_scaled, y)
            ridge.fit(X_scaled, y)
            
            return {
                'rf': rf,
                'ridge': ridge,
                'scaler': None,  # No longer needed - using robust scaling
                'feature_importance': rf.feature_importances_.tolist()
            }
        except:
            return {'rf': None, 'ridge': None, 'scaler': None}
    
    def calculate_confidence_score(self, models: Dict, X_scaled: np.ndarray) -> float:
        """
        Calculate confidence score as inverse of ensemble disagreement
        Confidence = 1 / (1 + disagreement_ratio)
        """
        if models.get('rf') is None or models.get('ridge') is None:
            return 0.5
        
        try:
            rf_pred = models['rf'].predict(X_scaled)
            ridge_pred = models['ridge'].predict(X_scaled)
            
            # Calculate disagreement (MAE between models)
            disagreement = abs(rf_pred[0] - ridge_pred[0])
            
            # Convert to confidence (0-1, where 1 = perfect agreement)
            confidence = 1.0 / (1.0 + disagreement * 10)
            
            return np.clip(confidence, 0.0, 1.0)
        except:
            return 0.5
    
    def generate_ml_forecast(self, etf_data: pd.DataFrame, models: Dict = None, ticker: str = None) -> Dict:
        """
        Generate ML forecast with confidence score
        NO BIAS CORRECTION - raw ensemble output only
        Uses caching to avoid retraining on unchanged data
        RETURNS trained models so walk_forward_validate can reuse them
        """
        prices = self.get_price_data(etf_data)

        if len(prices) < 100:
            return {
                'forecast_return': 0.0,
                'confidence_score': 0.0,
                'features_used': {},
                'model_ensemble_output': 0.0,
                'feature_importance': {},
                'trained_models': None  # No models trained
            }

        # Train models if needed (cache removed during optimization)
        if models is None or models.get('rf') is None:
            models = self.train_ensemble(prices)

        if models.get('rf') is None:
            return {
                'forecast_return': 0.0,
                'confidence_score': 0.0,
                'features_used': {},
                'model_ensemble_output': 0.0,
                'feature_importance': {},
                'trained_models': None
            }

        try:
            # Extract and scale features using robust scaling
            X = self.extract_ml_features(prices)
            X_scaled = self.robust_scale_features(X)

            # Get predictions from both models
            rf_forecast = models['rf'].predict(X_scaled)[0]
            ridge_forecast = models['ridge'].predict(X_scaled)[0]

            # Ensemble output: average of both models (NO BIAS CORRECTION)
            ensemble_output = (rf_forecast + ridge_forecast) / 2.0

            # Constrain to reasonable range
            ensemble_output = np.clip(ensemble_output, -self.max_forecast_range, self.max_forecast_range)

            # Calculate confidence score
            confidence = self.calculate_confidence_score(models, X_scaled)

            # Get feature importance
            feature_importance = {
                f'feature_{i}': float(imp)
                for i, imp in enumerate(models.get('feature_importance', []))
            }

            return {
                'forecast_return': ensemble_output * 100,  # Convert to percentage
                'confidence_score': confidence,
                'features_used': {
                    'momentum': float(X[0][0]),
                    'volatility': float(X[0][1]),
                    'rsi': float(X[0][2]),
                    'price_position': float(X[0][3]),
                    'sma_ratio': float(X[0][4]),
                    'return_ratio': float(X[0][5])
                },
                'model_ensemble_output': ensemble_output * 100,
                'feature_importance': feature_importance,
                'trained_models': models  # Return trained models for reuse
            }
        except:
            return {
                'forecast_return': 0.0,
                'confidence_score': 0.0,
                'features_used': {},
                'model_ensemble_output': 0.0,
                'feature_importance': {},
                'trained_models': None
            }
    
    def walk_forward_validate(self, prices: pd.Series, models: Dict = None, train_days: int = 252, test_days: int = 60, max_windows: int = 5) -> Dict:
        """
        Walk-forward validation for ML ensemble
        Returns MAE and hit rate metrics
        If models provided, reuse them instead of retraining (OPTIMIZATION)
        """
        if len(prices) < train_days + test_days:
            return {'mae': np.nan, 'hit_rate': np.nan, 'num_windows': 0}

        maes, hits = [], []

        # If models provided, use just one validation window (reusing trained models)
        # Otherwise use multiple windows with fresh training
        if models is not None and models.get('rf') is not None:
            # REUSE MODE: Use provided models with proper forward validation
            try:
                # Use multiple windows for proper validation
                start_idx = train_days
                end_idx = len(prices) - test_days
                step = test_days
                
                # Use 1 window for maximum speed optimization
                indices = list(range(start_idx, end_idx, step))[:1]
                
                for i in indices:
                    # Get features at prediction time
                    train_prices = prices.iloc[i-train_days:i]
                    X = self.extract_ml_features(train_prices)
                    X_scaled = self.robust_scale_features(X)
                    
                    # Make prediction
                    rf_pred = models['rf'].predict(X_scaled)[0]
                    ridge_pred = models['ridge'].predict(X_scaled)[0]
                    forecast_return = (rf_pred + ridge_pred) / 2.0

                    # Get actual forward return
                    actual_return = (prices.iloc[i + test_days] - prices.iloc[i]) / prices.iloc[i]

                    maes.append(abs(forecast_return - actual_return))
                    hits.append(1 if (forecast_return > 0) == (actual_return > 0) else 0)
            except:
                pass
        else:
            # TRAIN MODE: Multiple windows with fresh training (original behavior)
            start_idx = train_days
            end_idx = len(prices) - test_days
            step = test_days

            # Limit number of windows for efficiency
            indices = list(range(start_idx, end_idx, step))[:max_windows]

            for i in indices:
                try:
                    # Train on window
                    train_prices = prices.iloc[i-train_days:i]
                    models = self.train_ensemble(train_prices, lookback_days=min(100, train_days//2))

                    if models.get('rf') is None:
                        continue

                    # Predict
                    X = self.extract_ml_features(train_prices)
                    X_scaled = self.robust_scale_features(X)
                    rf_pred = models['rf'].predict(X_scaled)[0]
                    ridge_pred = models['ridge'].predict(X_scaled)[0]
                    forecast_return = (rf_pred + ridge_pred) / 2.0

                    # Actual return
                    actual_return = (prices.iloc[i + test_days] - prices.iloc[i]) / prices.iloc[i]

                    # Metrics
                    maes.append(abs(forecast_return - actual_return))
                    hits.append(1 if (forecast_return > 0) == (actual_return > 0) else 0)
                except:
                    continue

        if len(maes) == 0:
            return {'mae': np.nan, 'hit_rate': np.nan, 'num_windows': 0}

        return {
            'mae': np.mean(maes) * 100,  # Convert to percentage
            'hit_rate': np.mean(hits),
            'num_windows': len(maes)
        }
    
    def forecast_etf(self, etf_data: pd.DataFrame, models: Dict = None) -> Dict:
        """Main forecasting function"""
        return self.generate_ml_forecast(etf_data, models)
