"""
Dual-target ETF forecasting system
Maintains separate models for mean daily vs cumulative return prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import pickle
import os

class DualTargetForecaster:
    """
    Dual-model system for ETF forecasting with different prediction targets
    
    Model A: Mean Daily Return (for ranking ETFs)
    Model B: Cumulative Return (for absolute forecasting)
    """
    
    def __init__(self, forecast_horizon: int = 60):
        """
        Initialize dual-target forecasting system
        
        Args:
            forecast_horizon: Number of days to forecast ahead
        """
        self.forecast_horizon = forecast_horizon
        
        # Mean daily return models
        self.rf_daily = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_leaf=5, 
            random_state=42
        )
        self.ridge_daily = Ridge(alpha=1.0, random_state=42)
        self.scaler_daily = StandardScaler()
        
        # Cumulative return models
        self.rf_cumulative = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            min_samples_leaf=5, 
            random_state=42
        )
        self.ridge_cumulative = Ridge(alpha=1.0, random_state=42)
        self.scaler_cumulative = StandardScaler()
        
        # Production features (validated 10-feature set)
        self.production_features = [
            'volatility', 'gold_equity_corr', 'volatility_level', 
            'signal_quality', 'vix_rates_corr', 'cross_asset_dispersion',
            'macd_histogram', 'macd_signal', 'momentum', 'equity_bonds_corr'
        ]
        
        # Feature bounds for scaling
        self.feature_bounds = {
            'volatility': (0.0, 0.5),
            'gold_equity_corr': (-1.0, 1.0),
            'volatility_level': (0.0, 1.0),
            'signal_quality': (0.0, 1.0),
            'vix_rates_corr': (-1.0, 1.0),
            'cross_asset_dispersion': (0.0, 1.0),
            'macd_histogram': (-0.1, 0.1),
            'macd_signal': (-0.1, 0.1),
            'momentum': (-0.5, 0.5),
            'equity_bonds_corr': (-1.0, 1.0)
        }
        
        # Training targets for validation
        self.training_targets = {}
        
    def train(self, X: np.ndarray, returns: pd.Series) -> Dict:
        """
        Train both models on same features, different targets
        
        Args:
            X: Feature matrix (n_samples, 10_features)
            returns: Pandas Series of daily returns
            
        Returns:
            dict with both target values for logging
        """
        if len(returns) < self.forecast_horizon + 100:
            return None
            
        # Calculate both targets
        recent_returns = returns.tail(self.forecast_horizon)
        
        # Target 1: Mean daily return
        target_daily = recent_returns.mean()
        
        # Target 2: Cumulative return
        target_cumulative = recent_returns.sum()
        
        # Store for validation
        self.training_targets = {
            'mean_daily': target_daily,
            'cumulative': target_cumulative,
            'ratio': target_cumulative / (target_daily * self.forecast_horizon)
        }
        
        # Scale features
        X_scaled_daily = self.scaler_daily.fit_transform(X)
        X_scaled_cumulative = self.scaler_cumulative.fit_transform(X)
        
        # Train models
        self.rf_daily.fit(X_scaled_daily, [target_daily])
        self.ridge_daily.fit(X_scaled_daily, [target_daily])
        
        self.rf_cumulative.fit(X_scaled_cumulative, [target_cumulative])
        self.ridge_cumulative.fit(X_scaled_cumulative, [target_cumulative])
        
        return self.training_targets
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Generate predictions from both models
        
        Args:
            features: Feature vector (1, 10_features)
            
        Returns:
            dict with comprehensive forecast information
        """
        if not hasattr(self, 'rf_daily'):
            return None
            
        # Scale features
        features_scaled_daily = self.scaler_daily.transform(features.reshape(1, -1))
        features_scaled_cumulative = self.scaler_cumulative.transform(features.reshape(1, -1))
        
        # Get predictions
        rf_daily_pred = self.rf_daily.predict(features_scaled_daily)[0]
        ridge_daily_pred = self.ridge_daily.predict(features_scaled_daily)[0]
        
        rf_cumulative_pred = self.rf_cumulative.predict(features_scaled_cumulative)[0]
        ridge_cumulative_pred = self.ridge_cumulative.predict(features_scaled_cumulative)[0]
        
        # Ensemble predictions
        daily_pred = (rf_daily_pred + ridge_daily_pred) / 2
        cumulative_pred = (rf_cumulative_pred + ridge_cumulative_pred) / 2
        
        # Calculate derived metrics
        daily_to_period = daily_pred * self.forecast_horizon
        
        # Model agreement (how closely they align)
        if abs(cumulative_pred) > 0:
            agreement = 1 - abs(daily_to_period - cumulative_pred) / abs(cumulative_pred)
            agreement = max(0, min(1, agreement))
        else:
            agreement = 1.0
        
        return {
            'mean_daily_return': daily_pred,
            'cumulative_return': cumulative_pred,
            'daily_to_period_conversion': daily_to_period,
            'model_agreement': agreement,
            'forecast_horizon': self.forecast_horizon,
            'prediction_confidence': self._calculate_confidence(daily_pred, cumulative_pred)
        }
    
    def _calculate_confidence(self, daily_pred: float, cumulative_pred: float) -> float:
        """Calculate prediction confidence based on model agreement and magnitude"""
        # Basic confidence based on model agreement
        agreement = 1 - abs(daily_pred * self.forecast_horizon - cumulative_pred) / max(abs(cumulative_pred), 0.01)
        
        # Confidence based on reasonable ranges
        daily_reasonable = 0.0005 <= abs(daily_pred) <= 0.01  # 0.05% to 1% daily
        cumulative_reasonable = 0.01 <= abs(cumulative_pred) <= 0.5  # 1% to 50% cumulative
        
        confidence = agreement * (daily_reasonable + cumulative_reasonable) / 2
        return min(1.0, max(0.0, confidence))
    
    def save_models(self, directory: str = "models"):
        """Save trained models to disk"""
        os.makedirs(directory, exist_ok=True)
        
        model_data = {
            'rf_daily': self.rf_daily,
            'ridge_daily': self.ridge_daily,
            'scaler_daily': self.scaler_daily,
            'rf_cumulative': self.rf_cumulative,
            'ridge_cumulative': self.ridge_cumulative,
            'scaler_cumulative': self.scaler_cumulative,
            'forecast_horizon': self.forecast_horizon,
            'training_targets': self.training_targets
        }
        
        with open(f"{directory}/dual_target_forecaster.pkl", 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_models(self, directory: str = "models"):
        """Load trained models from disk"""
        try:
            with open(f"{directory}/dual_target_forecaster.pkl", 'rb') as f:
                model_data = pickle.load(f)
                
            self.rf_daily = model_data['rf_daily']
            self.ridge_daily = model_data['ridge_daily']
            self.scaler_daily = model_data['scaler_daily']
            self.rf_cumulative = model_data['rf_cumulative']
            self.ridge_cumulative = model_data['ridge_cumulative']
            self.scaler_cumulative = model_data['scaler_cumulative']
            self.forecast_horizon = model_data['forecast_horizon']
            self.training_targets = model_data.get('training_targets', {})
            
            return True
        except FileNotFoundError:
            return False
