#!/usr/bin/env python3
"""
ML Ensemble - Production Version with 10 Validated Features
Based on comprehensive statistical investigation results
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from utilities.shared_utils import extract_column

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Cache external data at module level to prevent repeated downloads
_CACHED_EXTERNAL_DATA = None

class MLEnsembleProduction:
    """
    Production ML Ensemble with 10 validated features
    Based on comprehensive statistical investigation and testing
    """
    
    def __init__(self):
        global _CACHED_EXTERNAL_DATA
        self.forecast_horizon = 20
        self.max_forecast_range = 0.10  # Reduced range for shorter horizon
        
        # Production feature set - 10 validated indicators
        self.production_features = [
            'volatility',           # Basic Technical - 0.744 balanced score
            'gold_equity_corr',     # Regime - 0.673 balanced score  
            'volatility_level',     # MACD-V - 0.660 balanced score
            'signal_quality',       # MACD-V - 0.659 balanced score
            'vix_rates_corr',       # Regime - 0.602 balanced score
            'cross_asset_dispersion', # Regime - 0.583 balanced score
            'macd_histogram',       # MACD-V - 0.569 balanced score
            'macd_signal',          # MACD-V - 0.569 balanced score
            'momentum',             # Basic Technical - 0.558 balanced score
            'equity_bonds_corr'     # Regime - 0.453 balanced score
        ]
        
        # Feature scaling bounds for production
        self.feature_bounds = {
            'volatility': (0.0, 1.0),           # 0-100% annualized vol
            'momentum': (-0.5, 0.5),             # -50% to +50% momentum
            'volatility_level': (0.5, 2.0),      # Volatility normalization
            'signal_quality': (0.0, 1.0),        # Signal strength 0-100%
            'macd_histogram': (-0.1, 0.1),       # MACD histogram range
            'macd_signal': (-0.05, 0.05),        # MACD signal range
            'gold_equity_corr': (-1.0, 1.0),     # Correlation range
            'vix_rates_corr': (-1.0, 1.0),       # Correlation range
            'cross_asset_dispersion': (0.0, 0.1), # Dispersion range
            'equity_bonds_corr': (-1.0, 1.0)     # Correlation range
        }
        
        # Use cached external data if available, otherwise fetch once
        if _CACHED_EXTERNAL_DATA is None:
            print(" Loading external market data for ML features...")
            self.external_data = self._load_external_data()
            _CACHED_EXTERNAL_DATA = self.external_data  # Cache for future instances
        else:
            self.external_data = _CACHED_EXTERNAL_DATA
            print(" Using cached external market data")
    
    def _load_external_data(self) -> Dict:
        """Load external market data for regime features"""
        try:
            from data_manager.external_data import fetch_external_data
            return fetch_external_data()
        except Exception as e:
            print(f"Warning: Could not load external data: {e}")
            return {}
    
    def forecast_etf(self, etf_data: pd.DataFrame) -> Dict:
        """
        Generate ML forecast using 10 production features
        
        Args:
            etf_data: DataFrame with OHLCV data
            
        Returns:
            Dict with forecast and metadata
        """
        if not ML_AVAILABLE:
            return self._get_default_forecast()
        
        try:
            # Extract price data
            prices = extract_column(etf_data, 'Close')
            volumes = extract_column(etf_data, 'Volume')
            
            if prices is None or len(prices) < 100:
                return self._get_default_forecast()
            
            # Extract 10 production features
            X = self._extract_production_features(prices, volumes)
            
            if X.shape[1] != 10:
                print(f"Warning: Expected 10 features, got {X.shape[1]}")
                return self._get_default_forecast()
            
            # Scale features
            X_scaled = self._scale_features(X)
            
            # Train models
            models = self._train_production_models(prices, X_scaled)
            
            if models.get('rf') is None:
                return self._get_default_forecast()
            
            # Generate predictions
            rf_forecast = models['rf'].predict(X_scaled)[0]
            ridge_forecast = models['ridge'].predict(X_scaled)[0]
            
            # Ensemble output
            ensemble_output = (rf_forecast + ridge_forecast) / 2.0
            ensemble_output = np.clip(ensemble_output, -self.max_forecast_range, self.max_forecast_range)
            
            # Calculate confidence
            confidence = self._calculate_confidence(models, X_scaled)
            
            # Get feature importance
            feature_importance = dict(zip(self.production_features, models['feature_importance']))
            
            return {
                'forecast_return': ensemble_output * 100,  # Convert to percentage
                'confidence_score': confidence,
                'features_used': dict(zip(self.production_features, X[0])),
                'feature_importance': feature_importance,
                'model_ensemble_output': ensemble_output,
                'trained_models': models
            }
            
        except Exception as e:
            print(f"ML forecast error: {e}")
            return self._get_default_forecast()
    
    def _extract_production_features(self, prices: pd.Series, volumes: pd.Series = None) -> np.ndarray:
        """
        Extract the 10 validated production features
        
        Returns:
            numpy array with shape (1, 10)
        """
        if len(prices) < 60:
            return np.array([[0.0] * 10])
        
        returns = prices.pct_change().dropna()
        features = []
        
        # 1. Volatility (Basic Technical) - 30-day rolling volatility
        volatility = returns.tail(30).std() * np.sqrt(252) if len(returns) > 30 else 0.15
        features.append(min(1.0, volatility))  # Cap at 100%
        
        # 2. Momentum (Basic Technical) - 20-day price momentum  
        momentum = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else 0
        features.append(np.clip(momentum, -0.5, 0.5))
        
        # 3-4. MACD Features (MACD-V indicators)
        macd_features = self._calculate_macd_features(prices)
        features.extend(macd_features)  # volatility_level, signal_quality, macd_histogram, macd_signal
        
        # 5-8. Regime Features (correlations and dispersion)
        regime_features = self._calculate_regime_features(prices)
        features.extend(regime_features)  # gold_equity_corr, vix_rates_corr, cross_asset_dispersion, equity_bonds_corr
        
        return np.array([features])
    
    def _calculate_macd_features(self, prices: pd.Series) -> list:
        """Calculate MACD-V production features"""
        try:
            # Calculate MACD
            exp1 = prices.ewm(span=12).mean()
            exp2 = prices.ewm(span=26).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            # Normalize by volatility for robustness
            volatility = prices.pct_change().tail(30).std()
            vol_normalized = 1.0 / max(volatility, 0.01)
            
            # Feature 3: volatility_level - volatility-normalized MACD
            volatility_level = float(macd_line.iloc[-1] * vol_normalized) if len(macd_line) > 0 else 0
            volatility_level = np.clip(volatility_level, 0.5, 2.0)
            
            # Feature 4: signal_quality - MACD signal strength
            signal_strength = abs(float(signal_line.iloc[-1])) if len(signal_line) > 0 else 0
            signal_quality = min(1.0, signal_strength * 20)  # Scale to 0-1
            signal_quality = np.clip(signal_quality, 0.0, 1.0)
            
            # Feature 7: macd_histogram - MACD histogram
            macd_hist = float(histogram.iloc[-1]) if len(histogram) > 0 else 0
            macd_histogram = np.clip(macd_hist / prices.iloc[-1], -0.1, 0.1)
            
            # Feature 8: macd_signal - Standard MACD signal
            macd_sig = float(macd_line.iloc[-1]) if len(macd_line) > 0 else 0
            macd_signal = np.clip(macd_sig / prices.iloc[-1], -0.05, 0.05)
            
            return [volatility_level, signal_quality, macd_histogram, macd_signal]
            
        except Exception as e:
            print(f"MACD feature calculation error: {e}")
            return [1.0, 0.5, 0.0, 0.0]  # Default values
    
    def _calculate_regime_features(self, prices: pd.Series) -> list:
        """Calculate regime correlation features"""
        features = []
        
        try:
            # Calculate ETF returns
            etf_returns = prices.pct_change().dropna()
            
            # Default values if external data unavailable
            default_features = [0.0, 0.0, 0.05, 0.0]
            
            if not self.external_data:
                return default_features
            
            # Feature 5: gold_equity_corr - Gold-equity correlation
            if 'Gold' in self.external_data:
                gold_returns = self.external_data['Gold'].pct_change().dropna()
                gold_corr = self._calculate_correlation(etf_returns, gold_returns)
                features.append(np.clip(gold_corr, -1.0, 1.0))
            else:
                features.append(0.0)
            
            # Feature 6: vix_rates_corr - VIX-rates correlation
            if 'VIX' in self.external_data and 'Rates' in self.external_data:
                vix_returns = self.external_data['VIX'].pct_change().dropna()
                rates_returns = self.external_data['Rates'].pct_change().dropna()
                vix_rates_corr = self._calculate_correlation(vix_returns, rates_returns)
                features.append(np.clip(vix_rates_corr, -1.0, 1.0))
            else:
                features.append(0.0)
            
            # Feature 9: cross_asset_dispersion - Cross-asset risk dispersion
            asset_returns = []
            for asset in ['SP500', 'NASDAQ', 'Gold', 'Bonds']:
                if asset in self.external_data:
                    asset_ret = self.external_data[asset].pct_change().dropna()
                    if len(asset_ret) > 0:
                        asset_returns.append(asset_ret.tail(60).std())
            
            if len(asset_returns) >= 2:
                dispersion = np.std(asset_returns)
                features.append(np.clip(dispersion, 0.0, 0.1))
            else:
                features.append(0.05)
            
            # Feature 10: equity_bonds_corr - Equity-bonds correlation
            if 'SP500' in self.external_data and 'Bonds' in self.external_data:
                equity_returns = self.external_data['SP500'].pct_change().dropna()
                bond_returns = self.external_data['Bonds'].pct_change().dropna()
                equity_bonds_corr = self._calculate_correlation(equity_returns, bond_returns)
                features.append(np.clip(equity_bonds_corr, -1.0, 1.0))
            else:
                features.append(0.0)
            
            return features
            
        except Exception as e:
            print(f"Regime feature calculation error: {e}")
            return [0.0, 0.0, 0.05, 0.0]  # Default values
    
    def _calculate_correlation(self, series1: pd.Series, series2: pd.Series, min_periods: int = 30) -> float:
        """Calculate rolling correlation with minimum period requirement"""
        try:
            # Align dates
            common_dates = series1.index.intersection(series2.index)
            if len(common_dates) < min_periods:
                return 0.0
            
            s1_aligned = series1.loc[common_dates]
            s2_aligned = series2.loc[common_dates]
            
            # Use last 60 days for correlation
            if len(s1_aligned) > 60:
                s1_aligned = s1_aligned.tail(60)
                s2_aligned = s2_aligned.tail(60)
            
            correlation = s1_aligned.corr(s2_aligned)
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features using production bounds"""
        X_scaled = X.copy()
        
        for i, feature_name in enumerate(self.production_features):
            if feature_name in self.feature_bounds:
                min_val, max_val = self.feature_bounds[feature_name]
                # Min-max scaling to [0, 1]
                X_scaled[0, i] = (X[0, i] - min_val) / (max_val - min_val)
                X_scaled[0, i] = np.clip(X_scaled[0, i], 0.0, 1.0)
        
        return X_scaled
    
    def _train_production_models(self, prices: pd.Series, X: np.ndarray) -> Dict:
        """Train Random Forest + Ridge ensemble for production"""
        if len(prices) < 50:  # Reduced minimum for 20-day forecasts
            return {'rf': None, 'ridge': None, 'scaler': None, 'feature_importance': []}
        
        try:
            # Prepare training data
            returns = prices.pct_change().dropna()
            
            # Create feature matrix for training
            lookback = len(prices) - self.forecast_horizon - 20
            if lookback < 30:  # Reduced lookback requirement
                return {'rf': None, 'ridge': None, 'scaler': None, 'feature_importance': []}
            
            # Train models
            rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
            ridge = Ridge(alpha=1.0, random_state=42)
            
            # For 20-day production forecasts
            cumulative_return = returns.tail(self.forecast_horizon).sum()
            rf.fit(X, [cumulative_return])
            ridge.fit(X, [cumulative_return])
            
            return {
                'rf': rf,
                'ridge': ridge,
                'scaler': StandardScaler(),
                'feature_importance': rf.feature_importances_.tolist()
            }
            
        except Exception as e:
            print(f"Model training error: {e}")
            return {'rf': None, 'ridge': None, 'scaler': None, 'feature_importance': []}
    
    def _calculate_confidence(self, models: Dict, X: np.ndarray) -> float:
        """Calculate prediction confidence"""
        try:
            if models.get('rf') is None:
                return 0.5
            
            # Simple confidence based on feature consistency
            rf_pred = models['rf'].predict(X)[0]
            ridge_pred = models['ridge'].predict(X)[0]
            
            # Higher confidence when models agree
            agreement = 1.0 - abs(rf_pred - ridge_pred) / (abs(rf_pred) + abs(ridge_pred) + 0.01)
            confidence = max(0.1, min(0.9, agreement))
            
            return confidence
            
        except Exception:
            return 0.5
    
    def walk_forward_validate(self, prices: pd.Series, models: Dict = None) -> Dict:
        """
        Simple validation for production ensemble
        
        Args:
            prices: Price series for validation
            models: Pre-trained models (optional)
            
        Returns:
            Dict with validation metrics
        """
        if not ML_AVAILABLE or len(prices) < 120:  # 100 train + 20 test minimum
            return {'mae': np.nan, 'hit_rate': np.nan}
        
        try:
            # Proper walk-forward validation for 20-day forecasts
            window_size = 100  # Reduced training window for faster validation
            forecast_horizon = 20
            step_size = 15  # Test every 15 days for more validation points
            
            forecasts = []
            actuals = []
            
            # Walk through data with rolling windows
            for end_idx in range(window_size + forecast_horizon, len(prices) - forecast_horizon, step_size):
                try:
                    # Define training and test periods
                    train_start = end_idx - window_size - forecast_horizon
                    train_end = end_idx - forecast_horizon
                    
                    if train_start < 0 or train_end >= len(prices):
                        continue
                    
                    # Training data
                    train_prices = prices.iloc[train_start:train_end]
                    test_prices = prices.iloc[train_end:end_idx]
                    
                    if len(train_prices) < 50 or len(test_prices) < forecast_horizon:
                        continue
                    
                    # Extract features from END of training period
                    X_train = self._extract_production_features(train_prices)
                    X_train_scaled = self._scale_features(X_train)
                    models = self._train_production_models(train_prices, X_train_scaled)
                    
                    if models.get('rf') is None:
                        continue
                    
                    # Make prediction using features at training end
                    X_pred = self._extract_production_features(train_prices.iloc[-1:])
                    X_pred_scaled = self._scale_features(X_pred)
                    
                    rf_pred = models['rf'].predict(X_pred_scaled)[0]
                    ridge_pred = models['ridge'].predict(X_pred_scaled)[0]
                    forecast = (rf_pred + ridge_pred) / 2
                    
                    # Calculate actual cumulative return for test period
                    actual_return = test_prices.pct_change().sum()
                    
                    forecasts.append(forecast)
                    actuals.append(actual_return)
                    
                except Exception as e:
                    continue
            
            if len(forecasts) < 3:  # Need multiple predictions for meaningful hit rate
                return {'hit_rate': np.nan}
            
            # Calculate hit rate (directional accuracy)
            hit_rate = np.mean(np.sign(forecasts) == np.sign(actuals))
            
            return {'hit_rate': hit_rate}
            
        except Exception as e:
            print(f"Walk-forward validation error: {e}")
            return {'hit_rate': np.nan}
    
    def _get_default_forecast(self) -> Dict:
        """Return default forecast when models unavailable"""
        return {
            'forecast_return': 0.0,
            'confidence_score': 0.5,
            'features_used': {feat: 0.0 for feat in self.production_features},
            'feature_importance': {feat: 0.1 for feat in self.production_features},
            'model_ensemble_output': 0.0,
            'trained_models': None
        }
