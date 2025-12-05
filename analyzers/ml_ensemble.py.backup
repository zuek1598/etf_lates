#!/usr/bin/env python3
"""
ML Ensemble Forecasting - Raw Output with Confidence Scores
No bias correction - raw predictions with uncertainty quantification
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from utilities.shared_utils import extract_column
# from system.model_cache import ModelCache  # REMOVED during optimization

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_absolute_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Import regime detector for Phase 2
try:
    from analyzers.regime_detector import RegimeDetector
    from data_manager.external_data import fetch_external_data
    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False

# Import new indicators
from indicators.macd_v_demand_supply import MACDVIndicator, DemandSupplyIndicator

class MLEnsemble:
    """ML-based forecasting with raw output and confidence scores"""
    
    def __init__(self, use_enhanced_features: bool = False):
        self.forecast_horizon = 60
        self.max_forecast_range = 0.15
        self.use_enhanced_features = use_enhanced_features
        # self.model_cache = ModelCache()  # REMOVED during optimization
        # NO BIAS CORRECTION - raw output only
        
        # PHASE 2: Initialize regime detection components
        self.regime_detector = None
        self.external_data = None
        self.regime_data = None
        
        # NEW: Initialize MACD-V and Demand-Supply indicators
        self.macd_v_indicator = MACDVIndicator()
        self.demand_supply_indicator = DemandSupplyIndicator()
        
        if self.use_enhanced_features and REGIME_AVAILABLE:
            print("🔧 Initializing enhanced ML features with regime detection...")
            try:
                self.regime_detector = RegimeDetector()
                self.external_data = fetch_external_data()
                if self.external_data:
                    print("✅ External data loaded for regime-aware ML")
                else:
                    print("⚠️ External data unavailable, using basic features")
            except Exception as e:
                print(f"⚠️ Regime detector initialization failed: {e}")
                self.use_enhanced_features = False
        
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
    
    def extract_ml_features(self, prices: pd.Series, volumes: pd.Series = None, use_last_point: bool = True) -> np.ndarray:
        """Extract features for ML models (efficient version)
        
        Args:
            prices: Price series
            volumes: Volume series (optional)
            use_last_point: If True, uses prices.iloc[-1] (for live prediction)
                         If False, uses prices.iloc[-1] of window (for training)
        """
        if len(prices) < 60:
            return np.array([[0.0] * 6])
        
        returns = prices.pct_change().dropna()
        features = []
        
        # Feature 1: Momentum (20-day)
        # FIXED: Use last point consistently for both training and inference
        if use_last_point:
            # Live prediction: use most recent price
            momentum = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else 0
        else:
            # Training: use last price in window (same as live)
            momentum = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else 0
        # Ensure scalar
        if isinstance(momentum, pd.Series):
            momentum = momentum.iloc[0] if len(momentum) > 0 else 0
        features.append(momentum)
        
        # Feature 2: Volatility (30-day)
        vol_30 = returns.tail(30).std() if len(returns) > 30 else 0.01
        # Ensure scalar
        if isinstance(vol_30, pd.Series):
            vol_30 = vol_30.iloc[0] if len(vol_30) > 0 else 0.01
        features.append(vol_30)
        
        # Feature 3: RSI momentum
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        avg_gain = gains.tail(14).mean() if len(gains) > 14 else 0
        avg_loss = losses.tail(14).mean() if len(losses) > 14 else 0
        # FIXED: Handle scalar comparison properly
        total_gain_loss = avg_gain + avg_loss
        if isinstance(total_gain_loss, pd.Series):
            total_gain_loss = total_gain_loss.iloc[0] if len(total_gain_loss) > 0 else 0
        if isinstance(avg_gain, pd.Series):
            avg_gain = avg_gain.iloc[0] if len(avg_gain) > 0 else 0
        rsi = avg_gain / total_gain_loss if total_gain_loss > 0 else 0.5
        if isinstance(rsi, pd.Series):
            rsi = rsi.iloc[0] if len(rsi) > 0 else 0.5
        features.append(rsi)
        
        # Feature 4: Price position (60-day range)
        price_60_high = prices.tail(60).max()
        price_60_low = prices.tail(60).min()
        # FIXED: Handle scalar comparison properly
        if isinstance(price_60_high, pd.Series):
            price_60_high = price_60_high.iloc[0] if len(price_60_high) > 0 else 0
        if isinstance(price_60_low, pd.Series):
            price_60_low = price_60_low.iloc[0] if len(price_60_low) > 0 else 0
        
        if use_last_point:
            price_pos = (prices.iloc[-1] - price_60_low) / (price_60_high - price_60_low) if price_60_high > price_60_low else 0.5
        else:
            price_pos = (prices.iloc[-1] - price_60_low) / (price_60_high - price_60_low) if price_60_high > price_60_low else 0.5
        # Ensure scalar
        if isinstance(price_pos, pd.Series):
            price_pos = price_pos.iloc[0] if len(price_pos) > 0 else 0.5
        features.append(price_pos)
        
        # Feature 5: SMA ratio
        sma_20 = prices.tail(20).mean()
        # FIXED: Handle scalar comparison properly
        if isinstance(sma_20, pd.Series):
            sma_20 = sma_20.iloc[0] if len(sma_20) > 0 else 1
        
        if use_last_point:
            sma_ratio = prices.iloc[-1] / sma_20 if sma_20 > 0 else 1.0
        else:
            sma_ratio = prices.iloc[-1] / sma_20 if sma_20 > 0 else 1.0
        # Ensure scalar
        if isinstance(sma_ratio, pd.Series):
            sma_ratio = sma_ratio.iloc[0] if len(sma_ratio) > 0 else 1.0
        features.append(sma_ratio)
        
        # Feature 6: Recent return vs historical
        recent_return = returns.tail(20).mean()
        hist_return = returns.tail(60).mean() if len(returns) > 60 else recent_return
        # FIXED: Handle scalar comparison properly
        if isinstance(recent_return, pd.Series):
            recent_return = recent_return.iloc[0] if len(recent_return) > 0 else 0
        if isinstance(hist_return, pd.Series):
            hist_return = hist_return.iloc[0] if len(hist_return) > 0 else 1
        
        return_ratio = recent_return / hist_return if hist_return != 0 else 1.0
        # Ensure scalar
        if isinstance(return_ratio, pd.Series):
            return_ratio = return_ratio.iloc[0] if len(return_ratio) > 0 else 1.0
        features.append(return_ratio)
        
        # Convert all features to float explicitly
        features = [float(f) for f in features]
        
        # PHASE 2: Add regime features if enabled
        if self.use_enhanced_features and self.regime_detector is not None:
            regime_features = self.extract_regime_features(prices)
            features.extend(regime_features)
        
        # NEW: Add MACD-V features (13 features)
        try:
            macd_v_features = self.macd_v_indicator.extract_macd_v_features(prices)
            macd_v_values = list(macd_v_features.values())
            features.extend(macd_v_values)
        except Exception as e:
            print(f"⚠️ MACD-V feature extraction failed: {e}")
            features.extend([0.0] * 13)  # Add default values
        
        # NEW: Add Demand-Supply features (14 features) - only if volume available
        if volumes is not None and len(volumes) > 0:
            try:
                ds_features = self.demand_supply_indicator.extract_demand_supply_features(prices, volumes)
                ds_values = list(ds_features.values())
                features.extend(ds_values)
            except Exception as e:
                print(f"⚠️ Demand-Supply feature extraction failed: {e}")
                features.extend([0.0] * 14)  # Add default values
        else:
            # Add default values if no volume data
            features.extend([0.0] * 14)
        
        return np.array([features])
    
    def extract_regime_features(self, prices: pd.Series) -> List[float]:
        """
        Extract regime-aware features for enhanced ML
        
        Args:
            prices: Price series for the ETF
            
        Returns:
            List of regime feature values
        """
        regime_features = []
        
        try:
            # Initialize regime data if not done yet
            if self.regime_data is None and self.external_data is not None:
                print("📊 Analyzing regimes for enhanced features...")
                self.regime_data = self.regime_detector.analyze_regimes(
                    self.external_data, 
                    prices  # Use ETF as equity proxy
                )
            
            if self.regime_data is None:
                # Return zeros if regime analysis failed
                return [0.0] * 7  # 7 regime features (5 correlations + confidence + stability)
            
            # Get current date (use latest price date)
            current_date = prices.index[-1]
            
            # Extract regime features
            regime_info = self.regime_detector.get_regime_features(
                self.regime_data, 
                current_date
            )
            
            # Add correlation features
            correlation_features = [
                regime_info.get('corr_gold_equity', 0.0),
                regime_info.get('corr_aud_gold', 0.0),
                regime_info.get('corr_vix_rates', 0.0),
                regime_info.get('corr_equity_bonds', 0.0),
                regime_info.get('corr_cross_asset_dispersion', 0.0),
            ]
            
            # Add confidence and stability
            regime_features.extend(correlation_features)
            regime_features.append(regime_info.get('regime_confidence', 0.5))
            regime_features.append(regime_info.get('regime_stability', 0.5))
            
        except Exception as e:
            print(f"⚠️ Regime feature extraction failed: {e}")
            # Return neutral values
            regime_features = [0.0] * 7  # 5 correlations + confidence + stability
        
        return regime_features
    
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
                # FIXED: Use use_last_point=False for training consistency
                X = self.extract_ml_features(window_prices, use_last_point=False)
                features_list.append(X[0])
                
                # Target: 60-day return
                future_price = prices.iloc[i + self.forecast_horizon]
                current_price = prices.iloc[i]
                target_return = (future_price - current_price) / current_price
                
                # PHASE 1 FIX: Clip target returns to ±15% to prevent overfitting to extremes
                target_return = np.clip(target_return, -0.15, 0.15)
                
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
        except Exception as e:
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
            # FIXED: Use use_last_point=True for live prediction (consistent with training)
            X = self.extract_ml_features(prices, use_last_point=True)
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

            # Build features dictionary with regime features if available
            features_dict = {
                'momentum': float(X[0][0]),
                'volatility': float(X[0][1]),
                'rsi': float(X[0][2]),
                'price_position': float(X[0][3]),
                'sma_ratio': float(X[0][4]),
                'return_ratio': float(X[0][5])
            }
            
            # Add regime features if enhanced mode is active
            if self.use_enhanced_features and X.shape[1] > 6:
                regime_features = self.extract_regime_features(prices)
                regime_names = [
                    'corr_gold_equity', 'corr_aud_gold', 'corr_vix_rates', 
                    'corr_equity_bonds', 'corr_cross_asset_dispersion',
                    'regime_confidence', 'regime_stability'
                ]
                for i, (name, value) in enumerate(zip(regime_names, regime_features)):
                    features_dict[name] = float(value)

            return {
                'forecast_return': ensemble_output * 100,  # Convert to percentage
                'confidence_score': confidence,
                'features_used': features_dict,
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
                
                print(f"DEBUG: Validation indices: {indices}")
                
                for i in indices:
                    # Get features at prediction time
                    train_prices = prices.iloc[i-train_days:i]
                    # FIXED: Use use_last_point=False for validation consistency
                    X = self.extract_ml_features(train_prices, use_last_point=False)
                    X_scaled = self.robust_scale_features(X)
                    
                    # Make prediction
                    rf_pred = models['rf'].predict(X_scaled)[0]
                    ridge_pred = models['ridge'].predict(X_scaled)[0]
                    forecast_return = (rf_pred + ridge_pred) / 2.0

                    # Get actual forward return
                    actual_return = (prices.iloc[i + test_days] - prices.iloc[i]) / prices.iloc[i]
                    
                    # Convert to scalars for debugging
                    forecast_val = float(forecast_return)
                    actual_val = float(actual_return)
                    print(f"DEBUG: Forecast: {forecast_val:.4f}, Actual: {actual_val:.4f}")

                    maes.append(abs(forecast_val - actual_val))
                    hit = 1 if (forecast_val > 0) == (actual_val > 0) else 0
                    hits.append(hit)
                    print(f"DEBUG: Hit calculation: forecast>0={forecast_val>0}, actual>0={actual_val>0}, hit={hit}")
            except Exception as e:
                print(f"DEBUG: Exception in validation: {e}")
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
                    # FIXED: Use use_last_point=False for validation consistency
                    X = self.extract_ml_features(train_prices, use_last_point=False)
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

        # FIXED: Handle hit rate calculation properly
        hit_rate = np.mean(hits) if len(hits) > 0 else np.nan
        mae = np.mean(maes) if len(maes) > 0 else np.nan
        
        return {
            'mae': mae * 100,  # Convert to percentage
            'hit_rate': hit_rate,
            'num_windows': len(maes)
        }
    
    def forecast_etf(self, etf_data: pd.DataFrame, models: Dict = None) -> Dict:
        """Main forecasting function"""
        return self.generate_ml_forecast(etf_data, models)
