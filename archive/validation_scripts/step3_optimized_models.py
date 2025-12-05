#!/usr/bin/env python3
"""
Step 3: Optimize ML Models with Statistically Validated Features
Create optimized ML models using only the features that passed statistical significance testing
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from analyzers.ml_ensemble import MLEnsemble
from utilities.shared_utils import extract_column
import time

class OptimizedMLEnsemble:
    """
    Optimized ML Ensemble using only statistically validated features
    """
    
    def __init__(self, feature_set='balanced'):
        """
        Initialize optimized ML ensemble
        
        Args:
            feature_set: 'conservative', 'balanced', or 'comprehensive'
        """
        self.feature_set = feature_set
        self.forecast_horizon = 60
        self.max_forecast_range = 0.15
        
        # Load feature significance analysis
        with open('data/feature_significance_analysis.json', 'r') as f:
            analysis = json.load(f)
        
        # Select features based on chosen set
        if feature_set == 'conservative':
            self.selected_features = analysis['recommendations']['conservative_set']
        elif feature_set == 'balanced':
            self.selected_features = analysis['recommendations']['balanced_set']
        elif feature_set == 'comprehensive':
            self.selected_features = analysis['recommendations']['comprehensive_set']
        else:
            raise ValueError("feature_set must be 'conservative', 'balanced', or 'comprehensive'")
        
        print(f"🎯 Initializing Optimized ML Ensemble")
        print(f"   Feature Set: {feature_set.upper()}")
        print(f"   Selected Features: {len(self.selected_features)}")
        print(f"   Features: {', '.join(self.selected_features[:5])}...")
        
        # Initialize base ML ensemble for feature extraction
        self.base_ml = MLEnsemble(use_enhanced_features=True)
        
        # Initialize models
        self.rf_model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        )
        self.ridge_model = Ridge(alpha=1.0, random_state=42)
        
        # Model performance tracking
        self.performance_history = {}
    
    def extract_optimized_features(self, prices: pd.Series, volumes: pd.Series = None) -> np.ndarray:
        """
        Extract only the statistically validated features
        
        Args:
            prices: Price series
            volumes: Volume series (optional)
            
        Returns:
            Array of selected features only
        """
        # Extract all features using base ML ensemble
        all_features = self.base_ml.extract_ml_features(prices, volumes, use_last_point=True)
        
        # Get feature names from base ensemble
        feature_names = (
            ['momentum', 'volatility', 'rsi', 'price_position', 'sma_ratio', 'return_ratio'] +
            ['gold_equity_corr', 'aud_gold_corr', 'vix_rates_corr', 'equity_bonds_corr', 
             'cross_asset_dispersion', 'regime_confidence', 'regime_stability'] +
            ['macd_signal', 'macd_histogram', 'macd_v_signal', 'macd_v_histogram',
             'macd_strength', 'macd_v_strength', 'volatility_level', 'volatility_regime',
             'macd_divergence', 'trend_consistency', 'macd_v_consistency', 
             'signal_quality', 'volatility_adjusted_momentum'] +
            ['volume_ratio', 'price_volume_correlation', 'money_flow_index', 'ad_trend',
             'obv_trend', 'volume_pressure', 'demand_strength', 'supply_pressure',
             'volume_confirmation', 'buying_pressure', 'selling_pressure',
             'demand_supply_balance', 'volume_trend_strength', 'price_volume_efficiency']
        )
        
        # Create feature mapping
        feature_dict = dict(zip(feature_names, all_features[0]))
        
        # Extract only selected features
        optimized_features = []
        for feature in self.selected_features:
            if feature in feature_dict:
                optimized_features.append(feature_dict[feature])
            else:
                print(f"⚠️ Feature {feature} not found, using 0.0")
                optimized_features.append(0.0)
        
        return np.array([optimized_features])
    
    def train_optimized_models(self, price_data: dict, sample_size: int = 100):
        """
        Train optimized models using only validated features
        
        Args:
            price_data: Dict of {ticker: price_data}
            sample_size: Number of ETFs to sample for training
        """
        print(f"🔧 Training Optimized Models with {len(self.selected_features)} features")
        
        # Sample ETFs for training
        tickers = list(price_data.keys())
        if len(tickers) > sample_size:
            tickers = np.random.choice(tickers, sample_size, replace=False)
        
        # Extract features and targets
        X_features = []
        y_targets = []
        
        print(f"📊 Extracting optimized features from {len(tickers)} ETFs...")
        for i, ticker in enumerate(tickers):
            try:
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i+1}/{len(tickers)}")
                
                etf_data = price_data[ticker]
                prices = extract_column(etf_data, 'Close')
                volumes = extract_column(etf_data, 'Volume') if 'Volume' in etf_data.columns else None
                
                if len(prices) < 100:
                    continue
                
                # Extract optimized features
                features = self.extract_optimized_features(prices, volumes)
                
                # Calculate target (60-day forward return)
                target = (prices.iloc[-1] / prices.iloc[-60] - 1) if len(prices) > 60 else 0
                
                X_features.append(features[0])
                y_targets.append(target)
                
            except Exception as e:
                print(f"⚠️ Feature extraction failed for {ticker}: {e}")
                continue
        
        if len(X_features) < 30:
            print(f"❌ Insufficient training data: {len(X_features)} < 30")
            return False
        
        # Convert to arrays
        X = np.array(X_features)
        y = np.array(y_targets)
        
        # Clean data
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        
        print(f"✅ Training data prepared: {len(X)} samples, {len(X[0])} features")
        
        # Train models
        print(f"🤖 Training Random Forest...")
        start_time = time.time()
        self.rf_model.fit(X, y)
        rf_time = time.time() - start_time
        
        print(f"🤖 Training Ridge Regression...")
        start_time = time.time()
        self.ridge_model.fit(X, y)
        ridge_time = time.time() - start_time
        
        # Evaluate models with cross-validation
        print(f"📈 Evaluating model performance...")
        tscv = TimeSeriesSplit(n_splits=5)
        
        rf_scores = cross_val_score(self.rf_model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        ridge_scores = cross_val_score(self.ridge_model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        
        # Calculate performance metrics
        rf_mae = -rf_scores.mean()
        ridge_mae = -ridge_scores.mean()
        
        # Baseline performance
        baseline_mae = mean_absolute_error(y, [np.mean(y)] * len(y))
        
        # Store performance
        self.performance_history = {
            'feature_set': self.feature_set,
            'num_features': len(self.selected_features),
            'training_samples': len(X),
            'rf_mae': rf_mae,
            'ridge_mae': ridge_mae,
            'baseline_mae': baseline_mae,
            'rf_improvement': (baseline_mae - rf_mae) / baseline_mae * 100,
            'ridge_improvement': (baseline_mae - ridge_mae) / baseline_mae * 100,
            'training_time_rf': rf_time,
            'training_time_ridge': ridge_time
        }
        
        print(f"✅ Model training completed!")
        print(f"   Random Forest MAE: {rf_mae:.4f} ({self.performance_history['rf_improvement']:.1f}% vs baseline)")
        print(f"   Ridge Regression MAE: {ridge_mae:.4f} ({self.performance_history['ridge_improvement']:.1f}% vs baseline)")
        print(f"   Baseline MAE: {baseline_mae:.4f}")
        
        return True
    
    def predict_optimized(self, prices: pd.Series, volumes: pd.Series = None) -> dict:
        """
        Make optimized prediction using trained models
        
        Args:
            prices: Price series
            volumes: Volume series (optional)
            
        Returns:
            Dict with prediction results
        """
        if not hasattr(self.rf_model, 'feature_importances_'):
            print("❌ Models not trained yet")
            return {}
        
        # Extract optimized features
        features = self.extract_optimized_features(prices, volumes)
        
        # Make predictions
        rf_pred = self.rf_model.predict(features)[0]
        ridge_pred = self.ridge_model.predict(features)[0]
        
        # Ensemble prediction (simple average)
        ensemble_pred = (rf_pred + ridge_pred) / 2
        
        # Calculate confidence based on model agreement
        model_disagreement = abs(rf_pred - ridge_pred)
        confidence = max(0, 1 - model_disagreement / 0.1)  # Normalize to [0,1]
        
        # Clip to reasonable range
        ensemble_pred = np.clip(ensemble_pred, -self.max_forecast_range, self.max_forecast_range)
        
        return {
            'prediction': ensemble_pred,
            'rf_prediction': rf_pred,
            'ridge_prediction': ridge_pred,
            'confidence': confidence,
            'feature_set': self.feature_set,
            'num_features': len(self.selected_features)
        }


def compare_optimized_models(price_data: dict):
    """
    Compare performance of different optimized feature sets
    
    Args:
        price_data: Dict of {ticker: price_data}
    """
    print("🔬 COMPARING OPTIMIZED MODEL CONFIGURATIONS")
    print("=" * 50)
    
    feature_sets = ['conservative', 'balanced', 'comprehensive']
    results = {}
    
    for feature_set in feature_sets:
        print(f"\n🎯 Testing {feature_set.upper()} feature set...")
        
        # Initialize optimized ensemble
        opt_ml = OptimizedMLEnsemble(feature_set=feature_set)
        
        # Train models
        success = opt_ml.train_optimized_models(price_data, sample_size=80)
        
        if success:
            results[feature_set] = opt_ml.performance_history
            print(f"✅ {feature_set.upper()}: {opt_ml.performance_history['rf_improvement']:.1f}% RF improvement")
        else:
            print(f"❌ {feature_set.upper()}: Training failed")
    
    # Compare results
    if results:
        print(f"\n📊 MODEL COMPARISON SUMMARY:")
        print("-" * 60)
        print(f"{'Feature Set':<15} {'Features':<10} {'RF MAE':<10} {'RF Improv':<10} {'Ridge Improv':<12}")
        print("-" * 60)
        
        for feature_set, perf in results.items():
            print(f"{feature_set.upper():<15} {perf['num_features']:<10} {perf['rf_mae']:<10.4f} "
                  f"{perf['rf_improvement']:<10.1f}% {perf['ridge_improvement']:<12.1f}%")
        
        # Find best performing model
        best_rf = max(results.items(), key=lambda x: x[1]['rf_improvement'])
        best_ridge = max(results.items(), key=lambda x: x[1]['ridge_improvement'])
        
        print(f"\n🏆 BEST PERFORMING MODELS:")
        print(f"   Random Forest: {best_rf[0].upper()} ({best_rf[1]['rf_improvement']:.1f}% improvement)")
        print(f"   Ridge Regression: {best_ridge[0].upper()} ({best_ridge[1]['ridge_improvement']:.1f}% improvement)")
        
        # Save comparison results
        with open('data/optimized_model_comparison.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Comparison results saved to data/optimized_model_comparison.json")
        
        return results
    
    return None


if __name__ == "__main__":
    # Load price data
    print("📊 Loading price data for optimization...")
    
    # Load ETF universe and historical data
    from data_manager.data_manager import ETFDataManager
    import os
    
    data_manager = ETFDataManager()
    universe_df = data_manager.load_universe()
    all_tickers = universe_df['ticker'].tolist()
    
    # Load price data
    price_data = {}
    loaded_count = 0
    
    for ticker in all_tickers:
        try:
            # Load historical price data
            possible_files = [
                f"data/historical/{ticker.replace('.AX', '_AX')}.parquet",
                f"data/historical/{ticker}.parquet"
            ]
            
            hist_data = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    hist_data = pd.read_parquet(file_path)
                    break
            
            if hist_data is not None and len(hist_data) >= 100:
                if 'Close' in hist_data.columns:
                    price_data[ticker] = hist_data
                    loaded_count += 1
                    
        except Exception as e:
            continue
        
        if loaded_count >= 150:  # Limit for performance
            break
    
    print(f"✅ Loaded {loaded_count} ETFs for optimization")
    
    if len(price_data) >= 50:
        # Compare optimized models
        results = compare_optimized_models(price_data)
        
        if results:
            print(f"\n🎯 STEP 3 COMPLETE - Ready for Step 4: Comprehensive Validation Report")
        else:
            print(f"\n❌ STEP 3 FAILED - Model optimization failed")
    else:
        print(f"❌ Insufficient data: {len(price_data)} < 50")
