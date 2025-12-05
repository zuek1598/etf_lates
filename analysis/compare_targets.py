#!/usr/bin/env python3
"""
Comprehensive comparison between mean daily vs cumulative return targets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
# No matplotlib - console output only

class TargetComparison:
    """
    Compare mean daily vs cumulative return prediction models
    """
    
    def __init__(self, forecast_horizon: int = 60):
        self.forecast_horizon = forecast_horizon
        self.results = {}
        
    def create_comparison_data(self, n_etfs: int = 100, n_days: int = 300):
        """Create realistic comparison dataset"""
        np.random.seed(42)
        
        all_results = []
        
        for etf_idx in range(n_etfs):
            # More realistic return generation
            base_return = np.random.normal(0.0008, 0.0002)  # 0.08% daily
            volatility = np.random.uniform(0.015, 0.035)    # 1.5-3.5% daily vol
            
            # Generate returns with some structure
            returns = []
            for i in range(n_days):
                # Add some momentum and mean reversion
                if i > 5:
                    momentum = np.mean(returns[-5:]) * 0.1
                else:
                    momentum = 0
                
                daily_return = np.random.normal(base_return + momentum, volatility)
                returns.append(daily_return)
            
            returns = pd.Series(returns)
            
            # Create features with some predictive power
            features = []
            for i in range(len(returns) - self.forecast_horizon - 100):
                recent_returns = returns.iloc[i:i+100]
                
                # Calculate features
                volatility = recent_returns.std()
                momentum = recent_returns.iloc[-20:].mean() - recent_returns.iloc[-60:].mean()
                rsi = self._calculate_rsi(recent_returns)
                
                feature = [
                    volatility,                    # volatility
                    np.random.uniform(-0.5, 0.5),  # gold_equity_corr (simulated)
                    volatility / 0.05,            # volatility_level
                    abs(momentum) / 0.01,         # signal_quality
                    np.random.uniform(-0.5, 0.5), # vix_rates_corr (simulated)
                    np.random.uniform(0, 1),      # cross_asset_dispersion
                    momentum,                     # macd_histogram
                    momentum * 0.8,              # macd_signal
                    momentum,                     # momentum
                    np.random.uniform(-0.5, 0.5)  # equity_bonds_corr (simulated)
                ]
                
                # Calculate targets
                future_returns = returns.iloc[i+100:i+100+self.forecast_horizon]
                target_daily = future_returns.mean()
                target_cumulative = future_returns.sum()
                
                features.append(feature)
                all_results.append({
                    'features': feature,
                    'target_daily': target_daily,
                    'target_cumulative': target_cumulative,
                    'etf': etf_idx
                })
        
        return all_results
    
    def _calculate_rsi(self, returns, period: int = 14):
        """Simple RSI calculation"""
        gains = returns[returns > 0]
        losses = abs(returns[returns < 0])
        
        if len(gains) == 0 or len(losses) == 0:
            return 0.5
        
        avg_gain = gains.mean() if len(gains) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 1.0
        
        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))
        return rsi
    
    def compare_models(self, n_etfs: int = 50):
        """Run comprehensive comparison"""
        
        print("🔍 Comparing Mean Daily vs Cumulative Return Targets...")
        
        # Generate data
        data = self.create_comparison_data(n_etfs)
        
        # Prepare data
        X = np.array([d['features'] for d in data])
        y_daily = np.array([d['target_daily'] for d in data])
        y_cumulative = np.array([d['target_cumulative'] for d in data])
        
        # Temporal split
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_daily_train, y_daily_test = y_daily[:split_idx], y_daily[split_idx:]
        y_cumulative_train, y_cumulative_test = y_cumulative[:split_idx], y_cumulative[split_idx:]
        
        # Train models
        rf_daily = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_cumulative = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rf_daily.fit(X_train, y_daily_train)
        rf_cumulative.fit(X_train, y_cumulative_train)
        
        # Predictions
        daily_pred = rf_daily.predict(X_test)
        cumulative_pred = rf_cumulative.predict(X_test)
        
        # Calculate metrics
        daily_mae = mean_absolute_error(y_daily_test, daily_pred)
        cumulative_mae = mean_absolute_error(y_cumulative_test, cumulative_pred)
        
        daily_r2 = r2_score(y_daily_test, daily_pred)
        cumulative_r2 = r2_score(y_cumulative_test, cumulative_pred)
        
        # Model agreement
        daily_to_cumulative = daily_pred * self.forecast_horizon
        model_agreement = 1 - np.mean(np.abs(daily_to_cumulative - cumulative_pred)) / np.mean(np.abs(cumulative_pred))
        
        # Ranking correlation
        rank_corr, _ = spearmanr(cumulative_pred, daily_to_cumulative)
        
        # Feature importance comparison
        daily_importance = rf_daily.feature_importances_
        cumulative_importance = rf_cumulative.feature_importances_
        
        # Store results
        self.results = {
            'performance': {
                'mean_daily': {
                    'mae': daily_mae,
                    'r2': daily_r2,
                    'target_mean': y_daily_test.mean(),
                    'target_std': y_daily_test.std()
                },
                'cumulative': {
                    'mae': cumulative_mae,
                    'r2': cumulative_r2,
                    'target_mean': y_cumulative_test.mean(),
                    'target_std': y_cumulative_test.std()
                }
            },
            'model_agreement': {
                'correlation': model_agreement,
                'ranking_correlation': rank_corr,
                'mean_difference': np.mean(np.abs(daily_to_cumulative - cumulative_pred))
            },
            'feature_importance': {
                'daily': daily_importance,
                'cumulative': cumulative_importance,
                'correlation': pearsonr(daily_importance, cumulative_importance)[0]
            }
        }
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        
        if not self.results:
            self.compare_models()
        
        print("\n" + "=" * 80)
        print("TARGET COMPARISON REPORT")
        print("=" * 80)
        
        # Performance comparison
        daily = self.results['performance']['mean_daily']
        cumulative = self.results['performance']['cumulative']
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"   Mean Daily Model:")
        print(f"     MAE: {daily['mae']:.4f} ({daily['mae']*100:.2f}%)")
        print(f"     R²: {daily['r2']:.3f}")
        print(f"     Target range: {daily['target_mean']:.4f} ± {daily['target_std']:.4f}")
        
        print(f"   Cumulative Model:")
        print(f"     MAE: {cumulative['mae']:.4f} ({cumulative['mae']*100:.2f}%)")
        print(f"     R²: {cumulative['r2']:.3f}")
        print(f"     Target range: {cumulative['target_mean']:.4f} ± {cumulative['target_std']:.4f}")
        
        # Model agreement
        agreement = self.results['model_agreement']
        print(f"\n🤝 MODEL AGREEMENT:")
        print(f"   Prediction correlation: {agreement['correlation']:.3f}")
        print(f"   Ranking correlation: {agreement['ranking_correlation']:.3f}")
        print(f"   Mean difference: {agreement['mean_difference']:.4f}")
        
        # Feature importance
        importance = self.results['feature_importance']
        print(f"\n🔍 FEATURE IMPORTANCE CORRELATION: {importance['correlation']:.3f}")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        
        if agreement['ranking_correlation'] > 0.85:
            print("   ✅ High ranking agreement - both models suitable for ranking")
        else:
            print("   ⚠️ Low ranking agreement - prefer mean daily for ranking")
        
        if abs(agreement['correlation']) > 0.8:
            print("   ✅ High prediction agreement - models consistent")
        else:
            print("   ⚠️ Low prediction agreement - investigate differences")
        
        print(f"   📊 Use mean daily model for: ETF ranking and comparison")
        print(f"   📈 Use cumulative model for: Absolute return forecasting")
        
        return self.results

if __name__ == "__main__":
    print("=" * 80)
    print("TARGET COMPARISON ANALYSIS")
    print("=" * 80)
    
    comparator = TargetComparison()
    results = comparator.compare_models(n_etfs=50)
    comparator.generate_report()
