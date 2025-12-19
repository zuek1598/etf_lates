#!/usr/bin/env python3
"""
Feature Validation Framework
Tests ML features for predictive power before implementation
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from data_manager.data_manager import ETFDataManager
from analyzers.ml_ensemble_production import MLEnsembleProduction
import yfinance as yf

class FeatureValidator:
    """Validates ML features for predictive power"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.data_manager = ETFDataManager(data_dir)
        self.ml_ensemble = MLEnsembleProduction()
        
    def download_test_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download test data for validation"""
        print(f"Downloading data for {len(tickers)} ETFs...")
        
        all_data = {}
        chunk_size = 50
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            try:
                data = yf.download(chunk, start=start_date, end=end_date, progress=False)
                if not data.empty and 'Close' in data.columns:
                    if isinstance(data.columns, pd.MultiIndex):
                        close_data = data['Close']
                        for ticker in close_data.columns:
                            if not close_data[ticker].isnull().all():
                                all_data[ticker] = close_data[ticker]
                    else:
                        all_data.update(data['Close'].to_dict())
            except Exception as e:
                print(f"    Warning: Failed chunk {i//chunk_size + 1}: {e}")
        
        return pd.DataFrame(all_data)
    
    def extract_feature_series(self, prices: pd.Series, feature_name: str) -> pd.Series:
        """Extract time series for a specific feature"""
        volumes = pd.Series(0, index=prices.index)  # Dummy volume
        
        # Create ETF data frame
        etf_data = pd.DataFrame({'Close': prices, 'Volume': volumes})
        
        # Get all features
        X = self.ml_ensemble._extract_production_features(prices, volumes)
        
        # Map feature index to name
        feature_map = {i: name for i, name in enumerate(self.ml_ensemble.production_features)}
        
        # Create series for the requested feature
        feature_values = []
        dates = []
        
        # Rolling window extraction
        for i in range(60, len(prices)):  # Need at least 60 days for features
            window_prices = prices.iloc[:i+1]
            window_volumes = volumes.iloc[:i+1]
            
            try:
                X_window = self.ml_ensemble._extract_production_features(window_prices, window_volumes)
                feature_idx = None
                
                for idx, name in feature_map.items():
                    if name == feature_name:
                        feature_idx = idx
                        break
                
                if feature_idx is not None and len(X_window[0]) > feature_idx:
                    feature_values.append(X_window[0][feature_idx])
                    dates.append(prices.index[i])
            except:
                continue
        
        return pd.Series(feature_values, index=dates)
    
    def calculate_feature_predictive_power(self, feature_series: pd.Series, 
                                         returns: pd.Series, 
                                         lag_days: int = 20) -> Dict:
        """Calculate predictive power metrics for a feature"""
        # Align feature and returns
        common_dates = feature_series.index.intersection(returns.index)
        if len(common_dates) < 100:
            return {'correlation': 0, 'p_value': 1, 'hit_rate': 0.5, 'valid': False}
        
        feature_aligned = feature_series.loc[common_dates]
        returns_aligned = returns.loc[common_dates]
        
        # Calculate forward returns
        forward_returns = returns_aligned.shift(-lag_days).dropna()
        feature_clean = feature_aligned.loc[forward_returns.index].dropna()
        
        if len(feature_clean) < 50:
            return {'correlation': 0, 'p_value': 1, 'hit_rate': 0.5, 'valid': False}
        
        # 1. Correlation analysis
        correlation, p_value = stats.pearsonr(feature_clean, forward_returns)
        
        # 2. Hit rate (directional accuracy)
        feature_direction = np.sign(feature_clean)
        return_direction = np.sign(forward_returns)
        hit_rate = np.mean(feature_direction == return_direction)
        
        # 3. Information coefficient
        ic = correlation
        
        # 4. Regression R-squared
        if len(feature_clean.unique()) > 1:  # Check for variance
            slope, intercept, r_value, r_p_value, std_err = stats.linregress(feature_clean, forward_returns)
            r_squared = r_value ** 2
        else:
            slope = 0
            r_squared = 0
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'hit_rate': hit_rate,
            'information_coefficient': ic,
            'r_squared': r_squared,
            'valid': True,
            'samples': len(feature_clean)
        }
    
    def validate_all_features(self, test_tickers: List[str] = None) -> pd.DataFrame:
        """Validate all ML features across multiple ETFs"""
        print("=" * 80)
        print("FEATURE VALIDATION FRAMEWORK")
        print("=" * 80)
        
        if test_tickers is None:
            # Use a diverse sample of ETFs
            test_tickers = ['VAS.AX', 'VGS.AX', 'IOZ.AX', 'MVB.AX', 'WAA.AX', 
                          'GOLD.AX', 'ETPMAG.AX', 'CURE.AX', 'TECH.AX', 'FANG.AX']
        
        # Download data
        price_data = self.download_test_data(test_tickers, '2020-01-01', '2024-12-31')
        
        results = []
        
        for feature in self.ml_ensemble.production_features:
            print(f"\nValidating feature: {feature}")
            feature_results = []
            
            for ticker in test_tickers:
                if ticker not in price_data.columns:
                    continue
                
                prices = price_data[ticker].dropna()
                if len(prices) < 300:
                    continue
                
                returns = prices.pct_change().dropna()
                
                # Extract feature series
                feature_series = self.extract_feature_series(prices, feature)
                
                # Calculate predictive power
                metrics = self.calculate_feature_predictive_power(feature_series, returns)
                
                if metrics['valid']:
                    feature_results.append({
                        'ticker': ticker,
                        'correlation': metrics['correlation'],
                        'p_value': metrics['p_value'],
                        'hit_rate': metrics['hit_rate'],
                        'r_squared': metrics['r_squared']
                    })
            
            # Aggregate results
            if feature_results:
                df_feature = pd.DataFrame(feature_results)
                
                avg_correlation = df_feature['correlation'].mean()
                avg_hit_rate = df_feature['hit_rate'].mean()
                avg_r_squared = df_feature['r_squared'].mean()
                significant = (df_feature['p_value'] < 0.05).mean()
                
                results.append({
                    'feature': feature,
                    'avg_correlation': avg_correlation,
                    'avg_hit_rate': avg_hit_rate,
                    'avg_r_squared': avg_r_squared,
                    'significant_ratio': significant,
                    'num_etfs': len(df_feature)
                })
                
                print(f"  Avg Correlation: {avg_correlation:.4f}")
                print(f"  Avg Hit Rate: {avg_hit_rate:.2%}")
                print(f"  Avg R²: {avg_r_squared:.4f}")
                print(f"  Significant (p<0.05): {significant:.1%}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            print("\n" + "=" * 80)
            print("FEATURE VALIDATION SUMMARY")
            print("=" * 80)
            print("\nFeatures sorted by predictive power (correlation):")
            print(results_df.sort_values('avg_correlation', ascending=False).to_string(index=False))
            
            # Identify significant features
            significant_features = results_df[results_df['significant_ratio'] > 0.5]
            print(f"\nSignificant features (p<0.05 in >50% of ETFs):")
            if not significant_features.empty:
                print(significant_features[['feature', 'avg_correlation', 'avg_hit_rate']].to_string(index=False))
            else:
                print("  None!")
        
        return results_df
    
    def test_momentum_vs_features(self, test_tickers: List[str] = None) -> Dict:
        """Compare momentum to other features"""
        print("\n" + "=" * 80)
        print("MOMENTUM VS OTHER FEATURES ANALYSIS")
        print("=" * 80)
        
        if test_tickers is None:
            test_tickers = ['VAS.AX', 'VGS.AX', 'IOZ.AX', 'MVB.AX', 'WAA.AX']
        
        price_data = self.download_test_data(test_tickers, '2020-01-01', '2024-12-31')
        
        momentum_results = []
        feature_results = []
        
        for ticker in test_tickers:
            if ticker not in price_data.columns:
                continue
            
            prices = price_data[ticker].dropna()
            returns = prices.pct_change().dropna()
            
            # Calculate momentum (20-day)
            momentum = prices.pct_change(20).dropna()
            forward_returns = returns.shift(-20).dropna()
            
            # Momentum predictive power
            common_dates = momentum.index.intersection(forward_returns.index)
            if len(common_dates) > 50:
                momentum_clean = momentum.loc[common_dates].dropna()
                forward_clean = forward_returns.loc[common_dates].dropna()
                
                corr, p_val = stats.pearsonr(momentum_clean, forward_clean)
                hit_rate = np.mean(np.sign(momentum_clean) == np.sign(forward_clean))
                
                momentum_results.append({
                    'ticker': ticker,
                    'correlation': corr,
                    'hit_rate': hit_rate
                })
        
        # Compare with best ML feature
        print("\nMomentum Performance:")
        momentum_df = pd.DataFrame(momentum_results)
        if not momentum_df.empty:
            print(f"  Avg Correlation: {momentum_df['correlation'].mean():.4f}")
            print(f"  Avg Hit Rate: {momentum_df['hit_rate'].mean():.2%}")
        
        return {'momentum': momentum_df}


def main():
    """Run feature validation"""
    validator = FeatureValidator()
    
    # Validate all features
    results = validator.validate_all_features()
    
    # Compare with momentum
    momentum_comparison = validator.test_momentum_vs_features()
    
    # Save results
    if not results.empty:
        results.to_csv('data/feature_validation_results.csv', index=False)
        print(f"\nResults saved to: data/feature_validation_results.csv")


if __name__ == "__main__":
    main()
