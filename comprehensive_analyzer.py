#!/usr/bin/env python3
"""
Comprehensive Feature Analysis
Tests different horizons and approaches to find predictive signals
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

class ComprehensiveAnalyzer:
    """Analyze features across multiple horizons and approaches"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.data_manager = ETFDataManager(data_dir)
        self.ml_ensemble = MLEnsembleProduction()
        
    def download_test_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Download test data for analysis"""
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
        volumes = pd.Series(0, index=prices.index)
        
        etf_data = pd.DataFrame({'Close': prices, 'Volume': volumes})
        
        feature_values = []
        dates = []
        
        for i in range(60, len(prices)):
            window_prices = prices.iloc[:i+1]
            window_volumes = volumes.iloc[:i+1]
            
            try:
                X_window = self.ml_ensemble._extract_production_features(window_prices, window_volumes)
                feature_map = {i: name for i, name in enumerate(self.ml_ensemble.production_features)}
                
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
    
    def test_multiple_horizons(self, test_tickers: List[str] = None) -> pd.DataFrame:
        """Test features across multiple time horizons"""
        print("=" * 80)
        print("MULTIPLE HORIZON ANALYSIS")
        print("=" * 80)
        
        if test_tickers is None:
            test_tickers = ['VAS.AX', 'VGS.AX', 'IOZ.AX', 'MVB.AX', 'WAA.AX']
        
        price_data = self.download_test_data(test_tickers, '2020-01-01', '2024-12-31')
        
        horizons = [5, 10, 20, 40, 60]  # Different forecast horizons
        results = []
        
        for horizon in horizons:
            print(f"\nTesting {horizon}-day horizon:")
            
            for feature in ['volatility', 'momentum', 'macd_signal']:
                feature_correlations = []
                
                for ticker in test_tickers:
                    if ticker not in price_data.columns:
                        continue
                    
                    prices = price_data[ticker].dropna()
                    returns = prices.pct_change().dropna()
                    
                    # Extract feature
                    if feature == 'momentum':
                        feature_series = prices.pct_change(20).dropna()
                    else:
                        feature_series = self.extract_feature_series(prices, feature)
                    
                    # Calculate forward returns
                    forward_returns = returns.shift(-horizon).dropna()
                    
                    # Align and calculate correlation
                    common_dates = feature_series.index.intersection(forward_returns.index)
                    if len(common_dates) > 50:
                        feature_clean = feature_series.loc[common_dates].dropna()
                        forward_clean = forward_returns.loc[common_dates].dropna()
                        
                        if len(feature_clean.unique()) > 1:
                            corr, p_val = stats.pearsonr(feature_clean, forward_clean)
                            feature_correlations.append(corr)
                
                if feature_correlations:
                    avg_corr = np.mean(feature_correlations)
                    significant = sum(abs(c) > 0.1 for c in feature_correlations) / len(feature_correlations)
                    
                    results.append({
                        'horizon': horizon,
                        'feature': feature,
                        'avg_correlation': avg_corr,
                        'significant_ratio': significant,
                        'num_etfs': len(feature_correlations)
                    })
                    
                    print(f"  {feature}: {avg_corr:.4f} avg correlation")
        
        results_df = pd.DataFrame(results)
        
        print("\n" + "=" * 80)
        print("HORIZON ANALYSIS SUMMARY")
        print("=" * 80)
        print(results_df.pivot(index='feature', columns='horizon', values='avg_correlation').round(4))
        
        return results_df
    
    def test_volatility_prediction(self, test_tickers: List[str] = None) -> Dict:
        """Test if features predict volatility better than returns"""
        print("\n" + "=" * 80)
        print("VOLATILITY PREDICTION ANALYSIS")
        print("=" * 80)
        
        if test_tickers is None:
            test_tickers = ['VAS.AX', 'VGS.AX', 'IOZ.AX', 'MVB.AX', 'WAA.AX']
        
        price_data = self.download_test_data(test_tickers, '2020-01-01', '2024-12-31')
        
        results = []
        
        for ticker in test_tickers:
            if ticker not in price_data.columns:
                continue
            
            prices = price_data[ticker].dropna()
            returns = prices.pct_change().dropna()
            
            # Calculate realized volatility (20-day)
            realized_vol = returns.rolling(20).std().dropna() * np.sqrt(252)
            
            # Test volatility prediction
            for feature in ['volatility', 'macd_signal']:
                if feature == 'volatility':
                    feature_series = returns.rolling(20).std().dropna()
                else:
                    feature_series = self.extract_feature_series(prices, feature)
                
                # Align with future volatility
                future_vol = realized_vol.shift(-20).dropna()
                
                common_dates = feature_series.index.intersection(future_vol.index)
                if len(common_dates) > 50:
                    feature_clean = feature_series.loc[common_dates].dropna()
                    vol_clean = future_vol.loc[common_dates].dropna()
                    
                    if len(feature_clean.unique()) > 1:
                        corr, p_val = stats.pearsonr(feature_clean, vol_clean)
                        
                        results.append({
                            'ticker': ticker,
                            'feature': feature,
                            'volatility_correlation': corr,
                            'p_value': p_val
                        })
        
        vol_df = pd.DataFrame(results)
        
        if not vol_df.empty:
            print("\nVolatility Prediction Results:")
            for feature in vol_df['feature'].unique():
                feature_data = vol_df[vol_df['feature'] == feature]
                print(f"  {feature}: {feature_data['volatility_correlation'].mean():.4f} avg correlation")
        
        return {'volatility_prediction': vol_df}
    
    def test_feature_combinations(self, test_tickers: List[str] = None) -> Dict:
        """Test combinations of features for better predictive power"""
        print("\n" + "=" * 80)
        print("FEATURE COMBINATION ANALYSIS")
        print("=" * 80)
        
        if test_tickers is None:
            test_tickers = ['VAS.AX', 'VGS.AX', 'IOZ.AX', 'MVB.AX', 'WAA.AX']
        
        price_data = self.download_test_data(test_tickers, '2020-01-01', '2024-12-31')
        
        combination_results = []
        
        for ticker in test_tickers:
            if ticker not in price_data.columns:
                continue
            
            prices = price_data[ticker].dropna()
            returns = prices.pct_change().dropna()
            
            # Extract multiple features
            features_data = {}
            for feature in ['volatility', 'macd_signal', 'signal_quality']:
                features_data[feature] = self.extract_feature_series(prices, feature)
            
            # Create feature matrix
            common_dates = returns.index
            for feature_series in features_data.values():
                common_dates = common_dates.intersection(feature_series.index)
            
            if len(common_dates) < 100:
                continue
            
            # Build feature matrix
            X = []
            y = []
            
            for date in common_dates:
                feature_row = []
                valid = True
                
                for feature in ['volatility', 'macd_signal', 'signal_quality']:
                    if date in features_data[feature].index:
                        val = features_data[feature].loc[date]
                        if not np.isnan(val):
                            feature_row.append(val)
                        else:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                
                if valid and len(feature_row) == 3:
                    # Get forward return
                    idx = returns.index.get_loc(date)
                    if idx + 20 < len(returns):
                        forward_return = returns.iloc[idx + 20]
                        X.append(feature_row)
                        y.append(forward_return)
            
            if len(X) > 50:
                X = np.array(X)
                y = np.array(y)
                
                # Test linear regression
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                
                # Individual feature correlations
                correlations = []
                for i in range(3):
                    corr, _ = stats.pearsonr(X[:, i], y)
                    correlations.append(abs(corr))
                
                combination_results.append({
                    'ticker': ticker,
                    'combined_r2': r2,
                    'max_individual_corr': max(correlations),
                    'improvement': r2 - max(correlations) ** 2
                })
        
        combo_df = pd.DataFrame(combination_results)
        
        if not combo_df.empty:
            print("\nFeature Combination Results:")
            print(f"  Average Combined R²: {combo_df['combined_r2'].mean():.4f}")
            print(f"  Average Max Individual Correlation²: {(combo_df['max_individual_corr']**2).mean():.4f}")
            print(f"  Average Improvement: {combo_df['improvement'].mean():.4f}")
        
        return {'combinations': combo_df}
    
    def analyze_market_regimes(self, test_tickers: List[str] = None) -> Dict:
        """Analyze if features work better in different market regimes"""
        print("\n" + "=" * 80)
        print("MARKET REGIME ANALYSIS")
        print("=" * 80)
        
        # Get market data for regime identification
        market_data = yf.download('^AXJO', start='2020-01-01', end='2024-12-31', progress=False)
        if isinstance(market_data.columns, pd.MultiIndex):
            market_prices = market_data['Close']['^AXJO']
        else:
            market_prices = market_data['Close']
        
        # Define regimes based on market performance
        market_returns = market_prices.pct_change().dropna()
        
        # Bull market: positive 60-day return
        bull_60d = market_returns.rolling(60).sum() > 0.05
        # Bear market: negative 60-day return  
        bear_60d = market_returns.rolling(60).sum() < -0.05
        
        if test_tickers is None:
            test_tickers = ['VAS.AX', 'VGS.AX', 'IOZ.AX']
        
        price_data = self.download_test_data(test_tickers, '2020-01-01', '2024-12-31')
        
        regime_results = []
        
        for ticker in test_tickers:
            if ticker not in price_data.columns:
                continue
            
            prices = price_data[ticker].dropna()
            returns = prices.pct_change().dropna()
            
            # Test momentum in different regimes
            momentum = prices.pct_change(20).dropna()
            forward_returns = returns.shift(-20).dropna()
            
            # Align all data
            common_dates = momentum.index.intersection(forward_returns.index)
            common_dates = common_dates.intersection(market_returns.index)
            
            if len(common_dates) < 100:
                continue
            
            for regime_name, regime_mask in [('Bull', bull_60d), ('Bear', bear_60d)]:
                regime_dates = [d for d in common_dates if d in regime_mask.index and regime_mask.loc[d]]
                
                if len(regime_dates) > 30:
                    regime_momentum = momentum.loc[regime_dates]
                    regime_forward = forward_returns.loc[regime_dates]
                    
                    corr, p_val = stats.pearsonr(regime_momentum, regime_forward)
                    hit_rate = np.mean(np.sign(regime_momentum) == np.sign(regime_forward))
                    
                    regime_results.append({
                        'ticker': ticker,
                        'regime': regime_name,
                        'correlation': corr,
                        'hit_rate': hit_rate,
                        'samples': len(regime_dates)
                    })
        
        regime_df = pd.DataFrame(regime_results)
        
        if not regime_df.empty:
            print("\nRegime-Based Performance:")
            for regime in regime_df['regime'].unique():
                regime_data = regime_df[regime_df['regime'] == regime]
                print(f"  {regime} Market:")
                print(f"    Avg Correlation: {regime_data['correlation'].mean():.4f}")
                print(f"    Avg Hit Rate: {regime_data['hit_rate'].mean():.2%}")
        
        return {'regimes': regime_df}


def main():
    """Run comprehensive analysis"""
    analyzer = ComprehensiveAnalyzer()
    
    # Test multiple horizons
    horizon_results = analyzer.test_multiple_horizons()
    
    # Test volatility prediction
    vol_results = analyzer.test_volatility_prediction()
    
    # Test feature combinations
    combo_results = analyzer.test_feature_combinations()
    
    # Test market regimes
    regime_results = analyzer.analyze_market_regimes()
    
    # Save all results
    horizon_results.to_csv('data/horizon_analysis.csv', index=False)
    print(f"\nAll results saved to data/ directory")


if __name__ == "__main__":
    main()
