#!/usr/bin/env python3
"""
CRITICAL INVESTIGATION - RIGOROUS STATISTICAL VALIDATION
Addressing red flags: data snooping bias, false significance, insufficient sample size
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from analysis.statistical_feature_validator import StatisticalFeatureValidator
from analyzers.ml_ensemble import MLEnsemble
from utilities.shared_utils import extract_column
import time
import warnings
warnings.filterwarnings('ignore')

class RigorousFeatureValidator:
    """
    Rigorous statistical validation addressing methodological flaws
    """
    
    def __init__(self):
        self.significance_level = 0.05
        self.min_samples_per_feature = 15  # Strict requirement
        self.cv_threshold = 0.02  # Require positive CV performance
        
    def load_full_dataset(self):
        """Load full 375 ETF dataset for proper validation"""
        print("🔍 LOADING FULL DATASET FOR RIGOROUS VALIDATION")
        print("=" * 60)
        
        # Load ETF universe and historical data
        from data_manager.data_manager import ETFDataManager
        import os
        
        data_manager = ETFDataManager()
        universe_df = data_manager.load_universe()
        all_tickers = universe_df['ticker'].tolist()
        
        # Load all available price data
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
        
        print(f"✅ Loaded {loaded_count} ETFs with sufficient data")
        return price_data
    
    def extract_features_with_leakage_check(self, price_data, sample_size=None):
        """
        Extract features with careful data leakage prevention
        """
        print(f"\n🔬 EXTRACTING FEATURES WITH LEAKAGE PREVENTION")
        print("-" * 50)
        
        # Initialize ML ensemble for feature extraction
        ml_ensemble = MLEnsemble(use_enhanced_features=True)
        
        # Get feature names
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
        
        # Sample ETFs if specified
        tickers = list(price_data.keys())
        if sample_size and len(tickers) > sample_size:
            np.random.seed(42)  # For reproducibility
            tickers = np.random.choice(tickers, sample_size, replace=False)
        
        print(f"📊 Processing {len(tickers)} ETFs for feature extraction...")
        
        all_features = []
        all_targets = []
        valid_tickers = []
        
        for i, ticker in enumerate(tickers):
            try:
                if (i + 1) % 50 == 0:
                    print(f"  Progress: {i+1}/{len(tickers)} ({(i+1)/len(tickers)*100:.1f}%)")
                
                etf_data = price_data[ticker]
                prices = extract_column(etf_data, 'Close')
                volumes = extract_column(etf_data, 'Volume') if 'Volume' in etf_data.columns else None
                
                if len(prices) < 120:  # Need enough data for 60-day target
                    continue
                
                # Extract features (using last point to avoid look-ahead bias)
                features = ml_ensemble.extract_ml_features(prices, volumes, use_last_point=True)
                
                # Calculate target (60-day forward return) - ensure no leakage
                # Use last 60 days for target, earlier data for features
                if len(prices) >= 120:
                    feature_prices = prices.iloc[:-60]
                    target_prices = prices.iloc[-60:]
                    
                    # Re-extract features using only historical data
                    features_hist = ml_ensemble.extract_ml_features(feature_prices, volumes.iloc[:-60] if volumes is not None else None, use_last_point=True)
                    
                    # Calculate actual 60-day forward return
                    target = (target_prices.iloc[-1] / target_prices.iloc[0] - 1)
                    
                    all_features.append(features_hist[0])
                    all_targets.append(target)
                    valid_tickers.append(ticker)
                
            except Exception as e:
                continue
        
        print(f"✅ Extracted features from {len(all_features)} ETFs")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features, columns=feature_names)
        targets_series = pd.Series(all_targets)
        
        # Clean data rigorously
        print(f"🧹 Rigorous data cleaning...")
        
        # Replace infinity and extreme values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        targets_series = targets_series.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with any NaN
        nan_mask = features_df.isna().any(axis=1) | targets_series.isna()
        if nan_mask.any():
            print(f"  Removing {nan_mask.sum()} rows with NaN/infinity values")
            features_df = features_df[~nan_mask]
            targets_series = targets_series[~nan_mask]
        
        # Clip extreme values (more aggressive)
        for col in features_df.columns:
            q1, q99 = features_df[col].quantile([0.01, 0.99])
            features_df[col] = features_df[col].clip(q1, q99)
        
        # Final NaN check
        features_df = features_df.fillna(0)
        targets_series = targets_series.fillna(0)
        
        print(f"✅ Clean data: {len(features_df)} samples, {len(features_df.columns)} features")
        
        return features_df, targets_series, valid_tickers
    
    def rigorous_validation(self, features_df, targets_series):
        """
        Apply rigorous statistical validation criteria
        """
        print(f"\n🔍 RIGOROUS STATISTICAL VALIDATION")
        print("-" * 50)
        
        validation_results = []
        
        # Check sample size sufficiency
        num_samples, num_features = features_df.shape
        samples_per_feature = num_samples / num_features
        
        print(f"📊 Sample Analysis:")
        print(f"   Samples: {num_samples}")
        print(f"   Features: {num_features}")
        print(f"   Samples per feature: {samples_per_feature:.1f}")
        print(f"   Required: {self.min_samples_per_feature}+")
        
        if samples_per_feature < self.min_samples_per_feature:
            print(f"⚠️ WARNING: Insufficient samples per feature!")
        
        for i, feature_name in enumerate(features_df.columns):
            feature_values = features_df[feature_name].values
            
            # Skip if feature has zero variance
            if np.var(feature_values) == 0:
                continue
            
            # 1. Correlation Analysis (strict)
            try:
                pearson_corr, pearson_p = pearsonr(feature_values, targets_series)
                spearman_corr, spearman_p = spearmanr(feature_values, targets_series)
                
                # Use maximum correlation, but require significance
                max_corr = max(abs(pearson_corr), abs(spearman_corr))
                min_p = min(pearson_p, spearman_p)
                corr_significant = (min_p < self.significance_level and max_corr > 0.1)
                
            except:
                pearson_corr, spearman_corr, min_p = 0, 0, 1
                corr_significant = False
            
            # 2. Cross-Validation Performance (CRITICAL TEST)
            try:
                # Time series cross-validation to prevent leakage
                tscv = TimeSeriesSplit(n_splits=5)
                
                # Test with Random Forest
                rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                cv_scores = cross_val_score(rf, feature_values.reshape(-1, 1), targets_series, 
                                          cv=tscv, scoring='neg_mean_absolute_error')
                
                # Baseline score (predict mean)
                baseline_mae = mean_absolute_error(targets_series, [np.mean(targets_series)] * len(targets_series))
                feature_mae = -cv_scores.mean()
                
                cv_improvement = (baseline_mae - feature_mae) / baseline_mae
                cv_predictive = (cv_improvement > self.cv_threshold)
                
            except:
                cv_improvement = -0.1
                cv_predictive = False
            
            # 3. Permutation Importance (secondary test)
            try:
                rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                rf.fit(features_df[[feature_name]], targets_series)
                baseline_score = rf.score(features_df[[feature_name]], targets_series)
                
                # Permutation test
                perm_scores = []
                for _ in range(10):
                    shuffled_targets = np.random.permutation(targets_series)
                    perm_score = rf.score(features_df[[feature_name]], shuffled_targets)
                    perm_scores.append(perm_score)
                
                importance = baseline_score - np.mean(perm_scores)
                perm_significant = (importance > 0.01)
                
            except:
                importance = 0
                perm_significant = False
            
            # 4. RIGOROUS SIGNIFICANCE CRITERIA
            # MUST have positive CV performance AND at least one other test
            overall_significant = cv_predictive and (corr_significant or perm_significant)
            
            # Calculate composite score (weighted toward CV)
            validation_score = (
                0.5 * max(0, cv_improvement) +  # CV performance (50% weight)
                0.3 * max(0, max_corr) +       # Correlation (30% weight)  
                0.2 * max(0, importance)       # Permutation (20% weight)
            )
            
            validation_results.append({
                'feature': feature_name,
                'pearson_corr': pearson_corr,
                'spearman_corr': spearman_corr,
                'max_correlation': max_corr,
                'corr_p_value': min_p,
                'corr_significant': corr_significant,
                'cv_improvement': cv_improvement,
                'cv_predictive': cv_predictive,
                'importance': importance,
                'perm_significant': perm_significant,
                'validation_score': validation_score,
                'overall_significant': overall_significant
            })
        
        return pd.DataFrame(validation_results)
    
    def analyze_multicollinearity(self, features_df, validation_results):
        """
        Analyze feature multicollinearity and independence
        """
        print(f"\n🔍 MULTICOLLINEARITY ANALYSIS")
        print("-" * 40)
        
        # Calculate correlation matrix
        corr_matrix = features_df.corr().abs()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        threshold = 0.7
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        print(f"📊 Multicollinearity Results:")
        print(f"   High correlation pairs (>0.7): {len(high_corr_pairs)}")
        
        # Group features by correlation clusters
        significant_features = validation_results[validation_results['overall_significant']]
        
        if len(high_corr_pairs) > 0:
            print(f"\n⚠️ HIGHLY CORRELATED FEATURE PAIRS:")
            for pair in high_corr_pairs[:10]:  # Show top 10
                f1_significant = pair['feature1'] in significant_features['feature'].values
                f2_significant = pair['feature2'] in significant_features['feature'].values
                sig_markers = "✓✓" if (f1_significant and f2_significant) else ("✓✗" if f1_significant else "✗✓")
                print(f"   {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f} ({sig_markers})")
        
        return high_corr_pairs
    
    def investigate_regime_features(self, features_df, validation_results):
        """
        Investigate regime features with zero permutation importance
        """
        print(f"\n🔍 REGIME FEATURES INVESTIGATION")
        print("-" * 40)
        
        regime_features = ['gold_equity_corr', 'aud_gold_corr', 'vix_rates_corr', 
                          'equity_bonds_corr', 'cross_asset_dispersion']
        
        print(f"📊 Regime Feature Analysis:")
        
        for feature in regime_features:
            if feature in validation_results['feature'].values:
                result = validation_results[validation_results['feature'] == feature].iloc[0]
                
                print(f"\n   📈 {feature}:")
                print(f"      Correlation: {result['max_correlation']:.3f} (p={result['corr_p_value']:.4f})")
                print(f"      CV Improvement: {result['cv_improvement']:.1%}")
                print(f"      Permutation Importance: {result['importance']:.4f}")
                print(f"      Overall Significant: {'✓' if result['overall_significant'] else '✗'}")
                
                # Test linear vs non-linear performance
                try:
                    feature_values = features_df[feature].values.reshape(-1, 1)
                    targets = targets_series  # This needs to be passed in or stored
                    
                    # Linear model (Ridge)
                    ridge = Ridge(alpha=1.0)
                    ridge_scores = cross_val_score(ridge, feature_values, targets, 
                                                 cv=TimeSeriesSplit(n_splits=5), 
                                                 scoring='neg_mean_absolute_error')
                    ridge_improvement = (mean_absolute_error(targets, [np.mean(targets)] * len(targets)) 
                                       - (-ridge_scores.mean())) / mean_absolute_error(targets, [np.mean(targets)] * len(targets))
                    
                    # Tree model (Random Forest)
                    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                    rf_scores = cross_val_score(rf, feature_values, targets,
                                              cv=TimeSeriesSplit(n_splits=5),
                                              scoring='neg_mean_absolute_error')
                    rf_improvement = (mean_absolute_error(targets, [np.mean(targets)] * len(targets))
                                    - (-rf_scores.mean())) / mean_absolute_error(targets, [np.mean(targets)] * len(targets))
                    
                    print(f"      Ridge Improvement: {ridge_improvement:.1%}")
                    print(f"      RF Improvement: {rf_improvement:.1%}")
                    
                    if ridge_improvement > rf_improvement * 1.5:
                        print(f"      → Works better in linear models")
                    elif rf_improvement > ridge_improvement * 1.5:
                        print(f"      → Works better in tree models")
                    else:
                        print(f"      → Similar performance in both")
                        
                except Exception as e:
                    print(f"      → Model comparison failed: {e}")
    
    def generate_corrected_report(self, validation_results, high_corr_pairs):
        """
        Generate corrected validation report with rigorous standards
        """
        print(f"\n📋 CORRECTED VALIDATION REPORT")
        print("=" * 60)
        
        # Summary statistics
        total_features = len(validation_results)
        truly_significant = validation_results['overall_significant'].sum()
        significance_rate = truly_significant / total_features
        
        print(f"🎯 RIGOROUS VALIDATION SUMMARY:")
        print(f"   Total Features Tested: {total_features}")
        print(f"   Truly Significant: {truly_significant} ({significance_rate:.1%})")
        print(f"   Significance Criteria: CV > 2% AND (Corr > 0.1 OR Perm > 0.01)")
        
        # Compare with original flawed results
        print(f"\n⚖️ COMPARISON WITH ORIGINAL VALIDATION:")
        print(f"   Original Significance Rate: 95.0% (38/40)")
        print(f"   Corrected Significance Rate: {significance_rate:.1%} ({truly_significant}/{total_features})")
        print(f"   Reduction in Significance: {(95.0 - significance_rate*100):.1f} percentage points")
        
        # Top truly significant features
        significant_features = validation_results[validation_results['overall_significant']].sort_values('validation_score', ascending=False)
        
        if len(significant_features) > 0:
            print(f"\n🏆 TRULY SIGNIFICANT FEATURES ({len(significant_features)}):")
            print("-" * 70)
            print(f"{'Rank':<5} {'Feature':<25} {'CV Improv':<10} {'Correlation':<12} {'Score':<8}")
            print("-" * 70)
            
            for i, (_, row) in enumerate(significant_features.head(10).iterrows()):
                print(f"{i+1:<5} {row['feature']:<25} {row['cv_improvement']:<10.1%} "
                      f"{row['max_correlation']:<12.3f} {row['validation_score']:<8.3f}")
        else:
            print(f"\n❌ NO FEATURES PASSED RIGOROUS VALIDATION!")
            print(f"   This suggests the original indicators may not have genuine predictive power")
        
        # Multicollinearity impact
        print(f"\n🔗 MULTICOLLINEARITY IMPACT:")
        print(f"   High correlation pairs: {len(high_corr_pairs)}")
        if len(high_corr_pairs) > 0:
            print(f"   Estimated independent signals: ~{max(4, total_features - len(high_corr_pairs))}")
        
        # Recommendations
        print(f"\n💡 CORRECTED RECOMMENDATIONS:")
        if len(significant_features) >= 4:
            print(f"   ✅ Use {len(significant_features)} truly significant features")
            print(f"   ✅ Expected genuine improvement (not inflated)")
            print(f"   ✅ Reduced overfitting risk")
        else:
            print(f"   ⚠️ Consider feature engineering - current indicators may be weak")
            print(f"   ⚠️ Focus on simple, robust features (price momentum, volatility)")
            print(f"   ⚠️ May need different approach or more data")
        
        return {
            'total_features': total_features,
            'truly_significant': truly_significant,
            'significance_rate': significance_rate,
            'significant_features': significant_features['feature'].tolist() if len(significant_features) > 0 else [],
            'high_corr_pairs': high_corr_pairs,
            'validation_results': validation_results.to_dict('records')
        }


def main():
    """Main investigation function"""
    print("🔍 CRITICAL STATISTICAL INVESTIGATION")
    print("Addressing methodological flaws in original validation")
    print("=" * 60)
    
    validator = RigorousFeatureValidator()
    
    # Load full dataset
    price_data = validator.load_full_dataset()
    
    if len(price_data) < 100:
        print("❌ Insufficient data for rigorous validation")
        return None
    
    # Extract features with leakage prevention
    features_df, targets_series, valid_tickers = validator.extract_features_with_leakage_check(price_data)
    
    # Store targets for regime feature analysis
    globals()['targets_series'] = targets_series
    
    if len(features_df) < 50:
        print("❌ Insufficient clean samples for validation")
        return None
    
    # Apply rigorous validation
    validation_results = validator.rigorous_validation(features_df, targets_series)
    
    # Analyze multicollinearity
    high_corr_pairs = validator.analyze_multicollinearity(features_df, validation_results)
    
    # Investigate regime features
    validator.investigate_regime_features(features_df, validation_results)
    
    # Generate corrected report
    corrected_results = validator.generate_corrected_report(validation_results, high_corr_pairs)
    
    # Save corrected results
    with open('data/rigorous_validation_results.json', 'w') as f:
        json.dump(corrected_results, f, indent=2, default=str)
    
    print(f"\n💾 Rigorous validation results saved to data/rigorous_validation_results.json")
    
    return corrected_results

if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n🎯 RIGOROUS INVESTIGATION COMPLETE")
        print(f"📊 Original validation likely overstated significance")
        print(f"🔧 Use corrected results for production decisions")
    else:
        print(f"\n❌ Investigation failed")
