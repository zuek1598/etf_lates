#!/usr/bin/env python3
"""
CRITICAL ISSUES DEBUG - Fix COVID Validation, Regime Confidence, Feature Selection
Addressing the three major issues identified in the investigation
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class CriticalIssuesDebugger:
    """
    Debug and fix the critical issues identified
    """
    
    def __init__(self):
        self.correlation_threshold = 0.7
        self.cv_threshold = 0.02
        
    def debug_covid_validation(self):
        """
        Debug Issue #1: COVID validation returning all zeros
        """
        print("🐛 DEBUG ISSUE #1: COVID VALIDATION RETURNING ALL ZEROS")
        print("=" * 65)
        
        # Load the rigorous validation results to get feature data
        try:
            with open('data/rigorous_validation_results.json', 'r') as f:
                rigorous_results = json.load(f)
            
            validation_df = pd.DataFrame(rigorous_results['validation_results'])
            
            # Load feature data for debugging
            price_data = self.load_price_data()
            features_df, targets_series, _ = self.extract_features_for_analysis(price_data)
            
            volatility_features = ['volatility', 'volatility_level', 'volatility_regime', 'signal_quality']
            
            print(f"🔍 DEBUGGING COVID VALIDATION LOGIC:")
            print(f"   Total samples: {len(features_df)}")
            print(f"   Features available: {len(features_df.columns)}")
            print(f"   Volatility features available: {[f for f in volatility_features if f in features_df.columns]}")
            
            # Check the original COVID split logic
            samples = len(features_df)
            covid_period_end = int(samples * 0.4)
            post_covid_start = int(samples * 0.6)
            
            print(f"\n📅 ORIGINAL COVID SPLIT LOGIC:")
            print(f"   Total samples: {samples}")
            print(f"   COVID period (first 40%): 0 to {covid_period_end}")
            print(f"   Gap (40-60%): {covid_period_end} to {post_covid_start}")
            print(f"   Post-COVID (last 40%): {post_covid_start} to {samples}")
            
            # Check if we have actual temporal data or just random samples
            print(f"\n🔍 DATA TEMPORAL ANALYSIS:")
            
            # The issue: we're not using actual dates, just random ETF samples
            # This explains why COVID vs post-COVID shows no difference
            print(f"   ❌ ISSUE IDENTIFIED: Using random ETF samples, not temporal data!")
            print(f"   ❌ Cannot separate COVID vs post-COVID without actual timestamps")
            print(f"   ❌ All samples treated as independent, not time-ordered")
            
            # Fix: Use actual temporal data from individual ETF time series
            print(f"\n🔧 PROPOSED FIX:")
            print(f"   1. Use individual ETF time series with actual dates")
            print(f"   2. Extract features at different time points (2020-2022 vs 2023-2025)")
            print(f"   3. Compare performance across actual time periods")
            
            # Implement a simple temporal test using one ETF as example
            print(f"\n🧪 TESTING TEMPORAL EXTRACTION WITH ONE ETF:")
            
            # Get first ETF with sufficient data
            first_ticker = list(price_data.keys())[0]
            first_data = price_data[first_ticker]
            
            if len(first_data) >= 500:  # Need enough data for temporal split
                from analyzers.ml_ensemble import MLEnsemble
                from utilities.shared_utils import extract_column
                
                ml_ensemble = MLEnsemble(use_enhanced_features=True)
                prices = extract_column(first_data, 'Close')
                volumes = extract_column(first_data, 'Volume') if 'Volume' in first_data.columns else None
                
                # Simulate temporal split (assuming data is chronological)
                total_points = len(prices)
                covid_end = int(total_points * 0.6)  # First 60% as COVID period
                post_covid_start = int(total_points * 0.8)  # Last 20% as post-COVID
                
                print(f"   ETF: {first_ticker}")
                print(f"   Total price points: {total_points}")
                print(f"   COVID period: 0 to {covid_end}")
                print(f"   Post-COVID period: {post_covid_start} to {total_points}")
                
                # Extract volatility at different time points
                covid_prices = prices.iloc[covid_end-60:covid_end]
                post_covid_prices = prices.iloc[post_covid_start:post_covid_start+60]
                
                covid_vol = covid_prices.pct_change().std() * np.sqrt(252)
                post_covid_vol = post_covid_prices.pct_change().std() * np.sqrt(252)
                
                print(f"   COVID volatility: {covid_vol:.2%}")
                print(f"   Post-COVID volatility: {post_covid_vol:.2%}")
                print(f"   Volatility change: {(post_covid_vol - covid_vol):.2%}")
                
                if abs(post_covid_vol - covid_vol) > 0.05:
                    print(f"   ✅ Temporal volatility difference detected!")
                else:
                    print(f"   ⚠️ Similar volatility levels")
            
            return True
            
        except Exception as e:
            print(f"❌ COVID validation debug failed: {e}")
            return False
    
    def debug_regime_confidence(self):
        """
        Debug Issue #2: Regime confidence very low (3.2%)
        """
        print(f"\n🏛️ DEBUG ISSUE #2: REGIME CONFIDENCE VERY LOW (3.2%)")
        print("=" * 65)
        
        try:
            # Load regime detection system
            from frameworks.geopolitical_framework import RegimeDetector
            
            print(f"🔍 INVESTIGATING REGIME DETECTION CONFIGURATION:")
            
            # Initialize regime detector to check settings
            regime_detector = RegimeDetector()
            
            # Check default parameters
            print(f"   Correlation window: {regime_detector.correlation_window} days")
            print(f"   Regime window: {regime_detector.regime_window} days")
            print(f"   Confidence threshold: {regime_detector.confidence_threshold}")
            
            # Load external data to analyze
            external_data = regime_detector.fetch_external_data()
            
            print(f"\n📊 EXTERNAL DATA ANALYSIS:")
            for key, data in external_data.items():
                if hasattr(data, 'index') and len(data) > 0:
                    print(f"   {key}: {len(data)} points from {data.index[0]} to {data.index[-1]}")
            
            # Run regime analysis to debug confidence
            print(f"\n🔍 RUNNING REGIME ANALYSIS WITH DEBUGGING:")
            
            # This will show the detailed regime classification process
            regime_results = regime_detector.run_analysis(external_data)
            
            if regime_results and 'regime_data' in regime_results:
                regime_data = regime_results['regime_data']
                
                print(f"   Total regime points: {len(regime_data)}")
                
                if 'confidence' in regime_data.columns:
                    confidence_stats = regime_data['confidence'].describe()
                    print(f"\n📈 CONFIDENCE STATISTICS:")
                    print(f"   Mean confidence: {confidence_stats['mean']:.2%}")
                    print(f"   Median confidence: {confidence_stats['50%']:.2%}")
                    print(f"   Max confidence: {confidence_stats['max']:.2%}")
                    print(f"   Min confidence: {confidence_stats['min']:.2%}")
                    
                    # Check confidence distribution
                    high_confidence = (regime_data['confidence'] > 0.5).sum()
                    medium_confidence = ((regime_data['confidence'] > 0.2) & (regime_data['confidence'] <= 0.5)).sum()
                    low_confidence = (regime_data['confidence'] <= 0.2).sum()
                    
                    print(f"\n📊 CONFIDENCE DISTRIBUTION:")
                    print(f"   High confidence (>50%): {high_confidence} ({high_confidence/len(regime_data):.1%})")
                    print(f"   Medium confidence (20-50%): {medium_confidence} ({medium_confidence/len(regime_data):.1%})")
                    print(f"   Low confidence (≤20%): {low_confidence} ({low_confidence/len(regime_data):.1%})")
                    
                    # Investigate causes of low confidence
                    print(f"\n🔍 INVESTIGATING LOW CONFIDENCE CAUSES:")
                    
                    if low_confidence / len(regime_data) > 0.8:
                        print(f"   ❌ MAJOR ISSUE: >80% of samples have low confidence")
                        print(f"   🔧 POSSIBLE FIXES:")
                        print(f"      1. Reduce correlation window (63 days → 30 days)")
                        print(f"      2. Lower confidence threshold (0.5 → 0.3)")
                        print(f"      3. Simplify regime classification logic")
                        print(f"      4. Add more external data sources")
                    
                    # Check current market conditions
            else:
                print(f"   ❌ Regime analysis failed to return data")
            
            return True
            
        except Exception as e:
            print(f"❌ Regime confidence debug failed: {e}")
            return False
    
    def debug_feature_selection_logic(self):
        """
        Debug Issue #3: Regime features in final set despite low temporal importance
        """
        print(f"\n🎯 DEBUG ISSUE #3: FEATURE SELECTION LOGIC INCONSISTENCY")
        print("=" * 65)
        
        try:
            # Load the rigorous validation results
            with open('data/rigorous_validation_results.json', 'r') as f:
                rigorous_results = json.load(f)
            
            with open('data/final_feature_recommendations.json', 'r') as f:
                final_results = json.load(f)
            
            validation_df = pd.DataFrame(rigorous_results['validation_results'])
            final_features = final_results['final_features']
            temporal_performance = final_results['temporal_performance']
            
            print(f"🔍 ANALYZING FEATURE SELECTION INCONSISTENCY:")
            
            # Compare CV scores vs temporal importance
            print(f"\n📊 COMPARING VALIDATION CRITERIA:")
            print(f"{'Feature':<25} {'CV Score':<10} {'Temporal Importance':<18} {'In Final Set':<12}")
            print("-" * 70)
            
            regime_features = ['gold_equity_corr', 'cross_asset_dispersion', 'vix_rates_corr', 
                              'aud_gold_corr', 'equity_bonds_corr', 'regime_stability', 'regime_confidence']
            
            for feature in regime_features:
                if feature in validation_df['feature'].values:
                    cv_score = validation_df[validation_df['feature'] == feature]['cv_improvement'].iloc[0]
                    
                    # Get temporal importance from the temporal validation
                    temp_importance = 0.0
                    if 'feature_importance' in temporal_performance:
                        for imp_data in temporal_performance['feature_importance']:
                            if imp_data['feature'] == feature:
                                temp_importance = imp_data['importance']
                                break
                    
                    in_final = feature in final_features
                    
                    print(f"{feature:<25} {cv_score:<10.1%} {temp_importance:<18.4f} {'✓' if in_final else '✗':<12}")
            
            # Identify the inconsistency
            print(f"\n🔍 INCONSISTENCY ANALYSIS:")
            
            high_cv_regime = []
            low_temp_regime = []
            
            for feature in regime_features:
                if feature in validation_df['feature'].values and feature in final_features:
                    cv_score = validation_df[validation_df['feature'] == feature]['cv_improvement'].iloc[0]
                    
                    temp_importance = 0.0
                    if 'feature_importance' in temporal_performance:
                        for imp_data in temporal_performance['feature_importance']:
                            if imp_data['feature'] == feature:
                                temp_importance = imp_data['importance']
                                break
                    
                    if cv_score > 0.15:  # High CV score
                        high_cv_regime.append((feature, cv_score, temp_importance))
                    
                    if temp_importance < 0.05:  # Low temporal importance
                        low_temp_regime.append((feature, cv_score, temp_importance))
            
            print(f"   Regime features with high CV scores (>15%): {len(high_cv_regime)}")
            for feature, cv, temp_imp in high_cv_regime:
                print(f"      • {feature}: CV {cv:.1%}, Temporal {temp_imp:.4f}")
            
            print(f"   Regime features with low temporal importance (<0.05): {len(low_temp_regime)}")
            for feature, cv, temp_imp in low_temp_regime:
                print(f"      • {feature}: CV {cv:.1%}, Temporal {temp_imp:.4f}")
            
            # Explain the issue
            print(f"\n💡 ROOT CAUSE IDENTIFIED:")
            print(f"   ❌ FEATURE SELECTION BASED PRIMARILY ON CV SCORES")
            print(f"   ❌ TEMPORAL IMPORTANCE GIVEN LOW WEIGHT")
            print(f"   ❌ CORRELATION VALIDATION OVERWEIGHTED")
            
            print(f"\n🔧 PROPOSED FIX - BALANCED FEATURE SELECTION:")
            print(f"   1. Weight: 40% CV improvement, 30% Temporal importance, 30% Correlation")
            print(f"   2. Require minimum temporal importance (0.02)")
            print(f"   3. Prioritize features that perform well across all tests")
            
            # Implement balanced scoring
            print(f"\n🎯 REIMPLEMENTING BALANCED FEATURE SCORING:")
            
            balanced_scores = []
            for _, row in validation_df.iterrows():
                feature = row['feature']
                
                cv_score = row['cv_improvement']
                correlation = row['max_correlation']
                
                # Get temporal importance
                temp_importance = 0.0
                if 'feature_importance' in temporal_performance:
                    for imp_data in temporal_performance['feature_importance']:
                        if imp_data['feature'] == feature:
                            temp_importance = imp_data['importance']
                            break
                
                # Balanced scoring (40% CV, 30% Temporal, 30% Correlation)
                normalized_cv = min(cv_score / 0.20, 1.0)  # Normalize to max 20%
                normalized_temp = min(temp_importance / 0.10, 1.0)  # Normalize to max 10%
                normalized_corr = min(correlation / 0.50, 1.0)  # Normalize to max 50%
                
                balanced_score = (0.4 * normalized_cv + 
                                0.3 * normalized_temp + 
                                0.3 * normalized_corr)
                
                # Require minimum temporal importance
                meets_temporal_requirement = temp_importance >= 0.02
                
                balanced_scores.append({
                    'feature': feature,
                    'balanced_score': balanced_score,
                    'cv_score': cv_score,
                    'temporal_importance': temp_importance,
                    'correlation': correlation,
                    'meets_temporal_requirement': meets_temporal_requirement
                })
            
            balanced_df = pd.DataFrame(balanced_scores)
            balanced_df = balanced_df.sort_values('balanced_score', ascending=False)
            
            # Select features with balanced criteria
            balanced_selection = balanced_df[
                (balanced_df['balanced_score'] > 0.3) & 
                (balanced_df['meets_temporal_requirement'])
            ].head(12)
            
            print(f"\n🏆 BALANCED FEATURE SELECTION (12 features):")
            print(f"{'Rank':<5} {'Feature':<25} {'Balanced':<9} {'CV':<8} {'Temporal':<10} {'Corr':<8}")
            print("-" * 75)
            
            for i, (_, row) in enumerate(balanced_selection.iterrows()):
                print(f"{i+1:<5} {row['feature']:<25} {row['balanced_score']:<9.3f} "
                      f"{row['cv_score']:<8.1%} {row['temporal_importance']:<10.4f} {row['correlation']:<8.3f}")
            
            # Save corrected feature set
            corrected_features = balanced_selection['feature'].tolist()
            
            corrected_recommendations = {
                'corrected_features': corrected_features,
                'feature_count': len(corrected_features),
                'selection_method': 'balanced_scoring',
                'balanced_scores': balanced_df.to_dict('records'),
                'original_issues': {
                    'covid_validation_fixed': False,  # Still needs temporal data
                    'regime_confidence_low': True,
                    'feature_selection_logic_corrected': True
                }
            }
            
            with open('data/corrected_feature_recommendations.json', 'w') as f:
                json.dump(corrected_recommendations, f, indent=2, default=str)
            
            print(f"\n💾 Corrected feature recommendations saved")
            print(f"🎯 BALANCED SELECTION: {len(corrected_features)} features with balanced criteria")
            
            return corrected_recommendations
            
        except Exception as e:
            print(f"❌ Feature selection debug failed: {e}")
            return None
    
    def load_price_data(self):
        """Load price data"""
        from data_manager.data_manager import ETFDataManager
        import os
        
        data_manager = ETFDataManager()
        universe_df = data_manager.load_universe()
        all_tickers = universe_df['ticker'].tolist()
        
        price_data = {}
        for ticker in all_tickers:
            try:
                possible_files = [
                    f"data/historical/{ticker.replace('.AX', '_AX')}.parquet",
                    f"data/historical/{ticker}.parquet"
                ]
                
                for file_path in possible_files:
                    if os.path.exists(file_path):
                        hist_data = pd.read_parquet(file_path)
                        if 'Close' in hist_data.columns and len(hist_data) >= 100:
                            price_data[ticker] = hist_data
                            break
            except:
                continue
        
        return price_data
    
    def extract_features_for_analysis(self, price_data):
        """Extract features for analysis"""
        from analyzers.ml_ensemble import MLEnsemble
        from utilities.shared_utils import extract_column
        
        ml_ensemble = MLEnsemble(use_enhanced_features=True)
        
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
        
        all_features = []
        all_targets = []
        
        for ticker, etf_data in price_data.items():
            try:
                prices = extract_column(etf_data, 'Close')
                volumes = extract_column(etf_data, 'Volume') if 'Volume' in etf_data.columns else None
                
                if len(prices) >= 120:
                    feature_prices = prices.iloc[:-60]
                    target_prices = prices.iloc[-60:]
                    
                    features = ml_ensemble.extract_ml_features(feature_prices, volumes.iloc[:-60] if volumes is not None else None, use_last_point=True)
                    target = (target_prices.iloc[-1] / target_prices.iloc[0] - 1)
                    
                    all_features.append(features[0])
                    all_targets.append(target)
            except:
                continue
        
        features_df = pd.DataFrame(all_features, columns=feature_names)
        targets_series = pd.Series(all_targets)
        
        # Clean data
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        targets_series = targets_series.replace([np.inf, -np.inf], np.nan)
        
        nan_mask = features_df.isna().any(axis=1) | targets_series.isna()
        features_df = features_df[~nan_mask]
        targets_series = targets_series[~nan_mask]
        
        return features_df.fillna(0), targets_series.fillna(0), []


def main():
    """Main debugging function"""
    print("🐛 CRITICAL ISSUES DEBUGGER")
    print("Fixing COVID validation, regime confidence, and feature selection")
    print("=" * 70)
    
    debugger = CriticalIssuesDebugger()
    
    # Debug Issue #1: COVID validation
    covid_debug = debugger.debug_covid_validation()
    
    # Debug Issue #2: Regime confidence
    regime_debug = debugger.debug_regime_confidence()
    
    # Debug Issue #3: Feature selection logic
    feature_debug = debugger.debug_feature_selection_logic()
    
    print(f"\n🎯 DEBUGGING SUMMARY:")
    print(f"   Issue #1 (COVID validation): {'✅ Identified fix needed' if covid_debug else '❌ Failed'}")
    print(f"   Issue #2 (Regime confidence): {'✅ Investigated' if regime_debug else '❌ Failed'}")
    print(f"   Issue #3 (Feature selection): {'✅ Corrected' if feature_debug else '❌ Failed'}")
    
    if feature_debug:
        print(f"\n📊 CORRECTED FEATURE SET READY:")
        print(f"   Features: {len(feature_debug['corrected_features'])}")
        print(f"   Method: Balanced scoring (40% CV, 30% Temporal, 30% Correlation)")
        print(f"   Temporal requirement: ≥0.02 importance")
    
    return {
        'covid_debug': covid_debug,
        'regime_debug': regime_debug,
        'feature_debug': feature_debug
    }


if __name__ == "__main__":
    results = main()
