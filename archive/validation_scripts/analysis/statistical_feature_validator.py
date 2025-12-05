#!/usr/bin/env python3
"""
Statistical Feature Validation System
Comprehensive statistical testing to identify which indicators actually contribute to ML forecasting
Uses multiple validation methods: correlation analysis, permutation importance, SHAP values, and predictive performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.inspection import permutation_importance
from scipy import stats

from analyzers.ml_ensemble import MLEnsemble
from utilities.shared_utils import extract_column


class StatisticalFeatureValidator:
    """
    Statistical validation system for ML feature importance
    Uses multiple statistical methods to identify truly predictive features
    """
    
    def __init__(self, significance_level: float = 0.05, min_samples: int = 30):
        """
        Initialize statistical feature validator
        
        Args:
            significance_level: Statistical significance threshold (default: 0.05)
            min_samples: Minimum samples required for validation (default: 30)
        """
        self.significance_level = significance_level
        self.min_samples = min_samples
        self.ml_ensemble = MLEnsemble(use_enhanced_features=True)
        
        # Feature categories for analysis
        self.feature_categories = {
            'basic': list(range(6)),  # First 6 features: basic technical indicators
            'regime': list(range(6, 13)),  # Next 7 features: regime indicators
            'macd_v': list(range(13, 26)),  # Next 13 features: MACD-V indicators
            'demand_supply': list(range(26, 40))  # Next 14 features: demand-supply indicators
        }
        
        self.validation_results = {}
    
    def extract_comprehensive_features(self, prices: pd.Series, volumes: pd.Series = None) -> Tuple[np.ndarray, Dict]:
        """
        Extract all features with proper naming
        
        Args:
            prices: Price series
            volumes: Volume series (optional)
            
        Returns:
            Tuple of (features_array, feature_info_dict)
        """
        # Get features from enhanced ML ensemble
        features = self.ml_ensemble.extract_ml_features(prices, volumes, use_last_point=True)
        
        # Create feature names
        feature_names = []
        
        # Basic features (6)
        basic_names = ['momentum', 'volatility', 'rsi', 'price_position', 'sma_ratio', 'return_ratio']
        feature_names.extend(basic_names)
        
        # Regime features (7)
        regime_names = [
            'gold_equity_corr', 'aud_gold_corr', 'vix_rates_corr', 
            'equity_bonds_corr', 'cross_asset_dispersion', 
            'regime_confidence', 'regime_stability'
        ]
        feature_names.extend(regime_names)
        
        # MACD-V features (13)
        macd_v_names = [
            'macd_signal', 'macd_histogram', 'macd_v_signal', 'macd_v_histogram',
            'macd_strength', 'macd_v_strength', 'volatility_level', 'volatility_regime',
            'macd_divergence', 'trend_consistency', 'macd_v_consistency', 
            'signal_quality', 'volatility_adjusted_momentum'
        ]
        feature_names.extend(macd_v_names)
        
        # Demand-Supply features (14)
        ds_names = [
            'volume_ratio', 'price_volume_correlation', 'money_flow_index', 'ad_trend',
            'obv_trend', 'volume_pressure', 'demand_strength', 'supply_pressure',
            'volume_confirmation', 'buying_pressure', 'selling_pressure',
            'demand_supply_balance', 'volume_trend_strength', 'price_volume_efficiency'
        ]
        feature_names.extend(ds_names)
        
        # Create feature info
        feature_info = {
            'names': feature_names,
            'categories': self.feature_categories,
            'total_features': len(feature_names)
        }
        
        return features, feature_info
    
    def correlation_analysis(self, features_df: pd.DataFrame, targets: pd.Series) -> Dict:
        """
        Perform statistical correlation analysis
        
        Args:
            features_df: DataFrame of features
            targets: Target variable series
            
        Returns:
            Dict with correlation analysis results
        """
        results = {}
        
        # Pearson correlation with significance testing
        pearson_corrs = []
        pearson_pvals = []
        
        for feature in features_df.columns:
            corr, pval = stats.pearsonr(features_df[feature], targets)
            pearson_corrs.append(abs(corr))  # Use absolute correlation
            pearson_pvals.append(pval)
        
        # Spearman correlation (non-parametric)
        spearman_corrs = []
        spearman_pvals = []
        
        for feature in features_df.columns:
            corr, pval = stats.spearmanr(features_df[feature], targets)
            spearman_corrs.append(abs(corr))
            spearman_pvals.append(pval)
        
        # Create results DataFrame
        corr_results = pd.DataFrame({
            'feature': features_df.columns,
            'pearson_corr': pearson_corrs,
            'pearson_pval': pearson_pvals,
            'spearman_corr': spearman_corrs,
            'spearman_pval': spearman_pvals
        })
        
        # Statistical significance
        corr_results['pearson_significant'] = corr_results['pearson_pval'] < self.significance_level
        corr_results['spearman_significant'] = corr_results['spearman_pval'] < self.significance_level
        
        # Combined significance
        corr_results['statistically_significant'] = (
            corr_results['pearson_significant'] | corr_results['spearman_significant']
        )
        
        # Average correlation
        corr_results['avg_correlation'] = (corr_results['pearson_corr'] + corr_results['spearman_corr']) / 2
        
        results['correlation_analysis'] = corr_results
        results['significant_features'] = corr_results[corr_results['statistically_significant']]
        results['summary'] = {
            'total_features': len(features_df.columns),
            'significant_count': corr_results['statistically_significant'].sum(),
            'significance_rate': corr_results['statistically_significant'].mean(),
            'top_correlation': corr_results['avg_correlation'].max(),
            'mean_correlation': corr_results['avg_correlation'].mean()
        }
        
        return results
    
    def permutation_importance_analysis(self, features_df: pd.DataFrame, targets: pd.Series) -> Dict:
        """
        Perform permutation importance analysis
        
        Args:
            features_df: DataFrame of features
            targets: Target variable series
            
        Returns:
            Dict with permutation importance results
        """
        results = {}
        
        # Use Random Forest for permutation importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Fit model
        rf.fit(features_df, targets)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            rf, features_df, targets, 
            n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Create results DataFrame
        perm_results = pd.DataFrame({
            'feature': features_df.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        })
        
        # Statistical significance (importance > 0)
        perm_results['significant'] = perm_results['importance_mean'] > 0
        
        # Sort by importance
        perm_results = perm_results.sort_values('importance_mean', ascending=False)
        
        results['permutation_importance'] = perm_results
        results['significant_features'] = perm_results[perm_results['significant']]
        results['summary'] = {
            'total_features': len(features_df.columns),
            'significant_count': perm_results['significant'].sum(),
            'significance_rate': perm_results['significant'].mean(),
            'top_importance': perm_results['importance_mean'].max(),
            'mean_importance': perm_results['importance_mean'].mean()
        }
        
        return results
    
    def cross_validation_performance(self, features_df: pd.DataFrame, targets: pd.Series) -> Dict:
        """
        Perform cross-validation performance analysis for each feature
        
        Args:
            features_df: DataFrame of features
            targets: Target variable series
            
        Returns:
            Dict with cross-validation results
        """
        results = {}
        
        # Time series split for temporal data
        tscv = TimeSeriesSplit(n_splits=5)
        
        feature_performance = []
        
        for feature in features_df.columns:
            feature_data = features_df[[feature]]
            
            # Test with Random Forest
            rf_scores = cross_val_score(
                RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
                feature_data, targets, cv=tscv, scoring='neg_mean_absolute_error'
            )
            
            # Test with Ridge Regression
            ridge_scores = cross_val_score(
                Ridge(alpha=1.0, random_state=42),
                feature_data, targets, cv=tscv, scoring='neg_mean_absolute_error'
            )
            
            # Calculate performance metrics
            rf_mae = -rf_scores.mean()
            ridge_mae = -ridge_scores.mean()
            best_mae = min(rf_mae, ridge_mae)
            
            # Baseline performance (predicting mean)
            baseline_mae = mean_absolute_error(targets, [targets.mean()] * len(targets))
            
            # Performance improvement over baseline
            improvement = (baseline_mae - best_mae) / baseline_mae
            
            feature_performance.append({
                'feature': feature,
                'rf_mae': rf_mae,
                'ridge_mae': ridge_mae,
                'best_mae': best_mae,
                'baseline_mae': baseline_mae,
                'improvement_pct': improvement * 100,
                'beats_baseline': improvement > 0
            })
        
        # Create results DataFrame
        cv_results = pd.DataFrame(feature_performance)
        cv_results = cv_results.sort_values('improvement_pct', ascending=False)
        
        results['cross_validation'] = cv_results
        results['predictive_features'] = cv_results[cv_results['beats_baseline']]
        results['summary'] = {
            'total_features': len(features_df.columns),
            'predictive_count': cv_results['beats_baseline'].sum(),
            'predictive_rate': cv_results['beats_baseline'].mean(),
            'best_improvement': cv_results['improvement_pct'].max(),
            'mean_improvement': cv_results['improvement_pct'].mean()
        }
        
        return results
    
    def comprehensive_feature_validation(self, price_data: Dict[str, pd.DataFrame], 
                                        sample_size: int = 50) -> Dict:
        """
        Perform comprehensive statistical validation of all features
        
        Args:
            price_data: Dict of {ticker: price_data}
            sample_size: Number of ETFs to sample for analysis
            
        Returns:
            Dict with comprehensive validation results
        """
        print(f"🔍 Performing comprehensive statistical feature validation...")
        print(f"📊 Analyzing {len(price_data)} ETFs (sample: {sample_size})")
        
        # Sample ETFs for analysis
        tickers = list(price_data.keys())
        if len(tickers) > sample_size:
            tickers = np.random.choice(tickers, sample_size, replace=False)
        
        # Extract features and targets for all ETFs
        all_features = []
        all_targets = []
        feature_info = None
        
        for ticker in tickers:
            try:
                etf_data = price_data[ticker]
                prices = extract_column(etf_data, 'Close')
                volumes = extract_column(etf_data, 'Volume') if 'Volume' in etf_data.columns else None
                
                if len(prices) < 100:
                    continue
                
                # Extract features
                features, info = self.extract_comprehensive_features(prices, volumes)
                if feature_info is None:
                    feature_info = info
                
                # Calculate target (60-day forward return)
                target = (prices.iloc[-1] / prices.iloc[-60] - 1) if len(prices) > 60 else 0
                
                all_features.append(features[0])  # Extract single row
                all_targets.append(target)
                
            except Exception as e:
                print(f"⚠️ Feature extraction failed for {ticker}: {e}")
                continue
        
        if len(all_features) < self.min_samples:
            print(f"❌ Insufficient data: {len(all_features)} < {self.min_samples}")
            return {}
        
        # Convert to DataFrame and clean data
        features_df = pd.DataFrame(all_features, columns=feature_info['names'])
        targets_series = pd.Series(all_targets)
        
        # Clean data: handle infinity and NaN values
        print(f"🧹 Cleaning data...")
        
        # Replace infinity with NaN
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        targets_series = targets_series.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with NaN values
        nan_mask = features_df.isna().any(axis=1) | targets_series.isna()
        if nan_mask.any():
            print(f"  Removing {nan_mask.sum()} rows with NaN/infinity values")
            features_df = features_df[~nan_mask]
            targets_series = targets_series[~nan_mask]
        
        # Clip extreme values to prevent numerical issues
        for col in features_df.columns:
            # Calculate reasonable bounds based on percentiles
            lower_bound = features_df[col].quantile(0.01)
            upper_bound = features_df[col].quantile(0.99)
            
            # Clip values beyond 1st and 99th percentiles
            features_df[col] = features_df[col].clip(lower_bound, upper_bound)
        
        # Final check for any remaining issues
        features_df = features_df.fillna(features_df.mean())
        targets_series = targets_series.fillna(targets_series.mean())
        
        print(f"✅ Cleaned data: {len(features_df)} samples, {len(features_df.columns)} features")
        
        print(f"✅ Extracted {len(features_df)} samples with {len(features_df.columns)} features")
        
        # Perform all statistical tests
        print(f"📈 Performing correlation analysis...")
        correlation_results = self.correlation_analysis(features_df, targets_series)
        
        print(f"🔀 Performing permutation importance analysis...")
        permutation_results = self.permutation_importance_analysis(features_df, targets_series)
        
        print(f"🎯 Performing cross-validation performance analysis...")
        cv_results = self.cross_validation_performance(features_df, targets_series)
        
        # Combine results
        combined_results = self._combine_validation_results(
            correlation_results, permutation_results, cv_results, feature_info
        )
        
        # Store results
        self.validation_results = combined_results
        
        return combined_results
    
    def _combine_validation_results(self, corr_results: Dict, perm_results: Dict, 
                                  cv_results: Dict, feature_info: Dict) -> Dict:
        """
        Combine results from all validation methods
        
        Args:
            corr_results: Correlation analysis results
            perm_results: Permutation importance results
            cv_results: Cross-validation results
            feature_info: Feature information
            
        Returns:
            Dict with combined validation results
        """
        # Create master DataFrame
        master_df = pd.DataFrame({'feature': feature_info['names']})
        
        # Add correlation results
        corr_df = corr_results['correlation_analysis'][['feature', 'avg_correlation', 'statistically_significant']]
        master_df = master_df.merge(corr_df, on='feature', how='left')
        master_df.rename(columns={'statistically_significant': 'corr_significant'}, inplace=True)
        
        # Add permutation importance results
        perm_df = perm_results['permutation_importance'][['feature', 'importance_mean', 'significant']]
        master_df = master_df.merge(perm_df, on='feature', how='left')
        master_df.rename(columns={'significant': 'perm_significant'}, inplace=True)
        
        # Add cross-validation results
        cv_df = cv_results['cross_validation'][['feature', 'improvement_pct', 'beats_baseline']]
        master_df = master_df.merge(cv_df, on='feature', how='left')
        master_df.rename(columns={'beats_baseline': 'cv_predictive'}, inplace=True)
        
        # Calculate composite scores
        master_df['validation_score'] = (
            (master_df['avg_correlation'].fillna(0) * 0.3) +
            (master_df['importance_mean'].fillna(0) * 0.4) +
            (master_df['improvement_pct'].fillna(0) / 100 * 0.3)
        )
        
        # Determine overall significance
        master_df['overall_significant'] = (
            master_df['corr_significant'].fillna(False) |
            master_df['perm_significant'].fillna(False) |
            master_df['cv_predictive'].fillna(False)
        )
        
        # Add feature categories
        def get_feature_category(idx):
            for category, indices in feature_info['categories'].items():
                if idx in indices:
                    return category
            return 'unknown'
        
        master_df['category'] = [get_feature_category(i) for i in range(len(master_df))]
        
        # Sort by validation score
        master_df = master_df.sort_values('validation_score', ascending=False)
        
        # Category-wise analysis
        category_analysis = {}
        for category in feature_info['categories'].keys():
            cat_features = master_df[master_df['category'] == category]
            if len(cat_features) > 0:
                category_analysis[category] = {
                    'total_features': len(cat_features),
                    'significant_features': cat_features['overall_significant'].sum(),
                    'significance_rate': cat_features['overall_significant'].mean(),
                    'avg_validation_score': cat_features['validation_score'].mean(),
                    'top_feature': cat_features.iloc[0]['feature'],
                    'top_score': cat_features.iloc[0]['validation_score']
                }
        
        return {
            'master_results': master_df,
            'category_analysis': category_analysis,
            'correlation_summary': corr_results['summary'],
            'permutation_summary': perm_results['summary'],
            'cv_summary': cv_results['summary'],
            'overall_summary': {
                'total_features': len(master_df),
                'significant_features': master_df['overall_significant'].sum(),
                'overall_significance_rate': master_df['overall_significant'].mean(),
                'top_features': master_df.head(10)['feature'].tolist(),
                'validation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'feature_info': feature_info
        }
    
    def generate_validation_report(self, results: Dict) -> str:
        """
        Generate comprehensive validation report
        
        Args:
            results: Validation results from comprehensive_feature_validation
            
        Returns:
            Formatted report string
        """
        if not results:
            return "❌ No validation results available"
        
        master_df = results['master_results']
        category_analysis = results['category_analysis']
        overall = results['overall_summary']
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║            STATISTICAL FEATURE VALIDATION REPORT              ║
╚══════════════════════════════════════════════════════════════╝

📊 VALIDATION SUMMARY:
• Total Features Analyzed: {overall['total_features']}
• Statistically Significant: {overall['significant_features']} ({overall['overall_significance_rate']:.1%})
• Validation Date: {overall['validation_date']}

🎯 TOP 10 MOST VALIDATED FEATURES:
"""
        
        for i, (_, row) in enumerate(master_df.head(10).iterrows()):
            significance = "✅" if row['overall_significant'] else "❌"
            score = row['validation_score']
            report += f"  {i+1:2d}. {significance} {row['feature']:25s} | Score: {score:.3f} | {row['category']:8s}\n"
        
        report += f"\n📈 CATEGORY PERFORMANCE ANALYSIS:\n"
        for category, analysis in category_analysis.items():
            report += f"• {category.upper():12s}: {analysis['significant_features']}/{analysis['total_features']} significant "
            report += f"({analysis['significance_rate']:.1%}) | Avg Score: {analysis['avg_validation_score']:.3f}\n"
            report += f"  └─ Top: {analysis['top_feature']} (Score: {analysis['top_score']:.3f})\n"
        
        report += f"\n🔍 STATISTICAL TEST RESULTS:\n"
        report += f"• Correlation Analysis: {results['correlation_summary']['significant_count']} significant "
        report += f"({results['correlation_summary']['significance_rate']:.1%})\n"
        report += f"• Permutation Importance: {results['permutation_summary']['significant_count']} significant "
        report += f"({results['permutation_summary']['significance_rate']:.1%})\n"
        report += f"• Cross-Validation: {results['cv_summary']['predictive_count']} predictive "
        report += f"({results['cv_summary']['predictive_rate']:.1%})\n"
        
        report += f"\n💡 RECOMMENDATIONS:\n"
        
        # Best performing category
        best_category = max(category_analysis.items(), key=lambda x: x[1]['avg_validation_score'])
        report += f"• BEST CATEGORY: {best_category[0].upper()} (Avg Score: {best_category[1]['avg_validation_score']:.3f})\n"
        
        # Feature recommendations
        significant_features = master_df[master_df['overall_significant']]
        if len(significant_features) > 0:
            report += f"• PRIORITIZE {len(significant_features)} statistically validated features\n"
        else:
            report += f"• ⚠️ NO features passed statistical significance tests\n"
        
        # Low performing features
        low_performers = master_df[master_df['validation_score'] < 0.01]
        if len(low_performers) > 0:
            report += f"• CONSIDER REMOVING {len(low_performers)} low-performing features\n"
        
        return report
    
    def get_validated_features(self, min_score: float = 0.05) -> List[str]:
        """
        Get list of statistically validated features
        
        Args:
            min_score: Minimum validation score threshold
            
        Returns:
            List of validated feature names
        """
        if not self.validation_results:
            return []
        
        master_df = self.validation_results['master_results']
        validated = master_df[
            (master_df['overall_significant']) & 
            (master_df['validation_score'] >= min_score)
        ]
        
        return validated['feature'].tolist()


if __name__ == "__main__":
    # Test the statistical validator
    print("🧪 Testing Statistical Feature Validator")
    print("=" * 50)
    
    validator = StatisticalFeatureValidator()
    
    # Create sample data for testing
    np.random.seed(42)
    sample_data = {}
    
    for i in range(10):
        ticker = f"TEST{i:02d}.AX"
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        
        # Simulate price data
        trend = np.linspace(100, 120, 200)
        noise = np.random.randn(200) * 2
        prices = pd.Series(trend + noise, index=dates)
        
        # Simulate volume data
        volumes = pd.Series(np.random.randint(100000, 1000000, 200), index=dates)
        
        df = pd.DataFrame({
            'Close': prices,
            'Volume': volumes
        })
        sample_data[ticker] = df
    
    print("📊 Running comprehensive validation...")
    results = validator.comprehensive_feature_validation(sample_data, sample_size=10)
    
    if results:
        print("\n" + validator.generate_validation_report(results))
        
        validated_features = validator.get_validated_features(min_score=0.01)
        print(f"\n✅ Validated Features ({len(validated_features)}):")
        for feature in validated_features[:5]:
            print(f"  • {feature}")
        if len(validated_features) > 5:
            print(f"  ... and {len(validated_features) - 5} more")
    else:
        print("❌ Validation failed")
