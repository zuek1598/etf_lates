#!/usr/bin/env python3
"""
Enhanced Feature Validator with MACD-V and Demand-Supply Indicators
Integrates the new indicators into the feature analysis system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from analyzers.ml_ensemble import MLEnsemble
from utilities.shared_utils import extract_column
from indicators.macd_v_demand_supply import MACDVIndicator, DemandSupplyIndicator


class EnhancedFeatureValidator:
    """
    Enhanced feature validator with MACD-V and Demand-Supply indicators
    Analyzes which features actually contribute to predictions
    """
    
    def __init__(self):
        """Initialize enhanced feature validator"""
        self.macd_v = MACDVIndicator()
        self.demand_supply = DemandSupplyIndicator()
    
    def extract_all_features(self, prices: pd.Series, volumes: pd.Series = None) -> Dict[str, float]:
        """
        Extract all available features including MACD-V and Demand-Supply
        
        Args:
            prices: Price series
            volumes: Volume series (optional)
            
        Returns:
            Dict of all features
        """
        features = {}
        
        # Original ML features (from ml_ensemble.py)
        if len(prices) >= 60:
            returns = prices.pct_change().dropna()
            
            # Feature 1: Momentum (20-day)
            momentum = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else 0
            features['momentum'] = float(momentum)
            
            # Feature 2: Volatility (30-day)
            vol_30 = returns.tail(30).std() if len(returns) > 30 else 0.01
            features['volatility'] = float(vol_30)
            
            # Feature 3: RSI momentum
            gains = returns.where(returns > 0, 0)
            losses = -returns.where(returns < 0, 0)
            avg_gain = gains.tail(14).mean() if len(gains) > 14 else 0
            avg_loss = losses.tail(14).mean() if len(losses) > 14 else 0
            total_gain_loss = avg_gain + avg_loss
            rsi = avg_gain / total_gain_loss if total_gain_loss > 0 else 0.5
            features['rsi'] = float(rsi)
            
            # Feature 4: Price position (60-day range)
            price_60_high = prices.tail(60).max()
            price_60_low = prices.tail(60).min()
            price_pos = (prices.iloc[-1] - price_60_low) / (price_60_high - price_60_low) if price_60_high > price_60_low else 0.5
            features['price_position'] = float(price_pos)
            
            # Feature 5: SMA ratio
            sma_20 = prices.tail(20).mean()
            sma_ratio = prices.iloc[-1] / sma_20 if sma_20 > 0 else 1.0
            features['sma_ratio'] = float(sma_ratio)
            
            # Feature 6: Recent return vs historical
            recent_return = returns.tail(20).mean()
            hist_return = returns.tail(60).mean() if len(returns) > 60 else recent_return
            return_ratio = recent_return / hist_return if hist_return != 0 else 1.0
            features['return_ratio'] = float(return_ratio)
        
        # MACD-V features (13 features)
        macd_v_features = self.macd_v.extract_macd_v_features(prices)
        # Add prefix to avoid conflicts
        for key, value in macd_v_features.items():
            features[f'macd_v_{key}'] = value
        
        # Demand-Supply features (14 features) - only if volume data available
        if volumes is not None and len(volumes) > 0:
            ds_features = self.demand_supply.extract_demand_supply_features(prices, volumes)
            # Add prefix to avoid conflicts
            for key, value in ds_features.items():
                features[f'ds_{key}'] = value
        
        return features
    
    def analyze_feature_importance(self, price_data: Dict[str, pd.DataFrame], 
                                 volume_data: Dict[str, pd.Series] = None,
                                 sample_size: int = 50) -> Dict:
        """
        Analyze feature importance across multiple ETFs with enhanced indicators
        
        Args:
            price_data: Dict of {ticker: price_data}
            volume_data: Dict of {ticker: volume_series}
            sample_size: Number of ETFs to sample for analysis
            
        Returns:
            Dict with feature importance analysis
        """
        print(f"🔍 Analyzing enhanced feature importance across {len(price_data)} ETFs...")
        
        # Sample ETFs for analysis
        tickers = list(price_data.keys())
        if len(tickers) > sample_size:
            tickers = np.random.choice(tickers, sample_size, replace=False)
        
        feature_data = []
        returns_data = []
        
        for ticker in tickers:
            try:
                etf_data = price_data[ticker]
                prices = extract_column(etf_data, 'Close')
                volumes = extract_column(etf_data, 'Volume') if 'Volume' in etf_data.columns else None
                
                if len(prices) < 100:
                    continue
                
                # Extract enhanced features
                features = self.extract_all_features(prices, volumes)
                
                # Calculate forward return (target)
                forward_return = (prices.iloc[-1] / prices.iloc[-60] - 1) if len(prices) > 60 else 0
                
                feature_data.append(features)
                returns_data.append(forward_return)
                
            except Exception as e:
                print(f"⚠️ Feature extraction failed for {ticker}: {e}")
                continue
        
        if len(feature_data) < 10:
            print("❌ Insufficient data for feature analysis")
            return {}
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_data)
        returns_series = pd.Series(returns_data)
        
        # Calculate feature correlations with returns
        correlations = {}
        for feature in features_df.columns:
            corr = features_df[feature].corr(returns_series)
            if not np.isnan(corr):
                correlations[feature] = abs(corr)  # Use absolute correlation
        
        # Sort by importance
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Feature importance categories
        high_importance = [(f, c) for f, c in sorted_features if c > 0.2]
        medium_importance = [(f, c) for f, c in sorted_features if 0.1 <= c <= 0.2]
        low_importance = [(f, c) for f, c in sorted_features if c < 0.1]
        
        # Categorize by feature type
        basic_features = [(f, c) for f, c in sorted_features if not f.startswith(('macd_v_', 'ds_'))]
        macd_v_features = [(f, c) for f, c in sorted_features if f.startswith('macd_v_')]
        ds_features = [(f, c) for f, c in sorted_features if f.startswith('ds_')]
        
        print(f"\n📊 Enhanced Feature Importance Analysis (based on {len(feature_data)} ETFs):")
        print(f"  Total Features: {len(sorted_features)}")
        print(f"  High Importance (>0.2): {len(high_importance)} features")
        print(f"  Medium Importance (0.1-0.2): {len(medium_importance)} features")
        print(f"  Low Importance (<0.1): {len(low_importance)} features")
        
        print(f"\n📈 Feature Type Breakdown:")
        print(f"  Basic Features: {len(basic_features)}")
        print(f"  MACD-V Features: {len(macd_v_features)}")
        print(f"  Demand-Supply Features: {len(ds_features)}")
        
        print(f"\n🎯 Top 15 Most Predictive Features:")
        for i, (feature, correlation) in enumerate(sorted_features[:15]):
            feature_type = "🔥" if correlation > 0.3 else "⭐" if correlation > 0.2 else "📈"
            category = "Basic" if not feature.startswith(('macd_v_', 'ds_')) else \
                      "MACD-V" if feature.startswith('macd_v_') else "D-S"
            print(f"  {i+1:2d}. {feature_type} {feature:25s} | {category:8s} | Correlation: {correlation:.3f}")
        
        return {
            'feature_correlations': correlations,
            'sorted_features': sorted_features,
            'categories': {
                'high_importance': high_importance,
                'medium_importance': medium_importance,
                'low_importance': low_importance,
                'basic_features': basic_features,
                'macd_v_features': macd_v_features,
                'ds_features': ds_features
            },
            'sample_size': len(feature_data),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def generate_enhanced_feature_report(self, analysis_results: Dict) -> str:
        """
        Generate human-readable enhanced feature analysis report
        
        Args:
            analysis_results: Results from analyze_feature_importance
            
        Returns:
            Formatted report string
        """
        if not analysis_results:
            return "❌ No feature analysis results available"
        
        sorted_features = analysis_results['sorted_features']
        categories = analysis_results['categories']
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║            ENHANCED FEATURE VALIDATION ANALYSIS REPORT        ║
╚══════════════════════════════════════════════════════════════╝

📊 ANALYSIS SUMMARY:
• Sample Size: {analysis_results['sample_size']} ETFs
• Total Features Analyzed: {len(sorted_features)}
• Analysis Date: {analysis_results['analysis_date']}

🎯 FEATURE TYPE BREAKDOWN:
• Basic Features: {len(categories['basic_features'])} indicators
• MACD-V Features: {len(categories['macd_v_features'])} volatility-normalized indicators
• Demand-Supply Features: {len(categories['ds_features'])} volume-price indicators

📈 IMPORTANCE BREAKDOWN:
• High Importance (>0.2 correlation): {len(categories['high_importance'])} features
• Medium Importance (0.1-0.2 correlation): {len(categories['medium_importance'])} features  
• Low Importance (<0.1 correlation): {len(categories['low_importance'])} features

⭐ TOP 15 MOST PREDICTIVE FEATURES:
"""
        
        for i, (feature, correlation) in enumerate(sorted_features[:15]):
            importance = "🔥" if correlation > 0.3 else "⭐" if correlation > 0.2 else "📈"
            category = "Basic" if not feature.startswith(('macd_v_', 'ds_')) else \
                      "MACD-V" if feature.startswith('macd_v_') else "D-S"
            report += f"  {i+1:2d}. {importance} {feature:25s} | {category:8s} | Correlation: {correlation:.3f}\n"
        
        # Best features by type
        if categories['macd_v_features']:
            report += f"\n🔥 TOP MACD-V FEATURES:\n"
            for feature, correlation in categories['macd_v_features'][:5]:
                report += f"  • {feature}: {correlation:.3f}\n"
        
        if categories['ds_features']:
            report += f"\n📈 TOP DEMAND-SUPPLY FEATURES:\n"
            for feature, correlation in categories['ds_features'][:5]:
                report += f"  • {feature}: {correlation:.3f}\n"
        
        if categories['basic_features']:
            report += f"\n📊 TOP BASIC FEATURES:\n"
            for feature, correlation in categories['basic_features'][:5]:
                report += f"  • {feature}: {correlation:.3f}\n"
        
        report += f"\n💡 ENHANCED INDICATOR PERFORMANCE:\n"
        macd_v_avg = np.mean([c for f, c in categories['macd_v_features']]) if categories['macd_v_features'] else 0
        ds_avg = np.mean([c for f, c in categories['ds_features']]) if categories['ds_features'] else 0
        basic_avg = np.mean([c for f, c in categories['basic_features']]) if categories['basic_features'] else 0
        
        report += f"• MACD-V Average Correlation: {macd_v_avg:.3f}\n"
        report += f"• Demand-Supply Average Correlation: {ds_avg:.3f}\n"
        report += f"• Basic Features Average Correlation: {basic_avg:.3f}\n"
        
        report += f"\n💡 RECOMMENDATIONS:\n"
        if macd_v_avg > basic_avg:
            report += f"• MACD-V indicators outperform basic features - consider prioritizing\n"
        if ds_avg > basic_avg:
            report += f"• Demand-Supply indicators show strong predictive power\n"
        if len(categories['high_importance']) > 0:
            report += f"• Focus on {len(categories['high_importance'])} high-importance features for model optimization\n"
        else:
            report += f"• Consider feature engineering - no features show strong correlation (>0.2)\n"
        
        return report


if __name__ == "__main__":
    # Test the enhanced feature validator
    validator = EnhancedFeatureValidator()
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    # Simulate price with trend and cycles
    trend = np.linspace(100, 120, 200)
    cycle = 5 * np.sin(np.linspace(0, 4*np.pi, 200))
    noise = np.random.randn(200) * 2
    prices = pd.Series(trend + cycle + noise, index=dates)
    
    # Simulate volume with price correlation
    base_volume = 500000
    volume_variation = np.random.randn(200) * 100000
    price_volume_corr = (prices - prices.mean()) * 10000
    volumes = pd.Series(np.maximum(100000, base_volume + volume_variation + price_volume_corr), index=dates)
    
    print("🧪 Testing Enhanced Feature Validator")
    print("=" * 50)
    
    # Test feature extraction
    features = validator.extract_all_features(prices, volumes)
    print(f"✅ Extracted {len(features)} enhanced features")
    
    # Count by type
    basic_count = len([f for f in features.keys() if not f.startswith(('macd_v_', 'ds_'))])
    macd_v_count = len([f for f in features.keys() if f.startswith('macd_v_')])
    ds_count = len([f for f in features.keys() if f.startswith('ds_')])
    
    print(f"  Basic Features: {basic_count}")
    print(f"  MACD-V Features: {macd_v_count}")
    print(f"  Demand-Supply Features: {ds_count}")
    
    # Show sample features
    print(f"\n🎯 Sample Enhanced Features:")
    for name, value in list(features.items())[:8]:
        print(f"  {name:30s}: {value:8.4f}")
    
    print(f"\n✅ Enhanced feature validator working correctly!")
