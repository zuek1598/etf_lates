#!/usr/bin/env python3
"""
Feature Validation Analysis and MACD-V Indicator
Analyzes which features actually contribute to predictions
Adds demand-supply indicators using volume-price analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from analyzers.ml_ensemble import MLEnsemble
from utilities.shared_utils import extract_column


class MACDVIndicator:
    """
    MACD-V (MACD with Volume) Demand-Supply Indicator
    Combines traditional MACD with volume analysis for demand-supply signals
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD-V indicator
        
        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate traditional MACD components
        
        Args:
            prices: Price series
            
        Returns:
            Dict with MACD components
        """
        ema_fast = self.calculate_ema(prices, self.fast_period)
        ema_slow = self.calculate_ema(prices, self.slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
    
    def calculate_volume_weighted_macd(self, prices: pd.Series, volumes: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate volume-weighted MACD for demand-supply analysis
        
        Args:
            prices: Price series
            volumes: Volume series
            
        Returns:
            Dict with volume-weighted MACD components
        """
        # Volume-weighted price (VWAP-like)
        volume_weighted_price = (prices * volumes).rolling(window=self.fast_period).sum() / volumes.rolling(window=self.fast_period).sum()
        volume_weighted_slow = (prices * volumes).rolling(window=self.slow_period).sum() / volumes.rolling(window=self.slow_period).sum()
        
        macd_v_line = volume_weighted_price - volume_weighted_slow
        signal_v_line = macd_v_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram_v = macd_v_line - signal_v_line
        
        return {
            'macd_v_line': macd_v_line,
            'signal_v_line': signal_v_line,
            'histogram_v': histogram_v
        }
    
    def extract_demand_supply_features(self, prices: pd.Series, volumes: pd.Series) -> Dict[str, float]:
        """
        Extract demand-supply features from MACD-V analysis
        
        Args:
            prices: Price series
            volumes: Volume series
            
        Returns:
            Dict of demand-supply features
        """
        if len(prices) < self.slow_period + self.signal_period or len(volumes) < self.slow_period + self.signal_period:
            return self._get_default_features()
        
        try:
            # Traditional MACD
            macd_components = self.calculate_macd(prices)
            
            # Volume-weighted MACD
            macd_v_components = self.calculate_volume_weighted_macd(prices, volumes)
            
            # Current values
            macd_current = macd_components['macd_line'].iloc[-1]
            signal_current = macd_components['signal_line'].iloc[-1]
            histogram_current = macd_components['histogram'].iloc[-1]
            
            macd_v_current = macd_v_components['macd_v_line'].iloc[-1]
            signal_v_current = macd_v_components['signal_v_line'].iloc[-1]
            histogram_v_current = macd_v_components['histogram_v'].iloc[-1]
            
            # Demand-Supply signals
            macd_bullish = macd_current > signal_current
            macd_v_bullish = macd_v_current > signal_v_current
            
            # Divergence signals (price vs MACD)
            price_momentum = (prices.iloc[-1] / prices.iloc[-20] - 1) if len(prices) > 20 else 0
            macd_momentum = (macd_current - macd_components['macd_line'].iloc[-20]) if len(macd_components['macd_line']) > 20 else 0
            volume_momentum = (volumes.iloc[-20:].mean() / volumes.iloc[-40:-20].mean() - 1) if len(volumes) > 40 else 0
            
            # Demand-Supply strength
            demand_strength = histogram_current if macd_bullish else -histogram_current
            supply_strength = histogram_v_current if macd_v_bullish else -histogram_v_current
            
            # Volume confirmation
            volume_confirmation = volume_momentum if macd_bullish else -volume_momentum
            
            features = {
                'macd_signal': float(macd_current - signal_current),  # MACD crossover strength
                'macd_histogram': float(histogram_current),          # MACD momentum
                'macd_v_signal': float(macd_v_current - signal_v_current),  # Volume-weighted signal
                'macd_v_histogram': float(histogram_v_current),      # Volume-weighted momentum
                'demand_strength': float(demand_strength),           # Demand indicator
                'supply_strength': float(supply_strength),           # Supply indicator
                'volume_confirmation': float(volume_confirmation),   # Volume-backed momentum
                'macd_divergence': float(price_momentum - macd_momentum),  # Price-MACD divergence
                'bullish_alignment': float(1.0 if macd_bullish and macd_v_bullish else 0.0),  # Bullish consensus
                'volume_pressure': float(volume_momentum)             # Volume pressure indicator
            }
            
            return features
            
        except Exception as e:
            print(f"⚠️ MACD-V calculation failed: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when calculation fails"""
        return {
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'macd_v_signal': 0.0,
            'macd_v_histogram': 0.0,
            'demand_strength': 0.0,
            'supply_strength': 0.0,
            'volume_confirmation': 0.0,
            'macd_divergence': 0.0,
            'bullish_alignment': 0.0,
            'volume_pressure': 0.0
        }


class FeatureValidator:
    """
    Analyzes which features actually contribute to prediction accuracy
    Uses statistical methods to rank feature importance
    """
    
    def __init__(self):
        """Initialize feature validator"""
        self.macd_v = MACDVIndicator()
    
    def extract_all_features(self, prices: pd.Series, volumes: pd.Series = None) -> Dict[str, float]:
        """
        Extract all available features including MACD-V
        
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
        
        # MACD-V demand-supply features
        if volumes is not None and len(volumes) > 0:
            macd_v_features = self.macd_v.extract_demand_supply_features(prices, volumes)
            features.update(macd_v_features)
        
        return features
    
    def analyze_feature_importance(self, price_data: Dict[str, pd.DataFrame], 
                                 volume_data: Dict[str, pd.Series] = None,
                                 sample_size: int = 50) -> Dict:
        """
        Analyze feature importance across multiple ETFs
        
        Args:
            price_data: Dict of {ticker: price_data}
            volume_data: Dict of {ticker: volume_series}
            sample_size: Number of ETFs to sample for analysis
            
        Returns:
            Dict with feature importance analysis
        """
        print(f"🔍 Analyzing feature importance across {len(price_data)} ETFs...")
        
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
                
                # Extract features
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
        
        print(f"\n📊 Feature Importance Analysis (based on {len(feature_data)} ETFs):")
        print(f"  High Importance (>0.2): {len(high_importance)} features")
        print(f"  Medium Importance (0.1-0.2): {len(medium_importance)} features")
        print(f"  Low Importance (<0.1): {len(low_importance)} features")
        
        print(f"\n🎯 Top 10 Most Predictive Features:")
        for i, (feature, correlation) in enumerate(sorted_features[:10]):
            print(f"  {i+1:2d}. {feature:20s} | Correlation: {correlation:.3f}")
        
        return {
            'feature_correlations': correlations,
            'sorted_features': sorted_features,
            'categories': {
                'high_importance': high_importance,
                'medium_importance': medium_importance,
                'low_importance': low_importance
            },
            'sample_size': len(feature_data),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def generate_feature_report(self, analysis_results: Dict) -> str:
        """
        Generate human-readable feature analysis report
        
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
║                 FEATURE VALIDATION ANALYSIS REPORT           ║
╚══════════════════════════════════════════════════════════════╝

📊 ANALYSIS SUMMARY:
• Sample Size: {analysis_results['sample_size']} ETFs
• Total Features Analyzed: {len(sorted_features)}
• Analysis Date: {analysis_results['analysis_date']}

🎯 FEATURE IMPORTANCE BREAKDOWN:
• High Importance (>0.2 correlation): {len(categories['high_importance'])} features
• Medium Importance (0.1-0.2 correlation): {len(categories['medium_importance'])} features  
• Low Importance (<0.1 correlation): {len(categories['low_importance'])} features

⭐ TOP 10 MOST PREDICTIVE FEATURES:
"""
        
        for i, (feature, correlation) in enumerate(sorted_features[:10]):
            importance = "🔥" if correlation > 0.3 else "⭐" if correlation > 0.2 else "📈"
            report += f"  {i+1:2d}. {importance} {feature:20s} | Correlation: {correlation:.3f}\n"
        
        if categories['high_importance']:
            report += f"\n🔥 HIGH IMPORTANCE FEATURES (>0.2):\n"
            for feature, correlation in categories['high_importance']:
                report += f"  • {feature}: {correlation:.3f}\n"
        
        if categories['medium_importance']:
            report += f"\n📈 MEDIUM IMPORTANCE FEATURES (0.1-0.2):\n"
            for feature, correlation in categories['medium_importance']:
                report += f"  • {feature}: {correlation:.3f}\n"
        
        report += f"\n💡 RECOMMENDATIONS:\n"
        if len(categories['high_importance']) > 0:
            report += f"• Focus on {len(categories['high_importance'])} high-importance features for model optimization\n"
        else:
            report += f"• Consider feature engineering - no features show strong correlation (>0.2)\n"
        
        if len(categories['low_importance']) > len(sorted_features) / 2:
            report += f"• {len(categories['low_importance'])} features show low predictive power - consider removal\n"
        
        return report


if __name__ == "__main__":
    # Test the feature validator
    validator = FeatureValidator()
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    prices = pd.Series(100 + np.random.randn(200).cumsum() * 0.5, index=dates)
    volumes = pd.Series(np.random.randint(100000, 1000000, 200), index=dates)
    
    print("=== Feature Validator Test ===")
    features = validator.extract_all_features(prices, volumes)
    print(f"Extracted {len(features)} features:")
    for name, value in list(features.items())[:5]:
        print(f"  {name}: {value:.4f}")
    print("  ...")
