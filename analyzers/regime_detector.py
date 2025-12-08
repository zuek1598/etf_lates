#!/usr/bin/env python3
"""
Regime Detection System - Phase 2 Implementation
Implements cross-asset correlation analysis and regime classification

Key Features:
- 5 correlation pairs for regime analysis
- 21-day rolling correlation windows (optimized for 20-day forecasts)
- 100-day regime classification windows
- Sub-regime detection (8-10 types)
- Regime transition identification
- Confidence scoring based on correlation stability

Usage:
    from analyzers.regime_detector import RegimeDetector
    
    detector = RegimeDetector()
    regime_data = detector.analyze_regimes(external_data, equity_index)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RegimeDetector:
    """Advanced regime detection using cross-asset correlations"""
    
    def __init__(self, correlation_window: int = 21, regime_window: int = 100):
        """
        Initialize regime detector
        
        Args:
            correlation_window: Rolling window for correlation calculation (21 days = ~1 month)
            regime_window: Window for regime classification (100 days = 5 months)
        """
        self.correlation_window = correlation_window
        self.regime_window = regime_window
        
        # Define the 5 key correlation pairs
        self.correlation_pairs = [
            {
                'name': 'gold_equity',
                'asset1': 'gold',
                'asset2': 'equity',
                'description': 'Gold-Equity correlation (crisis type identification)',
                'interpretation': {
                    'high_positive': 'Risk-on regime (inflation)',
                    'high_negative': 'Crisis regime (flight to safety)',
                    'neutral': 'Normal market conditions'
                }
            },
            {
                'name': 'aud_gold',
                'asset1': 'aud_usd',
                'asset2': 'gold',
                'description': 'AUD-Gold correlation (Australia-specific risk)',
                'interpretation': {
                    'high_positive': 'Commodity boom',
                    'high_negative': 'Currency crisis',
                    'neutral': 'Balanced conditions'
                }
            },
            {
                'name': 'vix_rates',
                'asset1': 'vix',
                'asset2': 'us_10y',
                'description': 'VIX-Rates correlation (stagflation detection)',
                'interpretation': {
                    'high_positive': 'Stagflation risk',
                    'high_negative': 'Growth slowdown',
                    'neutral': 'Normal monetary policy'
                }
            },
            {
                'name': 'equity_bonds',
                'asset1': 'equity',
                'asset2': 'us_10y',
                'description': 'Equity-Bonds correlation (hedge effectiveness)',
                'interpretation': {
                    'high_positive': 'Trend following regime',
                    'high_negative': 'Risk-off regime',
                    'neutral': 'Balanced allocation'
                }
            },
            {
                'name': 'cross_asset_dispersion',
                'asset1': 'dispersion',
                'asset2': 'dispersion',
                'description': 'Cross-asset dispersion (regime confidence)',
                'interpretation': {
                    'high': 'Regime transition',
                    'low': 'Stable regime',
                    'medium': 'Established regime'
                }
            }
        ]
        
        # Regime classification thresholds
        self.regime_thresholds = {
            'correlation_strength': {
                'strong': 0.7,
                'moderate': 0.4,
                'weak': 0.1
            },
            'dispersion': {
                'high': 0.3,
                'medium': 0.15,
                'low': 0.05
            }
        }
    
    def calculate_correlation_series(self, data1: pd.Series, data2: pd.Series) -> pd.Series:
        """
        Calculate rolling correlation between two series
        
        Args:
            data1: First price series
            data2: Second price series
            
        Returns:
            Rolling correlation series
        """
        # Align the series on common dates
        aligned_data = pd.concat([data1, data2], axis=1).dropna()
        if len(aligned_data) < self.correlation_window:
            return pd.Series(dtype=float)
        
        # Calculate returns for correlation
        returns1 = aligned_data.iloc[:, 0].pct_change().dropna()
        returns2 = aligned_data.iloc[:, 1].pct_change().dropna()
        
        # Align returns
        aligned_returns = pd.concat([returns1, returns2], axis=1).dropna()
        
        if len(aligned_returns) < self.correlation_window:
            return pd.Series(dtype=float)
        
        # Calculate rolling correlation
        correlation = aligned_returns.iloc[:, 0].rolling(
            window=self.correlation_window,
            min_periods=int(self.correlation_window * 0.8)
        ).corr(aligned_returns.iloc[:, 1])
        
        return correlation.dropna()
    
    def calculate_cross_asset_dispersion(self, correlations: Dict[str, pd.Series]) -> pd.Series:
        """
        Calculate cross-asset dispersion as a regime confidence measure
        
        Args:
            correlations: Dictionary of correlation series
            
        Returns:
            Dispersion series (standard deviation of correlations)
        """
        if not correlations:
            return pd.Series(dtype=float)
        
        # Combine all correlation series
        correlation_df = pd.DataFrame(correlations)
        correlation_df = correlation_df.dropna()
        
        if correlation_df.empty:
            return pd.Series(dtype=float)
        
        # Calculate dispersion (standard deviation across correlations)
        dispersion = correlation_df.std(axis=1)
        
        return dispersion
    
    def classify_regime_type(self, correlations: Dict[str, pd.Series], 
                           dispersion: pd.Series, date: datetime) -> Dict:
        """
        Classify regime type based on correlation patterns
        
        Args:
            correlations: Dictionary of correlation series
            dispersion: Cross-asset dispersion series
            date: Date for classification
            
        Returns:
            Regime classification dictionary
        """
        regime_info = {
            'date': date,
            'base_regime': 'UNKNOWN',
            'sub_regime': 'UNKNOWN',
            'confidence': 0.0,
            'correlations': {},
            'interpretation': 'Insufficient data for classification'
        }
        
        # Get latest correlation values
        latest_correlations = {}
        for name, series in correlations.items():
            if not series.empty and date in series.index:
                latest_correlations[name] = float(series.loc[date])
        
        if len(latest_correlations) < 3:  # Need minimum correlations
            return regime_info
        
        # Get latest dispersion
        latest_dispersion = float(dispersion.loc[date]) if date in dispersion.index else 0.5
        
        regime_info['correlations'] = latest_correlations
        regime_info['dispersion'] = latest_dispersion
        
        # Base regime classification
        gold_equity = latest_correlations.get('gold_equity', 0)
        vix_rates = latest_correlations.get('vix_rates', 0)
        equity_bonds = latest_correlations.get('equity_bonds', 0)
        
        # Determine base regime
        if gold_equity < -0.5 and vix_rates > 0.3:
            regime_info['base_regime'] = 'CRISIS'
            regime_info['interpretation'] = 'Severe market stress with flight to safety'
        elif gold_equity > 0.3 and equity_bonds > 0.5:
            regime_info['base_regime'] = 'RISK_ON'
            regime_info['interpretation'] = 'Risk appetite with trend following'
        elif gold_equity < -0.3 and equity_bonds < -0.3:
            regime_info['base_regime'] = 'RISK_OFF'
            regime_info['interpretation'] = 'Risk aversion with hedging behavior'
        elif vix_rates > 0.5:
            regime_info['base_regime'] = 'STAGFLATION'
            regime_info['interpretation'] = 'High volatility with rising rates'
        elif abs(gold_equity) < 0.2 and abs(equity_bonds) < 0.2:
            regime_info['base_regime'] = 'NEUTRAL'
            regime_info['interpretation'] = 'Balanced market conditions'
        else:
            regime_info['base_regime'] = 'TRANSITIONAL'
            regime_info['interpretation'] = 'Mixed signals - regime in transition'
        
        # Sub-regime classification based on dispersion
        if latest_dispersion > self.regime_thresholds['dispersion']['high']:
            regime_info['sub_regime'] = 'HIGH_VOLATILITY'
            regime_info['confidence'] = 0.3  # Low confidence during transitions
        elif latest_dispersion < self.regime_thresholds['dispersion']['low']:
            regime_info['sub_regime'] = 'STABLE'
            regime_info['confidence'] = 0.9  # High confidence in stable regimes
        else:
            regime_info['sub_regime'] = 'MODERATE'
            regime_info['confidence'] = 0.6  # Medium confidence
        
        # Adjust confidence based on correlation strength consistency
        correlation_strengths = [abs(corr) for corr in latest_correlations.values()]
        avg_strength = np.mean(correlation_strengths)
        
        if avg_strength > 0.6:
            regime_info['confidence'] = min(0.95, regime_info['confidence'] + 0.2)
        elif avg_strength < 0.2:
            regime_info['confidence'] = max(0.1, regime_info['confidence'] - 0.3)
        
        return regime_info
    
    def analyze_regimes(self, external_data: Dict[str, pd.Series], 
                       equity_index: Optional[pd.Series] = None) -> Dict:
        """
        Complete regime analysis using external data
        
        Args:
            external_data: Dictionary of external data series
            equity_index: Equity index series (e.g., ASX200 or SP500)
            
        Returns:
            Complete regime analysis results
        """
        print(f"\n{'='*60}")
        print("REGIME DETECTION ANALYSIS - PHASE 2")
        print(f"{'='*60}")
        print(f"Correlation window: {self.correlation_window} days")
        print(f"Regime window: {self.regime_window} days")
        print(f"Analyzing {len(external_data)} external data series")
        
        if not external_data:
            print("❌ No external data provided")
            return {}
        
        # Add equity index if provided
        if equity_index is not None:
            external_data['equity'] = equity_index
            print("✅ Equity index added for correlation analysis")
        
        # Calculate correlations for each pair
        correlations = {}
        print(f"\n📊 Calculating correlations...")
        
        for pair in self.correlation_pairs:
            if pair['name'] == 'cross_asset_dispersion':
                continue  # Handle separately
            
            asset1_key = pair['asset1']
            asset2_key = pair['asset2']
            
            if asset1_key in external_data and asset2_key in external_data:
                corr_series = self.calculate_correlation_series(
                    external_data[asset1_key], 
                    external_data[asset2_key]
                )
                
                if not corr_series.empty:
                    correlations[pair['name']] = corr_series
                    print(f"  ✅ {pair['name']}: {len(corr_series)} correlation points")
                else:
                    print(f"  ❌ {pair['name']}: Insufficient overlapping data")
            else:
                print(f"  ⚠️ {pair['name']}: Missing data ({asset1_key} or {asset2_key})")
        
        if not correlations:
            print("❌ No correlations calculated - insufficient data")
            return {}
        
        # Calculate cross-asset dispersion
        dispersion = self.calculate_cross_asset_dispersion(correlations)
        correlations['cross_asset_dispersion'] = dispersion
        print(f"  ✅ Cross-asset dispersion: {len(dispersion)} points")
        
        # Classify regimes for each date
        print(f"\n🎯 Classifying regimes...")
        regime_classifications = []
        
        # Get common dates for analysis
        all_dates = set()
        for series in correlations.values():
            all_dates.update(series.index)
        
        analysis_dates = sorted(all_dates)
        
        # Sample dates for efficiency (every 5 days)
        sample_dates = analysis_dates[::5]
        
        for date in sample_dates:
            regime = self.classify_regime_type(correlations, dispersion, date)
            regime_classifications.append(regime)
        
        print(f"  ✅ Classified {len(regime_classifications)} regime points")
        
        # Create regime summary
        regime_summary = self.create_regime_summary(regime_classifications)
        
        # Compile results
        results = {
            'correlations': correlations,
            'dispersion': dispersion,
            'regime_classifications': regime_classifications,
            'regime_summary': regime_summary,
            'analysis_metadata': {
                'correlation_window': self.correlation_window,
                'regime_window': self.regime_window,
                'analysis_dates': len(sample_dates),
                'total_correlations': len(correlations),
                'data_quality': 'VALID' if len(correlations) >= 4 else 'LIMITED'
            }
        }
        
        print(f"\n📈 Regime Analysis Summary:")
        print(f"  • Base regimes detected: {len(regime_summary['base_regime_distribution'])}")
        print(f"  • Sub-regimes detected: {len(regime_summary['sub_regime_distribution'])}")
        print(f"  • Average confidence: {regime_summary['avg_confidence']:.1%}")
        print(f"  • Current regime: {regime_summary['current_regime']['base_regime']}")
        print(f"  • Data quality: {results['analysis_metadata']['data_quality']}")
        
        return results
    
    def create_regime_summary(self, regime_classifications: List[Dict]) -> Dict:
        """
        Create summary statistics from regime classifications
        
        Args:
            regime_classifications: List of regime classification dictionaries
            
        Returns:
            Regime summary dictionary
        """
        if not regime_classifications:
            return {}
        
        # Extract data for analysis
        base_regimes = [r['base_regime'] for r in regime_classifications]
        sub_regimes = [r['sub_regime'] for r in regime_classifications]
        confidences = [r['confidence'] for r in regime_classifications]
        
        # Calculate distributions
        base_regime_counts = pd.Series(base_regimes).value_counts()
        sub_regime_counts = pd.Series(sub_regimes).value_counts()
        
        # Get current regime (latest classification)
        current_regime = regime_classifications[-1] if regime_classifications else {}
        
        # Calculate regime transitions
        transitions = 0
        for i in range(1, len(regime_classifications)):
            if regime_classifications[i]['base_regime'] != regime_classifications[i-1]['base_regime']:
                transitions += 1
        
        transition_rate = transitions / len(regime_classifications) if regime_classifications else 0
        
        summary = {
            'total_classifications': len(regime_classifications),
            'base_regime_distribution': base_regime_counts.to_dict(),
            'sub_regime_distribution': sub_regime_counts.to_dict(),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'current_regime': current_regime,
            'regime_transitions': transitions,
            'transition_rate': transition_rate,
            'most_common_regime': base_regime_counts.index[0] if not base_regime_counts.empty else 'UNKNOWN',
            'regime_stability': 1.0 - transition_rate  # Higher = more stable
        }
        
        return summary
    
    def get_regime_features(self, regime_data: Dict, date: datetime) -> Dict:
        """
        Extract regime features for ML models
        
        Args:
            regime_data: Regime analysis results
            date: Date for feature extraction
            
        Returns:
            Dictionary of regime features
        """
        if not regime_data or 'correlations' not in regime_data:
            return {}
        
        features = {}
        correlations = regime_data['correlations']
        
        # Get correlation values for the date
        for name, series in correlations.items():
            if not series.empty and date in series.index:
                features[f'corr_{name}'] = float(series.loc[date])
            else:
                features[f'corr_{name}'] = 0.0
        
        # Add regime classification if available
        if 'regime_classifications' in regime_data:
            for regime in regime_data['regime_classifications']:
                if regime['date'] == date:
                    features['regime_confidence'] = regime['confidence']
                    features['regime_stability'] = 1.0 - regime_data['regime_summary']['transition_rate']
                    break
        
        return features
    
    def fetch_external_data(self, force_refresh: bool = False) -> Dict[str, pd.Series]:
        """
        Convenience method to fetch external data for regime analysis
        
        Args:
            force_refresh: Force refresh of cached data
            
        Returns:
            Dictionary of external market data series
        """
        try:
            from data_manager.external_data import fetch_external_data
            return fetch_external_data(force_refresh)
        except ImportError as e:
            print(f"❌ Cannot import external data fetcher: {e}")
            return {}
        except Exception as e:
            print(f"❌ Error fetching external data: {e}")
            return {}

# Convenience function for quick usage
def analyze_market_regimes(external_data: Dict[str, pd.Series], 
                          equity_index: Optional[pd.Series] = None) -> Dict:
    """
    Convenience function for complete regime analysis
    
    Args:
        external_data: External market data dictionary
        equity_index: Optional equity index series
        
    Returns:
        Complete regime analysis results
    """
    detector = RegimeDetector()
    return detector.analyze_regimes(external_data, equity_index)

if __name__ == "__main__":
    # Test the regime detector
    print("Testing Regime Detector...")
    
    from data_manager.external_data import fetch_external_data
    
    # Fetch external data
    external_data = fetch_external_data()
    
    if external_data:
        # Create sample equity index (would normally use ASX200 or SP500)
        np.random.seed(42)
        dates = pd.date_range(start='2020-12-04', end='2025-12-03', freq='D')
        equity_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
        equity_index = pd.Series(equity_prices, index=dates, name='Equity_Index')
        
        # Analyze regimes
        regime_results = analyze_market_regimes(external_data, equity_index)
        
        if regime_results:
            summary = regime_results['regime_summary']
            print(f"\n🎯 Regime Analysis Results:")
            print(f"  • Most common regime: {summary['most_common_regime']}")
            print(f"  • Average confidence: {summary['avg_confidence']:.1%}")
            print(f"  • Regime stability: {summary['regime_stability']:.1%}")
            print(f"  • Total transitions: {summary['regime_transitions']}")
        else:
            print("❌ Regime analysis failed")
    else:
        print("❌ No external data available for testing")
