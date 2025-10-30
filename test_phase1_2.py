#!/usr/bin/env python3
"""Test Phase 1.2: Component Score Storage"""

from analyzers.scoring_system_growth import GrowthScoringSystem

print("Testing Phase 1.2: Component Score Storage")
print("=" * 50)

# Test the scoring system directly
scoring = GrowthScoringSystem()

# Test data
sample_data = {
    'risk_score': 0.3,
    'kalman_trend': 1,
    'kalman_signal_strength': 0.7,
    'kalman_efficiency_ratio': 0.6,
    'kalman_divergence': 'none',
    'ml_forecast': 5.0,
    'ml_confidence': 0.6,
    'hit_rate': 0.65,
    'volume_spike_score': 60.0,
    'volume_correlation': 0.5,
    'volume_ad_signal': 'neutral',
    'cvar': -0.1,
    'avg_daily_volume': 500_000
}

# Calculate score
result = scoring.calculate_composite_score(sample_data, 'MEDIUM')

print("Scoring Result Structure:")
print(f"  Keys: {list(result.keys())}")
print()
print("Component Scores:")
print(f"  Risk: {result['components']['risk']:.1f}")
print(f"  Momentum: {result['components']['momentum']:.1f}")
print(f"  Forecast: {result['components']['forecast']:.1f}")
print(f"  Volume: {result['components']['volume']:.1f}")
print()
print("Adjusted Components (with risk multipliers):")
print(f"  Risk: {result['adjusted_components']['risk']:.1f}")
print(f"  Momentum: {result['adjusted_components']['momentum']:.1f}")
print(f"  Forecast: {result['adjusted_components']['forecast']:.1f}")
print(f"  Volume: {result['adjusted_components']['volume']:.1f}")
print()
print("Other Data:")
print(f"  Composite Score: {result['composite_score']:.1f}")
print(f"  Position Size: {result['position_size']:.3f}")
print(f"  Risk Category: {result['risk_category']}")

# Test rankings functionality
analysis_results = {
    'TEST1.AX': sample_data,
    'TEST2.AX': {**sample_data, 'risk_score': 0.6, 'kalman_trend': -1}  # Different profile
}

risk_classifications = {'TEST1.AX': 'MEDIUM', 'TEST2.AX': 'HIGH'}

rankings = scoring.rank_etfs_by_category(analysis_results, risk_classifications)

print()
print("Rankings Structure Test:")
for category, etf_list in rankings.items():
    print(f"  {category}: {len(etf_list)} ETFs")
    for ticker, score_result in etf_list[:2]:  # Show first 2
        print(f"    {ticker}: {score_result['composite_score']:.1f} (components: {list(score_result['components'].keys())})")

print()
print("✓ Phase 1.2 COMPLETE: Component scores are properly structured and accessible")
