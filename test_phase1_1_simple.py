#!/usr/bin/env python3
"""Test GrowthScoringSystem directly"""

from analyzers.scoring_system_growth import GrowthScoringSystem

print('✓ GrowthScoringSystem import successful')

scoring = GrowthScoringSystem()
print('✓ GrowthScoringSystem initialization successful')

# Test basic scoring functionality
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

result = scoring.calculate_composite_score(sample_data, 'MEDIUM')
print('✓ Scoring calculation successful')
print(f'  Composite Score: {result["composite_score"]:.1f}')
print(f'  Component Scores Available: {list(result["components"].keys())}')
print(f'  Risk Score: {result["components"]["risk"]:.1f}')
print(f'  Momentum Score: {result["components"]["momentum"]:.1f}')
print(f'  Forecast Score: {result["components"]["forecast"]:.1f}')
print(f'  Volume Score: {result["components"]["volume"]:.1f}')

print('\n✓ Phase 1.1 COMPLETE: Growth Scoring System integrated and functional')
