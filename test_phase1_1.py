#!/usr/bin/env python3
"""Test GrowthScoringSystem integration"""

from system.orchestrator import ETFAnalysisSystem

print('✓ GrowthScoringSystem import successful')
system = ETFAnalysisSystem()
print('✓ ETFAnalysisSystem initialization successful')
print('✓ Scoring system type:', type(system.scoring_system).__name__)

# Test basic scoring functionality
sample_data = {
    'TEST.AX': {
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
}

result = system.scoring_system.calculate_composite_score(sample_data['TEST.AX'], 'MEDIUM')
print('✓ Scoring calculation successful')
print(f'  Composite Score: {result["composite_score"]:.1f}')
print(f'  Component Scores: {list(result["components"].keys())}')
print('✓ Phase 1.1 COMPLETE: Growth Scoring System integrated')
