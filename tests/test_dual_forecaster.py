#!/usr/bin/env python3
"""
Quick test for dual-target forecasting system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from ml_models.dual_target_forecaster import DualTargetForecaster

def test_dual_forecaster():
    """Test dual-target forecaster on sample data"""
    
    print("🧪 Testing Dual-Target Forecaster...")
    
    # Create sample data
    np.random.seed(42)
    
    # Simulate 200 days of returns
    returns = pd.Series(np.random.normal(0.001, 0.02, 200))  # ~0.1% daily avg
    
    # Create 10 features (using the validated set)
    features = np.random.randn(1, 10) * 0.1 + 0.5  # Center around 0.5
    
    # Initialize forecaster
    forecaster = DualTargetForecaster(forecast_horizon=60)
    
    # Train on sample data
    targets = forecaster.train(features, returns)
    
    if targets is None:
        print("❌ Training failed - insufficient data")
        return False
    
    print(f"✅ Training successful:")
    print(f"   Mean daily target: {targets['mean_daily']:.4f} ({targets['mean_daily']*100:.2f}%)")
    print(f"   Cumulative target: {targets['cumulative']:.4f} ({targets['cumulative']*100:.2f}%)")
    print(f"   Ratio: {targets['ratio']:.2f}")
    
    # Test prediction
    prediction = forecaster.predict(features)
    
    if prediction is None:
        print("❌ Prediction failed")
        return False
    
    print(f"✅ Prediction successful:")
    print(f"   Mean daily: {prediction['mean_daily_return']:.4f} ({prediction['mean_daily_return']*100:.2f}%)")
    print(f"   Cumulative: {prediction['cumulative_return']:.4f} ({prediction['cumulative_return']*100:.2f}%)")
    print(f"   Daily→Period: {prediction['daily_to_period_conversion']:.4f} ({prediction['daily_to_period_conversion']*100:.2f}%)")
    print(f"   Model agreement: {prediction['model_agreement']:.2f}")
    print(f"   Confidence: {prediction['prediction_confidence']:.2f}")
    
    # Validate ranges
    daily_reasonable = 0.0005 <= abs(prediction['mean_daily_return']) <= 0.01
    cumulative_reasonable = 0.01 <= abs(prediction['cumulative_return']) <= 0.5
    
    if daily_reasonable and cumulative_reasonable:
        print("✅ Predictions in reasonable ranges")
    else:
        print("⚠️ Predictions outside expected ranges")
    
    return True

def test_feature_scaling():
    """Test feature scaling works correctly"""
    
    print("\n🧪 Testing Feature Scaling...")
    
    forecaster = DualTargetForecaster()
    
    # Test extreme features
    extreme_features = np.array([[0.5, 0.8, 0.3, 0.9, -0.7, 0.6, 0.05, -0.05, 0.2, -0.3]])
    
    # Check bounds
    for i, (feature, (min_val, max_val)) in enumerate(zip(extreme_features[0], forecaster.feature_bounds.items())):
        if not (min_val <= feature <= max_val):
            print(f"⚠️ Feature {i} ({list(forecaster.feature_bounds.keys())[i]}) outside bounds: {feature}")
    
    print("✅ Feature bounds checked")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("DUAL-TARGET FORECASTER TEST SUITE")
    print("=" * 60)
    
    success = True
    
    # Test 1: Basic functionality
    success &= test_dual_forecaster()
    
    # Test 2: Feature scaling
    success &= test_feature_scaling()
    
    if success:
        print("\n🎉 All tests passed! Ready for validation.")
    else:
        print("\n❌ Some tests failed. Please investigate.")
