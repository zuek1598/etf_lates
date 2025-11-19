#!/usr/bin/env python3
"""
Quick integration test for Phase 2.3
Verifies that orchestrator imports and initializes correctly
"""

import sys
import os
from pathlib import Path

# Set output encoding to UTF-8
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all imports work without errors"""
    print("Testing imports...")

    try:
        from analyzers.percentile_ranker import PercentileRanker
        print("  [OK] PercentileRanker imported")
    except Exception as e:
        print(f"  [FAIL] PercentileRanker import failed: {e}")
        return False

    try:
        from analysis.factor_validator import FactorValidator
        print("  [OK] FactorValidator imported")
    except Exception as e:
        print(f"  [FAIL] FactorValidator import failed: {e}")
        return False

    try:
        from system.orchestrator import ETFAnalysisSystem
        print("  [OK] ETFAnalysisSystem imported")
    except Exception as e:
        print(f"  [FAIL] ETFAnalysisSystem import failed: {e}")
        return False

    return True


def test_orchestrator_init():
    """Test that orchestrator initializes without errors"""
    print("\nTesting orchestrator initialization...")

    try:
        from system.orchestrator import ETFAnalysisSystem
        system = ETFAnalysisSystem()

        # Check that percentile_ranker exists
        if hasattr(system, 'percentile_ranker'):
            print("  [OK] percentile_ranker initialized")
        else:
            print("  [FAIL] percentile_ranker not found")
            return False

        # Check that old scoring_system is gone
        if hasattr(system, 'scoring_system'):
            print("  [WARN] Old scoring_system still exists")
        else:
            print("  [OK] Old scoring_system removed")

        return True
    except Exception as e:
        print(f"  [FAIL] Orchestrator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_percentile_ranker():
    """Test basic PercentileRanker functionality"""
    print("\nTesting PercentileRanker functionality...")

    try:
        from analyzers.percentile_ranker import PercentileRanker
        import pandas as pd
        import numpy as np

        ranker = PercentileRanker(lookback_days=252)
        print("  [OK] PercentileRanker instantiated")

        # Test with dummy data
        test_metric = 0.5
        test_history = pd.Series(np.random.randn(100) + 0.5)
        percentile = ranker.calculate_percentile(test_metric, test_history, 'test_metric')

        if 0 <= percentile <= 100 or pd.isna(percentile):
            print(f"  [OK] Percentile calculation works (result: {percentile:.1f})")
        else:
            print(f"  [FAIL] Percentile out of range: {percentile}")
            return False

        return True
    except Exception as e:
        print(f"  [FAIL] PercentileRanker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weights_config():
    """Test weights configuration"""
    print("\nTesting weights configuration...")

    try:
        from pathlib import Path
        import json

        config_path = Path('config/weights_config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

            if 'weighting_mode' in config and 'factor_weights' in config:
                print(f"  [OK] weights_config.json valid (mode: {config['weighting_mode']})")
                print(f"  [OK] {len(config['factor_weights'])} factors configured")
                return True
            else:
                print("  [FAIL] weights_config.json missing required fields")
                return False
        else:
            print("  [FAIL] weights_config.json not found")
            return False
    except Exception as e:
        print(f"  [FAIL] Weights config test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("="*60)
    print("PHASE 2.3 INTEGRATION TEST")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Orchestrator Init", test_orchestrator_init),
        ("PercentileRanker", test_percentile_ranker),
        ("Weights Config", test_weights_config),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nFATAL ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{name:30s} {status}")

    total_pass = sum(1 for _, r in results if r)
    total_tests = len(results)

    print(f"\n{total_pass}/{total_tests} tests passed")

    if total_pass == total_tests:
        print("\n[SUCCESS] All integration tests passed! Ready for Phase 3.2")
        return 0
    else:
        print(f"\n[ERROR] {total_tests - total_pass} test(s) failed. Fix issues before proceeding.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
