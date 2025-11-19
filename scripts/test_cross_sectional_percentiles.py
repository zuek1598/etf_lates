#!/usr/bin/env python3
"""
Quick test of cross-sectional percentiles on a small sample.
Verifies that composite scores are no longer 0.0 and rankings are populated.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings('ignore')


def test_cross_sectional_ranking():
    """Test cross-sectional percentile ranking on 10 sample ETFs"""

    print("="*60)
    print("TESTING CROSS-SECTIONAL PERCENTILES")
    print("="*60)

    try:
        from system.orchestrator import ETFAnalysisSystem
        from data_manager.data_manager import ETFDataManager as ETFDatabase

        print("\n1. Initializing systems...")
        orchestrator = ETFAnalysisSystem()
        etf_db = ETFDatabase()
        all_tickers = list(etf_db.etf_data.keys())

        # Test on 10 sample ETFs (fast)
        sample_tickers = all_tickers[:10]
        print(f"   Testing on 10 sample ETFs: {sample_tickers}\n")

        print("2. Running analysis...")
        results = orchestrator.run_full_analysis(sample_tickers)

        analysis_results = results.get('analysis_results', {})
        print(f"   Analysis complete for {len(analysis_results)} ETFs\n")

        print("3. Checking composite percentiles...\n")

        percentiles = []
        for ticker in sample_tickers:
            if ticker in analysis_results:
                composite = analysis_results[ticker].get('composite_percentile', None)
                percentiles.append((ticker, composite))
                status = "✓" if composite is not None and composite != 0.0 else "✗"
                print(f"   {status} {ticker}: {composite}")

        # Check if all percentiles are non-zero and populated
        valid_percentiles = [p for _, p in percentiles if p is not None and p != 0.0]

        print(f"\n4. Results Summary:")
        print(f"   Total ETFs: {len(percentiles)}")
        print(f"   Non-zero percentiles: {len(valid_percentiles)}")
        print(f"   Percentage: {len(valid_percentiles)/len(percentiles)*100:.1f}%")

        if len(valid_percentiles) > 5:
            print(f"\n✓ SUCCESS: Cross-sectional percentiles working!")
            print(f"✓ Composite scores are properly populated (not 0.0)")
            return True
        else:
            print(f"\n✗ FAILED: Most percentiles are 0.0 or NaN")
            print(f"✗ Cross-sectional percentiles not working properly")
            return False

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cross_sectional_ranking()
    sys.exit(0 if success else 1)
