#!/usr/bin/env python3
"""
Phase 1.4-1.5: Test Parallel Analysis
Tests multiprocessing.Pool parallelization of ML Ensemble, Kalman Hull, Volume Intelligence
"""

import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from system.orchestrator import ETFAnalysisSystem
from analyzers.etf_risk_classifier import ETFRiskClassifier


def get_sample_etfs(num_etfs=5):
    """Get sample ETFs for testing"""
    sample_5 = ['VAS.AX', 'VGS.AX', 'NDQ.AX', 'MNRS.AX', 'HETH.AX']
    return sample_5[:num_etfs]


def run_test():
    """Run Phase 1.4-1.5 test"""
    print("="*80)
    print("PHASE 1.4-1.5: PARALLEL ANALYSIS TEST")
    print("="*80)
    print()

    # Get sample ETFs
    sample_tickers = get_sample_etfs(5)
    print(f"[SETUP] Testing with {len(sample_tickers)} ETFs: {sample_tickers}")
    print()

    # Initialize analyzer
    classifier = ETFRiskClassifier()
    system = ETFAnalysisSystem()

    # ========================================================================
    # SEQUENTIAL ANALYSIS (BASELINE)
    # ========================================================================
    print("[SEQUENTIAL] Running sequential analysis...")
    seq_start = time.time()

    # Classify ETFs
    seq_results, seq_summary = classifier.classify_etfs(sample_tickers)

    # Get classified groups
    seq_low = seq_results['low_risk_etfs']
    seq_medium = seq_results['medium_risk_etfs']
    seq_high = seq_results['high_risk_etfs']

    seq_time = time.time() - seq_start
    print(f"[SEQUENTIAL_COMPLETE] {seq_time:.2f}s (Low:{len(seq_low)}, Med:{len(seq_medium)}, High:{len(seq_high)})")
    print()

    # ========================================================================
    # PARALLEL ANALYSIS (NEW)
    # ========================================================================
    print("[PARALLEL] Running parallel analysis...")
    par_start = time.time()

    # Classify ETFs using parallel mode
    par_results, par_summary = classifier.classify_etfs_parallel(sample_tickers, max_workers=4)

    # Get classified groups
    par_low = par_results['low_risk_etfs']
    par_medium = par_results['medium_risk_etfs']
    par_high = par_results['high_risk_etfs']

    par_time = time.time() - par_start
    print(f"[PARALLEL_COMPLETE] {par_time:.2f}s (Low:{len(par_low)}, Med:{len(par_medium)}, High:{len(par_high)})")
    print()

    # ========================================================================
    # DETAILED RISK GROUP ANALYSIS (WITH PARALLEL analyze_risk_group_parallel)
    # ========================================================================
    print("[DETAILED] Testing analyze_risk_group_parallel()...")
    print()

    # Prepare risk groups for detailed analysis
    combined_groups = {**seq_high, **seq_medium}
    if len(combined_groups) < 3:
        combined_groups[sample_tickers[0]] = seq_high.get(sample_tickers[0]) or seq_medium.get(sample_tickers[0])

    if combined_groups:
        print(f"[ANALYZE] Running detailed parallel analysis on {len(combined_groups)} ETFs...")

        detailed_start = time.time()
        detailed_results = system.analyze_risk_group_parallel(
            combined_groups,
            risk_category='MIXED',
            max_workers=4
        )
        detailed_time = time.time() - detailed_start

        print(f"[ANALYZE_COMPLETE] {detailed_time:.2f}s for {len(detailed_results)} ETFs")
        print(f"  Avg per ETF: {detailed_time/len(detailed_results):.2f}s")
        print()

        # Show sample results
        print("[RESULTS_SAMPLE]")
        for i, (ticker, data) in enumerate(list(detailed_results.items())[:3]):
            print(f"  {ticker}:")
            print(f"    ML Forecast: {data.get('forecast_return', 0):.4f}")
            print(f"    Confidence: {data.get('confidence_score', 0):.2f}")
            print(f"    MAE Score: {data.get('mae_score', 'N/A')}")
            print(f"    Hit Rate: {data.get('hit_rate', 'N/A')}")
            print(f"    Trend: {data.get('trend', 0)}")
            print(f"    Signal Strength: {data.get('signal_strength', 0):.2f}")

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print()
    print("[COMPARISON]")
    print(f"  Sequential time: {seq_time:.2f}s")
    print(f"  Parallel time:   {par_time:.2f}s")
    print(f"  Speedup:         {seq_time/par_time:.2f}x")
    print()

    # Check consistency
    print("[CONSISTENCY]")
    consistency_check = {
        'low_match': len(seq_low) == len(par_low),
        'medium_match': len(seq_medium) == len(par_medium),
        'high_match': len(seq_high) == len(par_high),
    }

    for check, result in consistency_check.items():
        status = "PASS" if result else "FAIL"
        print(f"  {check}: {status}")

    # Final summary
    print()
    print("="*80)
    if all(consistency_check.values()):
        print("STATUS: PASS - Parallel implementation working correctly")
    else:
        print("STATUS: FAIL - Consistency check failed")
    print("="*80)


if __name__ == '__main__':
    run_test()
