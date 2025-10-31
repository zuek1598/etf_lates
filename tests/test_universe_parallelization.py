"""
Universe Parallelization Test
Tests parallel ETF downloads on 11-ETF sample and full universe
Validates that parallel processing produces identical results to sequential

Usage:
    python test_universe_parallelization.py               # Test 11-ETF sample
    python test_universe_parallelization.py --full        # Test all 385 ETFs
    python test_universe_parallelization.py --sample 50   # Test 50 ETFs
"""

import sys
from pathlib import Path
import time
import argparse
import json
from typing import Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analyzers.etf_risk_classifier import ETFRiskClassifier
from data_manager.data_manager import ETFDataManager


def get_sample_tickers(num_etfs: int = 11) -> list:
    """Get sample ETF tickers"""
    if num_etfs == 11:
        return [
            'VAS.AX', 'VGS.AX', 'NDQ.AX',  # LOW risk
            'MNRS.AX', 'HETH.AX', 'FOOD.AX', 'ROBO.AX',  # MEDIUM risk
            'CRYP.AX', 'URNM.AX', 'GDX.AX', 'ATEC.AX'  # HIGH risk
        ]
    else:
        # Return first n tickers from database
        etf_db = ETFDataManager()
        all_tickers = list(etf_db.etf_data.keys())
        return all_tickers[:num_etfs]


def extract_results_summary(results: Dict) -> Dict:
    """Extract summary from classification results"""
    return {
        'low_risk': set(results['low_risk_etfs'].keys()),
        'medium_risk': set(results['medium_risk_etfs'].keys()),
        'high_risk': set(results['high_risk_etfs'].keys())
    }


def compare_results(seq_summary: Dict, par_summary: Dict) -> Dict:
    """Compare sequential and parallel results"""
    return {
        'low_risk_match': seq_summary['low_risk'] == par_summary['low_risk'],
        'medium_risk_match': seq_summary['medium_risk'] == par_summary['medium_risk'],
        'high_risk_match': seq_summary['high_risk'] == par_summary['high_risk'],
        'total_match': (
            seq_summary['low_risk'] == par_summary['low_risk'] and
            seq_summary['medium_risk'] == par_summary['medium_risk'] and
            seq_summary['high_risk'] == par_summary['high_risk']
        ),
        'seq_summary': {
            'low': len(seq_summary['low_risk']),
            'medium': len(seq_summary['medium_risk']),
            'high': len(seq_summary['high_risk'])
        },
        'par_summary': {
            'low': len(par_summary['low_risk']),
            'medium': len(par_summary['medium_risk']),
            'high': len(par_summary['high_risk'])
        }
    }


def test_universe(num_etfs: int = 11, num_runs: int = 1):
    """
    Test parallelization on universe of ETFs

    Args:
        num_etfs: Number of ETFs to test (11, 50, or all available)
        num_runs: Number of times to run test for consistency checking
    """
    print("=" * 80)
    print(f"UNIVERSE PARALLELIZATION TEST ({num_etfs} ETFs, {num_runs} runs)")
    print("=" * 80)

    # Get sample ETFs
    print(f"\n[SETUP] Getting {num_etfs} sample ETFs...")
    sample_tickers = get_sample_tickers(num_etfs)
    print(f"  Tickers: {', '.join(sample_tickers[:5])}..." if num_etfs > 5 else f"  Tickers: {', '.join(sample_tickers)}")

    results_seq = None
    results_par = None

    for run in range(1, num_runs + 1):
        print(f"\n{'='*80}")
        print(f"RUN {run}/{num_runs}")
        print(f"{'='*80}")

        # Sequential classification
        print(f"\n[SEQUENTIAL] Testing sequential ETF classification...")
        classifier_seq = ETFRiskClassifier()

        start = time.time()
        results_seq, summary_seq = classifier_seq.classify_etfs(sample_tickers)
        time_seq = time.time() - start

        print(f"\n  Results Summary:")
        print(f"    - Low risk: {summary_seq['low_risk_count']}")
        print(f"    - Medium risk: {summary_seq['medium_risk_count']}")
        print(f"    - High risk: {summary_seq['high_risk_count']}")
        print(f"    - Failed: {len(summary_seq['failed_downloads'])}")
        print(f"    - Time: {time_seq:.2f} seconds")

        # Parallel classification
        print(f"\n[PARALLEL] Testing parallel ETF classification...")
        classifier_par = ETFRiskClassifier()

        start = time.time()
        results_par, summary_par = classifier_par.classify_etfs_parallel(
            sample_tickers,
            max_workers=8
        )
        time_par = time.time() - start

        print(f"\n  Results Summary:")
        print(f"    - Low risk: {summary_par['low_risk_count']}")
        print(f"    - Medium risk: {summary_par['medium_risk_count']}")
        print(f"    - High risk: {summary_par['high_risk_count']}")
        print(f"    - Failed: {len(summary_par['failed_downloads'])}")
        print(f"    - Time: {time_par:.2f} seconds")
        print(f"    - Speed: {num_etfs/time_par:.2f} ETFs/sec")

        # Compare results
        print(f"\n[COMPARISON] Comparing results...")
        seq_summary = extract_results_summary(results_seq)
        par_summary = extract_results_summary(results_par)
        comparison = compare_results(seq_summary, par_summary)

        print(f"\n  Consistency Check:")
        print(f"    - Low risk match: {'YES' if comparison['low_risk_match'] else 'NO'}")
        print(f"    - Medium risk match: {'YES' if comparison['medium_risk_match'] else 'NO'}")
        print(f"    - High risk match: {'YES' if comparison['high_risk_match'] else 'NO'}")

        if comparison['total_match']:
            print(f"\n  [PASS] Results are IDENTICAL!")
        else:
            print(f"\n  [WARNING] Results differ:")
            print(f"    Sequential: {comparison['seq_summary']}")
            print(f"    Parallel:   {comparison['par_summary']}")

        # Performance metrics
        speedup = time_seq / time_par if time_par > 0 else 0
        print(f"\n[PERFORMANCE]")
        print(f"  Sequential: {time_seq:.2f}s")
        print(f"  Parallel:   {time_par:.2f}s")
        print(f"  Speedup:    {speedup:.2f}x")

        if speedup > 1.0:
            print(f"  [PASS] Parallel is {speedup:.2f}x faster!")
        else:
            print(f"  [INFO] Parallel slightly slower (expected for small samples)")

    # Final summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")

    if comparison['total_match']:
        print(f"\n[PASS] All tests passed!")
        print(f"  - Results are consistent between sequential and parallel")
        print(f"  - Parallelization is working correctly")
        print(f"  - Ready for Phase 1.4-1.7 (ML and analysis loop parallelization)")
    else:
        print(f"\n[WARNING] Results differ between sequential and parallel")
        print(f"  Investigation needed before proceeding")

    # Save results to file
    results_file = Path('test_results_universe.json')
    test_data = {
        'num_etfs': num_etfs,
        'num_runs': num_runs,
        'time_sequential': time_seq,
        'time_parallel': time_par,
        'speedup': speedup,
        'results_match': comparison['total_match'],
        'sequential_summary': {k: len(v) for k, v in seq_summary.items()},
        'parallel_summary': {k: len(v) for k, v in par_summary.items()}
    }

    with open(results_file, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return comparison['total_match']


def main():
    parser = argparse.ArgumentParser(description='Test universe parallelization')
    parser.add_argument('--sample', type=int, default=11,
                        help='Number of ETFs to test (default: 11)')
    parser.add_argument('--full', action='store_true',
                        help='Test all 385 ETFs')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs for consistency testing')
    args = parser.parse_args()

    num_etfs = 11
    if args.full:
        etf_db = ETFDataManager()
        num_etfs = len(etf_db.etf_data)
    elif args.sample:
        num_etfs = args.sample

    print(f"\nPhase 1 - Universe Parallelization Test")
    print(f"======================================")
    print(f"Testing with: {num_etfs} ETFs")
    print(f"Runs: {args.runs}")
    print(f"\nNote: Full universe (385 ETFs) will take ~45-60 minutes")

    if num_etfs > 50:
        confirm = input("\nContinue with large dataset? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled")
            return 0

    try:
        success = test_universe(num_etfs=num_etfs, num_runs=args.runs)
        return 0 if success else 1
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
