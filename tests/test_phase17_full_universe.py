#!/usr/bin/env python3
"""
Phase 1.7: Full Universe Testing (385 ETFs)
Tests complete parallelization on all available ETFs with error handling validation
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
from data_manager.data_manager import ETFDataManager


def run_phase17_test():
    """Run Phase 1.7 full universe test"""
    print("=" * 80)
    print("PHASE 1.7: FULL UNIVERSE TESTING (385 ETFs)")
    print("=" * 80)
    print()

    # Get all ETFs from database
    etf_db = ETFDataManager()
    all_tickers = list(etf_db.etf_data.keys())
    print(f"[SETUP] Total ETFs in database: {len(all_tickers)}")
    print()

    # Test with progressively larger samples to measure scaling
    test_sizes = [11, 50, 100, len(all_tickers)]

    results_summary = {
        'timestamp': time.time(),
        'test_results': {}
    }

    for test_size in test_sizes:
        if test_size > len(all_tickers):
            continue

        sample_tickers = all_tickers[:test_size]
        print("=" * 80)
        print(f"TEST: {test_size} ETFs")
        print("=" * 80)
        print()

        # Initialize classifier and system
        classifier = ETFRiskClassifier()
        system = ETFAnalysisSystem()

        # Run classification with parallel mode
        print(f"[CLASSIFICATION] Classifying {test_size} ETFs (parallel mode)...")
        class_start = time.time()

        try:
            results, summary = classifier.classify_etfs_parallel(
                sample_tickers,
                max_workers=8
            )

            class_time = time.time() - class_start

            # Get risk counts
            low_count = summary['low_risk_count']
            med_count = summary['medium_risk_count']
            high_count = summary['high_risk_count']
            failed_count = len(summary['failed_downloads'])

            print(f"[CLASSIFICATION_COMPLETE] {class_time:.1f}s")
            print(f"  Low risk: {low_count}")
            print(f"  Medium risk: {med_count}")
            print(f"  High risk: {high_count}")
            print(f"  Failed: {failed_count}")
            print(f"  Speed: {test_size/class_time:.2f} ETFs/sec")
            print()

            # Categorize ETFs by risk for detailed analysis
            low_risk_etfs = results.get('low_risk_etfs', {})
            med_risk_etfs = results.get('medium_risk_etfs', {})
            high_risk_etfs = results.get('high_risk_etfs', {})

            # Run detailed analysis on each risk group
            print(f"[DETAILED_ANALYSIS] Running parallel analysis on risk groups...")
            analysis_start = time.time()

            group_results = {}
            total_results = 0

            for group_name, group_etfs in [
                ('LOW', low_risk_etfs),
                ('MEDIUM', med_risk_etfs),
                ('HIGH', high_risk_etfs)
            ]:
                if not group_etfs:
                    continue

                print(f"  Analyzing {group_name} risk group ({len(group_etfs)} ETFs)...")
                group_start = time.time()

                try:
                    group_analysis = system.analyze_risk_group_parallel(
                        group_etfs,
                        risk_category=group_name,
                        max_workers=8
                    )
                    group_time = time.time() - group_start
                    group_results[group_name] = {
                        'count': len(group_analysis),
                        'time': group_time,
                        'per_etf': group_time / len(group_analysis) if group_analysis else 0
                    }
                    total_results += len(group_analysis)

                except Exception as e:
                    print(f"  [ERROR] {group_name} group analysis failed: {str(e)}")
                    group_results[group_name] = {'error': str(e)}

            analysis_time = time.time() - analysis_start

            print()
            print(f"[DETAILED_ANALYSIS_COMPLETE] {analysis_time:.1f}s total for {total_results} ETFs")
            print()

            # Calculate statistics
            total_time = class_time + analysis_time
            success_rate = (test_size - failed_count) / test_size * 100 if test_size > 0 else 0

            # Store results
            results_summary['test_results'][test_size] = {
                'classification_time': class_time,
                'analysis_time': analysis_time,
                'total_time': total_time,
                'time_per_etf': total_time / (test_size - failed_count) if (test_size - failed_count) > 0 else 0,
                'low_risk': low_count,
                'medium_risk': med_count,
                'high_risk': high_count,
                'failed': failed_count,
                'success_rate': success_rate,
                'speed_etfs_per_sec': test_size / total_time if total_time > 0 else 0,
                'group_results': group_results
            }

            # Print summary for this test size
            print("[SUMMARY]")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Per ETF: {total_time / (test_size - failed_count):.2f}s")
            print(f"  Speed: {test_size/total_time:.2f} ETFs/sec")
            print(f"  Success rate: {success_rate:.1f}%")
            print()

        except Exception as e:
            print(f"[FATAL_ERROR] Test failed: {str(e)}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("=" * 80)
    print("PHASE 1.7 FINAL SUMMARY")
    print("=" * 80)
    print()

    for test_size, test_result in results_summary['test_results'].items():
        if 'error' in test_result:
            print(f"[{test_size} ETFs] FAILED: {test_result['error']}")
            continue

        print(f"[{test_size} ETFs]")
        print(f"  Classification: {test_result['classification_time']:.1f}s")
        print(f"  Analysis: {test_result['analysis_time']:.1f}s")
        print(f"  Total: {test_result['total_time']:.1f}s ({test_result['speed_etfs_per_sec']:.2f} ETFs/sec)")
        print(f"  Distribution: L:{test_result['low_risk']} M:{test_result['medium_risk']} H:{test_result['high_risk']}")
        print(f"  Success: {test_result['success_rate']:.1f}% ({test_result['failed']} failed)")
        print()

    # Save results
    results_file = Path('test_results_phase17_full_universe.json')
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results saved to: {results_file}")
    print()

    # Determine overall success
    largest_test = max([k for k in results_summary['test_results'].keys() if 'error' not in results_summary['test_results'][k]])
    largest_result = results_summary['test_results'][largest_test]

    print("=" * 80)
    if largest_result['success_rate'] >= 90:
        print("STATUS: PASS - Phase 1 parallelization validated on full universe")
        print(f"Achieved {largest_result['speed_etfs_per_sec']:.2f}x ETFs/sec processing speed")
    else:
        print(f"STATUS: WARNING - Success rate {largest_result['success_rate']:.1f}% below 90% threshold")
    print("=" * 80)

    return largest_result['success_rate'] >= 90


if __name__ == '__main__':
    try:
        success = run_phase17_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FATAL] Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
