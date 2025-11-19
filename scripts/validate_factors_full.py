#!/usr/bin/env python3
"""
Phase 3.3: Factor Validator Test on Full Universe (377 ETFs)
Comprehensive factor validation on complete ETF universe.

This script:
1. Runs analysis on all available ETFs
2. Extracts factor values for each ETF
3. Gets price data for forward return calculations
4. Runs comprehensive factor validation (5 tests)
5. Generates validated_factors.json with final results
6. Confirms which factors are validated for production use

Expected runtime: 15-30 minutes (depending on system performance)

Output:
- config/validated_factors.json: Final list of validated factors
- Console report with IC, hit rate, quintile, correlation, decay results
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')


def run_factor_validation_full():
    """
    Run factor validation on full ETF universe (377 ETFs).

    Returns:
        dict: Validation results with validated factors and test results
    """
    print("="*80)
    print("PHASE 3.3: FACTOR VALIDATION TEST (FULL 377 ETFs)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Expected runtime: 15-30 minutes\n")

    # Step 1: Initialize systems
    print("Step 1: Initializing systems...")
    try:
        from system.orchestrator import ETFAnalysisSystem
        from data_manager.data_manager import ETFDataManager as ETFDatabase
        from analysis.factor_validator import FactorValidator
        print("  [OK] All imports successful\n")
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return None

    # Step 2: Load database and get all tickers
    print("Step 2: Loading ETF database...")
    try:
        etf_db = ETFDatabase()
        all_tickers = list(etf_db.etf_data.keys())
        print(f"  [OK] Loaded {len(all_tickers)} ETFs from universe\n")
    except Exception as e:
        print(f"  [FAIL] Database load failed: {e}")
        return None

    # Step 3: Run analysis on all ETFs
    print("Step 3: Running analysis on full ETF universe...")
    print(f"  (Analyzing {len(all_tickers)} ETFs - this will take 15-30 minutes)\n")
    try:
        orchestrator = ETFAnalysisSystem()
        full_results = orchestrator.run_full_analysis(all_tickers)
        analysis_results = full_results.get('analysis_results', {})
        print(f"\n  [OK] Analysis complete for {len(analysis_results)} ETFs\n")
    except Exception as e:
        print(f"  [FAIL] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Step 4: Extract factor values
    print("Step 4: Extracting factor values...")
    try:
        # Define factors to test (matching config/weights_config.json)
        factor_names = [
            'ml_forecast',
            'ml_confidence',
            'hit_rate',
            'mae_score',
            'kalman_signal_strength',
            'kalman_efficiency_ratio',
            'volume_correlation',
            'volume_spike_score'
        ]

        factor_values = {}  # {factor_name: {ticker: value}}

        for factor in factor_names:
            factor_values[factor] = {}
            for ticker in all_tickers:
                if ticker in analysis_results:
                    value = analysis_results[ticker].get(factor)
                    if value is not None:
                        factor_values[factor][ticker] = value

        print(f"  [OK] Extracted {len(factor_names)} factors")
        for factor_name, values_dict in factor_values.items():
            valid_count = len([v for v in values_dict.values() if v is not None])
            print(f"      - {factor_name}: {valid_count}/{len(all_tickers)} valid values")
        print()
    except Exception as e:
        print(f"  [FAIL] Factor extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Step 5: Get price data for forward returns
    print("Step 5: Downloading price data for forward return calculation...")
    try:
        import yfinance as yf
        price_data = {}  # {ticker: close_prices_series}

        for i, ticker in enumerate(all_tickers):
            if (i + 1) % 50 == 0:
                print(f"  Downloading... {i+1}/{len(all_tickers)} ETFs")
            try:
                # Download 2 years of data for reliable forward return calculations
                hist = yf.download(ticker, period="2y", progress=False, quiet=True)
                if isinstance(hist, pd.DataFrame) and 'Close' in hist.columns and len(hist) > 0:
                    price_data[ticker] = hist['Close']
            except Exception as e:
                # Skip failed downloads
                pass

        print(f"  [OK] Downloaded price data for {len(price_data)} ETFs\n")
    except Exception as e:
        print(f"  [FAIL] Price data load failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Step 6: Run comprehensive factor validation
    print("Step 6: Running comprehensive factor validation...")
    print("  (This tests: IC, Hit Rate, Quintile, Correlation, Decay)\n")

    try:
        validator = FactorValidator()
        validation_results = validator.run_comprehensive_validation(
            factor_values,
            price_data,
            forward_days=20
        )
        print("  [OK] Full validation complete\n")
    except Exception as e:
        print(f"  [FAIL] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Step 7: Print validation summary
    print("="*80)
    print("FULL VALIDATION RESULTS (377 ETFs)")
    print("="*80)

    try:
        validator.print_summary(validation_results)
    except Exception as e:
        print(f"Warning: Could not print full summary: {e}")
        # Print basic results anyway
        print("\nValidated Factors:")
        for factor in validation_results.get('validated_factors', []):
            print(f"  ✓ {factor}")

        print("\nRejected Factors:")
        for factor in validation_results.get('rejected_factors', []):
            print(f"  ✗ {factor}")

    # Step 8: Export results to config/validated_factors.json
    print("\nStep 7: Exporting validation results...")
    try:
        config_path = Path('config/validated_factors.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Save validation results
        validator.export_validation_results(validation_results)
        print(f"  [OK] Results saved to {config_path}")
        print(f"  [OK] config/validated_factors.json ready for Phase 4\n")
    except Exception as e:
        print(f"  [FAIL] Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Step 9: Summary statistics
    print("="*80)
    print("FULL VALIDATION SUMMARY")
    print("="*80)
    print(f"ETFs Tested:           {len(all_tickers)} (full universe)")
    print(f"Factors Tested:        {len(factor_names)}")
    print(f"Validated Factors:     {len(validation_results.get('validated_factors', []))}")
    print(f"Rejected Factors:      {len(validation_results.get('rejected_factors', []))}")
    print(f"Redundant Factors:     {len(validation_results.get('redundant_factors', []))}")

    validated = validation_results.get('validated_factors', [])
    if len(validated) >= 4:
        print(f"\n✓ SUCCESS: {len(validated)} factors validated (threshold: >= 4)")
        print("  ➜ Ready to proceed to Phase 4 (Orchestrator Integration)")
    else:
        print(f"\n⚠ WARNING: Only {len(validated)} factors validated (threshold: >= 4)")
        print("  ➜ Consider relaxing IC threshold from 0.02 to 0.01 if needed")

    print("\nTop Factors by IC:")
    ic_results = validation_results.get('ic_results', {})
    for factor_name, stats in sorted(ic_results.items(),
                                     key=lambda x: x[1].get('ic', 0),
                                     reverse=True)[:5]:
        ic = stats.get('ic', 0)
        status = stats.get('status', 'UNKNOWN')
        pvalue = stats.get('pvalue', 1.0)
        print(f"  - {factor_name}: IC={ic:.4f} (p={pvalue:.4f}) [{status}]")

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    return validation_results


def main():
    """Main entry point for full factor validation."""
    results = run_factor_validation_full()

    if results is None:
        print("Full validation failed!")
        return 1

    validated_count = len(results.get('validated_factors', []))
    if validated_count >= 4:
        print("✓ Ready for Phase 4 (Orchestrator Integration with Validated Factors)")
        return 0
    else:
        print(f"⚠ {validated_count} factors validated. Consider next steps carefully.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
