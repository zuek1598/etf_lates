#!/usr/bin/env python3
"""
Phase 3.2: Factor Validator Test on Sample ETFs (50 sample)
Tests which factors have predictive power before using in final ranking system.

This script:
1. Runs analysis on first 50 ETFs
2. Extracts factor values for each ETF
3. Gets price data for forward return calculations
4. Runs comprehensive factor validation (5 tests)
5. Generates validated_factors.json with results
6. Reports which factors pass validation thresholds

Expected output:
- IC test: Factor correlation with forward returns (threshold: IC > 0.02)
- Hit rate: Directional accuracy (threshold: > 52%)
- Quintile: Monotonic relationship (Q1 < Q2 < Q3 < Q4 < Q5)
- Correlation: Factor redundancy (flag if corr > 0.70)
- Decay: Optimal holding period (5, 10, 20, 40, 60 days)
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import warnings
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')


def run_factor_validation_sample():
    """
    Run factor validation on 50 sample ETFs.

    Returns:
        dict: Validation results with validated factors and test results
    """
    print("="*80)
    print("PHASE 3.2: FACTOR VALIDATION TEST (50 SAMPLE ETFs)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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

    # Step 2: Load database and get sample tickers
    print("Step 2: Loading ETF database...")
    try:
        etf_db = ETFDatabase()
        all_tickers = list(etf_db.etf_data.keys())
        sample_tickers = all_tickers[:50]  # First 50 ETFs
        print(f"  [OK] Loaded {len(all_tickers)} ETFs")
        print(f"  [OK] Selected sample of {len(sample_tickers)} ETFs for testing\n")
    except Exception as e:
        print(f"  [FAIL] Database load failed: {e}")
        return None

    # Step 3: Run analysis on sample
    print("Step 3: Running analysis on sample ETFs...")
    print(f"  (This may take 2-3 minutes for 50 ETFs)\n")
    try:
        orchestrator = ETFAnalysisSystem()
        sample_results = orchestrator.run_full_analysis(sample_tickers)
        analysis_results = sample_results.get('analysis_results', {})
        print(f"  [OK] Analysis complete for {len(analysis_results)} ETFs\n")
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
            for ticker in sample_tickers:
                if ticker in analysis_results:
                    value = analysis_results[ticker].get(factor)
                    if value is not None:
                        factor_values[factor][ticker] = value

        print(f"  [OK] Extracted {len(factor_names)} factors")
        for factor_name, values_dict in factor_values.items():
            valid_count = len([v for v in values_dict.values() if v is not None])
            print(f"      - {factor_name}: {valid_count}/{len(sample_tickers)} valid values")
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

        for i, ticker in enumerate(sample_tickers):
            try:
                # Download 2 years of data for reliable forward return calculations
                hist = yf.download(ticker, period="2y", progress=False)
                if isinstance(hist, pd.DataFrame) and len(hist) > 0:
                    # Handle MultiIndex columns from yfinance
                    if isinstance(hist.columns, pd.MultiIndex):
                        # Columns are ('Close', 'TICKER'), ('High', 'TICKER'), etc.
                        close_col = [col for col in hist.columns if col[0] == 'Close']
                        if close_col:
                            price_data[ticker] = hist[close_col[0]]
                    elif 'Close' in hist.columns:
                        price_data[ticker] = hist['Close']
            except Exception as e:
                # Skip failed downloads - print error for debugging
                print(f"    Error downloading {ticker}: {e}")
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Downloaded {i+1}/{len(sample_tickers)} ETFs... ({len(price_data)} successful)")

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
        print("  [OK] Validation complete\n")
    except Exception as e:
        print(f"  [FAIL] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Step 7: Print validation summary
    print("="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    try:
        validator.print_summary(validation_results)
    except Exception as e:
        print(f"Warning: Could not print full summary: {e}")
        # Print basic results anyway
        print("\nValidated Factors:")
        for factor in validation_results.get('validated_factors', []):
            print(f"  [OK] {factor}")

        print("\nRejected Factors:")
        for factor in validation_results.get('rejected_factors', []):
            print(f"  [FAIL] {factor}")

    # Step 8: Export results
    print("\nStep 7: Exporting validation results...")
    try:
        config_path = Path('config/validated_factors.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Save validation results
        validator.export_validation_results(validation_results)
        print(f"  [OK] Results saved to {config_path}\n")
    except Exception as e:
        print(f"  [FAIL] Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Step 9: Summary statistics
    print("="*80)
    print("SAMPLE VALIDATION SUMMARY")
    print("="*80)
    print(f"ETFs Tested:           {len(sample_tickers)} (sample)")
    print(f"Factors Tested:        {len(factor_names)}")
    print(f"Validated Factors:     {len(validation_results.get('validated_factors', []))}")
    print(f"Rejected Factors:      {len(validation_results.get('rejected_factors', []))}")
    print(f"Redundant Factors:     {len(validation_results.get('redundant_factors', []))}")

    validated = validation_results.get('validated_factors', [])
    if len(validated) >= 3:
        print(f"\n[OK] SUCCESS: {len(validated)} factors validated (threshold: >= 3)")
        print("  Ready to proceed to Phase 3.3 (full validation)")
    else:
        print(f"\n[WARN] WARNING: Only {len(validated)} factors validated (threshold: >= 3)")
        print("  Consider relaxing thresholds or investigating factor quality")

    print("\nTop Factors by IC:")
    ic_results = validation_results.get('ic_results', {})
    for factor_name, stats in sorted(ic_results.items(),
                                     key=lambda x: x[1].get('ic', 0),
                                     reverse=True)[:5]:
        ic = stats.get('ic', 0)
        status = stats.get('status', 'UNKNOWN')
        print(f"  - {factor_name}: IC={ic:.4f} [{status}]")

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    return validation_results


def main():
    """Main entry point for sample factor validation."""
    results = run_factor_validation_sample()

    if results is None:
        print("Sample validation failed!")
        return 1

    validated_count = len(results.get('validated_factors', []))
    if validated_count >= 3:
        print(f"\n[OK] Ready for Phase 3.3 (Full validation on 377 ETFs)")
        return 0
    else:
        print(f"\n[WARN] Only {validated_count} factors validated. Review results before continuing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
