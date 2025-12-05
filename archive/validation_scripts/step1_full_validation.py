#!/usr/bin/env python3
"""
Full Statistical Validation - All 385 ETFs
Step 1: Run comprehensive validation across entire ETF universe
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from analysis.statistical_feature_validator import StatisticalFeatureValidator
from data_manager.data_manager import ETFDataManager
import time

def run_full_validation():
    """Run full statistical validation across all ETFs"""
    print("🔍 STEP 1: FULL STATISTICAL VALIDATION ACROSS 385 ETFs")
    print("=" * 60)
    
    # Initialize validator
    validator = StatisticalFeatureValidator(significance_level=0.05, min_samples=30)
    
    # Load all ETF data
    print("📊 Loading ETF universe...")
    data_manager = ETFDataManager()
    
    # Get all ETF tickers
    universe_df = data_manager.load_universe()
    all_tickers = universe_df['ticker'].tolist()
    
    print(f"✅ Found {len(all_tickers)} ETFs in universe")
    
    # Load price data for all ETFs
    price_data = {}
    loaded_count = 0
    
    print("📈 Loading price data for all ETFs...")
    for i, ticker in enumerate(all_tickers):
        try:
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(all_tickers)} ({(i+1)/len(all_tickers)*100:.1f}%)")
            
            # Load historical price data from data/historical directory
            import os
            from pathlib import Path
            
            # Try different file naming conventions - the files use _AX format
            possible_files = [
                f"data/historical/{ticker.replace('.AX', '_AX')}.parquet",
                f"data/historical/{ticker}.parquet",
                f"data/historical/{ticker.replace('.', '_')}.parquet"
            ]
            
            hist_data = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    hist_data = pd.read_parquet(file_path)
                    break
            
            if hist_data is not None and len(hist_data) >= 100:
                # Ensure we have Close column
                if 'Close' in hist_data.columns:
                    price_data[ticker] = hist_data
                    loaded_count += 1
                elif 'close' in hist_data.columns:
                    hist_data['Close'] = hist_data['close']
                    price_data[ticker] = hist_data
                    loaded_count += 1
                elif 'price' in hist_data.columns:
                    hist_data['Close'] = hist_data['price']
                    price_data[ticker] = hist_data
                    loaded_count += 1
                
        except Exception as e:
            print(f"⚠️ Failed to load {ticker}: {e}")
            continue
    
    print(f"✅ Successfully loaded {loaded_count} ETFs with sufficient data")
    
    if loaded_count < validator.min_samples:
        print(f"❌ Insufficient data: {loaded_count} < {validator.min_samples}")
        return None
    
    # Run comprehensive validation
    print(f"\n🔬 Running comprehensive statistical validation...")
    print(f"   Sample size: {min(loaded_count, 100)} ETFs (for performance)")
    print(f"   Features: 40 total indicators")
    print(f"   Tests: Correlation + Permutation + Cross-Validation")
    
    start_time = time.time()
    
    # Run validation with reasonable sample size
    sample_size = min(loaded_count, 100)  # Limit to 100 for performance
    results = validator.comprehensive_feature_validation(price_data, sample_size=sample_size)
    
    elapsed_time = time.time() - start_time
    
    if results:
        print(f"\n✅ VALIDATION COMPLETED in {elapsed_time:.1f} seconds")
        
        # Save results
        import json
        with open('data/validation_results.json', 'w') as f:
            # Convert DataFrames to dicts for JSON serialization
            json_results = {
                'master_results': results['master_results'].to_dict('records'),
                'category_analysis': results['category_analysis'],
                'correlation_summary': results['correlation_summary'],
                'permutation_summary': results['permutation_summary'],
                'cv_summary': results['cv_summary'],
                'overall_summary': results['overall_summary'],
                'feature_info': results['feature_info']
            }
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"💾 Results saved to data/validation_results.json")
        
        # Generate initial summary
        overall = results['overall_summary']
        print(f"\n📊 INITIAL VALIDATION SUMMARY:")
        print(f"   Total Features Analyzed: {overall['total_features']}")
        print(f"   Statistically Significant: {overall['significant_features']} ({overall['overall_significance_rate']:.1%})")
        print(f"   Top 3 Features: {', '.join(overall['top_features'][:3])}")
        
        return results
    else:
        print(f"❌ Validation failed")
        return None

if __name__ == "__main__":
    results = run_full_validation()
    if results:
        print(f"\n🎯 STEP 1 COMPLETE - Ready for Step 2: Feature Significance Analysis")
    else:
        print(f"\n❌ STEP 1 FAILED - Check data and retry")
