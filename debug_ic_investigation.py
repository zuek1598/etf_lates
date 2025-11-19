"""
Deep investigation into why IC test is failing.
Tests the IC calculation step-by-step with detailed logging.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import yfinance as yf

from system.orchestrator import ETFAnalysisSystem
from data_manager.data_manager import ETFDataManager as ETFDatabase

def main():
    print("=" * 80)
    print("IC TEST INVESTIGATION - STEP BY STEP DEBUGGING")
    print("=" * 80)

    # Step 1: Load data
    print("\n[STEP 1] Loading ETF database and running analysis...")
    etf_db = ETFDatabase()
    all_tickers = list(etf_db.etf_data.keys())
    sample_tickers = all_tickers[:5]  # Use just 5 for debugging
    print(f"  Sample tickers: {sample_tickers}")

    orchestrator = ETFAnalysisSystem()
    sample_results = orchestrator.run_full_analysis(sample_tickers)
    analysis_results = sample_results.get('analysis_results', {})
    print(f"  Analysis results for {len(analysis_results)} ETFs")

    # Step 2: Extract factor values
    print("\n[STEP 2] Extracting factor values...")
    factor_names = ['ml_forecast', 'ml_confidence', 'mae_score']
    factor_values = {}

    for factor in factor_names:
        factor_values[factor] = {}
        for ticker in sample_tickers:
            if ticker in analysis_results:
                value = analysis_results[ticker].get(factor)
                if value is not None:
                    factor_values[factor][ticker] = value

    print(f"  Extracted factors:")
    for factor_name, values_dict in factor_values.items():
        print(f"    {factor_name}: {values_dict}")

    # Step 3: Normalize to Series
    print("\n[STEP 3] Normalizing factor_values to Series...")
    for factor_name, factor_data in factor_values.items():
        factor_values[factor_name] = pd.Series(factor_data)
        print(f"  {factor_name}:")
        print(f"    Type: {type(factor_values[factor_name])}")
        print(f"    Index: {list(factor_values[factor_name].index)}")
        print(f"    Values: {list(factor_values[factor_name].values)}")

    # Step 4: Download price data
    print("\n[STEP 4] Downloading price data...")
    price_data = {}
    for ticker in sample_tickers:
        try:
            hist = yf.download(ticker, period="2y", progress=False)
            if isinstance(hist, pd.DataFrame) and len(hist) > 0:
                if isinstance(hist.columns, pd.MultiIndex):
                    close_col = [col for col in hist.columns if col[0] == 'Close']
                    if close_col:
                        price_data[ticker] = hist[close_col[0]]
                elif 'Close' in hist.columns:
                    price_data[ticker] = hist['Close']
        except Exception as e:
            print(f"    Error downloading {ticker}: {e}")

    print(f"  Downloaded price data for {len(price_data)} tickers:")
    for ticker, prices in price_data.items():
        print(f"    {ticker}: {len(prices)} rows, date range {prices.index[0]} to {prices.index[-1]}")

    # Step 5: Test IC calculation for one factor
    print("\n[STEP 5] Testing IC calculation for ml_forecast...")
    factor_name = 'ml_forecast'
    factor_series = factor_values[factor_name]
    forward_days = 20

    print(f"\n  Factor series: {factor_series}")
    print(f"  Factor series index: {list(factor_series.index)}")
    print(f"  Factor series values: {list(factor_series.values)}")

    ics = []
    for ticker in factor_series.index:
        print(f"\n  Processing ticker: {ticker}")
        
        if ticker not in price_data:
            print(f"    [SKIP] Ticker not in price_data")
            continue
        
        factor_val = factor_series[ticker]
        prices = price_data[ticker]
        
        print(f"    Factor value: {factor_val} (type: {type(factor_val)})")
        print(f"    Price data shape: {prices.shape}")
        print(f"    Price data type: {type(prices)}")
        print(f"    Price data index type: {type(prices.index)}")
        print(f"    Price data first 3 rows:\n{prices.head(3)}")
        
        if pd.isna(factor_val):
            print(f"    [SKIP] Factor value is NaN")
            continue
        
        # Calculate forward returns
        forward_returns = prices.shift(-forward_days) / prices - 1
        print(f"    Forward returns shape: {forward_returns.shape}")
        print(f"    Forward returns first 3 rows:\n{forward_returns.head(3)}")
        print(f"    Forward returns last 3 rows:\n{forward_returns.tail(3)}")
        
        # Get common dates
        common_dates = forward_returns.dropna().index
        print(f"    Common dates (non-NaN forward returns): {len(common_dates)} dates")
        
        if len(common_dates) < 30:
            print(f"    [SKIP] Not enough common dates ({len(common_dates)} < 30)")
            continue
        
        returns_clean = forward_returns.loc[common_dates].dropna()
        print(f"    Returns clean shape: {returns_clean.shape}")
        print(f"    Returns clean first 3 rows:\n{returns_clean.head(3)}")
        print(f"    Returns clean stats: mean={returns_clean.mean():.6f}, std={returns_clean.std():.6f}")
        
        if len(returns_clean) >= 30:
            # Create factor values series
            factor_vals_clean = pd.Series([factor_val] * len(returns_clean), index=returns_clean.index)
            print(f"    Factor values clean: all values = {factor_val}")
            print(f"    Factor values clean shape: {factor_vals_clean.shape}")
            print(f"    Factor values clean std: {factor_vals_clean.std()}")
            
            # Check if we can compute correlation
            if factor_vals_clean.std() > 0:
                print(f"    [OK] Factor std > 0, computing correlation...")
                ic, p_value = spearmanr(factor_vals_clean, returns_clean)
                print(f"    IC: {ic}, p-value: {p_value}")
                if not np.isnan(ic):
                    ics.append({'ic': ic, 'p_value': p_value})
                    print(f"    [OK] IC added to list")
                else:
                    print(f"    [WARN] IC is NaN, not adding to list")
            else:
                print(f"    [SKIP] Factor std = 0 (constant value)")

    print(f"\n  Final IC list: {ics}")
    if ics:
        avg_ic = np.mean([r['ic'] for r in ics])
        print(f"  Average IC: {avg_ic}")
    else:
        print(f"  [RESULT] No valid IC values collected - INSUFFICIENT DATA")

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
