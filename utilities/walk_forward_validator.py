#!/usr/bin/env python3
"""
Walk-Forward Validation System
Validates ML forecasting models using sliding window approach
"""

# Fix imports for new folder structure
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from analyzers.forecasting_engine import ForecastingEngine


def validate_etf(ticker: str, train_days: int = 252, test_days: int = 60) -> Dict:
    """
    Walk-forward validation for single ETF
    
    Args:
        ticker: ETF ticker (e.g., 'VAS.AX')
        train_days: Training window size (default 252 = 1 year)
        test_days: Test window size (default 60 = 3 months)
        
    Returns:
        Dictionary with validation metrics
    """
    print(f"\nValidating {ticker}...")
    
    # Download data
    try:
        data = yf.download(ticker, period='max', progress=False)
        if data.empty or len(data) < train_days + test_days:
            return {'ticker': ticker, 'error': 'Insufficient data', 'mae': np.nan, 'hit_rate': np.nan}
        
        prices = data['Close'] if isinstance(data['Close'], pd.Series) else data['Close'].iloc[:, 0]
    except Exception as e:
        return {'ticker': ticker, 'error': str(e), 'mae': np.nan, 'hit_rate': np.nan}
    
    # Initialize forecasting engine
    engine = ForecastingEngine()
    
    # Walk-forward validation
    maes = []
    hits = []
    biases = []
    num_windows = 0
    
    # Sliding window
    start_idx = train_days
    end_idx = len(prices) - test_days
    step = test_days  # Move forward by test_days each time
    
    for i in range(start_idx, end_idx, step):
        try:
            # Training window
            train_prices = prices.iloc[i-train_days:i]
            
            # Generate forecast
            train_returns = train_prices.pct_change().dropna()
            if len(train_returns) < 30:
                continue
            
            # Simple trend forecast
            x = np.arange(len(train_returns))
            y = np.log(train_returns.values + 1)
            coeffs = np.polyfit(x, y, 1)
            forecast_return = (np.exp(coeffs[0] * test_days) - 1)
            
            # Actual return
            actual_price_start = prices.iloc[i]
            actual_price_end = prices.iloc[i + test_days]
            actual_return = (actual_price_end - actual_price_start) / actual_price_start
            
            # Calculate metrics
            mae = abs(forecast_return - actual_return)
            hit = 1 if (forecast_return > 0) == (actual_return > 0) else 0
            bias = forecast_return - actual_return
            
            maes.append(mae)
            hits.append(hit)
            biases.append(bias)
            num_windows += 1
            
        except Exception:
            continue
    
    if num_windows == 0:
        return {'ticker': ticker, 'error': 'No valid windows', 'mae': np.nan, 'hit_rate': np.nan}
    
    # Aggregate results
    return {
        'ticker': ticker,
        'avg_mae': np.mean(maes) * 100,  # Convert to percentage
        'std_mae': np.std(maes) * 100,
        'hit_rate': np.mean(hits),
        'avg_bias': np.mean(biases) * 100,
        'num_windows': num_windows,
        'error': None
    }


def save_validation_results(results: Dict, output_path: str = 'data/validation_results.json'):
    """Save validation results to JSON"""
    output = Path(output_path)
    output.parent.mkdir(exist_ok=True)
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved validation results to {output}")


def main():
    """Run walk-forward validation on sample ETFs"""
    import sys
    
    print("Walk-Forward Validation System")
    print("=" * 60)
    
    # Get tickers from command line or use defaults
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    else:
        tickers = ['VAS.AX', 'VGS.AX', 'NDQ.AX', 'A200.AX', 'IOZ.AX']
    
    print(f"\nValidating {len(tickers)} ETFs...")
    
    results = {}
    for ticker in tickers:
        result = validate_etf(ticker)
        results[ticker] = result
        
        if result.get('error'):
            print(f"  ✗ {ticker}: {result['error']}")
        else:
            print(f"  ✓ {ticker}: MAE={result['avg_mae']:.2f}%, Hit Rate={result['hit_rate']:.1%}, Windows={result['num_windows']}")
    
    # Save results
    save_validation_results(results)
    
    print("\n" + "=" * 60)
    print("Validation Complete")


if __name__ == "__main__":
    main()

