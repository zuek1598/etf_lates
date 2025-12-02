#!/usr/bin/env python3
"""
Performance Analysis Script
Identifies bottlenecks in ETF analysis system
"""
import time
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.orchestrator import ETFAnalysisSystem
from data_manager.data_manager import ETFDataManager

def analyze_performance():
    print("🔍 PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Initialize system
    start_init = time.time()
    system = ETFAnalysisSystem()
    init_time = time.time() - start_init
    print(f"⚡ System Initialization: {init_time:.2f}s")
    
    # Get ETF list
    data_manager = ETFDataManager()
    all_tickers = data_manager.get_all_tickers()
    
    # Test with different sample sizes
    sample_sizes = [10, 50, 100]
    
    for size in sample_sizes:
        print(f"\n📊 Testing with {size} ETFs:")
        print("-" * 30)
        
        # Sample ETFs
        sample_tickers = all_tickers[:size]
        etf_data = {}
        
        # Load data
        start_load = time.time()
        for ticker in sample_tickers:
            try:
                data = data_manager.get_etf_data(ticker)
                if data is not None and len(data) > 100:
                    etf_data[ticker] = {'data': data}
            except:
                continue
        load_time = time.time() - start_load
        print(f"  Data Loading: {load_time:.2f}s ({len(etf_data)} ETFs)")
        
        if len(etf_data) == 0:
            continue
            
        # Analyze risk categories
        start_classify = time.time()
        risk_classifications = system.classify_etfs_by_risk(list(etf_data.keys()))
        classify_time = time.time() - start_classify
        print(f"  Risk Classification: {classify_time:.2f}s")
        
        # Analyze each risk category
        total_analysis_time = 0
        for risk_category, etfs in risk_classifications.items():
            if etfs:
                start_analysis = time.time()
                results = system.analyze_risk_group_parallel(etfs, risk_category, max_workers=2)
                analysis_time = time.time() - start_analysis
                total_analysis_time += analysis_time
                print(f"  {risk_category} Analysis: {analysis_time:.2f}s ({len(etfs)} ETFs)")
        
        print(f"  📈 Total Analysis: {total_analysis_time:.2f}s")
        print(f"  📊 Time per ETF: {total_analysis_time/len(etf_data):.2f}s")
        
        # Extrapolate to full universe
        estimated_full_time = total_analysis_time * (377 / len(etf_data))
        print(f"  🎯 Estimated Full 377 ETFs: {estimated_full_time:.1f}s")
        
    print(f"\n🎯 PERFORMANCE INSIGHTS:")
    print("=" * 50)
    
    # Test individual components
    print("\n🔬 Component Analysis:")
    
    # Test ML Ensemble
    ticker = list(etf_data.keys())[0] if etf_data else 'VAS.AX'
    if ticker in etf_data:
        print(f"  Testing {ticker}...")
        
        # ML Ensemble
        start_ml = time.time()
        ml_result = system.ml_ensemble.forecast_etf(ticker, etf_data[ticker]['data'])
        ml_time = time.time() - start_ml
        print(f"    ML Ensemble: {ml_time:.3f}s")
        
        # Kalman Hull
        start_kalman = time.time()
        from utilities.shared_utils import extract_column
        prices = extract_column(etf_data[ticker]['data'], 'Close')
        volume = extract_column(etf_data[ticker]['data'], 'Volume')
        kalman_result = system.calculate_kalman_hull(ticker, prices, volume, 'MEDIUM', etf_data[ticker]['data'])
        kalman_time = time.time() - start_kalman
        print(f"    Kalman Hull: {kalman_time:.3f}s")
        
        # Risk Component
        start_risk = time.time()
        risk_result = system.risk_component.calculate_risk_scores(
            etf_data[ticker]['data'], 
            {'name': ticker, 'sector': 'Unknown'},
            system.vix_data, 
            system.benchmark_data.get('ASX200')
        )
        risk_time = time.time() - start_risk
        print(f"    Risk Component: {risk_time:.3f}s")

if __name__ == "__main__":
    analyze_performance()
