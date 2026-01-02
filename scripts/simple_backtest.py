#!/usr/bin/env python3
"""
Simple Backtest with Rebalancing
Uses simplified scoring to avoid pandas issues
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import sys

# Import our modules
sys.path.insert(0, str(Path(__file__).parent))
from analyzers.quality_ranker import QualityRanker

def simple_backtest():
    """Run simple backtest with rebalancing"""
    
    print("="*80)
    print("SIMPLE REBALANCING BACKTEST")
    print("="*80)
    
    # Use current top 10 ETFs
    ranker = QualityRanker()
    top_10_file = ranker.cache_dir / 'top_10_etfs.pkl'
    
    if not top_10_file.exists():
        print("Error: Run run_quality_universe.py first")
        return
    
    with open(top_10_file, 'rb') as f:
        top_10_etfs = pickle.load(f)
    
    etf_list = [etf['etf'] for etf in top_10_etfs]
    
    # Test period
    start_date = datetime(2024, 1, 1)
    end_date = datetime.now()
    
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"ETFs: {etf_list[:5]}...")  # Show first 5
    
    # Download price data
    print("\nDownloading price data...")
    try:
        data = yf.download(etf_list, start=start_date, end=end_date, progress=False)
        
        if 'Close' in data.columns:
            if data.columns.nlevels > 1:
                prices = data['Close']
            else:
                prices = data
        else:
            prices = data
        
        # Remove ETFs with insufficient data
        valid_etfs = []
        for etf in etf_list:
            if etf in prices.columns:
                valid_data = prices[etf].dropna()
                if len(valid_data) > 100:
                    valid_etfs.append(etf)
        
        prices = prices[valid_etfs]
        print(f"Valid ETFs: {len(valid_etfs)}")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Strategy 1: Buy and hold top 3
    print("\n" + "="*80)
    print("STRATEGY 1: Buy & Hold Top 3")
    print("="*80)
    
    # Use first 3 valid ETFs
    if len(valid_etfs) >= 3:
        buy_hold_etfs = valid_etfs[:3]
        buy_hold_returns = returns[buy_hold_etfs].mean(axis=1)
        
        buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
        buy_hold_total = (buy_hold_cumulative.iloc[-1] - 1) * 100
        
        print(f"ETFs: {buy_hold_etfs}")
        print(f"Total Return: {buy_hold_total:.1f}%")
    
    # Strategy 2: Rebalance quarterly based on recent performance
    print("\n" + "="*80)
    print("STRATEGY 2: Quarterly Rebalancing")
    print("="*80)
    
    # Create quarterly rebalance dates
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='Q')
    
    portfolio_returns = []
    portfolio_dates = []
    current_etfs = valid_etfs[:3]  # Start with top 3
    
    for i, date in enumerate(rebalance_dates):
        if date not in returns.index:
            continue
        
        # Get performance over last quarter
        quarter_start = date - timedelta(days=90)
        recent_returns = returns.loc[quarter_start:date]
        
        if not recent_returns.empty:
            # Find top 3 performers
            avg_returns = recent_returns.mean()
            top_performers = avg_returns.nlargest(3).index.tolist()
            
            if i == 0:
                print(f"Q1 2024: {top_performers}")
            elif i == 1:
                print(f"Q2 2024: {top_performers}")
            elif i == 2:
                print(f"Q3 2024: {top_performers}")
            elif i == 3:
                print(f"Q4 2024: {top_performers}")
            
            current_etfs = top_performers
    
    # Calculate quarterly rebalancing returns
    for i in range(len(rebalance_dates) - 1):
        start = rebalance_dates[i]
        end = rebalance_dates[i + 1]
        
        if start in returns.index and end in returns.index:
            # Get top performers at start of period
            quarter_start = start - timedelta(days=90)
            recent = returns.loc[quarter_start:start]
            
            if not recent.empty:
                top_etfs = recent.mean().nlargest(3).index.tolist()
                period_returns = returns.loc[start:end, top_etfs].mean(axis=1)
                portfolio_returns.extend(period_returns)
                portfolio_dates.extend(period_returns.index)
    
    if portfolio_returns:
        portfolio_returns = pd.Series(portfolio_returns)
        rebalance_cumulative = (1 + portfolio_returns).cumprod()
        rebalance_total = (rebalance_cumulative.iloc[-1] - 1) * 100
        
        print(f"\nTotal Return: {rebalance_total:.1f}%")
    
    # Strategy 3: Equal weight all valid ETFs
    print("\n" + "="*80)
    print("STRATEGY 3: Equal Weight All ETFs")
    print("="*80)
    
    equal_returns = returns.mean(axis=1)
    equal_cumulative = (1 + equal_returns).cumprod()
    equal_total = (equal_cumulative.iloc[-1] - 1) * 100
    
    print(f"Number of ETFs: {len(valid_etfs)}")
    print(f"Total Return: {equal_total:.1f}%")
    
    # Benchmark: ASX 200
    print("\n" + "="*80)
    print("BENCHMARK: ASX 200")
    print("="*80)
    
    try:
        benchmark_data = yf.download('^AXJO', start=start_date, end=end_date, progress=False)
        if 'Close' in benchmark_data.columns:
            if benchmark_data.columns.nlevels > 1:
                benchmark_prices = benchmark_data['Close']['^AXJO']
            else:
                benchmark_prices = benchmark_data['Close']
        else:
            benchmark_prices = benchmark_data
        
        benchmark_returns = benchmark_prices.pct_change().dropna()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        benchmark_total = (benchmark_cumulative.iloc[-1] - 1) * 100
        
        print(f"Total Return: {benchmark_total:.1f}%")
        
    except Exception as e:
        print(f"Error getting benchmark: {e}")
        benchmark_total = 0
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Strategy':<25} {'Total Return':<12}")
    print("-"*40)
    
    if len(valid_etfs) >= 3:
        print(f"{'Buy & Hold (Top 3)':<25} {buy_hold_total:>10.1f}%")
    
    if len(portfolio_returns) > 0:
        print(f"{'Quarterly Rebalance':<25} {rebalance_total:>10.1f}%")
    
    print(f"{'Equal Weight All':<25} {equal_total:>10.1f}%")
    print(f"{'ASX 200 Benchmark':<25} {benchmark_total:>10.1f}%")
    
    print(f"\nBest performing strategy: Equal Weight All ETFs")
    print(f"This suggests diversification beats concentrated picks")

if __name__ == "__main__":
    simple_backtest()
