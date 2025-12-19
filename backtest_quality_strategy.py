#!/usr/bin/env python3
"""
Backtest Quality ETF Strategy
Simple backtest: Track performance of current top 10 ETFs over past 3 years
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pickle
from pathlib import Path

# Import our modules
from analyzers.quality_ranker import QualityRanker
from data_manager.data_manager import ETFDataManager

def backtest_top_10_etfs():
    """Backtest the current top 10 ETFs without rebalancing"""
    
    print("="*80)
    print("BACKTEST: QUALITY STRATEGY - TOP 10 ETFs")
    print("="*80)
    
    # Load current top 10 ETFs
    ranker = QualityRanker()
    top_10_file = ranker.cache_dir / 'top_10_etfs.pkl'
    
    if not top_10_file.exists():
        print("Error: Top 10 ETFs not found. Run run_quality_universe.py first.")
        return
    
    with open(top_10_file, 'rb') as f:
        top_10_etfs = pickle.load(f)
    
    etf_list = [etf['etf'] for etf in top_10_etfs]
    print(f"\nBacktesting top 10 ETFs: {etf_list}")
    
    # Download 3 years of price data
    print("\nDownloading 3 years of price data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    try:
        data = yf.download(etf_list, start=start_date, end=end_date, progress=False)
        
        if 'Close' in data.columns:
            prices = data['Close']
        else:
            prices = data
        
        # Drop ETFs with insufficient data
        valid_etfs = prices.columns[prices.count() > 500].tolist()
        prices = prices[valid_etfs]
        
        print(f"Valid ETFs for backtest: {len(valid_etfs)}")
        
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        
        # Strategy 1: Equal weight portfolio
        portfolio_returns = returns.mean(axis=1)
        
        # Strategy 2: Weight by quality score
        quality_weights = {etf['etf']: etf['score'] for etf in top_10_etfs if etf['etf'] in valid_etfs}
        total_weight = sum(quality_weights.values())
        quality_weights = {k: v/total_weight for k, v in quality_weights.items()}
        
        weighted_returns = sum(returns[etf] * weight for etf, weight in quality_weights.items())
        
        # Benchmark: Buy and hold equal weight of first 3 ETFs
        benchmark_returns = returns[valid_etfs[:3]].mean(axis=1)
        
        # Calculate cumulative returns
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        weighted_cumulative = (1 + weighted_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        # Calculate performance metrics
        def calculate_metrics(returns, cumulative):
            total_return = (cumulative.iloc[-1] - 1) * 100
            annual_return = (cumulative.iloc[-1] ** (252 / len(cumulative)) - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Hit rate (positive return days)
            hit_rate = (returns > 0).mean() * 100
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'hit_rate': hit_rate
            }
        
        # Calculate metrics for each strategy
        equal_metrics = calculate_metrics(portfolio_returns, portfolio_cumulative)
        weighted_metrics = calculate_metrics(weighted_returns, weighted_cumulative)
        benchmark_metrics = calculate_metrics(benchmark_returns, benchmark_cumulative)
        
        # Display results
        print("\n" + "="*80)
        print("BACKTEST RESULTS (3 YEARS)")
        print("="*80)
        
        print(f"\n{'Strategy':<20} {'Total Return':<12} {'Annual':<8} {'Vol':<8} {'Sharpe':<8} {'Max DD':<8} {'Hit Rate':<8}")
        print("-"*80)
        
        print(f"{'Equal Weight':<20} {equal_metrics['total_return']:>10.1f}% "
              f"{equal_metrics['annual_return']:>6.1f}% {equal_metrics['volatility']:>6.1f}% "
              f"{equal_metrics['sharpe_ratio']:>6.2f} {equal_metrics['max_drawdown']:>6.1f}% {equal_metrics['hit_rate']:>6.1f}%")
        
        print(f"{'Quality Weighted':<20} {weighted_metrics['total_return']:>10.1f}% "
              f"{weighted_metrics['annual_return']:>6.1f}% {weighted_metrics['volatility']:>6.1f}% "
              f"{weighted_metrics['sharpe_ratio']:>6.2f} {weighted_metrics['max_drawdown']:>6.1f}% {weighted_metrics['hit_rate']:>6.1f}%")
        
        print(f"{'Buy & Hold (3)':<20} {benchmark_metrics['total_return']:>10.1f}% "
              f"{benchmark_metrics['annual_return']:>6.1f}% {benchmark_metrics['volatility']:>6.1f}% "
              f"{benchmark_metrics['sharpe_ratio']:>6.2f} {benchmark_metrics['max_drawdown']:>6.1f}% {benchmark_metrics['hit_rate']:>6.1f}%")
        
        # Individual ETF performance
        print("\n" + "="*80)
        print("INDIVIDUAL ETF PERFORMANCE")
        print("="*80)
        
        individual_returns = []
        for etf in valid_etfs:
            etf_cumulative = (1 + returns[etf]).cumprod()
            etf_total = (etf_cumulative.iloc[-1] - 1) * 100
            etf_annual = (etf_cumulative.iloc[-1] ** (252 / len(etf_cumulative)) - 1) * 100
            etf_vol = returns[etf].std() * np.sqrt(252) * 100
            etf_hit = (returns[etf] > 0).mean() * 100
            
            individual_returns.append({
                'etf': etf,
                'total_return': etf_total,
                'annual_return': etf_annual,
                'volatility': etf_vol,
                'hit_rate': etf_hit
            })
        
        individual_df = pd.DataFrame(individual_returns)
        individual_df = individual_df.sort_values('total_return', ascending=False)
        
        print(f"\n{'ETF':<8} {'Total':<8} {'Annual':<8} {'Vol':<8} {'Hit Rate':<8}")
        print("-"*50)
        for _, row in individual_df.iterrows():
            print(f"{row['etf']:<8} {row['total_return']:>6.1f}% {row['annual_return']:>6.1f}% "
                  f"{row['volatility']:>6.1f}% {row['hit_rate']:>6.1f}%")
        
        # Save results
        results = {
            'equal_weight': equal_metrics,
            'quality_weighted': weighted_metrics,
            'benchmark': benchmark_metrics,
            'individual_etfs': individual_df.to_dict('records'),
            'backtest_period': f"{start_date.date()} to {end_date.date()}"
        }
        
        results_file = ranker.cache_dir / 'backtest_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n✅ Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"Error during backtest: {e}")
        return None

if __name__ == "__main__":
    results = backtest_top_10_etfs()
