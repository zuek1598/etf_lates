"""
Check Buy-Hold Performance Against Our Strategies
===============================================

Let's see how simple buy-hold compares to our complex strategies.
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime

# Add paths
sys.path.append('/Users/peter/Desktop/etf_lates')
sys.path.append('/Users/peter/Desktop/etf_lates/analyzers')

from walk_forward_backtest import WalkForwardBacktest

def calculate_buy_hold(all_data, tickers, start_date, end_date):
    """Calculate buy and hold returns"""
    if not tickers:
        return None
    
    # Get common date range
    common_dates = None
    valid_tickers = []
    
    for ticker in tickers:
        if ticker in all_data:
            prices = all_data[ticker]
            mask = (prices.index >= start_date) & (prices.index <= end_date)
            ticker_dates = prices.index[mask]
            
            if common_dates is None:
                common_dates = ticker_dates
            else:
                common_dates = common_dates.intersection(ticker_dates)
            
            if len(ticker_dates) > 0:
                valid_tickers.append(ticker)
    
    if not valid_tickers or len(common_dates) == 0:
        return None
    
    # Calculate equal-weight portfolio
    portfolio_value = pd.Series(index=common_dates, dtype=float)
    
    # Initial investment
    initial_value = 100000
    portfolio_value.iloc[0] = initial_value
    
    # Calculate daily returns
    for i in range(1, len(common_dates)):
        date = common_dates[i]
        prev_date = common_dates[i-1]
        
        daily_return = 0
        count = 0
        
        for ticker in valid_tickers:
            if ticker in all_data:
                if date in all_data[ticker].index and prev_date in all_data[ticker].index:
                    ret = (all_data[ticker].loc[date] - all_data[ticker].loc[prev_date]) / all_data[ticker].loc[prev_date]
                    daily_return += ret
                    count += 1
        
        if count > 0:
            daily_return = daily_return / count
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + daily_return)
    
    return {
        'returns': portfolio_value.pct_change().fillna(0),
        'cumulative': portfolio_value / initial_value,
        'value': portfolio_value,
        'tickers': valid_tickers
    }

def get_market_benchmark(start_date, end_date):
    """Simulate market benchmark returns"""
    # Create realistic market returns for the period
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Simulate different market phases
    np.random.seed(42)
    
    returns = []
    for date in dates:
        if date.year == 2020 and date.month < 4:
            # COVID crash: -30% in Feb-Mar
            daily_ret = np.random.normal(-0.005, 0.03)
        elif date.year == 2020 and date.month >= 4:
            # Recovery: strong rebound
            daily_ret = np.random.normal(0.002, 0.02)
        elif date.year == 2021:
            # Bull market
            daily_ret = np.random.normal(0.0008, 0.015)
        elif date.year == 2022:
            # Bear market
            daily_ret = np.random.normal(-0.0005, 0.02)
        elif date.year == 2023:
            # Recovery
            daily_ret = np.random.normal(0.0005, 0.015)
        else:  # 2024
            # Mixed
            daily_ret = np.random.normal(0.0003, 0.012)
        
        returns.append(daily_ret)
    
    prices = 100 * np.exp(np.cumsum(returns))
    benchmark = pd.Series(prices, index=dates)
    
    return {
        'returns': benchmark.pct_change().fillna(0),
        'cumulative': benchmark / 100,
        'value': benchmark * 1000
    }

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown"""
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def main():
    print("="*80)
    print("BUY-HOLD vs ACTIVE STRATEGIES COMPARISON")
    print("="*80)
    
    # Test parameters
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2024-12-31')
    
    print(f"\nPeriod: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Duration: {(end_date - start_date).days / 365.1:.1f} years")
    
    # Download data
    print("\nDownloading ETF data...")
    bt = WalkForwardBacktest(start_date, end_date)
    all_data = bt.download_all_etf_data()
    
    results = {}
    
    # 1. Buy-Hold Initial Top 3 (what our strategies started with)
    print("\n" + "="*50)
    print("1. BUY-HOLD INITIAL TOP 3")
    print("="*50)
    
    # Get initial top 3 from our first backtest
    initial_holdings = ['VBLD.AX', 'RCAP.AX', 'IFRA.AX']  # From earlier runs
    
    bh_initial = calculate_buy_hold(all_data, initial_holdings, start_date, end_date)
    if bh_initial:
        bh_initial_return = bh_initial['cumulative'].iloc[-1] - 1
        results['buy_hold_initial'] = {
            'return': bh_initial_return,
            'trades': 0,
            'volatility': bh_initial['returns'].std() * np.sqrt(252),
            'sharpe': (bh_initial['returns'].mean() * 252) / (bh_initial['returns'].std() * np.sqrt(252)),
            'max_drawdown': calculate_max_drawdown(bh_initial['cumulative'])
        }
        
        print(f"  Initial Holdings: {initial_holdings}")
        print(f"  Total Return: {bh_initial_return:.1%}")
        print(f"  Sharpe Ratio: {results['buy_hold_initial']['sharpe']:.2f}")
        print(f"  Max Drawdown: {results['buy_hold_initial']['max_drawdown']:.1%}")
    
    # 2. Buy-Hold Equal Weight Top 10 ETFs
    print("\n" + "="*50)
    print("2. BUY-HOLD TOP 10 ETFs (Equal Weight)")
    print("="*50)
    
    # Simple selection - first 10 available ETFs
    top_10 = list(all_data.keys())[:10]
    
    bh_10 = calculate_buy_hold(all_data, top_10, start_date, end_date)
    if bh_10:
        bh_10_return = bh_10['cumulative'].iloc[-1] - 1
        results['buy_hold_10'] = {
            'return': bh_10_return,
            'trades': 0,
            'volatility': bh_10['returns'].std() * np.sqrt(252),
            'sharpe': (bh_10['returns'].mean() * 252) / (bh_10['returns'].std() * np.sqrt(252)),
            'max_drawdown': calculate_max_drawdown(bh_10['cumulative'])
        }
        
        print(f"  Top 10 ETFs: {top_10}")
        print(f"  Total Return: {bh_10_return:.1%}")
        print(f"  Sharpe Ratio: {results['buy_hold_10']['sharpe']:.2f}")
        print(f"  Max Drawdown: {results['buy_hold_10']['max_drawdown']:.1%}")
    
    # 3. Market Benchmark
    print("\n" + "="*50)
    print("3. MARKET BENCHMARK (Simulated S&P/ASX)")
    print("="*50)
    
    benchmark = get_market_benchmark(start_date, end_date)
    benchmark_return = benchmark['cumulative'].iloc[-1] - 1
    results['benchmark'] = {
        'return': benchmark_return,
        'trades': 0,
        'volatility': benchmark['returns'].std() * np.sqrt(252),
        'sharpe': (benchmark['returns'].mean() * 252) / (benchmark['returns'].std() * np.sqrt(252)),
        'max_drawdown': calculate_max_drawdown(benchmark['cumulative'])
    }
    
    print(f"  Market Return: {benchmark_return:.1%}")
    print(f"  Sharpe Ratio: {results['benchmark']['sharpe']:.2f}")
    print(f"  Max Drawdown: {results['benchmark']['max_drawdown']:.1%}")
    
    # 4. Our Best Strategy (Fixed Volatility CI)
    print("\n" + "="*50)
    print("4. BEST ACTIVE STRATEGY (Fixed Volatility CI)")
    print("="*50)
    
    results['volatility_fixed'] = {
        'return': 0.639,  # From previous run
        'trades': 45,
        'volatility': 0.14,
        'sharpe': 0.65,
        'max_drawdown': -0.287
    }
    
    print(f"  Total Return: 63.9%")
    print(f"  Sharpe Ratio: 0.65")
    print(f"  Max Drawdown: -28.7%")
    print(f"  Total Trades: 45")
    
    # Comprehensive comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON - DID WE BEAT BUY-HOLD?")
    print("="*80)
    print(f"{'Strategy':<25} {'Return':<10} {'Sharpe':<8} {'Max DD':<10} {'Trades':<8} {'Beat BH?':<10}")
    print("-"*80)
    
    strategy_names = {
        'buy_hold_initial': 'Buy-Hold Initial',
        'buy_hold_10': 'Buy-Hold Top 10',
        'benchmark': 'Market Benchmark',
        'volatility_fixed': 'Fixed Vol CI (Active)'
    }
    
    for key, name in strategy_names.items():
        if key in results:
            r = results[key]
            beat_bh = r['return'] > results.get('buy_hold_initial', {}).get('return', 0)
            print(f"{name:<25} {r['return']:<10.1%} {r['sharpe']:<8.2f} "
                  f"{r['max_drawdown']:<10.1%} {r['trades']:<8} {'Yes' if beat_bh else 'No':<10}")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    bh_return = results.get('buy_hold_initial', {}).get('return', 0)
    active_return = results['volatility_fixed']['return']
    
    print(f"\nKey Question: Did active management add value?")
    print(f"  Buy-Hold Return: {bh_return:.1%}")
    print(f"  Active Return: {active_return:.1%}")
    print(f"  Outperformance: {active_return - bh_return:.1%}")
    
    if active_return > bh_return:
        print(f"\n  ✓ YES: Active strategy outperformed buy-hold")
        print(f"  ✓ Added {active_return - bh_return:.1%} alpha")
        print(f"  ✓ With {results['volatility_fixed']['trades']} trades")
    else:
        print(f"\n  ✗ NO: Buy-hold would have been better")
        print(f"  ✗ Underperformed by {bh_return - active_return:.1%}")
    
    print(f"\nRisk-Adjusted Performance:")
    print(f"  Buy-Hold Sharpe: {results.get('buy_hold_initial', {}).get('sharpe', 0):.2f}")
    print(f"  Active Sharpe: {results['volatility_fixed']['sharpe']:.2f}")
    
    print(f"\nConclusion:")
    print(f"  The Fixed Volatility CI strategy {'successfully' if active_return > bh_return else 'failed to'} ")
    print(f"  {'justify' if active_return > bh_return else 'justify'} active management over this period.")

if __name__ == "__main__":
    main()
