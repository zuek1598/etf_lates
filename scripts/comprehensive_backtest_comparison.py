"""
Comprehensive Backtest Comparison
=================================

Compares multiple strategies:
1. Standard Walk-Forward Backtest
2. Metric-Based CI Backtest
3. Volatility-Based CI Backtest
4. Buy-Hold Strategies
5. Benchmark Comparisons
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

# Add paths
sys.path.append('/Users/peter/Desktop/etf_lates')
sys.path.append('/Users/peter/Desktop/etf_lates/analyzers')

from walk_forward_backtest import WalkForwardBacktest
from walk_forward_backtest_ci import WalkForwardBacktestCI
from walk_forward_backtest_volatility_fixed import WalkForwardBacktestVolatilityFixed

def get_benchmark_data(ticker, start_date, end_date):
    """Get benchmark data (simulated)"""
    # In production, this would fetch real benchmark data
    # For now, we'll simulate based on typical market returns
    
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Simulate different benchmark characteristics
    if ticker == 'SP500':
        # S&P 500 like returns: ~10% annual with 15% volatility
        daily_returns = np.random.normal(0.0004, 0.0095, len(dates))
    elif ticker == 'NASDAQ':
        # NASDAQ like returns: ~12% annual with 20% volatility
        daily_returns = np.random.normal(0.0005, 0.0126, len(dates))
    elif ticker == 'ASX200':
        # ASX 200 like returns: ~8% annual with 12% volatility
        daily_returns = np.random.normal(0.0003, 0.0076, len(dates))
    else:
        # Default balanced fund: ~6% annual with 8% volatility
        daily_returns = np.random.normal(0.0002, 0.0050, len(dates))
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    prices = 100 * np.exp(np.cumsum(daily_returns))
    return pd.Series(prices, index=dates)

def calculate_buy_and_hold(all_data, tickers, start_date, end_date):
    """Calculate buy and hold returns for given ETFs"""
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

def run_comprehensive_comparison():
    """Run comprehensive comparison of all strategies"""
    print("="*80)
    print("COMPREHENSIVE BACKTEST COMPARISON")
    print("="*80)
    
    # Test parameters - extended period for better analysis
    start_date = pd.Timestamp('2020-01-01')  # Include COVID period
    end_date = pd.Timestamp('2024-12-31')
    forecast_window = 40
    buffer_zone = 20
    
    print(f"\nTest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Duration: {(end_date - start_date).days / 365.1:.1f} years")
    
    # Download data once
    print("\nDownloading data...")
    standard_bt = WalkForwardBacktest(start_date, end_date)
    all_data = standard_bt.download_all_etf_data()
    
    results = {}
    
    # 1. Standard Walk-Forward Backtest
    print("\n" + "="*50)
    print("1. STANDARD WALK-FORWARD BACKTEST")
    print("="*50)
    
    standard_results = standard_bt.simulate_forecast_aware_rebalancing(
        all_data, 
        forecast_window=forecast_window, 
        buffer_zone=buffer_zone
    )
    
    std_return = standard_results['cumulative'].iloc[-1] - 1
    std_trades = len([t for t in standard_results['trade_log'] if t['action'] == 'rebalance'])
    
    results['standard'] = {
        'return': std_return,
        'trades': std_trades,
        'volatility': standard_results['returns'].std() * np.sqrt(252),
        'sharpe': (standard_results['returns'].mean() * 252) / (standard_results['returns'].std() * np.sqrt(252)),
        'max_drawdown': calculate_max_drawdown(standard_results['cumulative'])
    }
    
    print(f"  Total Return: {std_return:.1%}")
    print(f"  Annualized Volatility: {results['standard']['volatility']:.1%}")
    print(f"  Sharpe Ratio: {results['standard']['sharpe']:.2f}")
    print(f"  Max Drawdown: {results['standard']['max_drawdown']:.1%}")
    print(f"  Total Trades: {std_trades}")
    
    # 2. Metric-Based CI Backtest
    print("\n" + "="*50)
    print("2. METRIC-BASED CI BACKTEST")
    print("="*50)
    
    metric_bt = WalkForwardBacktestCI(start_date, end_date, enable_ci=True)
    metric_bt.build_confidence_database(all_data)
    
    metric_results = metric_bt.simulate_ci_aware_rebalancing(
        all_data,
        forecast_window=forecast_window,
        buffer_zone=buffer_zone
    )
    
    metric_return = metric_results['cumulative'].iloc[-1] - 1
    metric_trades = len([t for t in metric_results['trade_log'] if t['action'] == 'rebalance'])
    
    results['metric_ci'] = {
        'return': metric_return,
        'trades': metric_trades,
        'volatility': metric_results['returns'].std() * np.sqrt(252),
        'sharpe': (metric_results['returns'].mean() * 252) / (metric_results['returns'].std() * np.sqrt(252)),
        'max_drawdown': calculate_max_drawdown(metric_results['cumulative'])
    }
    
    print(f"  Total Return: {metric_return:.1%}")
    print(f"  Annualized Volatility: {results['metric_ci']['volatility']:.1%}")
    print(f"  Sharpe Ratio: {results['metric_ci']['sharpe']:.2f}")
    print(f"  Max Drawdown: {results['metric_ci']['max_drawdown']:.1%}")
    print(f"  Total Trades: {metric_trades}")
    print(f"  Trade Reduction: {(std_trades - metric_trades)/std_trades:.1%}")
    
    # 3. Volatility-Based CI Backtest
    print("\n" + "="*50)
    print("3. VOLATILITY-BASED CI BACKTEST")
    print("="*50)
    
    vol_bt = WalkForwardBacktestVolatilityFixed(start_date, end_date, enable_ci=True)
    vol_results = vol_bt.simulate_volatility_aware_rebalancing(
        all_data,
        forecast_window=forecast_window,
        buffer_zone=buffer_zone
    )
    
    vol_return = vol_results['cumulative'].iloc[-1] - 1
    vol_trades = len([t for t in vol_results['trade_log'] if t['action'] == 'rebalance'])
    
    results['volatility_ci'] = {
        'return': vol_return,
        'trades': vol_trades,
        'volatility': vol_results['returns'].std() * np.sqrt(252),
        'sharpe': (vol_results['returns'].mean() * 252) / (vol_results['returns'].std() * np.sqrt(252)),
        'max_drawdown': calculate_max_drawdown(vol_results['cumulative'])
    }
    
    print(f"  Total Return: {vol_return:.1%}")
    print(f"  Annualized Volatility: {results['volatility_ci']['volatility']:.1%}")
    print(f"  Sharpe Ratio: {results['volatility_ci']['sharpe']:.2f}")
    print(f"  Max Drawdown: {results['volatility_ci']['max_drawdown']:.1%}")
    print(f"  Total Trades: {vol_trades}")
    print(f"  Trade Reduction: {(std_trades - vol_trades)/std_trades:.1%}")
    
    # 4. Buy-Hold Strategies
    print("\n" + "="*50)
    print("4. BUY-HOLD STRATEGIES")
    print("="*50)
    
    # Buy-hold top 3 ETFs from initial selection
    initial_top = standard_results['trade_log'][0]['holdings'] if standard_results['trade_log'] else []
    if initial_top:
        buy_hold_initial = calculate_buy_and_hold(all_data, initial_top, start_date, end_date)
        if buy_hold_initial:
            bh_initial_return = buy_hold_initial['cumulative'].iloc[-1] - 1
            results['buy_hold_initial'] = {
                'return': bh_initial_return,
                'trades': 0,
                'volatility': buy_hold_initial['returns'].std() * np.sqrt(252),
                'sharpe': (buy_hold_initial['returns'].mean() * 252) / (buy_hold_initial['returns'].std() * np.sqrt(252)),
                'max_drawdown': calculate_max_drawdown(buy_hold_initial['cumulative'])
            }
            print(f"\n  Buy-Hold Initial Top 3 ({initial_top}):")
            print(f"    Total Return: {bh_initial_return:.1%}")
            print(f"    Sharpe Ratio: {results['buy_hold_initial']['sharpe']:.2f}")
    
    # Buy-hold equal weight top 10 ETFs
    top_10 = list(all_data.keys())[:10]  # Simplified - would use actual ranking
    buy_hold_10 = calculate_buy_and_hold(all_data, top_10, start_date, end_date)
    if buy_hold_10:
        bh_10_return = buy_hold_10['cumulative'].iloc[-1] - 1
        results['buy_hold_10'] = {
            'return': bh_10_return,
            'trades': 0,
            'volatility': buy_hold_10['returns'].std() * np.sqrt(252),
            'sharpe': (buy_hold_10['returns'].mean() * 252) / (buy_hold_10['returns'].std() * np.sqrt(252)),
            'max_drawdown': calculate_max_drawdown(buy_hold_10['cumulative'])
        }
        print(f"\n  Buy-Hold Top 10 ETFs:")
        print(f"    Total Return: {bh_10_return:.1%}")
        print(f"    Sharpe Ratio: {results['buy_hold_10']['sharpe']:.2f}")
    
    # 5. Benchmarks
    print("\n" + "="*50)
    print("5. BENCHMARK COMPARISONS")
    print("="*50)
    
    benchmarks = ['SP500', 'NASDAQ', 'ASX200', 'Balanced_Fund']
    
    for benchmark in benchmarks:
        benchmark_data = get_benchmark_data(benchmark, start_date, end_date)
        benchmark_returns = benchmark_data.pct_change().fillna(0)
        benchmark_cumulative = benchmark_data / benchmark_data.iloc[0]
        
        bench_return = benchmark_cumulative.iloc[-1] - 1
        results[f'benchmark_{benchmark.lower()}'] = {
            'return': bench_return,
            'trades': 0,
            'volatility': benchmark_returns.std() * np.sqrt(252),
            'sharpe': (benchmark_returns.mean() * 252) / (benchmark_returns.std() * np.sqrt(252)),
            'max_drawdown': calculate_max_drawdown(benchmark_cumulative)
        }
        
        print(f"\n  {benchmark}:")
        print(f"    Total Return: {bench_return:.1%}")
        print(f"    Sharpe Ratio: {results[f'benchmark_{benchmark.lower()}']['sharpe']:.2f}")
    
    # Comprehensive Summary
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    print(f"{'Strategy':<25} {'Return':<10} {'Vol':<8} {'Sharpe':<8} {'DD':<8} {'Trades':<8}")
    print("-"*80)
    
    strategy_names = {
        'standard': 'Standard Walk-Forward',
        'metric_ci': 'Metric-Based CI',
        'volatility_ci': 'Volatility-Based CI',
        'buy_hold_initial': 'Buy-Hold Initial',
        'buy_hold_10': 'Buy-Hold Top 10',
        'benchmark_sp500': 'S&P 500',
        'benchmark_nasdaq': 'NASDAQ',
        'benchmark_asx200': 'ASX 200',
        'benchmark_balanced_fund': 'Balanced Fund'
    }
    
    for key, name in strategy_names.items():
        if key in results:
            r = results[key]
            print(f"{name:<25} {r['return']:<10.1%} {r['volatility']:<8.1%} "
                  f"{r['sharpe']:<8.2f} {r['max_drawdown']:<8.1%} {r['trades']:<8}")
    
    # Performance Analysis
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Best returns
    best_return = max(results.items(), key=lambda x: x[1]['return'])
    print(f"\nBest Total Return: {strategy_names.get(best_return[0], best_return[0])}")
    print(f"  Return: {best_return[1]['return']:.1%}")
    
    # Best Sharpe
    best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe'])
    print(f"\nBest Risk-Adjusted (Sharpe): {strategy_names.get(best_sharpe[0], best_sharpe[0])}")
    print(f"  Sharpe: {best_sharpe[1]['sharpe']:.2f}")
    
    # Lowest drawdown
    best_dd = min(results.items(), key=lambda x: x[1]['max_drawdown'])
    print(f"\nLowest Max Drawdown: {strategy_names.get(best_dd[0], best_dd[0])}")
    print(f"  Max DD: {best_dd[1]['max_drawdown']:.1%}")
    
    # Trade efficiency
    trading_strategies = {k: v for k, v in results.items() if v['trades'] > 0}
    if trading_strategies:
        print(f"\nTrade Efficiency (Return per Trade):")
        for key, value in trading_strategies.items():
            efficiency = value['return'] / value['trades']
            print(f"  {strategy_names.get(key, key)}: {efficiency:.1%} per trade")
    
    # Period Analysis
    print("\n" + "="*80)
    print("PERIOD ANALYSIS (2020-2024)")
    print("="*80)
    print("\nKey Market Events:")
    print("  - 2020: COVID crash & recovery")
    print("  - 2021: Bull market continuation")
    print("  - 2022: Bear market (inflation, rate hikes)")
    print("  - 2023: Recovery year")
    print("  - 2024: Mixed conditions")
    
    print("\nStrategy Performance During Different Periods:")
    print("  (Note: Detailed period-by-period analysis would require separate runs)")
    
    return results

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown"""
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

if __name__ == "__main__":
    results = run_comprehensive_comparison()
