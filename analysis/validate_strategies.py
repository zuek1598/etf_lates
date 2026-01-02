import pandas as pd
import numpy as np
import yfinance as yf
from walk_forward_backtest import WalkForwardBacktest
import warnings
warnings.filterwarnings('ignore')

def validate_top_strategies():
    """
    Validate top 3 performing strategies on 2018-2021 period (including COVID)
    """
    print("="*80)
    print("VALIDATING TOP STRATEGIES ON 2020-2021 (INCLUDING COVID CRASH)")
    print("="*80)
    
    # Initialize backtest with 2020-2021 period (start when more ETFs are available)
    backtest = WalkForwardBacktest(
        start_date=pd.Timestamp('2020-01-01'),  # Start when more ETFs have history
        end_date=pd.Timestamp('2021-12-31')
    )
    
    # Download data
    all_data = backtest.download_all_etf_data()
    
    if not all_data:
        print("No data downloaded")
        return
    
    # Top 3 strategies from 2021-2025 optimization
    strategies = [
        (40, 20, "40d forecast + Top 20 buffer"),
        (20, 10, "20d forecast + Top 10 buffer"),
        (60, 20, "60d forecast + Top 20 buffer")
    ]
    
    results = {}
    
    for forecast_window, buffer_zone, description in strategies:
        print(f"\n{'='*60}")
        print(f"VALIDATING: {description}")
        print(f"{'='*60}")
        
        # Test forecast-aware rebalancing
        strategy_result = backtest.simulate_forecast_aware_rebalancing(
            all_data,
            forecast_window=forecast_window,
            buffer_zone=buffer_zone
        )
        
        # Test buy & hold for same period
        buy_hold_result = backtest.simulate_buy_and_hold(
            all_data,
            forecast_window=forecast_window
        )
        
        if strategy_result and buy_hold_result:
            strategy_metrics = backtest.calculate_metrics(strategy_result['returns'])
            buy_hold_metrics = backtest.calculate_metrics(buy_hold_result['returns'])
            
            edge = (strategy_metrics['total_return'] - buy_hold_metrics['total_return']) * 100
            
            results[description] = {
                'strategy_return': strategy_metrics['total_return'],
                'buy_hold_return': buy_hold_metrics['total_return'],
                'edge': edge,
                'sharpe': strategy_metrics['sharpe_ratio'],
                'rebalances': len(strategy_result['trade_log']),
                'max_dd': strategy_metrics['max_drawdown']
            }
            
            print(f"\nResults 2020-2021:")
            print(f"  Strategy: {strategy_metrics['total_return']*100:.1f}%")
            print(f"  Buy & Hold: {buy_hold_metrics['total_return']*100:.1f}%")
            print(f"  Edge: {edge:+.1f}%")
            print(f"  Sharpe: {strategy_metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {strategy_metrics['max_drawdown']*100:.1f}%")
            print(f"  Rebalances: {len(strategy_result['trade_log'])}")
    
    # Get benchmark
    print("\n" + "="*60)
    print("GETTING BENCHMARK DATA (2020-2021)")
    print("="*60)
    
    try:
        benchmark_data = yf.download(
            '^AXJO',
            start='2020-01-01',
            end='2021-12-31',
            progress=False
        )
        
        if 'Close' in benchmark_data.columns:
            if hasattr(benchmark_data['Close'], 'columns'):
                benchmark_prices = benchmark_data['Close']['^AXJO']
            else:
                benchmark_prices = benchmark_data['Close']
        else:
            benchmark_prices = benchmark_data
        
        benchmark_returns = benchmark_prices.pct_change().dropna()
        benchmark_metrics = backtest.calculate_metrics(benchmark_returns)
        
        print(f"ASX 2018-2021: {benchmark_metrics['total_return']*100:.1f}%")
        
    except Exception as e:
        print(f"Error getting benchmark: {e}")
        benchmark_metrics = {}
    
    # Display comparison
    print("\n" + "="*80)
    print("VALIDATION RESULTS COMPARISON")
    print("="*80)
    
    print(f"\n{'Strategy':<35} {'Return':<10} {'Edge':<10} {'Sharpe':<8} {'Max DD':<10} {'Trades':<8}")
    print("-"*85)
    
    for desc, data in results.items():
        print(f"{desc:<35} "
              f"{data['strategy_return']*100:>8.1f}% "
              f"{data['edge']:>+8.1f}% "
              f"{data['sharpe']:>6.2f} "
              f"{data['max_dd']*100:>8.1f}% "
              f"{data['rebalances']:>6}")
    
    if benchmark_metrics:
        print(f"{'ASX 200 Benchmark':<35} "
              f"{benchmark_metrics['total_return']*100:>8.1f}% "
              f"{'—':>8} "
              f"{benchmark_metrics['sharpe_ratio']:>6.2f} "
              f"{benchmark_metrics['max_drawdown']*100:>8.1f}% "
              f"{'N/A':>6}")
    
    # Analysis
    print(f"\n📊 Validation Analysis:")
    
    # Find best performer
    best_strategy = max(results.items(), key=lambda x: x[1]['edge'])
    print(f"\n🏆 Best Strategy 2018-2021: {best_strategy[0]}")
    print(f"   Edge vs Buy & Hold: {best_strategy[1]['edge']:+.1f}%")
    
    # Compare to 2021-2025 results
    print(f"\n🔄 2021-2025 vs 2018-2021 Comparison:")
    print(f"   2021-2025 winner: 40d+Top20 (+28.6% edge)")
    print(f"   2018-2021 winner: {best_strategy[0]} ({best_strategy[1]['edge']:+.1f}% edge)")
    
    # Check consistency
    consistent = []
    for desc, data in results.items():
        if data['edge'] > 0:
            consistent.append(desc)
    
    print(f"\n✅ Strategies with positive edge in both periods:")
    if consistent:
        for desc in consistent:
            print(f"   - {desc}")
    else:
        print(f"   None - all strategies underperformed buy & hold in 2018-2021")
    
    # Risk analysis
    print(f"\n⚠️  Risk Analysis (COVID period):")
    for desc, data in results.items():
        risk_adj = data['edge'] / abs(data['max_dd']) if data['max_dd'] != 0 else 0
        print(f"   {desc}: Risk-adjusted edge = {risk_adj:.2f}")
    
    return results, benchmark_metrics

if __name__ == "__main__":
    results, benchmark = validate_top_strategies()
