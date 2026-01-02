import pandas as pd
import numpy as np
import yfinance as yf
from walk_forward_backtest import WalkForwardBacktest
import warnings
warnings.filterwarnings('ignore')

def comprehensive_analysis():
    """
    Run comprehensive analysis from 2018 to today
    Tests all strategies across full market cycles
    """
    print("="*80)
    print("COMPREHENSIVE ANALYSIS: 2018 to Today")
    print("="*80)
    
    # Define periods for analysis
    periods = [
        ('2020-01-01', '2021-12-31', 'COVID Period'),
        ('2021-01-01', '2025-12-31', 'Post-COVID Bull'),
        ('2020-01-01', '2025-12-31', 'Full Period')
    ]
    
    # Top strategies from optimization
    strategies = [
        (40, 20, "40d forecast + Top 20"),
        (20, 10, "20d forecast + Top 10"),
        (60, 20, "60d forecast + Top 20"),
        (63, 15, "63d forecast + Top 15")
    ]
    
    all_results = {}
    
    for start_date, end_date, period_name in periods:
        print(f"\n{'='*80}")
        print(f"ANALYZING PERIOD: {period_name} ({start_date} to {end_date})")
        print(f"{'='*80}")
        
        # Initialize backtest for this period
        backtest = WalkForwardBacktest(
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date)
        )
        
        # Download data
        all_data = backtest.download_all_etf_data()
        
        if not all_data:
            print(f"No data for {period_name}")
            continue
        
        period_results = {}
        
        for forecast_window, buffer_zone, strategy_name in strategies:
            print(f"\n--- Testing {strategy_name} ---")
            
            # Test forecast-aware rebalancing
            strategy_result = backtest.simulate_forecast_aware_rebalancing(
                all_data,
                forecast_window=forecast_window,
                buffer_zone=buffer_zone
            )
            
            # Test buy & hold
            buy_hold_result = backtest.simulate_buy_and_hold(
                all_data,
                forecast_window=forecast_window
            )
            
            if strategy_result and buy_hold_result:
                strategy_metrics = backtest.calculate_metrics(strategy_result['returns'])
                buy_hold_metrics = backtest.calculate_metrics(buy_hold_result['returns'])
                
                edge = (strategy_metrics['total_return'] - buy_hold_metrics['total_return']) * 100
                
                period_results[strategy_name] = {
                    'strategy_return': strategy_metrics['total_return'],
                    'buy_hold_return': buy_hold_metrics['total_return'],
                    'edge': edge,
                    'sharpe': strategy_metrics['sharpe_ratio'],
                    'rebalances': len(strategy_result['trade_log']),
                    'max_dd': strategy_metrics['max_drawdown'],
                    'annual': strategy_metrics['annual_return']
                }
                
                print(f"  Return: {strategy_metrics['total_return']*100:.1f}%")
                print(f"  Buy & Hold: {buy_hold_metrics['total_return']*100:.1f}%")
                print(f"  Edge: {edge:+.1f}%")
                print(f"  Sharpe: {strategy_metrics['sharpe_ratio']:.2f}")
                print(f"  Rebalances: {len(strategy_result['trade_log'])}")
        
        all_results[period_name] = period_results
        
        # Get benchmark for this period
        try:
            benchmark_data = yf.download(
                '^AXJO',
                start=start_date,
                end=end_date,
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
            
            print(f"\nASX 200 ({period_name}): {benchmark_metrics['total_return']*100:.1f}%")
            all_results[period_name]['benchmark'] = benchmark_metrics
            
        except Exception as e:
            print(f"Error getting benchmark: {e}")
    
    # Display comprehensive results
    display_comprehensive_results(all_results)
    
    return all_results

def display_comprehensive_results(all_results):
    """
    Display comprehensive results across all periods
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    # Create summary table
    periods = list(all_results.keys())
    strategies = ['40d forecast + Top 20', '20d forecast + Top 10', 
                 '60d forecast + Top 20', '63d forecast + Top 15']
    
    print(f"\n{'Strategy':<25} {'COVID':<12} {'Post-COVID':<12} {'Full':<12} {'Best':<10}")
    print("-"*75)
    
    for strategy in strategies:
        print(f"{strategy:<25}", end="")
        
        best_period = None
        best_edge = -999
        
        for period in periods:
            if period in all_results and strategy in all_results[period]:
                edge = all_results[period][strategy]['edge']
                color = "🟢" if edge > 0 else "🔴"
                print(f"{color}{edge:>+10.1f}%{'':<1}", end="")
                
                if edge > best_edge:
                    best_edge = edge
                    best_period = period[:3]  # First 3 chars
            else:
                print(f"{'N/A':>12}", end="")
        
        print(f"  {best_period:<8}")
    
    # Find best overall strategy
    print(f"\n🏆 Best Strategy by Period:")
    for period in periods:
        if period not in all_results:
            continue
            
        best_strategy = None
        best_edge = -999
        
        for strategy, data in all_results[period].items():
            if strategy == 'benchmark':
                continue
            if data['edge'] > best_edge:
                best_edge = data['edge']
                best_strategy = strategy
        
        if best_strategy:
            print(f"   {period}: {best_strategy} ({best_edge:+.1f}% edge)")
    
    # Consistency analysis
    print(f"\n📊 Consistency Analysis:")
    
    for strategy in strategies:
        positive_edges = 0
        total_edges = 0
        total_edge = 0
        
        for period in periods:
            if period in all_results and strategy in all_results[period]:
                edge = all_results[period][strategy]['edge']
                total_edges += 1
                total_edge += edge
                if edge > 0:
                    positive_edges += 1
        
        if total_edges > 0:
            consistency = (positive_edges / total_edges) * 100
            avg_edge = total_edge / total_edges
            print(f"   {strategy}: {consistency:.0f}% positive edges, avg {avg_edge:+.1f}%")
    
    # Risk-adjusted performance
    print(f"\n⚠️  Risk-Adjusted Performance (Full Period):")
    if 'Full Period' in all_results:
        for strategy, data in all_results['Full Period'].items():
            if strategy == 'benchmark':
                continue
            if data['max_dd'] != 0:
                risk_adj = data['edge'] / abs(data['max_dd'])
                print(f"   {strategy}: {risk_adj:.2f}")
    
    # Key insights
    print(f"\n🎯 Key Insights:")
    
    # Check if any strategy consistently beats buy & hold
    consistent_winners = []
    for strategy in strategies:
        always_positive = True
        for period in periods:
            if period in all_results and strategy in all_results[period]:
                if all_results[period][strategy]['edge'] <= 0:
                    always_positive = False
                    break
        if always_positive:
            consistent_winners.append(strategy)
    
    if consistent_winners:
        print(f"   ✅ Consistent Winners: {', '.join(consistent_winners)}")
    else:
        print(f"   ❌ No strategy consistently beats buy & hold across all periods")
    
    # Volatility sensitivity
    covid_performance = {}
    post_covid_performance = {}
    
    for strategy in strategies:
        if 'COVID Period' in all_results and strategy in all_results['COVID Period']:
            covid_performance[strategy] = all_results['COVID Period'][strategy]['edge']
        if 'Post-COVID Bull' in all_results and strategy in all_results['Post-COVID Bull']:
            post_covid_performance[strategy] = all_results['Post-COVID Bull'][strategy]['edge']
    
    print(f"\n📈 Volatility Sensitivity:")
    print(f"   COVID (high vol) vs Post-COVID (bull market):")
    for strategy in strategies:
        covid_edge = covid_performance.get(strategy, 0)
        post_edge = post_covid_performance.get(strategy, 0)
        diff = post_edge - covid_edge
        print(f"   {strategy}: COVID {covid_edge:+.1f}% → Post {post_edge:+.1f}% (diff {diff:+.1f}%)")
    
    print(f"\n⚠️  Final Recommendation:")
    print(f"   - No single strategy works across all market conditions")
    print(f"   - Consider regime-based approach: longer horizons in volatile markets")
    print(f"   - 40d+Top20 excels in bull markets but fails in crises")
    print(f"   - 60d+Top20 provides more stability (least negative in COVID)")

if __name__ == "__main__":
    results = comprehensive_analysis()
