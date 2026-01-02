import pandas as pd
import numpy as np
import yfinance as yf
from walk_forward_backtest import WalkForwardBacktest
import warnings
warnings.filterwarnings('ignore')

def test_1_out_of_sample():
    """
    Test 1: Out-of-Sample Validation
    Train on 2021-2024, Test on 2024-2025
    """
    print("="*80)
    print("TEST 1: OUT-OF-SAMPLE VALIDATION")
    print("="*80)
    print("Training: 2021-2024 | Testing: 2024-2025")
    print("="*80)
    
    # Train period
    backtest_train = WalkForwardBacktest(
        start_date=pd.Timestamp('2021-01-01'),
        end_date=pd.Timestamp('2023-12-31')
    )
    
    # Test period  
    backtest_test = WalkForwardBacktest(
        start_date=pd.Timestamp('2024-01-01'),
        end_date=pd.Timestamp('2025-12-31')
    )
    
    # Download data for full period
    backtest_full = WalkForwardBacktest(
        start_date=pd.Timestamp('2021-01-01'),
        end_date=pd.Timestamp('2025-12-31')
    )
    
    all_data = backtest_full.download_all_etf_data()
    
    if not all_data:
        print("No data downloaded")
        return
    
    # Test 40d+Top20 on test period
    print("\nTesting 40d+Top20 on unseen data (2024-2025)...")
    test_result = backtest_test.simulate_forecast_aware_rebalancing(
        all_data,
        forecast_window=40,
        buffer_zone=20
    )
    
    # Buy & hold for same period
    bh_result = backtest_test.simulate_buy_and_hold(
        all_data,
        forecast_window=40
    )
    
    if test_result and bh_result:
        test_metrics = backtest_test.calculate_metrics(test_result['returns'])
        bh_metrics = backtest_test.calculate_metrics(bh_result['returns'])
        
        edge = (test_metrics['total_return'] - bh_metrics['total_return']) * 100
        
        print(f"\nOut-of-Sample Results (2024-2025):")
        print(f"  40d+Top20: {test_metrics['total_return']*100:.1f}%")
        print(f"  Buy & Hold: {bh_metrics['total_return']*100:.1f}%")
        print(f"  Edge: {edge:+.1f}%")
        print(f"  Rebalances: {len(test_result['trade_log'])}")
        
        return edge > 0
    
    return False

def test_2_stress_periods():
    """
    Test 2: Stress Test Different Regimes
    """
    print("\n" + "="*80)
    print("TEST 2: STRESS TEST DIFFERENT REGIMES")
    print("="*80)
    
    # Define stress periods
    stress_periods = [
        ('2020-02-01', '2020-12-31', 'COVID Crash & Recovery'),
        ('2022-01-01', '2022-12-31', 'Rate Hike Tech Selloff'),
        ('2023-01-01', '2023-12-31', 'Post-Rate Hike Recovery')
    ]
    
    results = {}
    
    for start, end, period_name in stress_periods:
        print(f"\n--- Testing {period_name} ---")
        
        backtest = WalkForwardBacktest(
            start_date=pd.Timestamp(start),
            end_date=pd.Timestamp(end)
        )
        
        # Download data
        all_data = backtest.download_all_etf_data()
        
        if not all_data:
            print(f"No data for {period_name}")
            continue
        
        # Test 40d+Top20
        strategy_result = backtest.simulate_forecast_aware_rebalancing(
            all_data,
            forecast_window=40,
            buffer_zone=20
        )
        
        # Buy & hold
        bh_result = backtest.simulate_buy_and_hold(
            all_data,
            forecast_window=40
        )
        
        if strategy_result and bh_result:
            strategy_metrics = backtest.calculate_metrics(strategy_result['returns'])
            bh_metrics = backtest.calculate_metrics(bh_result['returns'])
            
            edge = (strategy_metrics['total_return'] - bh_metrics['total_return']) * 100
            
            results[period_name] = {
                'strategy_return': strategy_metrics['total_return'],
                'bh_return': bh_metrics['total_return'],
                'edge': edge,
                'rebalances': len(strategy_result['trade_log']),
                'max_dd': strategy_metrics['max_drawdown']
            }
            
            print(f"  40d+Top20: {strategy_metrics['total_return']*100:.1f}%")
            print(f"  Buy & Hold: {bh_metrics['total_return']*100:.1f}%")
            print(f"  Edge: {edge:+.1f}%")
            print(f"  Rebalances: {len(strategy_result['trade_log'])}")
            print(f"  Max DD: {strategy_metrics['max_drawdown']*100:.1f}%")
            
            # Analyze trades during stress
            analyze_stress_trades(strategy_result['trade_log'], period_name)
    
    return results

def analyze_stress_trades(trade_log, period_name):
    """Analyze trading behavior during stress periods"""
    print(f"\n  Trade Analysis during {period_name}:")
    
    # Count rebalances by month
    rebalance_dates = [t['date'] for t in trade_log if t['action'] == 'rebalance']
    
    if len(rebalance_dates) > 0:
        print(f"    First rebalance: {rebalance_dates[0].date()}")
        print(f"    Last rebalance: {rebalance_dates[-1].date()}")
        print(f"    Avg days between rebalances: {(rebalance_dates[-1] - rebalance_dates[0]).days / max(len(rebalance_dates)-1, 1):.0f}")
        
        # Check if caught the bottom
        if 'COVID' in period_name and len(rebalance_dates) > 0:
            covid_bottom = pd.Timestamp('2020-03-23')
            first_rebalance = rebalance_dates[0]
            days_after_bottom = (first_rebalance - covid_bottom).days
            print(f"    Days after COVID bottom: {days_after_bottom}")

def test_3_walk_forward_cv():
    """
    Test 3: Walk-Forward Cross-Validation
    """
    print("\n" + "="*80)
    print("TEST 3: WALK-FORWARD CROSS-VALIDATION")
    print("="*80)
    
    # Define expanding windows
    cv_windows = [
        ('2021-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
        ('2021-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
        ('2021-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
        ('2021-01-01', '2024-12-31', '2025-01-01', '2025-12-31')
    ]
    
    cv_results = []
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(cv_windows):
        print(f"\n--- Fold {i+1}: Train {train_start[:4]}-{train_end[:4]}, Test {test_start[:4]} ---")
        
        # Download full data
        backtest = WalkForwardBacktest(
            start_date=pd.Timestamp(train_start),
            end_date=pd.Timestamp(test_end)
        )
        
        all_data = backtest.download_all_etf_data()
        
        if not all_data:
            continue
        
        # Test on test period
        test_backtest = WalkForwardBacktest(
            start_date=pd.Timestamp(test_start),
            end_date=pd.Timestamp(test_end)
        )
        
        strategy_result = test_backtest.simulate_forecast_aware_rebalancing(
            all_data,
            forecast_window=40,
            buffer_zone=20
        )
        
        bh_result = test_backtest.simulate_buy_and_hold(
            all_data,
            forecast_window=40
        )
        
        if strategy_result and bh_result:
            strategy_metrics = test_backtest.calculate_metrics(strategy_result['returns'])
            bh_metrics = test_backtest.calculate_metrics(bh_result['returns'])
            
            edge = (strategy_metrics['total_return'] - bh_metrics['total_return']) * 100
            
            cv_results.append(edge)
            
            print(f"  Edge: {edge:+.1f}%")
            print(f"  Rebalances: {len(strategy_result['trade_log'])}")
    
    if cv_results:
        avg_edge = np.mean(cv_results)
        positive_folds = sum(1 for e in cv_results if e > 0)
        
        print(f"\nWalk-Forward CV Results:")
        print(f"  Average edge: {avg_edge:+.1f}%")
        print(f"  Positive folds: {positive_folds}/{len(cv_results)} ({positive_folds/len(cv_results)*100:.0f}%)")
        print(f"  Fold edges: {cv_results}")
        
        return avg_edge, positive_folds/len(cv_results)
    
    return 0, 0

def test_4_baseline_comparison():
    """
    Test 4: Compare to Simpler Baselines
    """
    print("\n" + "="*80)
    print("TEST 4: BASELINE COMPARISON")
    print("="*80)
    
    # Test period
    backtest = WalkForwardBacktest(
        start_date=pd.Timestamp('2021-01-01'),
        end_date=pd.Timestamp('2025-12-31')
    )
    
    all_data = backtest.download_all_etf_data()
    
    if not all_data:
        return
    
    baseline_results = {}
    
    # Baseline 1: 40d+Top20 (complex)
    print("\n--- Testing 40d+Top20 (Complex) ---")
    complex_result = backtest.simulate_forecast_aware_rebalancing(
        all_data,
        forecast_window=40,
        buffer_zone=20
    )
    
    if complex_result:
        complex_metrics = backtest.calculate_metrics(complex_result['returns'])
        baseline_results['40d+Top20'] = {
            'return': complex_metrics['total_return'],
            'sharpe': complex_metrics['sharpe_ratio'],
            'rebalances': len(complex_result['trade_log'])
        }
        print(f"  Return: {complex_metrics['total_return']*100:.1f}%")
        print(f"  Rebalances: {len(complex_result['trade_log'])}")
    
    # Baseline 2: Simple momentum quarterly
    print("\n--- Testing Simple Momentum (Quarterly) ---")
    momentum_result = backtest.simulate_portfolio(
        all_data,
        rebalance_frequency='Q',
        forecast_window=40
    )
    
    if momentum_result:
        momentum_metrics = backtest.calculate_metrics(momentum_result['returns'])
        baseline_results['Momentum_Q'] = {
            'return': momentum_metrics['total_return'],
            'sharpe': momentum_metrics['sharpe_ratio'],
            'rebalances': len(momentum_result['trade_log'])
        }
        print(f"  Return: {momentum_metrics['total_return']*100:.1f}%")
        print(f"  Rebalances: {len(momentum_result['trade_log'])}")
    
    # Baseline 3: Equal weight 10 ETFs annually
    print("\n--- Testing Equal Weight 10 ETFs (Annual) ---")
    equal_weight_result = backtest.simulate_equal_weight(
        all_data,
        num_etfs=10,
        rebalance_frequency='A'
    )
    
    if equal_weight_result:
        ew_metrics = backtest.calculate_metrics(equal_weight_result['returns'])
        baseline_results['Equal_10_A'] = {
            'return': ew_metrics['total_return'],
            'sharpe': ew_metrics['sharpe_ratio'],
            'rebalances': len(equal_weight_result['trade_log'])
        }
        print(f"  Return: {ew_metrics['total_return']*100:.1f}%")
        print(f"  Rebalances: {len(equal_weight_result['trade_log'])}")
    
    # Buy & Hold
    print("\n--- Testing Buy & Hold ---")
    bh_result = backtest.simulate_buy_and_hold(all_data, forecast_window=40)
    
    if bh_result:
        bh_metrics = backtest.calculate_metrics(bh_result['returns'])
        baseline_results['Buy_Hold'] = {
            'return': bh_metrics['total_return'],
            'sharpe': bh_metrics['sharpe_ratio'],
            'rebalances': 1
        }
        print(f"  Return: {bh_metrics['total_return']*100:.1f}%")
    
    # Display comparison
    print(f"\n{'Strategy':<15} {'Return':<10} {'Sharpe':<8} {'Rebalances':<12}")
    print("-"*50)
    
    for strategy, data in baseline_results.items():
        print(f"{strategy:<15} "
              f"{data['return']*100:>8.1f}% "
              f"{data['sharpe']:>6.2f} "
              f"{data['rebalances']:>10}")
    
    # Analysis
    print(f"\n📊 Baseline Analysis:")
    
    if '40d+Top20' in baseline_results and 'Buy_Hold' in baseline_results:
        edge_vs_bh = (baseline_results['40d+Top20']['return'] - 
                     baseline_results['Buy_Hold']['return']) * 100
        print(f"   40d+Top20 edge vs Buy & Hold: {edge_vs_bh:+.1f}%")
    
    if 'Momentum_Q' in baseline_results:
        edge_vs_mom = (baseline_results['40d+Top20']['return'] - 
                      baseline_results['Momentum_Q']['return']) * 100
        print(f"   40d+Top20 edge vs Momentum Q: {edge_vs_mom:+.1f}%")
    
    if 'Equal_10_A' in baseline_results:
        edge_vs_ew = (baseline_results['40d+Top20']['return'] - 
                     baseline_results['Equal_10_A']['return']) * 100
        print(f"   40d+Top20 edge vs Equal Weight: {edge_vs_ew:+.1f}%")
    
    return baseline_results

def simulate_equal_weight(self, all_data, num_etfs=10, rebalance_frequency='Q'):
    """
    Simulate equal-weight portfolio with periodic rebalancing
    """
    print(f"\nSimulating equal-weight {num_etfs} ETFs with {rebalance_frequency} rebalancing...")
    
    # Get all available ETFs at start
    start_date = self.start_date
    available_etfs = []
    
    for etf, prices in all_data.items():
        if start_date in prices.index:
            available_etfs.append(etf)
    
    if len(available_etfs) < num_etfs:
        print(f"Not enough ETFs: {len(available_etfs)} available, need {num_etfs}")
        return None
    
    # Select first N ETFs (simple selection)
    selected_etfs = available_etfs[:num_etfs]
    print(f"Selected {len(selected_etfs)} ETFs")
    
    # Simulate with periodic rebalancing
    return self.simulate_portfolio(all_data, rebalance_frequency=rebalance_frequency, 
                                forecast_window=63)

# Add method to WalkForwardBacktest class
WalkForwardBacktest.simulate_equal_weight = simulate_equal_weight

def main():
    """
    Run all rigorous validation tests
    """
    print("RIGOROUS VALIDATION OF 40d+Top20 STRATEGY")
    print("="*80)
    
    results = {}
    
    # Test 1: Out-of-sample
    results['oos_success'] = test_1_out_of_sample()
    
    # Test 2: Stress periods
    results['stress'] = test_2_stress_periods()
    
    # Test 3: Walk-forward CV
    avg_edge, win_rate = test_3_walk_forward_cv()
    results['cv'] = {'avg_edge': avg_edge, 'win_rate': win_rate}
    
    # Test 4: Baseline comparison
    results['baselines'] = test_4_baseline_comparison()
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\n✅ Out-of-Sample Success: {'YES' if results['oos_success'] else 'NO'}")
    print(f"📊 CV Average Edge: {results['cv']['avg_edge']:+.1f}%")
    print(f"📈 CV Win Rate: {results['cv']['win_rate']*100:.0f}%")
    
    print(f"\n🎯 Final Verdict:")
    
    if results['oos_success'] and results['cv']['win_rate'] > 0.5:
        print("   ✅ Strategy shows genuine edge")
    elif results['cv']['win_rate'] > 0.25:
        print("   ⚠️  Strategy has potential but needs refinement")
    else:
        print("   ❌ Strategy likely overfit to bull market")
    
    return results

if __name__ == "__main__":
    results = main()
