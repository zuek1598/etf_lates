"""
Quick test of CI system to verify fixes
"""

import pandas as pd
import numpy as np
import sys

# Add paths
sys.path.append('/Users/peter/Desktop/etf_lates')
sys.path.append('/Users/peter/Desktop/etf_lates/analyzers')

from walk_forward_backtest_ci import WalkForwardBacktestCI

def quick_test():
    """Quick test of CI system"""
    print("Quick CI System Test")
    print("-" * 40)
    
    # Initialize with shorter period for testing
    ci_backtest = WalkForwardBacktestCI(
        start_date=pd.Timestamp('2023-01-01'),
        end_date=pd.Timestamp('2023-12-31'),
        enable_ci=True
    )
    
    # Download data
    print("Downloading data...")
    all_data = ci_backtest.download_all_etf_data()
    
    # Build confidence database
    print("Building confidence intervals...")
    ci_backtest.build_confidence_database(all_data)
    
    # Run strategy
    print("Running CI-enhanced strategy...")
    result = ci_backtest.simulate_ci_aware_rebalancing(
        all_data=all_data,
        forecast_window=40,
        buffer_zone=20
    )
    
    if result:
        final_return = (result['cumulative'].iloc[-1] - 1) * 100
        sharpe = result['returns'].mean() / result['returns'].std() * np.sqrt(252)
        
        print(f"\nResults:")
        print(f"  Total return: {final_return:.1f}%")
        print(f"  Sharpe ratio: {sharpe:.2f}")
        print(f"  Number of trades: {len(result['trade_log'])}")
        
        if result['ci_stats']:
            print(f"  Noise prevented: {result['ci_stats']['noise_prevented']}")
        
        print("\n✓ CI system is working!")
        return True
    else:
        print("\n✗ CI system failed")
        return False

if __name__ == "__main__":
    quick_test()
