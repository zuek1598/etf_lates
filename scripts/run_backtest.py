#!/usr/bin/env python3
"""
Standalone Backtest Runner
Runs full universe backtest without interactive prompts

Usage:
    python3 run_backtest.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from utilities.backtest_engine import run_backtest_on_universe


def main():
    """Run full universe backtest"""
    print("\n" + "="*80)
    print("RUNNING FULL UNIVERSE BACKTEST")
    print("="*80)
    print("\n⚠️  This will backtest all ETFs with sufficient historical data")
    print("    Estimated time: 10-20 minutes for ~350 ETFs")
    print()
    
    # Run backtest on all available ETFs
    results = run_backtest_on_universe()
    
    if not results.empty:
        print("\n✅ Full universe backtest complete!")
        print(f"   Results saved to: data/backtest_results.parquet")
        print(f"   Tested {len(results)} ETFs")
        print("\nView results in dashboard:")
        print("  python3 run_dashboard.py")
        print("  Open: http://127.0.0.1:8050")
    else:
        print("\n⚠️  No backtest results generated")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

