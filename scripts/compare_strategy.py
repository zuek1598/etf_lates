#!/usr/bin/env python3
"""
Compare Strategy vs Buy-Hold and Benchmarks

Usage:
  python scripts/compare_strategy.py VAS.AX
  python scripts/compare_strategy.py VAS.AX VGS.AX IOZ.AX
  python scripts/compare_strategy.py --top 10
  python scripts/compare_strategy.py --all
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utilities.strategy_comparator import StrategyComparator, compare_multiple_etfs


def main():
    parser = argparse.ArgumentParser(
        description='Compare strategy returns vs buy-and-hold and benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_strategy.py VAS.AX
  python scripts/compare_strategy.py VAS.AX VGS.AX IOZ.AX
  python scripts/compare_strategy.py --top 10
  python scripts/compare_strategy.py --all
  python scripts/compare_strategy.py VAS.AX --period 6m
        """
    )

    parser.add_argument('tickers', nargs='*', help='ETF ticker(s) to compare')
    parser.add_argument('--top', type=int, help='Compare top N performers from backtest')
    parser.add_argument('--bottom', type=int, help='Compare bottom N performers from backtest')
    parser.add_argument('--all', action='store_true', help='Compare all ETFs from backtest')
    parser.add_argument('--period', default='full', choices=['1m', '3m', '6m', '1y', 'full'],
                        help='Backtest period (default: full)')
    parser.add_argument('--output', help='Output CSV file for results')
    parser.add_argument('--save-report', help='Save detailed report to file')

    args = parser.parse_args()

    # Initialize comparator with selected period
    comparator = StrategyComparator(period=args.period)

    # Determine which ETFs to compare
    tickers_to_compare = []

    if args.tickers:
        tickers_to_compare = args.tickers
    elif args.top:
        # Get top N performers
        top_performers = comparator.results_df.nlargest(args.top, 'total_return')
        tickers_to_compare = top_performers['ticker'].tolist()
        print(f"\nComparing TOP {args.top} performers:\n")

    elif args.bottom:
        # Get bottom N performers
        bottom_performers = comparator.results_df.nsmallest(args.bottom, 'total_return')
        tickers_to_compare = bottom_performers['ticker'].tolist()
        print(f"\nComparing BOTTOM {args.bottom} performers:\n")

    elif args.all:
        # Get all ETFs
        tickers_to_compare = comparator.results_df['ticker'].tolist()
        print(f"\nComparing ALL {len(tickers_to_compare)} ETFs:\n")

    else:
        # Interactive mode - ask user for input
        print("\n" + "="*80)
        print("STRATEGY COMPARATOR - INTERACTIVE MODE")
        print("="*80)

        # First, ask for period selection
        print("\nSelect Analysis Period:")
        print("  1. Last 1 month")
        print("  2. Last 3 months")
        print("  3. Last 6 months")
        print("  4. Last 1 year")
        print("  5. Full history")
        print("")

        period_choice = input("Enter period (1-5, default 5): ").strip()
        period_map = {
            '1': '1m',
            '2': '3m',
            '3': '6m',
            '4': '1y',
            '5': 'full',
            '': 'full'
        }
        selected_period = period_map.get(period_choice, 'full')

        # Reinitialize comparator with selected period
        try:
            comparator = StrategyComparator(period=selected_period)
        except Exception as e:
            print(f"\nWarning: Could not load backtest results for {selected_period}: {e}")
            print("Falling back to full history...\n")
            comparator = StrategyComparator(period='full')
            selected_period = 'full'

        print(f"\nUsing backtest results for: {selected_period.upper()}")
        print("\n" + "="*80)
        print("ANALYSIS OPTIONS")
        print("="*80)
        print("\nOptions:")
        print("  1. Compare single ETF (e.g., VAS.AX)")
        print("  2. Compare multiple ETFs (e.g., VAS.AX VGS.AX IOZ.AX)")
        print("  3. Top 10 performers")
        print("  4. Bottom 10 performers")
        print("  5. All ETFs (export to CSV)")
        print("  6. Exit")
        print("")

        while True:
            choice = input("Enter choice (1-6): ").strip()

            if choice == '1':
                ticker = input("Enter ETF ticker (e.g., VAS.AX): ").strip().upper()
                if not ticker:
                    print("Invalid ticker. Try again.\n")
                    continue
                tickers_to_compare = [ticker]
                break

            elif choice == '2':
                tickers_str = input("Enter ETF tickers separated by spaces (e.g., VAS.AX VGS.AX IOZ.AX): ").strip().upper()
                if not tickers_str:
                    print("Invalid input. Try again.\n")
                    continue
                tickers_to_compare = tickers_str.split()
                break

            elif choice == '3':
                top_performers = comparator.results_df.nlargest(10, 'total_return')
                tickers_to_compare = top_performers['ticker'].tolist()
                print(f"\nComparing TOP 10 performers:\n")
                break

            elif choice == '4':
                bottom_performers = comparator.results_df.nsmallest(10, 'total_return')
                tickers_to_compare = bottom_performers['ticker'].tolist()
                print(f"\nComparing BOTTOM 10 performers:\n")
                break

            elif choice == '5':
                output_file = input("Enter output filename (default: strategy_results.csv): ").strip()
                if not output_file:
                    output_file = "strategy_results.csv"
                args.output = output_file
                tickers_to_compare = comparator.results_df['ticker'].tolist()
                print(f"\nComparing ALL {len(tickers_to_compare)} ETFs...\n")
                break

            elif choice == '6':
                print("Exiting.\n")
                return 0

            else:
                print("Invalid choice. Try again.\n")
                continue

    # Process single ticker
    if len(tickers_to_compare) == 1:
        ticker = tickers_to_compare[0]
        try:
            comparison = comparator.compare_etf(ticker)
            print(comparator.format_comparison(comparison))

            # Save detailed report if requested
            if args.save_report:
                with open(args.save_report, 'w') as f:
                    f.write(comparator.format_comparison(comparison))
                print(f"\n[OK] Report saved to: {args.save_report}")

        except Exception as e:
            print(f"Error: {e}")
            return 1

    # Process multiple tickers
    else:
        results_df = compare_multiple_etfs(tickers_to_compare, show_summary=True, period=args.period)

        # Save to CSV if requested
        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"[OK] Results saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
