#!/usr/bin/env python3
"""
Single Ticker Analysis Entry Point
Analyzes individual stocks or crypto coins using the validated system

Usage:
    python3 run_ticker_analysis.py AAPL          # Analyze Apple stock
    python3 run_ticker_analysis.py BTC-USD         # Analyze Bitcoin
    python3 run_ticker_analysis.py TSLA MSFT       # Analyze multiple tickers
    python3 run_ticker_analysis.py --help          # Show help

Examples:
    Stocks: AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN
    Crypto: BTC-USD, ETH-USD, SOL-USD, ADA-USD
"""

import sys
import argparse
from analyzers.single_ticker_analyzer import SingleTickerAnalyzer


def main():
    """Main entry point for single ticker analysis"""
    parser = argparse.ArgumentParser(
        description='Analyze individual stocks or crypto coins using validated system components',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s AAPL                    # Analyze Apple stock
  %(prog)s BTC-USD                 # Analyze Bitcoin
  %(prog)s TSLA MSFT GOOGL         # Analyze multiple tickers
  %(prog)s AAPL --period 5y        # Use 5 years of historical data
        """
    )
    
    parser.add_argument(
        'tickers',
        nargs='*',  # Allow zero arguments for interactive mode
        help='Stock or crypto ticker symbol(s) (e.g., AAPL, BTC-USD, TSLA). If not provided, will prompt interactively.'
    )
    
    parser.add_argument(
        '--period',
        type=str,
        default='2y',
        choices=['1y', '2y', '5y', '10y', 'max'],
        help='Historical data period (default: 2y)'
    )
    
    args = parser.parse_args()
    
    # If no tickers provided, ask interactively
    tickers_to_analyze = args.tickers
    if not tickers_to_analyze:
        print("\n" + "="*70)
        print("SINGLE TICKER ANALYZER")
        print("="*70)
        print("\nEnter ticker symbol(s) to analyze.")
        print("Examples: AAPL, BTC-USD, TSLA MSFT GOOGL")
        print("\nPopular Tickers:")
        print("  Stocks: AAPL, MSFT, GOOGL, TSLA, NVDA, AMZN, META, NFLX")
        print("  Crypto: BTC-USD, ETH-USD, SOL-USD, ADA-USD, BNB-USD")
        print("\n" + "-"*70)
        
        user_input = input("\nEnter ticker(s): ").strip()
        
        if not user_input:
            print("\n❌ No ticker provided. Exiting.")
            sys.exit(1)
        
        # Split by spaces to handle multiple tickers
        tickers_to_analyze = user_input.split()
    
    # Initialize analyzer
    print("\n🚀 Initializing Single Ticker Analyzer...")
    analyzer = SingleTickerAnalyzer()
    
    # Analyze each ticker
    all_results = []
    for ticker in tickers_to_analyze:
        try:
            result = analyzer.analyze(ticker, period=args.period)
            if result:
                analyzer.print_summary(result)
                all_results.append(result)
            else:
                print(f"\n❌ Failed to analyze {ticker}\n")
        except KeyboardInterrupt:
            print("\n\n⚠️  Analysis interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error analyzing {ticker}: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Summary if multiple tickers
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"{'Ticker':<12} {'Name':<30} {'Score':<8} {'Forecast':<10} {'Trend':<10}")
        print("-" * 70)
        
        for result in sorted(all_results, key=lambda x: x['composite_score'], reverse=True):
            trend_str = "🟢 BULL" if result['kalman_trend'] == 1 else "🔴 BEAR"
            name_short = result['name'][:28] + "..." if len(result['name']) > 30 else result['name']
            print(f"{result['ticker']:<12} {name_short:<30} {result['composite_score']:>6.1f}   "
                  f"{result['ml_forecast']:>+7.2f}%   {trend_str:<10}")
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

