#!/usr/bin/env python3
"""
Multi-Period Backtest Runner
Executes backtesting for multiple lookback periods: 1m, 3m, 6m, 1y, full history
Generates separate result files for each period to match strategy comparator periods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utilities.professional_backtester import ProfessionalBacktester
from utilities.shared_utils import extract_column


def load_etf_data(ticker: str, historical_dir: Path) -> tuple:
    """Load ETF historical data"""
    try:
        file_path = historical_dir / f"{ticker.replace('.', '_')}.parquet"
        if not file_path.exists():
            return None, None

        data = pd.read_parquet(file_path)
        prices = extract_column(data, 'Close')

        if prices is None or len(prices) < 60:
            return None, None

        return prices, data

    except Exception as e:
        return None, None


def run_backtest_for_period(period_name: str, lookback_months: int = None):
    """Run backtest for a specific lookback period"""

    print("\n" + "="*80)
    print(f"PROFESSIONAL BACKTESTING - {period_name.upper()}")
    print("="*80)
    if lookback_months:
        print(f"Lookback Period: {lookback_months} months")
    else:
        print(f"Lookback Period: Full history")

    # Load ETF classifications
    classification_path = project_root / 'data' / 'etf_data_classification.parquet'
    if not classification_path.exists():
        print(f"ERROR: Classification file not found: {classification_path}")
        print("Run: python scripts/validate_historical_data.py first")
        return None

    classifications = pd.read_parquet(classification_path)
    
    # Filter ETFs based on period-specific requirements
    if lookback_months is None:
        # Full history - use standard eligibility
        backtest_eligible = classifications[classifications['can_backtest'] == True]
        period_desc = "full history"
    else:
        # Specific lookback period - require sufficient trading days for that period
        required_days = lookback_months * 21  # ~21 trading days per month
        backtest_eligible = classifications[classifications['total_days'] >= required_days]
        period_desc = f"last {lookback_months} months"
    
    print(f"\nBacktesting {len(backtest_eligible)} eligible ETFs for {period_desc}:")
    print(f"  Mature (312+ days):      {(backtest_eligible['category'] == 'mature').sum():>4}")
    print(f"  Immature (60-311 days):  {(backtest_eligible['category'] == 'immature').sum():>4}")

    # Initialize backtester with lookback_months parameter
    backtester = ProfessionalBacktester(
        min_hold_days=60,
        capital_per_trade=10.0,
        rebalance_frequency=30,
        buy_threshold=50.0,
        sell_threshold=40.0,
        target_return=0.125,
        stop_loss=-0.08,
        stale_days=180,
        lookback_months=lookback_months
    )

    # Historical data directory
    historical_dir = project_root / 'data' / 'historical'

    # Run backtests
    results = []
    successful = 0
    failed = 0

    print(f"\nRunning backtests...")
    start_time = time.time()

    for idx, (_, row) in enumerate(backtest_eligible.iterrows(), 1):
        ticker = row['ticker']
        category = row['category']
        peer_proxy = row['peer_proxy_ticker'] if pd.notna(row['peer_proxy_ticker']) else None

        # Load data
        prices, ohlc_data = load_etf_data(ticker, historical_dir)
        if prices is None:
            failed += 1
            continue

        # Determine risk category (simplified)
        returns = prices.pct_change().dropna()
        if len(returns) >= 60:
            vol_annual = returns.std() * np.sqrt(252)
            if vol_annual < 0.12:
                risk_category = 'LOW'
            elif vol_annual < 0.25:
                risk_category = 'MEDIUM'
            else:
                risk_category = 'HIGH'
        else:
            risk_category = 'MEDIUM'

        # Run appropriate backtest
        try:
            if category == 'mature':
                result = backtester.backtest_mature_etf(
                    ticker, prices, ohlc_data, risk_category
                )
            else:  # immature
                result = backtester.backtest_immature_etf(
                    ticker, prices, ohlc_data, risk_category, peer_proxy
                )

            if result['status'] == 'success':
                successful += 1
                # Suppress debug output for cleaner multi-period display
                print(f"  [{idx}/{len(backtest_eligible)}] {ticker:12} [{category:7}] "
                      f"Return: {result['total_return']:+7.2%} | "
                      f"Trades: {result['num_trades']:3} | "
                      f"Win: {result['win_rate']:5.1%} | "
                      f"Sharpe: {result['sharpe_ratio']:6.2f}")

                # Remove trades/positions lists for storage (too verbose)
                result.pop('trades', None)
                result.pop('positions', None)
                results.append(result)
            else:
                failed += 1
                print(f"  [{idx}/{len(backtest_eligible)}] {ticker:12} [{category:7}] "
                      f"Failed: {result.get('reason', 'Unknown')}")

        except Exception as e:
            failed += 1
            print(f"  [{idx}/{len(backtest_eligible)}] {ticker:12} [{category:7}] Error: {str(e)[:40]}")

        # Progress every 50
        if idx % 50 == 0:
            elapsed = time.time() - start_time
            per_etf = elapsed / idx
            remaining_etfs = len(backtest_eligible) - idx
            eta_mins = (per_etf * remaining_etfs) / 60
            print(f"     Progress: {idx}/{len(backtest_eligible)}, ETA: {eta_mins:.1f} min")

    # Save results
    elapsed = time.time() - start_time

    print(f"\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)

    if results:
        results_df = pd.DataFrame(results)

        # Save to parquet with period-specific filename
        output_filename = f'backtest_results_{period_name}.parquet'
        output_path = project_root / 'data' / output_filename
        results_df.to_parquet(output_path, compression='snappy', index=False)

        # Print summary statistics
        print(f"\nResults Summary:")
        print(f"  Successful:     {successful}")
        print(f"  Failed:         {failed}")
        print(f"  Success Rate:   {successful / len(backtest_eligible) * 100:.1f}%")
        print(f"  Time Elapsed:   {elapsed:.1f}s ({elapsed/60:.1f} min)")

        print(f"\nPerformance Metrics:")
        print(f"  Total Return:   {results_df['total_return'].mean():+7.2%}")
        print(f"  Avg Win Rate:   {results_df['win_rate'].mean():7.1%}")
        print(f"  Avg Sharpe:     {results_df['sharpe_ratio'].mean():7.2f}")
        print(f"  Avg Max DD:     {results_df['max_drawdown'].mean():7.1%}")
        print(f"  Avg Trades:     {results_df['num_trades'].mean():7.1f}")
        print(f"  Avg Hold Days:  {results_df['avg_hold_days'].mean():7.1f}")

        print(f"\nTrade Statistics:")
        total_trades = results_df['num_trades'].sum()
        total_capital = results_df['total_capital'].sum()
        total_pnl = results_df['total_pnl'].sum()
        total_return_portfolio = total_pnl / total_capital if total_capital > 0 else 0

        print(f"  Total Trades:   {total_trades:.0f}")
        print(f"  Total Capital:  ${total_capital:,.0f}")
        print(f"  Total P&L:      ${total_pnl:,.0f}")
        print(f"  Portfolio Return: {total_return_portfolio:+.2%}")

        print(f"\nTop 5 Performers:")
        top_5 = results_df.nlargest(5, 'total_return')[['ticker', 'total_return', 'win_rate', 'num_trades']]
        for idx, row in top_5.iterrows():
            print(f"  {row['ticker']:12} {row['total_return']:+7.2%} | "
                  f"Win: {row['win_rate']:5.1%} | Trades: {row['num_trades']:3.0f}")

        print(f"\nResults saved to: {output_path}")
        print()

        return results_df

    else:
        print("No successful backtests!")
        print()
        return None


def main():
    """Run backtests for multiple periods"""

    print("\n" + "="*80)
    print("MULTI-PERIOD BACKTEST RUNNER")
    print("="*80)
    print("\nThis will run backtests for periods: 1m, 3m, 6m, 1y, full")
    print("Each period will be saved to a separate results file")

    # Define periods to backtest
    # Format: (period_name, lookback_months)
    # None for lookback_months means full history
    periods = [
        ('1m', 1),
        ('3m', 3),
        ('6m', 6),
        ('1y', 12),
        ('full', None)
    ]

    results_summary = {}

    # Run backtest for each period
    for period_name, lookback_months in periods:
        results = run_backtest_for_period(period_name, lookback_months)
        if results is not None:
            results_summary[period_name] = {
                'count': len(results),
                'mean_return': results['total_return'].mean(),
                'positive_count': (results['total_return'] > 0).sum(),
                'positive_pct': (results['total_return'] > 0).sum() / len(results) * 100
            }

    # Print overall summary
    print("\n" + "="*80)
    print("MULTI-PERIOD SUMMARY")
    print("="*80)

    if results_summary:
        print(f"\n{'Period':<10} {'Count':<8} {'Mean Return':<15} {'Positive':<12}")
        print("-" * 50)
        for period in ['1m', '3m', '6m', '1y', 'full']:
            if period in results_summary:
                stats = results_summary[period]
                print(f"{period:<10} {stats['count']:<8} {stats['mean_return']:+10.2%}    "
                      f"{stats['positive_count']:.0f}/{stats['count']:.0f} "
                      f"({stats['positive_pct']:.1f}%)")

    print("\n" + "="*80)
    print("MULTI-PERIOD BACKTESTS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
