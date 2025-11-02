#!/usr/bin/env python3
"""
Professional Backtest Runner
Executes full universe backtesting on all classified ETFs
Handles both mature and immature ETFs with appropriate strategies
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


def main():
    """Run professional backtest on universe"""

    print("\n" + "="*80)
    print("PROFESSIONAL BACKTESTING - FULL UNIVERSE")
    print("="*80)

    # Load ETF classifications
    classification_path = project_root / 'data' / 'etf_data_classification.parquet'
    if not classification_path.exists():
        print(f"ERROR: Classification file not found: {classification_path}")
        print("Run: python scripts/validate_historical_data.py first")
        return

    classifications = pd.read_parquet(classification_path)
    backtest_eligible = classifications[classifications['can_backtest'] == True]

    print(f"\nBacktesting {len(backtest_eligible)} eligible ETFs:")
    print(f"  Mature (312+ days):      {(backtest_eligible['category'] == 'mature').sum():>4}")
    print(f"  Immature (60-311 days):  {(backtest_eligible['category'] == 'immature').sum():>4}")

    # Initialize backtester
    backtester = ProfessionalBacktester(
        min_hold_days=60,
        capital_per_trade=10.0,
        rebalance_frequency=30,
        buy_threshold=50.0,
        sell_threshold=40.0,
        target_return=0.125,
        stop_loss=-0.08,
        stale_days=180
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

        # Save to parquet
        output_path = project_root / 'data' / 'professional_backtest_results.parquet'
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

        print(f"\nTop 10 Performers:")
        top_10 = results_df.nlargest(10, 'total_return')[['ticker', 'total_return', 'win_rate', 'num_trades']]
        for idx, row in top_10.iterrows():
            print(f"  {row['ticker']:12} {row['total_return']:+7.2%} | "
                  f"Win: {row['win_rate']:5.1%} | Trades: {row['num_trades']:3.0f}")

        print(f"\nBottom 10 Performers:")
        bottom_10 = results_df.nsmallest(10, 'total_return')[['ticker', 'total_return', 'win_rate', 'num_trades']]
        for idx, row in bottom_10.iterrows():
            print(f"  {row['ticker']:12} {row['total_return']:+7.2%} | "
                  f"Win: {row['win_rate']:5.1%} | Trades: {row['num_trades']:3.0f}")

        print(f"\nResults saved to: {output_path}")

    else:
        print("No successful backtests!")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
