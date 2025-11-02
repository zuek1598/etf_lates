"""
Strategy Comparator - Compare strategy returns vs buy-and-hold and benchmarks

Implements two comparison approaches:
1. Capital-Deployed: Return on capital deployed per signal
2. Full-Period: Return on total capital allocated (including idle periods)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import yfinance as yf


class StrategyComparator:
    """Compare strategy returns against buy-and-hold and benchmarks"""

    def __init__(self, backtest_results_path: str = None, period: str = 'full'):
        """
        Initialize comparator with backtest results

        Args:
            backtest_results_path: Path to backtest results file (None = auto-detect)
            period: Period for backtest results ('1m', '3m', '6m', '1y', 'full')
        """
        self.period = period
        self.results_df = None
        self.etf_data = {}
        self.benchmark_data = {}

        # Determine backtest results path
        if backtest_results_path is None:
            project_root = Path(__file__).parent.parent
            # Map period to filename
            if period.lower() == 'full':
                # Use default filename for backward compatibility
                backtest_results_path = project_root / 'data' / 'professional_backtest_results.parquet'
            else:
                # Use period-specific filename
                backtest_results_path = project_root / 'data' / f'backtest_results_{period}.parquet'

        self.backtest_results_path = Path(backtest_results_path)
        self._load_backtest_results()

    def _load_backtest_results(self):
        """Load backtest results from parquet"""
        if self.backtest_results_path.exists():
            self.results_df = pd.read_parquet(self.backtest_results_path)
        else:
            raise FileNotFoundError(f"Backtest results not found: {self.backtest_results_path}")

    def _get_etf_data(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.Series:
        """Download ETF price data"""
        try:
            # Add .AX suffix if not present
            if not ticker.endswith('.AX'):
                ticker = ticker + '.AX'

            etf = yf.Ticker(ticker)
            hist = etf.history(start=start_date, end=end_date)

            if hist.empty:
                raise ValueError(f"No data found for {ticker}")

            return hist['Close']
        except Exception as e:
            print(f"Warning: Could not fetch data for {ticker}: {e}")
            return None

    def _get_benchmark_data(self, benchmark: str, start_date: str = None, end_date: str = None) -> pd.Series:
        """Download benchmark data"""
        try:
            bench = yf.Ticker(benchmark)
            hist = bench.history(start=start_date, end=end_date)

            if hist.empty:
                raise ValueError(f"No data found for {benchmark}")

            return hist['Close']
        except Exception as e:
            print(f"Warning: Could not fetch benchmark {benchmark}: {e}")
            return None

    def compare_etf(self, ticker: str, start_date: str = None, end_date: str = None) -> Dict:
        """
        Compare strategy performance against buy-hold and benchmarks

        Returns: Dict with comparison metrics
        """
        # Get backtest result for this ETF
        result_row = self.results_df[self.results_df['ticker'] == ticker.upper()]
        if result_row.empty:
            raise ValueError(f"ETF {ticker} not found in backtest results")

        result = result_row.iloc[0].to_dict()
        strategy_return = float(result['total_return'])
        num_trades = int(result['num_trades'])
        win_rate = float(result['win_rate'])
        sharpe = float(result['sharpe_ratio'])
        avg_hold_days = float(result['avg_hold_days'])

        # Fetch ETF data
        etf_ticker = ticker.upper()
        if not etf_ticker.endswith('.AX'):
            etf_ticker = etf_ticker + '.AX'

        etf_prices = self._get_etf_data(etf_ticker, start_date, end_date)
        if etf_prices is None or len(etf_prices) < 2:
            raise ValueError(f"Insufficient data for {ticker}")

        # Set date range from ETF data
        actual_start = etf_prices.index[0].strftime('%Y-%m-%d')
        actual_end = etf_prices.index[-1].strftime('%Y-%m-%d')

        # Calculate buy-hold return
        buy_hold_return = (etf_prices.iloc[-1] - etf_prices.iloc[0]) / etf_prices.iloc[0]

        # Fetch benchmark data
        asx200_prices = self._get_benchmark_data('^AXJO', actual_start, actual_end)
        sp500_prices = self._get_benchmark_data('^GSPC', actual_start, actual_end)

        asx200_return = None
        sp500_return = None

        if asx200_prices is not None and len(asx200_prices) > 1:
            asx200_return = (asx200_prices.iloc[-1] - asx200_prices.iloc[0]) / asx200_prices.iloc[0]

        if sp500_prices is not None and len(sp500_prices) > 1:
            sp500_return = (sp500_prices.iloc[-1] - sp500_prices.iloc[0]) / sp500_prices.iloc[0]

        # Calculate period days
        period_days = (etf_prices.index[-1] - etf_prices.index[0]).days
        total_capital_deployed = num_trades * 10.0

        # Calculate Approach B metrics upfront
        capital_per_day = (total_capital_deployed / period_days) if period_days > 0 else 0
        utilization_pct = (capital_per_day / 10.0) * 100 if period_days > 0 else 0

        # Calculate metrics for both approaches
        comparison = {
            'ticker': ticker.upper(),
            'period_start': actual_start,
            'period_end': actual_end,
            'period_days': period_days,

            # Strategy metrics
            'strategy_deployed_return': strategy_return,
            'strategy_trades': num_trades,
            'strategy_win_rate': win_rate,
            'strategy_sharpe': sharpe,
            'strategy_avg_hold_days': avg_hold_days,

            # Buy-and-hold ETF
            'buyhold_return': buy_hold_return,

            # APPROACH A: Capital-Deployed Comparison
            'approach_a_strategy_return': strategy_return,
            'approach_a_buyhold_return': buy_hold_return,
            'approach_a_alpha_vs_buyhold': strategy_return - buy_hold_return,
            'approach_a_outperforms': strategy_return > buy_hold_return,

            # APPROACH B: Full-Period Comparison
            'total_capital_deployed': total_capital_deployed,
            'approach_b_capital_per_day': capital_per_day,
            'approach_b_capital_utilization': capital_per_day,
            'approach_b_utilization_pct': utilization_pct,

            # Benchmarks
            'asx200_return': asx200_return,
            'sp500_return': sp500_return,

            # Alpha vs benchmarks (Approach A)
            'alpha_vs_asx200': strategy_return - asx200_return if asx200_return is not None else None,
            'alpha_vs_sp500': strategy_return - sp500_return if sp500_return is not None else None,
        }

        return comparison

    def format_comparison(self, comparison: Dict) -> str:
        """Format comparison results for display"""
        ticker = comparison['ticker']
        period_start = comparison['period_start']
        period_end = comparison['period_end']
        period_days = comparison['period_days']

        lines = []
        lines.append('=' * 90)
        lines.append(f"STRATEGY COMPARISON: {ticker}")
        lines.append('=' * 90)
        lines.append(f"Period: {period_start} to {period_end} ({period_days} days)")
        lines.append('')

        # Strategy metrics
        lines.append('STRATEGY PERFORMANCE')
        lines.append('-' * 90)
        lines.append(f"  Return (Deployed Capital):    {comparison['strategy_deployed_return']*100:+7.2f}%")
        lines.append(f"  Trades Executed:              {comparison['strategy_trades']:3.0f}")
        lines.append(f"  Win Rate:                     {comparison['strategy_win_rate']:6.1f}%")
        lines.append(f"  Sharpe Ratio:                 {comparison['strategy_sharpe']:7.2f}")
        lines.append(f"  Avg Hold Period:              {comparison['strategy_avg_hold_days']:6.1f} days")
        lines.append(f"  Total Capital Deployed:       ${comparison['total_capital_deployed']:7.2f}")
        lines.append('')

        # Buy-and-hold
        lines.append('BUY-AND-HOLD ETF (Full Period)')
        lines.append('-' * 90)
        lines.append(f"  Total Return:                 {comparison['buyhold_return']*100:+7.2f}%")
        lines.append(f"  Capital Deployed:             $10.00")
        lines.append('')

        # Approach A
        lines.append('APPROACH A: CAPITAL-DEPLOYED COMPARISON')
        lines.append('-' * 90)
        lines.append(f"  Strategy Return:              {comparison['approach_a_strategy_return']*100:+7.2f}%")
        lines.append(f"  Buy-Hold Return:              {comparison['approach_a_buyhold_return']*100:+7.2f}%")
        lines.append(f"  Alpha (Strategy vs Buy-Hold): {comparison['approach_a_alpha_vs_buyhold']*100:+7.2f}%")

        if comparison['approach_a_outperforms']:
            lines.append(f"  Result: [YES] STRATEGY OUTPERFORMS")
        else:
            lines.append(f"  Result: [NO] BUY-HOLD OUTPERFORMS")
        lines.append('')
        lines.append("  How to interpret: This compares return on deployed capital.")
        lines.append("  Strategy deploys $10 per signal (total: ${:.0f})".format(comparison['total_capital_deployed']))
        lines.append("  Buy-hold has $10 invested from day 1.")
        lines.append('')

        # Approach B
        lines.append('APPROACH B: FULL-PERIOD COMPARISON')
        lines.append('-' * 90)
        util_pct = comparison['approach_b_utilization_pct']
        lines.append(f"  Strategy Capital Utilization: {util_pct:6.2f}%")
        lines.append(f"  (Average ${comparison['approach_b_capital_utilization']:.3f}/day deployed)")
        lines.append(f"  Buy-Hold Capital:             100.00%")
        lines.append('')
        lines.append("  Interpretation: Strategy deployed ${:.0f} total across {} days".format(
            comparison['total_capital_deployed'], period_days))
        lines.append("  Buy-hold had $10.00 deployed for entire period (100% utilization).")
        lines.append('')
        lines.append("  Fair Comparison:")
        lines.append(f"    If strategy had ${comparison['total_capital_deployed']:.0f} to invest like buy-hold,")
        lines.append(f"    but spread it using our signal timing instead of lump-sum,")
        lines.append(f"    the return would be {comparison['strategy_deployed_return']*100:+.2f}% on that capital.")
        lines.append('')

        # Benchmarks
        lines.append('BENCHMARK COMPARISONS')
        lines.append('-' * 90)

        if comparison['asx200_return'] is not None:
            alpha_asx = comparison['alpha_vs_asx200']
            lines.append(f"  ASX200 (^AXJO):               {comparison['asx200_return']*100:+7.2f}%")
            lines.append(f"  Alpha vs ASX200:              {alpha_asx*100:+7.2f}%")
            if comparison['strategy_deployed_return'] > comparison['asx200_return']:
                lines.append(f"  Result: [YES] STRATEGY OUTPERFORMS ASX200")
            else:
                lines.append(f"  Result: [NO] ASX200 OUTPERFORMS STRATEGY")
            lines.append('')

        if comparison['sp500_return'] is not None:
            alpha_sp = comparison['alpha_vs_sp500']
            lines.append(f"  S&P500 (^GSPC):               {comparison['sp500_return']*100:+7.2f}%")
            lines.append(f"  Alpha vs S&P500:              {alpha_sp*100:+7.2f}%")
            if comparison['strategy_deployed_return'] > comparison['sp500_return']:
                lines.append(f"  Result: [YES] STRATEGY OUTPERFORMS S&P500")
            else:
                lines.append(f"  Result: [NO] S&P500 OUTPERFORMS STRATEGY")
            lines.append('')

        # Summary table
        lines.append('SUMMARY TABLE')
        lines.append('-' * 90)
        lines.append(f"{'Metric':<35} {'Strategy':>15} {'Buy-Hold':>15} {'Benchmark':>15}")
        lines.append('-' * 90)

        # Return row with conditional benchmark
        return_line = f"{'Return':<35} {comparison['strategy_deployed_return']*100:>14.2f}% {comparison['buyhold_return']*100:>14.2f}%"
        if comparison['asx200_return'] is not None:
            return_line += f" {comparison['asx200_return']*100:>14.2f}%"
        else:
            return_line += " {'N/A':>14}"
        lines.append(return_line)

        lines.append(f"{'Trades/Capital Util':<35} {comparison['strategy_trades']:>14.0f} {'100.00%':>14} {'N/A':>15}")
        lines.append('')
        lines.append('=' * 90)

        return '\n'.join(lines)


def compare_multiple_etfs(tickers: list, show_summary: bool = True, period: str = 'full') -> pd.DataFrame:
    """
    Compare multiple ETFs at once

    Args:
        tickers: List of ETF tickers to compare
        show_summary: Whether to print summary table
        period: Backtest period ('1m', '3m', '6m', '1y', 'full')
    """
    comparator = StrategyComparator(period=period)
    results = []

    for ticker in tickers:
        try:
            comparison = comparator.compare_etf(ticker)
            results.append({
                'Ticker': ticker.upper(),
                'Strategy Return': f"{comparison['strategy_deployed_return']*100:.2f}%",
                'Buy-Hold Return': f"{comparison['buyhold_return']*100:.2f}%",
                'Alpha': f"{comparison['approach_a_alpha_vs_buyhold']*100:+.2f}%",
                'Outperforms': "Yes" if comparison['approach_a_outperforms'] else "No",
                'Trades': int(comparison['strategy_trades']),
                'Win Rate': f"{comparison['strategy_win_rate']:.1f}%",
                'Sharpe': f"{comparison['strategy_sharpe']:.2f}",
            })
        except Exception as e:
            print(f"Error comparing {ticker}: {e}")

    df = pd.DataFrame(results)

    if show_summary and len(df) > 0:
        print('\n' + '=' * 90)
        print('MULTIPLE ETF COMPARISON SUMMARY')
        print('=' * 90)
        print(df.to_string(index=False))
        print('')

    return df
