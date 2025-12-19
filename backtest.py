#!/usr/bin/env python3
"""
Backtesting Engine for Top 10 ETF Portfolio Strategy
Tests performance of top 10 ETFs by composite score in MEDIUM and HIGH risk categories
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ETFBacktestEngine:
    """Backtest engine for top N ETF portfolio strategy"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.results = {}
        
    def load_rankings(self):
        """Load current rankings for medium and high risk ETFs"""
        print("Loading ETF rankings...")
        
        # Load rankings by risk category
        medium_rankings = pd.read_parquet(self.data_dir / 'rankings_medium_risk.parquet')
        high_rankings = pd.read_parquet(self.data_dir / 'rankings_high_risk.parquet')
        
        # Combine medium and high risk
        combined = pd.concat([medium_rankings, high_rankings], ignore_index=True)
        
        # Sort by composite score (descending)
        combined = combined.sort_values('composite_score', ascending=False)
        
        print(f"  Loaded {len(medium_rankings)} medium risk ETFs")
        print(f"  Loaded {len(high_rankings)} high risk ETFs")
        print(f"  Total: {len(combined)} ETFs")
        
        return combined
    
    def get_top_n_etfs(self, rankings, n=10):
        """Get top N ETFs by composite score"""
        return rankings.head(n)
    
    def download_price_data(self, tickers, period='60d'):
        """Download historical price data for backtesting"""
        print(f"Downloading price data for {len(tickers)} ETFs...")
        
        # Download data
        data = yf.download(tickers, period=period, progress=False)['Close']
        
        # Check for missing data
        missing = data.columns[data.isnull().any()].tolist()
        if missing:
            print(f"  Warning: Missing data for {len(missing)} ETFs: {missing[:5]}")
        
        return data
    
    def calculate_portfolio_returns(self, price_data, top_etfs, rebalance_days=20):
        """Calculate portfolio returns with periodic rebalancing"""
        print(f"Calculating portfolio returns (rebalance every {rebalance_days} days)...")
        
        # Get tickers of top ETFs
        top_tickers = top_etfs['ticker'].tolist()
        
        # Filter price data to top ETFs
        available_tickers = [t for t in top_tickers if t in price_data.columns]
        portfolio_prices = price_data[available_tickers].copy()
        
        if len(available_tickers) == 0:
            print("  Error: No price data available for selected ETFs")
            return None
        
        print(f"  Using {len(available_tickers)} ETFs in portfolio")
        
        # Calculate daily returns
        returns = portfolio_prices.pct_change().dropna()
        
        # Equal-weight portfolio returns
        portfolio_returns = returns.mean(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Check if we have data
        if len(cumulative_returns) == 0:
            print("  Error: No return data available")
            return None
        
        # Calculate metrics
        total_return = cumulative_returns.iloc[-1] - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (portfolio_returns.mean() * 252 - 0.0435) / volatility  # Using risk-free rate
        
        # Calculate drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        results = {
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(portfolio_returns)),
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'etf_count': len(available_tickers)
        }
        
        return results
    
    def run_backtest(self, top_n=10, rebalance_days=20):
        """Run complete backtest"""
        print("=" * 60)
        print("ETF PORTFOLIO BACKTEST")
        print("=" * 60)
        print(f"Strategy: Top {top_n} ETFs (Medium + High Risk)")
        print(f"Rebalance: Every {rebalance_days} days")
        print()
        
        # Load rankings
        rankings = self.load_rankings()
        
        # Get top N ETFs
        top_etfs = self.get_top_n_etfs(rankings, n=top_n)
        
        print("\nTop ETFs Selected:")
        for i, row in top_etfs.iterrows():
            print(f"  {i+1:2d}. {row['ticker']:8s} - Score: {row['composite_score']:.3f} - {row.get('etf_name', 'Unknown')}")
        
        # Download price data
        tickers = top_etfs['ticker'].tolist()
        price_data = self.download_price_data(tickers)
        
        # Calculate returns
        results = self.calculate_portfolio_returns(price_data, top_etfs, rebalance_days)
        
        if results is None:
            print("\nBacktest failed: No price data available")
            return None
        
        # Display results
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period: Last {len(results['portfolio_returns'])} trading days")
        print(f"ETFs in Portfolio: {results['etf_count']}")
        print()
        print(f"Total Return:        {results['total_return']:+.2%}")
        print(f"Annualized Return:   {results['annualized_return']:+.2%}")
        print(f"Volatility:          {results['volatility']:.2%}")
        print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:        {results['max_drawdown']:.2%}")
        
        # Save results
        self.save_results(results, top_etfs)
        
        return results
    
    def save_results(self, results, top_etfs):
        """Save backtest results to file"""
        output_dir = self.data_dir / 'backtest_results'
        output_dir.mkdir(exist_ok=True)
        
        # Save portfolio composition
        top_etfs.to_csv(output_dir / 'portfolio_composition.csv', index=False)
        
        # Save returns data
        results['portfolio_returns'].to_csv(output_dir / 'portfolio_returns.csv')
        
        # Save summary
        summary = {
            'total_return': results['total_return'],
            'annualized_return': results['annualized_return'],
            'volatility': results['volatility'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'etf_count': results['etf_count'],
            'backtest_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(output_dir / 'backtest_summary.csv', index=False)
        
        print(f"\nResults saved to: {output_dir}")


def main():
    """Run backtest"""
    engine = ETFBacktestEngine()
    
    # Run backtest with default parameters
    results = engine.run_backtest(top_n=10, rebalance_days=20)
    
    if results:
        print("\n✅ Backtest completed successfully!")
    else:
        print("\n❌ Backtest failed!")


if __name__ == "__main__":
    main()
