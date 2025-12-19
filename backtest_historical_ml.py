#!/usr/bin/env python3
"""
Historical Backtest with Actual ML Ensemble
Uses MLEnsembleProduction for true historical analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import warnings
import yfinance as yf
warnings.filterwarnings('ignore')

# Import system components
from data_manager.data_manager import ETFDataManager
from analyzers.ml_ensemble_production import MLEnsembleProduction

class HistoricalMLBacktest:
    """Historical backtest using actual ML ensemble"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / 'historical_ml_cache'
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.data_manager = ETFDataManager(data_dir)
        self.portfolio = {}
        
        # Initialize ML ensemble
        self.ml_ensemble = MLEnsembleProduction()
        
    def download_historical_prices(self, tickers, start_date, end_date):
        """Download and cache historical price data"""
        cache_file = self.cache_dir / f"historical_prices_{start_date}_{end_date}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        print(f"Downloading price data for {len(tickers)} ETFs...")
        
        all_data = {}
        chunk_size = 50
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            print(f"  Downloading chunk {i//chunk_size + 1}/{(len(tickers)-1)//chunk_size + 1}...")
            
            try:
                data = yf.download(chunk, start=start_date, end=end_date, progress=False)
                if not data.empty and 'Close' in data.columns:
                    if isinstance(data.columns, pd.MultiIndex):
                        close_data = data['Close']
                        valid_tickers = [col for col in close_data.columns if not close_data[col].isnull().all()]
                        for ticker in valid_tickers:
                            all_data[ticker] = close_data[ticker]
                    else:
                        all_data.update(data['Close'].to_dict())
            except Exception as e:
                print(f"    Warning: Failed chunk {i//chunk_size + 1}: {e}")
        
        price_df = pd.DataFrame(all_data)
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.ffill()
        
        # Cache results
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(price_df, f)
        except:
            pass
        
        print(f"  Downloaded data for {len(price_df.columns)} ETFs")
        return price_df
    
    def download_external_data(self, start_date, end_date):
        """Download historical external data for ML features"""
        cache_file = self.cache_dir / f"external_data_{start_date}_{end_date}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        print("Downloading external market data for ML features...")
        
        external_data = {}
        
        # Download key market indicators
        tickers = {
            'Gold': 'GC=F',
            'VIX': '^VIX',
            'Rates': '^TNX',  # 10-year Treasury yield
            'SP500': '^GSPC',
            'NASDAQ': '^IXIC',
            'Bonds': 'TLT'
        }
        
        for name, ticker in tickers.items():
            try:
                print(f"  Downloading {name}...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty and 'Close' in data.columns:
                    if isinstance(data.columns, pd.MultiIndex):
                        external_data[name] = data['Close'][ticker]
                    else:
                        external_data[name] = data['Close']
            except Exception as e:
                print(f"    Warning: Failed to download {name}: {e}")
        
        # Cache results
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(external_data, f)
        except:
            pass
        
        print(f"  Downloaded data for {len(external_data)} external indicators")
        return external_data
    
    def generate_ml_rankings(self, price_data, external_data, date, lookback_days=252):
        """Generate ML rankings for a specific historical date"""
        print(f"\nAnalyzing ETFs for {date.date()}...")
        
        # Debug: Show top 10 ETFs by ML forecast
        ml_rankings = []
        
        # Get data up to the ranking date
        end_date = pd.to_datetime(date)
        start_date = end_date - pd.Timedelta(days=lookback_days + 60)
        
        # Prepare external data for ML ensemble
        historical_external = {}
        for name, data in external_data.items():
            historical_external[name] = data.loc[:end_date]
        
        # Modify ML ensemble to remove forecast cap for testing
        original_max_range = self.ml_ensemble.max_forecast_range
        self.ml_ensemble.max_forecast_range = 0.50  # Increase to 50% cap
        
        # Temporarily replace external data in ML ensemble
        original_external = self.ml_ensemble.external_data
        self.ml_ensemble.external_data = historical_external
        
        for ticker in price_data.columns:
            prices = price_data[ticker].loc[:end_date].dropna()
            
            if len(prices) < 100:  # Need minimum data for ML
                continue
            
            # Prepare ETF data for ML
            etf_data = pd.DataFrame({
                'Close': prices,
                'Volume': pd.Series(0, index=prices.index)  # Dummy volume
            })
            
            try:
                # Get ML forecast
                ml_result = self.ml_ensemble.forecast_etf(etf_data)
                
                if ml_result and 'forecast_return' in ml_result:
                    # Calculate additional metrics
                    returns = prices.pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) if len(returns) > 20 else 0.15
                    
                    # Determine risk category based on volatility
                    if volatility < 0.15:
                        risk_category = 'LOW'
                    elif volatility < 0.25:
                        risk_category = 'MEDIUM'
                    else:
                        risk_category = 'HIGH'
                    
                    # Calculate composite score (ML forecast + confidence)
                    forecast = ml_result['forecast_return'] / 100  # Convert back from percentage
                    confidence = ml_result.get('confidence_score', 0.5)
                    composite_score = (forecast * 0.7 + confidence * 0.3) * 100  # Scale to 0-100
                    
                    # Simulate hit rate based on confidence
                    hit_rate = min(0.70, 0.45 + confidence * 0.5)
                    
                    # Calculate momentum for comparison
                    momentum = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else 0
                    
                    ml_rankings.append({
                        'ticker': ticker,
                        'composite_score': composite_score,
                        'ml_forecast': forecast,
                        'confidence_score': confidence,
                        'risk_category': risk_category,
                        'volatility': volatility,
                        'hit_rate': hit_rate,
                        'momentum': momentum,
                        'feature_importance': ml_result.get('feature_importance', {})
                    })
                    
            except Exception as e:
                print(f"    Warning: ML analysis failed for {ticker}: {e}")
                continue
        
        # Convert to DataFrame and sort
        ml_df = pd.DataFrame(ml_rankings)
        
        if not ml_df.empty:
            print(f"\n  Top 10 ETFs by ML Forecast:")
            top_ml = ml_df.nlargest(10, 'ml_forecast')
            for _, row in top_ml.iterrows():
                print(f"    {row['ticker']}: Forecast={row['ml_forecast']:.2%}, Composite={row['composite_score']:.1f}, Momentum={row['momentum']:.2%}")
            
            print(f"\n  Top 10 ETFs by Composite Score:")
            top_composite = ml_df.nlargest(10, 'composite_score')
            for _, row in top_composite.iterrows():
                print(f"    {row['ticker']}: Composite={row['composite_score']:.1f}, Forecast={row['ml_forecast']:.2%}, Confidence={row['confidence_score']:.2f}")
        
        # Restore original external data
        self.ml_ensemble.external_data = original_external
        
        # Restore original max range
        self.ml_ensemble.max_forecast_range = original_max_range
        
        return ml_df
    
    def update_portfolio(self, rankings, date, position_drop_threshold=5):
        """Update portfolio based on ML rankings"""
        # Group by risk category
        medium_rankings = rankings[rankings['risk_category'] == 'MEDIUM'].copy()
        high_rankings = rankings[rankings['risk_category'] == 'HIGH'].copy()
        
        # Add position rankings
        medium_rankings['position'] = range(1, len(medium_rankings) + 1)
        high_rankings['position'] = range(1, len(high_rankings) + 1)
        
        # Track positions to sell
        to_sell = []
        
        for ticker, position_info in list(self.portfolio.items()):
            risk_cat = position_info['risk_category']
            original_rank = position_info['rank']
            
            # Find current rank
            current_rankings = medium_rankings if risk_cat == 'MEDIUM' else high_rankings
            current_row = current_rankings[current_rankings['ticker'] == ticker]
            
            if len(current_row) == 0:
                to_sell.append(ticker)
                print(f"    {date.date()}: {ticker} no longer ranked - SELL")
            else:
                current_position = current_row['position'].iloc[0]
                position_drop = current_position - original_rank
                
                if position_drop >= position_drop_threshold:
                    to_sell.append(ticker)
                    print(f"    {date.date()}: {ticker} dropped {position_drop} positions - SELL")
        
        # Execute sells
        for ticker in to_sell:
            del self.portfolio[ticker]
        
        # Buy new positions (top 5 total)
        to_buy = []
        
        # Select top 3 from MEDIUM and top 2 from HIGH
        medium_available = medium_rankings[~medium_rankings['ticker'].isin(self.portfolio.keys())]
        high_available = high_rankings[~high_rankings['ticker'].isin(self.portfolio.keys())]
        
        if len(medium_available) >= len(high_available):
            to_buy.extend(medium_available.head(3)['ticker'].tolist())
            to_buy.extend(high_available.head(2)['ticker'].tolist())
        else:
            to_buy.extend(medium_available.head(2)['ticker'].tolist())
            to_buy.extend(high_available.head(3)['ticker'].tolist())
        
        # Add new positions
        for ticker in to_buy:
            if len(self.portfolio) < 5:
                risk_cat = rankings[rankings['ticker'] == ticker]['risk_category'].iloc[0]
                current_rankings = medium_rankings if risk_cat == 'MEDIUM' else high_rankings
                position = current_rankings[current_rankings['ticker'] == ticker]['position'].iloc[0]
                
                self.portfolio[ticker] = {
                    'buy_date': date,
                    'risk_category': risk_cat,
                    'rank': position,
                    'composite_score': rankings[rankings['ticker'] == ticker]['composite_score'].iloc[0],
                    'ml_forecast': rankings[rankings['ticker'] == ticker]['ml_forecast'].iloc[0],
                    'confidence_score': rankings[rankings['ticker'] == ticker]['confidence_score'].iloc[0],
                    'hit_rate': rankings[rankings['ticker'] == ticker]['hit_rate'].iloc[0],
                    'buy_price': None
                }
                print(f"    {date.date()}: Buying {ticker} (Score: {self.portfolio[ticker]['composite_score']:.1f}, Forecast: {self.portfolio[ticker]['ml_forecast']:.2%})")
    
    def run_historical_ml_backtest(self, start_date='2022-01-01', end_date='2024-12-31', 
                                  frequency='quarterly', initial_capital=100000):
        """Run historical backtest with actual ML analysis"""
        print("=" * 80)
        print("HISTORICAL ML BACKTEST WITH ACTUAL ML ENSEMBLE")
        print("=" * 80)
        print(f"Period: {start_date} to {end_date}")
        print(f"Rebalance: {frequency}")
        print(f"Strategy: Top 5 ETFs using ML ensemble forecasts")
        print()
        
        # Generate rebalance dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if frequency == 'quarterly':
            rebalance_dates = pd.date_range(start, end, freq='3M')
        else:
            rebalance_dates = pd.date_range(start, end, freq='3M')
        
        # Adjust to trading days
        trading_dates = []
        for date in rebalance_dates:
            if date.weekday() >= 5:
                date = date + pd.Timedelta(days=(7 - date.weekday()))
            trading_dates.append(date)
        
        print(f"Generated {len(trading_dates)} rebalance dates")
        
        # Get ETF universe
        universe_df = self.data_manager.load_universe()
        if universe_df.empty:
            print("Error: No universe data found")
            return None
        
        medium_high = universe_df[universe_df['risk_category'].isin(['MEDIUM', 'HIGH'])]
        tickers = medium_high['ticker'].tolist()
        
        print(f"Found {len(tickers)} medium and high risk ETFs")
        
        # Download price data
        price_data = self.download_historical_prices(tickers, start_date, end_date)
        
        if price_data.empty:
            print("Error: No price data available")
            return None
        
        # Download external data
        external_data = self.download_external_data(start_date, end_date)
        
        # Get benchmark
        benchmark_data = yf.download('^AXJO', start=start_date, end=end_date, progress=False)
        if isinstance(benchmark_data.columns, pd.MultiIndex):
            benchmark_data = benchmark_data['Close']['^AXJO']
        else:
            benchmark_data = benchmark_data['Close']
        
        benchmark_data = benchmark_data.reindex(price_data.index, method='ffill').fillna(method='bfill')
        
        # Initialize tracking
        portfolio_values = []
        dates = list(price_data.index)
        capital_per_position = initial_capital / 5
        current_portfolio = {}
        current_rankings_idx = 0
        rankings_history = {}
        
        print("\nRunning ML analysis and tracking portfolio:")
        
        # Process each day
        for date in dates:
            # Check if we need to rebalance
            if current_rankings_idx < len(trading_dates) and date >= trading_dates[current_rankings_idx]:
                # Generate ML rankings for this date
                rankings = self.generate_ml_rankings(price_data, external_data, date)
                
                if not rankings.empty:
                    rankings_history[trading_dates[current_rankings_idx]] = rankings
                    self.update_portfolio(rankings, date)
                    
                    # Set buy prices
                    for ticker in self.portfolio:
                        if ticker in price_data.columns:
                            buy_price = price_data.loc[date, ticker]
                            if not pd.isna(buy_price):
                                self.portfolio[ticker]['buy_price'] = buy_price
                    
                    # Copy portfolio
                    current_portfolio = {}
                    for ticker, info in self.portfolio.items():
                        current_portfolio[ticker] = info.copy()
                
                current_rankings_idx += 1
            
            # Calculate portfolio value
            total_value = 0
            
            for ticker, position_info in current_portfolio.items():
                if ticker in price_data.columns:
                    current_price = price_data.loc[date, ticker]
                    buy_price = position_info.get('buy_price')
                    
                    if not pd.isna(current_price) and buy_price is not None and not pd.isna(buy_price):
                        price_return = (current_price / buy_price) - 1
                        position_value = capital_per_position * (1 + price_return)
                        total_value += position_value
            
            # Add cash for empty positions
            total_value += (5 - len(current_portfolio)) * capital_per_position
            
            # Safety check
            if total_value == 0 and len(current_portfolio) > 0 and len(portfolio_values) > 0:
                total_value = portfolio_values[-1]
            
            portfolio_values.append(total_value)
            
            # Monthly summary
            if len(portfolio_values) % 20 == 0:
                print(f"  {date.date()}: Portfolio = ${total_value:,.2f} ({len(current_portfolio)} positions)")
        
        # Calculate performance
        portfolio_series = pd.Series(portfolio_values, index=dates)
        portfolio_returns = portfolio_series.pct_change().dropna()
        
        # Winsorize extreme returns
        portfolio_returns = portfolio_returns.clip(lower=-0.5, upper=0.5)
        
        total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
        volatility = float(portfolio_returns.std() * np.sqrt(252))
        
        risk_free_rate = 0.0435
        sharpe_ratio = (portfolio_returns.mean() * 252 - risk_free_rate) / volatility if volatility > 0 else 0
        max_drawdown = (portfolio_series / portfolio_series.expanding().max() - 1).min()
        
        # Benchmark metrics
        benchmark_normalized = (benchmark_data / benchmark_data.iloc[0]) * initial_capital
        benchmark_returns = benchmark_normalized.pct_change().dropna()
        
        if benchmark_data.iloc[0] == 0 or pd.isna(benchmark_data.iloc[0]):
            benchmark_return = 0.0
            benchmark_vol = 0.0
            benchmark_dd = 0.0
        else:
            benchmark_return = float((benchmark_normalized.iloc[-1] / benchmark_normalized.iloc[0]) - 1)
            benchmark_vol = float(benchmark_returns.std() * np.sqrt(252))
            benchmark_dd = float((benchmark_normalized / benchmark_normalized.expanding().max() - 1).min())
        
        # Display results
        print("\n" + "=" * 80)
        print("HISTORICAL ML BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"\nPortfolio Performance:")
        print(f"  Total Return:        {total_return:+.2%}")
        print(f"  Annualized Return:   {((1 + total_return) ** (1/3) - 1):+.2%}")
        print(f"  Volatility:          {volatility:.2%}")
        print(f"  Sharpe Ratio:        {sharpe_ratio:.2f}")
        print(f"  Max Drawdown:        {max_drawdown:.2%}")
        
        print(f"\nBenchmark (^AXJO) Performance:")
        print(f"  Total Return:        {benchmark_return:+.2%}")
        print(f"  Annualized Return:   {((1 + benchmark_return) ** (1/3) - 1):+.2%}")
        print(f"  Volatility:          {benchmark_vol:.2%}")
        print(f"  Max Drawdown:        {benchmark_dd:.2%}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Alpha (Total):        {total_return - benchmark_return:+.2%}")
        print(f"  Alpha (Annualized):   {((1 + total_return) ** (1/3) - 1) - ((1 + benchmark_return) ** (1/3) - 1):+.2%}")
        print(f"  Number of Rebalances: {len(trading_dates)}")
        
        # Year-by-year breakdown
        print(f"\nYear-by-Year Performance:")
        for year in [2022, 2023, 2024]:
            year_start = f"{year}-01-01"
            year_end = f"{year}-12-31"
            year_portfolio = portfolio_series.loc[year_start:year_end]
            year_benchmark = benchmark_normalized.loc[year_start:year_end]
            
            if len(year_portfolio) > 1:
                portfolio_return = float((year_portfolio.iloc[-1] / year_portfolio.iloc[0]) - 1)
                benchmark_return = float((year_benchmark.iloc[-1] / year_benchmark.iloc[0]) - 1)
                print(f"  {year}: Portfolio {portfolio_return:+.2%}, Benchmark {benchmark_return:+.2%}, Alpha {portfolio_return - benchmark_return:+.2%}")
        
        # Save results
        self.save_results(portfolio_series, benchmark_normalized, total_return, volatility, sharpe_ratio)
        
        return {
            'portfolio_return': total_return,
            'benchmark_return': benchmark_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def save_results(self, portfolio_values, benchmark_values, total_return, volatility, sharpe_ratio):
        """Save backtest results"""
        output_dir = self.data_dir / 'historical_ml_results'
        output_dir.mkdir(exist_ok=True)
        
        portfolio_values.to_csv(output_dir / 'historical_ml_portfolio.csv')
        benchmark_values.to_csv(output_dir / 'historical_ml_benchmark.csv')
        
        summary = pd.DataFrame([{
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'backtest_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        
        summary.to_csv(output_dir / 'historical_ml_summary.csv', index=False)
        
        print(f"\nResults saved to: {output_dir}")


def main():
    """Run historical ML backtest"""
    engine = HistoricalMLBacktest()
    
    results = engine.run_historical_ml_backtest(
        start_date='2022-01-01',
        end_date='2024-12-31',
        frequency='quarterly',
        initial_capital=100000
    )
    
    if results:
        print("\n✅ Historical ML backtest completed successfully!")
    else:
        print("\n❌ Backtest failed!")


if __name__ == "__main__":
    main()
