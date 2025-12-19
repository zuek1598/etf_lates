#!/usr/bin/env python3
"""
Volatility-Targeting ML Backtest
Uses ML to predict volatility and adjust position sizes accordingly
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import warnings
import yfinance as yf
from scipy import stats
warnings.filterwarnings('ignore')

from data_manager.data_manager import ETFDataManager
from analyzers.ml_ensemble_production import MLEnsembleProduction

class VolatilityTargetingBacktest:
    """Backtest using ML volatility predictions for risk-adjusted positioning"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / 'volatility_cache'
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.data_manager = ETFDataManager(data_dir)
        self.portfolio = {}
        self.ml_ensemble = MLEnsembleProduction()
        
    def download_historical_prices(self, tickers, start_date, end_date):
        """Download and cache historical price data"""
        cache_file = self.cache_dir / f"prices_{start_date}_{end_date}.pkl"
        
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
            try:
                data = yf.download(chunk, start=start_date, end=end_date, progress=False)
                if not data.empty and 'Close' in data.columns:
                    if isinstance(data.columns, pd.MultiIndex):
                        close_data = data['Close']
                        for ticker in close_data.columns:
                            if not close_data[ticker].isnull().all():
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
        
        return price_df
    
    def predict_volatility(self, prices: pd.Series) -> float:
        """Use ML to predict future volatility"""
        volumes = pd.Series(0, index=prices.index)
        etf_data = pd.DataFrame({'Close': prices, 'Volume': volumes})
        
        try:
            # Get historical volatility for training
            returns = prices.pct_change().dropna()
            if len(returns) < 100:
                return 0.15
            
            # Calculate realized volatilities for training
            historical_vols = returns.rolling(20).std().dropna() * np.sqrt(252)
            
            # Extract features for each point in time
            X_features = []
            y_targets = []
            
            for i in range(60, len(prices) - 20):
                window_prices = prices.iloc[:i+1]
                window_returns = returns.iloc[:i+1]
                
                # Extract ML features at this point
                try:
                    X = self.ml_ensemble._extract_production_features(window_prices, volumes.iloc[:i+1])
                    if len(X[0]) == 10:  # Ensure all features present
                        X_features.append(X[0])
                        # Target is volatility 20 days later
                        if i + 20 < len(historical_vols):
                            y_targets.append(historical_vols.iloc[i + 20])
                except:
                    continue
            
            if len(X_features) < 50:
                # Fallback to simple method
                current_vol = returns.tail(20).std() * np.sqrt(252)
                return np.clip(current_vol, 0.05, 0.50)
            
            # Train simple model to predict volatility
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            
            X = np.array(X_features)
            y = np.array(y_targets)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y)
            
            # Predict current volatility
            current_X = self.ml_ensemble._extract_production_features(prices, volumes)
            current_X_scaled = scaler.transform(current_X)
            predicted_vol = model.predict(current_X_scaled)[0]
            
            # Apply bounds
            predicted_vol = np.clip(predicted_vol, 0.05, 0.50)
            
            return predicted_vol
            
        except Exception as e:
            # Fallback to historical volatility
            returns = prices.pct_change().dropna()
            if len(returns) > 20:
                vol = returns.tail(20).std() * np.sqrt(252)
                return np.clip(vol, 0.05, 0.50)
            return 0.15
    
    def calculate_position_size(self, predicted_vol: float, target_vol: float = 0.15) -> float:
        """Calculate position size based on volatility prediction"""
        # Inverse volatility scaling
        vol_scalar = target_vol / predicted_vol
        return np.clip(vol_scalar, 0.5, 2.0)  # Limit to 0.5x-2x position
    
    def generate_volatility_rankings(self, price_data, date, lookback_days=252):
        """Generate rankings based on risk-adjusted momentum"""
        rankings = []
        
        end_date = pd.to_datetime(date)
        start_date = end_date - pd.Timedelta(days=lookback_days + 60)
        
        for ticker in price_data.columns:
            prices = price_data[ticker].loc[:end_date].dropna()
            
            if len(prices) < 100:
                continue
            
            try:
                # Calculate metrics
                returns = prices.pct_change().dropna()
                
                # 1. Momentum (20-day)
                momentum = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) > 20 else 0
                
                # 2. Predict future volatility
                predicted_vol = self.predict_volatility(prices)
                
                # 3. Risk-adjusted momentum (momentum / predicted volatility)
                risk_adjusted_momentum = momentum / predicted_vol if predicted_vol > 0 else 0
                
                # 4. Determine risk category
                current_vol = returns.std() * np.sqrt(252) if len(returns) > 20 else 0.15
                if current_vol < 0.15:
                    risk_category = 'LOW'
                elif current_vol < 0.25:
                    risk_category = 'MEDIUM'
                else:
                    risk_category = 'HIGH'
                
                # 5. Calculate position size
                position_size = self.calculate_position_size(predicted_vol)
                
                # 6. Composite score (risk-adjusted momentum)
                composite_score = risk_adjusted_momentum * 100
                
                rankings.append({
                    'ticker': ticker,
                    'composite_score': composite_score,
                    'risk_adjusted_momentum': risk_adjusted_momentum,
                    'momentum': momentum,
                    'predicted_volatility': predicted_vol,
                    'position_size': position_size,
                    'risk_category': risk_category,
                    'current_volatility': current_vol
                })
                
            except Exception as e:
                continue
        
        return pd.DataFrame(rankings)
    
    def update_portfolio(self, rankings, date, position_drop_threshold=5):
        """Update portfolio with volatility-adjusted positions"""
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
                
                row_data = rankings[rankings['ticker'] == ticker].iloc[0]
                
                self.portfolio[ticker] = {
                    'buy_date': date,
                    'risk_category': risk_cat,
                    'rank': position,
                    'composite_score': row_data['composite_score'],
                    'risk_adjusted_momentum': row_data['risk_adjusted_momentum'],
                    'momentum': row_data['momentum'],
                    'predicted_volatility': row_data['predicted_volatility'],
                    'position_size': row_data['position_size'],
                    'buy_price': None
                }
                print(f"    {date.date()}: Buying {ticker} (Score: {row_data['composite_score']:.1f}, Size: {row_data['position_size']:.2f}x)")
    
    def run_volatility_backtest(self, start_date='2022-01-01', end_date='2024-12-31', 
                               initial_capital=100000, target_volatility=0.15):
        """Run volatility-targeting backtest"""
        print("=" * 80)
        print("VOLATILITY-TARGETING ML BACKTEST")
        print("=" * 80)
        print(f"Period: {start_date} to {end_date}")
        print(f"Strategy: Risk-adjusted momentum with volatility prediction")
        print(f"Target Volatility: {target_volatility:.1%}")
        print()
        
        # Generate rebalance dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
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
        medium_high = universe_df[universe_df['risk_category'].isin(['MEDIUM', 'HIGH'])]
        tickers = medium_high['ticker'].tolist()
        
        print(f"Found {len(tickers)} medium and high risk ETFs")
        
        # Download price data
        price_data = self.download_historical_prices(tickers, start_date, end_date)
        
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
        base_position_size = initial_capital / 5
        current_portfolio = {}
        current_rankings_idx = 0
        
        print("\nRunning volatility-targeting strategy:")
        
        # Process each day
        for date in dates:
            # Check if we need to rebalance
            if current_rankings_idx < len(trading_dates) and date >= trading_dates[current_rankings_idx]:
                rankings = self.generate_volatility_rankings(price_data, date)
                
                if not rankings.empty:
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
            
            # Calculate portfolio value with volatility-adjusted positions
            total_value = 0
            
            for ticker, position_info in current_portfolio.items():
                if ticker in price_data.columns:
                    current_price = price_data.loc[date, ticker]
                    buy_price = position_info.get('buy_price')
                    position_size = position_info.get('position_size', 1.0)
                    
                    if not pd.isna(current_price) and buy_price is not None and not pd.isna(buy_price):
                        price_return = (current_price / buy_price) - 1
                        adjusted_position = base_position_size * position_size
                        position_value = adjusted_position * (1 + price_return)
                        total_value += position_value
            
            # Add cash for empty positions
            total_value += (5 - len(current_portfolio)) * base_position_size
            
            portfolio_values.append(total_value)
            
            # Monthly summary
            if len(portfolio_values) % 20 == 0:
                print(f"  {date.date()}: Portfolio = ${total_value:,.2f} ({len(current_portfolio)} positions)")
        
        # Calculate performance
        portfolio_series = pd.Series(portfolio_values, index=dates)
        portfolio_returns = portfolio_series.pct_change().dropna()
        
        total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
        volatility = float(portfolio_returns.std() * np.sqrt(252))
        
        risk_free_rate = 0.0435
        sharpe_ratio = (portfolio_returns.mean() * 252 - risk_free_rate) / volatility if volatility > 0 else 0
        max_drawdown = (portfolio_series / portfolio_series.expanding().max() - 1).min()
        
        # Benchmark metrics
        benchmark_normalized = (benchmark_data / benchmark_data.iloc[0]) * initial_capital
        benchmark_returns = benchmark_normalized.pct_change().dropna()
        benchmark_return = float((benchmark_normalized.iloc[-1] / benchmark_normalized.iloc[0]) - 1)
        benchmark_vol = float(benchmark_returns.std() * np.sqrt(252))
        benchmark_dd = float((benchmark_normalized / benchmark_normalized.expanding().max() - 1).min())
        
        # Display results
        print("\n" + "=" * 80)
        print("VOLATILITY-TARGETING BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"\nPortfolio Performance:")
        print(f"  Total Return:        {total_return:+.2%}")
        print(f"  Annualized Return:   {((1 + total_return) ** (1/3) - 1):+.2%}")
        print(f"  Volatility:          {volatility:.2%}")
        print(f"  Sharpe Ratio:        {sharpe_ratio:.2f}")
        print(f"  Max Drawdown:        {max_drawdown:.2%}")
        
        print(f"\nBenchmark (^AXJO) Performance:")
        print(f"  Total Return:        {benchmark_return:+.2%}")
        print(f"  Volatility:          {benchmark_vol:.2%}")
        print(f"  Sharpe Ratio:        {(benchmark_returns.mean() * 252 - risk_free_rate) / benchmark_vol:.2f}")
        print(f"  Max Drawdown:        {benchmark_dd:.2%}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Alpha (Total):        {total_return - benchmark_return:+.2%}")
        print(f"  Volatility Reduction: {(benchmark_vol - volatility) / benchmark_vol:+.1%}")
        print(f"  Sharpe Improvement:   {sharpe_ratio - ((benchmark_returns.mean() * 252 - risk_free_rate) / benchmark_vol):+.2f}")
        
        # Save results
        output_dir = self.data_dir / 'volatility_results'
        output_dir.mkdir(exist_ok=True)
        portfolio_series.to_csv(output_dir / 'volatility_portfolio.csv')
        benchmark_normalized.to_csv(output_dir / 'volatility_benchmark.csv')
        
        print(f"\nResults saved to: {output_dir}")
        
        return {
            'portfolio_return': total_return,
            'benchmark_return': benchmark_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }


def main():
    """Run volatility-targeting backtest"""
    engine = VolatilityTargetingBacktest()
    
    results = engine.run_volatility_backtest(
        start_date='2022-01-01',
        end_date='2024-12-31',
        initial_capital=100000,
        target_volatility=0.15
    )
    
    if results:
        print("\n✅ Volatility-targeting backtest completed successfully!")
    else:
        print("\n❌ Backtest failed!")


if __name__ == "__main__":
    main()
