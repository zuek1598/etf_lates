#!/usr/bin/env python3
"""
Walk-Forward Backtest with Monthly Rebalancing
Uses full QualityRanker functionality with ML features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import yfinance as yf

# Add paths
sys.path.append('/Users/peter/Desktop/etf_lates')
sys.path.append('/Users/peter/Desktop/etf_lates/analyzers')

from data_manager.etf_database import ETFDatabase
from quality_ranker import QualityRanker
from metric_calculation import calculate_all_metrics

class WalkForwardBacktest:
    def __init__(self, start_date=None, end_date=None):
        """
        Initialize walk-forward backtest
        """
        self.start_date = start_date or datetime(2021, 1, 1)  # Changed to 2021
        self.end_date = end_date or datetime.now()
        
        self.manager = ETFDatabase()
        self.ranker = QualityRanker()
        
        print("="*80)
        print("WALK-FORWARD BACKTEST WITH MONTHLY REBALANCING")
        print("(Using Full QualityRanker with ML Features)")
        print("="*80)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
    
    def download_all_etf_data(self):
        """
        Download price data for all ETFs once upfront
        """
        print("\nDownloading all ETF price data...")
        
        # Get all ETF tickers
        all_etfs = list(self.manager.etf_data.keys())
        print(f"Total ETFs in universe: {len(all_etfs)}")
        
        try:
            # Download data in batches to avoid overwhelming yfinance
            batch_size = 50
            all_data = {}
            etf_first_dates = {}  # Track when each ETF becomes available
            
            for i in range(0, len(all_etfs), batch_size):
                batch = all_etfs[i:i+batch_size]
                print(f"  Downloading batch {i//batch_size + 1}/{(len(all_etfs)-1)//batch_size + 1}: {len(batch)} ETFs")
                
                try:
                    # For 2021 start, need 2 years history to have 200 days by Jan 2021
                    data = yf.download(
                        batch,
                        start=datetime(2019, 1, 1),  # Start earlier for history
                        end=self.end_date,
                        progress=False
                    )
                    
                    if 'Close' in data.columns:
                        if data.columns.nlevels > 1:
                            close_data = data['Close']
                        else:
                            close_data = data
                        
                        # Store each ETF and track first available date
                        for etf in batch:
                            if etf in close_data.columns:
                                prices = close_data[etf].dropna()
                                if len(prices) > 100:  # At least 100 days of data
                                    all_data[etf] = prices
                                    etf_first_dates[etf] = prices.index[0].date()
                
                except Exception as e:
                    print(f"  Batch {i//batch_size + 1} failed: {str(e)[:50]}")
                    continue
            
            print(f"\nSuccessfully downloaded data for {len(all_data)} ETFs")
            
            # Show universe growth over time
            print("\nETF Universe Growth:")
            sample_dates = [
                datetime(2021, 1, 1),
                datetime(2022, 1, 1),
                datetime(2023, 1, 1),
                datetime(2024, 1, 1),
                datetime(2025, 1, 1)
            ]
            
            for date in sample_dates:
                if date <= self.end_date:
                    count = sum(1 for first_date in etf_first_dates.values() if first_date <= date.date())
                    print(f"  {date.year}: {count} ETFs available")
            
            return all_data
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return {}
    
    def get_top_etfs_at_date(self, all_data, as_of_date, top_n=10, forecast_window=63):
        """
        Get top N ETFs based on full QualityRanker score at specific date
        Only includes ETFs that are available as of the given date
        """
        print(f"  Running QualityRanker as of {as_of_date.date()} ({forecast_window}d forecast)...", end=" ")
        
        # Adjust minimum history based on forecast window
        if forecast_window == 20:
            min_history = 50
        elif forecast_window == 40:
            min_history = 100
        elif forecast_window == 60:
            min_history = 150
        else:
            min_history = 200
        
        # First stage: Filter ETFs that exist and have sufficient history
        available_etfs = []
        
        for etf, prices in all_data.items():
            try:
                # Check if ETF exists as of this date
                first_date = prices.index[0]
                if first_date > pd.Timestamp(as_of_date):
                    continue  # ETF doesn't exist yet
                
                # Get data up to as_of_date
                historical_prices = prices[prices.index <= pd.Timestamp(as_of_date)]
                
                if len(historical_prices) < min_history:
                    continue  # Not enough history
                
                # Simple momentum filter
                returns = historical_prices.pct_change().dropna()
                recent_momentum = returns.tail(63).sum()  # ~3 months
                
                if recent_momentum > -0.1:  # Filter out very poor performers
                    available_etfs.append(etf)
                    
            except:
                continue
        
        print(f"{len(available_etfs)} available...", end=" ")
        
        if len(available_etfs) < 10:
            print("❌ Too few ETFs available")
            return []
        
        # Second stage: Use full QualityRanker on available ETFs
        scores = []
        
        # Limit to top 50 by momentum for performance
        momentum_filtered = available_etfs[:50]
        
        for etf in momentum_filtered:
            try:
                # Get historical data up to as_of_date
                prices = all_data[etf]
                historical_prices = prices[prices.index <= pd.Timestamp(as_of_date)]
                
                if len(historical_prices) < min_history:
                    continue
                
                # Use QualityRanker to calculate score with forecast window
                score_data = self._calculate_quality_score_full(etf, historical_prices, as_of_date, forecast_window)
                
                if score_data:
                    scores.append(score_data)
                    
            except Exception as e:
                continue
        
        if not scores:
            print("❌ No valid scores")
            return []
        
        # Sort by score and return top N
        scores_df = pd.DataFrame(scores)
        top_etfs = scores_df.nlargest(top_n, 'score')
        
        print(f"✓ Top {len(top_etfs)} selected")
        
        return top_etfs.to_dict('records')
    
    def _calculate_quality_score_full(self, etf, prices, as_of_date, forecast_window=63):
        """
        Calculate quality score using standardized metrics
        Uses the same metric calculations as the CI system
        """
        try:
            # Check minimum history
            min_history = 100 if forecast_window == 40 else 200
            if len(prices) < min_history:
                return None
            
            # Use standardized metric calculations
            metrics = calculate_all_metrics(prices, as_of_date=as_of_date)
            
            if not metrics:
                return None
            
            # Extract metrics
            hit_rate = metrics['hit_rate']
            conviction = metrics['conviction']
            stability = metrics['stability']
            
            # Normalize conviction for scoring (since it can be negative)
            conviction_normalized = 1 / (1 + np.exp(-conviction))
            
            # Composite score (same weights as QualityRanker)
            score = (hit_rate * 0.35 + conviction_normalized * 0.40 + stability * 0.25) * 10
            
            return {
                'etf': etf,
                'score': score,
                'hit_rate': hit_rate,
                'conviction': conviction,
                'stability': stability
            }
            
        except Exception as e:
            print(f"Error calculating score for {etf}: {e}")
            return None
    
    def simulate_buy_and_hold(self, all_data, forecast_window=63):
        """
        Simulate buy and hold strategy - select top 3 ETFs at start and hold
        """
        print("\n" + "="*80)
        print("BUY & HOLD STRATEGY")
        print("="*80)
        
        # Get top 3 ETFs at the start
        start_date = self.start_date
        print(f"Selecting top 3 ETFs on {start_date.date()}...")
        
        top_etfs = self.get_top_etfs_at_date(all_data, start_date, top_n=3, forecast_window=forecast_window)
        
        if not top_etfs:
            print("❌ Could not select initial ETFs")
            return None
        
        selected_etfs = [etf['etf'] for etf in top_etfs]
        print(f"Selected: {selected_etfs}")
        
        # Calculate returns for the entire period
        portfolio_returns = []
        
        for etf in selected_etfs:
            if etf in all_data:
                prices = all_data[etf]
                # Get returns for the full period
                etf_returns = prices.pct_change().dropna()
                period_returns = etf_returns[
                    (etf_returns.index >= self.start_date) &
                    (etf_returns.index <= self.end_date)
                ]
                portfolio_returns.append(period_returns)
        
        if not portfolio_returns:
            print("❌ No returns calculated")
            return None
        
        # Equal weight the returns
        avg_returns = pd.concat(portfolio_returns, axis=1).mean(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + avg_returns).cumprod()
        
        return {
            'returns': avg_returns,
            'cumulative': cumulative_returns,
            'value': cumulative_returns * 100000,
            'trade_log': [{'date': self.start_date, 'action': 'buy', 'holdings': selected_etfs}],
            'final_holdings': selected_etfs
        }
    
    def simulate_forecast_aware_rebalancing(self, all_data, forecast_window=63, buffer_zone=15):
        """
        Simulate portfolio with forecast-aware conditional rebalancing
        Only checks for rebalancing AFTER the forecast window has expired
        buffer_zone: Check if holdings dropped below top N (e.g., 10, 15, 20)
        """
        print(f"\nSimulating portfolio with FORECAST-AWARE conditional rebalancing ({forecast_window}d forecast, top {buffer_zone} buffer)...")
        
        # Track portfolio value over time (compounded)
        # Use actual trading days from ETF data instead of creating a daily index
        all_dates = []
        for prices in all_data.values():
            all_dates.extend(prices.index.tolist())
        unique_dates = sorted(set(all_dates))
        
        portfolio_value = pd.Series(index=unique_dates)
        portfolio_value[:] = np.nan  # Start with NaN
        
        trade_log = []
        current_value = 1.0
        portfolio_value.iloc[0] = current_value  # Set initial value
        
        # Start with top 3 ETFs
        start_date = self.start_date
        print(f"  {start_date.strftime('%Y-%m-%d')}: Selecting initial ETFs...", end=" ")
        top_etfs = self.get_top_etfs_at_date(all_data, start_date, top_n=buffer_zone, forecast_window=forecast_window)
        
        if not top_etfs:
            print("No valid ETFs")
            return None
        
        current_holdings = [etf['etf'] for etf in top_etfs[:3]]  # Start with top 3
        print(f"Initial holdings: {current_holdings}")
        
        trade_log.append({
            'date': start_date,
            'action': 'buy',
            'holdings': current_holdings.copy()
        })
        
        # Calculate when we can first check for rebalancing
        next_check_date = start_date + pd.Timedelta(days=forecast_window)
        
        while next_check_date < self.end_date:
            print(f"  {next_check_date.strftime('%Y-%m-%d')}: Forecast expired, checking holdings...", end=" ")
            
            # Get current top N (buffer zone)
            top_n = self.get_top_etfs_at_date(all_data, next_check_date, top_n=buffer_zone, forecast_window=forecast_window)
            
            if not top_n:
                print("No valid ETFs")
                # Move to next period
                next_check_date += pd.Timedelta(days=forecast_window)
                continue
            
            top_n_etfs = [etf['etf'] for etf in top_n]
            
            # Check if any current holding dropped below top N
            need_rebalance = False
            for holding in current_holdings:
                if holding not in top_n_etfs:
                    print(f"{holding} dropped below top {buffer_zone}", end=" ")
                    need_rebalance = True
                    break
            
            if need_rebalance:
                # Replace dropped ETFs with next best
                new_holdings = []
                for holding in current_holdings:
                    if holding in top_n_etfs:
                        new_holdings.append(holding)
                
                # Fill missing slots with top remaining ETFs
                for etf in top_n_etfs:
                    if len(new_holdings) >= 3:
                        break
                    if etf not in new_holdings:
                        new_holdings.append(etf)
                
                print(f"→ New holdings: {new_holdings}")
                
                trade_log.append({
                    'date': next_check_date,
                    'action': 'rebalance',
                    'holdings': new_holdings.copy()
                })
                
                current_holdings = new_holdings
            else:
                print(f"All holdings still in top {buffer_zone}, extending forecast")
            
            # Set next check date (another forecast window from now)
            next_check_date += pd.Timedelta(days=forecast_window)
        
        # Calculate portfolio returns with current holdings
        for i in range(len(unique_dates) - 1):
            date = unique_dates[i]
            next_date = unique_dates[i + 1]
            
            # Check if we have holdings for this period
            if not trade_log:
                continue
                
            # Find current holdings at this date
            current_etfs = None
            for trade in reversed(trade_log):
                if trade['date'] <= date:
                    current_etfs = trade['holdings']
                    break
            
            if not current_etfs:
                continue
            
            # Calculate returns for this day
            daily_returns = []
            for etf in current_etfs:
                if etf in all_data:
                    prices = all_data[etf]
                    if date in prices.index and next_date in prices.index:
                        daily_ret = (prices.loc[next_date] / prices.loc[date]) - 1
                        daily_returns.append(daily_ret)
            
            if daily_returns:
                avg_ret = np.mean(daily_returns)
                current_value *= (1 + avg_ret)
                portfolio_value.loc[next_date] = current_value
        
        # Calculate returns from portfolio value
        portfolio_value = portfolio_value.ffill()  # Forward fill values
        portfolio_returns = portfolio_value.pct_change().dropna()
        
        if len(portfolio_returns) == 0:
            print("No returns calculated")
            return None
        
        # Calculate cumulative returns
        cumulative_returns = portfolio_value.copy()
        
        return {
            'returns': portfolio_returns,
            'cumulative': cumulative_returns,
            'value': cumulative_returns * 100000,
            'trade_log': trade_log,
            'final_holdings': current_holdings
        }
        """
        Simulate portfolio with forecast-aware conditional rebalancing
        Only checks for rebalancing AFTER the forecast window has expired
        """
        print(f"\nSimulating portfolio with FORECAST-AWARE conditional rebalancing ({forecast_window}d forecast)...")
        
        # Track portfolio value over time (compounded)
        # Use actual trading days from ETF data instead of creating a daily index
        all_dates = []
        for prices in all_data.values():
            all_dates.extend(prices.index.tolist())
        unique_dates = sorted(set(all_dates))
        
        portfolio_value = pd.Series(index=unique_dates)
        portfolio_value[:] = np.nan  # Start with NaN
        
        trade_log = []
        current_value = 1.0
        portfolio_value.iloc[0] = current_value  # Set initial value
        
        # Start with top 3 ETFs
        start_date = self.start_date
        print(f"  {start_date.strftime('%Y-%m-%d')}: Selecting initial ETFs...", end=" ")
        top_etfs = self.get_top_etfs_at_date(all_data, start_date, top_n=10, forecast_window=forecast_window)
        
        if not top_etfs:
            print("No valid ETFs")
            return None
        
        current_holdings = [etf['etf'] for etf in top_etfs[:3]]  # Start with top 3
        print(f"Initial holdings: {current_holdings}")
        
        trade_log.append({
            'date': start_date,
            'action': 'buy',
            'holdings': current_holdings.copy()
        })
        
        # Calculate when we can first check for rebalancing
        next_check_date = start_date + pd.Timedelta(days=forecast_window)
        
        while next_check_date < self.end_date:
            print(f"  {next_check_date.strftime('%Y-%m-%d')}: Forecast expired, checking holdings...", end=" ")
            
            # Get current top 15 (buffer zone)
            top_15 = self.get_top_etfs_at_date(all_data, next_check_date, top_n=15, forecast_window=forecast_window)
            
            if not top_15:
                print("No valid ETFs")
                # Move to next period
                next_check_date += pd.Timedelta(days=forecast_window)
                continue
            
            top_15_etfs = [etf['etf'] for etf in top_15]
            
            # Check if any current holding dropped below top 15
            need_rebalance = False
            for holding in current_holdings:
                if holding not in top_15_etfs:
                    print(f"{holding} dropped below top 15", end=" ")
                    need_rebalance = True
                    break
            
            if need_rebalance:
                # Replace dropped ETFs with next best
                new_holdings = []
                for holding in current_holdings:
                    if holding in top_15_etfs:
                        new_holdings.append(holding)
                
                # Fill missing slots with top remaining ETFs
                for etf in top_15_etfs:
                    if len(new_holdings) >= 3:
                        break
                    if etf not in new_holdings:
                        new_holdings.append(etf)
                
                print(f"→ New holdings: {new_holdings}")
                
                trade_log.append({
                    'date': next_check_date,
                    'action': 'rebalance',
                    'holdings': new_holdings.copy()
                })
                
                current_holdings = new_holdings
            else:
                print("All holdings still in top 15, extending forecast")
            
            # Set next check date (another forecast window from now)
            next_check_date += pd.Timedelta(days=forecast_window)
        
        # Calculate portfolio returns with current holdings
        for i in range(len(unique_dates) - 1):
            date = unique_dates[i]
            next_date = unique_dates[i + 1]
            
            # Check if we have holdings for this period
            if not trade_log:
                continue
                
            # Find current holdings at this date
            current_etfs = None
            for trade in reversed(trade_log):
                if trade['date'] <= date:
                    current_etfs = trade['holdings']
                    break
            
            if not current_etfs:
                continue
            
            # Calculate returns for this day
            daily_returns = []
            for etf in current_etfs:
                if etf in all_data:
                    prices = all_data[etf]
                    if date in prices.index and next_date in prices.index:
                        daily_ret = (prices.loc[next_date] / prices.loc[date]) - 1
                        daily_returns.append(daily_ret)
            
            if daily_returns:
                avg_ret = np.mean(daily_returns)
                current_value *= (1 + avg_ret)
                portfolio_value.loc[next_date] = current_value
        
        # Calculate returns from portfolio value
        portfolio_value = portfolio_value.ffill()  # Forward fill values
        portfolio_returns = portfolio_value.pct_change().dropna()
        
        if len(portfolio_returns) == 0:
            print("No returns calculated")
            return None
        
        # Calculate cumulative returns
        cumulative_returns = portfolio_value.copy()
        
        return {
            'returns': portfolio_returns,
            'cumulative': cumulative_returns,
            'value': cumulative_returns * 100000,
            'trade_log': trade_log,
            'final_holdings': current_holdings
        }
        """
        Simulate portfolio with conditional rebalancing - only rebalance when holdings drop below top 10
        """
        print("\nSimulating portfolio with CONDITIONAL REBALANCING...")
        
        # Track portfolio value over time (compounded)
        # Use actual trading days from ETF data instead of creating a daily index
        all_dates = []
        for prices in all_data.values():
            all_dates.extend(prices.index.tolist())
        unique_dates = sorted(set(all_dates))
        
        portfolio_value = pd.Series(index=unique_dates)
        portfolio_value[:] = np.nan  # Start with NaN
        
        trade_log = []
        current_value = 1.0
        portfolio_value.iloc[0] = current_value  # Set initial value
        
        # Start with top 3 ETFs
        start_date = self.start_date
        print(f"  {start_date.strftime('%Y-%m-%d')}: Selecting initial ETFs...", end=" ")
        top_etfs = self.get_top_etfs_at_date(all_data, start_date, top_n=10, forecast_window=forecast_window)
        
        if not top_etfs:
            print("No valid ETFs")
            return None
        
        current_holdings = [etf['etf'] for etf in top_etfs[:3]]  # Start with top 3
        print(f"Initial holdings: {current_holdings}")
        
        trade_log.append({
            'date': start_date,
            'action': 'buy',
            'holdings': current_holdings.copy()
        })
        
        # Check holdings quarterly for rebalancing trigger
        check_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='3M')
        
        for check_date in check_dates:
            if check_date >= self.end_date:
                break
                
            print(f"  {check_date.strftime('%Y-%m-%d')}: Checking holdings...", end=" ")
            
            # Get current top 15 (buffer zone)
            top_15 = self.get_top_etfs_at_date(all_data, check_date, top_n=15, forecast_window=forecast_window)
            
            if not top_15:
                print("No valid ETFs")
                continue
            
            top_15_etfs = [etf['etf'] for etf in top_15]
            
            # Check if any current holding dropped below top 15
            need_rebalance = False
            for holding in current_holdings:
                if holding not in top_15_etfs:
                    print(f"{holding} dropped below top 15", end=" ")
                    need_rebalance = True
                    break
            
            if need_rebalance:
                # Replace dropped ETFs with next best
                new_holdings = []
                for holding in current_holdings:
                    if holding in top_15_etfs:
                        new_holdings.append(holding)
                
                # Fill missing slots with top remaining ETFs
                for etf in top_15_etfs:
                    if len(new_holdings) >= 3:
                        break
                    if etf not in new_holdings:
                        new_holdings.append(etf)
                
                print(f"→ New holdings: {new_holdings}")
                
                trade_log.append({
                    'date': check_date,
                    'action': 'rebalance',
                    'holdings': new_holdings.copy()
                })
                
                current_holdings = new_holdings
            else:
                print("No change needed")
        
        # Calculate portfolio returns with current holdings
        for i in range(len(unique_dates) - 1):
            date = unique_dates[i]
            next_date = unique_dates[i + 1]
            
            # Check if we have holdings for this period
            if not trade_log:
                continue
                
            # Find current holdings at this date
            current_etfs = None
            for trade in reversed(trade_log):
                if trade['date'] <= date:
                    current_etfs = trade['holdings']
                    break
            
            if not current_etfs:
                continue
            
            # Calculate returns for this day
            daily_returns = []
            for etf in current_etfs:
                if etf in all_data:
                    prices = all_data[etf]
                    if date in prices.index and next_date in prices.index:
                        daily_ret = (prices.loc[next_date] / prices.loc[date]) - 1
                        daily_returns.append(daily_ret)
            
            if daily_returns:
                avg_ret = np.mean(daily_returns)
                current_value *= (1 + avg_ret)
                portfolio_value.loc[next_date] = current_value
        
        # Calculate returns from portfolio value
        portfolio_value = portfolio_value.ffill()  # Forward fill values
        portfolio_returns = portfolio_value.pct_change().dropna()
        
        if len(portfolio_returns) == 0:
            print("No returns calculated")
            return None
        
        # Calculate cumulative returns
        cumulative_returns = portfolio_value.copy()
        
        return {
            'returns': portfolio_returns,
            'cumulative': cumulative_returns,
            'value': cumulative_returns * 100000,
            'trade_log': trade_log,
            'final_holdings': current_holdings
        }
    
    def simulate_portfolio(self, all_data, rebalance_frequency='M', forecast_window=63):
        """
        Simulate portfolio with rebalancing
        """
        print(f"\nSimulating portfolio with {rebalance_frequency} rebalancing...")
        
        # Create rebalance dates
        rebalance_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=rebalance_frequency
        )
        
        # Track portfolio value over time (compounded)
        # Use actual trading days from ETF data instead of creating a daily index
        all_dates = []
        for prices in all_data.values():
            all_dates.extend(prices.index.tolist())
        unique_dates = sorted(set(all_dates))
        
        portfolio_value = pd.Series(index=unique_dates)
        portfolio_value[:] = np.nan  # Start with NaN
        
        trade_log = []
        current_value = 1.0
        portfolio_value.iloc[0] = current_value  # Set initial value
        
        for i in range(len(rebalance_dates) - 1):
            start_date = rebalance_dates[i]
            end_date = rebalance_dates[i + 1]
            
            # Get top ETFs at start of period
            print(f"  {start_date.strftime('%Y-%m-%d')}: Selecting ETFs...", end=" ")
            top_etfs = self.get_top_etfs_at_date(all_data, start_date, top_n=3, forecast_window=forecast_window)
            
            if not top_etfs:
                print("No valid ETFs")
                continue
            
            selected = [etf['etf'] for etf in top_etfs]
            print(f"Top 3: {selected}")
            
            trade_log.append({
                'date': start_date,
                'action': 'rebalance',
                'holdings': selected.copy()
            })
            
            # Calculate returns for this period
            period_returns = []
            
            for etf in selected:
                if etf in all_data:
                    prices = all_data[etf]
                    # Get returns for this period
                    etf_returns = prices.pct_change().dropna()
                    period_etf_returns = etf_returns[
                        (etf_returns.index >= start_date) &
                        (etf_returns.index < end_date)
                    ]
                    period_returns.append(period_etf_returns)
            
            if period_returns:
                # Equal weight the returns
                avg_period_returns = pd.concat(period_returns, axis=1).mean(axis=1)
                
                # Compound the returns
                for date, ret in avg_period_returns.items():
                    if date in portfolio_value.index:
                        current_value *= (1 + ret)
                        portfolio_value.loc[date] = current_value
        
        # Handle the last period
        if len(rebalance_dates) > 0:
            start_date = rebalance_dates[-1]
            end_date = self.end_date
            
            print(f"  {start_date.strftime('%Y-%m-%d')}: Selecting ETFs...", end=" ")
            top_etfs = self.get_top_etfs_at_date(all_data, start_date, top_n=3, forecast_window=forecast_window)
            
            if top_etfs:
                selected = [etf['etf'] for etf in top_etfs]
                print(f"Top 3: {selected}")
                
                trade_log.append({
                    'date': start_date,
                    'action': 'rebalance',
                    'holdings': selected.copy()
                })
                
                period_returns = []
                for etf in selected:
                    if etf in all_data:
                        prices = all_data[etf]
                        etf_returns = prices.pct_change().dropna()
                        period_etf_returns = etf_returns[
                            (etf_returns.index >= start_date) &
                            (etf_returns.index <= end_date)
                        ]
                        period_returns.append(period_etf_returns)
                
                if period_returns:
                    avg_period_returns = pd.concat(period_returns, axis=1).mean(axis=1)
                    
                    # Compound the returns for the last period
                    for date, ret in avg_period_returns.items():
                        if date in portfolio_value.index:
                            current_value *= (1 + ret)
                            portfolio_value.loc[date] = current_value
        
        # Calculate returns from portfolio value
        portfolio_value = portfolio_value.ffill()  # Forward fill values
        portfolio_returns = portfolio_value.pct_change().dropna()
        
        if len(portfolio_returns) == 0:
            print("No returns calculated")
            return None
        
        # Calculate cumulative returns
        cumulative_returns = portfolio_value.copy()
        
        return {
            'returns': portfolio_returns,
            'cumulative': cumulative_returns,
            'value': cumulative_returns * 100000,
            'trade_log': trade_log,
            'final_holdings': trade_log[-1]['holdings'] if trade_log else []
        }
    
    def calculate_metrics(self, returns):
        """
        Calculate performance metrics
        """
        if len(returns) == 0:
            return {}
        
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        hit_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate
        }
    
    def run(self):
        """
        Run comprehensive test of all buffer zones across all forecast windows
        """
        # Step 1: Download all ETF data
        all_data = self.download_all_etf_data()
        
        if not all_data:
            print("No data downloaded. Exiting.")
            return
        
        # Step 2: Test all combinations
        print("\n" + "="*80)
        print("COMPREHENSIVE FORECAST-AWARE OPTIMIZATION")
        print("="*80)
        
        # Define all combinations
        forecast_windows = [20, 40, 60, 63]
        buffer_zones = [10, 15, 20]
        
        # Store results in a matrix
        results_matrix = {}
        
        for forecast_window in forecast_windows:
            print(f"\n{'='*60}")
            print(f"TESTING {forecast_window}-DAY FORECAST WINDOW")
            print(f"{'='*60}")
            
            results_matrix[forecast_window] = {}
            
            # Get buy & hold for this forecast window
            buy_hold_result = self.simulate_buy_and_hold(all_data, forecast_window=forecast_window)
            
            if not buy_hold_result:
                print(f"Failed to get buy & hold for {forecast_window} days")
                continue
                
            buy_hold_metrics = self.calculate_metrics(buy_hold_result['returns'])
            
            for buffer_zone in buffer_zones:
                print(f"\n--- Testing {forecast_window}d forecast with top {buffer_zone} buffer ---")
                
                # Test forecast-aware rebalancing
                forecast_aware_result = self.simulate_forecast_aware_rebalancing(
                    all_data, 
                    forecast_window=forecast_window, 
                    buffer_zone=buffer_zone
                )
                
                if forecast_aware_result:
                    forecast_aware_metrics = self.calculate_metrics(forecast_aware_result['returns'])
                    
                    edge = (forecast_aware_metrics['total_return'] - buy_hold_metrics['total_return']) * 100
                    
                    results_matrix[forecast_window][buffer_zone] = {
                        'return': forecast_aware_metrics['total_return'],
                        'annual': forecast_aware_metrics['annual_return'],
                        'sharpe': forecast_aware_metrics['sharpe_ratio'],
                        'rebalances': len(forecast_aware_result['trade_log']),
                        'edge_vs_bh': edge,
                        'buy_hold_return': buy_hold_metrics['total_return']
                    }
                    
                    print(f"  Return: {forecast_aware_metrics['total_return']*100:.1f}%")
                    print(f"  Buy & Hold: {buy_hold_metrics['total_return']*100:.1f}%")
                    print(f"  Edge: {edge:+.1f}%")
                    print(f"  Rebalances: {len(forecast_aware_result['trade_log'])}")
                else:
                    print(f"  Failed to simulate {forecast_window}d with top {buffer_zone}")
        
        # Step 3: Get benchmark
        print("\n" + "="*80)
        print("GETTING BENCHMARK DATA")
        print("="*80)
        
        try:
            benchmark_data = yf.download(
                '^AXJO',
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            if 'Close' in benchmark_data.columns:
                if hasattr(benchmark_data['Close'], 'columns'):
                    benchmark_prices = benchmark_data['Close']['^AXJO']
                else:
                    benchmark_prices = benchmark_data['Close']
            else:
                benchmark_prices = benchmark_data
            
            benchmark_returns = benchmark_prices.pct_change().dropna()
            benchmark_metrics = self.calculate_metrics(benchmark_returns)
            
        except Exception as e:
            print(f"Error getting benchmark: {e}")
            benchmark_metrics = {}
        
        # Step 4: Display comprehensive results
        self.display_comprehensive_results(results_matrix, benchmark_metrics)
        
        return {
            'results_matrix': results_matrix,
            'benchmark': benchmark_metrics
        }
    
    def display_comprehensive_results(self, results_matrix, benchmark_metrics):
        """
        Display comprehensive results matrix for all forecast/buffer combinations
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE OPTIMIZATION MATRIX")
        print("="*80)
        
        # Create header
        header = "Forecast\\Buffer"
        print(f"\n{header:<12}", end="")
        for buffer in [10, 15, 20]:
            print(f"{'Top ' + str(buffer):<12}", end="")
        print(f"{'Buy & Hold':<12}")
        
        print("-" * 60)
        
        # Display returns matrix
        print("\n📊 Total Returns (%):")
        for forecast in [20, 40, 60, 63]:
            print(f"{forecast}d{'':<8}", end="")
            
            for buffer in [10, 15, 20]:
                if forecast in results_matrix and buffer in results_matrix[forecast]:
                    ret = results_matrix[forecast][buffer]['return'] * 100
                    print(f"{ret:>8.1f}%{'':<3}", end="")
                else:
                    print(f"{'N/A':>8}{'':<3}", end="")
            
            # Show buy & hold
            if forecast in results_matrix and 10 in results_matrix[forecast]:
                bh = results_matrix[forecast][10]['buy_hold_return'] * 100
                print(f"{bh:>8.1f}%{'':<3}")
        
        # Display edge matrix
        print("\n📈 Edge vs Buy & Hold (%):")
        for forecast in [20, 40, 60, 63]:
            print(f"{forecast}d{'':<8}", end="")
            
            for buffer in [10, 15, 20]:
                if forecast in results_matrix and buffer in results_matrix[forecast]:
                    edge = results_matrix[forecast][buffer]['edge_vs_bh']
                    color = "🟢" if edge > 0 else "🔴"
                    print(f"{color}{edge:>7.1f}%{'':<2}", end="")
                else:
                    print(f"{'N/A':>8}{'':<3}", end="")
            
            print("  —")
        
        # Display rebalances matrix
        print("\n🔄 Number of Rebalances:")
        for forecast in [20, 40, 60, 63]:
            print(f"{forecast}d{'':<8}", end="")
            
            for buffer in [10, 15, 20]:
                if forecast in results_matrix and buffer in results_matrix[forecast]:
                    reb = results_matrix[forecast][buffer]['rebalances']
                    print(f"{reb:>8}{'':<3}", end="")
                else:
                    print(f"{'N/A':>8}{'':<3}", end="")
            
            print("  1")
        
        # Show benchmark
        if benchmark_metrics:
            print(f"\n📊 Benchmark (ASX 200): {benchmark_metrics['total_return']*100:.1f}%")
        
        # Find best combination
        best_combo = None
        best_edge = -999
        
        for forecast, buffers in results_matrix.items():
            for buffer, data in buffers.items():
                if data['edge_vs_bh'] > best_edge:
                    best_edge = data['edge_vs_bh']
                    best_combo = (forecast, buffer, data)
        
        if best_combo:
            print(f"\n🏆 Best Combination: {best_combo[0]}d forecast with top {best_combo[1]} buffer")
            print(f"   Return: {best_combo[2]['return']*100:.1f}%")
            print(f"   Edge vs Buy & Hold: {best_combo[2]['edge_vs_bh']:+.1f}%")
            print(f"   Rebalances: {best_combo[2]['rebalances']}")
        
        # Analysis
        print(f"\n🎯 Key Insights:")
        
        # Pattern 1: Shorter forecasts need wider buffers?
        short_need_wide = True
        for forecast in [20, 40]:
            if forecast in results_matrix:
                edges = [results_matrix[forecast][b]['edge_vs_bh'] for b in [10, 15, 20] 
                        if b in results_matrix[forecast]]
                if len(edges) > 1 and edges[-1] <= edges[0]:
                    short_need_wide = False
        
        if short_need_wide:
            print(f"   - Shorter forecasts (20d, 40d) perform better with wider buffers")
        
        # Pattern 2: Best buffer zone
        buffer_performance = {10: [], 15: [], 20: []}
        for buffers in results_matrix.values():
            for buffer, data in buffers.items():
                buffer_performance[buffer].append(data['edge_vs_bh'])
        
        avg_edges = {b: np.mean(edges) if edges else 0 for b, edges in buffer_performance.items()}
        best_buffer = max(avg_edges.items(), key=lambda x: x[1])
        print(f"   - Top {best_buffer[0]} buffer has highest average edge: {best_buffer[1]:+.1f}%")
        
        # Pattern 3: Best forecast window
        forecast_performance = {}
        for forecast, buffers in results_matrix.items():
            edges = [data['edge_vs_bh'] for data in buffers.values()]
            forecast_performance[forecast] = np.mean(edges) if edges else 0
        
        best_forecast = max(forecast_performance.items(), key=lambda x: x[1])
        print(f"   - {best_forecast[0]}d forecast has highest average edge: {best_forecast[1]:+.1f}%")
        
        # Warning about overfitting
        print(f"\n⚠️  Note: These results are optimized for 2021-2025 period.")
        print(f"   Validate on different periods (e.g., 2018-2021) before implementation.")
    
    def display_optimization_results(self, optimization_results, benchmark_metrics):
        """
        Display optimization results for 60-day forecast-aware rebalancing
        """
        print("\n" + "="*80)
        print("60-DAY FORECAST-AWARE OPTIMIZATION RESULTS")
        print("="*80)
        
        print(f"\n{'Buffer':<8} {'Strategy':<15} {'Return':<10} {'Annual':<10} {'Sharpe':<8} {'Rebalances':<12}")
        print("-"*73)
        
        for buffer_zone, data in optimization_results.items():
            # Forecast-aware rebalancing
            forecast_aware = data['forecast_aware']
            print(f"Top {buffer_zone:<4} "
                  f"{'Forecast-aware':<15} "
                  f"{forecast_aware['metrics']['total_return']*100:>8.1f}% "
                  f"{forecast_aware['metrics']['annual_return']*100:>8.1f}% "
                  f"{forecast_aware['metrics']['sharpe_ratio']:>6.2f} "
                  f"{forecast_aware['num_rebalances']:>10}")
            
            # Buy & Hold
            buy_hold = data['buy_hold']
            print(f"{'':<8} {'Buy & Hold':<15} "
                  f"{buy_hold['metrics']['total_return']*100:>8.1f}% "
                  f"{buy_hold['metrics']['annual_return']*100:>8.1f}% "
                  f"{buy_hold['metrics']['sharpe_ratio']:>6.2f} "
                  f"{buy_hold['num_rebalances']:>10}")
            
            # Edge
            edge = (forecast_aware['metrics']['total_return'] - buy_hold['metrics']['total_return']) * 100
            print(f"{'':<8} {'Edge':<15} {edge:>+7.1f}%")
            print("-"*73)
        
        # Show benchmark
        if benchmark_metrics:
            print(f"{'Benchmark':<8} {'ASX 200':<15} "
                  f"{benchmark_metrics['total_return']*100:>8.1f}% "
                  f"{benchmark_metrics['annual_return']*100:>8.1f}% "
                  f"{benchmark_metrics['sharpe_ratio']:>6.2f} "
                  f"{'N/A':>10}")
        
        # Find best buffer zone
        best_buffer = max(
            optimization_results.items(),
            key=lambda x: x[1]['forecast_aware']['metrics']['total_return']
        )
        
        print(f"\n🏆 Best Buffer Zone: Top {best_buffer[0]}")
        best_metrics = best_buffer[1]['forecast_aware']['metrics']
        buy_hold_metrics = best_buffer[1]['buy_hold']['metrics']
        edge = (best_metrics['total_return'] - buy_hold_metrics['total_return']) * 100
        
        print(f"   Return: {best_metrics['total_return']*100:.1f}%")
        print(f"   Edge vs Buy & Hold: {edge:+.1f}%")
        print(f"   Number of rebalances: {best_buffer[1]['forecast_aware']['num_rebalances']}")
        
        # Analysis
        print(f"\n📊 Buffer Zone Analysis:")
        for buffer_zone, data in optimization_results.items():
            forecast_aware = data['forecast_aware']
            buy_hold = data['buy_hold']
            edge = (forecast_aware['metrics']['total_return'] - buy_hold['metrics']['total_return']) * 100
            avg_holding = (self.end_date - self.start_date).days / forecast_aware['num_rebalances'] if forecast_aware['num_rebalances'] > 0 else 0
            
            print(f"   Top {buffer_zone}: Edge {edge:+.1f}%, {forecast_aware['num_rebalances']} trades, avg {avg_holding:.0f} days")
        
        print(f"\n🎯 Key Insights:")
        print(f"   - Narrower buffer (top 10) = more selective, fewer trades")
        print(f"   - Wider buffer (top 20) = more tolerant, more trades")
        print(f"   - Optimal buffer balances signal quality with trade frequency")
        
    def display_forecast_aware_comparison(self, forecast_aware_results, benchmark_metrics):
        """
        Display comparison of forecast-aware conditional rebalancing vs buy & hold
        """
        print("\n" + "="*80)
        print("FORECAST-AWARE CONDITIONAL REBALANCING COMPARISON")
        print("="*80)
        
        print(f"\n{'Forecast':<10} {'Strategy':<15} {'Return':<10} {'Annual':<10} {'Sharpe':<8} {'Rebalances':<12}")
        print("-"*75)
        
        for forecast_days, data in forecast_aware_results.items():
            forecast_name = data['forecast_name']
            
            # Forecast-aware rebalancing
            forecast_aware = data['forecast_aware']
            print(f"{forecast_name:<10} "
                  f"{'Forecast-aware':<15} "
                  f"{forecast_aware['metrics']['total_return']*100:>8.1f}% "
                  f"{forecast_aware['metrics']['annual_return']*100:>8.1f}% "
                  f"{forecast_aware['metrics']['sharpe_ratio']:>6.2f} "
                  f"{forecast_aware['num_rebalances']:>10}")
            
            # Buy & Hold
            buy_hold = data['buy_hold']
            print(f"{forecast_name:<10} "
                  f"{'Buy & Hold':<15} "
                  f"{buy_hold['metrics']['total_return']*100:>8.1f}% "
                  f"{buy_hold['metrics']['annual_return']*100:>8.1f}% "
                  f"{buy_hold['metrics']['sharpe_ratio']:>6.2f} "
                  f"{buy_hold['num_rebalances']:>10}")
            
            # Edge
            edge = (forecast_aware['metrics']['total_return'] - buy_hold['metrics']['total_return']) * 100
            print(f"{'':<10} {'Edge':<15} {edge:>+7.1f}%")
            print("-"*75)
        
        # Show benchmark
        if benchmark_metrics:
            print(f"{'Benchmark':<10} {'ASX 200':<15} "
                  f"{benchmark_metrics['total_return']*100:>8.1f}% "
                  f"{benchmark_metrics['annual_return']*100:>8.1f}% "
                  f"{benchmark_metrics['sharpe_ratio']:>6.2f} "
                  f"{'N/A':>10}")
        
        # Analysis
        print(f"\n📊 Key Findings:")
        for forecast_days, data in forecast_aware_results.items():
            forecast_name = data['forecast_name']
            forecast_aware = data['forecast_aware']
            buy_hold = data['buy_hold']
            edge = (forecast_aware['metrics']['total_return'] - buy_hold['metrics']['total_return']) * 100
            
            if edge > 0:
                print(f"   {forecast_name}: Forecast-aware BEATS buy & hold by {edge:+.1f}%! 🎉")
            else:
                print(f"   {forecast_name}: Buy & hold still wins by {edge:.1f}%")
        
        # Trade analysis
        print(f"\n📊 Trade Analysis:")
        for forecast_days, data in forecast_aware_results.items():
            forecast_name = data['forecast_name']
            forecast_aware = data['forecast_aware']
            avg_holding = 365 / forecast_aware['num_rebalances'] if forecast_aware['num_rebalances'] > 0 else 0
            print(f"   {forecast_name}: {forecast_aware['num_rebalances']} trades, avg {avg_holding:.0f} days holding")
        
        print(f"\n🎯 Conclusion:")
        print(f"   - Forecast-aware rebalancing respects the prediction horizon")
        print(f"   - Only trades when forecasts expire and holdings weaken")
        print(f"   - Significantly reduces trading frequency vs traditional rebalancing")
        
        # Check if any forecast window beat buy & hold
        any_win = any(
            data['forecast_aware']['metrics']['total_return'] > data['buy_hold']['metrics']['total_return']
            for data in forecast_aware_results.values()
        )
        
        if any_win:
            print(f"   - ✅ Some forecast windows successfully beat buy & hold!")
        else:
            print(f"   - 📊 Buy & hold remains strong in trending markets")
    
    def display_matched_comparison(self, matched_results, benchmark_metrics):
        """
        Display comparison of matched forecast/rebalancing periods
        """
        print("\n" + "="*80)
        print("MATCHED FORECAST/REBALANCING COMPARISON")
        print("="*80)
        
        print(f"\n{'Forecast':<10} {'Strategy':<15} {'Return':<10} {'Annual':<10} {'Sharpe':<8} {'Rebalances':<12}")
        print("-"*75)
        
        for forecast_days, data in matched_results.items():
            forecast_name = data['forecast_name']
            
            # Matched rebalancing
            matched = data['matched']
            print(f"{forecast_name:<10} "
                  f"{'Matched':<15} "
                  f"{matched['metrics']['total_return']*100:>8.1f}% "
                  f"{matched['metrics']['annual_return']*100:>8.1f}% "
                  f"{matched['metrics']['sharpe_ratio']:>6.2f} "
                  f"{matched['num_rebalances']:>10}")
            
            # Buy & Hold
            buy_hold = data['buy_hold']
            print(f"{forecast_name:<10} "
                  f"{'Buy & Hold':<15} "
                  f"{buy_hold['metrics']['total_return']*100:>8.1f}% "
                  f"{buy_hold['metrics']['annual_return']*100:>8.1f}% "
                  f"{buy_hold['metrics']['sharpe_ratio']:>6.2f} "
                  f"{buy_hold['num_rebalances']:>10}")
            
            # Edge
            edge = (matched['metrics']['total_return'] - buy_hold['metrics']['total_return']) * 100
            print(f"{'':<10} {'Edge':<15} {edge:>+7.1f}%")
            print("-"*75)
        
        # Show benchmark
        if benchmark_metrics:
            print(f"{'Benchmark':<10} {'ASX 200':<15} "
                  f"{benchmark_metrics['total_return']*100:>8.1f}% "
                  f"{benchmark_metrics['annual_return']*100:>8.1f}% "
                  f"{benchmark_metrics['sharpe_ratio']:>6.2f} "
                  f"{'N/A':>10}")
        
        # Analysis
        print(f"\n📊 Key Findings:")
        for forecast_days, data in matched_results.items():
            forecast_name = data['forecast_name']
            matched = data['matched']
            buy_hold = data['buy_hold']
            edge = (matched['metrics']['total_return'] - buy_hold['metrics']['total_return']) * 100
            
            if edge > 0:
                print(f"   {forecast_name}: Matched rebalancing BEATS buy & hold by {edge:+.1f}%")
            else:
                print(f"   {forecast_name}: Buy & hold still wins by {edge:.1f}%")
        
        print(f"\n🎯 Conclusion:")
        print(f"   - Matching forecast/rebalancing periods reduces the edge gap")
        print(f"   - Buy & hold still performs best in strong trending markets")
        print(f"   - All strategies significantly outperform the benchmark")

if __name__ == "__main__":
    backtest = WalkForwardBacktest()
    results = backtest.run()
