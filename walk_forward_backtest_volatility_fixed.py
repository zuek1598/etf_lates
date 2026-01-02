"""
Walk Forward Backtest with CORRECTED Volatility-Based Confidence Intervals
==========================================================================

Fixed version that correctly distinguishes market-wide stress from ETF-specific issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Add paths
sys.path.append('/Users/peter/Desktop/etf_lates')
sys.path.append('/Users/peter/Desktop/etf_lates/analyzers')

from walk_forward_backtest import WalkForwardBacktest
from analyzers.volatility_ci_corrected import CorrectedVolatilityCI

class WalkForwardBacktestVolatilityFixed(WalkForwardBacktest):
    """Walk-forward backtest with corrected volatility-based CI"""
    
    def __init__(self, start_date, end_date, enable_ci=True):
        super().__init__(start_date, end_date)
        self.enable_ci = enable_ci
        
        if enable_ci:
            self.vci = CorrectedVolatilityCI()
    
    def simulate_volatility_aware_rebalancing(self, all_data, forecast_window=40, buffer_zone=20):
        """
        Simulate rebalancing with corrected volatility-based confidence intervals
        """
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD BACKTEST WITH CORRECTED VOLATILITY CI")
        print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        print(f"\nSimulating Corrected Volatility-Aware Strategy")
        print(f"  Forecast window: {forecast_window} days")
        print(f"  Buffer zone: Top {buffer_zone}")
        print(f"  Volatility CI: {'Enabled' if self.enable_ci else 'Disabled'}")
        print(f"  Logic: Distinguish market stress from ETF-specific issues")
        print("-"*60)
        
        # Get trading days
        trading_days = []
        for prices in all_data.values():
            trading_days.extend(prices.index.tolist())
        unique_dates = sorted(set(trading_days))
        
        # Initialize portfolio
        portfolio_value = pd.Series(index=unique_dates)
        portfolio_value[:] = np.nan
        portfolio_value.iloc[0] = 1.0
        
        trade_log = []
        current_holdings = []
        last_rebalance_date = self.start_date
        
        # Initial selection
        print(f"\n  {self.start_date.strftime('%Y-%m-%d')}: Initial selection")
        top_etfs = self.get_top_etfs_at_date(all_data, self.start_date, top_n=buffer_zone, forecast_window=forecast_window)
        
        if top_etfs is not None and len(top_etfs) > 0:
            if isinstance(top_etfs, list):
                current_holdings = [etf['etf'] if isinstance(etf, dict) else etf for etf in top_etfs[:3]]
            else:
                current_holdings = list(top_etfs['etf'].head(3))
        
        trade_log.append({
            'date': self.start_date,
            'action': 'initial',
            'holdings': current_holdings.copy(),
            'reason': 'Initial selection'
        })
        
        print(f"    Initial holdings: {current_holdings}")
        
        # Schedule rebalancing checks
        next_check_date = self.start_date + timedelta(days=forecast_window)
        
        while next_check_date < self.end_date:
            print(f"\n  {next_check_date.strftime('%Y-%m-%d')}: Rebalancing check")
            
            # Get current rankings
            top_etfs = self.get_top_etfs_at_date(all_data, next_check_date, top_n=buffer_zone, forecast_window=forecast_window)
            
            if top_etfs is None or len(top_etfs) == 0:
                next_check_date += timedelta(days=forecast_window)
                continue
            
            # Get rankings as list
            if isinstance(top_etfs, list):
                ranked_etfs = [etf['etf'] if isinstance(etf, dict) else etf for etf in top_etfs]
            else:
                ranked_etfs = list(top_etfs['etf'])
            
            # Check each holding if volatility CI is enabled
            if self.enable_ci:
                new_holdings = []
                rotation_decisions = []
                
                # For each current holding, decide whether to keep or rotate
                for etf in current_holdings:
                    rank = ranked_etfs.index(etf) + 1 if etf in ranked_etfs else 999
                    
                    # Get price data up to check date
                    if etf in all_data:
                        prices = all_data[etf]
                        historical_prices = prices[prices.index <= next_check_date]
                        
                        if len(historical_prices) >= 50:
                            # Use corrected volatility CI to decide
                            should_rotate, decision = self.vci.should_rotate(
                                etf, historical_prices, all_data, rank
                            )
                            rotation_decisions.append(decision)
                            
                            if should_rotate:
                                # Find replacement from top ranked ETFs
                                for candidate in ranked_etfs[:5]:
                                    if candidate not in new_holdings and candidate != etf:
                                        new_holdings.append(candidate)
                                        break
                            else:
                                # Keep the ETF
                                new_holdings.append(etf)
                        else:
                            # Not enough history - use ranking
                            if etf in ranked_etfs[:3]:
                                new_holdings.append(etf)
                            else:
                                # Replace with top ranked
                                for candidate in ranked_etfs[:3]:
                                    if candidate not in new_holdings:
                                        new_holdings.append(candidate)
                                        break
                    else:
                        # No data - replace
                        for candidate in ranked_etfs[:3]:
                            if candidate not in new_holdings:
                                new_holdings.append(candidate)
                                break
                
                # Ensure we have exactly 3 holdings
                while len(new_holdings) < 3 and len(ranked_etfs) > 0:
                    for candidate in ranked_etfs:
                        if candidate not in new_holdings:
                            new_holdings.append(candidate)
                            break
                
                # Log decisions
                for decision in rotation_decisions:
                    print(f"    {decision['etf']}: {decision['decision']} - {decision['reasoning']}")
                
                # Update holdings if changed
                if set(new_holdings) != set(current_holdings):
                    trade_log.append({
                        'date': next_check_date,
                        'action': 'rebalance',
                        'holdings': new_holdings.copy(),
                        'reason': 'Portfolio rebalanced based on corrected volatility analysis'
                    })
                    current_holdings = new_holdings
                    print(f"    New holdings: {current_holdings}")
                else:
                    trade_log.append({
                        'date': next_check_date,
                        'action': 'hold',
                        'holdings': current_holdings.copy(),
                        'reason': 'No changes needed'
                    })
                    print(f"    No changes needed")
            else:
                # Standard rebalancing without CI
                new_holdings = []
                
                for etf in current_holdings:
                    if etf in ranked_etfs[:buffer_zone]:
                        new_holdings.append(etf)
                
                # Fill missing slots
                for etf in ranked_etfs[:3]:
                    if len(new_holdings) >= 3:
                        break
                    if etf not in new_holdings:
                        new_holdings.append(etf)
                
                if set(new_holdings) != set(current_holdings):
                    trade_log.append({
                        'date': next_check_date,
                        'action': 'rebalance',
                        'holdings': new_holdings.copy(),
                        'reason': 'Portfolio rebalanced'
                    })
                    current_holdings = new_holdings
                    print(f"    New holdings: {current_holdings}")
                else:
                    print(f"    No changes needed")
            
            # Move to next check date
            next_check_date += timedelta(days=forecast_window)
        
        # Calculate portfolio returns
        print(f"\nCalculating portfolio returns...")
        current_value = 1.0
        current_position_size = 1.0 / 3.0  # Equal weight
        
        for i, date in enumerate(unique_dates[1:], 1):
            if date < self.start_date or date > self.end_date:
                continue
            
            # Get previous date's value
            prev_value = portfolio_value.iloc[i-1]
            if np.isnan(prev_value):
                prev_value = 1.0
            
            # Calculate daily return based on holdings
            daily_return = 0
            valid_holdings = 0
            
            for etf in current_holdings:
                if etf in all_data:
                    if date in all_data[etf].index:
                        prev_date = unique_dates[i-1]
                        if prev_date in all_data[etf].index:
                            etf_return = (all_data[etf].loc[date] - all_data[etf].loc[prev_date]) / all_data[etf].loc[prev_date]
                            daily_return += etf_return
                            valid_holdings += 1
            
            if valid_holdings > 0:
                daily_return = daily_return / valid_holdings
            
            portfolio_value.iloc[i] = prev_value * (1 + daily_return)
        
        # Create results dictionary
        results = {
            'returns': portfolio_value.pct_change().fillna(0),
            'cumulative': portfolio_value,
            'value': portfolio_value * 100000,
            'trade_log': trade_log
        }
        
        return results
