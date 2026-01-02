"""
Walk-Forward Backtest with Confidence Intervals
===============================================

Enhanced version of the backtest engine that includes confidence interval
noise detection to prevent unnecessary rotations during market stress.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Add paths
sys.path.append('/Users/peter/Desktop/etf_lates')
sys.path.append('/Users/peter/Desktop/etf_lates/analyzers')

from walk_forward_backtest import WalkForwardBacktest
from confidence_intervals import ConfidenceIntervalCalculator, NoiseDetectionLogic
from quality_ranker import QualityRanker
from metric_calculation import calculate_all_metrics
import yfinance as yf

class WalkForwardBacktestCI(WalkForwardBacktest):
    """Enhanced backtest with confidence interval noise detection"""
    
    def __init__(self, start_date, end_date, enable_ci=True):
        super().__init__(start_date, end_date)
        self.enable_ci = enable_ci
        
        # Initialize CI components
        if enable_ci:
            self.ci_calc = ConfidenceIntervalCalculator(window_size=100, confidence_level=0.95)
            self.noise_logic = NoiseDetectionLogic(self.ci_calc)
            self.current_metrics = {}  # Store current metrics for all ETFs
    
    def build_confidence_database(self, all_data):
        """
        Build confidence interval database before backtest
        """
        if not self.enable_ci:
            return
        
        print("\nBuilding confidence interval database...")
        self.ci_calc.build_confidence_database(all_data, self.start_date, self.end_date)
        self.ci_calc.save_confidence_intervals()
    
    def get_etf_metrics(self, etf_ticker, prices, date):
        """
        Get current metrics for an ETF using standardized calculation methods
        """
        try:
            # Use the standardized calculation functions for consistency
            metrics = calculate_all_metrics(prices, as_of_date=date)
            
            if metrics:
                return metrics
        except Exception as e:
            print(f"DEBUG {etf_ticker}: Error - {e}")
        
        return None
    
    def simulate_ci_aware_rebalancing(self, all_data, forecast_window=40, buffer_zone=20):
        """
        Simulate strategy with confidence interval noise detection
        """
        print(f"\nSimulating CI-Enhanced Strategy")
        print(f"  Forecast window: {forecast_window} days")
        print(f"  Buffer zone: Top {buffer_zone}")
        print(f"  Confidence intervals: {'Enabled' if self.enable_ci else 'Disabled'}")
        print("-" * 60)
        
        # Build confidence database if needed
        if self.enable_ci:
            self.ci_calc.load_confidence_intervals()
        
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
        print(f"  {self.start_date.strftime('%Y-%m-%d')}: Initial selection")
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
            
            # Calculate current metrics for CI checking
            if self.enable_ci:
                print(f"DEBUG: Populating current_metrics for {len(ranked_etfs[:20])} ETFs")
                self.current_metrics = {}
                for etf in ranked_etfs[:20]:  # Only need top 20
                    if etf in all_data:
                        prices = all_data[etf]
                        historical_prices = prices[prices.index <= next_check_date]
                        if len(historical_prices) >= 30:  # Reduced from 100 to 30
                            print(f"DEBUG: Calculating metrics for {etf} (history: {len(historical_prices)})")
                            metrics = self.get_etf_metrics(etf, historical_prices, next_check_date)
                            if metrics:
                                self.current_metrics[etf] = metrics
                                print(f"DEBUG: Successfully calculated metrics for {etf}")
                            else:
                                print(f"DEBUG: Failed to calculate metrics for {etf}")
                print(f"DEBUG: current_metrics populated for {len(self.current_metrics)} ETFs")
            
            # Check market regime first
            market_regime = 'NORMAL'
            if self.enable_ci and len(self.current_metrics) > 0:
                regime_decision = self.noise_logic.check_market_regime(
                    ranked_etfs[:20], self.current_metrics
                )
                market_regime = regime_decision['regime']
                
                if market_regime == 'STRESS':
                    print(f"    Market regime: STRESS ({regime_decision['signal_percentage']:.1%} showing signals)")
                    print(f"    Action: HOLD all positions (market-wide stress)")
                    trade_log.append({
                        'date': next_check_date,
                        'action': 'hold',
                        'holdings': current_holdings.copy(),
                        'reason': f'Market stress regime ({regime_decision["signal_percentage"]:.1%} signals)'
                    })
                else:
                    print(f"    Market regime: NORMAL")
            
            # Check each holding only if market is normal
            if market_regime == 'NORMAL':
                new_holdings = []
                rotation_decisions = []
                
                # Get the top 3 candidates
                top_3_candidates = ranked_etfs[:3]
                
                # Default to top 3 if no CI data
                if not self.enable_ci or len(self.current_metrics) == 0:
                    new_holdings = top_3_candidates.copy()
                else:
                    # Compare current holdings with proposed candidates
                    for etf in current_holdings:
                        # If current holding is in top 3, keep it
                        if etf in top_3_candidates:
                            new_holdings.append(etf)
                            rotation_decisions.append({
                                'etf': etf,
                                'decision': 'HOLD',
                                'reason': 'In top 3',
                                'signal_type': 'HOLD'
                            })
                        else:
                            # Current holding dropped out - find replacement
                            best_candidate = None
                            best_signal = None
                            
                            for candidate in top_3_candidates:
                                if candidate not in new_holdings:
                                    # Check if candidate shows strong signal
                                    if candidate in self.current_metrics:
                                        _, candidate_decision = self.noise_logic.should_rotate(
                                            candidate, self.current_metrics[candidate], 
                                            ranked_etfs.index(candidate) + 1, buffer_zone, market_regime
                                        )
                                        
                                        # Check if current holding shows weak signal (noise)
                                        if etf in self.current_metrics:
                                            _, current_decision = self.noise_logic.should_rotate(
                                                etf, self.current_metrics[etf], 
                                                ranked_etfs.index(etf) + 1, buffer_zone, market_regime
                                            )
                                            current_signal = current_decision['signal_type']
                                        else:
                                            # No CI data for current holding - assume weak signal
                                            current_signal = 'WAIT'
                                        
                                        # Only rotate if candidate is strong AND current is weak
                                        if (candidate_decision['signal_type'] == 'ROTATE' and 
                                            current_signal in ['HOLD', 'WAIT']):
                                            best_candidate = candidate
                                            best_signal = candidate_decision
                                            break
                            
                            if best_candidate:
                                new_holdings.append(best_candidate)
                                rotation_decisions.append({
                                    'etf': etf,
                                    'decision': 'ROTATE',
                                    'reason': f'Replaced by {best_candidate}: {best_signal["reason"]}',
                                    'signal_type': 'ROTATE'
                                })
                            else:
                                # No strong candidate - keep current
                                new_holdings.append(etf)
                                rotation_decisions.append({
                                    'etf': etf,
                                    'decision': 'HOLD',
                                    'reason': 'No strong replacement candidates',
                                    'signal_type': 'HOLD'
                                })
                    
                    # Fill remaining slots with top candidates
                    while len(new_holdings) < 3:
                        for candidate in top_3_candidates:
                            if candidate not in new_holdings:
                                new_holdings.append(candidate)
                                break
                
                # Log decisions
                for decision in rotation_decisions:
                    print(f"    {decision['etf']}: {decision['decision']} - {decision['reason']}")
                
                # Update holdings if changed
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
                    trade_log.append({
                        'date': next_check_date,
                        'action': 'hold',
                        'holdings': current_holdings.copy(),
                        'reason': 'No changes needed'
                    })
            
            # Schedule next check
            next_check_date += timedelta(days=forecast_window)
        
        # Calculate portfolio returns
        print("\nCalculating portfolio returns...")
        
        # Build holdings schedule
        holdings_schedule = []
        for trade in trade_log:
            if 'holdings' in trade:
                holdings_schedule.append({
                    'date': trade['date'],
                    'holdings': trade['holdings']
                })
        
        # Calculate daily returns
        current_value = 1.0
        for i in range(len(unique_dates) - 1):
            date = unique_dates[i]
            next_date = unique_dates[i + 1]
            
            # Find current holdings
            current_etfs = None
            for change in reversed(holdings_schedule):
                if change['date'] <= date:
                    current_etfs = change['holdings']
                    break
            
            if not current_etfs:
                portfolio_value.loc[next_date] = portfolio_value.loc[date]
                continue
            
            # Calculate returns
            daily_returns = []
            for etf in current_etfs:
                if etf in all_data:
                    prices = all_data[etf]
                    if date in prices.index and next_date in prices.index:
                        daily_ret = (prices.loc[next_date] / prices.loc[date]) - 1
                        daily_returns.append(daily_ret)
            
            if daily_returns:
                avg_ret = np.mean(daily_returns)
                current_value = portfolio_value.loc[date] * (1 + avg_ret)
                portfolio_value.loc[next_date] = current_value
            else:
                portfolio_value.loc[next_date] = portfolio_value.loc[date]
        
        # Calculate returns
        portfolio_value = portfolio_value.ffill()
        portfolio_returns = portfolio_value.pct_change().dropna()
        cumulative_returns = portfolio_value.copy()
        
        # Add CI statistics to results
        ci_stats = {}
        if self.enable_ci:
            ci_stats = {
                'total_decisions': len(self.noise_logic.decision_log),
                'rotations': sum(1 for d in self.noise_logic.decision_log if d['decision'] == 'ROTATE'),
                'noise_prevented': sum(1 for d in self.noise_logic.decision_log 
                                     if d['decision'] == 'HOLD' and 'noise' in d['reason'].lower())
            }
        
        return {
            'returns': portfolio_returns,
            'cumulative': cumulative_returns,
            'value': portfolio_value,
            'trade_log': trade_log,
            'ci_stats': ci_stats
        }

def test_ci_enhanced_strategy():
    """Test the CI-enhanced strategy"""
    print("="*60)
    print("CI-ENHANCED STRATEGY TEST")
    print("="*60)
    
    # Initialize backtest
    backtest = WalkForwardBacktestCI(
        start_date=pd.Timestamp('2021-01-01'),
        end_date=pd.Timestamp('2025-12-31'),
        enable_ci=True
    )
    
    # Download data
    print("Downloading data...")
    all_data = backtest.download_all_etf_data()
    
    # Build confidence database
    backtest.build_confidence_database(all_data)
    
    # Run CI-enhanced strategy
    result = backtest.simulate_ci_aware_rebalancing(
        all_data=all_data,
        forecast_window=40,
        buffer_zone=20
    )
    
    if result:
        # Calculate metrics
        final_value = result['cumulative'].iloc[-1]
        total_return = (final_value - 1) * 100
        returns = result['returns']
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        print("\n" + "="*60)
        print("CI-ENHANCED STRATEGY RESULTS")
        print("="*60)
        print(f"  Total return: {total_return:.1f}%")
        print(f"  Sharpe ratio: {sharpe:.2f}")
        print(f"  Number of trades: {len(result['trade_log'])}")
        
        if result['ci_stats']:
            print(f"\nCI Statistics:")
            print(f"  Total decisions: {result['ci_stats']['total_decisions']}")
            print(f"  Rotations: {result['ci_stats']['rotations']}")
            print(f"  Noise prevented: {result['ci_stats']['noise_prevented']}")
        
        # Yearly breakdown
        print("\nYear-by-Year Performance:")
        print("-" * 40)
        for year in range(2021, 2026):
            year_start = pd.Timestamp(f'{year}-01-01')
            year_end = pd.Timestamp(f'{year}-12-31')
            
            year_value = result['cumulative'].loc[year_start:year_end]
            if len(year_value) > 0:
                year_return = (year_value.iloc[-1] / year_value.iloc[0] - 1) * 100
                print(f"  {year}: {year_return:.1f}%")
        
        return result
    
    return None

if __name__ == "__main__":
    test_ci_enhanced_strategy()
