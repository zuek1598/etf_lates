"""
Confidence Interval System for Noise Detection
==============================================

This module implements statistical confidence intervals to distinguish
between normal metric fluctuations (noise) and genuine regime changes (signals).

Key Features:
- 95% confidence intervals for hit_rate, conviction, and stability
- 100-day rolling window calculations
- Market-wide signal detection
- Noise vs signal classification
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime, timedelta

# Add analyzers path
sys.path.append('/Users/peter/Desktop/etf_lates/analyzers')
from quality_ranker import QualityRanker
from metric_calculation import (
    calculate_all_metrics,
    calculate_distance_weight,
    make_rotation_decision
)

class ConfidenceIntervalCalculator:
    """Calculate and manage confidence intervals for ETF metrics"""
    
    def __init__(self, window_size=100, confidence_level=0.95):
        self.window_size = window_size  # 100 trading days
        self.confidence_level = confidence_level
        self.z_score = 1.96  # For 95% confidence
        
        # Storage for historical metrics
        self.metric_history = {}  # {etf: {date: {hit_rate, conviction, stability}}}
        self.confidence_intervals = {}  # {etf: {metric: {mean, std, lower, upper}}}
        
        # Initialize QualityRanker for metric calculations
        self.qr = QualityRanker()
    
    def calculate_metrics_for_etf(self, etf_ticker, prices, as_of_date):
        """
        Calculate hit_rate, conviction, and stability for an ETF
        Using standardized metric calculation functions
        """
        try:
            # Use the standardized calculation functions
            metrics = calculate_all_metrics(prices, as_of_date=as_of_date)
            
            if metrics:
                return metrics
        except Exception as e:
            print(f"DEBUG {etf_ticker}: Error - {e}")
        
        return None
    
    def update_metric_history(self, etf_ticker, date, metrics):
        """
        Update the rolling history for an ETF
        """
        if etf_ticker not in self.metric_history:
            self.metric_history[etf_ticker] = {}
        
        self.metric_history[etf_ticker][date] = metrics
        
        # Keep only last window_size days
        if len(self.metric_history[etf_ticker]) > self.window_size:
            # Sort by date and keep most recent
            sorted_dates = sorted(self.metric_history[etf_ticker].keys())
            to_remove = sorted_dates[:-self.window_size]
            for date_to_remove in to_remove:
                del self.metric_history[etf_ticker][date_to_remove]
    
    def calculate_confidence_intervals(self, etf_ticker):
        """
        Calculate confidence intervals for all metrics of an ETF using percentiles
        """
        if etf_ticker not in self.metric_history:
            print(f"DEBUG: No history for {etf_ticker}")
            return None
        
        history = self.metric_history[etf_ticker]
        
        # Extract metric values
        hit_rates = [m['hit_rate'] for m in history.values()]
        convictions = [m['conviction'] for m in history.values()]
        stabilities = [m['stability'] for m in history.values()]
        
        # Apply outlier capping to conviction (winsorization at 1st/99th percentile)
        if convictions:
            p1 = np.percentile(convictions, 1)
            p99 = np.percentile(convictions, 99)
            convictions = np.clip(convictions, p1, p99).tolist()
        
        # Calculate confidence intervals using percentiles (more robust)
        ci_data = {}
        
        for metric_name, values in [
            ('hit_rate', hit_rates),
            ('conviction', convictions),
            ('stability', stabilities)
        ]:
            # Use 2.5th and 97.5th percentiles for 95% CI
            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            ci_data[metric_name] = {
                'lower': lower,
                'upper': upper,
                'mean': mean_val,
                'std': std_val,
                'median': np.median(values)
            }
        
        # Store in confidence intervals
        self.confidence_intervals[etf_ticker] = ci_data
        
        return ci_data
    
    def check_metric_outside_ci(self, etf_ticker, current_metrics):
        """
        Check how many metrics are outside their confidence intervals using weighted distance
        """
        if etf_ticker not in self.confidence_intervals:
            return 0, {}, 'NO_CI'
        
        ci = self.confidence_intervals[etf_ticker]
        total_weight = 0.0
        outside_details = {}
        
        for metric in ['hit_rate', 'conviction', 'stability']:
            current_value = current_metrics.get(metric, 0)
            
            if metric in ci:
                lower = ci[metric]['lower']
                upper = ci[metric]['upper']
                
                # Calculate weighted distance
                weight, distance_pct = calculate_distance_weight(current_value, lower, upper)
                
                outside_details[metric] = {
                    'value': current_value,
                    'ci_range': [lower, upper],
                    'outside': weight > 0,
                    'weight': weight,
                    'distance_pct': distance_pct
                }
                
                total_weight += weight
        
        # Make decision based on total weight
        signal_type = make_rotation_decision(total_weight)
        
        # Convert to numeric for backward compatibility
        if signal_type == 'HOLD':
            outside_count = 0
        elif signal_type == 'WAIT':
            outside_count = 1
        else:  # ROTATE
            outside_count = 2
        
        return outside_count, outside_details, signal_type
    
    def detect_market_wide_signal(self, etf_list, current_metrics_dict):
        """
        Detect if a market-wide regime change is occurring using weighted signals
        """
        total_etfs = len(etf_list)
        strong_signals = 0
        
        for etf in etf_list:
            if etf in current_metrics_dict:
                count, _, signal_type = self.check_metric_outside_ci(etf, current_metrics_dict[etf])
                if signal_type == 'ROTATE':  # Strong signal based on weighted distance
                    strong_signals += 1
        
        # If >30% of ETFs show strong signals, it's market-wide (lowered from 60%)
        signal_percentage = strong_signals / total_etfs if total_etfs > 0 else 0
        
        if signal_percentage > 0.3:
            return True, signal_percentage
        
        return False, signal_percentage
    
    def build_confidence_database(self, all_data, start_date, end_date):
        """
        Build the complete confidence interval database
        """
        print("Building confidence interval database...")
        print(f"  Window size: {self.window_size} days")
        print(f"  Confidence level: {self.confidence_level * 100}%")
        
        # Get all trading days
        trading_days = []
        for prices in all_data.values():
            trading_days.extend(prices.index.tolist())
        unique_dates = sorted(set(trading_days))
        
        # Find start date for CI calculation (need 100 days before backtest)
        ci_start = start_date - timedelta(days=self.window_size * 2)
        
        # Filter dates
        dates_to_process = [d for d in unique_dates if ci_start <= d <= end_date]
        
        print(f"  Processing {len(dates_to_process)} trading days...")
        
        # Process each day
        for i, date in enumerate(dates_to_process):
            if i % 100 == 0:
                print(f"    Progress: {i}/{len(dates_to_process)} ({i/len(dates_to_process)*100:.1f}%)")
            
            # Calculate metrics for all ETFs
            for etf_ticker, prices in all_data.items():
                try:
                    # Get historical prices up to this date
                    historical_prices = prices[prices.index <= date]
                    
                    if len(historical_prices) >= 100:  # Need enough history
                        metrics = self.calculate_metrics_for_etf(etf_ticker, historical_prices, date)
                        
                        if metrics:
                            self.update_metric_history(etf_ticker, date, metrics)
                            
                            # Calculate CIs for this ETF
                            ci_data = self.calculate_confidence_intervals(etf_ticker)
                            if ci_data:
                                self.confidence_intervals[etf_ticker] = ci_data
                
                except:
                    continue
        
        print(f"  Built confidence intervals for {len(self.confidence_intervals)} ETFs")
        return self.confidence_intervals
    
    def save_confidence_intervals(self, filename='confidence_intervals.pkl'):
        """
        Save confidence intervals to file
        """
        filepath = os.path.join('/Users/peter/Desktop/etf_lates/data', filename)
        
        save_data = {
            'confidence_intervals': self.confidence_intervals,
            'window_size': self.window_size,
            'confidence_level': self.confidence_level,
            'created_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved confidence intervals to {filepath}")
    
    def load_confidence_intervals(self, filename='confidence_intervals.pkl'):
        """
        Load confidence intervals from file
        """
        filepath = os.path.join('/Users/peter/Desktop/etf_lates/data', filename)
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.confidence_intervals = save_data['confidence_intervals']
            self.window_size = save_data['window_size']
            self.confidence_level = save_data['confidence_level']
            
            print(f"Loaded confidence intervals for {len(self.confidence_intervals)} ETFs")
            return True
        
        print("No saved confidence intervals found")
        return False

class NoiseDetectionLogic:
    """Logic to decide whether to rotate based on confidence intervals"""
    
    def __init__(self, confidence_calculator):
        self.ci_calc = confidence_calculator
        self.decision_log = []  # Track all decisions
    
    def should_rotate(self, etf_ticker, current_metrics, current_rank, buffer_zone=20, market_regime='NORMAL'):
        """
        Decide whether to rotate an ETF based on noise detection and market regime
        """
        decision = {
            'etf': etf_ticker,
            'date': datetime.now(),
            'current_rank': current_rank,
            'in_buffer': current_rank <= buffer_zone,
            'decision': 'HOLD',
            'reason': '',
            'total_weight': 0,
            'signal_type': 'HOLD'
        }
        
        # If market is in stress regime, always hold
        if market_regime == 'STRESS':
            decision['decision'] = 'HOLD'
            decision['reason'] = 'Market stress regime - hold all positions'
            self.decision_log.append(decision)
            return False, decision
        
        # Check if metrics are outside confidence intervals
        outside_count, outside_details, signal_type = self.ci_calc.check_metric_outside_ci(
            etf_ticker, current_metrics
        )
        
        decision['total_weight'] = sum(d.get('weight', 0) for d in outside_details.values())
        decision['signal_type'] = signal_type
        
        # DEBUG: Log what's happening
        print(f"DEBUG CI: {etf_ticker} rank={current_rank}, signal={signal_type}, weight={decision['total_weight']:.1f}")
        
        # Apply decision rules (only in normal market)
        if signal_type == 'HOLD':
            decision['decision'] = 'HOLD'
            decision['reason'] = 'Pure noise - metrics within normal variation'
        elif signal_type == 'WAIT':
            decision['decision'] = 'HOLD'
            decision['reason'] = 'Minor signal - wait for confirmation'
        elif signal_type == 'ROTATE':
            decision['decision'] = 'ROTATE'
            decision['reason'] = 'Strong signal - metrics outside normal range'
        else:
            decision['decision'] = 'ROTATE'  # Default to original behavior if no CI data
            decision['reason'] = 'No confidence intervals - using original logic'
        
        self.decision_log.append(decision)
        
        # Return True if should rotate
        return decision['decision'] == 'ROTATE', decision
    
    def check_market_regime(self, etf_list, current_metrics_dict):
        """
        Check if we're in a market-wide regime change
        """
        is_market_signal, signal_percentage = self.ci_calc.detect_market_wide_signal(
            etf_list, current_metrics_dict
        )
        
        regime_decision = {
            'is_market_signal': is_market_signal,
            'signal_percentage': signal_percentage,
            'regime': 'STRESS' if is_market_signal else 'NORMAL'
        }
        
        return regime_decision

def test_confidence_intervals():
    """Test the confidence interval system"""
    print("="*60)
    print("CONFIDENCE INTERVAL SYSTEM TEST")
    print("="*60)
    
    # Initialize calculator
    ci_calc = ConfidenceIntervalCalculator(window_size=100, confidence_level=0.95)
    
    # Try to load existing data
    if not ci_calc.load_confidence_intervals():
        print("No existing confidence intervals found.")
        print("Run build_confidence_database() first.")
        return
    
    # Test noise detection
    noise_logic = NoiseDetectionLogic(ci_calc)
    
    # Example test with HACK
    test_metrics = {
        'hit_rate': 0.56,
        'conviction': 2.7,
        'stability': 0.70
    }
    
    should_rotate, decision = noise_logic.should_rotate(
        'HACK.AX', test_metrics, current_rank=22, buffer_zone=20
    )
    
    print(f"\nTest Decision for HACK.AX:")
    print(f"  Current Rank: {decision['current_rank']}")
    print(f"  Decision: {decision['decision']}")
    print(f"  Reason: {decision['reason']}")
    print(f"  Signal Type: {decision['signal_type']}")
    print(f"  Metrics Outside CI: {decision['metrics_outside']}")
    
    return ci_calc, noise_logic

if __name__ == "__main__":
    test_confidence_intervals()
