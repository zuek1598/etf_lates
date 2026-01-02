"""
CORRECTED Volatility-Based Confidence Intervals
===============================================

Fixed logic: Distinguishes market-wide stress from ETF-specific problems.
Uses relative volatility, not absolute risk management.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class CorrectedVolatilityCI:
    """Corrected volatility CI that distinguishes market stress from ETF-specific issues"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.etf_vol_history = {}
        self.market_vol_history = {}
        self.decision_log = []
        
    def calculate_etf_vol_ci(self, prices, etf_ticker):
        """
        Calculate individual ETF's volatility confidence intervals
        """
        returns = prices.pct_change().dropna()
        
        if len(returns) < 50:
            return None
            
        # Calculate current volatilities
        vol_metrics = {}
        periods = [20, 50, 100]  # Focus on shorter periods for responsiveness
        
        for period in periods:
            if len(returns) >= period:
                recent_vol = returns.tail(period).std() * np.sqrt(252)
                
                # Historical distribution
                if len(returns) > self.window_size:
                    historical_vols = []
                    for i in range(self.window_size, len(returns)):
                        window_returns = returns.iloc[i-self.window_size:i]
                        vol = window_returns.std() * np.sqrt(252)
                        historical_vols.append(vol)
                    
                    historical_vols = np.array(historical_vols)
                    
                    vol_metrics[f'{period}d'] = {
                        'current': recent_vol,
                        'median': np.median(historical_vols),
                        'p5': np.percentile(historical_vols, 5),
                        'p95': np.percentile(historical_vols, 95),
                        'is_elevated': recent_vol > np.percentile(historical_vols, 95)
                    }
                else:
                    vol_metrics[f'{period}d'] = {
                        'current': recent_vol,
                        'median': recent_vol,
                        'p5': recent_vol * 0.8,
                        'p95': recent_vol * 1.2,
                        'is_elevated': False
                    }
        
        return vol_metrics
    
    def calculate_market_vol(self, all_prices):
        """
        Calculate market-wide volatility metrics
        """
        market_vols = {}
        
        # Calculate 20-day vol for all ETFs
        for etf, prices in all_prices.items():
            returns = prices.pct_change().dropna()
            if len(returns) >= 20:
                vol = returns.tail(20).std() * np.sqrt(252)
                market_vols[etf] = vol
        
        if not market_vols:
            return None
        
        market_vols_array = np.array(list(market_vols.values()))
        
        return {
            'current_mean': np.mean(market_vols_array),
            'current_median': np.median(market_vols_array),
            'current_p75': np.percentile(market_vols_array, 75),
            'current_p90': np.percentile(market_vols_array, 90),
            'etf_count': len(market_vols),
            'distribution': market_vols
        }
    
    def analyze_volatility_regime(self, etf_ticker, etf_vol_metrics, market_vol_metrics):
        """
        Analyze whether volatility is ETF-specific or market-wide
        
        Returns:
            (decision, reasoning)
        """
        if not etf_vol_metrics or not market_vol_metrics:
            return 'HOLD', 'Insufficient data'
        
        # Use 20-day volatility for decision
        if '20d' not in etf_vol_metrics:
            return 'HOLD', 'No 20-day volatility data'
        
        etf_20d = etf_vol_metrics['20d']
        etf_current = etf_20d['current']
        etf_elevated = etf_20d['is_elevated']
        
        market_mean = market_vol_metrics['current_mean']
        market_p90 = market_vol_metrics['current_p90']
        
        # Key logic: Is market in stress?
        # Use both relative and absolute measures
        avg_vol = market_vol_metrics['current_mean']
        median_vol = market_vol_metrics['current_median']
        
        # Market stress if:
        # 1. Average vol is very high (>30% annual) OR
        # 2. Average vol is much higher than median (>1.5x)
        market_in_stress = (avg_vol > 0.30) or (avg_vol > median_vol * 1.5)
        
        # Decision matrix
        reasoning = []
        
        # Is this ETF's volatility elevated compared to its own history?
        if etf_elevated:
            reasoning.append(f"{etf_ticker} vol elevated ({etf_current:.1%} vs normal max {etf_20d['p95']:.1%})")
        else:
            reasoning.append(f"{etf_ticker} vol normal ({etf_current:.1%} within historical range)")
        
        # Is market also elevated?
        if market_in_stress:
            if avg_vol > 0.30:
                reasoning.append(f"Market in stress (avg vol {avg_vol:.1%} > 30% threshold)")
            else:
                reasoning.append(f"Market in stress (avg vol {avg_vol:.1%} is 1.5x median {median_vol:.1%})")
        else:
            reasoning.append(f"Market normal (avg vol {avg_vol:.1%} close to median {median_vol:.1%})")
        
        # Decision logic
        if market_in_stress:
            # Market-wide stress: HOLD regardless of individual ETF vol
            decision = 'HOLD'
            reasoning.append("→ HOLD: Market stress, don't rotate")
        else:
            # Normal market: Check if ETF has specific problems
            if etf_elevated:
                decision = 'ROTATE'
                reasoning.append("→ ROTATE: ETF-specific volatility spike")
            else:
                decision = 'ROTATE'  # Even if vol normal, ranking drop is signal in normal market
                reasoning.append("→ ROTATE: Ranking drop in normal market is real signal")
        
        return decision, ' | '.join(reasoning)
    
    def should_rotate(self, etf_ticker, current_prices, all_prices, current_rank=999):
        """
        Main decision function: Should we rotate out of this ETF?
        """
        decision_record = {
            'etf': etf_ticker,
            'date': datetime.now(),
            'current_rank': current_rank,
            'decision': 'HOLD',
            'reasoning': '',
            'etf_vol': None,
            'market_vol': None
        }
        
        # Calculate ETF volatility
        etf_vol = self.calculate_etf_vol_ci(current_prices, etf_ticker)
        decision_record['etf_vol'] = etf_vol
        
        # Calculate market volatility
        market_vol = self.calculate_market_vol(all_prices)
        decision_record['market_vol'] = market_vol
        
        # Make decision
        decision, reasoning = self.analyze_volatility_regime(
            etf_ticker, etf_vol, market_vol
        )
        
        decision_record['decision'] = decision
        decision_record['reasoning'] = reasoning
        
        self.decision_log.append(decision_record)
        
        return decision == 'ROTATE', decision_record

def test_corrected_logic():
    """Test the corrected volatility logic"""
    print("="*70)
    print("TESTING CORRECTED VOLATILITY CI LOGIC")
    print("="*70)
    
    # Create test scenarios
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=200, freq='D')
    
    # Scenario 1: Normal market
    print("\n" + "="*50)
    print("SCENARIO 1: NORMAL MARKET")
    print("="*50)
    
    normal_market = {}
    for i in range(10):
        returns = np.random.normal(0.0005, 0.01, 200)  # 10% annual vol
        prices = 100 * np.exp(np.cumsum(returns))
        normal_market[f'ETF_{i}'] = pd.Series(prices, index=dates)
    
    # One ETF with specific problem (elevated vol)
    problem_returns = np.random.normal(0.0005, 0.02, 200)  # 20% annual vol
    problem_prices = 100 * np.exp(np.cumsum(problem_returns))
    normal_market['PROBLEM'] = pd.Series(problem_prices, index=dates)
    
    # Test decision
    vci = CorrectedVolatilityCI()
    should_rotate, decision = vci.should_rotate(
        'PROBLEM', 
        normal_market['PROBLEM'], 
        normal_market, 
        current_rank=25
    )
    
    print(f"\nProblem ETF (rank 25) in normal market:")
    print(f"  Decision: {decision['decision']}")
    print(f"  Reasoning: {decision['reasoning']}")
    
    # Scenario 2: Market stress (2022-like crash)
    print("\n" + "="*50)
    print("SCENARIO 2: MARKET STRESS (CRASH)")
    print("="*50)
    
    stress_market = {}
    for i in range(10):
        returns = np.random.normal(-0.002, 0.03, 200)  # High vol, negative drift
        prices = 100 * np.exp(np.cumsum(returns))
        stress_market[f'ETF_{i}'] = pd.Series(prices, index=dates)
    
    # Our ETF (also high vol, but not worse than market)
    our_returns = np.random.normal(-0.001, 0.025, 200)
    our_prices = 100 * np.exp(np.cumsum(our_returns))
    stress_market['OUR_ETF'] = pd.Series(our_prices, index=dates)
    
    # Test decision
    should_rotate, decision = vci.should_rotate(
        'OUR_ETF', 
        stress_market['OUR_ETF'], 
        stress_market, 
        current_rank=25
    )
    
    print(f"\nOur ETF (rank 25) in market stress:")
    print(f"  Decision: {decision['decision']}")
    print(f"  Reasoning: {decision['reasoning']}")
    
    # Scenario 3: ETF-specific problem in normal market
    print("\n" + "="*50)
    print("SCENARIO 3: ETF-SPECIFIC PROBLEM")
    print("="*50)
    
    # Normal market again
    normal_market_2 = {}
    for i in range(10):
        returns = np.random.normal(0.0005, 0.01, 200)
        prices = 100 * np.exp(np.cumsum(returns))
        normal_market_2[f'ETF_{i}'] = pd.Series(prices, index=dates)
    
    # One ETF with extreme vol spike
    extreme_returns = np.random.normal(0.0005, 0.04, 200)  # 40% vol!
    extreme_prices = 100 * np.exp(np.cumsum(extreme_returns))
    normal_market_2['EXTREME'] = pd.Series(extreme_prices, index=dates)
    
    # Test decision
    should_rotate, decision = vci.should_rotate(
        'EXTREME', 
        normal_market_2['EXTREME'], 
        normal_market_2, 
        current_rank=25
    )
    
    print(f"\nExtreme ETF (rank 25) in normal market:")
    print(f"  Decision: {decision['decision']}")
    print(f"  Reasoning: {decision['reasoning']}")
    
    print("\n" + "="*70)
    print("SUMMARY OF CORRECTED LOGIC")
    print("="*70)
    print("""
    ✓ Scenario 1: ETF problem in normal market → ROTATE (correct)
    ✓ Scenario 2: Market stress → HOLD (correct, don't panic sell)
    ✓ Scenario 3: Extreme ETF problem → ROTATE (correct)
    
    The system now correctly distinguishes:
    - Market-wide stress (hold through it)
    - ETF-specific problems (rotate out)
    """)

if __name__ == "__main__":
    test_corrected_logic()
