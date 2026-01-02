"""
ETF Metric Calculation Module
=============================

Standardized functions for calculating ETF quality metrics.
These functions MUST be used identically when building confidence intervals
AND during backtesting to ensure consistency.

All functions follow the specification in /docs/METRIC_SPECIFICATION.md
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def calculate_hit_rate(prices: pd.Series, benchmark_prices: pd.Series = None, 
                      forecast_window: int = 40, lookback_window: int = 100) -> float:
    """
    Calculate hit rate: percentage of positive return periods.
    
    Args:
        prices: ETF price series
        benchmark_prices: Not used - kept for backward compatibility
        forecast_window: Days to look forward for performance
        lookback_window: Number of past days to analyze
        
    Returns:
        Hit rate between 0 and 1
    """
    if len(prices) < lookback_window:
        return 0.0
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    if len(returns) < lookback_window:
        return 0.0
    
    # Take last lookback_window periods
    recent_returns = returns.tail(lookback_window)
    
    # Hit rate is percentage of positive returns
    hit_rate = (recent_returns > 0).mean()
    
    return hit_rate


def calculate_conviction(prices: pd.Series, 
                        recent_period: int = 10, 
                        long_period: int = 40) -> float:
    """
    Calculate conviction: strength of recent momentum relative to historical.
    
    Args:
        prices: ETF price series
        recent_period: Days for recent momentum calculation
        long_period: Days for historical momentum average
        
    Returns:
        Conviction value (can be negative or positive)
    """
    if len(prices) < long_period:
        return 0.0
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    if len(returns) < long_period:
        return 0.0
    
    # Calculate momentum
    recent_momentum = returns.tail(recent_period).mean()
    avg_momentum = returns.tail(long_period).mean()
    
    # Calculate conviction
    if avg_momentum != 0:
        conviction = recent_momentum / abs(avg_momentum)
    else:
        conviction = 0.0
    
    return conviction


def calculate_stability(prices: pd.Series, 
                       volatility_period: int = 30) -> float:
    """
    Calculate stability: bounded inverse volatility.
    
    Args:
        prices: ETF price series
        volatility_period: Days for volatility calculation
        
    Returns:
        Stability between 0 and 1 (higher = more stable)
    """
    if len(prices) < volatility_period:
        return 0.0
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    if len(returns) < volatility_period:
        return 0.0
    
    # Calculate volatility
    volatility = returns.tail(volatility_period).std()
    
    # Apply bounded inverse formula
    stability = 1.0 / (1.0 + volatility * 10)
    
    return stability


def calculate_all_metrics(prices: pd.Series, 
                         benchmark_prices: Optional[pd.Series] = None,
                         as_of_date: Optional[pd.Timestamp] = None) -> Dict[str, float]:
    """
    Calculate all ETF quality metrics using standardized methods.
    
    Args:
        prices: ETF price series
        benchmark_prices: Benchmark price series (optional, defaults to ASX 200)
        as_of_date: Date for calculation (for logging)
        
    Returns:
        Dictionary with hit_rate, conviction, and stability
    """
    # If no benchmark provided, use a simple relative calculation
    if benchmark_prices is None:
        # Use a simple benchmark based on the ETF's own historical performance
        # This is a fallback - ideally always provide a real benchmark
        benchmark_prices = prices
    
    # Calculate all metrics
    hit_rate = calculate_hit_rate(prices, benchmark_prices)
    conviction = calculate_conviction(prices)
    stability = calculate_stability(prices)
    
    return {
        'hit_rate': hit_rate,
        'conviction': conviction,
        'stability': stability
    }


def calculate_distance_weight(current_value: float, 
                            ci_lower: float, 
                            ci_upper: float) -> Tuple[float, float]:
    """
    Calculate weighted distance of a metric from its confidence interval.
    
    Args:
        current_value: Current metric value
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        
    Returns:
        Tuple of (weight, distance_percentage)
    """
    # Check if value is within CI
    if ci_lower <= current_value <= ci_upper:
        return 0.0, 0.0
    
    # Calculate distance beyond nearest bound
    if current_value < ci_lower:
        distance = ci_lower - current_value
        nearest_bound = ci_lower
    else:  # current_value > ci_upper
        distance = current_value - ci_upper
        nearest_bound = ci_upper
    
    # Calculate percentage distance
    if nearest_bound != 0:
        percentage_distance = distance / abs(nearest_bound)
    else:
        percentage_distance = distance
    
    # Assign weight based on distance tier
    if percentage_distance <= 0.10:  # 0-10% beyond CI
        weight = 0.3
    elif percentage_distance <= 0.20:  # 10-20% beyond CI
        weight = 0.7
    else:  # 20%+ beyond CI
        weight = 1.0
    
    return weight, percentage_distance


def make_rotation_decision(total_weight: float) -> str:
    """
    Make rotation decision based on total weighted distance.
    
    Args:
        total_weight: Sum of weights from all metrics outside CI
        
    Returns:
        Decision: 'HOLD', 'WAIT', or 'ROTATE'
    """
    if total_weight < 0.5:
        return 'HOLD'
    elif total_weight <= 1.5:
        return 'WAIT'
    else:
        return 'ROTATE'


# Test cases for verification
def _run_test_cases():
    """Run test cases to verify metric calculations."""
    print("Running metric calculation test cases...")
    
    # Create test price series
    dates = pd.date_range('2021-01-01', periods=100)
    
    # Stable ETF (low volatility)
    stable_prices = pd.Series(100 * (1 + np.random.randn(100) * 0.005), index=dates)
    
    # Volatile ETF (high volatility)
    volatile_prices = pd.Series(100 * (1 + np.random.randn(100) * 0.02), index=dates)
    
    # Trending ETF (positive momentum)
    trend = np.linspace(0, 0.2, 100)
    trending_prices = pd.Series(100 * (1 + trend + np.random.randn(100) * 0.01), index=dates)
    
    # Test stability
    stable_stability = calculate_stability(stable_prices)
    volatile_stability = calculate_stability(volatile_prices)
    
    print(f"Stable ETF stability: {stable_stability:.3f}")
    print(f"Volatile ETF stability: {volatile_stability:.3f}")
    assert stable_stability > volatile_stability, "Stable ETF should have higher stability"
    
    # Test conviction
    trending_conviction = calculate_conviction(trending_prices)
    print(f"Trending ETF conviction: {trending_conviction:.3f}")
    
    # Test distance weighting
    weight, pct = calculate_distance_weight(0.3, 0.4, 0.6)
    print(f"Below CI: weight={weight}, distance={pct:.1%}")
    assert weight == 0.3, "Should be borderline weight"
    
    weight, pct = calculate_distance_weight(0.8, 0.4, 0.6)
    print(f"Above CI: weight={weight}, distance={pct:.1%}")
    assert weight == 1.0, "Should be strong weight"
    
    # Test decisions
    assert make_rotation_decision(0.3) == 'HOLD'
    assert make_rotation_decision(1.0) == 'WAIT'
    assert make_rotation_decision(2.0) == 'ROTATE'
    
    print("✓ All test cases passed!")


if __name__ == "__main__":
    _run_test_cases()
