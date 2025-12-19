"""
Shared Utilities Module - Updated
Common helper functions for new indicators and analysis
Removed: KAMA, RSI, Stochastic, VWAP, Holdings, Bias Correction helpers
Added: Volume Intelligence, Kalman Hull helpers
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple

# ============================================================================
# DATA EXTRACTION UTILITIES (KEPT)
# ============================================================================

def extract_column(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Extract column from DataFrame (handles multi-level columns from yfinance)"""
    if col_name not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    col = df[col_name]
    
    if isinstance(col, pd.Series):
        return col
    else:
        return col.iloc[:, 0]


def extract_price_data(etf_data: pd.DataFrame) -> pd.Series:
    """Extract close price data from ETF DataFrame"""
    return extract_column(etf_data, 'Close')


def extract_adjusted_price(etf_data: pd.DataFrame) -> Union[pd.Series, None]:
    """Extract adjusted close price data from ETF DataFrame"""
    # Try 'Adj Close' first (yfinance standard)
    if 'Adj Close' in etf_data.columns:
        return extract_column(etf_data, 'Adj Close')
    
    # Fallback to regular Close if Adj Close not available
    if 'Close' in etf_data.columns:
        return extract_column(etf_data, 'Close')
    
    # Return None if no price data available
    return None


def extract_volume_data(etf_data: pd.DataFrame) -> pd.Series:
    """Extract volume data from ETF DataFrame"""
    if 'Volume' in etf_data.columns:
        return extract_column(etf_data, 'Volume')
    return pd.Series(index=etf_data.index, dtype=float)


def transform_to_returns(prices: pd.Series) -> pd.Series:
    """Transform price series to returns"""
    return prices.pct_change().dropna()


# ============================================================================
# QUALITY ASSESSMENT UTILITIES (KEPT)
# ============================================================================

def calculate_quality_flag(hit_rate: float, confidence: float) -> str:
    """Evaluate forecast quality and return appropriate flag"""
    if hit_rate > 0.65 and confidence > 0.7:
        return '[EMOJI]'  # EXCELLENT
    elif hit_rate > 0.55 and confidence > 0.5:
        return '~'   # FAIR
    elif hit_rate < 0.45 or confidence < 0.15:
        return '[EMOJI]'  # POOR
    else:
        return '[EMOJI]'  # WARNING


def get_quality_tier(years_available: float) -> Tuple[str, float]:
    """Determine data quality tier based on years available"""
    if years_available >= 3:
        return 'tier_1', 0.0
    elif years_available >= 2:
        return 'tier_2', -2.0
    elif years_available >= 1:
        return 'tier_3', -5.0
    else:
        return 'tier_4', -10.0


# ============================================================================
# NEW VOLUME INTELLIGENCE HELPERS (ADDED)
# ============================================================================

def calculate_relative_volume_ratio(volumes: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate relative volume ratio: current_vol / MA(vol, period)
    Used for spike score calculation
    """
    vol_ma = volumes.rolling(window=period).mean()
    return volumes / vol_ma.replace(0, np.nan)


def calculate_rolling_correlation(series1: pd.Series, series2: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate rolling correlation between two series
    Used for price-volume correlation calculation
    """
    return series1.rolling(window=period).corr(series2)


def z_score_normalize(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate z-score normalization for a series
    Used for spike score z-score calculation
    """
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    return (series - rolling_mean) / rolling_std.replace(0, 1)


def detect_trend_divergence(trend1: pd.Series, trend2: pd.Series, period: int = 5) -> str:
    """
    Compare two trends and detect divergence
    Returns: 'bullish', 'bearish', or 'none'
    Used for Kalman Hull and A/D divergence detection
    """
    if len(trend1) < period or len(trend2) < period:
        return 'none'
    
    # Get recent trends
    recent_trend1 = trend1.tail(period).mean()
    recent_trend2 = trend2.tail(period).mean()
    
    # Detect divergence
    if recent_trend1 > 0 and recent_trend2 > 0:
        return 'bullish'
    elif recent_trend1 < 0 and recent_trend2 < 0:
        return 'bearish'
    else:
        return 'none'


# ============================================================================
# TECHNICAL UTILITIES (SIMPLIFIED - kept only essentials)
# ============================================================================

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()


# ============================================================================
# RISK UTILITIES (KEPT)
# ============================================================================

def calculate_annual_volatility(returns: pd.Series, period: int = 252) -> float:
    """Calculate annualized volatility"""
    if len(returns) < period:
        return returns.std() * np.sqrt(252)
    return returns.tail(period).std() * np.sqrt(252)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0435) -> float:
    """Calculate Sharpe ratio"""
    if returns.std() == 0:
        return np.nan
    return (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252))


# ============================================================================
# NOTE: REMOVED FUNCTIONS
# ============================================================================
# - KAMA calculations (Kaufman Adaptive Moving Average)
# - RSI calculations (Relative Strength Index)
# - Stochastic calculations (%K, %D, regime)
# - VWAP calculations (Volume Weighted Average Price)
# - Holdings-related utilities
# - Bias correction helpers
# - VaR-only functions (kept CVaR in main components)
