"""
Macro Economic Cycle Overlay Framework
Efficient implementation with minimal lines
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DATA FETCHING
# ============================================

def fetch_all_data(lookback_days: int = 365) -> Dict[str, pd.Series]:
    """Fetch all required data in one batch for efficiency"""
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    
    # All tickers needed
    tickers = {
        # Credit & Bonds
        'HYG': 'HYG', 'LQD': 'LQD', 'TLT': 'TLT',
        # Dollar
        'DXY': 'DX-Y.NYB',
        # Sectors
        'XLF': 'XLF', 'XLI': 'XLI', 'XLY': 'XLY', 'XLE': 'XLE',
        'XLK': 'XLK', 'XLV': 'XLV', 'XLU': 'XLU', 'XLP': 'XLP',
        # Benchmarks
        'SPY': 'SPY',
    }
    
    # Batch download
    data = yf.download(list(tickers.values()), start=start, end=end, progress=False)['Close']
    
    # Get FRED data (Treasury yields) with fallback
    try:
        from pandas_datareader import data as web
        us10y = web.DataReader('DGS10', 'fred', start, end)['DGS10']
        us2y = web.DataReader('DGS2', 'fred', start, end)['DGS2']
        fed_funds = web.DataReader('FEDFUNDS', 'fred', start, end)['FEDFUNDS']
        cpi = web.DataReader('CPIAUCSL', 'fred', start, end)['CPIAUCSL']
    except Exception as e:
        # Fallback: use yfinance treasury ETFs as proxies
        print(f" FRED data unavailable ({e}), using ETF proxies")
        try:
            tef_data = yf.download(['^TNX', '^FVX'], start=start, end=end, progress=False)['Close']
            us10y = tef_data['^TNX'] if '^TNX' in tef_data.columns else None
            us2y = tef_data['^FVX'] if '^FVX' in tef_data.columns else None
        except:
            us10y = us2y = None
        fed_funds = cpi = None
    
    # Organize into clean dictionary
    result = {k: data[v] if v in data.columns else data for k, v in tickers.items()}
    result.update({'US10Y': us10y, 'US2Y': us2y, 'FEDFUNDS': fed_funds, 'CPI': cpi})
    
    return result

# ============================================
# UTILITIES
# ============================================

def pct_change(series: pd.Series, periods: int) -> float:
    """Calculate percentage change over periods"""
    return series.pct_change(periods).iloc[-1] * 100 if len(series) > periods else 0

def normalize(value: float, center: float, scale: float, min_val: float = 0, max_val: float = 100) -> float:
    """Normalize value to 0-100 scale"""
    return np.clip(center + (value / scale) * 50, min_val, max_val)

# ============================================
# FACTOR 1: SYSTEMATIC RISK (35%)
# ============================================

def calc_credit_spreads(hyg: pd.Series, tlt: pd.Series) -> float:
    """Credit spread dynamics (40% of factor)"""
    # Divergence: TLT outperforming HYG = risk-off
    hyg_30d, tlt_30d = pct_change(hyg, 30), pct_change(tlt, 30)
    divergence = tlt_30d - hyg_30d
    
    # Acceleration vs 60-day average
    hyg_60d, tlt_60d = pct_change(hyg, 60), pct_change(tlt, 60)
    divergence_60d = tlt_60d - hyg_60d
    acceleration = ((divergence / divergence_60d) - 1) * 100 if divergence_60d != 0 else 0
    
    # Score: spreads widening (divergence > 0) = risk-off
    spread_score = normalize(divergence, 50, 5)
    accel_score = normalize(acceleration, 50, 50)
    
    return 0.70 * spread_score + 0.30 * accel_score

def calc_dollar_strength(dxy: pd.Series) -> float:
    """Dollar strength momentum (35% of factor)"""
    # Multi-timeframe momentum
    dxy_20d, dxy_60d = pct_change(dxy, 20), pct_change(dxy, 60)
    momentum = 0.60 * dxy_20d + 0.40 * dxy_60d
    
    # Acceleration
    dxy_10d_ago = dxy.iloc[-11:-1]
    mom_10d_ago = 0.60 * pct_change(dxy_10d_ago, 20) + 0.40 * pct_change(dxy_10d_ago, 60)
    acceleration = ((momentum / mom_10d_ago) - 1) * 100 if mom_10d_ago != 0 else 0
    
    # Strong dollar (positive momentum) = risk-off for international
    dollar_score = normalize(momentum, 50, 3)
    accel_score = normalize(acceleration, 50, 10)
    
    return 0.75 * dollar_score + 0.25 * accel_score

def calc_yield_curve(us10y: Optional[pd.Series], us2y: Optional[pd.Series]) -> float:
    """Yield curve dynamics (25% of factor)"""
    if us10y is None or us2y is None or len(us10y) < 30 or len(us2y) < 30:
        return 50.0  # Neutral if no data
    
    # Current spread in bps
    spread_bps = (us10y.iloc[-1] - us2y.iloc[-1]) * 100
    
    # Spread momentum (steepening/flattening)
    spread_30d_ago = (us10y.iloc[-30] - us2y.iloc[-30]) * 100
    spread_change = spread_bps - spread_30d_ago
    
    # Score based on curve shape
    if spread_bps > 100:
        curve_score = 25 + min(75, (spread_bps - 100) / 2)
    elif spread_bps > 0:
        curve_score = 25 + (spread_bps / 100) * 25
    else:  # Inverted
        curve_score = max(0, 25 + spread_bps / 2)
    
    # Factor in momentum
    momentum_score = normalize(spread_change, 50, 25)
    
    return 0.70 * curve_score + 0.30 * momentum_score

def calc_systematic_risk(data: Dict) -> float:
    """Factor 1: Systematic Risk Assessment"""
    credit = calc_credit_spreads(data['HYG'], data['TLT'])
    dollar = calc_dollar_strength(data['DXY'])
    curve = calc_yield_curve(data['US10Y'], data['US2Y'])
    
    return 0.40 * credit + 0.35 * dollar + 0.25 * curve

# ============================================
# FACTOR 2: GROWTH MOMENTUM (30%)
# ============================================

def calc_pmi_proxy(xli: pd.Series, spy: pd.Series) -> float:
    """Economic indicator momentum - use XLI/SPY as PMI proxy (45% of factor)"""
    # XLI outperformance = manufacturing strength
    xli_60d, spy_60d = pct_change(xli, 60), pct_change(spy, 60)
    industrial_momentum = xli_60d - spy_60d
    
    # Normalize: >5% outperformance = strong growth
    return normalize(industrial_momentum, 50, 5)

def calc_earnings_revisions(data: Dict) -> float:
    """Earnings revision trends via sector rotation (35% of factor)"""
    # Cyclicals vs Defensives
    cyclical_60d = np.mean([pct_change(data[s], 60) for s in ['XLI', 'XLY', 'XLF']])
    defensive_60d = np.mean([pct_change(data[s], 60) for s in ['XLU', 'XLV', 'XLP']])
    cyclical_outperformance = cyclical_60d - defensive_60d
    
    # Normalize: >5% = strong earnings optimism
    return normalize(cyclical_outperformance, 50, 5)

def calc_sector_rotation(data: Dict) -> float:
    """Relative sector strength (20% of factor)"""
    # Calculate 20-day returns for all sectors
    sectors = {
        'XLF': pct_change(data['XLF'], 20), 'XLI': pct_change(data['XLI'], 20),
        'XLY': pct_change(data['XLY'], 20), 'XLE': pct_change(data['XLE'], 20),
        'XLK': pct_change(data['XLK'], 20), 'XLV': pct_change(data['XLV'], 20),
        'XLU': pct_change(data['XLU'], 20), 'XLP': pct_change(data['XLP'], 20),
    }
    
    # Get top 3 leaders
    sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
    leaders = [s[0] for s in sorted_sectors[:3]]
    
    # Score based on leadership pattern
    if 'XLF' in leaders or 'XLI' in leaders:
        rotation_score = 75  # Early cycle
    elif 'XLK' in leaders or 'XLY' in leaders:
        rotation_score = 60  # Mid cycle
    elif 'XLE' in leaders:
        rotation_score = 50  # Late cycle
    else:  # Defensives leading
        rotation_score = 25  # Recession
    
    # Adjust for breadth
    top3_avg = np.mean([s[1] for s in sorted_sectors[:3]])
    bottom3_avg = np.mean([s[1] for s in sorted_sectors[-3:]])
    breadth = top3_avg - bottom3_avg
    breadth_score = normalize(breadth, 0, 10)
    
    return 0.75 * rotation_score + 0.25 * breadth_score

def calc_growth_momentum(data: Dict) -> float:
    """Factor 2: Growth Momentum Assessment"""
    pmi = calc_pmi_proxy(data['XLI'], data['SPY'])
    earnings = calc_earnings_revisions(data)
    rotation = calc_sector_rotation(data)
    
    return 0.45 * pmi + 0.35 * earnings + 0.20 * rotation

# ============================================
# FACTOR 3: MONETARY POLICY (25%)
# ============================================

def calc_real_rates(us10y: Optional[pd.Series], cpi: Optional[pd.Series]) -> float:
    """Real interest rates (45% of factor)"""
    if us10y is None or cpi is None or len(cpi) < 12:
        return 50.0  # Neutral
    
    # Calculate real rate
    us10y_yield = us10y.iloc[-1]
    inflation_rate = ((cpi.iloc[-1] / cpi.iloc[-12]) - 1) * 100
    real_rate = us10y_yield - inflation_rate
    
    # Real rate momentum
    if len(cpi) > 72:
        inflation_60d_ago = ((cpi.iloc[-60] / cpi.iloc[-72]) - 1) * 100
        real_rate_60d_ago = us10y.iloc[-60] - inflation_60d_ago
        real_rate_change = real_rate - real_rate_60d_ago
    else:
        real_rate_change = 0
    
    # Score: negative real rates = growth tailwind
    if real_rate < -1:
        rate_score = max(0, 50 - ((real_rate + 1) / 2) * 50)
    elif real_rate < 2:
        rate_score = 50 - (real_rate / 2) * 25
    else:
        rate_score = max(0, 25 - ((real_rate - 2) / 3) * 25)
    
    # Falling rates = improving for growth
    momentum_score = normalize(-real_rate_change, 50, 0.5)
    
    return 0.70 * rate_score + 0.30 * momentum_score

def calc_inflation_trajectory(cpi: Optional[pd.Series]) -> float:
    """Inflation trajectory (30% of factor)"""
    if cpi is None or len(cpi) < 12:
        return 50.0
    
    # 12-month inflation
    inflation_12m = ((cpi.iloc[-1] / cpi.iloc[-12]) - 1) * 100
    
    # Score: low/falling inflation = favor growth
    if inflation_12m < 0:
        inflation_score = 85
    elif inflation_12m < 2:
        inflation_score = 75
    elif inflation_12m < 4:
        inflation_score = 50
    elif inflation_12m < 6:
        inflation_score = 30
    else:
        inflation_score = 15
    
    # Calculate acceleration if enough data
    if len(cpi) > 9:
        inflation_3m = ((cpi.iloc[-1] / cpi.iloc[-3]) - 1) * 100 * 4
        inflation_6m = ((cpi.iloc[-3] / cpi.iloc[-6]) - 1) * 100 * 4
        acceleration = inflation_3m - inflation_6m
        accel_score = normalize(-acceleration, 50, 2)
        return 0.70 * inflation_score + 0.30 * accel_score
    
    return inflation_score

def calc_policy_stance(fed_funds: Optional[pd.Series]) -> float:
    """Central bank policy stance (25% of factor)"""
    if fed_funds is None or len(fed_funds) < 6:
        return 50.0
    
    # Policy direction
    current = fed_funds.iloc[-1]
    rate_3m = current - fed_funds.iloc[-3] if len(fed_funds) > 3 else 0
    rate_6m = current - fed_funds.iloc[-6] if len(fed_funds) > 6 else 0
    
    # Classify stance
    if rate_3m < -0.25:
        base_score = 75  # Cutting
    elif rate_3m > 0.25:
        base_score = 25  # Hiking
    elif current < 2:
        base_score = 65  # Accommodative
    elif current > 4:
        base_score = 35  # Restrictive
    else:
        base_score = 50  # Neutral
    
    # Adjust for momentum
    momentum_adj = (rate_6m - rate_3m) * 10
    
    return np.clip(base_score - momentum_adj, 0, 100)

def calc_monetary_policy(data: Dict) -> float:
    """Factor 3: Monetary Policy Conditions"""
    real_rates = calc_real_rates(data['US10Y'], data['CPI'])
    inflation = calc_inflation_trajectory(data['CPI'])
    policy = calc_policy_stance(data['FEDFUNDS'])
    
    return 0.45 * real_rates + 0.30 * inflation + 0.25 * policy

# ============================================
# FACTOR 4: MARKET REGIME (10%)
# ============================================

def classify_market_regime(systematic_risk: float, growth_momentum: float, monetary_policy: float) -> Dict:
    """Classify market regime and return dynamic weights"""
    
    # CRISIS REGIME
    if systematic_risk > 60 or growth_momentum < 30 or (systematic_risk > 50 and growth_momentum < 40):
        return {
            'regime': 'CRISIS',
            'score': max(0, 30 - (growth_momentum / 3)),
            'weights': {'systematic_risk': 0.45, 'growth_momentum': 0.25, 'monetary_policy': 0.20, 'regime': 0.10}
        }
    
    # GOLDILOCKS REGIME
    elif systematic_risk < 40 and growth_momentum > 60 and monetary_policy > 55:
        return {
            'regime': 'GOLDILOCKS',
            'score': min(100, 70 + ((growth_momentum - 60) / 2)),
            'weights': {'systematic_risk': 0.20, 'growth_momentum': 0.40, 'monetary_policy': 0.25, 'regime': 0.15}
        }
    
    # TRANSITIONAL REGIME
    else:
        return {
            'regime': 'TRANSITIONAL',
            'score': 50,
            'weights': {'systematic_risk': 0.35, 'growth_momentum': 0.30, 'monetary_policy': 0.25, 'regime': 0.10}
        }


# ============================================
# MAIN CALCULATION
# ============================================

def calculate_macro_framework(data: Optional[Dict] = None) -> Dict:
    """
    Main function: Calculate complete macro framework
    Returns comprehensive macro result dictionary
    """
    # Fetch data if not provided
    if data is None:
        print("Fetching market data...")
        data = fetch_all_data()
    
    print(" Calculating macro factors...")
    
    # Calculate all four factors
    systematic_risk = calc_systematic_risk(data)
    growth_momentum = calc_growth_momentum(data)
    monetary_policy = calc_monetary_policy(data)
    
    # Classify regime and get dynamic weights
    regime_result = classify_market_regime(systematic_risk, growth_momentum, monetary_policy)
    
    # Calculate weighted composite score
    weights = regime_result['weights']
    composite_score = (
        weights['systematic_risk'] * systematic_risk +
        weights['growth_momentum'] * growth_momentum +
        weights['monetary_policy'] * monetary_policy +
        weights['regime'] * regime_result['score']
    )
    
    # Convert to multiplier (0.75 - 1.25)
    multiplier = 0.75 + (composite_score / 100) * 0.50
    
    return {
        'multiplier': round(multiplier, 4),
        'composite_score': round(composite_score, 2),
        'regime': regime_result['regime'],
        'regime_weights': regime_result['weights'],
        'factors': {
            'systematic_risk': round(systematic_risk, 2),
            'growth_momentum': round(growth_momentum, 2),
            'monetary_policy': round(monetary_policy, 2),
            'regime_classification': round(regime_result['score'], 2)
        }
    }



# ============================================
# EXECUTION
# ============================================

if __name__ == "__main__":
    # Calculate macro framework
    result = calculate_macro_framework()
    
    print("\n" + "="*60)
    print("MACRO ECONOMIC CYCLE OVERLAY - RESULTS")
    print("="*60)
    print(f"\nMULTIPLIER: {result['multiplier']:.4f} (0.75-1.25)")
    print(f"COMPOSITE SCORE: {result['composite_score']:.2f}/100")
    print(f" REGIME: {result['regime']}")
    print(f"\nFACTOR BREAKDOWN:")
    print(f"   • Systematic Risk:      {result['factors']['systematic_risk']:>6.2f}/100 (weight: {result['regime_weights']['systematic_risk']:.0%})")
    print(f"   • Growth Momentum:      {result['factors']['growth_momentum']:>6.2f}/100 (weight: {result['regime_weights']['growth_momentum']:.0%})")
    print(f"   • Monetary Policy:      {result['factors']['monetary_policy']:>6.2f}/100 (weight: {result['regime_weights']['monetary_policy']:.0%})")
    print(f"   • Regime Classification:{result['factors']['regime_classification']:>6.2f}/100 (weight: {result['regime_weights']['regime']:.0%})")
    print("\n" + "="*60 + "\n")

