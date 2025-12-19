"""
Geopolitical Risk Overlay Framework
Protects capital from tail risks and black swan events
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DATA FETCHING
# ============================================

def fetch_geo_data(lookback_days: int = 365) -> Dict[str, pd.Series]:
    """Fetch all geopolitical data in one batch"""
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    
    # All tickers needed (DIFFERENT from macro framework)
    tickers = {
        # Taiwan/China tension
        'TSM': 'TSM', 'INTC': 'INTC', 'SAMSUNG': '005930.KS',
        'EWT': 'EWT', 'EWY': 'EWY', 'EWJ': 'EWJ',
        # Defense
        'LMT': 'LMT', 'RTX': 'RTX', 'NOC': 'NOC',
        # Safe havens & volatility
        'VIX': '^VIX', 'GLD': 'GLD', 'TLT': 'TLT',
        # Energy
        'WTI': 'CL=F', 'BRENT': 'BZ=F', 'NG': 'NG=F', 'XLE': 'XLE',
        # China & currencies
        'FXI': 'FXI', 'CNY': 'CNY=X',
        # Sectors for stress detection
        'XLI': 'XLI', 'XLY': 'XLY', 'XRT': 'XRT',
        'XLV': 'XLV', 'XLU': 'XLU', 'EWG': 'EWG',
        # Benchmark
        'SPY': 'SPY',
    }
    
    # Batch download with error handling
    try:
        data = yf.download(list(tickers.values()), start=start, end=end, progress=False)['Close']
        result = {}
        for key, ticker in tickers.items():
            if ticker in data.columns:
                result[key] = data[ticker]
            else:
                result[key] = data  # Single series case
        return result
    except Exception as e:
        print(f" Data fetch warning: {e}")
        return {}

# ============================================
# UTILITIES
# ============================================

def pct_change(series: pd.Series, periods: int) -> float:
    """Calculate percentage change over periods"""
    try:
        return series.pct_change(periods).iloc[-1] * 100 if len(series) > periods else 0
    except:
        return 0

def volatility(series: pd.Series, window: int) -> float:
    """Calculate annualized volatility"""
    try:
        return series.pct_change().rolling(window).std().iloc[-1] * np.sqrt(252) * 100
    except:
        return 0

def normalize(value: float, center: float, scale: float, min_val: float = 0, max_val: float = 100) -> float:
    """Normalize value to 0-100 scale"""
    return np.clip(center + (value / scale) * 50, min_val, max_val)

# ============================================
# PILLAR 1: US-CHINA-TAIWAN TENSION (30%)
# ============================================

def calc_semiconductor_stress(tsm: pd.Series, intc: pd.Series, samsung: pd.Series) -> float:
    """Semiconductor supply chain stress (40% of pillar)"""
    # TSM underperforming alternatives = Taiwan risk rising
    tsm_30d = pct_change(tsm, 30)
    intc_30d = pct_change(intc, 30)
    samsung_30d = pct_change(samsung, 30)
    
    alternative_avg = (intc_30d + samsung_30d) / 2
    divergence = alternative_avg - tsm_30d
    
    # Normalize: >8% divergence = crisis
    return normalize(divergence, 50, 8)

def calc_taiwan_risk_premium(ewt: pd.Series, ewy: pd.Series, ewj: pd.Series) -> float:
    """Taiwan equity risk premium (30% of pillar)"""
    # Valuation discount
    ewt_30d = pct_change(ewt, 30)
    peer_avg = (pct_change(ewy, 30) + pct_change(ewj, 30)) / 2
    valuation_discount = peer_avg - ewt_30d
    
    # Volatility spike
    ewt_vol_30d = volatility(ewt, 30)
    ewt_vol_252d = volatility(ewt, 252)
    vol_spike = ((ewt_vol_30d / ewt_vol_252d) - 1) * 100 if ewt_vol_252d > 0 else 0
    
    # Composite
    discount_score = normalize(valuation_discount, 0, 5)
    vol_score = normalize(vol_spike, 0, 50)
    
    return 0.60 * discount_score + 0.40 * vol_score

def calc_defense_positioning(lmt: pd.Series, rtx: pd.Series, noc: pd.Series, spy: pd.Series) -> float:
    """Defense contractor positioning (30% of pillar)"""
    # Defense basket outperformance
    defense_avg = (pct_change(lmt, 30) + pct_change(rtx, 30) + pct_change(noc, 30)) / 3
    spy_30d = pct_change(spy, 30)
    defense_premium = defense_avg - spy_30d
    
    # Normalize: >5% = strong conflict pricing
    return normalize(defense_premium, 50, 5)

def calc_us_china_taiwan_index(data: Dict) -> float:
    """Pillar 1: US-China-Taiwan Tension Index"""
    semi = calc_semiconductor_stress(data['TSM'], data['INTC'], data['SAMSUNG'])
    taiwan = calc_taiwan_risk_premium(data['EWT'], data['EWY'], data['EWJ'])
    defense = calc_defense_positioning(data['LMT'], data['RTX'], data['NOC'], data['SPY'])
    
    return 0.40 * semi + 0.30 * taiwan + 0.30 * defense

# ============================================
# PILLAR 2: MILITARY CONFLICT RISK (25%)
# ============================================

def calc_vix_fear(vix: pd.Series) -> float:
    """VIX spike detection (30% of pillar)"""
    vix_current = vix.iloc[-1]
    
    # VIX regime score
    if vix_current < 16:
        vix_regime = (vix_current / 16) * 30
    elif vix_current < 20:
        vix_regime = 30 + ((vix_current - 16) / 4) * 20
    elif vix_current < 30:
        vix_regime = 50 + ((vix_current - 20) / 10) * 30
    else:
        vix_regime = min(100, 80 + ((vix_current - 30) / 10) * 20)
    
    # VIX acceleration
    vix_20d_avg = vix.rolling(20).mean().iloc[-1]
    vix_spike_rate = ((vix_current / vix_20d_avg) - 1) * 100 if vix_20d_avg > 0 else 0
    
    return 0.60 * vix_regime + 0.40 * min(100, max(0, vix_spike_rate * 2))

def calc_safe_haven_flows(gld: pd.Series, tlt: pd.Series, spy: pd.Series) -> float:
    """Safe-haven flows (40% of pillar)"""
    # Gold outperformance
    gold_30d = pct_change(gld, 30)
    spy_30d = pct_change(spy, 30)
    gold_premium = gold_30d - spy_30d
    
    # Treasury rally
    tlt_30d = pct_change(tlt, 30)
    
    # Composite
    gold_score = normalize(gold_premium, 50, 5)
    tlt_score = normalize(tlt_30d, 50, 3)
    
    return 0.60 * gold_score + 0.40 * tlt_score

def calc_energy_shocks(wti: pd.Series, brent: pd.Series, ng: pd.Series) -> float:
    """Energy supply shocks (30% of pillar)"""
    # Oil spike
    oil_current = wti.iloc[-1]
    oil_50d_ma = wti.rolling(50).mean().iloc[-1]
    oil_spike = ((oil_current / oil_50d_ma) - 1) * 100 if oil_50d_ma > 0 else 0
    
    # Brent-WTI spread anomaly
    spread = brent.iloc[-1] - oil_current
    spread_anomaly = abs(spread - 5.0)  # Normal ~$5
    
    # Natural gas volatility
    ng_vol_30d = volatility(ng, 30)
    ng_vol_252d = volatility(ng, 252)
    ng_vol_spike = ((ng_vol_30d / ng_vol_252d) - 1) * 100 if ng_vol_252d > 0 else 0
    
    # Composite
    return (
        0.40 * normalize(oil_spike, 50, 15) +
        0.30 * normalize(spread_anomaly, 0, 10) +
        0.30 * normalize(ng_vol_spike, 0, 50)
    )

def calc_military_conflict_index(data: Dict) -> float:
    """Pillar 2: Military Conflict Risk Index"""
    vix = calc_vix_fear(data['VIX'])
    safe_haven = calc_safe_haven_flows(data['GLD'], data['TLT'], data['SPY'])
    energy = calc_energy_shocks(data['WTI'], data['BRENT'], data['NG'])
    
    return 0.30 * vix + 0.40 * safe_haven + 0.30 * energy

# ============================================
# PILLAR 3: TRADE WAR ESCALATION (20%)
# ============================================

def calc_tariff_stress(xli: pd.Series, xly: pd.Series, xrt: pd.Series, xlv: pd.Series, xlu: pd.Series, spy: pd.Series) -> float:
    """Tariff-sensitive sector stress (35% of pillar)"""
    # Vulnerable sectors underperforming
    vulnerable_avg = (pct_change(xli, 30) + pct_change(xly, 30) + pct_change(xrt, 30)) / 3
    spy_30d = pct_change(spy, 30)
    vulnerable_underperformance = spy_30d - vulnerable_avg
    
    # Defensives outperforming
    defensive_avg = (pct_change(xlv, 30) + pct_change(xlu, 30)) / 2
    defensive_outperformance = defensive_avg - spy_30d
    
    # Composite tariff stress
    tariff_stress = 0.60 * vulnerable_underperformance + 0.40 * defensive_outperformance
    
    return normalize(tariff_stress, 50, 5)

def calc_us_china_divergence(spy: pd.Series, fxi: pd.Series) -> float:
    """US-China equity divergence (35% of pillar)"""
    # 60-day divergence
    spy_60d = pct_change(spy, 60)
    fxi_60d = pct_change(fxi, 60)
    divergence = spy_60d - fxi_60d
    
    # Normalize: >10% = significant trade war
    return normalize(divergence, 50, 10)

def calc_currency_war(cny: pd.Series) -> float:
    """CNY devaluation signal (30% of pillar)"""
    # CNY weakening (USDCNY rising)
    usdcny_current = cny.iloc[-1]
    baseline = 7.00  # Neutral level
    cny_devaluation = ((usdcny_current - baseline) / baseline) * 100
    
    # Volatility suppression (managed float indicator)
    cny_vol_30d = volatility(cny, 30)
    cny_vol_252d = volatility(cny, 252)
    vol_suppression = max(0, cny_vol_252d - cny_vol_30d)
    
    return (
        0.70 * normalize(cny_devaluation, 50, 5) +
        0.30 * normalize(vol_suppression, 0, 2)
    )

def calc_trade_war_index(data: Dict) -> float:
    """Pillar 3: Trade War Escalation Index"""
    tariff = calc_tariff_stress(data['XLI'], data['XLY'], data['XRT'], data['XLV'], data['XLU'], data['SPY'])
    us_china = calc_us_china_divergence(data['SPY'], data['FXI'])
    currency = calc_currency_war(data['CNY'])
    
    return 0.35 * tariff + 0.35 * us_china + 0.30 * currency

# ============================================
# PILLAR 4: FINANCIAL STRESS (15%)
# ============================================

def calc_yield_inversion(data: Dict) -> float:
    """Yield curve inversion (40% of pillar) - uses FRED if available"""
    try:
        from pandas_datareader import data as web
        end = datetime.now()
        start = end - timedelta(days=365)
        us10y = web.DataReader('DGS10', 'fred', start, end)['DGS10'].iloc[-1]
        us2y = web.DataReader('DGS2', 'fred', start, end)['DGS2'].iloc[-1]
        yield_spread = us10y - us2y
        
        # Inverted = high stress
        if yield_spread < 0:
            return min(100, abs(yield_spread / 1.0) * 100)
        else:
            return 0
    except:
        return 0  # No inversion detectable

def calc_equity_drawdown(spy: pd.Series) -> float:
    """Equity drawdown (35% of pillar)"""
    spy_current = spy.iloc[-1]
    spy_252d_high = spy.rolling(252).max().iloc[-1]
    drawdown_pct = ((spy_current / spy_252d_high) - 1) * 100
    
    # Convert to stress score: -20% = 100 stress
    return min(100, abs(drawdown_pct / 20) * 100)

def calc_pmi_contraction(xli: pd.Series, spy: pd.Series) -> float:
    """PMI contraction proxy (25% of pillar)"""
    # Industrial weakness vs market
    xli_60d = pct_change(xli, 60)
    spy_60d = pct_change(spy, 60)
    industrial_weakness = spy_60d - xli_60d
    
    return normalize(industrial_weakness, 50, 10)

def calc_financial_stress_index(data: Dict) -> float:
    """Pillar 4: Financial Stress Indicator"""
    inversion = calc_yield_inversion(data)
    drawdown = calc_equity_drawdown(data['SPY'])
    pmi = calc_pmi_contraction(data['XLI'], data['SPY'])
    
    return 0.40 * inversion + 0.35 * drawdown + 0.25 * pmi

# ============================================
# PILLAR 5: ENERGY SECURITY RISK (10%)
# ============================================

def calc_european_stress(ewg: pd.Series, spy: pd.Series) -> float:
    """European energy stress (40% of pillar)"""
    # Germany underperformance = EU energy vulnerability
    ewg_30d = pct_change(ewg, 30)
    spy_30d = pct_change(spy, 30)
    eu_stress = spy_30d - ewg_30d
    
    return normalize(eu_stress, 50, 5)

def calc_oil_chokepoint(wti: pd.Series, brent: pd.Series) -> float:
    """Oil chokepoint premium (35% of pillar)"""
    # Brent-WTI spread anomaly
    spread = brent.iloc[-1] - wti.iloc[-1]
    spread_anomaly = abs(spread - 4.0)  # Normal $4 premium
    
    return min(100, (spread_anomaly / 8.0) * 100)

def calc_energy_volatility(xle: pd.Series) -> float:
    """Energy volatility (25% of pillar)"""
    xle_vol_30d = volatility(xle, 30)
    xle_vol_252d = volatility(xle, 252)
    vol_spike = ((xle_vol_30d / xle_vol_252d) - 1) * 100 if xle_vol_252d > 0 else 0
    
    return normalize(vol_spike, 0, 50)

def calc_energy_security_index(data: Dict) -> float:
    """Pillar 5: Energy Security Risk"""
    eu = calc_european_stress(data['EWG'], data['SPY'])
    chokepoint = calc_oil_chokepoint(data['WTI'], data['BRENT'])
    energy_vol = calc_energy_volatility(data['XLE'])
    
    return 0.40 * eu + 0.35 * chokepoint + 0.25 * energy_vol

# ============================================
# MAIN CALCULATION
# ============================================

def calculate_geopolitical_framework(data: Optional[Dict] = None) -> Dict:
    """
    Main function: Calculate complete geopolitical framework
    Returns comprehensive geopolitical risk result dictionary
    """
    # Fetch data if not provided
    if data is None:
        print("Fetching geopolitical data...")
        data = fetch_geo_data()
    
    if not data:
        print(" Warning: No data available")
        return {
            'risk_score': 50.0,
            'risk_level': 'UNKNOWN',
            'pillars': {}
        }
    
    print(" Calculating geopolitical risk...")
    
    # Calculate all five pillars
    us_china_taiwan = calc_us_china_taiwan_index(data)
    military_conflict = calc_military_conflict_index(data)
    trade_war = calc_trade_war_index(data)
    financial_stress = calc_financial_stress_index(data)
    energy_security = calc_energy_security_index(data)
    
    # Calculate weighted composite risk score
    risk_score = (
        0.30 * us_china_taiwan +
        0.25 * military_conflict +
        0.20 * trade_war +
        0.15 * financial_stress +
        0.10 * energy_security
    )
    
    # Classify risk level
    if risk_score < 20:
        risk_level = "LOW"
    elif risk_score < 35:
        risk_level = "MODERATE"
    elif risk_score < 50:
        risk_level = "HIGH"
    elif risk_score < 65:
        risk_level = "SEVERE"
    else:
        risk_level = "EXTREME"
    
    return {
        'risk_score': round(risk_score, 2),
        'risk_level': risk_level,
        'pillars': {
            'us_china_taiwan': round(us_china_taiwan, 2),
            'military_conflict': round(military_conflict, 2),
            'trade_war': round(trade_war, 2),
            'financial_stress': round(financial_stress, 2),
            'energy_security': round(energy_security, 2)
        }
    }

# ============================================
# EXECUTION
# ============================================

if __name__ == "__main__":
    # Calculate geopolitical framework
    result = calculate_geopolitical_framework()
    
    print("\n" + "="*60)
    print(" GEOPOLITICAL RISK OVERLAY - RESULTS")
    print("="*60)
    print(f"\nRISK SCORE: {result['risk_score']:.2f}/100")
    print(f" RISK LEVEL: {result['risk_level']}")
    print(f"\nPILLAR BREAKDOWN:")
    print(f"   • US-China-Taiwan:  {result['pillars']['us_china_taiwan']:>6.2f}/100 (weight: 30%)")
    print(f"   • Military Conflict:{result['pillars']['military_conflict']:>6.2f}/100 (weight: 25%)")
    print(f"   • Trade War:        {result['pillars']['trade_war']:>6.2f}/100 (weight: 20%)")
    print(f"   • Financial Stress: {result['pillars']['financial_stress']:>6.2f}/100 (weight: 15%)")
    print(f"   • Energy Security:  {result['pillars']['energy_security']:>6.2f}/100 (weight: 10%)")
    print("\n" + "="*60 + "\n")

