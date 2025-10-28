#!/usr/bin/env python3
"""
Scoring, Weighting & Ranking System - Modified
Integrates Risk Component (30/30/20/20), ML Ensemble, Volume Intelligence, and Kalman Hull
"""

import numpy as np
from typing import Dict, List, Tuple

class ScoringRankingSystem:
    """Optimized scoring system with new components"""
    
    def __init__(self):
        # Component weights: Risk(40%), Technical(30%), ML+Volume(30%)
        self.weights = {
            'risk': 0.40,        # Risk Component (CVaR, Ulcer, Beta, IR)
            'technical': 0.30,   # Kalman Hull
            'ml_volume': 0.30    # ML Ensemble + Volume Intelligence
        }
        
        # Risk adjustments by category
        self.risk_multipliers = {
            'LOW': [1.1, 0.9, 0.8],      # [risk, technical, ml_volume]
            'MEDIUM': [1.0, 1.0, 1.0],
            'HIGH': [0.9, 1.1, 1.2]
        }
    
    def score_risk_component(self, risk_score: float) -> float:
        """Score Risk Component (0-100) - inverted since lower risk_score = better"""
        if np.isnan(risk_score): return 50.0
        # risk_score is 0-1, where 0=best, 1=worst
        return max(0.0, min(100.0, (1.0 - risk_score) * 100.0))
    
    def score_kalman_hull(self, trend: int, signal_strength: float, divergence: str) -> float:
        """Score Kalman Hull indicator (0-100)"""
        if np.isnan(signal_strength): return 50.0
        
        # Base score from trend direction
        if trend == 1:  # Uptrend
            base_score = 70.0
        elif trend == -1:  # Downtrend
            base_score = 30.0
        else:  # Neutral
            base_score = 50.0
        
        # Adjust by signal strength (0-1)
        strength_adj = (signal_strength - 0.5) * 40.0  # ±20 points
        
        # Adjust by divergence
        div_adj = 0.0
        if divergence == 'bullish':
            div_adj = 10.0
        elif divergence == 'bearish':
            div_adj = -10.0
        
        return max(0.0, min(100.0, base_score + strength_adj + div_adj))
    
    def score_ml_forecast(self, forecast: float, confidence: float) -> float:
        """Score ML Forecast with confidence weighting (0-100)"""
        if np.isnan(forecast) or np.isnan(confidence): return 50.0
        
        # Clamp forecast to ±15%
        forecast = max(-0.15, min(0.15, forecast / 100.0))  # Convert from % to decimal
        
        # Base score: 50 ± (forecast * 333.33) to map ±15% to ±50 points
        base_score = 50.0 + forecast * 333.33
        
        # Weight by confidence
        return max(0.0, min(100.0, 50.0 + (base_score - 50.0) * confidence))
    
    def score_volume_intelligence(self, spike_score: float, correlation: float, ad_signal: str) -> float:
        """Score Volume Intelligence (0-100)"""
        if np.isnan(spike_score): return 50.0
        
        # Base from spike score (0-100 already)
        base = spike_score * 0.4  # 40% weight
        
        # Correlation component (30% weight)
        if not np.isnan(correlation):
            # Positive correlation is good (price + volume up together)
            corr_score = (correlation + 1.0) * 50.0  # Map [-1,1] to [0,100]
            base += corr_score * 0.3
        else:
            base += 50.0 * 0.3
        
        # A/D signal component (30% weight)
        ad_scores = {'accumulation': 70.0, 'neutral': 50.0, 'distribution': 30.0}
        base += ad_scores.get(ad_signal, 50.0) * 0.3
        
        return max(0.0, min(100.0, base))
    
    def calculate_component_scores(self, analysis: Dict, risk_category: str) -> Tuple[float, float, float]:
        """Calculate all component scores"""
        # Risk Component score
        risk_score = self.score_risk_component(analysis.get('risk_score', np.nan))
        
        # Technical score (Kalman Hull)
        tech_score = self.score_kalman_hull(
            analysis.get('kalman_trend', 0),
            analysis.get('kalman_signal_strength', 0.5),
            analysis.get('kalman_divergence', 'none')
        )
        
        # ML + Volume score (combined)
        ml_score = self.score_ml_forecast(
            analysis.get('ml_forecast', np.nan),
            analysis.get('ml_confidence', 0.5)
        )
        vol_score = self.score_volume_intelligence(
            analysis.get('volume_spike_score', np.nan),
            analysis.get('volume_correlation', np.nan),
            analysis.get('volume_ad_signal', 'neutral')
        )
        ml_volume_score = (ml_score * 0.6 + vol_score * 0.4)  # 60% ML, 40% Volume
        
        return risk_score, tech_score, ml_volume_score
    
    def calculate_composite_score(self, analysis: Dict, risk_category: str) -> float:
        """Calculate final composite score with risk adjustments"""
        risk_score, tech_score, ml_volume_score = self.calculate_component_scores(analysis, risk_category)
        
        # Apply risk category multipliers
        multipliers = self.risk_multipliers.get(risk_category, [1.0, 1.0, 1.0])
        risk_score *= multipliers[0]
        tech_score *= multipliers[1]
        ml_volume_score *= multipliers[2]
        
        # Weighted composite score
        composite = (
            risk_score * self.weights['risk'] +
            tech_score * self.weights['technical'] +
            ml_volume_score * self.weights['ml_volume']
        )
        
        # Apply quality adjustments
        young_penalty = analysis.get('young_etf_penalty', 0.0)
        composite *= (1 - young_penalty)
        
        # Apply graduated, percentage-based penalties
        composite = self.apply_risk_adjusted_penalties(composite, analysis)
        
        return max(0.0, min(100.0, composite))
    
    def apply_risk_adjusted_penalties(self, composite: float, analysis: Dict) -> float:
        """
        Apply graduated, proportional penalties based on risk and liquidity
        Uses percentage-based penalties instead of fixed deductions
        """
        # CVaR penalty (graduated)
        cvar = analysis.get('cvar', 0.0)
        if not np.isnan(cvar):
            if cvar < -0.25:  # Extreme tail risk (< -25%)
                composite *= 0.75  # 25% penalty
            elif cvar < -0.15:  # High tail risk (-15% to -25%)
                composite *= 0.90  # 10% penalty
            elif cvar < -0.10:  # Moderate tail risk (-10% to -15%)
                composite *= 0.95  # 5% penalty
        
        # Liquidity penalty (graduated)
        avg_daily_volume = analysis.get('avg_daily_volume', np.nan)
        if not np.isnan(avg_daily_volume):
            if avg_daily_volume < 100_000:  # Very low liquidity
                composite *= 0.70  # 30% penalty
            elif avg_daily_volume < 500_000:  # Low liquidity
                composite *= 0.85  # 15% penalty
            elif avg_daily_volume < 1_000_000:  # Moderate liquidity
                composite *= 0.95  # 5% penalty
        
        # Amihud illiquidity penalty
        amihud = analysis.get('amihud', np.nan)
        if not np.isnan(amihud):
            if amihud > 5.0:  # Very illiquid
                composite *= 0.85  # 15% penalty
            elif amihud > 2.0:  # Illiquid
                composite *= 0.95  # 5% penalty
        
        # Zero volume days penalty
        zero_volume_days = analysis.get('zero_volume_days', 0)
        if zero_volume_days > 20:  # Frequent zero volume
            composite *= 0.80  # 20% penalty
        elif zero_volume_days > 10:
            composite *= 0.90  # 10% penalty
        elif zero_volume_days > 5:
            composite *= 0.95  # 5% penalty
        
        # Expense ratio penalty (graduated)
        expense_ratio = analysis.get('expense_ratio', np.nan)
        if not np.isnan(expense_ratio):
            if expense_ratio > 0.0100:  # > 1.00% (very high)
                composite *= 0.75  # 25% penalty
            elif expense_ratio > 0.0075:  # 0.75-1.00% (high)
                composite *= 0.85  # 15% penalty
            elif expense_ratio > 0.0050:  # 0.50-0.75% (above average)
                composite *= 0.95  # 5% penalty
        
        # AUM penalty (small ETFs have higher risk)
        aum_aud = analysis.get('aum_aud', np.nan)
        if not np.isnan(aum_aud):
            if aum_aud < 25_000_000:  # < 25M (very small)
                composite *= 0.75  # 25% penalty
            elif aum_aud < 50_000_000:  # 25-50M (small)
                composite *= 0.85  # 15% penalty
            elif aum_aud < 100_000_000:  # 50-100M (below average)
                composite *= 0.95  # 5% penalty
        
        return max(0.0, min(100.0, composite))
    
    def rank_etfs_by_category(self, analysis_results: Dict, risk_classifications: Dict) -> Dict:
        """Rank ETFs within their risk categories"""
        category_groups = {'LOW': [], 'MEDIUM': [], 'HIGH': []}
        
        for ticker, analysis in analysis_results.items():
            category = risk_classifications.get(ticker, 'MEDIUM')
            if category not in category_groups: category = 'MEDIUM'
            
            score = self.calculate_composite_score(analysis, category)
            category_groups[category].append((ticker, score))
        
        # Sort each category by score (descending)
        return {cat: sorted(etfs, key=lambda x: x[1], reverse=True) 
                for cat, etfs in category_groups.items()}
    
    def get_top_etfs(self, rankings: Dict, top_n: int = 10) -> List[Dict]:
        """Get top N ETFs across all categories"""
        all_etfs = []
        for category, etf_list in rankings.items():
            for ticker, score in etf_list:
                all_etfs.append({'ticker': ticker, 'score': score, 'category': category})
        
        return sorted(all_etfs, key=lambda x: x['score'], reverse=True)[:top_n]


def main():
    """Example usage"""
    sample_analysis = {
        'VAF.AX': {
            'risk_score': 0.3,  # Risk Component output
            'kalman_trend': 1, 'kalman_signal_strength': 0.7, 'kalman_divergence': 'bullish',
            'ml_forecast': 2.5, 'ml_confidence': 0.75,
            'volume_spike_score': 65.0, 'volume_correlation': 0.6, 'volume_ad_signal': 'accumulation',
            'young_etf_penalty': 0.0, 'cvar': -0.03
        }
    }
    
    scoring_system = ScoringRankingSystem()
    
    print("ETF Scoring System - Modified")
    print("=" * 40)
    
    for ticker, analysis in sample_analysis.items():
        score = scoring_system.calculate_composite_score(analysis, 'LOW')
        print(f"{ticker}: {score:.1f}/100")


if __name__ == "__main__":
    main()

