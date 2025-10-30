#!/usr/bin/env python3
"""
Growth-Optimized Scoring System
Reweighted for capital growth focus (MEDIUM/HIGH risk emphasis)
"""

import numpy as np
from typing import Dict, List, Tuple

class GrowthScoringSystem:
    """
    Optimized scoring for growth-focused trading strategy
    Emphasizes momentum and forecast over risk aversion
    """
    
    def __init__(self):
        # REWEIGHTED: Momentum(35%), Forecast(25%), Risk(25%), Volume(15%)
        # Shifted from Risk(40%), Technical(30%), ML+Volume(30%)
        self.weights = {
            'momentum': 0.35,     # Kalman Hull momentum (↑ from 30%)
            'forecast': 0.25,     # ML Ensemble directional bias (↑ from 18%)
            'risk': 0.25,         # Risk Component (↓ from 40%)
            'volume': 0.15        # Volume Intelligence (↓ from 12%)
        }
        
        # Risk category adjustments (MORE aggressive for HIGH risk)
        self.risk_multipliers = {
            'LOW': {
                'momentum': 0.8,   # Suppress momentum signals
                'forecast': 0.9,   # Reduce forecast weight
                'risk': 1.3,       # Emphasize risk quality
                'volume': 1.0
            },
            'MEDIUM': {
                'momentum': 1.0,   # Neutral
                'forecast': 1.0,
                'risk': 1.0,
                'volume': 1.0
            },
            'HIGH': {
                'momentum': 1.3,   # BOOST momentum signals (growth focus)
                'forecast': 1.2,   # BOOST forecast importance
                'risk': 0.7,       # Reduce risk weight (accept volatility)
                'volume': 1.1      # Slightly boost volume confirmation
            }
        }
    
    def score_risk_component(self, risk_score: float) -> float:
        """Score Risk Component (0-100) - inverted since lower risk_score = better"""
        if np.isnan(risk_score): return 50.0
        return max(0.0, min(100.0, (1.0 - risk_score) * 100.0))
    
    def score_momentum(self, trend: int, signal_strength: float, efficiency_ratio: float, divergence: str) -> float:
        """
        Score momentum using Kalman Hull + efficiency ratio
        This is the PRIMARY signal for growth trading
        """
        if np.isnan(signal_strength): 
            signal_strength = 0.5
        if np.isnan(efficiency_ratio):
            efficiency_ratio = 0.5
        
        # Base score from trend direction
        if trend == 1:  # Uptrend
            base_score = 75.0  # Higher base (was 70)
        elif trend == -1:  # Downtrend
            base_score = 25.0  # Lower base (was 30)
        else:  # Neutral
            base_score = 45.0  # Slightly bearish neutral (was 50)
        
        # MOMENTUM SCORE: Combine signal_strength + efficiency_ratio
        momentum_score = (signal_strength * 0.6) + (efficiency_ratio * 0.4)
        
        # Adjust by momentum strength (more aggressive than original)
        if momentum_score > 0.75:
            strength_adj = 25.0   # Strong momentum (was ±20)
        elif momentum_score > 0.60:
            strength_adj = 15.0
        elif momentum_score > 0.45:
            strength_adj = 5.0
        elif momentum_score > 0.30:
            strength_adj = -10.0  # Weak momentum = penalty
        else:
            strength_adj = -20.0  # Very weak = large penalty
        
        # Adjust by divergence
        div_adj = 0.0
        if divergence == 'bullish':
            div_adj = 15.0  # Stronger bullish signal (was 10)
        elif divergence == 'bearish':
            div_adj = -15.0  # Stronger bearish warning (was -10)
        
        return max(0.0, min(100.0, base_score + strength_adj + div_adj))
    
    def score_forecast(self, forecast: float, confidence: float, hit_rate: float) -> float:
        """
        Score ML Forecast with confidence + hit rate weighting
        Uses walk-forward validation hit rate for reliability
        """
        if np.isnan(forecast): 
            forecast = 0.0
        if np.isnan(confidence): 
            confidence = 0.5
        if np.isnan(hit_rate):
            hit_rate = 0.5
        
        # Clamp forecast to ±15%
        forecast = max(-0.15, min(0.15, forecast / 100.0))
        
        # Base score: 50 ± (forecast * 333.33) to map ±15% to ±50 points
        base_score = 50.0 + forecast * 333.33
        
        # Reliability factor: Average of confidence and hit_rate
        reliability = (confidence * 0.6) + (hit_rate * 0.4)
        
        # Weight by reliability (more aggressive than original)
        final_score = 50.0 + (base_score - 50.0) * reliability
        
        # Bonus for high-confidence + high hit rate forecasts
        if reliability > 0.7 and abs(forecast) > 0.05:  # Strong reliable signal
            final_score += 5.0 if forecast > 0 else -5.0
        
        return max(0.0, min(100.0, final_score))
    
    def score_volume(self, spike_score: float, correlation: float, ad_signal: str) -> float:
        """Score Volume Intelligence (0-100) - confirmation signal"""
        if np.isnan(spike_score): 
            spike_score = 50.0
        
        # Base from spike score (40% weight)
        base = spike_score * 0.4
        
        # Correlation component (30% weight)
        if not np.isnan(correlation):
            corr_score = (correlation + 1.0) * 50.0
            base += corr_score * 0.3
        else:
            base += 50.0 * 0.3
        
        # A/D signal component (30% weight) - MORE aggressive scores
        ad_scores = {
            'accumulation': 80.0,   # Strong buy signal (was 70)
            'neutral': 50.0,
            'distribution': 20.0    # Strong sell signal (was 30)
        }
        base += ad_scores.get(ad_signal, 50.0) * 0.3
        
        return max(0.0, min(100.0, base))
    
    def calculate_component_scores(self, analysis: Dict, risk_category: str) -> Dict[str, float]:
        """Calculate all component scores and return as dict"""
        # Risk Component
        risk_score = self.score_risk_component(analysis.get('risk_score', np.nan))
        
        # Momentum (Kalman Hull + efficiency ratio)
        momentum_score = self.score_momentum(
            analysis.get('kalman_trend', 0),
            analysis.get('kalman_signal_strength', 0.5),
            analysis.get('kalman_efficiency_ratio', 0.5),
            analysis.get('kalman_divergence', 'none')
        )
        
        # Forecast (ML with hit rate)
        forecast_score = self.score_forecast(
            analysis.get('ml_forecast', np.nan),
            analysis.get('ml_confidence', 0.5),
            analysis.get('hit_rate', 0.5)
        )
        
        # Volume
        volume_score = self.score_volume(
            analysis.get('volume_spike_score', np.nan),
            analysis.get('volume_correlation', np.nan),
            analysis.get('volume_ad_signal', 'neutral')
        )
        
        return {
            'risk': risk_score,
            'momentum': momentum_score,
            'forecast': forecast_score,
            'volume': volume_score
        }
    
    def calculate_position_size(self, component_scores: Dict, risk_category: str, 
                                signal_strength: float, efficiency_ratio: float) -> float:
        """
        Calculate recommended position size based on signal quality
        Returns: 0.0 - 1.0 multiplier
        """
        # Base position sizes by risk category
        base_sizes = {
            'LOW': 0.15,      # 15% max (conservative)
            'MEDIUM': 0.12,   # 12% max (moderate)
            'HIGH': 0.08      # 8% max (aggressive but smaller due to volatility)
        }
        base = base_sizes.get(risk_category, 0.10)
        
        # Momentum quality multiplier
        momentum_quality = (signal_strength * 0.6) + (efficiency_ratio * 0.4)
        
        if momentum_quality > 0.75:
            multiplier = 1.0      # Full position (strong momentum)
        elif momentum_quality > 0.60:
            multiplier = 0.85     # 85% position
        elif momentum_quality > 0.45:
            multiplier = 0.65     # 65% position
        elif momentum_quality > 0.30:
            multiplier = 0.40     # 40% position (weak)
        else:
            multiplier = 0.20     # 20% position (very weak, consider exit)
        
        return base * multiplier
    
    def calculate_composite_score(self, analysis: Dict, risk_category: str) -> Dict:
        """
        Calculate final composite score with growth adjustments
        Returns dict with score, components, and position size
        """
        # Get component scores
        components = self.calculate_component_scores(analysis, risk_category)
        
        # Apply risk category multipliers
        multipliers = self.risk_multipliers.get(risk_category, self.risk_multipliers['MEDIUM'])
        adjusted_components = {
            key: components[key] * multipliers[key] 
            for key in components.keys()
        }
        
        # Weighted composite score
        composite = sum(
            adjusted_components[key] * self.weights[key]
            for key in self.weights.keys()
        )
        
        # Apply quality penalties (LESS aggressive than original for HIGH risk)
        composite = self.apply_growth_penalties(composite, analysis, risk_category)
        
        # Calculate position size
        position_size = self.calculate_position_size(
            components,
            risk_category,
            analysis.get('kalman_signal_strength', 0.5),
            analysis.get('kalman_efficiency_ratio', 0.5)
        )
        
        return {
            'composite_score': max(0.0, min(100.0, composite)),
            'components': components,
            'adjusted_components': adjusted_components,
            'position_size': position_size,
            'risk_category': risk_category
        }
    
    def apply_growth_penalties(self, composite: float, analysis: Dict, risk_category: str) -> float:
        """
        Apply additive penalties with maximum caps - prevents over-penalization
        """
        # Penalty scaling by risk category (less aggressive for HIGH risk)
        penalty_scales = {
            'LOW': 1.0,      # Full penalties
            'MEDIUM': 0.7,   # 70% of penalties
            'HIGH': 0.4      # 40% of penalties (accept risk for growth)
        }
        scale = penalty_scales.get(risk_category, 1.0)
        
        penalty_points = 0.0  # Additive penalty system
        
        # CVaR penalty (additive)
        cvar = analysis.get('cvar', 0.0)
        if not np.isnan(cvar):
            if cvar < -0.50:  # Extreme tail risk (< -50%)
                penalty_points += 20 * scale  # 20 points
            elif cvar < -0.30:  # High tail risk (-30% to -50%)
                penalty_points += 15 * scale  # 15 points
            elif cvar < -0.20:  # Moderate tail risk (-20% to -30%)
                penalty_points += 10 * scale  # 10 points
        
        # Liquidity penalty (additive)
        avg_daily_volume = analysis.get('avg_daily_volume', np.nan)
        if not np.isnan(avg_daily_volume):
            if avg_daily_volume < 50_000:  # Very low liquidity
                penalty_points += 15 * scale  # 15 points
            elif avg_daily_volume < 200_000:  # Low liquidity
                penalty_points += 10 * scale  # 10 points
            elif avg_daily_volume < 500_000:  # Moderate liquidity
                penalty_points += 5 * scale   # 5 points
        
        # Amihud penalty (additive)
        amihud = analysis.get('amihud', np.nan)
        if not np.isnan(amihud):
            if amihud > 10.0:  # Very illiquid
                penalty_points += 10 * scale  # 10 points
            elif amihud > 5.0:  # Illiquid
                penalty_points += 5 * scale   # 5 points
        
        # Zero volume days penalty (additive)
        zero_volume_days = analysis.get('zero_volume_days', 0)
        if zero_volume_days > 25:  # Frequent zero volume
            penalty_points += 15 * scale  # 15 points
        elif zero_volume_days > 15:
            penalty_points += 10 * scale  # 10 points
        elif zero_volume_days > 8:
            penalty_points += 5 * scale   # 5 points
        
        # Expense ratio penalty (additive)
        expense_ratio = analysis.get('expense_ratio', np.nan)
        if not np.isnan(expense_ratio):
            if expense_ratio > 0.0100:  # > 1.00% (very high)
                penalty_points += 15 * scale  # 15 points
            elif expense_ratio > 0.0075:  # 0.75-1.00% (high)
                penalty_points += 10 * scale  # 10 points
            elif expense_ratio > 0.0050:  # 0.50-0.75% (above average)
                penalty_points += 5 * scale   # 5 points
        
        # AUM penalty (additive)
        aum_aud = analysis.get('aum_aud', np.nan)
        if not np.isnan(aum_aud):
            if aum_aud < 25_000_000:  # < 25M (very small)
                penalty_points += 15 * scale  # 15 points
            elif aum_aud < 50_000_000:  # 25-50M (small)
                penalty_points += 10 * scale  # 10 points
            elif aum_aud < 100_000_000:  # 50-100M (below average)
                penalty_points += 5 * scale   # 5 points
        
        # Cap total penalties at 30 points maximum (prevents over-penalization)
        penalty_points = min(penalty_points, 30.0)
        
        # Apply as percentage reduction from composite score
        penalty_factor = 1.0 - (penalty_points / 100.0)
        return max(0.0, composite * penalty_factor)
    
    def rank_etfs_by_category(self, analysis_results: Dict, risk_classifications: Dict) -> Dict:
        """Rank ETFs within their risk categories"""
        category_groups = {'LOW': [], 'MEDIUM': [], 'HIGH': []}
        
        for ticker, analysis in analysis_results.items():
            category = risk_classifications.get(ticker, 'MEDIUM')
            if category not in category_groups:
                category = 'MEDIUM'
            
            result = self.calculate_composite_score(analysis, category)
            category_groups[category].append((ticker, result))
        
        # Sort by composite_score (descending)
        return {
            cat: sorted(etfs, key=lambda x: x[1]['composite_score'], reverse=True) 
            for cat, etfs in category_groups.items()
        }
    
    def get_top_opportunities(self, rankings: Dict, top_n: int = 20, 
                            min_score: float = 60.0, focus_categories: List[str] = None) -> List[Dict]:
        """
        Get top opportunities for growth strategy
        
        Args:
            rankings: Output from rank_etfs_by_category
            top_n: Max number to return
            min_score: Minimum composite score threshold
            focus_categories: ['MEDIUM', 'HIGH'] for growth focus
        """
        if focus_categories is None:
            focus_categories = ['MEDIUM', 'HIGH']  # Default to growth categories
        
        opportunities = []
        for category in focus_categories:
            if category not in rankings:
                continue
            
            for ticker, result in rankings[category]:
                score = result['composite_score']
                if score >= min_score:
                    opportunities.append({
                        'ticker': ticker,
                        'composite_score': score,
                        'risk_category': category,
                        'position_size': result['position_size'],
                        'momentum_score': result['components']['momentum'],
                        'forecast_score': result['components']['forecast'],
                        'volume_score': result['components']['volume'],
                        'risk_score': result['components']['risk']
                    })
        
        # Sort by composite score
        opportunities.sort(key=lambda x: x['composite_score'], reverse=True)
        return opportunities[:top_n]


def main():
    """Example usage"""
    sample_analysis = {
        'HACK.AX': {  # High risk tech ETF
            'risk_score': 0.6,
            'kalman_trend': 1,
            'kalman_signal_strength': 0.78,
            'kalman_efficiency_ratio': 0.62,
            'kalman_divergence': 'bullish',
            'ml_forecast': 8.5,
            'ml_confidence': 0.65,
            'hit_rate': 0.72,
            'volume_spike_score': 68.0,
            'volume_correlation': 0.67,
            'volume_ad_signal': 'accumulation',
            'cvar': -0.28,
            'avg_daily_volume': 1_500_000
        },
        'VAS.AX': {  # Low risk broad market
            'risk_score': 0.25,
            'kalman_trend': 1,
            'kalman_signal_strength': 0.55,
            'kalman_efficiency_ratio': 0.48,
            'kalman_divergence': 'none',
            'ml_forecast': 3.2,
            'ml_confidence': 0.58,
            'hit_rate': 0.63,
            'volume_spike_score': 52.0,
            'volume_correlation': 0.45,
            'volume_ad_signal': 'neutral',
            'cvar': -0.08,
            'avg_daily_volume': 8_000_000
        }
    }
    
    scoring = GrowthScoringSystem()
    
    print("Growth-Optimized Scoring System")
    print("=" * 60)
    print(f"Weights: Momentum({scoring.weights['momentum']:.0%}), "
          f"Forecast({scoring.weights['forecast']:.0%}), "
          f"Risk({scoring.weights['risk']:.0%}), "
          f"Volume({scoring.weights['volume']:.0%})\n")
    
    for ticker, analysis in sample_analysis.items():
        # Determine risk category
        category = 'HIGH' if 'HACK' in ticker else 'LOW'
        
        # Calculate score
        result = scoring.calculate_composite_score(analysis, category)
        
        print(f"\n{ticker} ({category} Risk)")
        print(f"  Composite Score: {result['composite_score']:.1f}/100")
        print(f"  Position Size: {result['position_size']*100:.1f}%")
        print(f"  Components:")
        print(f"    - Momentum: {result['components']['momentum']:.1f}")
        print(f"    - Forecast: {result['components']['forecast']:.1f}")
        print(f"    - Risk: {result['components']['risk']:.1f}")
        print(f"    - Volume: {result['components']['volume']:.1f}")


if __name__ == "__main__":
    main()

