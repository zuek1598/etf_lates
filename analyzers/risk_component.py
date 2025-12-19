#!/usr/bin/env python3
"""
Risk Component - Statistical and Risk Metrics Analysis
Calculates CVaR (30%), Ulcer Index (30%), Beta (20%), Information Ratio (20%)
with new weighting system for comprehensive risk assessment
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from utilities.shared_utils import extract_column, transform_to_returns

class RiskComponent:
    """Risk calculation - VALIDATED FACTORS ONLY (CVaR only)"""
    
    def __init__(self):
        self.risk_free_rate = 0.0435  # RBA rate Oct 2024
        self.group_params = {
            'Australian_Equity': [60, 100, 60],
            'International_Equity': [60, 80, 40],
            'Bonds': [90, 100, 60],
            'Commodities': [30, 60, 30]
        }
        # VALIDATED FACTORS ONLY: Only CVaR was statistically significant
        # Other risk metrics were rejected (p > 0.05 or negative IC)
        self.validated_only = True
    
    def fit_t_distribution(self, returns: pd.Series) -> Dict:
        """Fit t-distribution to return series with edge case handling"""
        if len(returns) < 30:
            return {'degrees_of_freedom': 2.1, 'location': 0.0, 'scale': 0.02}
        
        try:
            clean_returns = returns.dropna()
            n = len(clean_returns)
            t_params = stats.t.fit(clean_returns)
            df = t_params[0]
            
            if df <= 2.0:
                df = 2.1
            
            if n < 100:
                df = df * (n - 1) / max(n - 3, 1)
                df = max(df, 2.1)
            
            return {'degrees_of_freedom': df, 'location': t_params[1], 'scale': t_params[2]}
        except:
            return {'degrees_of_freedom': 5.0, 'location': returns.mean(), 'scale': returns.std()}
    
    def calculate_cvar(self, returns: pd.Series, t_params: Dict, confidence: float = 0.95) -> float:
        """
        Calculate CVaR (Conditional Value-at-Risk) using parametric t-distribution
        
        CVaR = μ + σ * E[T | T < VaR] where T ~ t(df, μ, σ²)
        
        For standardized t-distribution:
        E[T | T < q] = -(df + q²) / ((df - 1) * α) * pdf(q)
        
        where q = t.ppf(α, df) is the VaR quantile
        """
        if len(returns) < 30:
            return np.nan
        
        clean_returns = returns.dropna()
        
        # Parametric CVaR using t-distribution
        df = t_params.get('degrees_of_freedom', 5.0)
        loc = t_params.get('location', 0.0)
        scale = t_params.get('scale', 0.02)
        
        # Ensure df > 2 for valid moments
        if df <= 2.0:
            df = 2.1
        
        alpha = 1 - confidence  # Tail probability (e.g., 0.01 for 99% confidence)
        
        # VaR quantile (negative, since we're looking at left tail losses)
        var_quantile = stats.t.ppf(alpha, df)
        
        # PDF at the VaR quantile
        pdf_at_quantile = stats.t.pdf(var_quantile, df)
        
        # Expected shortfall for standardized t-distribution
        # Formula from Wikipedia: ES = -(pdf(q) * (df + q²)) / ((df-1) * alpha)
        if pdf_at_quantile > 1e-10:  # Avoid division by zero
            standardized_es = -(pdf_at_quantile * (df + var_quantile**2)) / ((df - 1) * alpha)
        else:
            standardized_es = var_quantile  # Fallback to VaR if PDF too small
        
        # Transform back to original scale: CVaR = μ + σ * E[X]
        param_cvar = loc + scale * standardized_es
        
        # Annualize (assuming daily returns) - multiply by sqrt(252)
        param_cvar_annual = param_cvar * np.sqrt(252)
        
        return param_cvar_annual
    
    def classify_etf_group(self, etf_info: Dict) -> str:
        """Classify ETF into group"""
        etf_type = etf_info.get('type', 'unknown')
        region = etf_info.get('region', 'unknown')
        
        if etf_type == 'bond': return 'Bonds'
        if etf_type == 'commodity': return 'Commodities'
        if region == 'AUSTRALIA' and etf_type in ['broad_market', 'growth']: return 'Australian_Equity'
        return 'International_Equity'
    
    def calculate_conditional_sharpe(self, returns: pd.Series, vix_data: pd.Series, t_params: Dict) -> float:
        """Calculate conditional Sharpe ratio during market stress"""
        if len(returns) < 252 or len(vix_data) < 252: return np.nan
        
        try:
            common_dates = returns.index.intersection(vix_data.index)
            if len(common_dates) < 100: return np.nan
            
            aligned_returns = returns.loc[common_dates]
            aligned_vix = vix_data.loc[common_dates]
        except:
            min_len = min(len(returns), len(vix_data))
            if min_len < 100: return np.nan
            aligned_returns = returns.tail(min_len)
            aligned_vix = vix_data.tail(min_len)
        
        vix_threshold = np.percentile(aligned_vix, 85)
        stress_mask = aligned_vix > vix_threshold
        
        if stress_mask.sum() < 20: return np.nan
        
        stress_returns = aligned_returns[stress_mask]
        mean_return = stress_returns.mean()
        
        df = t_params.get('degrees_of_freedom', 5.0)
        scale = t_params.get('scale', 0.02)

        if df <= 2.1 or np.isnan(df):
            df = 5.0
        if scale <= 0.001 or np.isnan(scale):
            scale = 0.02
        
        if df > 2.1 and scale > 0:
            t_volatility = scale * np.sqrt(df / (df - 2)) * np.sqrt(252)
        else:
            t_volatility = stress_returns.std() * np.sqrt(252)
        
        if t_volatility <= 0 or np.isnan(t_volatility):
            return np.nan
        
        conditional_sharpe = (mean_return * 252 - self.risk_free_rate) / t_volatility
        
        return max(-3.0, min(3.0, conditional_sharpe))
    
    def normalize_metric(self, value: float, lower: float, upper: float) -> float:
        """Normalize metric to 0-1 scale for weighting"""
        if np.isnan(value):
            return np.nan
        return np.clip((value - lower) / (upper - lower), 0, 1)
    
    def scale_cvar(self, cvar: float) -> float:
        """
        Scale CVaR to [0, 1] where 1 is best (lowest risk)
        
        Typical CVaR ranges (annualized):
        - Excellent: > -0.05  (-5%, low tail risk)
        - Good: -0.05 to -0.10
        - Average: -0.10 to -0.15
        - Poor: -0.15 to -0.25
        - Very Poor: < -0.25 (-25%)
        """
        if np.isnan(cvar):
            return 0.5
        
        # CVaR is negative, more negative = worse
        # Map [-0.30, 0.0] to [0, 1]
        scaled = (cvar + 0.30) / 0.30
        return np.clip(scaled, 0, 1)
    
    def calculate_liquidity_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate liquidity metrics
        Returns: amihud ratio, avg_daily_volume, zero_volume_days
        """
        volume = extract_column(data, 'Volume')
        close = extract_column(data, 'Close')
        
        if volume is None or close is None:
            return {'amihud': np.nan, 'avg_daily_volume': np.nan, 'zero_volume_days': 0}
        
        # Average daily volume (last 60 days)
        avg_daily_volume = volume.tail(60).mean()
        
        # Zero volume days (last 60 days)
        zero_volume_days = int((volume.tail(60) == 0).sum())
        
        # Amihud illiquidity ratio: mean(|return| / dollar_volume)
        returns = close.pct_change().abs()
        dollar_volume = close * volume
        
        # Avoid division by zero
        valid_mask = dollar_volume > 0
        if valid_mask.sum() > 30:  # Need sufficient data
            amihud_ratios = returns[valid_mask] / dollar_volume[valid_mask]
            amihud = amihud_ratios.tail(60).mean() * 1e6  # Scale to millions
        else:
            amihud = np.nan
        
        return {
            'amihud': amihud,
            'avg_daily_volume': avg_daily_volume,
            'zero_volume_days': zero_volume_days
        }
    
    def calculate_risk_scores(self, etf_data: pd.DataFrame, etf_info: Dict, vix_data: pd.Series = None, 
                             benchmark_data: pd.Series = None, beta: float = np.nan) -> Dict:
        """
        Risk scoring - VALIDATED FACTORS ONLY (CVaR only when optimized)
        Normal mode: CVaR (30%), Ulcer (30%), Beta (20%), Information Ratio (20%)
        Validated mode: CVaR only (statistically significant factor)
        """
        close_col = extract_column(etf_data, 'Close')
        volume_col = extract_column(etf_data, 'Volume')
        returns = transform_to_returns(close_col)
        
        if len(returns) < 30:
            return {
                'cvar': np.nan,
                'ulcer_index': np.nan,
                'beta': np.nan,
                'information_ratio': np.nan,
                'risk_score': 0.5,
                'risk_category': 'UNKNOWN',
                'quality_flag': '[EMOJI]'
            }
        
        if getattr(self, 'validated_only', False):
            # OPTIMIZED MODE: Calculate only CVaR (validated factor)
            t_params = self.fit_t_distribution(returns)
            cvar = self.calculate_cvar(returns, t_params)
            
            return {
                'cvar': cvar,
                'ulcer_index': np.nan,  # Skipped - not validated
                'beta': np.nan,         # Skipped - not validated  
                'information_ratio': np.nan,  # Skipped - not validated
                'risk_score': 0.5,
                'risk_category': 'UNKNOWN',
                'quality_flag': '[EMOJI]'
            }
        
        # NORMAL MODE: Calculate all components (for backward compatibility)
        # Note: ulcer_index and information_ratio removed - only CVaR is validated
        t_params = self.fit_t_distribution(returns)
        etf_group = self.classify_etf_group(etf_info)
        
        cvar = self.calculate_cvar(returns, t_params)
        ulcer = np.nan  # Removed - not validated
        ir = np.nan    # Removed - not validated
        
        # Calculate liquidity metrics
        liquidity = self.calculate_liquidity_metrics(etf_data)
        
        # Use provided beta or default to 1.0
        if np.isnan(beta):
            beta = 1.0
        
        # Scale each metric using data-driven bounds (0-1 scale, 1 = best)
        cvar_scaled = self.scale_cvar(cvar)
        ulcer_scaled = self.scale_ulcer(ulcer)
        beta_scaled = self.scale_beta(beta)
        ir_scaled = self.scale_information_ratio(ir)
        
        # Calculate weighted risk score (lower = less risk)
        components_available = sum([
            not np.isnan(cvar),
            not np.isnan(ulcer),
            not np.isnan(beta),
            not np.isnan(ir)
        ])
        
        if components_available == 0:
            risk_score = 0.5
        else:
            total_weight = 0
            weighted_sum = 0
            if not np.isnan(cvar):
                weighted_sum += self.weights['cvar'] * cvar_scaled
                total_weight += self.weights['cvar']
            if not np.isnan(ulcer):
                weighted_sum += self.weights['ulcer'] * ulcer_scaled
                total_weight += self.weights['ulcer']
            if not np.isnan(beta):
                weighted_sum += self.weights['beta'] * beta_scaled
                total_weight += self.weights['beta']
            if not np.isnan(ir):
                weighted_sum += self.weights['ir'] * ir_scaled
                total_weight += self.weights['ir']
            
            risk_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Classify risk (based on overall score)
        if risk_score < 0.35:
            risk_category = 'LOW'
            quality_flag = '[EMOJI]'
        elif risk_score < 0.65:
            risk_category = 'MEDIUM'
            quality_flag = '~'
        else:
            risk_category = 'HIGH'
            quality_flag = '[EMOJI]'
        
        return {
            'cvar': cvar,
            'ulcer_index': ulcer,
            'beta': beta,
            'information_ratio': ir,
            'risk_score': risk_score,
            'risk_category': risk_category,
            'quality_flag': quality_flag,
            't_distribution_params': t_params,
            'group_classification': etf_group,
            'amihud': liquidity['amihud'],
            'avg_daily_volume': liquidity['avg_daily_volume'],
            'zero_volume_days': liquidity['zero_volume_days']
        }
    
    def analyze_etf(self, etf_data: pd.DataFrame, etf_info: Dict, vix_data: pd.Series = None, 
                    benchmark_data: pd.Series = None, beta: float = np.nan) -> Dict:
        """Comprehensive risk analysis for single ETF"""
        return self.calculate_risk_scores(etf_data, etf_info, vix_data, benchmark_data, beta)
    
    def analyze_risk_group(self, risk_group_etfs: Dict, vix_data: pd.Series = None, 
                          benchmark_data: Dict = None) -> Dict:
        """Analyze all ETFs in a risk group"""
        results = {}
        
        for ticker, etf_data in risk_group_etfs.items():
            data = etf_data['data']
            etf_info = etf_data.get('etf_info', {})
            
            benchmark_series = None
            if benchmark_data and etf_data.get('best_benchmark'):
                benchmark_df = benchmark_data.get(etf_data['best_benchmark'])
                if benchmark_df is not None and not benchmark_df.empty:
                    benchmark_prices = extract_column(benchmark_df, 'Close')
                    benchmark_series = transform_to_returns(benchmark_prices)
            
            beta = etf_data.get('beta', np.nan)
            analysis = self.analyze_etf(data, etf_info, vix_data, benchmark_series, beta)
            results[ticker] = analysis
        
        return results
