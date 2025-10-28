#!/usr/bin/env python3
"""
ETF Risk Classification System
Sophisticated volatility and beta analysis for ETF risk categorization
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import time
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from utilities.shared_utils import extract_column

class ETFRiskClassifier:
    """
    Advanced ETF risk classification using volatility and beta analysis
    """
    
    def __init__(self):
        from data_manager.data_manager import ETFDataManager
        self.etf_database = ETFDataManager()
        
        self.benchmarks = {
            'ASX200': '^AXJO',
            'MSCI_World': 'URTH', 
            'SP500': '^GSPC',
            'MSCI_EM': 'EEM',
            'Global_Bonds': 'BND',
            'NASDAQ': '^IXIC',
            'VIX': '^VIX',
            'DXY': 'DX-Y.NYB',
            'Gold': 'GLD'
        }
        
        # Risk classification thresholds
        self.LOW_VOL_THRESHOLD = 0.12
        self.HIGH_VOL_THRESHOLD = 0.22
        self.LOW_BETA_THRESHOLD = 0.8
        self.HIGH_BETA_THRESHOLD = 1.2
        
        # Data quality tiers
        self.QUALITY_TIERS = {
            'tier_1': 3.0,  # 3+ years
            'tier_2': 2.0,  # 2-3 years
            'tier_3': 1.0,  # 1-2 years
            'tier_4': 0.5   # 6m-1y
        }
        
        self.benchmark_data = {}
        self.cache = {}
        
    def download_benchmark_data(self) -> Dict[str, pd.DataFrame]:
        """Download all benchmark data for correlation analysis"""
        print("Downloading benchmark data...")
        
        for name, ticker in self.benchmarks.items():
            try:
                print(f"  Downloading {name} ({ticker})...")
                data = yf.download(ticker, period="max", progress=False)
                if not data.empty:
                    self.benchmark_data[name] = data
                    print(f"    ✓ {name}: {len(data)} days")
                else:
                    print(f"    ✗ {name}: No data")
            except Exception as e:
                print(f"    ✗ {name}: Error - {str(e)}")
                
        print(f"Successfully downloaded {len(self.benchmark_data)} benchmarks")
        return self.benchmark_data
    
    def download_etf_data(self, ticker: str, period: str = "max") -> Tuple[pd.DataFrame, str, float]:
        """
        Download ETF data with quality assessment and retry logic
        Returns: (data, quality_tier, quality_score)
        """
        max_retries = 3
        retry_delay = 1  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                # Check cache first
                cache_key = f"{ticker}_{period}"
                if cache_key in self.cache:
                    return self.cache[cache_key]
                
                data = yf.download(ticker, period=period, progress=False)
                
                if data is None or data.empty or len(data) < 30:  # Minimum 30 days
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  {ticker}: No data returned, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        print(f"  ❌ {ticker}: Insufficient data after {max_retries} attempts")
                        return None, "insufficient", 0.0
                
                # Calculate data quality
                years_available = len(data) / 252
                # Handle multi-level columns from yfinance
                close_col = extract_column(data, 'Close')
                completeness = (1 - close_col.isna().sum() / len(data))
                
                # Determine quality tier
                if years_available >= 3:
                    quality_tier = "tier_1"
                elif years_available >= 2:
                    quality_tier = "tier_2"
                elif years_available >= 1:
                    quality_tier = "tier_3"
                else:
                    quality_tier = "tier_4"
                
                quality_score = completeness * min(years_available / 3, 1.0)
                
                # Cache the result
                result = (data, quality_tier, quality_score)
                self.cache[cache_key] = result
                
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠️  {ticker}: Error ({str(e)}), retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"  ❌ {ticker}: Failed after {max_retries} attempts - {str(e)}")
                    return None, "error", 0.0
        
        return None, "error", 0.0
    
    def calculate_annual_volatility(self, data: pd.DataFrame, periods: int) -> float:
        """Calculate annualized volatility for given periods"""
        if len(data) < periods:
            return np.nan
            
        # Handle multi-level columns from yfinance
        close_col = extract_column(data, 'Close')
        returns = close_col.pct_change().dropna()
        if len(returns) < periods:
            return np.nan
            
        # Use last N periods
        recent_returns = returns.tail(periods)
        return recent_returns.std() * np.sqrt(252)
    
    def calculate_enhanced_volatility(self, data: pd.DataFrame, ticker: str = "Unknown") -> float:
        """
        Enhanced volatility calculation with simple 1-year approach and t-distribution
        Now with fallback for ETFs with <252 days
        """
        # Step 2A: Base Volatility (Simple 1-Year to match Beta approach)
        vol_1yr = self.calculate_annual_volatility(data, 252)

        # Handle missing data with fallback for newer ETFs
        if np.isnan(vol_1yr):
            # Fallback: use whatever data is available (minimum 90 days)
            close_col = extract_column(data, 'Close')
            returns = close_col.pct_change().dropna()
            
            if len(returns) >= 90:  # At least 90 days
                actual_periods = len(returns)
                vol_actual = returns.std() * np.sqrt(252)
                print(f"  ℹ️  {ticker}: Using {actual_periods}-day volatility (insufficient 1-year data)")
                final_volatility = vol_actual
            else:
                print(f"  ❌ {ticker}: Insufficient data for volatility calculation ({len(returns)} < 90 days)")
                return np.nan
        else:
            # Simple final volatility (consistent with 1-year beta)
            final_volatility = vol_1yr
        
        # Step 2B: T-Distribution Enhancement
        close_col = extract_column(data, 'Close')
        returns = close_col.pct_change().dropna()
        
        if len(returns) < 30:  # Need minimum data for t-distribution
            return final_volatility
            
        try:
            # Fit t-distribution to estimate degrees of freedom (ν)
            params = stats.t.fit(returns)
            degrees_of_freedom = params[0]
            
            # Adjust volatility for fat tails
            if degrees_of_freedom > 2.1:
                adjusted_volatility = final_volatility * np.sqrt(degrees_of_freedom / (degrees_of_freedom - 2))
                return adjusted_volatility
            else:
                return final_volatility
            
        except Exception:
            return final_volatility
    
    def calculate_beta(self, etf_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate beta between ETF and benchmark returns
        
        Beta = Cov(R_etf, R_benchmark) / Var(R_benchmark)
        
        Uses consistent ddof=1 (sample statistics) throughout
        """
        if len(etf_returns) < 30 or len(benchmark_returns) < 30:
            return np.nan
            
        # Align the series by date
        common_dates = etf_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < 30:
            return np.nan
            
        etf_aligned = etf_returns.loc[common_dates].dropna()
        bench_aligned = benchmark_returns.loc[common_dates].dropna()
        
        # Ensure both series have clean overlapping data
        common_clean = etf_aligned.index.intersection(bench_aligned.index)
        if len(common_clean) < 30:
            return np.nan
        
        etf_clean = etf_aligned.loc[common_clean]
        bench_clean = bench_aligned.loc[common_clean]
        
        # Calculate covariance matrix with consistent ddof=1 (sample covariance)
        # Note: np.cov uses ddof=1 by default
        cov_matrix = np.cov(etf_clean, bench_clean, ddof=1)
        covariance = cov_matrix[0, 1]
        
        # Variance of benchmark with ddof=1 (sample variance)
        benchmark_variance = np.var(bench_clean, ddof=1)
        
        if benchmark_variance == 0 or np.isnan(covariance) or np.isnan(benchmark_variance):
            return np.nan
        
        beta = covariance / benchmark_variance
        
        # Sanity check: beta should typically be between -3 and 3
        if abs(beta) > 10:
            return np.nan
            
        return beta
    
    def identify_highest_correlation(self, etf_returns: pd.Series) -> Tuple[str, float]:
        """Find benchmark with highest correlation to ETF"""
        best_correlation = -1
        best_benchmark = None
        
        for benchmark_name, benchmark_data in self.benchmark_data.items():
            if benchmark_data.empty:
                continue
                
            close_col = extract_column(benchmark_data, 'Close')
            benchmark_returns = close_col.pct_change().dropna()
            
            # Align dates
            common_dates = etf_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) < 30:
                continue
                
            etf_aligned = etf_returns.loc[common_dates]
            bench_aligned = benchmark_returns.loc[common_dates]
            
            correlation = etf_aligned.corr(bench_aligned)
            
            if not np.isnan(correlation) and correlation > best_correlation:
                best_correlation = correlation
                best_benchmark = benchmark_name
                
        return best_benchmark, best_correlation
    
    def calculate_volatility_beta(self, data: pd.DataFrame, ticker: str = "Unknown") -> Tuple[float, str]:
        """
        Calculate volatility beta with correlation analysis
        Returns: (final_beta, best_benchmark)
        Added ticker parameter for better logging
        """
        close_col = extract_column(data, 'Close')
        etf_returns = close_col.pct_change().dropna()
        
        if len(etf_returns) < 30:
            print(f"  ⚠️  Beta calculation failed for {ticker}: Insufficient returns ({len(etf_returns)} < 30)")
            return np.nan, None
            
        # Find best correlated benchmark
        best_benchmark, correlation = self.identify_highest_correlation(etf_returns)
        
        if best_benchmark is None:
            print(f"  ⚠️  Beta calculation failed for {ticker}: No suitable benchmark found")
            return np.nan, None
            
        benchmark_data = self.benchmark_data[best_benchmark]
        close_col = extract_column(benchmark_data, 'Close')
        benchmark_returns = close_col.pct_change().dropna()
        
        # Calculate 1-year beta (most recent and relevant)
        periods_1yr = min(252, len(etf_returns))  # Use up to 1 year of data
        
        if periods_1yr < 30:  # Need at least 1 month of data (reduced from 60 to support newer ETFs)
            print(f"  ⚠️  Beta calculation failed for {ticker}: Insufficient periods ({periods_1yr} < 30)")
            return np.nan, best_benchmark
            
        # Calculate 1-year beta
        final_beta = self.calculate_beta(
            etf_returns.tail(periods_1yr), 
            benchmark_returns.tail(periods_1yr)
        )
        
        return final_beta, best_benchmark
    
    def classify_risk(self, volatility: float, beta: float) -> Tuple[str, float]:
        """
        Matrix-based risk classification
        Primary Classification (Volatility 70% + Beta 30%)
        
        Volatility Bands:
        - < 12%: Low volatility (bonds, defensive)
        - 12-22%: Medium volatility (standard equity)
        - > 22%: High volatility (growth, emerging)
        
        Beta Bands:
        - < 0.8: Low market sensitivity
        - 0.8-1.2: Market-aligned
        - > 1.2: High market sensitivity
        """
        if np.isnan(volatility) or np.isnan(beta):
            return "UNKNOWN", 0.0
        
        # Calculate risk score for transparency (even though not used for classification)
        risk_score = volatility * 0.7 + abs(beta - 1.0) * 0.3
        
        # Volatility Band: < 12%
        if volatility < 0.12:
            if beta < 0.8:
                return "LOW", risk_score
            elif beta <= 1.2:
                return "LOW", risk_score
            else:  # beta > 1.2
                return "MEDIUM", risk_score  # upgrade due to market sensitivity
                
        # Volatility Band: 12-22%
        elif volatility <= 0.22:
            if beta < 0.8:
                return "MEDIUM", risk_score
            elif beta <= 1.2:
                return "MEDIUM", risk_score
            else:  # beta > 1.2
                return "HIGH", risk_score  # upgrade due to market sensitivity  
                
        # Volatility Band: > 22%
        else:
            if beta < 0.8:
                return "MEDIUM", risk_score  # downgrade due to market independence
            elif beta <= 1.2:
                return "HIGH", risk_score
            else:  # beta > 1.2
                return "HIGH", risk_score
    
    def process_etf(self, ticker: str) -> Dict:
        """Process a single ETF and return risk classification data"""
        print(f"Processing {ticker}...")
        
        # Download data
        data, quality_tier, quality_score = self.download_etf_data(ticker)
        
        if data is None or quality_tier == "insufficient":
            return None
            
        # Calculate enhanced volatility (passing ticker for better logging)
        volatility = self.calculate_enhanced_volatility(data, ticker)
        
        if pd.isna(volatility):
            print(f"  ❌ {ticker}: Volatility calculation failed")
            return None
            
        # Calculate beta (passing ticker for better logging)
        beta, best_benchmark = self.calculate_volatility_beta(data, ticker)
        
        # Use fallback beta for new ETFs if calculation fails
        beta_confidence = 'normal'
        if pd.isna(beta):
            if len(data) >= 90:  # Only use fallback for ETFs with sufficient data
                beta = 1.0
                best_benchmark = list(self.benchmark_data.keys())[0] if self.benchmark_data else 'VTS.AX'
                beta_confidence = 'low'
                print(f"  Using fallback beta=1.0 for {ticker} (insufficient benchmark overlap)")
            else:
                return None  # Still reject if data is truly insufficient
            
        # Classify risk
        risk_category, risk_score = self.classify_risk(volatility, beta)
        
        return {
            'ticker': ticker,
            'data': data,
            'volatility': volatility,
            'beta': beta,
            'beta_confidence': beta_confidence,
            'best_benchmark': best_benchmark,
            'quality_tier': quality_tier,
            'quality_score': quality_score,
            'risk_category': risk_category,
            'risk_score': risk_score
        }
    
    def classify_etfs(self, etf_tickers: List[str]) -> Tuple[Dict, Dict]:
        """
        Main classification function
        Returns: (classified_etfs, summary)
        """
        print("Starting ETF risk classification...")
        start_time = time.time()
        
        # Download benchmark data first
        self.download_benchmark_data()
        
        # Initialize result dictionaries
        low_risk_etfs = {}
        medium_risk_etfs = {}
        high_risk_etfs = {}
        failed_downloads = []
        
        total_etfs = len(etf_tickers)
        
        for i, ticker in enumerate(etf_tickers, 1):
            print(f"\n[{i}/{total_etfs}] Processing {ticker}...")
            
            try:
                result = self.process_etf(ticker)
                
                if result is None:
                    failed_downloads.append(ticker)
                    print(f"  ✗ Failed to process {ticker}")
                    continue
                
                # Store in appropriate risk category
                etf_data = {
                    'data': result['data'],
                    'volatility': result['volatility'],
                    'beta': result['beta'],
                    'best_benchmark': result['best_benchmark'],
                    'quality_tier': result['quality_tier'],
                    'quality_score': result['quality_score'],
                    'risk_score': result['risk_score'],
                    'etf_info': self.etf_database.etf_data.get(ticker, {})
                }
                
                if result['risk_category'] == 'LOW':
                    low_risk_etfs[ticker] = etf_data
                elif result['risk_category'] == 'MEDIUM':
                    medium_risk_etfs[ticker] = etf_data
                elif result['risk_category'] == 'HIGH':
                    high_risk_etfs[ticker] = etf_data
                
                print(f"  ✓ {ticker}: {result['risk_category']} risk "
                      f"(Vol: {result['volatility']:.3f}, Beta: {result['beta']:.3f})")
                
            except Exception as e:
                print(f"  ✗ Error processing {ticker}: {str(e)}")
                failed_downloads.append(ticker)
        
        # Create summary
        processing_time = time.time() - start_time
        summary = {
            'total_processed': total_etfs,
            'low_risk_count': len(low_risk_etfs),
            'medium_risk_count': len(medium_risk_etfs),
            'high_risk_count': len(high_risk_etfs),
            'failed_downloads': failed_downloads,
            'processing_time_seconds': processing_time
        }
        
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total processed: {summary['total_processed']}")
        print(f"Low risk: {summary['low_risk_count']}")
        print(f"Medium risk: {summary['medium_risk_count']}")
        print(f"High risk: {summary['high_risk_count']}")
        print(f"Failed downloads: {len(failed_downloads)}")
        print(f"Processing time: {processing_time:.1f} seconds")
        
        return {
            'low_risk_etfs': low_risk_etfs,
            'medium_risk_etfs': medium_risk_etfs,
            'high_risk_etfs': high_risk_etfs
        }, summary


def main():
    """Example usage of the ETF Risk Classifier"""
    
    # Initialize classifier
    classifier = ETFRiskClassifier()
    
    # Example ETF tickers (using some from the database)
    sample_etfs = [
        'VAF.AX', 'VGB.AX', 'AAA.AX',  # Low risk examples
        'VAS.AX', 'IOZ.AX', 'STW.AX',  # Medium risk examples  
        'NDQ.AX', 'HACK.AX', 'CRYP.AX' # High risk examples
    ]
    
    print("ETF Risk Classification System")
    print("=" * 40)
    
    # Classify ETFs
    results, summary = classifier.classify_etfs(sample_etfs)
    
    # Display results
    print(f"\nDetailed Results:")
    for category, etfs in results.items():
        print(f"\n{category.upper()}:")
        for ticker, data in etfs.items():
            print(f"  {ticker}: Vol={data['volatility']:.3f}, "
                  f"Beta={data['beta']:.3f}, "
                  f"Benchmark={data['best_benchmark']}")


if __name__ == "__main__":
    main()
