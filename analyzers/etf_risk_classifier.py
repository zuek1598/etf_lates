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
import os
from datetime import datetime
from typing import Dict, Tuple, List
import warnings
import threading
warnings.filterwarnings('ignore')

from utilities.shared_utils import extract_column
from utilities.etf_validator import ETFActivityValidator

class ETFRiskClassifier:
    """
    Advanced ETF risk classification using volatility and beta analysis
    Thread-safe implementation for parallel processing
    """

    def __init__(self, enable_cache=True):
        from data_manager.data_manager import ETFDataManager
        self.etf_database = ETFDataManager()
        self.enable_cache = enable_cache
        self.activity_validator = ETFActivityValidator()
        self._cache_lock = threading.Lock()
        self._benchmark_lock = threading.Lock()  # Lock for benchmark data access
        self.enable_cache = enable_cache  # Can be disabled for parallel mode

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
        """Download all benchmark data for correlation analysis with timeout"""
        print("Downloading benchmark data...")
        
        # Check if we have recent cached data first
        cache_file = os.path.join("data", "benchmark_cache.parquet")
        if os.path.exists(cache_file):
            try:
                cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
                if cache_age.total_seconds() < 24 * 3600:  # Less than 24 hours old
                    print(f"  Using cached benchmark data ({cache_age.total_seconds()/3600:.1f}h old)")
                    cached_data = pd.read_parquet(cache_file)
                    for name in self.benchmarks.keys():
                        if name in cached_data.columns:
                            # Create a DataFrame with the benchmark data (don't drop all NaNs)
                            benchmark_df = pd.DataFrame({name: cached_data[name]})
                            self.benchmark_data[name] = benchmark_df
                    print(f"    Loaded {len(self.benchmark_data)} benchmarks from cache")
                    return self.benchmark_data
            except Exception as e:
                print(f"  Cache read failed: {e}")
        
        for name, ticker in self.benchmarks.items():
            try:
                print(f"  Downloading {name} ({ticker})...")
                
                # Add timeout by using a shorter period for recent data
                data = yf.download(ticker, period="1y", progress=False, timeout=10)
                if not data.empty:
                    self.benchmark_data[name] = data
                    print(f"    {name}: {len(data)} days (from {data.index[0].date()} to {data.index[-1].date()})")
                else:
                    print(f"    {name}: No data")
            except Exception as e:
                print(f"    {name}: Error - {str(e)}")
                
        # Cache the downloaded data for future use
        if self.benchmark_data:
            try:
                os.makedirs("data", exist_ok=True)
                combined_data = pd.DataFrame()
                for name, data in self.benchmark_data.items():
                    if 'Close' in data.columns:
                        combined_data[name] = data['Close']
                combined_data.to_parquet(cache_file)
                print(f"  Cached benchmark data for future use")
            except Exception as e:
                print(f"  Cache save failed: {e}")
        
        print(f"Successfully downloaded {len(self.benchmark_data)} benchmarks")
        
        # Debug: Print benchmark data status
        for name, data in self.benchmark_data.items():
            if 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                print(f"  Debug {name}: {len(data)} price points, {len(returns)} returns")
        
        return self.benchmark_data
    
    def download_etf_data(self, ticker: str, period: str = "max") -> Tuple[pd.DataFrame, str, float]:
        """
        Download ETF data with quality assessment and retry logic
        Thread-safe with optional caching (disabled in parallel mode for thread isolation)

        NOTE: yfinance is NOT thread-safe! Downloads are serialized with a lock
        to prevent data corruption when multiple threads request simultaneously.
        Processing is still parallel, only downloads are serialized.

        Returns: (data, quality_tier, quality_score)
        """
        max_retries = 3
        retry_delay = 1  # Start with 1 second
        cache_key = f"{ticker}_{period}"

        # Check cache only if enabled (skipped in parallel mode for thread safety)
        if self.enable_cache:
            with self._cache_lock:
                if cache_key in self.cache:
                    cached_data, _, _ = self.cache[cache_key]
                    # Check if cached data is fresh (has data from today or yesterday)
                    if not cached_data.empty:
                        latest_date = cached_data.index[-1]
                        days_old = (datetime.now().date() - latest_date.date()).days
                        # If data is more than 1 day old, refresh it
                        if days_old <= 1:
                            return self.cache[cache_key]
                        else:
                            # Cache is stale, remove it and download fresh
                            del self.cache[cache_key]

        for attempt in range(max_retries):
            try:
                # CRITICAL: yfinance concurrent downloads corrupt data!
                # Must serialize with lock to prevent response mixing
                # Use _yfinance_lock if available (parallel mode), else _cache_lock (sequential)
                lock = getattr(self, '_yfinance_lock', None) or self._cache_lock
                with lock:
                    data = yf.download(ticker, period=period, progress=False)

                if data is None or data.empty or len(data) < 30:  # Minimum 30 days
                    if attempt < max_retries - 1:
                        print(f"   {ticker}: No data returned, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        print(f"  {ticker}: Insufficient data after {max_retries} attempts")
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

                # Calculate quality score (0-1)
                quality_score = min(1.0, (years_available / 5) * completeness)

                # Cache result if enabled
                if self.enable_cache:
                    with self._cache_lock:
                        self.cache[cache_key] = (data, quality_tier, quality_score)

                return data, quality_tier, quality_score

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"   {ticker}: Download failed, retrying in {retry_delay}s... ({str(e)[:50]})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"  {ticker}: Download failed after {max_retries} attempts: {str(e)[:50]}")
                    return None, "error", 0.0

        return None, "error", 0.0
    
    def download_etf_data_batch(self, tickers: List[str], period: str = "max") -> Dict[str, Tuple[pd.DataFrame, str, float]]:
        """
        Download multiple ETFs in parallel using batch fetching
        Much faster than individual downloads while maintaining the same interface
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print(f"🚀 Starting batch download: {len(tickers)} ETFs")
        
        results = {}
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all download tasks
            future_to_ticker = {
                executor.submit(self.download_etf_data, ticker, period): ticker 
                for ticker in tickers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data, quality_tier, quality_score = future.result()
                    results[ticker] = (data, quality_tier, quality_score)
                    if data is not None:
                        print(f"  ✅ {ticker}: {len(data)} days")
                    else:
                        print(f"  ❌ {ticker}: Failed")
                except Exception as e:
                    print(f"  ⚠️  {ticker}: Error - {str(e)[:50]}")
                    results[ticker] = (None, "error", 0.0)
        
        successful = sum(1 for result in results.values() if result[0] is not None)
        print(f"🎯 Batch Download Complete: {successful}/{len(tickers)} successful")
        
        return results
    
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
                print(f"  [EMOJI] {ticker}: Using {actual_periods}-day volatility (insufficient 1-year data)")
                final_volatility = vol_actual
            else:
                print(f"  {ticker}: Insufficient data for volatility calculation ({len(returns)} < 90 days)")
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

        # No lock needed - in parallel mode, self.benchmark_data contains deep copies
        # Each thread works with independent DataFrames, no concurrent modifications
        for benchmark_name, benchmark_data in self.benchmark_data.items():
            if benchmark_data.empty:
                continue

            # Benchmark data has the benchmark name as column (not 'Close')
            if benchmark_name in benchmark_data.columns:
                benchmark_prices = benchmark_data[benchmark_name]
            else:
                # Fallback to 'Close' column if it exists
                close_col = extract_column(benchmark_data, 'Close')
                benchmark_prices = close_col if close_col is not None else benchmark_data.iloc[:, 0]
            
            benchmark_returns = benchmark_prices.pct_change().dropna()

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
            print(f"   Beta calculation failed for {ticker}: Insufficient returns ({len(etf_returns)} < 30)")
            return np.nan, None
            
        # Find best correlated benchmark
        best_benchmark, correlation = self.identify_highest_correlation(etf_returns)

        if best_benchmark is None:
            print(f"   Beta calculation failed for {ticker}: No suitable benchmark found")
            return np.nan, None

        # Thread-safe access to benchmark data
        with self._benchmark_lock:
            benchmark_data = self.benchmark_data[best_benchmark]
            
            # Benchmark data has the benchmark name as column (not 'Close')
            if best_benchmark in benchmark_data.columns:
                benchmark_prices = benchmark_data[best_benchmark]
            else:
                # Fallback to 'Close' column if it exists
                close_col = extract_column(benchmark_data, 'Close')
                benchmark_prices = close_col if close_col is not None else benchmark_data.iloc[:, 0]
            
            benchmark_returns = benchmark_prices.pct_change().dropna()
        
        # Calculate 1-year beta (most recent and relevant)
        periods_1yr = min(252, len(etf_returns))  # Use up to 1 year of data
        
        if periods_1yr < 30:  # Need at least 1 month of data (reduced from 60 to support newer ETFs)
            print(f"   Beta calculation failed for {ticker}: Insufficient periods ({periods_1yr} < 30)")
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
        # Download data
        data, quality_tier, quality_score = self.download_etf_data(ticker)

        if data is None or quality_tier == "insufficient":
            return None

        # Validate ETF activity - filter out delisted/inactive ETFs
        prices = extract_column(data, 'Close')
        activity_result = self.activity_validator.validate_etf_activity(prices, ticker)
        
        if not activity_result['is_active']:
            print(f"  ⚠️  {ticker}: Skipping - {activity_result['reason']}")
            return None

        # Calculate enhanced volatility (passing ticker for better logging)
        volatility = self.calculate_enhanced_volatility(data, ticker)

        if pd.isna(volatility):
            return None

        # Calculate beta (passing ticker for better logging)
        beta, best_benchmark = self.calculate_volatility_beta(data, ticker)

        # Use fallback beta for new ETFs if calculation fails
        beta_confidence = 'normal'
        if pd.isna(beta):
            if len(data) >= 90:  # Only use fallback for ETFs with sufficient data
                beta = 1.0
                # Get first benchmark key (no lock needed in parallel - reading only)
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

        # Download benchmark data first (FIX 2: Skip if already loaded)
        if not self.benchmark_data:
            self.download_benchmark_data()
        else:
            print("  Using cached benchmark data (already loaded)")
        
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
                    print(f"  Failed to process {ticker}")
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
                
                print(f"  {ticker}: {result['risk_category']} risk "
                      f"(Vol: {result['volatility']:.3f}, Beta: {result['beta']:.3f})")
                
            except Exception as e:
                print(f"  Error processing {ticker}: {str(e)}")
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

    def classify_etfs_parallel(self, etf_tickers: List[str], max_workers: int = 8) -> Tuple[Dict, Dict]:
        """
        Parallel version of ETF classification using ThreadPoolExecutor

        Much faster for downloading data from yfinance (I/O bound)
        Uses threading instead of multiprocessing for I/O bound tasks

        Args:
            etf_tickers: List of ETF tickers to classify
            max_workers: Number of parallel download threads (default: 8)

        Returns:
            (classified_etfs, summary) - Same format as classify_etfs
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"Starting parallel ETF risk classification ({max_workers} threads)...")
        start_time = time.time()

        # Disable cache for parallel execution to ensure thread isolation
        # Each thread gets its own fresh copy of data, avoiding race conditions
        original_cache_setting = self.enable_cache
        self.enable_cache = False

        try:
            # Download benchmark data once (to be shared by all threads) (FIX 2: Skip if already loaded)
            if not self.benchmark_data:
                self.download_benchmark_data()
            else:
                print("  [CACHE] Using cached benchmark data (already loaded)")

            # Deep copy ALL benchmark data ONCE in main thread (before threads start)
            # This prevents race conditions when threads read from shared dict
            benchmark_copies = {}
            for name, df in self.benchmark_data.items():
                benchmark_copies[name] = df.copy(deep=True)

            # Initialize result dictionaries
            low_risk_etfs = {}
            medium_risk_etfs = {}
            high_risk_etfs = {}
            failed_downloads = []
            results_lock = __import__('threading').Lock()  # Thread-safe access

            total_etfs = len(etf_tickers)
            completed = 0

            # Create SHARED yfinance lock - ONE lock for ALL threads to prevent concurrent yfinance calls
            # (yfinance is NOT thread-safe, concurrent downloads corrupt data!)
            yfinance_lock = __import__('threading').Lock()

            # Create thread-local storage for classifiers
            thread_local = __import__('threading').local()

            def get_thread_classifier():
                """Get or create thread-local classifier with isolated benchmark data"""
                if not hasattr(thread_local, 'classifier'):
                    # Create new classifier instance for this thread
                    thread_local.classifier = ETFRiskClassifier(enable_cache=False)

                    # CRITICAL: Override the classifier's cache lock with the SHARED yfinance lock
                    # This ensures all threads serialize yfinance calls through one lock
                    thread_local.classifier._yfinance_lock = yfinance_lock

                    # Each thread gets its OWN deep copy of benchmark data from the pre-copies
                    # This ensures complete isolation - no shared mutable state
                    thread_local.classifier.benchmark_data = {}
                    for name, df in benchmark_copies.items():
                        thread_local.classifier.benchmark_data[name] = df.copy(deep=True)

                return thread_local.classifier

            def process_ticker_wrapper(ticker: str) -> Tuple[str, Dict]:
                """Wrapper that processes one ticker with thread-local classifier"""
                try:
                    # Get thread-local classifier with isolated data
                    thread_classifier = get_thread_classifier()
                    result = thread_classifier.process_etf(ticker)

                    if result is None:
                        return ticker, None

                    # Create isolated dict with copies of all values (no shared references)
                    isolated_result = {
                        'data': result['data'],  # CRITICAL: Include raw price data for downstream analysis
                        'ticker': str(result['ticker']),
                        'volatility': float(result['volatility']),
                        'beta': float(result['beta']),
                        'best_benchmark': str(result['best_benchmark']) if result['best_benchmark'] else None,
                        'quality_tier': str(result['quality_tier']),
                        'quality_score': float(result['quality_score']),
                        'risk_category': str(result['risk_category']),
                        'risk_score': float(result['risk_score']),
                        'beta_confidence': str(result['beta_confidence'])
                    }
                    return ticker, isolated_result
                except Exception as e:
                    print(f"  [ERROR] {ticker}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return ticker, None

            print(f"Downloading data for {total_etfs} ETFs using {max_workers} threads...")

            # Use ThreadPoolExecutor for parallel I/O-bound downloads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_ticker = {
                    executor.submit(process_ticker_wrapper, ticker): ticker
                    for ticker in etf_tickers
                }

                # Process results as they complete
                for future in as_completed(future_to_ticker):
                    completed += 1

                    try:
                        ticker, result = future.result(timeout=120)

                        if result is None:
                            failed_downloads.append(ticker)
                            print(f"  [{completed}/{total_etfs}] {ticker}: Failed")
                            continue

                        # Thread-safe storage of results
                        with results_lock:
                            etf_data = {
                                'data': result['data'],  # CRITICAL: Include raw price data for downstream analysis
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

                        print(f"  [{completed}/{total_etfs}] {ticker}: {result['risk_category']} "
                              f"(Vol: {result['volatility']:.3f}, Beta: {result['beta']:.3f})")

                    except Exception as e:
                        print(f"  [{completed}/{total_etfs}] {ticker}: Exception - {str(e)}")
                        failed_downloads.append(ticker)

            # Create summary
            processing_time = time.time() - start_time
            summary = {
                'total_processed': total_etfs,
                'low_risk_count': len(low_risk_etfs),
                'medium_risk_count': len(medium_risk_etfs),
                'high_risk_count': len(high_risk_etfs),
                'failed_downloads': failed_downloads,
                'processing_time_seconds': processing_time,
                'parallel_mode': True,
                'num_threads': max_workers
            }

            print(f"\n{'='*60}")
            print(f"PARALLEL CLASSIFICATION COMPLETE")
            print(f"{'='*60}")
            print(f"Total processed: {summary['total_processed']}")
            print(f"Low risk: {summary['low_risk_count']}")
            print(f"Medium risk: {summary['medium_risk_count']}")
            print(f"High risk: {summary['high_risk_count']}")
            print(f"Failed downloads: {len(failed_downloads)}")
            print(f"Processing time: {processing_time:.1f} seconds ({total_etfs/processing_time:.1f} ETF/sec)")

            return {
                'low_risk_etfs': low_risk_etfs,
                'medium_risk_etfs': medium_risk_etfs,
                'high_risk_etfs': high_risk_etfs
            }, summary

        finally:
            # Restore original cache setting
            # (benchmark_data unchanged - thread-local classifiers have their own copies)
            self.enable_cache = original_cache_setting


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
