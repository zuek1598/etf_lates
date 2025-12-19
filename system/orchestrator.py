#!/usr/bin/env python3
"""
ETF Analysis System Orchestrator - Modified
Integrates Risk Component, ML Ensemble, and Kalman Hull
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import new components
from utilities.shared_utils import extract_column, extract_adjusted_price
from analyzers.risk_component import RiskComponent
from analyzers.ml_ensemble_production import MLEnsembleProduction
# from analyzers.volume_intelligence import VolumeIntelligence  # REMOVED - no validated factors
from analyzers.etf_risk_classifier import ETFRiskClassifier
from analyzers.percentile_ranker import PercentileRanker
from analyzers.kalman_hull import calculate_adaptive_kalman_hull
from data_manager.data_manager import ETFDataManager as ETFDatabase
from utilities.validators import validate_output
from multiprocessing import Pool, TimeoutError
import time
import logging

# Configure logging for error tracking
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# MODULE-LEVEL HELPER FUNCTIONS FOR MULTIPROCESSING (PHASE 1.4-1.5)
# These must be at module level to be pickleable by multiprocessing.Pool
# ============================================================================

def _process_ml_ensemble_etf(args):
    """Process single ETF for ML Ensemble (CPU-bound, multiprocessing candidate)"""
    ticker, etf_data = args
    print(f"[ML_ENSEMBLE_START] {ticker}")

    try:
        ml_ensemble = MLEnsembleProduction()  # Create fresh instance per process
        data = etf_data['data']

        # Run ML Forecast
        ml_output = ml_ensemble.forecast_etf(data)

        # Run walk-forward validation (FIX 1: Reuse trained models instead of retraining)
        prices = extract_column(data, 'Close')
        if prices is not None and len(prices) >= 312:  # 252 train + 60 test
            # Capture trained models from forecast and reuse in validation
            trained_models = ml_output.get('trained_models')
            validation = ml_ensemble.walk_forward_validate(prices, models=trained_models)
            ml_output['hit_rate'] = validation['hit_rate']
            print(f"[ML_ENSEMBLE_OK] {ticker}: forecast={ml_output.get('forecast_return', 0):.4f}, hit_rate={validation['hit_rate']:.2f}")
        else:
            ml_output['hit_rate'] = np.nan
            print(f"[ML_ENSEMBLE_NODATA] {ticker}: insufficient data for validation")

        # Remove trained_models from output (not needed downstream)
        ml_output.pop('trained_models', None)
        return ticker, ml_output
    except Exception as e:
        print(f"[ML_ENSEMBLE_ERROR] {ticker}: {str(e)}")
        return ticker, {
            'forecast_return': 0.0, 'confidence_score': 0.5,
            'features_used': {}, 'model_ensemble_output': 0.0,
            'feature_importance': {}, 'hit_rate': np.nan,
            'error': str(e)
        }


def _process_kalman_hull_etf(args):
    """Process single ETF for Kalman Hull (CPU-bound, multiprocessing candidate)"""
    ticker, etf_data, risk_category = args
    print(f"[KALMAN_HULL_START] {ticker}")

    try:
        data = etf_data['data']
        prices = extract_column(data, 'Close')
        volume = extract_column(data, 'Volume')

        if prices is not None and len(prices) >= 30:
            result = calculate_adaptive_kalman_hull(
                prices, volume, risk_category=risk_category, ohlc_data=data
            )
            print(f"[KALMAN_HULL_OK] {ticker}: signal_strength={result.get('signal_strength', 0):.2f}")
            return ticker, result
        else:
            print(f"[KALMAN_HULL_NODATA] {ticker}: insufficient price data")
            return ticker, {
                'signal_strength': 0.0
            }
    except Exception as e:
        print(f"[KALMAN_HULL_ERROR] {ticker}: {str(e)}")
        return ticker, {
            'signal_strength': 0.0,
            'error': str(e)
        }


# ============================================================================
# PHASE 1.6: TIMEOUT AND ERROR RECOVERY HELPERS
# ============================================================================

def _apply_with_timeout(pool, func, args_list, timeout_sec: int = 120, max_retries: int = 1) -> Tuple[Dict, Dict]:
    """
    Execute pool.map with timeout handling and retry logic

    Args:
        pool: multiprocessing.Pool instance
        func: Function to apply to each item
        args_list: List of arguments to pass to func
        timeout_sec: Timeout per ETF (120s default)
        max_retries: Number of retry attempts on timeout

    Returns:
        Tuple of (results_dict, errors_dict) where errors_dict tracks failed ETFs
    """
    results = {}
    errors = {}

    for ticker, args in enumerate(args_list):
        retries = 0
        success = False

        while retries < max_retries and not success:
            try:
                # Use apply_async to get timeout control
                async_result = pool.apply_async(func, (args,))
                ticker_key, output = async_result.get(timeout=timeout_sec)

                # Check for errors in output
                if isinstance(output, dict) and 'error' in output:
                    errors[ticker_key] = output['error']
                    print(f"[TIMEOUT_RECOVERY] {ticker_key}: Processing error - {output['error']}")
                else:
                    results[ticker_key] = output
                    success = True

            except TimeoutError:
                retries += 1
                print(f"[TIMEOUT_ATTEMPT] {args[0]}: Attempt {retries}/{max_retries} (timeout after {timeout_sec}s)")

                if retries >= max_retries:
                    # Store timeout error
                    ticker_key = args[0]
                    errors[ticker_key] = f"Timeout after {timeout_sec}s x {max_retries} retries"
                    print(f"[TIMEOUT_FAILED] {ticker_key}: Failed after {max_retries} retries")

            except Exception as e:
                ticker_key = args[0]
                errors[ticker_key] = str(e)
                print(f"[RECOVERY_ERROR] {ticker_key}: {str(e)}")
                success = True  # Don't retry on non-timeout errors

    return results, errors


def _get_ml_ensemble_fallback(ticker: str) -> Dict:
    """Fallback ML Ensemble output for failed processing"""
    return {
        'forecast_return': 0.0,
        'confidence_score': 0.5,
        'features_used': {},
        'model_ensemble_output': 0.0,
        'feature_importance': {},
        'hit_rate': np.nan,
        'is_fallback': True
    }


def _get_kalman_hull_fallback(ticker: str) -> Dict:
    """Fallback Kalman Hull output for failed processing"""
    return {
        'signal_strength': 0.0,
        'is_fallback': True
    }




class ETFAnalysisSystem:
    """Streamlined ETF analysis system with new components"""
    
    def __init__(self):
        self.risk_component = RiskComponent()
        self.risk_component.validated_only = True  # ENABLE OPTIMIZED MODE
        self.ml_ensemble = MLEnsembleProduction()
        # self.volume_intelligence = VolumeIntelligence()  # REMOVED - no validated factors
        self.risk_classifier = ETFRiskClassifier()
        self.etf_database = ETFDatabase()
        self.percentile_ranker = PercentileRanker(lookback_days=252)
        self.vix_data = None
        self.benchmark_data = {}
        self._historical_data_saved = False

        # Download market data on initialization
        self.download_market_data()
    
    def download_market_data(self):
        """Download VIX and benchmark data with caching"""
        import pickle
        from datetime import datetime, timedelta
        
        cache_file = 'data/market_data_cache.pkl'
        cache_age_hours = 24  # Cache for 24 hours
        
        # Check if cached data exists and is fresh
        if os.path.exists(cache_file):
            try:
                cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
                cache_age = datetime.now() - cache_mtime
                
                if cache_age < timedelta(hours=cache_age_hours):
                    print(f"Loading market data from cache (age: {cache_age.total_seconds()/3600:.1f}h)...")
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        self.vix_data = cached_data['vix_data']
                        self.benchmark_data = cached_data['benchmark_data']
                        print(f"  VIX data: {len(self.vix_data)} days (cached)")
                        print(f"  Benchmarks: {len(self.benchmark_data)} indices (cached)")
                        return
                else:
                    print(f"Cache expired ({cache_age.total_seconds()/3600:.1f}h old), downloading fresh data...")
            except Exception as e:
                print(f"Cache load failed: {e}, downloading fresh data...")
        
        # Download fresh data
        print("Downloading market data...")
        
        # Download VIX
        try:
            import yfinance as yf
            print("  - Downloading VIX data...")
            vix = yf.download('^VIX', period='max', progress=False, timeout=30)
            if not vix.empty:
                close_col = vix['Close'] if isinstance(vix['Close'], pd.Series) else vix['Close'].iloc[:, 0]
                self.vix_data = close_col
                print(f"  VIX data: {len(self.vix_data)} days")
        except Exception as e:
            print(f"  VIX download failed: {e}")
        
        # Download benchmarks
        print("  - Downloading benchmark data...")
        self.risk_classifier.download_benchmark_data()
        self.benchmark_data = self.risk_classifier.benchmark_data
        print(f"  Benchmarks: {len(self.benchmark_data)} indices")
        
        # Save to cache
        try:
            os.makedirs('data', exist_ok=True)
            cache_data = {
                'vix_data': self.vix_data,
                'benchmark_data': self.benchmark_data,
                'timestamp': datetime.now()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"  Market data cached for {cache_age_hours} hours")
        except Exception as e:
            print(f"  Cache save failed: {e}")
    
    def _save_historical_data(self, risk_group_etfs: Dict):
        """Save historical data to disk for backtesting"""
        from pathlib import Path
        import traceback
        
        print(f"      DEBUG: _save_historical_data called with {len(risk_group_etfs)} ETFs")
        
        historical_dir = Path('data/historical')
        historical_dir.mkdir(exist_ok=True, parents=True)
        print(f"      DEBUG: Directory exists: {historical_dir.exists()}, Is dir: {historical_dir.is_dir()}")
        
        saved_count = 0
        failed_count = 0
        
        # Check first ETF structure
        if risk_group_etfs:
            first_ticker = list(risk_group_etfs.keys())[0]
            first_data = risk_group_etfs[first_ticker]
            print(f"      DEBUG: First ETF ({first_ticker}) keys: {list(first_data.keys())}")
            print(f"      DEBUG: First ETF 'data' type: {type(first_data.get('data'))}")
        
        for ticker, etf_data in risk_group_etfs.items():
            try:
                data = etf_data.get('data')
                
                if data is None:
                    print(f"       {ticker}: data is None")
                    failed_count += 1
                    continue
                
                if data.empty:
                    print(f"       {ticker}: data is empty")
                    failed_count += 1
                    continue
                
                # Flatten multi-level columns if present (yfinance creates these)
                if isinstance(data.columns, pd.MultiIndex):
                    # Keep only the first level (e.g., 'Close' instead of ('Close', 'TICKER'))
                    data = data.copy()
                    data.columns = data.columns.get_level_values(0)
                
                # Save
                file_path = historical_dir / f"{ticker.replace('.', '_')}.parquet"
                data.to_parquet(file_path)
                saved_count += 1
                
            except Exception as e:
                print(f"      {ticker}: {type(e).__name__}: {str(e)}")
                traceback.print_exc()
                failed_count += 1
        
        print(f"      Saved {saved_count} files, {failed_count} skipped")
    
    def classify_etfs_by_risk(self, etf_tickers: List[str]) -> Dict:
        """Classify ETFs into risk categories"""
        print(f"Classifying {len(etf_tickers)} ETFs by risk...")
        results, summary = self.risk_classifier.classify_etfs(etf_tickers)
        
        for category, etfs in results.items():
            for ticker, data in etfs.items():
                data['etf_info'] = self.etf_database.etf_data.get(ticker, {})
        
        print(f"  Low: {summary['low_risk_count']}, Medium: {summary['medium_risk_count']}, High: {summary['high_risk_count']}")
        return results
    
    def analyze_risk_group_parallel(self, risk_group_etfs: Dict, risk_category: str, max_workers: int = None) -> Dict:
        """
        PHASE 1.4-1.5: Parallel version of analyze_risk_group using multiprocessing.Pool

        Parallelizes CPU-bound analysis:
        - ML Ensemble (forecast_etf + walk_forward_validate)
        - Kalman Hull indicator
        - Kalman HullComponent and result combining remain sequential (smaller overhead)

        Args:
            risk_group_etfs: Dict of {ticker: etf_data}
            risk_category: 'LOW', 'MEDIUM', or 'HIGH'
            max_workers: Number of processes (auto-optimized if None)
        """
        # Optimize worker count based on ETF count
        import multiprocessing
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            etf_count = len(risk_group_etfs)
            
            # Intelligent worker allocation
            if etf_count <= 10:
                max_workers = min(2, cpu_count)  # Small groups don't need many workers
            elif etf_count <= 50:
                max_workers = min(4, cpu_count)  # Medium groups
            else:
                max_workers = min(cpu_count - 1, 6)  # Large groups, but cap at 6 for efficiency
            
            print(f"  Optimized: {etf_count} ETFs → {max_workers} workers (CPU: {cpu_count})")
        
        print(f"  Analyzing {risk_category} group ({len(risk_group_etfs)} ETFs) [PARALLEL MODE]...")
        start_time = time.time()

        # Save historical data to disk for backtesting (FIX 4: Only save once per run)
        if not self._historical_data_saved:
            self._save_historical_data(risk_group_etfs)
            self._historical_data_saved = True
        else:
            print(f"      [CACHE] Skipping duplicate historical data save (already saved)")

        # Run Risk Component analysis (already handled separately, not parallelized)
        risk_results = self.risk_component.analyze_risk_group(risk_group_etfs, self.vix_data, self.benchmark_data)

        # ====================================================================
        # PHASE 1.4: Parallelize ML Ensemble (50% of runtime)
        # PHASE 1.6: With timeout handling and error recovery
        # FIX 1: Use true parallel execution - submit ALL tasks first, then collect
        # ====================================================================
        print(f"    [ML_ENSEMBLE] Processing {len(risk_group_etfs)} ETFs with {max_workers or 'auto'} workers...")
        ml_start = time.time()

        ml_results = {}
        ml_errors = {}
        ml_timeouts = 0

        etf_list = list(risk_group_etfs.items())

        try:
            with Pool(processes=max_workers) as pool:
                # PHASE 1: Submit ALL tasks at once (non-blocking)
                async_results = []
                for ticker, etf_data in etf_list:
                    async_result = pool.apply_async(_process_ml_ensemble_etf, ((ticker, etf_data),))
                    async_results.append((ticker, async_result))

                print(f"    [ML_ENSEMBLE] Submitted {len(async_results)} tasks, collecting results...")

                # PHASE 2: Collect results (with timeout handling per ETF)
                for i, (ticker, async_result) in enumerate(async_results):
                    try:
                        ticker_key, ml_output = async_result.get(timeout=120)  # 120s per ETF

                        # Progress indicator
                        if (i + 1) % 10 == 0 or (i + 1) == len(async_results):
                            print(f"    [ML_ENSEMBLE_PROGRESS] {i + 1}/{len(async_results)} completed")

                        # Check for errors
                        if isinstance(ml_output, dict) and 'error' in ml_output:
                            ml_errors[ticker_key] = ml_output['error']
                            ml_results[ticker_key] = _get_ml_ensemble_fallback(ticker_key)
                            print(f"    [ML_ENSEMBLE_ERROR] {ticker_key}: {ml_output['error']}")
                        else:
                            ml_results[ticker_key] = ml_output

                    except TimeoutError:
                        ml_timeouts += 1
                        ml_errors[ticker_key] = "Timeout (120s)"
                        ml_results[ticker_key] = _get_ml_ensemble_fallback(ticker_key)
                        print(f"    [ML_ENSEMBLE_TIMEOUT] {ticker_key}: Using fallback values")

                    except Exception as e:
                        # Handle unexpected errors during result collection
                        ml_errors[ticker_key] = f"Collection error: {str(e)}"
                        ml_results[ticker_key] = _get_ml_ensemble_fallback(ticker_key)
                        print(f"    [ML_ENSEMBLE_COLLECT_ERROR] {ticker_key}: {str(e)}")

        except Exception as e:
            logger.error(f"ML Ensemble pool error: {str(e)}")
            # Fallback: use default values for all remaining ETFs
            for ticker in risk_group_etfs.keys():
                if ticker not in ml_results:
                    ml_results[ticker] = _get_ml_ensemble_fallback(ticker)

        ml_time = time.time() - ml_start
        success_count = len(ml_results) - len(ml_errors)
        print(f"    [ML_ENSEMBLE_COMPLETE] {success_count}/{len(ml_results)} ETFs in {ml_time:.1f}s ({ml_timeouts} timeouts)")

        # ====================================================================
        # Phase 1.5a: Parallelize Kalman Hull
        # PHASE 1.6: With timeout handling and error recovery
        # FIX 2a: Use true parallel execution - submit ALL tasks first, then collect
        # ====================================================================
        print(f"    [KALMAN_HULL] Processing {len(risk_group_etfs)} ETFs...")
        kalman_start = time.time()

        kalman_hull_results = {}
        kalman_errors = {}
        kalman_timeouts = 0

        kalman_args = [(ticker, etf_data, risk_category) for ticker, etf_data in risk_group_etfs.items()]

        try:
            with Pool(processes=max_workers) as pool:
                # PHASE 1: Submit ALL tasks at once (non-blocking)
                async_results = []
                for ticker, etf_data, risk_cat in kalman_args:
                    async_result = pool.apply_async(_process_kalman_hull_etf, ((ticker, etf_data, risk_cat),))
                    async_results.append((ticker, async_result))

                print(f"    [KALMAN_HULL] Submitted {len(async_results)} tasks, collecting results...")

                # PHASE 2: Collect results (with timeout handling per ETF)
                for i, (ticker, async_result) in enumerate(async_results):
                    try:
                        ticker_key, kalman_output = async_result.get(timeout=60)  # 60s per ETF

                        # Progress indicator
                        if (i + 1) % 10 == 0 or (i + 1) == len(async_results):
                            print(f"    [KALMAN_HULL_PROGRESS] {i + 1}/{len(async_results)} completed")

                        # Check for errors
                        if isinstance(kalman_output, dict) and 'error' in kalman_output:
                            kalman_errors[ticker_key] = kalman_output['error']
                            kalman_hull_results[ticker_key] = _get_kalman_hull_fallback(ticker_key)
                            print(f"    [KALMAN_HULL_ERROR] {ticker_key}: {kalman_output['error']}")
                        else:
                            kalman_hull_results[ticker_key] = kalman_output

                    except TimeoutError:
                        kalman_timeouts += 1
                        kalman_errors[ticker_key] = "Timeout (60s)"
                        kalman_hull_results[ticker_key] = _get_kalman_hull_fallback(ticker_key)
                        print(f"    [KALMAN_HULL_TIMEOUT] {ticker_key}: Using fallback values")

                    except Exception as e:
                        # Handle unexpected errors during result collection
                        kalman_errors[ticker_key] = f"Collection error: {str(e)}"
                        kalman_hull_results[ticker_key] = _get_kalman_hull_fallback(ticker_key)
                        print(f"    [KALMAN_HULL_COLLECT_ERROR] {ticker_key}: {str(e)}")

        except Exception as e:
            logger.error(f"Kalman Hull pool error: {str(e)}")
            # Fallback: use default values for all remaining ETFs
            for ticker in risk_group_etfs.keys():
                if ticker not in kalman_hull_results:
                    kalman_hull_results[ticker] = _get_kalman_hull_fallback(ticker)

        kalman_time = time.time() - kalman_start
        success_count = len(kalman_hull_results) - len(kalman_errors)
        print(f"    [KALMAN_HULL_COMPLETE] {success_count}/{len(kalman_hull_results)} ETFs in {kalman_time:.1f}s ({kalman_timeouts} timeouts)")

        # ====================================================================
        # Phase 1.5b: Volume Intelligence removed (no validated factors)
        print(f"    [VOLUME_INTEL] SKIPPED - No validated volume factors")
        vol_time = 0.0

        # ====================================================================
        # Combine results (sequential, small overhead)
        # ====================================================================
        print(f"    [COMBINING] Aggregating {len(risk_group_etfs)} results...")
        combined_results = {}
        for ticker in risk_group_etfs.keys():
            etf_data = risk_group_etfs[ticker]

            # Get component outputs (VALIDATED COMPONENTS ONLY)
            risk_output = risk_results.get(ticker, {})
            ml_output = ml_results.get(ticker, {})
            kalman_output = kalman_hull_results.get(ticker, {})
            # volume_output = volume_intelligence_results.get(ticker, {})  # SKIPPED - no validated factors

            combined_results[ticker] = {
                # VALIDATED FACTORS ONLY (4 factors with p < 0.05, positive IC)
                
                # Risk Component - VALIDATED
                'cvar': risk_output.get('cvar', np.nan),
                
                # ML Ensemble - VALIDATED
                'ml_forecast': -ml_output.get('forecast_return', 0.0),  # FIXED: direction corrected
                'hit_rate': ml_output.get('hit_rate', np.nan),
                
                # Kalman Hull - VALIDATED
                'kalman_signal_strength': kalman_output.get('signal_strength', 0.0),
                
                # Composite Score - CALCULATED
                'composite_score': self._calculate_composite_score(risk_output, ml_output, kalman_output),
            }

            # Add ETF info
            combined_results[ticker]['etf_name'] = etf_data.get('name', 'Unknown')

            # Calculate returns (FIX 4: Calculate once and reuse)
            prices = extract_column(etf_data.get('data'), 'Close')
            if prices is not None and len(prices) > 0:
                combined_results[ticker]['latest_price'] = float(prices.iloc[-1]) if isinstance(prices.iloc[-1], (int, float)) else 0.0
                # Calculate return once and reuse for both fields
                annual_return = self.calculate_adjusted_return(prices, 252)
                combined_results[ticker]['ytd_return'] = annual_return
                combined_results[ticker]['one_year_return'] = annual_return
            else:
                combined_results[ticker]['latest_price'] = 0.0
                combined_results[ticker]['ytd_return'] = 0.0
                combined_results[ticker]['one_year_return'] = 0.0

        total_time = time.time() - start_time
        print(f"    [PARALLEL_COMPLETE] Total time: {total_time:.1f}s (ML:{ml_time:.1f}s, Kalman:{kalman_time:.1f}s, VolIntel:SKIPPED)")

        # ====================================================================
        # Save daily factor values to historical files (Option A)
        # ====================================================================
        self._save_daily_factors(risk_group_etfs, combined_results)

        return combined_results
    
    def _calculate_composite_score(self, risk_output: Dict, ml_output: Dict, kalman_output: Dict) -> float:
        """
        Calculate composite score from validated factors
        
        Args:
            risk_output: Risk component results
            ml_output: ML ensemble results  
            kalman_output: Kalman Hull results
            
        Returns:
            Composite score (0-1)
        """
        try:
            # Risk score (40%) - use CVaR inverted (lower CVaR = higher score)
            cvar = risk_output.get('cvar', 0.0)
            if not np.isnan(cvar):
                # Invert CVaR and normalize to 0-1 (assuming CVaR range -0.1 to 0.0)
                risk_score = max(0.0, min(1.0, (cvar + 0.1) / 0.1))
            else:
                risk_score = 0.5
            
            # ML score (30%) - use forecast and confidence
            ml_forecast = ml_output.get('forecast_return', 0.0)
            confidence = ml_output.get('confidence_score', 0.5)
            # Normalize forecast to 0-1 (assuming range -15% to +15%)
            ml_normalized = max(0.0, min(1.0, (ml_forecast + 15) / 30))
            ml_score = ml_normalized * confidence
            
            # Technical score (30%) - use Kalman signal strength
            signal_strength = kalman_output.get('signal_strength', 0.0)
            # Normalize signal strength to 0-1 (assuming range -1 to +1)
            tech_score = max(0.0, min(1.0, (signal_strength + 1) / 2))
            
            # Weighted composite
            composite = (risk_score * 0.4 + ml_score * 0.3 + tech_score * 0.3)
            
            return max(0.0, min(1.0, composite))
            
        except Exception as e:
            print(f"Composite score calculation error: {e}")
            return 0.5

    def calculate_adjusted_return(self, prices: pd.Series, period_days: int = 252) -> float:
        """
        Calculate return over period, flagging extreme corporate actions.

        Args:
            prices: Adjusted price series
            period_days: Number of trading days to look back

        Returns:
            Float: Adjusted return percentage, or NaN if unreliable due to corporate actions
        """
        if len(prices) < period_days:
            return 0.0

        current_price = prices.iloc[-1]
        past_price = prices.iloc[-period_days]

        # Calculate raw return
        raw_return = (current_price - past_price) / past_price

        # Check for extreme jumps that indicate corporate actions
        # If there's a single day change > 300% or < -50%, mark as unreliable
        daily_returns = prices.pct_change()
        max_daily_jump = daily_returns.max()
        min_daily_jump = daily_returns.min()

        if max_daily_jump > 3.0 or min_daily_jump < -0.5:
            # Extreme corporate action detected - return is not meaningful
            return float('nan')

        return float(raw_return)


    def run_full_analysis(self, etf_tickers: List[str] = None) -> Dict:
        """Run complete analysis pipeline"""
        if etf_tickers is None:
            etf_tickers = list(self.etf_database.etf_data.keys())
        
        print(f"\n{'='*60}")
        print(f"ETF ANALYSIS SYSTEM - Modified")
        print(f"{'='*60}")
        print(f"Analyzing {len(etf_tickers)} ETFs\n")
        
        # Step 1: Classify by risk
        risk_classifications = self.classify_etfs_by_risk(etf_tickers)
        
        # Step 2: Analyze each risk group
        all_results = {}
        risk_categories_map = {}
        
        for risk_category, etfs in risk_classifications.items():
            if etfs:
                group_results = self.analyze_risk_group_parallel(etfs, risk_category)
                all_results.update(group_results)
                
                # Normalize risk category names for scoring system
                # low_risk_etfs -> LOW, medium_risk_etfs -> MEDIUM, high_risk_etfs -> HIGH
                normalized_category = risk_category.replace('_risk_etfs', '').replace('_', '').upper()
                for ticker in etfs.keys():
                    risk_categories_map[ticker] = normalized_category
        
        # Step 3: Calculate percentile rankings
        print("\nCalculating percentile rankings...")

        # Define metrics to use in ranking (STATISTICALLY VALIDATED FACTORS ONLY)
        # Only factors that passed statistical significance testing (p < 0.05, positive IC)
        ranking_metrics = [
            'ml_forecast',              # ML Ensemble forecast (IC=+0.229, p=0.027) ✅
            'hit_rate',                # ML Ensemble directional accuracy (IC=+0.344, p=0.001) ✅
            'kalman_signal_strength',  # Kalman Hull momentum strength (IC=+0.234, p=0.023) ✅
            'cvar',                    # Conditional Value at Risk (IC=+0.261, p=0.011) ✅
        ]

        # Build risk category mapping (reverse lookup)
        risk_categories_dict = {}
        for ticker, category in risk_categories_map.items():
            if category not in risk_categories_dict:
                risk_categories_dict[category] = []
            risk_categories_dict[category].append(ticker)

        # Run percentile ranking
        rankings = self.percentile_ranker.rank_etf_universe(
            all_results,
            risk_categories_dict,
            ranking_metrics
        )

        # Step 3.1: Apply risk filters (CVaR, liquidity)
        risk_filters = {
            'cvar': {'threshold': 10},      # Bottom 10% risk removed
            'risk_score': {'threshold': 15}  # Bottom 15% risk removed
        }
        rankings = self.percentile_ranker.apply_risk_filters(rankings, risk_filters)

        # Add percentile scores back to analysis results
        for risk_category, category_data in rankings.items():
            for rank_entry in category_data.get('rankings', []):
                ticker = rank_entry['ticker']
                if ticker in all_results:
                    all_results[ticker]['composite_percentile'] = rank_entry['composite_percentile']
                    all_results[ticker]['individual_percentiles'] = rank_entry['individual_percentiles']
                    all_results[ticker]['num_factors'] = rank_entry['num_factors']

        # Step 4: Get top 3 per risk category
        top_etfs = []
        print(f"\n{'='*60}")
        print("TOP 3 ETFs BY PERCENTILE RANKING")
        print(f"{'='*60}")

        for risk_category in ['LOW', 'MEDIUM', 'HIGH']:
            if risk_category in rankings and rankings[risk_category].get('top_3'):
                print(f"\n{risk_category} RISK:")
                for i, rank_entry in enumerate(rankings[risk_category]['top_3'], 1):
                    ticker = rank_entry['ticker']
                    percentile = rank_entry['composite_percentile']
                    individual_percentiles = rank_entry.get('individual_percentiles', {})
                    etf_data = all_results.get(ticker, {})

                    # Define the 4 VALIDATED metrics only
                    metric_names = [
                        'ml_forecast', 'hit_rate', 'kalman_signal_strength', 'cvar'
                    ]

                    metric_display_names = [
                        'ML Forecast', 'Hit Rate', 'Kalman Signal', 'CVaR (inv)'
                    ]

                    # Collect all percentile data
                    metric_percentiles = []
                    for metric_name in metric_names:
                        if metric_name in individual_percentiles:
                            perc = individual_percentiles[metric_name]
                            raw_val = etf_data.get(metric_name, 0.0)
                            metric_percentiles.append((metric_name, perc, raw_val))

                    # Get ETF name from database
                    etf_name = self.etf_database.etf_data.get(ticker, {}).get('name', ticker)
                    
                    # Print header with composite percentile and full name
                    print(f"\n  {i}. {ticker} - {etf_name}")
                    print(f"     Composite: {percentile:.1f}th Percentile")

                    # Print detailed metric breakdown table (ASCII-safe for Windows)
                    print(f"     +{'-'*27}+{'-'*12}+{'-'*13}+")
                    print(f"     | Metric                  | Raw Value  | Percentile  |")
                    print(f"     +{'-'*27}+{'-'*12}+{'-'*13}+")

                    for idx, (metric_name, perc, raw_val) in enumerate(metric_percentiles):
                        display_name = metric_display_names[idx] if idx < len(metric_display_names) else metric_name

                        # Format raw value based on metric type
                        if 'forecast' in metric_name:
                            raw_str = f"{raw_val:+.2f}%"
                        elif metric_name in ['hit_rate']:
                            raw_str = f"{raw_val:.2f}"
                        elif metric_name == 'cvar':
                            raw_str = f"{raw_val:.3f}"
                        elif metric_name == 'kalman_signal_strength':
                            raw_str = f"{raw_val:.3f}"
                        else:
                            raw_str = f"{raw_val:.2f}"

                        perc_str = f"{perc:.0f}th"
                        print(f"     | {display_name:<25} | {raw_str:>10} | {perc_str:>11} |")

                    print(f"     +{'-'*27}+{'-'*12}+{'-'*13}+")

                    # Calculate and display analysis
                    sorted_metrics = sorted(metric_percentiles, key=lambda x: x[1], reverse=True)
                    above_70_count = len([x for x in metric_percentiles if x[1] >= 70])

                    # Top 3 strengths
                    strengths = sorted_metrics[:3]
                    strength_str = ", ".join([f"{metric_display_names[metric_names.index(m[0])]}" +
                                             f" ({m[1]:.0f}th)" for m in strengths])

                    # Bottom 2 concerns
                    concerns = sorted_metrics[-2:]
                    concern_str = ", ".join([f"{metric_display_names[metric_names.index(m[0])]}" +
                                            f" ({m[1]:.0f}th)" for m in reversed(concerns)])

                    # Conviction level (adjusted for 4 validated factors)
                    if above_70_count >= 4:
                        conviction = "Very High"
                    elif above_70_count >= 3:
                        conviction = "High"
                    elif above_70_count >= 2:
                        conviction = "Medium"
                    else:
                        conviction = "Low"

                    print(f"\n     STRENGTHS: {strength_str}")
                    print(f"     CONCERNS: {concern_str}")
                    print(f"     CONVICTION: {conviction} ({above_70_count}/4 validated factors > 70th percentile)")

                    top_etfs.append({
                        'ticker': ticker,
                        'percentile': percentile,
                        'category': risk_category
                    })
        
        # Create original risk classifications map (with _risk_etfs suffix) for file saving
        original_risk_map = {}
        for risk_category, etfs in risk_classifications.items():
            for ticker in etfs.keys():
                original_risk_map[ticker] = risk_category
        
        # Step 5: Export rankings to CSV (delegates to percentile_ranker)
        # CSV export removed - only saving to Parquet
        # self.percentile_ranker.export_rankings_to_csv(rankings, 'data/rankings_percentile.csv')

        return {
            'analysis_results': all_results,
            'rankings': rankings,
            'top_etfs': top_etfs,
            'risk_classifications': original_risk_map,
            'risk_categories_normalized': risk_categories_map
        }

    def _save_daily_factors(self, risk_group_etfs: Dict, combined_results: Dict):
        """
        Save daily factor values to historical parquet files.
        This enables time-series factor validation (Option A).
        
        Args:
            risk_group_etfs: Dict of {ticker: etf_data}
            combined_results: Dict of {ticker: factor_values}
        """
        from pathlib import Path
        
        historical_dir = Path('data/historical')
        historical_dir.mkdir(parents=True, exist_ok=True)
        
        for ticker, etf_data in risk_group_etfs.items():
            try:
                # Get today's date from price data
                data = etf_data.get('data')
                if data is None or data.empty:
                    continue
                
                # Get the latest date from the price data
                if isinstance(data.index, pd.DatetimeIndex):
                    today = data.index[-1]
                else:
                    continue
                
                # Get factor values for this ticker
                factors = combined_results.get(ticker, {})
                
                # Load existing historical file
                hist_file = historical_dir / f'{ticker}.parquet'
                if hist_file.exists():
                    try:
                        df_hist = pd.read_parquet(hist_file)
                    except Exception:
                        df_hist = pd.DataFrame()
                else:
                    df_hist = pd.DataFrame()
                
                # Create new row with today's VALIDATED factors only
                factor_row = {
                    'Date': today,
                    'ml_forecast': factors.get('ml_forecast', np.nan),
                    'hit_rate': factors.get('hit_rate', np.nan),
                    'kalman_signal_strength': factors.get('kalman_signal_strength', np.nan),
                    'cvar': factors.get('cvar', np.nan),
                }
                
                # Append new row
                df_new = pd.DataFrame([factor_row])
                if df_hist.empty:
                    df_combined = df_new
                else:
                    # Remove duplicate if today already exists
                    if 'Date' in df_hist.columns:
                        df_hist = df_hist[df_hist['Date'] != today]
                    df_combined = pd.concat([df_hist, df_new], ignore_index=True)
                
                # Set Date as index and save
                if 'Date' in df_combined.columns:
                    df_combined['Date'] = pd.to_datetime(df_combined['Date'])
                    df_combined = df_combined.set_index('Date')
                
                df_combined.to_parquet(hist_file, compression='snappy')
                
            except Exception as e:
                # Silently skip errors - don't interrupt main analysis
                pass


def main():
    """Run analysis on all ETFs"""
    system = ETFAnalysisSystem()
    results = system.run_full_analysis()
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Total ETFs analyzed: {len(results['analysis_results'])}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

