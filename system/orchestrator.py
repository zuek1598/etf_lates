#!/usr/bin/env python3
"""
ETF Analysis System Orchestrator - Modified
Integrates Risk Component, ML Ensemble, Volume Intelligence, and Kalman Hull
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import new components
from utilities.shared_utils import extract_column, extract_adjusted_price
from analyzers.risk_component import RiskComponent
from analyzers.ml_ensemble import MLEnsemble
from analyzers.volume_intelligence import VolumeIntelligence
from analyzers.etf_risk_classifier import ETFRiskClassifier
from analyzers.scoring_system_growth import GrowthScoringSystem
from indicators.kalman_hull import calculate_adaptive_kalman_hull
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
        ml_ensemble = MLEnsemble()  # Create fresh instance per process
        data = etf_data['data']

        # Run ML Forecast
        ml_output = ml_ensemble.forecast_etf(data)

        # Run walk-forward validation
        prices = extract_column(data, 'Close')
        if prices is not None and len(prices) >= 312:  # 252 train + 60 test
            validation = ml_ensemble.walk_forward_validate(prices)
            ml_output['mae_score'] = validation['mae']
            ml_output['hit_rate'] = validation['hit_rate']
            print(f"[ML_ENSEMBLE_OK] {ticker}: forecast={ml_output.get('forecast_return', 0):.4f}, mae={validation['mae']:.4f}, hit_rate={validation['hit_rate']:.2f}")
        else:
            ml_output['mae_score'] = np.nan
            ml_output['hit_rate'] = np.nan
            print(f"[ML_ENSEMBLE_NODATA] {ticker}: insufficient data for validation")

        return ticker, ml_output
    except Exception as e:
        print(f"[ML_ENSEMBLE_ERROR] {ticker}: {str(e)}")
        return ticker, {
            'forecast_return': 0.0, 'confidence_score': 0.5,
            'features_used': {}, 'model_ensemble_output': 0.0,
            'feature_importance': {}, 'mae_score': np.nan, 'hit_rate': np.nan,
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
            print(f"[KALMAN_HULL_OK] {ticker}: trend={result.get('trend')}, signal={result.get('signal_strength', 0):.2f}")
            return ticker, result
        else:
            print(f"[KALMAN_HULL_NODATA] {ticker}: insufficient price data")
            return ticker, {
                'trend': 0, 'kalman_price': np.nan, 'upper_band': np.nan,
                'lower_band': np.nan, 'efficiency_ratio': 0.5,
                'divergence': 'none', 'trend_consistency': False, 'signal_strength': 0.0
            }
    except Exception as e:
        print(f"[KALMAN_HULL_ERROR] {ticker}: {str(e)}")
        return ticker, {
            'trend': 0, 'kalman_price': np.nan, 'upper_band': np.nan,
            'lower_band': np.nan, 'efficiency_ratio': 0.5,
            'divergence': 'none', 'trend_consistency': False, 'signal_strength': 0.0,
            'error': str(e)
        }


def _process_volume_intelligence_etf(args):
    """Process single ETF for Volume Intelligence (CPU-bound, multiprocessing candidate)"""
    ticker, etf_data = args
    print(f"[VOLUME_INTEL_START] {ticker}")

    try:
        volume_intelligence = VolumeIntelligence()  # Create fresh instance per process
        data = etf_data['data']
        prices = extract_column(data, 'Close')
        volume = extract_column(data, 'Volume')

        if prices is not None and volume is not None and len(prices) >= 20:
            result = volume_intelligence.analyze_volume(prices, volume, ohlc_data=data)
            print(f"[VOLUME_INTEL_OK] {ticker}: spike_score={result.get('spike_score', 0):.2f}, correlation={result.get('price_volume_correlation', 0):.2f}")
            return ticker, result
        else:
            print(f"[VOLUME_INTEL_NODATA] {ticker}: insufficient data")
            return ticker, {
                'spike_score': 0.0, 'price_volume_correlation': 0.0,
                'accumulation_distribution': 'neutral', 'volume_confidence': 0.0
            }
    except Exception as e:
        print(f"[VOLUME_INTEL_ERROR] {ticker}: {str(e)}")
        return ticker, {
            'spike_score': 0.0, 'price_volume_correlation': 0.0,
            'accumulation_distribution': 'neutral', 'volume_confidence': 0.0,
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
        'mae_score': np.nan,
        'hit_rate': np.nan,
        'is_fallback': True
    }


def _get_kalman_hull_fallback(ticker: str) -> Dict:
    """Fallback Kalman Hull output for failed processing"""
    return {
        'trend': 0,
        'kalman_price': np.nan,
        'upper_band': np.nan,
        'lower_band': np.nan,
        'efficiency_ratio': 0.5,
        'divergence': 'none',
        'trend_consistency': False,
        'signal_strength': 0.0,
        'is_fallback': True
    }


def _get_volume_intelligence_fallback(ticker: str) -> Dict:
    """Fallback Volume Intelligence output for failed processing"""
    return {
        'spike_score': 0.0,
        'price_volume_correlation': 0.0,
        'accumulation_distribution': 'neutral',
        'volume_confidence': 0.0,
        'is_fallback': True
    }


class ETFAnalysisSystem:
    """Streamlined ETF analysis system with new components"""
    
    def __init__(self):
        self.risk_component = RiskComponent()
        self.ml_ensemble = MLEnsemble()
        self.volume_intelligence = VolumeIntelligence()
        self.risk_classifier = ETFRiskClassifier()
        self.etf_database = ETFDatabase()
        self.scoring_system = GrowthScoringSystem()
        self.vix_data = None
        self.benchmark_data = {}
        
        # Download market data on initialization
        self.download_market_data()
    
    def download_market_data(self):
        """Download VIX and benchmark data"""
        print("Downloading market data...")
        
        # Download VIX
        try:
            import yfinance as yf
            print("  - Downloading VIX data...")
            vix = yf.download('^VIX', period='max', progress=False, timeout=30)
            if not vix.empty:
                close_col = vix['Close'] if isinstance(vix['Close'], pd.Series) else vix['Close'].iloc[:, 0]
                self.vix_data = close_col
                print(f"  [EMOJI] VIX data: {len(self.vix_data)} days")
        except Exception as e:
            print(f"  [EMOJI] VIX download failed: {e}")
        
        # Download benchmarks
        print("  - Downloading benchmark data...")
        self.risk_classifier.download_benchmark_data()
        self.benchmark_data = self.risk_classifier.benchmark_data
        print(f"  [EMOJI] Benchmarks: {len(self.benchmark_data)} indices")
    
    def _save_historical_data(self, risk_group_etfs: Dict):
        """Save historical data to disk for backtesting"""
        from pathlib import Path
        import traceback
        
        print(f"      [EMOJI] DEBUG: _save_historical_data called with {len(risk_group_etfs)} ETFs")
        
        historical_dir = Path('data/historical')
        historical_dir.mkdir(exist_ok=True, parents=True)
        print(f"      [EMOJI] DEBUG: Directory exists: {historical_dir.exists()}, Is dir: {historical_dir.is_dir()}")
        
        saved_count = 0
        failed_count = 0
        
        # Check first ETF structure
        if risk_group_etfs:
            first_ticker = list(risk_group_etfs.keys())[0]
            first_data = risk_group_etfs[first_ticker]
            print(f"      [EMOJI] DEBUG: First ETF ({first_ticker}) keys: {list(first_data.keys())}")
            print(f"      [EMOJI] DEBUG: First ETF 'data' type: {type(first_data.get('data'))}")
        
        for ticker, etf_data in risk_group_etfs.items():
            try:
                data = etf_data.get('data')
                
                if data is None:
                    print(f"      [EMOJI]  {ticker}: data is None")
                    failed_count += 1
                    continue
                
                if data.empty:
                    print(f"      [EMOJI]  {ticker}: data is empty")
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
                print(f"      [EMOJI] {ticker}: {type(e).__name__}: {str(e)}")
                traceback.print_exc()
                failed_count += 1
        
        print(f"      [EMOJI] Saved {saved_count} files, {failed_count} skipped")
    
    def classify_etfs_by_risk(self, etf_tickers: List[str]) -> Dict:
        """Classify ETFs into risk categories"""
        print(f"Classifying {len(etf_tickers)} ETFs by risk...")
        results, summary = self.risk_classifier.classify_etfs(etf_tickers)
        
        for category, etfs in results.items():
            for ticker, data in etfs.items():
                data['etf_info'] = self.etf_database.etf_data.get(ticker, {})
        
        print(f"  Low: {summary['low_risk_count']}, Medium: {summary['medium_risk_count']}, High: {summary['high_risk_count']}")
        return results
    
    def analyze_risk_group(self, risk_group_etfs: Dict, risk_category: str) -> Dict:
        """Analyze all ETFs in a risk group with new components"""
        print(f"  Analyzing {risk_category} group ({len(risk_group_etfs)} ETFs)...")
        
        # Save historical data to disk for backtesting
        self._save_historical_data(risk_group_etfs)
        
        # Run Risk Component analysis
        risk_results = self.risk_component.analyze_risk_group(risk_group_etfs, self.vix_data, self.benchmark_data)
        
        # Run ML Ensemble analysis with validation
        ml_results = {}
        for ticker, etf_data in risk_group_etfs.items():
            try:
                data = etf_data['data']  # Extract DataFrame from dict
                ml_output = self.ml_ensemble.forecast_etf(data)
                
                # Run walk-forward validation
                prices = extract_column(data, 'Close')
                if prices is not None and len(prices) >= 312:  # 252 train + 60 test
                    validation = self.ml_ensemble.walk_forward_validate(prices)
                    ml_output['mae_score'] = validation['mae']
                    ml_output['hit_rate'] = validation['hit_rate']
                else:
                    ml_output['mae_score'] = np.nan
                    ml_output['hit_rate'] = np.nan
                
                ml_results[ticker] = ml_output
            except Exception as e:
                print(f"    Warning: ML Ensemble failed for {ticker}: {e}")
                ml_results[ticker] = {
                    'forecast_return': 0.0, 'confidence_score': 0.5,
                    'features_used': {}, 'model_ensemble_output': 0.0,
                    'feature_importance': {}, 'mae_score': np.nan, 'hit_rate': np.nan
                }
        
        # Run Kalman Hull analysis
        kalman_hull_results = {}
        for ticker, etf_data in risk_group_etfs.items():
            try:
                data = etf_data['data']  # Extract DataFrame from dict
                prices = extract_column(data, 'Close')
                volume = extract_column(data, 'Volume')
                if prices is not None and len(prices) >= 30:
                    kalman_hull_results[ticker] = calculate_adaptive_kalman_hull(
                        prices, volume, risk_category=risk_category, ohlc_data=data
                    )
                else:
                    kalman_hull_results[ticker] = {
                        'trend': 0, 'kalman_price': np.nan, 'upper_band': np.nan,
                        'lower_band': np.nan, 'efficiency_ratio': 0.5,
                        'divergence': 'none', 'trend_consistency': False, 'signal_strength': 0.0
                    }
            except Exception as e:
                print(f"    Warning: Kalman Hull failed for {ticker}: {e}")
                kalman_hull_results[ticker] = {
                    'trend': 0, 'kalman_price': np.nan, 'upper_band': np.nan,
                    'lower_band': np.nan, 'efficiency_ratio': 0.5,
                    'divergence': 'none', 'trend_consistency': False, 'signal_strength': 0.0
                }
        
        # Run Volume Intelligence analysis
        volume_intelligence_results = {}
        for ticker, etf_data in risk_group_etfs.items():
            try:
                data = etf_data['data']  # Extract DataFrame from dict
                prices = extract_column(data, 'Close')
                volume = extract_column(data, 'Volume')
                if prices is not None and volume is not None and len(prices) >= 20:
                    volume_intelligence_results[ticker] = self.volume_intelligence.analyze_volume(prices, volume, ohlc_data=data)
                else:
                    volume_intelligence_results[ticker] = {
                        'spike_score': 0.0, 'price_volume_correlation': 0.0,
                        'accumulation_distribution': 'neutral', 'volume_confidence': 0.0
                    }
            except Exception as e:
                print(f"    Warning: Volume Intelligence failed for {ticker}: {e}")
                volume_intelligence_results[ticker] = {
                    'spike_score': 0.0, 'price_volume_correlation': 0.0,
                    'accumulation_distribution': 'neutral', 'volume_confidence': 0.0
                }
        
        # Combine results
        combined_results = {}
        for ticker in risk_group_etfs.keys():
            etf_data = risk_group_etfs[ticker]
            
            # Get component outputs
            risk_output = risk_results.get(ticker, {})
            ml_output = ml_results.get(ticker, {})
            kalman_output = kalman_hull_results.get(ticker, {})
            volume_output = volume_intelligence_results.get(ticker, {})
            
            combined_results[ticker] = {
                # Risk Component (30/30/20/20)
                'cvar': risk_output.get('cvar', np.nan),
                'ulcer_index': risk_output.get('ulcer_index', np.nan),
                'beta': risk_output.get('beta', etf_data.get('beta', 0.0)),
                'information_ratio': risk_output.get('information_ratio', np.nan),
                'risk_score': risk_output.get('risk_score', 0.5),
                'risk_category': risk_output.get('risk_category', risk_category),
                'quality_flag': risk_output.get('quality_flag', '~'),
                't_distribution_params': risk_output.get('t_distribution_params', {}),
                
                # ML Ensemble (NO bias correction)
                'ml_forecast': ml_output.get('forecast_return', 0.0),
                'ml_confidence': ml_output.get('confidence_score', 0.5),
                'mae_score': ml_output.get('mae_score', np.nan),
                'hit_rate': ml_output.get('hit_rate', np.nan),
                'features_used': ml_output.get('features_used', {}),
                'model_ensemble_output': ml_output.get('model_ensemble_output', 0.0),
                'feature_importance': ml_output.get('feature_importance', {}),
                
                # Liquidity metrics (from Risk Component)
                'amihud': risk_output.get('amihud', np.nan),
                'avg_daily_volume': risk_output.get('avg_daily_volume', np.nan),
                'zero_volume_days': risk_output.get('zero_volume_days', 0),
                
                # Kalman Hull Supertrend (NEW - integrated)
                'kalman_trend': kalman_output.get('trend', 0),
                'kalman_price': kalman_output.get('kalman_price', np.nan),
                'kalman_upper_band': kalman_output.get('upper_band', np.nan),
                'kalman_lower_band': kalman_output.get('lower_band', np.nan),
                'kalman_efficiency_ratio': kalman_output.get('efficiency_ratio', 0.5),
                'kalman_divergence': kalman_output.get('divergence', 'none'),
                'kalman_consistency': kalman_output.get('trend_consistency', False),
                'kalman_signal_strength': kalman_output.get('signal_strength', 0.0),
                
                # Volume Intelligence (NEW - integrated)
                'volume_spike_score': volume_output.get('spike_score', 0.0),
                'volume_correlation': volume_output.get('price_volume_correlation', 0.0),
                'volume_ad_signal': volume_output.get('accumulation_distribution', 'neutral'),
                'volume_confidence': volume_output.get('volume_confidence', 0.0),
                
                # Risk classification data
                'volatility': etf_data.get('volatility', 0.0),
                
                # Metadata
                'young_etf_penalty': risk_output.get('young_etf_penalty', 0.0),
                'group_classification': risk_output.get('group_classification', 'Unknown')
            }
            
            # Add fundamental data if available
            etf_info = self.etf_database.etf_data.get(ticker, {})
            combined_results[ticker]['expense_ratio'] = etf_info.get('expense_ratio', np.nan)
            combined_results[ticker]['aum_aud'] = etf_info.get('aum_aud', np.nan)
            
            # Calculate returns and latest price from data
            data = etf_data['data']
            if len(data) > 0:
                prices = extract_adjusted_price(data)
                if prices is not None and len(prices) > 0:
                    latest_price = float(prices.iloc[-1])
                    combined_results[ticker]['latest_price'] = latest_price
                    
                    # YTD return (from January 1st of current year)
                    from datetime import datetime
                    current_year = datetime.now().year
                    year_start = pd.Timestamp(f'{current_year}-01-01')
                    
                    # Filter prices from year start
                    ytd_prices = prices[prices.index >= year_start]
                    
                    if len(ytd_prices) > 1:
                        ytd_return = (ytd_prices.iloc[-1] - ytd_prices.iloc[0]) / ytd_prices.iloc[0]
                        combined_results[ticker]['ytd_return'] = float(ytd_return)
                    else:
                        combined_results[ticker]['ytd_return'] = 0.0
                    
                    # 1-year return (252 trading days) - adjusted for corporate actions
                    if len(prices) >= 252:
                        one_year_return = self.calculate_adjusted_return(prices, 252)
                        combined_results[ticker]['one_year_return'] = float(one_year_return)
                    else:
                        combined_results[ticker]['one_year_return'] = 0.0
                else:
                    combined_results[ticker]['latest_price'] = 0.0
                    combined_results[ticker]['ytd_return'] = 0.0
                    combined_results[ticker]['one_year_return'] = 0.0
            else:
                combined_results[ticker]['latest_price'] = 0.0
                combined_results[ticker]['ytd_return'] = 0.0
                combined_results[ticker]['one_year_return'] = 0.0
        
        return combined_results

    def analyze_risk_group_parallel(self, risk_group_etfs: Dict, risk_category: str, max_workers: int = None) -> Dict:
        """
        PHASE 1.4-1.5: Parallel version of analyze_risk_group using multiprocessing.Pool

        Parallelizes CPU-bound analysis:
        - ML Ensemble (forecast_etf + walk_forward_validate)
        - Kalman Hull indicator
        - Volume Intelligence

        Risk Component and result combining remain sequential (smaller overhead)

        Args:
            risk_group_etfs: Dict of {ticker: etf_data}
            risk_category: 'LOW', 'MEDIUM', or 'HIGH'
            max_workers: Number of processes (default: CPU count - 1)
        """
        print(f"  Analyzing {risk_category} group ({len(risk_group_etfs)} ETFs) [PARALLEL MODE]...")
        start_time = time.time()

        # Save historical data to disk for backtesting
        self._save_historical_data(risk_group_etfs)

        # Run Risk Component analysis (already handled separately, not parallelized)
        risk_results = self.risk_component.analyze_risk_group(risk_group_etfs, self.vix_data, self.benchmark_data)

        # ====================================================================
        # PHASE 1.4: Parallelize ML Ensemble (50% of runtime)
        # PHASE 1.6: With timeout handling and error recovery
        # ====================================================================
        print(f"    [ML_ENSEMBLE] Processing {len(risk_group_etfs)} ETFs with {max_workers or 'auto'} workers...")
        ml_start = time.time()

        ml_results = {}
        ml_errors = {}
        ml_timeouts = 0

        etf_list = list(risk_group_etfs.items())

        try:
            with Pool(processes=max_workers) as pool:
                for i, (ticker, etf_data) in enumerate(etf_list):
                    try:
                        async_result = pool.apply_async(_process_ml_ensemble_etf, ((ticker, etf_data),))
                        ticker_key, ml_output = async_result.get(timeout=120)  # 120s per ETF

                        # Check for errors
                        if isinstance(ml_output, dict) and 'error' in ml_output:
                            ml_errors[ticker] = ml_output['error']
                            ml_results[ticker] = _get_ml_ensemble_fallback(ticker)
                            print(f"    [ML_ENSEMBLE_ERROR] {ticker}: {ml_output['error']}")
                        else:
                            ml_results[ticker] = ml_output

                    except TimeoutError:
                        ml_timeouts += 1
                        ml_errors[ticker] = "Timeout (120s)"
                        ml_results[ticker] = _get_ml_ensemble_fallback(ticker)
                        print(f"    [ML_ENSEMBLE_TIMEOUT] {ticker}: Using fallback values")

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
        # ====================================================================
        print(f"    [KALMAN_HULL] Processing {len(risk_group_etfs)} ETFs...")
        kalman_start = time.time()

        kalman_hull_results = {}
        kalman_errors = {}
        kalman_timeouts = 0

        kalman_args = [(ticker, etf_data, risk_category) for ticker, etf_data in risk_group_etfs.items()]

        try:
            with Pool(processes=max_workers) as pool:
                for ticker, etf_data, risk_cat in kalman_args:
                    try:
                        async_result = pool.apply_async(_process_kalman_hull_etf, ((ticker, etf_data, risk_cat),))
                        ticker_key, kalman_output = async_result.get(timeout=60)  # 60s per ETF

                        # Check for errors
                        if isinstance(kalman_output, dict) and 'error' in kalman_output:
                            kalman_errors[ticker] = kalman_output['error']
                            kalman_hull_results[ticker] = _get_kalman_hull_fallback(ticker)
                            print(f"    [KALMAN_HULL_ERROR] {ticker}: {kalman_output['error']}")
                        else:
                            kalman_hull_results[ticker] = kalman_output

                    except TimeoutError:
                        kalman_timeouts += 1
                        kalman_errors[ticker] = "Timeout (60s)"
                        kalman_hull_results[ticker] = _get_kalman_hull_fallback(ticker)
                        print(f"    [KALMAN_HULL_TIMEOUT] {ticker}: Using fallback values")

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
        # Phase 1.5b: Parallelize Volume Intelligence
        # PHASE 1.6: With timeout handling and error recovery
        # ====================================================================
        print(f"    [VOLUME_INTEL] Processing {len(risk_group_etfs)} ETFs...")
        vol_start = time.time()

        volume_intelligence_results = {}
        vol_errors = {}
        vol_timeouts = 0

        vol_args = list(risk_group_etfs.items())

        try:
            with Pool(processes=max_workers) as pool:
                for ticker, etf_data in vol_args:
                    try:
                        async_result = pool.apply_async(_process_volume_intelligence_etf, ((ticker, etf_data),))
                        ticker_key, vol_output = async_result.get(timeout=60)  # 60s per ETF

                        # Check for errors
                        if isinstance(vol_output, dict) and 'error' in vol_output:
                            vol_errors[ticker] = vol_output['error']
                            volume_intelligence_results[ticker] = _get_volume_intelligence_fallback(ticker)
                            print(f"    [VOLUME_INTEL_ERROR] {ticker}: {vol_output['error']}")
                        else:
                            volume_intelligence_results[ticker] = vol_output

                    except TimeoutError:
                        vol_timeouts += 1
                        vol_errors[ticker] = "Timeout (60s)"
                        volume_intelligence_results[ticker] = _get_volume_intelligence_fallback(ticker)
                        print(f"    [VOLUME_INTEL_TIMEOUT] {ticker}: Using fallback values")

        except Exception as e:
            logger.error(f"Volume Intelligence pool error: {str(e)}")
            # Fallback: use default values for all remaining ETFs
            for ticker in risk_group_etfs.keys():
                if ticker not in volume_intelligence_results:
                    volume_intelligence_results[ticker] = _get_volume_intelligence_fallback(ticker)

        vol_time = time.time() - vol_start
        success_count = len(volume_intelligence_results) - len(vol_errors)
        print(f"    [VOLUME_INTEL_COMPLETE] {success_count}/{len(volume_intelligence_results)} ETFs in {vol_time:.1f}s ({vol_timeouts} timeouts)")

        # ====================================================================
        # Combine results (sequential, small overhead)
        # ====================================================================
        print(f"    [COMBINING] Aggregating {len(risk_group_etfs)} results...")
        combined_results = {}
        for ticker in risk_group_etfs.keys():
            etf_data = risk_group_etfs[ticker]

            # Get component outputs
            risk_output = risk_results.get(ticker, {})
            ml_output = ml_results.get(ticker, {})
            kalman_output = kalman_hull_results.get(ticker, {})
            volume_output = volume_intelligence_results.get(ticker, {})

            combined_results[ticker] = {
                # Risk Component (30/30/20/20)
                'cvar': risk_output.get('cvar', np.nan),
                'ulcer_index': risk_output.get('ulcer_index', np.nan),
                'beta': risk_output.get('beta', etf_data.get('beta', 0.0)),
                'information_ratio': risk_output.get('information_ratio', np.nan),
                'risk_score': risk_output.get('risk_score', 0.5),

                # ML Ensemble (25/10/10)
                'forecast_return': ml_output.get('forecast_return', 0.0),
                'confidence_score': ml_output.get('confidence_score', 0.5),
                'model_ensemble_output': ml_output.get('model_ensemble_output', 0.0),
                'mae_score': ml_output.get('mae_score', np.nan),
                'hit_rate': ml_output.get('hit_rate', np.nan),

                # Kalman Hull (20/20/20)
                'trend': kalman_output.get('trend', 0),
                'kalman_price': kalman_output.get('kalman_price', np.nan),
                'upper_band': kalman_output.get('upper_band', np.nan),
                'lower_band': kalman_output.get('lower_band', np.nan),
                'efficiency_ratio': kalman_output.get('efficiency_ratio', 0.5),
                'trend_consistency': kalman_output.get('trend_consistency', False),
                'signal_strength': kalman_output.get('signal_strength', 0.0),

                # Volume Intelligence (15/15/15)
                'spike_score': volume_output.get('spike_score', 0.0),
                'price_volume_correlation': volume_output.get('price_volume_correlation', 0.0),
                'accumulation_distribution': volume_output.get('accumulation_distribution', 'neutral'),
                'volume_confidence': volume_output.get('volume_confidence', 0.0),
            }

            # Add ETF info
            combined_results[ticker]['etf_name'] = etf_data.get('name', 'Unknown')

            # Calculate returns
            prices = extract_column(etf_data.get('data'), 'Close')
            if prices is not None and len(prices) > 0:
                combined_results[ticker]['latest_price'] = float(prices.iloc[-1]) if isinstance(prices.iloc[-1], (int, float)) else 0.0
                combined_results[ticker]['ytd_return'] = self.calculate_adjusted_return(prices, 252)
                combined_results[ticker]['one_year_return'] = self.calculate_adjusted_return(prices, 252)
            else:
                combined_results[ticker]['latest_price'] = 0.0
                combined_results[ticker]['ytd_return'] = 0.0
                combined_results[ticker]['one_year_return'] = 0.0

        total_time = time.time() - start_time
        print(f"    [PARALLEL_COMPLETE] Total time: {total_time:.1f}s (ML:{ml_time:.1f}s, Kalman:{kalman_time:.1f}s, VolIntel:{vol_time:.1f}s)")

        return combined_results

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
        
        # Step 3: Calculate composite scores and rankings
        print("\nCalculating composite scores and rankings...")
        rankings = self.scoring_system.rank_etfs_by_category(all_results, risk_categories_map)
        
        # Add composite scores back to analysis results
        for category, etf_list in rankings.items():
            for ticker, result in etf_list:
                if ticker in all_results:
                    all_results[ticker]['composite_score'] = result['composite_score']
                    # Store component scores (NEW - Phase 1.2)
                    all_results[ticker]['component_scores'] = result.get('components', {})
                    all_results[ticker]['adjusted_components'] = result.get('adjusted_components', {})
                    # Store position size recommendation
                    all_results[ticker]['position_size'] = result.get('position_size', 0.0)
        
        # Step 4: Get top ETFs
        top_opportunities = self.scoring_system.get_top_opportunities(rankings, top_n=10, min_score=0.0, focus_categories=['LOW', 'MEDIUM', 'HIGH'])
        
        # Convert to format expected by rest of system
        top_etfs = []
        for opp in top_opportunities:
            top_etfs.append({
                'ticker': opp['ticker'],
                'score': opp['composite_score'],
                'category': opp['risk_category']
            })
        
        print(f"\n{'='*60}")
        print("TOP 10 ETFs BY COMPOSITE SCORE")
        print(f"{'='*60}")
        for i, etf in enumerate(top_etfs, 1):
            print(f"{i:2d}. {etf['ticker']:10s} | Score: {etf['score']:5.1f} | Category: {etf['category']}")
        
        # Create original risk classifications map (with _risk_etfs suffix) for file saving
        original_risk_map = {}
        for risk_category, etfs in risk_classifications.items():
            for ticker in etfs.keys():
                original_risk_map[ticker] = risk_category
        
        return {
            'analysis_results': all_results,
            'rankings': rankings,
            'top_etfs': top_etfs,
            'risk_classifications': original_risk_map,  # Original names for file saving
            'risk_categories_normalized': risk_categories_map  # Normalized names for display
        }


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

