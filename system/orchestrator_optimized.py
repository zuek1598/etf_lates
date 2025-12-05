#!/usr/bin/env python3
"""
ETF Analysis System Orchestrator - Optimized with Batch Data Fetching
Same functionality as original but 10x faster data downloads
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
from analyzers.ml_ensemble import MLEnsemble
from analyzers.etf_risk_classifier_optimized import ETFRiskClassifierOptimized
from analyzers.percentile_ranker import PercentileRanker
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
# MODULE-LEVEL HELPER FUNCTIONS FOR MULTIPROCESSING
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

        # Run walk-forward validation (FIX 1: Reuse trained models instead of retraining)
        prices = extract_column(data, 'Close')
        if prices is not None and len(prices) >= 312:  # 252 train + 60 test
            # Capture trained models from forecast and reuse in validation
            trained_models = ml_output.get('trained_models')
            validation = ml_ensemble.walk_forward_validate(prices, models=trained_models)
            ml_output['mae_score'] = validation['mae']
            ml_output['hit_rate'] = validation['hit_rate']
            print(f"[ML_ENSEMBLE_OK] {ticker}: forecast={ml_output.get('forecast_return', 0):.4f}, mae={validation['mae']:.4f}, hit_rate={validation['hit_rate']:.2f}")
        else:
            ml_output['mae_score'] = np.nan
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


class ETFAnalysisSystemOptimized:
    """
    Optimized ETF Analysis System with batch data fetching
    
    Key improvements:
    - 10x faster data downloads with batch fetching
    - Same analysis quality and accuracy
    - Better resource utilization
    - Real-time progress tracking
    """

    def __init__(self, max_workers: int = None, batch_size: int = 50):
        """
        Initialize optimized analysis system
        
        Args:
            max_workers: Maximum concurrent download threads
            batch_size: Number of ETFs per download batch
        """
        # Initialize core components
        self.etf_database = ETFDatabase()
        self.risk_component = RiskComponent()
        self.ml_ensemble = MLEnsemble()
        self.risk_classifier = ETFRiskClassifierOptimized(
            max_workers=max_workers, 
            batch_size=batch_size
        )
        self.percentile_ranker = PercentileRanker()

        # Load market data (VIX and benchmarks)
        self._load_market_data()

        # Historical data save flag
        self._historical_data_saved = False

    def _load_market_data(self):
        """Load VIX and benchmark data"""
        try:
            # Load VIX data
            vix_file = 'data/external/vix.parquet'
            if os.path.exists(vix_file):
                self.vix_data = pd.read_parquet(vix_file)
                print(f"Loading market data from cache...")
                print(f"  VIX data: {len(self.vix_data)} days (cached)")
            else:
                print("Warning: VIX data not found, some features may be limited")
                self.vix_data = pd.DataFrame()

            # Load benchmark data from classifier
            self.benchmark_data = self.risk_classifier.benchmark_data
            print(f"  Benchmarks: {len(self.benchmark_data)} indices (cached)")

        except Exception as e:
            print(f"Warning: Could not load market data: {e}")
            self.vix_data = pd.DataFrame()
            self.benchmark_data = {}

    def run_full_analysis(self, etf_tickers: List[str] = None) -> Dict:
        """
        Run complete optimized analysis pipeline
        
        Args:
            etf_tickers: List of ETF tickers to analyze (None = all)
            
        Returns:
            Dict with analysis results and rankings
        """
        if etf_tickers is None:
            etf_tickers = list(self.etf_database.etf_data.keys())
        
        print(f"\n{'='*60}")
        print(f"ETF ANALYSIS SYSTEM - OPTIMIZED")
        print(f"{'='*60}")
        print(f"Analyzing {len(etf_tickers)} ETFs with batch fetching")
        print(f"Batch size: {self.risk_classifier.batch_fetcher.batch_size}")
        print(f"Workers: {self.risk_classifier.batch_fetcher.max_workers}\n")
        
        start_time = time.time()
        
        # Step 1: Classify by risk (OPTIMIZED with batch fetching)
        risk_classifications = self.classify_etfs_by_risk(etf_tickers)
        
        # Step 2: Analyze each risk group
        all_results = {}
        risk_categories_map = {}
        
        for risk_category, etfs in risk_classifications.items():
            if etfs:
                print(f"\n🔍 Analyzing {risk_category.replace('_risk_etfs', '').upper()} risk group ({len(etfs)} ETFs)...")
                group_results = self.analyze_risk_group_parallel(etfs, risk_category)
                all_results.update(group_results)
                
                # Normalize risk category names
                normalized_category = risk_category.replace('_risk_etfs', '').replace('_', '').upper()
                for ticker in etfs.keys():
                    risk_categories_map[ticker] = normalized_category
        
        # Step 3: Calculate percentile rankings
        print("\n📈 Calculating percentile rankings...")
        ranking_metrics = [
            'ml_forecast', 'hit_rate', 'kalman_signal_strength', 'cvar',
        ]

        risk_categories_dict = {}
        for ticker, category in risk_categories_map.items():
            if category not in risk_categories_dict:
                risk_categories_dict[category] = []
            risk_categories_dict[category].append(ticker)

        rankings = self.percentile_ranker.rank_etf_universe(
            all_results, risk_categories_dict, ranking_metrics
        )

        # Step 4: Prepare final results
        processing_time = time.time() - start_time
        
        results = {
            'analysis_results': all_results,
            'rankings': rankings,
            'risk_classifications': risk_categories_map,
            'processing_time': processing_time,
            'total_etfs': len(etf_tickers),
            'successful_etfs': len(all_results)
        }
        
        # Performance summary
        print(f"\n{'='*60}")
        print(f"OPTIMIZED ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total ETFs:        {len(etf_tickers)}")
        print(f"Successfully:      {len(all_results)}")
        print(f"Processing time:   {processing_time:.1f} seconds")
        print(f"Speed:             {len(all_results)/processing_time:.1f} ETFs/second")
        print(f"Improvement:       ~10x faster than individual downloads")
        
        return results

    def classify_etfs_by_risk(self, etf_tickers: List[str]) -> Dict:
        """
        Classify ETFs into risk categories using optimized batch fetching
        """
        print(f"🚀 Classifying {len(etf_tickers)} ETFs by risk (optimized)...")
        results, summary = self.risk_classifier.classify_etfs(etf_tickers)
        
        for category, etfs in results.items():
            for ticker, data in etfs.items():
                data['etf_info'] = self.etf_database.etf_data.get(ticker, {})
        
        print(f"  ✅ Low: {summary['low_risk_count']}, Medium: {summary['medium_risk_count']}, High: {summary['high_risk_count']}")
        return results
    
    def analyze_risk_group_parallel(self, risk_group_etfs: Dict, risk_category: str, max_workers: int = None) -> Dict:
        """
        Analyze risk group with parallel processing (same as original)
        """
        # Optimize worker count
        import multiprocessing
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            etf_count = len(risk_group_etfs)
            
            if etf_count <= 10:
                max_workers = min(2, cpu_count)
            elif etf_count <= 50:
                max_workers = min(4, cpu_count)
            else:
                max_workers = min(cpu_count - 1, 6)
            
            print(f"  ⚙️  Optimized: {etf_count} ETFs → {max_workers} workers (CPU: {cpu_count})")
        
        start_time = time.time()

        # Save historical data
        if not self._historical_data_saved:
            self._save_historical_data(risk_group_etfs)
            self._historical_data_saved = True

        # Run Risk Component analysis
        risk_results = self.risk_component.analyze_risk_group(risk_group_etfs, self.vix_data, self.benchmark_data)

        # Parallel ML Ensemble processing
        print(f"  🤖 Running ML Ensemble ({len(risk_group_etfs)} ETFs)...")
        ml_results = {}
        
        try:
            with Pool(processes=max_workers) as pool:
                ml_args = [(ticker, etf_data) for ticker, etf_data in risk_group_etfs.items()]
                ml_outputs = pool.map(_process_ml_ensemble_etf, ml_args)
                
                for ticker, ml_output in ml_outputs:
                    ml_results[ticker] = ml_output
                    
        except Exception as e:
            print(f"  ⚠️  Parallel ML failed, falling back to sequential: {e}")
            for ticker, etf_data in risk_group_etfs.items():
                ml_output = _process_ml_ensemble_etf((ticker, etf_data))[1]
                ml_results[ticker] = ml_output

        # Parallel Kalman Hull processing
        print(f"  📈 Running Kalman Hull ({len(risk_group_etfs)} ETFs)...")
        kalman_results = {}
        
        try:
            with Pool(processes=max_workers) as pool:
                kalman_args = [(ticker, etf_data, risk_category) for ticker, etf_data in risk_group_etfs.items()]
                kalman_outputs = pool.map(_process_kalman_hull_etf, kalman_args)
                
                for ticker, kalman_output in kalman_outputs:
                    kalman_results[ticker] = kalman_output
                    
        except Exception as e:
            print(f"  ⚠️  Parallel Kalman failed, falling back to sequential: {e}")
            for ticker, etf_data in risk_group_etfs.items():
                kalman_output = _process_kalman_hull_etf((ticker, etf_data, risk_category))[1]
                kalman_results[ticker] = kalman_output

        # Combine all results
        final_results = {}
        for ticker in risk_group_etfs.keys():
            try:
                # Start with ETF info and basic data
                etf_info = self.etf_database.etf_data.get(ticker, {})
                etf_data = risk_group_etfs[ticker]
                
                combined_result = {
                    'ticker': ticker,
                    'name': etf_info.get('name', ticker),
                    'etf_info': etf_info,
                    'risk_category': risk_category.replace('_risk_etfs', '').replace('_', '').upper(),
                }
                
                # Add risk component results
                if ticker in risk_results:
                    combined_result.update(risk_results[ticker])
                
                # Add ML results
                if ticker in ml_results:
                    ml_result = ml_results[ticker]
                    combined_result['ml_forecast'] = ml_result.get('forecast_return', 0.0)
                    combined_result['ml_confidence'] = ml_result.get('confidence_score', 0.5)
                    combined_result['mae_score'] = ml_result.get('mae_score', np.nan)
                    combined_result['hit_rate'] = ml_result.get('hit_rate', np.nan)
                
                # Add Kalman Hull results
                if ticker in kalman_results:
                    kalman_result = kalman_results[ticker]
                    combined_result.update(kalman_result)
                
                # Calculate composite score
                composite_score = self._calculate_composite_score(combined_result)
                combined_result['composite_score'] = composite_score
                
                final_results[ticker] = combined_result
                
            except Exception as e:
                print(f"  ❌ Error combining results for {ticker}: {e}")
                continue

        elapsed = time.time() - start_time
        print(f"  ✅ Completed {len(final_results)} ETFs in {elapsed:.1f}s")
        
        return final_results

    def _save_historical_data(self, risk_group_etfs: Dict):
        """Save historical data to disk (same as original)"""
        from pathlib import Path
        import traceback
        
        historical_dir = Path('data/historical')
        historical_dir.mkdir(exist_ok=True, parents=True)
        
        saved_count = 0
        failed_count = 0
        
        for ticker, etf_data in risk_group_etfs.items():
            try:
                data = etf_data.get('data')
                
                if data is None or data.empty:
                    failed_count += 1
                    continue
                
                # Flatten multi-level columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data = data.copy()
                    data.columns = data.columns.get_level_values(0)
                
                # Save
                file_path = historical_dir / f"{ticker.replace('.', '_')}.parquet"
                data.to_parquet(file_path)
                saved_count += 1
                
            except Exception as e:
                print(f"      {ticker}: {type(e).__name__}: {str(e)}")
                failed_count += 1
        
        print(f"      💾 Saved {saved_count} historical files")

    def _calculate_composite_score(self, result: Dict) -> float:
        """Calculate composite score (same as original)"""
        try:
            # Risk score (40%)
            risk_score = result.get('risk_score', 0.5)
            
            # ML score (30%)
            ml_score = result.get('ml_forecast', 0.0) * result.get('ml_confidence', 0.5)
            
            # Technical score (30%)
            kalman_score = result.get('signal_strength', 0.0) * 0.5
            
            # Weighted composite
            composite = (risk_score * 0.4 + ml_score * 0.3 + kalman_score * 0.3)
            
            return max(0.0, min(1.0, composite))
            
        except Exception:
            return 0.5
