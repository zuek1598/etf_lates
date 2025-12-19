#!/usr/bin/env python3
"""
ETF Data Manager - Consolidated Data Access Layer
Combines: etf_database.py + dashboard_data_loader.py + fundamental_data_scraper.py

Single source of truth for all data operations:
- ETF metadata (367 ETFs)
- Analysis results (Parquet files)
- Fundamental data (expense ratios, AUM)
- Historical price data
- Validation results

Author: ETF Analysis System
Date: October 2025
Version: 2.0 (Consolidated)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, List
import json
import warnings
warnings.filterwarnings('ignore')


class ETFDataManager:
    """
    Unified data manager for all ETF data operations
    
    Responsibilities:
    1. Load and manage ETF metadata (367 ETFs)
    2. Load analysis results from Parquet files
    3. Manage fundamental data (expense ratios, AUM)
    4. Load and save all data formats
    5. Provide caching for performance
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize data manager
        
        Args:
            data_dir: Directory containing all data files
        """
        self.data_dir = Path(data_dir)
        
        # ETF metadata (from original etf_database.py)
        self.etf_data = {}
        self._load_etf_metadata()
        
        # Validate data directory exists
        if not self.data_dir.exists():
            print(f" Data directory not found: {self.data_dir.absolute()}")
            print(f"   Creating directory...")
            self.data_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # ETF METADATA (from etf_database.py)
    # ============================================================
    
    def _load_etf_metadata(self):
        """Load ETF metadata - 385 Australian ETFs"""
        # Try to import from etf_database in same directory
        try:
            from data_manager.etf_database import ETFDatabase
            original_db = ETFDatabase()
            self.etf_data = original_db.etf_data.copy()
            print(f"Loaded metadata for {len(self.etf_data)} ETFs from etf_database.py")
        except ImportError as e:
            # If old file not available, initialize empty (will load from Parquet)
            print(f" etf_database.py not found: {e}")
            print(" Will attempt to load metadata from Parquet")
            self.etf_data = {}
        except Exception as e:
            print(f" Error loading ETF database: {e}")
            self.etf_data = {}
    
    def get_etf_info(self, ticker: str) -> Optional[Dict]:
        """
        Get metadata for a specific ETF
        
        Args:
            ticker: ETF ticker symbol (e.g., 'VAS.AX')
            
        Returns:
            Dictionary with ETF metadata or None if not found
        """
        return self.etf_data.get(ticker)
    
    def get_all_tickers(self) -> List[str]:
        """
        Get list of all ETF tickers
        
        Loads from either etf_data dictionary or etf_universe.parquet
        """
        # First try from loaded metadata
        if self.etf_data:
            return list(self.etf_data.keys())
        
        # Fallback: load from universe parquet
        universe = self.load_universe()
        if not universe.empty and 'ticker' in universe.columns:
            tickers = universe['ticker'].tolist()
            print(f"Loaded {len(tickers)} tickers from universe parquet")
            return tickers
        
        # Last resort: empty list
        print(" No ETF tickers found")
        return []
    
    def get_etfs_by_region(self, region: str) -> List[str]:
        """Get tickers for ETFs in a specific region"""
        return [ticker for ticker, info in self.etf_data.items() 
                if info.get('region') == region]
    
    def get_etfs_by_type(self, etf_type: str) -> List[str]:
        """Get tickers for ETFs of a specific type"""
        return [ticker for ticker, info in self.etf_data.items() 
                if info.get('type') == etf_type]
    
    # ============================================================
    # ANALYSIS RESULTS (from dashboard_data_loader.py)
    # ============================================================
    
    @lru_cache(maxsize=1)
    def load_universe(self) -> pd.DataFrame:
        """
        Load complete ETF universe with all metrics
        
        Returns:
            DataFrame with all ETFs and their analysis results
            
        Columns include:
            ticker, name, subcategory, risk_category, beta, volatility,
            conditional_sharpe, ulcer_index, information_ratio,
            kama_signal, stochastic_k, stochastic_regime, vwap_position,
            ml_forecast, forecast_confidence, hit_rate,
            ytd_return, one_year_return, composite_score,
            param_cvar, param_var, hist_cvar, hist_var,
            amihud, avg_daily_volume, zero_volume_days, risk_score, etc.
        """
        file_path = self.data_dir / 'etf_universe.parquet'
        
        if not file_path.exists():
            print(f" Universe file not found: {file_path}")
            print(f"   Please run analysis first: python run_analysis.py")
            return pd.DataFrame()
        
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} ETFs from {file_path.name}")
        return df
    
    @lru_cache(maxsize=3)
    def load_rankings(self, risk_category: str) -> pd.DataFrame:
        """
        Load rankings for specific risk category
        
        Args:
            risk_category: 'low_risk', 'medium_risk', or 'high_risk'
            
        Returns:
            DataFrame with ranked ETFs for the specified risk category
        """
        valid_categories = ['low_risk', 'medium_risk', 'high_risk']
        
        if risk_category not in valid_categories:
            raise ValueError(
                f"Invalid risk_category: {risk_category}\n"
                f"Must be one of: {', '.join(valid_categories)}"
            )
        
        file_path = self.data_dir / f'rankings_{risk_category}.parquet'
        
        if not file_path.exists():
            print(f" No rankings file for {risk_category}")
            return pd.DataFrame()
        
        return pd.read_parquet(file_path)
    
    @lru_cache(maxsize=1)
    def load_metadata(self) -> Dict:
        """
        Load analysis metadata
        
        Returns:
            Dictionary with analysis date, ETF counts, processing time, etc.
        """
        file_path = self.data_dir / 'analysis_metadata.parquet'
        
        if not file_path.exists():
            return {
                'analysis_date': None,
                'total_etfs': 0,
                'processing_time': 0
            }
        
        df = pd.read_parquet(file_path)
        return df.iloc[0].to_dict()
    
    def get_etf_by_ticker(self, ticker: str) -> Optional[pd.Series]:
        """
        Get complete analysis data for a specific ETF
        
        Args:
            ticker: ETF ticker symbol (e.g., 'VAS.AX')
            
        Returns:
            Series with all ETF data, or None if not found
        """
        universe = self.load_universe()
        
        if universe.empty:
            return None
        
        matches = universe[universe['ticker'] == ticker]
        
        if matches.empty:
            return None
        
        return matches.iloc[0]
    
    def filter_universe(
        self, 
        risk_categories: Optional[List[str]] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        search_term: Optional[str] = None,
        min_liquidity: Optional[float] = None,
        max_expense_ratio: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter universe with multiple criteria
        
        Args:
            risk_categories: List of risk categories to include
            min_score: Minimum composite score
            max_score: Maximum composite score
            search_term: Search in ticker, name, or subcategory
            min_liquidity: Minimum average daily volume
            max_expense_ratio: Maximum expense ratio
            
        Returns:
            Filtered DataFrame
        """
        df = self.load_universe()
        
        if df.empty:
            return df
        
        # Filter by risk category
        if risk_categories:
            df = df[df['risk_category'].isin(risk_categories)]
        
        # Filter by score range
        if min_score is not None:
            df = df[df['composite_score'] >= min_score]
        if max_score is not None:
            df = df[df['composite_score'] <= max_score]
        
        # Filter by liquidity
        if min_liquidity is not None and 'avg_daily_volume' in df.columns:
            df = df[df['avg_daily_volume'] >= min_liquidity]
        
        # Filter by expense ratio
        if max_expense_ratio is not None and 'expense_ratio' in df.columns:
            df = df[df['expense_ratio'] <= max_expense_ratio]
        
        # Search filter
        if search_term:
            search_term = search_term.lower()
            mask = (
                df['ticker'].str.lower().str.contains(search_term, na=False) |
                df['name'].str.lower().str.contains(search_term, na=False) |
                df['subcategory'].str.lower().str.contains(search_term, na=False)
            )
            df = df[mask]
        
        return df
    
    # ============================================================
    # SPECIFIC DATA LOADERS
    # ============================================================
    
    def load_var_cvar(self) -> pd.DataFrame:
        """Load VaR and CVaR data for all ETFs"""
        universe = self.load_universe()
        if universe.empty:
            return pd.DataFrame()
        
        columns = ['ticker', 'hist_var', 'hist_cvar', 'param_var', 'param_cvar']
        available_cols = [col for col in columns if col in universe.columns]
        return universe[available_cols]
    
    def load_liquidity_metrics(self) -> pd.DataFrame:
        """Load liquidity metrics for all ETFs"""
        universe = self.load_universe()
        if universe.empty:
            return pd.DataFrame()
        
        columns = ['ticker', 'amihud', 'avg_daily_volume', 'zero_volume_days']
        available_cols = [col for col in columns if col in universe.columns]
        return universe[available_cols]
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics across all ETFs
        
        Returns:
            Dictionary with aggregate statistics
        """
        universe = self.load_universe()
        
        if universe.empty:
            return {
                'total_etfs': 0,
                'avg_composite_score': 0,
                'avg_sharpe': 0,
                'avg_ulcer': 0,
                'avg_ytd_return': 0,
                'avg_cvar': 0,
                'avg_daily_volume': 0,
                'risk_breakdown': {},
                'top_etf': 'N/A',
                'top_score': 0,
                'risk_free_rate': 0.0435
            }
        
        return {
            'total_etfs': len(universe),
            'avg_composite_score': universe['composite_score'].mean() if 'composite_score' in universe.columns else 0,
            'avg_sharpe': universe['conditional_sharpe'].mean() if 'conditional_sharpe' in universe.columns else 0,
            'avg_ulcer': universe['ulcer_index'].mean() if 'ulcer_index' in universe.columns else 0,
            'avg_ytd_return': universe['ytd_return'].mean() if 'ytd_return' in universe.columns else 0,
            'avg_cvar': universe['param_cvar'].mean() if 'param_cvar' in universe.columns else 0,
            'avg_daily_volume': universe['avg_daily_volume'].mean() if 'avg_daily_volume' in universe.columns else 0,
            'risk_breakdown': universe['risk_category'].value_counts().to_dict() if 'risk_category' in universe.columns else {},
            'top_etf': universe.nlargest(1, 'composite_score')['ticker'].iloc[0] if 'composite_score' in universe.columns and len(universe) > 0 else 'N/A',
            'top_score': universe['composite_score'].max() if 'composite_score' in universe.columns else 0,
            'risk_free_rate': 0.0435  # From config
        }
    
    # ============================================================
    # FUNDAMENTAL DATA (from fundamental_data_scraper.py)
    # ============================================================
    
    @lru_cache(maxsize=1)
    def load_fundamental_data(self) -> Dict:
        """
        Load fundamental data (expense ratios, AUM, inception dates)
        
        Returns:
            Dictionary with ticker -> {expense_ratio, aum_aud, inception_date}
        """
        fund_file = self.data_dir / 'fundamental_data.json'
        
        if not fund_file.exists():
            print(f" Fundamental data file not found: {fund_file}")
            return {}
        
        try:
            with open(fund_file, 'r') as f:
                data = json.load(f)
            print(f"Loaded fundamental data for {len(data)} ETFs")
            return data
        except Exception as e:
            print(f" Error loading fundamental data: {e}")
            return {}
    
    def get_fundamental_info(self, ticker: str) -> Optional[Dict]:
        """Get fundamental data for a specific ETF"""
        fundamentals = self.load_fundamental_data()
        return fundamentals.get(ticker)
    
    def save_fundamental_data(self, data: Dict):
        """
        Save fundamental data to JSON file
        
        Args:
            data: Dictionary with ticker -> {expense_ratio, aum_aud, inception_date}
        """
        fund_file = self.data_dir / 'fundamental_data.json'
        fund_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(fund_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved fundamental data for {len(data)} ETFs to {fund_file.name}")
        
        # Clear cache
        self.load_fundamental_data.cache_clear()
    
    # ============================================================
    # VALIDATION RESULTS
    # ============================================================
    
    @lru_cache(maxsize=1)
    def load_validation_results(self) -> Dict:
        """
        Load walk-forward validation results
        
        Returns:
            Dictionary with validation metrics per ETF
        """
        val_file = self.data_dir / 'validation_results.json'
        
        if not val_file.exists():
            return {}
        
        try:
            with open(val_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_validation_results(self, results: Dict):
        """Save walk-forward validation results"""
        val_file = self.data_dir / 'validation_results.json'
        val_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(val_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved validation results to {val_file.name}")
        
        # Clear cache
        self.load_validation_results.cache_clear()
    
    # ============================================================
    # HISTORICAL PRICE DATA
    # ============================================================
    
    def load_historical_prices(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load historical price data for a specific ETF
        
        Args:
            ticker: ETF ticker symbol
            
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        hist_file = self.data_dir / 'historical' / f'{ticker}.parquet'
        
        if not hist_file.exists():
            return None
        
        try:
            return pd.read_parquet(hist_file)
        except Exception as e:
            print(f" Error loading {ticker}: {e}")
            return None
    
    def save_historical_prices(self, ticker: str, data: pd.DataFrame):
        """Save historical price data for an ETF"""
        hist_dir = self.data_dir / 'historical'
        hist_dir.mkdir(parents=True, exist_ok=True)
        
        hist_file = hist_dir / f'{ticker}.parquet'
        data.to_parquet(hist_file)
    
    def load_historical_factors(self, ticker: str) -> pd.DataFrame:
        """
        Load historical daily factor values for a specific ETF.
        
        Returns:
            DataFrame with DatetimeIndex and factor columns, or None if not found
        """
        hist_file = self.data_dir / 'historical' / f'{ticker}.parquet'
        
        if not hist_file.exists():
            return None
        
        try:
            df = pd.read_parquet(hist_file)
            # Ensure Date is DatetimeIndex
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            return None
    
    # ============================================================
    # CACHE MANAGEMENT
    # ============================================================
    
    def clear_cache(self):
        """Clear all cached data (useful for reloading after new analysis)"""
        self.load_universe.cache_clear()
        self.load_rankings.cache_clear()
        self.load_metadata.cache_clear()
        self.load_fundamental_data.cache_clear()
        self.load_validation_results.cache_clear()
        print("Cache cleared")
    
    # ============================================================
    # CONVENIENCE METHODS
    # ============================================================
    
    def get_etf_complete(self, ticker: str) -> Dict:
        """
        Get ALL available data for one ETF
        
        Args:
            ticker: ETF ticker symbol
            
        Returns:
            Dictionary combining metadata, analysis results, fundamentals, and validation
        """
        result = {}
        
        # Metadata
        metadata = self.get_etf_info(ticker)
        if metadata:
            result.update(metadata)
        
        # Analysis results
        analysis = self.get_etf_by_ticker(ticker)
        if analysis is not None:
            result.update(analysis.to_dict())
        
        # Fundamental data
        fundamentals = self.get_fundamental_info(ticker)
        if fundamentals:
            result.update(fundamentals)
        
        # Validation results
        validation = self.load_validation_results().get(ticker)
        if validation:
            result['validation'] = validation
        
        return result
    
    def get_top_etfs(self, n: int = 10, risk_category: Optional[str] = None) -> pd.DataFrame:
        """
        Get top N ETFs by composite score
        
        Args:
            n: Number of ETFs to return
            risk_category: Optional risk category filter
            
        Returns:
            DataFrame with top ETFs
        """
        universe = self.load_universe()
        
        if universe.empty:
            return pd.DataFrame()
        
        if risk_category:
            universe = universe[universe['risk_category'] == risk_category]
        
        return universe.nlargest(n, 'composite_score')


# ============================================================
# GLOBAL INSTANCE (Singleton Pattern)
# ============================================================

_data_manager = None

def get_data_manager(data_dir='data') -> ETFDataManager:
    """
    Get global data manager instance (singleton pattern)
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        ETFDataManager instance
    """
    global _data_manager
    if _data_manager is None:
        _data_manager = ETFDataManager(data_dir)
    return _data_manager


# ============================================================
# CONVENIENCE FUNCTIONS (For Backward Compatibility)
# ============================================================

def load_universe(data_dir='data') -> pd.DataFrame:
    """Load complete ETF universe"""
    return get_data_manager(data_dir).load_universe()


def load_rankings(risk_category: str, data_dir='data') -> pd.DataFrame:
    """Load rankings for specific risk category"""
    return get_data_manager(data_dir).load_rankings(risk_category)


def load_metadata(data_dir='data') -> Dict:
    """Load analysis metadata"""
    return get_data_manager(data_dir).load_metadata()


def get_etf(ticker: str, data_dir='data') -> Optional[pd.Series]:
    """Get specific ETF by ticker"""
    return get_data_manager(data_dir).get_etf_by_ticker(ticker)


# ============================================================
# EXAMPLE USAGE & TESTING
# ============================================================

if __name__ == "__main__":
    import time
    
    print("ETF DATA MANAGER - Performance Test")
    print("=" * 60)
    
    try:
        # Initialize
        manager = ETFDataManager()
        
        # Test metadata
        print("\n1. ETF Metadata:")
        print(f"   Total ETFs: {len(manager.get_all_tickers())}")
        vas_info = manager.get_etf_info('VAS.AX')
        if vas_info:
            print(f"   VAS.AX: {vas_info}")
        
        # Test universe loading
        print("\n2. Loading universe...")
        start = time.time()
        universe = manager.load_universe()
        load_time = time.time() - start
        if not universe.empty:
            print(f"   Loaded {len(universe)} ETFs in {load_time*1000:.1f}ms")
            
            # Test cached loading
            start = time.time()
            universe = manager.load_universe()
            cached_time = time.time() - start
            speedup = load_time/cached_time if cached_time > 0 else float('inf')
            print(f"   Cached load: {cached_time*1000:.1f}ms ({speedup:.0f}x faster)")
        else:
            print(f"    No universe data found (run analysis first)")
        
        # Test rankings
        print("\n3. Risk Rankings:")
        for risk in ['low_risk', 'medium_risk', 'high_risk']:
            rankings = manager.load_rankings(risk)
            print(f"   {risk}: {len(rankings)} ETFs")
        
        # Test fundamental data
        print("\n4. Fundamental Data:")
        fundamentals = manager.load_fundamental_data()
        print(f"   Loaded data for {len(fundamentals)} ETFs")
        if 'VAS.AX' in fundamentals:
            print(f"   VAS.AX: {fundamentals['VAS.AX']}")
        
        # Test summary stats
        print("\n5. Summary Statistics:")
        stats = manager.get_summary_stats()
        print(f"   Total ETFs: {stats['total_etfs']}")
        print(f"   Avg Score: {stats['avg_composite_score']:.1f}")
        print(f"   Top ETF: {stats['top_etf']} ({stats['top_score']:.1f})")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run analysis first:")
        print("   python run_analysis.py")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
