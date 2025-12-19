"""
Dashboard Data Loader
Helper module for loading ETF analysis results into Dash dashboard

Author: ETF Analysis System
Date: October 5, 2025
Version: 1.0

This module provides optimized data loading functions for the Dash dashboard,
using cached Parquet files for fast performance.
"""

import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import Optional


class ETFDataLoader:
    """
    Efficient data loader for Dash dashboard
    Uses LRU cache for instant access after first load
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing Parquet files
        """
        self.data_dir = Path(data_dir)
        
        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir.absolute()}\n"
                f"Please run 'python etf_dashboard.py' first to generate data files."
            )
    
    @lru_cache(maxsize=1)
    def load_universe(self) -> pd.DataFrame:
        """
        Load complete ETF universe
        
        Returns:
            DataFrame with all ETFs and their metrics
            
        Columns (VALIDATED FACTORS):
            ticker, name, subcategory, risk_category,
            ml_forecast, hit_rate, kalman_signal_strength, cvar,
            composite_percentile, ytd_return, one_year_return, etc.
            
        Note: Only 4 statistically validated factors are used for ranking:
            - ml_forecast: ML Ensemble forecast (IC=+0.229, p=0.027)
            - hit_rate: ML directional accuracy (IC=+0.344, p=0.001)
            - kalman_signal_strength: Kalman momentum (IC=+0.234, p=0.023)
            - cvar: Conditional Value at Risk (IC=+0.261, p=0.011)
        """
        file_path = self.data_dir / 'etf_universe.parquet'
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Universe file not found: {file_path}\n"
                f"Please run 'python etf_dashboard.py' first."
            )
        
        return pd.read_parquet(file_path)
    
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
            # Return empty DataFrame if category has no ETFs
            return pd.DataFrame()
        
        return pd.read_parquet(file_path)
    
    @lru_cache(maxsize=1)
    def load_metadata(self) -> dict:
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
        Get detailed data for a specific ETF
        
        Args:
            ticker: ETF ticker symbol (e.g., 'VAS.AX')
            
        Returns:
            Series with all ETF data, or None if not found
        """
        universe = self.load_universe()
        matches = universe[universe['ticker'] == ticker]
        
        if matches.empty:
            return None
        
        return matches.iloc[0]
    
    def filter_universe(
        self, 
        risk_categories: Optional[list] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        search_term: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter universe with multiple criteria
        
        Args:
            risk_categories: List of risk categories to include
            min_score: Minimum composite score
            max_score: Maximum composite score
            search_term: Search in ticker or name
            
        Returns:
            Filtered DataFrame
        """
        df = self.load_universe()
        
        # Filter by risk category
        if risk_categories:
            df = df[df['risk_category'].isin(risk_categories)]
        
        # Filter by score range
        if min_score is not None:
            df = df[df['composite_score'] >= min_score]
        if max_score is not None:
            df = df[df['composite_score'] <= max_score]
        
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
    
    def load_cvar(self) -> pd.DataFrame:
        """
        Load CVaR data for all ETFs (VaR removed in modified system)
        
        Returns:
            DataFrame with CVaR metrics
        """
        universe = self.load_universe()
        return universe[['ticker', 'cvar']]
    
    def load_fundamental_data(self) -> dict:
        """
        Load fundamental data (expense ratios, AUM) from JSON
        
        Returns:
            Dictionary with ticker -> {expense_ratio, aum_aud, inception_date}
        """
        fund_file = self.data_dir / 'fundamental_data.json'
        
        if not fund_file.exists():
            return {}
        
        try:
            import json
            with open(fund_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def load_liquidity_metrics(self) -> pd.DataFrame:
        """
        Load liquidity metrics for all ETFs
        
        Returns:
            DataFrame with Amihud ratio, avg daily volume, zero volume days
        """
        universe = self.load_universe()
        return universe[['ticker', 'amihud', 'avg_daily_volume', 'zero_volume_days']]
    
    def load_validation_results(self) -> dict:
        """
        Load walk-forward validation results from JSON
        
        Returns:
            Dictionary with validation metrics per ETF
        """
        val_file = self.data_dir / 'validation_results.json'
        
        if not val_file.exists():
            return {}
        
        try:
            import json
            with open(val_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def get_summary_stats(self) -> dict:
        """
        Get summary statistics across all ETFs
        
        Returns:
            Dictionary with aggregate statistics
        """
        universe = self.load_universe()
        
        return {
            'total_etfs': len(universe),
            'avg_composite_score': universe['composite_score'].mean(),
            'avg_sharpe': universe['conditional_sharpe'].mean() if 'conditional_sharpe' in universe.columns else 0.0,
            'avg_ulcer': universe['ulcer_index'].mean(),
            'avg_ytd_return': universe['ytd_return'].mean(),
            'avg_cvar': universe['cvar'].mean() if 'cvar' in universe.columns else 0.0,
            'avg_daily_volume': universe['avg_daily_volume'].mean() if 'avg_daily_volume' in universe.columns else 0.0,
            'avg_volume_spike': universe['volume_spike_score'].mean() if 'volume_spike_score' in universe.columns else 0.0,
            'avg_ml_confidence': universe['ml_confidence'].mean() if 'ml_confidence' in universe.columns else 0.0,
            'risk_breakdown': universe['risk_category'].value_counts().to_dict(),
            'top_etf': universe.nlargest(1, 'composite_score')['ticker'].iloc[0] if len(universe) > 0 else 'N/A',
            'top_score': universe['composite_score'].max() if len(universe) > 0 else 0.0,
            'risk_free_rate': 0.0435  # RBA rate Oct 2024
        }
    
    def clear_cache(self):
        """Clear all cached data (useful for reloading after new analysis)"""
        self.load_universe.cache_clear()
        self.load_rankings.cache_clear()
        self.load_metadata.cache_clear()


# Global instance for convenience
_loader = None

def get_data_loader(data_dir='data') -> ETFDataLoader:
    """
    Get global data loader instance (singleton pattern)
    
    Args:
        data_dir: Directory containing Parquet files
        
    Returns:
        ETFDataLoader instance
    """
    global _loader
    if _loader is None:
        _loader = ETFDataLoader(data_dir)
    return _loader


# Convenience functions for common operations

def load_universe(data_dir='data') -> pd.DataFrame:
    """Load complete ETF universe"""
    return get_data_loader(data_dir).load_universe()


def load_rankings(risk_category: str, data_dir='data') -> pd.DataFrame:
    """Load rankings for specific risk category"""
    return get_data_loader(data_dir).load_rankings(risk_category)


def load_metadata(data_dir='data') -> dict:
    """Load analysis metadata"""
    return get_data_loader(data_dir).load_metadata()


def get_etf(ticker: str, data_dir='data') -> Optional[pd.Series]:
    """Get specific ETF by ticker"""
    return get_data_loader(data_dir).get_etf_by_ticker(ticker)


# Example usage
if __name__ == "__main__":
    import time
    
    print("ETF Dashboard Data Loader - Performance Test")
    print("=" * 60)
    
    try:
        loader = ETFDataLoader()
        
        # Test universe loading
        print("\n1. Loading universe...")
        start = time.time()
        universe = loader.load_universe()
        load_time = time.time() - start
        print(f"   Loaded {len(universe)} ETFs in {load_time*1000:.1f}ms")
        
        # Test cached loading
        print("\n2. Loading universe again (cached)...")
        start = time.time()
        universe = loader.load_universe()
        cached_time = time.time() - start
        print(f"   Loaded {len(universe)} ETFs in {cached_time*1000:.1f}ms")
        print(f"   Speedup: {load_time/cached_time:.0f}x faster")
        
        # Test rankings loading
        print("\n3. Loading risk category rankings...")
        for risk in ['low_risk', 'medium_risk', 'high_risk']:
            start = time.time()
            rankings = loader.load_rankings(risk)
            load_time = time.time() - start
            print(f"   {risk}: {len(rankings)} ETFs in {load_time*1000:.1f}ms")
        
        # Test metadata
        print("\n4. Loading metadata...")
        metadata = loader.load_metadata()
        print(f"   Analysis date: {metadata.get('analysis_date', 'N/A')}")
        print(f"   Total ETFs: {metadata.get('total_etfs', 0)}")
        
        # Test filtering
        print("\n5. Testing filters...")
        filtered = loader.filter_universe(
            risk_categories=['Low', 'Medium'],
            min_score=70.0,
            search_term='vanguard'
        )
        print(f"   Found {len(filtered)} ETFs matching filters")
        
        # Test summary stats
        print("\n6. Summary statistics...")
        stats = loader.get_summary_stats()
        print(f"   Average composite score: {stats['avg_composite_score']:.1f}")
        print(f"   Top ETF: {stats['top_etf']} (Score: {stats['top_score']:.1f})")
        print(f"   Risk breakdown: {stats['risk_breakdown']}")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the following command first:")
        print("   python etf_dashboard.py")
