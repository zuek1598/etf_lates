#!/usr/bin/env python3
"""
External Data Manager for Regime Detection
Fetches and manages VIX, AUD/USD, AU/US yields, and Gold price data

Data Sources:
- VIX: ^VIX (CBOE Volatility Index)
- AUD/USD: AUDUSD=X (FX rate)
- AU 10Y Yield: GACGB10Y (Australian Government Bond)
- US 10Y Yield: ^TNX (US Treasury Note)
- Gold: GOLD-AU.AX (ASX-listed Gold ETF)

Usage:
    from data_manager.external_data import ExternalDataManager
    
    data_mgr = ExternalDataManager()
    external_data = data_mgr.fetch_all_data()
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import os
import warnings
warnings.filterwarnings('ignore')

class ExternalDataManager:
    """Manages external market data for regime detection and correlation analysis"""
    
    def __init__(self, data_dir: str = "data/external"):
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # Data source configuration
        self.data_sources = {
            'vix': {
                'ticker': '^VIX',
                'name': 'VIX Volatility Index',
                'description': 'CBOE Volatility Index - Market fear gauge'
            },
            'aud_usd': {
                'ticker': 'AUDUSD=X',
                'name': 'AUD/USD Exchange Rate',
                'description': 'Australian Dollar to US Dollar exchange rate'
            },
            'au_10y': {
                'ticker': '^TNX',  # Use US 10Y as proxy for AU bonds
                'name': 'US 10Y Treasury Yield (AU Proxy)',
                'description': 'US 10-Year Treasury Note Yield used as AU bond proxy'
            },
            'us_10y': {
                'ticker': '^TNX',
                'name': 'US 10Y Treasury Yield',
                'description': 'US 10-Year Treasury Note Yield'
            },
            'gold': {
                'ticker': 'GOLD.AX',
                'name': 'Gold Price (AUD)',
                'description': 'Gold price in Australian Dollars via ASX ETF'
            }
        }
        
        # Data parameters
        self.lookback_years = 5  # 5 years as per strategy
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=self.lookback_years * 365)
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_single_series(self, key: str, force_refresh: bool = False) -> Optional[pd.Series]:
        """
        Fetch a single external data series
        
        Args:
            key: Data source key (e.g., 'vix', 'aud_usd')
            force_refresh: Force download even if cached data exists
            
        Returns:
            pandas Series with date index and values
        """
        if key not in self.data_sources:
            print(f"❌ Unknown data source: {key}")
            return None
        
        source_config = self.data_sources[key]
        ticker = source_config['ticker']
        cache_file = os.path.join(self.data_dir, f"{key}.parquet")
        
        # Check if cached data exists and is recent
        if not force_refresh and os.path.exists(cache_file):
            try:
                cached_data = pd.read_parquet(cache_file)
                # Check if data is recent (less than 1 day old)
                cache_age = (datetime.now() - cached_data.index[-1]).days
                if cache_age < 1:
                    print(f"📁 Using cached {source_config['name']} ({len(cached_data)} points)")
                    return cached_data
            except Exception as e:
                print(f"⚠️ Cache read error for {key}: {e}")
        
        # Fetch fresh data
        print(f"📥 Fetching {source_config['name']} ({ticker})...")
        
        try:
            # Download data using yfinance
            data = yf.download(
                ticker,
                start=self.start_date.strftime('%Y-%m-%d'),
                end=self.end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=False
            )
            
            if data.empty:
                print(f"❌ No data found for {ticker}")
                return None
            
            # Extract Close prices and clean data
            series = data['Close'].copy()
            series = series.dropna()
            
            if len(series) < 100:  # Minimum data requirement
                print(f"❌ Insufficient data for {ticker}: {len(series)} points")
                return None
            
            # Save to cache
            try:
                series.to_parquet(cache_file)
                print(f"💾 Cached {source_config['name']} ({len(series)} points)")
            except Exception as e:
                print(f"⚠️ Cache write error for {key}: {e}")
            
            print(f"✅ Fetched {source_config['name']}: {len(series)} points ({series.index[0].date()} to {series.index[-1].date()})")
            return series
            
        except Exception as e:
            print(f"❌ Failed to fetch {ticker}: {e}")
            return None
    
    def fetch_all_data(self, force_refresh: bool = False) -> Dict[str, pd.Series]:
        """
        Fetch all external data series
        
        Args:
            force_refresh: Force download even if cached data exists
            
        Returns:
            Dictionary with data keys as keys and pandas Series as values
        """
        print(f"\n{'='*60}")
        print("EXTERNAL DATA FETCH - PHASE 2 INITIALIZATION")
        print(f"{'='*60}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Sources: {len(self.data_sources)} series")
        
        external_data = {}
        
        for key in self.data_sources.keys():
            series = self.fetch_single_series(key, force_refresh)
            if series is not None:
                external_data[key] = series
            else:
                print(f"⚠️ Skipping {key} due to fetch failure")
        
        print(f"\n✅ Successfully fetched {len(external_data)}/{len(self.data_sources)} series")
        
        # Display data summary
        if external_data:
            print(f"\n📊 Data Summary:")
            for key, series in external_data.items():
                config = self.data_sources[key]
                print(f"  • {config['name']}: {len(series)} points")
                # Fix pandas Series formatting
                min_val = float(series.min())
                max_val = float(series.max())
                latest_val = float(series.iloc[-1])
                print(f"    Range: {min_val:.2f} to {max_val:.2f}")
                print(f"    Latest: {latest_val:.2f} ({series.index[-1].date()})")
        
        return external_data
    
    def get_data_summary(self, data: Dict[str, pd.Series]) -> Dict:
        """
        Generate summary statistics for fetched data
        
        Args:
            data: Dictionary of external data series
            
        Returns:
            Summary statistics dictionary
        """
        if not data:
            return {}
        
        summary = {
            'total_series': len(data),
            'date_range': {},
            'statistics': {}
        }
        
        # Find common date range
        all_dates = []
        for series in data.values():
            all_dates.extend(series.index)
        
        if all_dates:
            summary['date_range'] = {
                'start': min(all_dates).date(),
                'end': max(all_dates).date(),
                'total_days': len(set(all_dates))
            }
        
        # Calculate statistics for each series
        for key, series in data.items():
            config = self.data_sources[key]
            summary['statistics'][key] = {
                'name': config['name'],
                'count': len(series),
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'std': float(series.std()),
                'latest': float(series.iloc[-1]),
                'latest_date': series.index[-1].date()
            }
        
        return summary
    
    def validate_data_quality(self, data: Dict[str, pd.Series]) -> Dict[str, bool]:
        """
        Validate data quality for each series
        
        Args:
            data: Dictionary of external data series
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        for key, series in data.items():
            is_valid = True
            issues = []
            
            # Check minimum data length
            if len(series) < 100:
                is_valid = False
                issues.append(f"Insufficient data: {len(series)} < 100")
            
            # Check for excessive missing data
            missing_count = series.isna().sum()
            missing_pct = float(missing_count / len(series) * 100)
            if missing_pct > 5:
                is_valid = False
                issues.append(f"Too much missing data: {missing_pct:.1f}%")
            
            # Check for stale data
            days_since_last = (datetime.now().date() - series.index[-1].date()).days
            if days_since_last > 7:
                is_valid = False
                issues.append(f"Stale data: {days_since_last} days old")
            
            # Check for reasonable value ranges
            config = self.data_sources[key]
            series_min = float(series.min())
            series_max = float(series.max())
            
            if key == 'vix' and (series_min < 5 or series_max > 100):
                is_valid = False
                issues.append("VIX values outside reasonable range (5-100)")
            elif key == 'aud_usd' and (series_min < 0.3 or series_max > 2.0):
                is_valid = False
                issues.append("AUD/USD values outside reasonable range (0.3-2.0)")
            elif key in ['au_10y', 'us_10y'] and (series_min < 0 or series_max > 20):
                is_valid = False
                issues.append("Yield values outside reasonable range (0-20%)")
            elif key == 'gold' and (series_min < 10 or series_max > 500):
                is_valid = False
                issues.append("Gold values outside reasonable range (10-500 AUD)")
            
            validation_results[key] = {
                'valid': is_valid,
                'issues': issues
            }
            
            if is_valid:
                print(f"✅ {config['name']}: Data quality OK")
            else:
                print(f"❌ {config['name']}: {', '.join(issues)}")
        
        return validation_results

# Convenience function for quick usage
def fetch_external_data(force_refresh: bool = False) -> Dict[str, pd.Series]:
    """
    Convenience function to fetch all external data
    
    Args:
        force_refresh: Force download even if cached data exists
        
    Returns:
        Dictionary of external data series
    """
    manager = ExternalDataManager()
    return manager.fetch_all_data(force_refresh)

if __name__ == "__main__":
    # Test the external data manager
    print("Testing External Data Manager...")
    
    manager = ExternalDataManager()
    data = manager.fetch_all_data()
    
    if data:
        summary = manager.get_data_summary(data)
        validation = manager.validate_data_quality(data)
        
        print(f"\n📈 Final Summary:")
        print(f"  Series fetched: {summary['total_series']}")
        print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"  Total trading days: {summary['date_range']['total_days']}")
        
        valid_count = sum(1 for v in validation.values() if v['valid'])
        print(f"  Quality validation: {valid_count}/{len(validation)} series passed")
    else:
        print("❌ No data fetched successfully")
