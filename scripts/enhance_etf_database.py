#!/usr/bin/env python3
"""
ETF Database Enhancement Script
Caches ETF names from Yahoo Finance to avoid repeated API calls
Updates the ETF database with full names and additional metadata
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yfinance as yf
import json
import time
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from data_manager.etf_database import ETFDatabase
from data_manager.data_manager import ETFDataManager


class ETFDatabaseEnhancer:
    """
    Enhances ETF database with cached names and metadata from Yahoo Finance
    Avoids repeated API calls during analysis
    """
    
    def __init__(self, cache_file: str = "data/etf_names_cache.json"):
        """
        Initialize database enhancer
        
        Args:
            cache_file: Path to cache file for ETF names
        """
        self.cache_file = cache_file
        self.etf_db = ETFDatabase()
        self.data_manager = ETFDataManager()
        self.name_cache = self._load_cache()
        
        # Rate limiting settings
        self.request_delay = 0.5  # Seconds between requests
        self.max_retries = 3
        self.batch_size = 50      # Process in batches to avoid timeouts
    
    def _load_cache(self) -> Dict:
        """Load cached ETF names from file"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"📝 Creating new cache file: {self.cache_file}")
            return {}
    
    def _save_cache(self):
        """Save cached ETF names to file"""
        try:
            import os
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.name_cache, f, indent=2, default=str)
            print(f"💾 Cache saved: {len(self.name_cache)} ETFs cached")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
    
    def fetch_etf_info(self, ticker: str) -> Dict:
        """
        Fetch ETF information from Yahoo Finance with retry logic
        
        Args:
            ticker: ETF ticker symbol
            
        Returns:
            Dict with ETF information
        """
        # Check cache first
        if ticker in self.name_cache:
            return self.name_cache[ticker]
        
        for attempt in range(self.max_retries):
            try:
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info
                
                # Extract relevant information
                etf_info = {
                    'ticker': ticker,
                    'longName': info.get('longName', ''),
                    'shortName': info.get('shortName', ''),
                    'category': info.get('category', ''),
                    'fundFamily': info.get('fundFamily', ''),
                    'expenseRatio': info.get('expenseRatio', 0.0),
                    'totalAssets': info.get('totalAssets', 0.0),
                    'ytdReturn': info.get('ytdReturn', 0.0),
                    'threeYearAverageReturn': info.get('threeYearAverageReturn', 0.0),
                    'fiveYearAverageReturn': info.get('fiveYearAverageReturn', 0.0),
                    'currency': info.get('currency', 'AUD'),
                    'exchange': info.get('exchange', 'ASX'),
                    'market': info.get('market', 'au_market'),
                    'quoteType': info.get('quoteType', 'ETF'),
                    'firstTradeDateEpochUtc': info.get('firstTradeDateEpochUtc', ''),
                    'last_updated': pd.Timestamp.now().isoformat()
                }
                
                # Cache the result
                self.name_cache[ticker] = etf_info
                
                # Rate limiting
                time.sleep(self.request_delay)
                
                return etf_info
                
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.request_delay * 2)  # Longer delay on retry
                else:
                    # Create placeholder for failed tickers
                    etf_info = {
                        'ticker': ticker,
                        'longName': self.etf_db.etf_data.get(ticker, {}).get('name', ticker),
                        'shortName': ticker,
                        'category': 'Unknown',
                        'fundFamily': 'Unknown',
                        'expenseRatio': 0.0,
                        'totalAssets': 0.0,
                        'ytdReturn': 0.0,
                        'threeYearAverageReturn': 0.0,
                        'fiveYearAverageReturn': 0.0,
                        'currency': 'AUD',
                        'exchange': 'ASX',
                        'market': 'au_market',
                        'quoteType': 'ETF',
                        'firstTradeDateEpochUtc': '',
                        'last_updated': pd.Timestamp.now().isoformat(),
                        'fetch_error': str(e)
                    }
                    self.name_cache[ticker] = etf_info
                    return etf_info
    
    def update_etf_database(self, force_update: bool = False) -> Dict:
        """
        Update ETF database with cached names and metadata
        
        Args:
            force_update: Force refresh of all cached data
            
        Returns:
            Dict with update results
        """
        print("🔄 Updating ETF database with enhanced metadata...")
        
        # Get all ETF tickers
        all_tickers = list(self.etf_db.etf_data.keys())
        print(f"📊 Found {len(all_tickers)} ETFs in database")
        
        # Check which ones need updating
        if force_update:
            tickers_to_update = all_tickers
            print("🔄 Force updating all ETFs...")
        else:
            # Check cache age
            current_time = pd.Timestamp.now()
            tickers_to_update = []
            
            for ticker in all_tickers:
                if ticker not in self.name_cache:
                    tickers_to_update.append(ticker)
                else:
                    last_updated = pd.Timestamp(self.name_cache[ticker].get('last_updated', '1970-01-01'))
                    if (current_time - last_updated).days > 7:  # Update if older than 7 days
                        tickers_to_update.append(ticker)
            
            print(f"📝 Need to update {len(tickers_to_update)} ETFs")
        
        # Process in batches
        updated_count = 0
        failed_count = 0
        
        for i in range(0, len(tickers_to_update), self.batch_size):
            batch = tickers_to_update[i:i + self.batch_size]
            print(f"🔄 Processing batch {i//self.batch_size + 1}/{(len(tickers_to_update)-1)//self.batch_size + 1} ({len(batch)} ETFs)...")
            
            for ticker in batch:
                try:
                    info = self.fetch_etf_info(ticker)
                    if 'fetch_error' not in info:
                        updated_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"❌ Failed to fetch {ticker}: {e}")
                    failed_count += 1
            
            # Save cache after each batch
            self._save_cache()
            
            # Brief pause between batches
            if i + self.batch_size < len(tickers_to_update):
                time.sleep(2)
        
        results = {
            'total_etfs': len(all_tickers),
            'updated_count': updated_count,
            'failed_count': failed_count,
            'cache_size': len(self.name_cache),
            'cache_file': self.cache_file
        }
        
        print(f"\n✅ Database update complete:")
        print(f"  Total ETFs: {results['total_etfs']}")
        print(f"  Successfully updated: {results['updated_count']}")
        print(f"  Failed: {results['failed_count']}")
        print(f"  Cache size: {results['cache_size']}")
        
        return results
    
    def generate_enhanced_database(self, output_file: str = "data/etf_database_enhanced.json") -> Dict:
        """
        Generate enhanced ETF database with cached information
        
        Args:
            output_file: Output file for enhanced database
            
        Returns:
            Enhanced database dictionary
        """
        print("📊 Generating enhanced ETF database...")
        
        enhanced_db = {}
        
        for ticker, base_info in self.etf_db.etf_data.items():
            # Get cached information
            cached_info = self.name_cache.get(ticker, {})
            
            # Combine base and cached information
            enhanced_info = {
                **base_info,  # Original database info
                **cached_info,  # Cached Yahoo Finance info
                'data_source': 'enhanced',
                'last_enhanced': pd.Timestamp.now().isoformat()
            }
            
            enhanced_db[ticker] = enhanced_info
        
        # Save enhanced database
        try:
            import os
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(enhanced_db, f, indent=2, default=str)
            print(f"💾 Enhanced database saved: {output_file}")
        except Exception as e:
            print(f"⚠️ Failed to save enhanced database: {e}")
        
        return enhanced_db
    
    def get_etf_name(self, ticker: str) -> str:
        """
        Get ETF name from cache (fast lookup)
        
        Args:
            ticker: ETF ticker
            
        Returns:
            ETF name or ticker if not found
        """
        if ticker in self.name_cache:
            return self.name_cache[ticker].get('longName', 
                   self.name_cache[ticker].get('shortName', 
                   self.etf_db.etf_data.get(ticker, {}).get('name', ticker)))
        else:
            return self.etf_db.etf_data.get(ticker, {}).get('name', ticker)
    
    def get_etf_metadata(self, ticker: str) -> Dict:
        """
        Get full ETF metadata from cache
        
        Args:
            ticker: ETF ticker
            
        Returns:
            ETF metadata dictionary
        """
        if ticker in self.name_cache:
            return self.name_cache[ticker]
        else:
            # Return basic info from original database
            base_info = self.etf_db.etf_data.get(ticker, {})
            return {
                'ticker': ticker,
                'longName': base_info.get('name', ticker),
                'shortName': ticker,
                'category': 'Unknown',
                'fundFamily': 'Unknown'
            }
    
    def print_cache_summary(self):
        """Print summary of cached data"""
        print(f"\n📊 ETF Name Cache Summary:")
        print(f"  Cache file: {self.cache_file}")
        print(f"  Total cached ETFs: {len(self.name_cache)}")
        
        # Analyze cache quality
        with_errors = sum(1 for info in self.name_cache.values() if 'fetch_error' in info)
        without_errors = len(self.name_cache) - with_errors
        
        print(f"  Successfully cached: {without_errors}")
        print(f"  With errors: {with_errors}")
        
        # Show some examples
        if self.name_cache:
            print(f"\n📝 Sample cached ETFs:")
            sample_tickers = list(self.name_cache.keys())[:5]
            for ticker in sample_tickers:
                info = self.name_cache[ticker]
                name = info.get('longName', 'Unknown')
                status = "✅" if 'fetch_error' not in info else "❌"
                print(f"  {status} {ticker}: {name[:50]}{'...' if len(name) > 50 else ''}")


def create_etf_name_lookup_function() -> callable:
    """
    Create a fast lookup function for ETF names
    This can be imported and used throughout the system
    
    Returns:
        Function that returns ETF name for given ticker
    """
    enhancer = ETFDatabaseEnhancer()
    
    def get_etf_name_fast(ticker: str) -> str:
        """Fast ETF name lookup"""
        return enhancer.get_etf_name(ticker)
    
    return get_etf_name_fast


# Module-level fast lookup function for easy importing
_enhancer_instance = None

def get_etf_name_fast(ticker: str) -> str:
    """
    Fast ETF name lookup from cache
    
    Args:
        ticker: ETF ticker symbol
        
    Returns:
        ETF name or ticker if not found in cache
    """
    global _enhancer_instance
    if _enhancer_instance is None:
        _enhancer_instance = ETFDatabaseEnhancer()
    
    return _enhancer_instance.get_etf_name(ticker)


if __name__ == "__main__":
    print("🚀 ETF Database Enhancement Script")
    print("=" * 50)
    
    # Create enhancer
    enhancer = ETFDatabaseEnhancer()
    
    # Print current cache status
    enhancer.print_cache_summary()
    
    # Update database
    print(f"\n🔄 Starting database update...")
    results = enhancer.update_etf_database(force_update=False)
    
    # Generate enhanced database
    print(f"\n📊 Generating enhanced database...")
    enhanced_db = enhancer.generate_enhanced_database()
    
    # Test lookup function
    print(f"\n🧪 Testing fast lookup function...")
    test_tickers = ['VAS.AX', 'IOZ.AX', 'MVB.AX']
    
    for ticker in test_tickers:
        name = enhancer.get_etf_name(ticker)
        print(f"  {ticker}: {name}")
    
    print(f"\n✅ Enhancement complete!")
    print(f"💡 Usage: Import get_etf_name_fast() for fast name lookups")
