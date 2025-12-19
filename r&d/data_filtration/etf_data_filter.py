#!/usr/bin/env python3
"""
ETF Data Filtration Layer - R&D Prototype
Filters ETFs based on data availability, risk class, and dynamic classification

Purpose: 
- Eliminate need for EODHD subscription
- Focus on ETFs with rich holdings data from yfinance/yahooquery
- Exclude low risk ETFs for growth focus
- Dynamic region/sector identification from actual holdings
- Analyze entire database to identify ETFs with actual price data
"""

import yahooquery as yq
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import warnings
from datetime import datetime, timedelta
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Add parent directory to path to import ETF database
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data_manager.etf_database import ETFDatabase


class ETFDataFilter:
    """
    Advanced ETF filtration system using yfinance/yahooquery data
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize ETF data filter
        
        Args:
            debug: Enable detailed logging
        """
        self.debug = debug
        self.filtered_etfs = {}
        
        # Enhanced exchange-to-region mapping
        self.exchange_region_map = {
            '-HK': 'Hong Kong', '-US': 'United States', '-CA': 'Canada',
            '-GB': 'United Kingdom', '-IN': 'India', '-JP': 'Japan',
            '-JKT': 'Indonesia', '-PH': 'Philippines', '-AU': 'Australia',
            '-TH': 'Thailand', '-NZ': 'New Zealand', '-ZA': 'South Africa',
            '-SG': 'Singapore', '-SE': 'Sweden', '-CN': 'China',
            '-BM': 'Bermuda', '-NE': 'Canada',  # NEO Exchange
            '-BR': 'Brazil', '-DE': 'Germany', '-FR': 'France',
            '.AX': 'Australia',  # ASX format
            '.L': 'United Kingdom', '.TO': 'Canada', '.HK': 'Hong Kong',
            '.T': 'Japan', '.DE': 'Germany', '.PA': 'France',
            '.SI': 'Singapore', '.NZ': 'New Zealand',
            '': 'United States'  # No suffix = US
        }
    
    def check_price_data_availability(self, ticker: str, min_days: int = 30) -> Dict:
        """
        Check if ETF has sufficient price/trading data available via yfinance
        
        Args:
            ticker: ETF ticker symbol
            min_days: Minimum number of days of data required
            
        Returns:
            dict: Data availability info with 'has_data', 'days_available', 'latest_date', 'status'
        """
        result = {
            'has_data': False,
            'days_available': 0,
            'latest_date': None,
            'status': 'unknown',
            'error': None
        }
        
        try:
            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(period="2y")
            
            if hist.empty:
                result['status'] = 'no_data'
                result['error'] = 'Empty history'
                return result
            
            # Check data quality
            prices = hist['Close'] if 'Close' in hist.columns else None
            if prices is None or len(prices) == 0:
                result['status'] = 'no_price_data'
                result['error'] = 'No price column'
                return result
            
            # Check if we have enough data
            days_available = len(prices)
            latest_date = prices.index[-1] if len(prices) > 0 else None
            
            if days_available < min_days:
                result['status'] = 'insufficient_data'
                result['days_available'] = days_available
                result['latest_date'] = latest_date
                result['error'] = f'Only {days_available} days (need {min_days})'
                return result
            
            # Check if data is recent (within last 30 days)
            if latest_date:
                days_since_update = (datetime.now() - latest_date.replace(tzinfo=None)).days
                if days_since_update > 30:
                    result['status'] = 'stale_data'
                    result['days_available'] = days_available
                    result['latest_date'] = latest_date
                    result['error'] = f'Data is {days_since_update} days old'
                    return result
            
            result['has_data'] = True
            result['days_available'] = days_available
            result['latest_date'] = latest_date
            result['status'] = 'available'
            
            return result
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            return result
    
    def check_holdings_data_availability(self, ticker: str) -> bool:
        """
        Check if ETF has sufficient holdings data available (original method)
        
        Args:
            ticker: ETF ticker symbol
            
        Returns:
            bool: True if data is available and sufficient
        """
        try:
            yq_ticker = yq.Ticker(ticker)
            
            # Check if fund holdings info exists
            if ticker not in yq_ticker.fund_holding_info:
                if self.debug:
                    print(f"❌ {ticker}: No holdings data available")
                return False
            
            holdings_info = yq_ticker.fund_holding_info[ticker]
            holdings_data = holdings_info.get('holdings', [])
            
            # Check if we have meaningful holdings data
            if len(holdings_data) < 5:  # Require at least 5 holdings
                if self.debug:
                    print(f"❌ {ticker}: Insufficient holdings data ({len(holdings_data)} holdings)")
                return False
            
            # Check if we have sector weightings
            sector_weightings = holdings_info.get('sectorWeightings', [])
            if not sector_weightings:
                if self.debug:
                    print(f"⚠️  {ticker}: No sector weightings available")
                # Still acceptable if we have holdings data
            
            if self.debug:
                print(f"✅ {ticker}: Data available ({len(holdings_data)} holdings)")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"❌ {ticker}: Error checking data - {str(e)}")
            return False
    
    def check_data_availability(self, ticker: str) -> bool:
        """
        Legacy method - checks holdings data availability
        For price data, use check_price_data_availability()
        """
        return self.check_holdings_data_availability(ticker)
    
    def classify_asset_class(self, holdings_info: Dict, ticker: str) -> str:
        """
        Classify primary asset class from fund allocation
        
        Args:
            holdings_info: YahooQuery holdings information
            ticker: ETF ticker for fallback name lookup
            
        Returns:
            str: Primary asset class (Equity, Bond, Commodity, Crypto, Mixed, Other)
        """
        try:
            stock_position = holdings_info.get('stockPosition', 0)
            bond_position = holdings_info.get('bondPosition', 0)
            other_position = holdings_info.get('otherPosition', 0)
            
            # Classify primary asset type
            if stock_position > 0.5:
                return "Equity"
            elif bond_position > 0.5:
                return "Bond"
            elif other_position > 0.5:
                # Fallback to name analysis for other assets
                yq_ticker = yq.Ticker(ticker)
                fund_name = yq_ticker.quote_type[ticker].get('longName', '').lower()
                
                crypto_keywords = ['bitcoin', 'ethereum', 'crypto', 'blockchain']
                commodity_keywords = ['gold', 'silver', 'oil', 'commodity', 'metal']
                
                if any(keyword in fund_name for keyword in crypto_keywords):
                    return "Crypto"
                elif any(keyword in fund_name for keyword in commodity_keywords):
                    return "Commodity"
                else:
                    return "Other"
            else:
                return "Mixed"
                
        except Exception as e:
            if self.debug:
                print(f"⚠️  {ticker}: Error classifying asset class - {str(e)}")
            return "Other"
    
    def extract_primary_sector(self, holdings_info: Dict, asset_class: str) -> Tuple[str, float]:
        """
        Extract primary sector from sector weightings
        
        Args:
            holdings_info: YahooQuery holdings information
            asset_class: Primary asset class for fallback
            
        Returns:
            tuple: (primary_sector, sector_weight)
        """
        try:
            sector_weightings = holdings_info.get('sectorWeightings', [])
            
            if sector_weightings:
                sector_dict = {list(s.keys())[0]: list(s.values())[0] for s in sector_weightings}
                primary_sector = max(sector_dict, key=sector_dict.get)
                primary_sector_weight = sector_dict[primary_sector]
                return primary_sector, primary_sector_weight
            else:
                # Fallback for commodity/crypto ETFs
                if asset_class in ["Commodity", "Crypto"]:
                    return asset_class, 1.0
                else:
                    return "N/A", 0
                    
        except Exception as e:
            if self.debug:
                print(f"⚠️  Error extracting sector: {str(e)}")
            return "N/A", 0
    
    def extract_primary_region(self, holdings_data: List[Dict]) -> Tuple[str, float]:
        """
        Extract primary region from holdings distribution
        
        Args:
            holdings_data: List of holding dictionaries
            
        Returns:
            tuple: (primary_region, region_weight)
        """
        try:
            regions = []
            
            for holding in holdings_data:
                symbol = holding['symbol']
                weight = holding.get('holdingPercent', 0)
                
                # Check for dash suffix first (e.g., AAPL-US)
                if '-' in symbol:
                    suffix = '-' + symbol.split('-')[-1]
                # Then check for dot suffix (e.g., BHP.AX)
                elif '.' in symbol:
                    suffix = '.' + symbol.split('.')[-1]
                else:
                    suffix = ''
                
                region = self.exchange_region_map.get(suffix, 'United States')
                regions.append((region, weight))
            
            # Calculate primary region by weight
            if regions:
                region_weights = {}
                for region, weight in regions:
                    region_weights[region] = region_weights.get(region, 0) + weight
                primary_region = max(region_weights, key=region_weights.get)
                primary_region_weight = region_weights[primary_region]
                return primary_region, primary_region_weight
            else:
                return "N/A", 0
                
        except Exception as e:
            if self.debug:
                print(f"⚠️  Error extracting region: {str(e)}")
            return "N/A", 0
    
    def should_exclude_risk_class(self, risk_category: str) -> bool:
        """
        Determine if risk category should be excluded
        
        Args:
            risk_category: Risk category from existing system
            
        Returns:
            bool: True if should exclude
        """
        # Exclude LOW risk ETFs to focus on growth opportunities
        return risk_category == "LOW"
    
    def filter_single_etf(self, ticker: str, risk_category: str = None) -> Optional[Dict]:
        """
        Filter and analyze a single ETF
        
        Args:
            ticker: ETF ticker symbol
            risk_category: Existing risk category (optional)
            
        Returns:
            dict: Filter results or None if excluded
        """
        try:
            # Check data availability first
            if not self.check_data_availability(ticker):
                return None
            
            # Check risk class exclusion
            if risk_category and self.should_exclude_risk_class(risk_category):
                if self.debug:
                    print(f"❌ {ticker}: Excluded - {risk_category} risk class")
                return None
            
            # Extract detailed information
            yq_ticker = yq.Ticker(ticker)
            holdings_info = yq_ticker.fund_holding_info[ticker]
            holdings_data = holdings_info.get('holdings', [])
            
            # Classify asset class
            asset_class = self.classify_asset_class(holdings_info, ticker)
            
            # Extract sector and region
            primary_sector, sector_weight = self.extract_primary_sector(holdings_info, asset_class)
            primary_region, region_weight = self.extract_primary_region(holdings_data)
            
            # Get basic fund info
            fund_info = yq_ticker.quote_type[ticker]
            fund_name = fund_info.get('longName', ticker)
            
            result = {
                'ticker': ticker,
                'name': fund_name,
                'risk_category': risk_category,
                'asset_class': asset_class,
                'primary_sector': primary_sector,
                'sector_weight': sector_weight,
                'primary_region': primary_region,
                'region_weight': region_weight,
                'holdings_count': len(holdings_data),
                'data_quality': 'RICH' if len(holdings_data) > 20 else 'BASIC'
            }
            
            if self.debug:
                print(f"✅ {ticker}: {asset_class} | {primary_sector} ({sector_weight:.1%}) | {primary_region} ({region_weight:.1%})")
            
            return result
            
        except Exception as e:
            if self.debug:
                print(f"❌ {ticker}: Error during filtering - {str(e)}")
            return None
    
    def filter_etf_universe(self, etf_list: List[str], risk_categories: Dict[str, str] = None) -> Dict:
        """
        Filter entire ETF universe using batch processing for speed
        
        Args:
            etf_list: List of ETF tickers to filter
            risk_categories: Dictionary of ticker -> risk_category mapping
            
        Returns:
            dict: Filtering results and statistics
        """
        print(f"🔍 Starting BATCH filtration of {len(etf_list)} ETFs...")
        
        filtered_results = []
        excluded_count = 0
        data_unavailable = 0
        risk_excluded = 0
        
        # Process in batches for better performance
        batch_size = 20  # Reduced from 50 to avoid timeouts
        for i in range(0, len(etf_list), batch_size):
            batch_tickers = etf_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(etf_list) + batch_size - 1) // batch_size
            
            print(f"  Batch {batch_num}/{total_batches}: {len(batch_tickers)} ETFs")
            
            # Batch check price data first
            price_batch = self._batch_check_price_data(batch_tickers)
            price_available = [t for t in batch_tickers if price_batch.get(t, {}).get('has_data', False)]
            
            # Then batch check holdings for those with price data
            if price_available:
                holdings_batch = self._batch_check_holdings_data(price_available)
                
                # Process individual results for detailed analysis
                for ticker in price_available:
                    risk_category = risk_categories.get(ticker) if risk_categories else None
                    
                    # Check risk class exclusion
                    if risk_category and self.should_exclude_risk_class(risk_category):
                        if self.debug:
                            print(f"❌ {ticker}: Excluded - {risk_category} risk class")
                        excluded_count += 1
                        risk_excluded += 1
                        continue
                    
                    # Only proceed if holdings data is available
                    if holdings_batch.get(ticker, {}).get('has_data', False):
                        result = self._analyze_single_etf(ticker, risk_category)
                        if result:
                            filtered_results.append(result)
                    else:
                        excluded_count += 1
                        data_unavailable += 1
            else:
                excluded_count += len(batch_tickers)
                data_unavailable += len(batch_tickers)
            
            # Brief pause between batches
            if batch_num < total_batches:
                time.sleep(0.5)
        
        # Create summary statistics
        summary = {
            'total_etfs': len(etf_list),
            'filtered_etfs': len(filtered_results),
            'excluded_etfs': excluded_count,
            'data_unavailable': data_unavailable,
            'risk_excluded': risk_excluded,
            'filtering_rate': len(filtered_results) / len(etf_list) * 100
        }
        
        print(f"\n📊 Filtration Summary:")
        print(f"  Total ETFs: {summary['total_etfs']}")
        print(f"  Filtered ETFs: {summary['filtered_etfs']}")
        print(f"  Excluded ETFs: {summary['excluded_etfs']}")
        print(f"    - Data unavailable: {summary['data_unavailable']}")
        print(f"    - Risk class excluded: {summary['risk_excluded']}")
        print(f"  Success rate: {summary['filtering_rate']:.1f}%")
        
        return {
            'filtered_etfs': filtered_results,
            'summary': summary
        }
    
    def _batch_check_price_data(self, tickers: List[str], min_days: int = 30) -> Dict[str, Dict]:
        """Check price data for multiple tickers in batch"""
        results = {}
        
        try:
            # Download batch price data
            batch_data = yf.download(
                tickers,
                period="2y",
                progress=False,
                group_by='ticker',
                threads=True
            )
            
            # Process each ticker
            for ticker in tickers:
                result = {'has_data': False, 'status': 'unknown'}
                
                try:
                    if ticker in batch_data.columns.levels[0]:
                        ticker_data = batch_data[ticker]
                        if 'Close' in ticker_data.columns:
                            prices = ticker_data['Close'].dropna()
                            if len(prices) >= min_days:
                                result['has_data'] = True
                                result['status'] = 'available'
                            else:
                                result['status'] = 'insufficient_data'
                        else:
                            result['status'] = 'no_price_data'
                    else:
                        result['status'] = 'no_data'
                except Exception as e:
                    result['status'] = 'error'
                    result['error'] = str(e)
                
                results[ticker] = result
                
        except Exception as e:
            # Mark all as failed if batch fails
            for ticker in tickers:
                results[ticker] = {'has_data': False, 'status': 'error', 'error': str(e)}
        
        return results
    
    def _batch_check_holdings_data(self, tickers: List[str], min_holdings: int = 5) -> Dict[str, Dict]:
        """Check holdings data for multiple tickers in batch with timeout protection"""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        
        def fetch_holdings_data(tickers_list):
            """Helper function to fetch holdings data"""
            batch_query = ' '.join(tickers_list)
            yq_ticker = yq.Ticker(batch_query)
            return yq_ticker.fund_holding_info
        
        results = {}
        
        try:
            # Use ThreadPoolExecutor for timeout protection
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_holdings_data, tickers)
                try:
                    holdings_data = future.result(timeout=30)  # 30 second timeout
                except FutureTimeoutError:
                    raise Exception("Batch holdings query timed out after 30 seconds")
            
            # Process each ticker
            for ticker in tickers:
                result = {'has_data': False, 'status': 'unknown', 'asset_class': 'Unknown'}
                
                try:
                    if ticker in holdings_data:
                        holdings_info = holdings_data[ticker]
                        
                        # Check if holdings_info is a string (error case) or dict (expected)
                        if isinstance(holdings_info, str):
                            # This ticker returned an error message instead of data
                            result['status'] = 'error'
                            result['error'] = f"API returned string: {holdings_info[:100]}"
                        elif isinstance(holdings_info, dict):
                            holdings_list = holdings_info.get('holdings', [])
                            
                            # Classify asset class
                            asset_class = self.classify_asset_class(holdings_info, ticker)
                            result['asset_class'] = asset_class
                            
                            # Check holdings based on asset class
                            if asset_class == "Bond":
                                bond_position = holdings_info.get('bondPosition', 0)
                                bond_holdings = holdings_info.get('bondHoldings', [])
                                holdings_count = len(bond_holdings if bond_holdings else holdings_list)
                                
                                if bond_position > 0 or holdings_count >= min_holdings:
                                    result['has_data'] = True
                                    result['status'] = 'available'
                            elif asset_class in ["Crypto", "Commodity"]:
                                other_position = holdings_info.get('otherPosition', 0)
                                holdings_count = len(holdings_list)
                                
                                if other_position > 0 or holdings_count >= 1:
                                    result['has_data'] = True
                                    result['status'] = 'available'
                            else:
                                # Equity or Mixed
                                holdings_count = len(holdings_list)
                                if holdings_count >= min_holdings:
                                    result['has_data'] = True
                                    result['status'] = 'available'
                            
                            result['holdings_count'] = holdings_count
                        else:
                            result['status'] = 'error'
                            result['error'] = f"Unexpected data type: {type(holdings_info)}"
                    else:
                        result['status'] = 'no_holdings_data'
                        
                except Exception as e:
                    result['status'] = 'error'
                    result['error'] = str(e)
                
                results[ticker] = result
                
        except Exception as e:
            # Mark all as failed if batch fails
            error_msg = str(e)
            if "timed out" in error_msg:
                error_msg = "Batch query timed out - trying individual tickers"
                # Fall back to individual processing with timeout
                for ticker in tickers:
                    try:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(self._check_single_holdings, ticker, min_holdings)
                            result = future.result(timeout=10)
                            results[ticker] = result
                    except:
                        results[ticker] = {'has_data': False, 'status': 'error', 'error': 'Individual query failed'}
            else:
                for ticker in tickers:
                    results[ticker] = {'has_data': False, 'status': 'error', 'error': error_msg}
        
        return results
    
    def _check_single_holdings(self, ticker: str, min_holdings: int) -> Dict:
        """Check holdings for a single ticker"""
        result = {'has_data': False, 'status': 'unknown', 'asset_class': 'Unknown'}
        
        try:
            yq_ticker = yq.Ticker(ticker)
            holdings_info = yq_ticker.fund_holding_info.get(ticker, {})
            holdings_list = holdings_info.get('holdings', [])
            
            # Classify asset class
            asset_class = self.classify_asset_class(holdings_info, ticker)
            result['asset_class'] = asset_class
            
            # Check holdings
            holdings_count = len(holdings_list)
            result['holdings_count'] = holdings_count
            
            if holdings_count >= min_holdings:
                result['has_data'] = True
                result['status'] = 'available'
            else:
                result['status'] = 'insufficient_holdings'
                
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
        
        return result
    
    def _analyze_single_etf(self, ticker: str, risk_category: str = None) -> Optional[Dict]:
        """Analyze a single ETF with available data"""
        try:
            yq_ticker = yq.Ticker(ticker)
            holdings_info = yq_ticker.fund_holding_info[ticker]
            holdings_data = holdings_info.get('holdings', [])
            
            # Classify asset class
            asset_class = self.classify_asset_class(holdings_info, ticker)
            
            # Extract sector and region
            primary_sector, sector_weight = self.extract_primary_sector(holdings_info, asset_class)
            primary_region, region_weight = self.extract_primary_region(holdings_data)
            
            # Get basic fund info
            fund_info = yq_ticker.quote_type[ticker]
            fund_name = fund_info.get('longName', ticker)
            
            result = {
                'ticker': ticker,
                'name': fund_name,
                'risk_category': risk_category,
                'asset_class': asset_class,
                'primary_sector': primary_sector,
                'sector_weight': sector_weight,
                'primary_region': primary_region,
                'region_weight': region_weight,
                'holdings_count': len(holdings_data),
                'data_quality': 'RICH' if len(holdings_data) > 20 else 'BASIC'
            }
            
            if self.debug:
                print(f"✅ {ticker}: {asset_class} | {primary_sector} ({sector_weight:.1%}) | {primary_region} ({region_weight:.1%})")
            
            return result
            
        except Exception as e:
            if self.debug:
                print(f"❌ {ticker}: Error during filtering - {str(e)}")
            return None


    def analyze_database_holdings_availability(self, min_holdings: int = 5, rate_limit: float = 0.1) -> Dict:
        """
        Analyze the entire ETF database to check holdings data availability using batch processing
        
        Args:
            min_holdings: Minimum number of holdings required
            rate_limit: Seconds to wait between API calls (now only between batches)
            
        Returns:
            dict: Comprehensive analysis results
        """
        print("="*80)
        print("ETF DATABASE HOLDINGS DATA AVAILABILITY ANALYSIS (BATCH)")
        print("="*80)
        print(f"\nLoading ETF database...")
        
        # Load entire database
        db = ETFDatabase()
        all_tickers = list(db.etf_data.keys())
        total_etfs = len(all_tickers)
        
        print(f"✅ Loaded {total_etfs} ETFs from database\n")
        print(f"Analyzing holdings data availability (min {min_holdings} holdings required)...")
        print(f"Using batch processing for speed\n")
        
        results = {
            'available': [],
            'insufficient_holdings': [],
            'no_holdings_data': [],
            'no_sector_data': [],
            'errors': [],
            'by_asset_class': {
                'Equity': [],
                'Bond': [],
                'Crypto': [],
                'Commodity': [],
                'Mixed': [],
                'Other': []
            },
            'summary': {}
        }
        
        start_time = time.time()
        
        # Process in batches
        batch_size = 20  # Reduced from 50 to avoid timeouts
        error_details = []  # Store error details for debugging
        
        for i in range(0, len(all_tickers), batch_size):
            batch_tickers = all_tickers[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(all_tickers) + batch_size - 1) // batch_size
            
            # Progress indicator
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total_etfs - i) / rate if rate > 0 else 0
            print(f"  Progress: {i}/{total_etfs} ({i/total_etfs*100:.1f}%) | "
                  f"Rate: {rate:.1f} ETFs/s | ETA: {eta/60:.1f} min")
            
            # Batch process holdings data
            batch_results = self._batch_check_holdings_data(batch_tickers, min_holdings)
            
            # Process results
            for ticker in batch_tickers:
                etf_info = db.etf_data.get(ticker, {})
                etf_name = etf_info.get('name', ticker)
                
                result_entry = {
                    'ticker': ticker,
                    'name': etf_name,
                    'region': etf_info.get('region', 'Unknown'),
                    'subcategory': etf_info.get('subcategory', 'Unknown'),
                    'asset_class': 'Unknown',
                    'holdings_count': 0,
                    'has_sector_data': False,
                    'status': 'unknown',
                    'error': None
                }
                
                batch_result = batch_results.get(ticker, {})
                result_entry['status'] = batch_result.get('status', 'unknown')
                result_entry['error'] = batch_result.get('error')
                result_entry['asset_class'] = batch_result.get('asset_class', 'Unknown')
                result_entry['holdings_count'] = batch_result.get('holdings_count', 0)
                
                # Categorize results
                if batch_result.get('has_data', False):
                    result_entry['status'] = 'available'
                    results['available'].append(result_entry)
                    results['by_asset_class'][result_entry['asset_class']].append(result_entry)
                elif result_entry['status'] == 'insufficient_holdings':
                    results['insufficient_holdings'].append(result_entry)
                    results['by_asset_class'][result_entry['asset_class']].append(result_entry)
                elif result_entry['status'] == 'no_holdings_data':
                    results['no_holdings_data'].append(result_entry)
                else:
                    results['errors'].append(result_entry)
                    # Store error details for debugging
                    if result_entry.get('error'):
                        error_details.append({
                            'ticker': ticker,
                            'error': result_entry['error'],
                            'status': result_entry['status']
                        })
            
            # Brief pause between batches
            if batch_num < total_batches:
                time.sleep(0.5)
        
        elapsed_time = time.time() - start_time
        
        # Calculate summary statistics
        total_available = len(results['available']) + len(results['no_sector_data'])
        
        # Count by asset class
        asset_class_counts = {}
        for asset_class, etfs in results['by_asset_class'].items():
            asset_class_counts[asset_class] = len(etfs)
        
        results['summary'] = {
            'total_etfs': total_etfs,
            'available_count': len(results['available']),
            'available_no_sector_count': len(results['no_sector_data']),
            'total_available_count': total_available,
            'insufficient_holdings_count': len(results['insufficient_holdings']),
            'no_holdings_data_count': len(results['no_holdings_data']),
            'error_count': len(results['errors']),
            'availability_rate': total_available / total_etfs * 100,
            'full_data_rate': len(results['available']) / total_etfs * 100,
            'analysis_time_seconds': elapsed_time,
            'analysis_time_minutes': elapsed_time / 60,
            'by_asset_class': asset_class_counts,
            'error_details': error_details  # Add error details
        }
        
        return results
    
    def analyze_database_data_availability(self, min_days: int = 30, rate_limit: float = 0.1) -> Dict:
        """
        Analyze the entire ETF database to check price data availability
        
        Args:
            min_days: Minimum days of data required
            rate_limit: Seconds to wait between API calls
            
        Returns:
            dict: Comprehensive analysis results
        """
        print("="*80)
        print("ETF DATABASE DATA AVAILABILITY ANALYSIS")
        print("="*80)
        print(f"\nLoading ETF database...")
        
        # Load entire database
        db = ETFDatabase()
        all_tickers = list(db.etf_data.keys())
        total_etfs = len(all_tickers)
        
        print(f"✅ Loaded {total_etfs} ETFs from database\n")
        print(f"Analyzing price data availability (min {min_days} days required)...")
        print(f"Rate limit: {rate_limit}s between requests\n")
        
        results = {
            'available': [],
            'insufficient': [],
            'stale': [],
            'no_data': [],
            'errors': [],
            'summary': {}
        }
        
        start_time = time.time()
        
        for i, ticker in enumerate(all_tickers, 1):
            # Progress indicator
            if i % 10 == 0 or i == 1:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total_etfs - i) / rate if rate > 0 else 0
                print(f"  Progress: {i}/{total_etfs} ({i/total_etfs*100:.1f}%) | "
                      f"Rate: {rate:.1f} ETFs/s | ETA: {eta/60:.1f} min")
            
            # Check price data availability
            data_check = self.check_price_data_availability(ticker, min_days)
            
            etf_info = db.etf_data.get(ticker, {})
            etf_name = etf_info.get('name', ticker)
            
            result_entry = {
                'ticker': ticker,
                'name': etf_name,
                'region': etf_info.get('region', 'Unknown'),
                'subcategory': etf_info.get('subcategory', 'Unknown'),
                'days_available': data_check['days_available'],
                'latest_date': data_check['latest_date'],
                'status': data_check['status'],
                'error': data_check.get('error')
            }
            
            # Categorize results
            if data_check['has_data']:
                results['available'].append(result_entry)
            elif data_check['status'] == 'insufficient_data':
                results['insufficient'].append(result_entry)
            elif data_check['status'] == 'stale_data':
                results['stale'].append(result_entry)
            elif data_check['status'] in ['no_data', 'no_price_data']:
                results['no_data'].append(result_entry)
            else:
                results['errors'].append(result_entry)
            
            # Rate limiting
            if i < total_etfs:
                time.sleep(rate_limit)
        
        elapsed_time = time.time() - start_time
        
        # Calculate summary statistics
        results['summary'] = {
            'total_etfs': total_etfs,
            'available_count': len(results['available']),
            'insufficient_count': len(results['insufficient']),
            'stale_count': len(results['stale']),
            'no_data_count': len(results['no_data']),
            'error_count': len(results['errors']),
            'availability_rate': len(results['available']) / total_etfs * 100,
            'analysis_time_seconds': elapsed_time,
            'analysis_time_minutes': elapsed_time / 60
        }
        
        return results
    
    def print_holdings_analysis_report(self, results: Dict):
        """Print comprehensive holdings data analysis report"""
        summary = results['summary']
        
        print("\n" + "="*80)
        print("HOLDINGS DATA AVAILABILITY ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"  Total ETFs Analyzed: {summary['total_etfs']}")
        print(f"  Analysis Time: {summary['analysis_time_minutes']:.1f} minutes")
        print(f"\n✅ AVAILABLE (with sector data): {summary['available_count']} ({summary['full_data_rate']:.1f}%)")
        print(f"✅ AVAILABLE (no sector data): {summary['available_no_sector_count']}")
        print(f"📈 TOTAL AVAILABLE: {summary['total_available_count']} ({summary['availability_rate']:.1f}%)")
        print(f"⚠️  INSUFFICIENT HOLDINGS: {summary['insufficient_holdings_count']}")
        print(f"❌ NO HOLDINGS DATA: {summary['no_holdings_data_count']}")
        print(f"🔴 ERRORS: {summary['error_count']}")
        
        # Show error details if there are errors
        if summary['error_count'] > 0 and 'error_details' in summary:
            print(f"\n🔍 ERROR BREAKDOWN (first 10 errors):")
            error_types = {}
            for error in summary['error_details'][:10]:
                error_msg = error.get('error', 'Unknown error')
                error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg[:50]
                error_types[error_type] = error_types.get(error_type, 0) + 1
                print(f"  {error['ticker']}: {error_msg[:100]}")
            
            print(f"\n📊 ERROR TYPES:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count} occurrences")
        if 'by_asset_class' in summary:
            print(f"\n📦 BREAKDOWN BY ASSET CLASS:")
            for asset_class, count in sorted(summary['by_asset_class'].items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"  {asset_class}: {count} ETFs")
        
        # Available ETFs breakdown by region
        if results['available']:
            print(f"\n📈 AVAILABLE ETFs BY REGION (with sector data):")
            region_counts = {}
            for etf in results['available']:
                region = etf['region']
                region_counts[region] = region_counts.get(region, 0) + 1
            
            for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {region}: {count} ETFs")
        
        # Available ETFs breakdown by subcategory
        if results['available']:
            print(f"\n📊 AVAILABLE ETFs BY SUBCATEGORY (with sector data):")
            subcat_counts = {}
            for etf in results['available']:
                subcat = etf['subcategory']
                subcat_counts[subcat] = subcat_counts.get(subcat, 0) + 1
            
            for subcat, count in sorted(subcat_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
                print(f"  {subcat}: {count} ETFs")
        
        # Holdings count statistics
        if results['available']:
            holdings_counts = [etf['holdings_count'] for etf in results['available']]
            print(f"\n📦 HOLDINGS COUNT STATISTICS (available ETFs):")
            print(f"  Average: {np.mean(holdings_counts):.1f} holdings")
            print(f"  Median: {np.median(holdings_counts):.1f} holdings")
            print(f"  Min: {min(holdings_counts)} holdings")
            print(f"  Max: {max(holdings_counts)} holdings")
        
        # Sample of available ETFs
        if results['available']:
            print(f"\n✅ SAMPLE AVAILABLE ETFs (with sector data, first 15):")
            for etf in sorted(results['available'], key=lambda x: x['holdings_count'], reverse=True)[:15]:
                print(f"  {etf['ticker']:<12} {etf['name'][:50]:<50} {etf['holdings_count']:>4} holdings")
        
        # ETFs with no holdings data
        if results['no_holdings_data']:
            print(f"\n❌ ETFs WITH NO HOLDINGS DATA (first 15):")
            for etf in results['no_holdings_data'][:15]:
                print(f"  {etf['ticker']:<12} {etf['name'][:50]:<50}")
        
        # ETFs with insufficient holdings
        if results['insufficient_holdings']:
            print(f"\n⚠️  ETFs WITH INSUFFICIENT HOLDINGS (first 15):")
            for etf in results['insufficient_holdings'][:15]:
                print(f"  {etf['ticker']:<12} {etf['name'][:50]:<50} {etf['holdings_count']} holdings")
        
        print("\n" + "="*80)
        
        # Export recommendations
        available_tickers = [etf['ticker'] for etf in results['available']]
        available_no_sector_tickers = [etf['ticker'] for etf in results['no_sector_data']]
        all_available_tickers = available_tickers + available_no_sector_tickers
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"  Use {len(available_tickers)} ETFs with full holdings + sector data")
        print(f"  Use {len(all_available_tickers)} ETFs total with holdings data")
        print(f"  This represents {summary['availability_rate']:.1f}% of the database")
        
        return {
            'full_data': available_tickers,
            'all_available': all_available_tickers
        }
    
    def print_analysis_report(self, results: Dict):
        """Print comprehensive price data analysis report"""
        summary = results['summary']
        
        print("\n" + "="*80)
        print("PRICE DATA AVAILABILITY ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"  Total ETFs Analyzed: {summary['total_etfs']}")
        print(f"  Analysis Time: {summary['analysis_time_minutes']:.1f} minutes")
        print(f"\n✅ AVAILABLE (≥{30} days, recent data): {summary['available_count']} ({summary['availability_rate']:.1f}%)")
        print(f"⚠️  INSUFFICIENT DATA (<{30} days): {summary['insufficient_count']}")
        print(f"⏰ STALE DATA (>30 days old): {summary['stale_count']}")
        print(f"❌ NO DATA: {summary['no_data_count']}")
        print(f"🔴 ERRORS: {summary['error_count']}")
        
        # Available ETFs breakdown by region
        if results['available']:
            print(f"\n📈 AVAILABLE ETFs BY REGION:")
            region_counts = {}
            for etf in results['available']:
                region = etf['region']
                region_counts[region] = region_counts.get(region, 0) + 1
            
            for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {region}: {count} ETFs")
        
        # Available ETFs breakdown by subcategory
        if results['available']:
            print(f"\n📊 AVAILABLE ETFs BY SUBCATEGORY:")
            subcat_counts = {}
            for etf in results['available']:
                subcat = etf['subcategory']
                subcat_counts[subcat] = subcat_counts.get(subcat, 0) + 1
            
            for subcat, count in sorted(subcat_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {subcat}: {count} ETFs")
        
        # Sample of available ETFs
        if results['available']:
            print(f"\n✅ SAMPLE AVAILABLE ETFs (first 10):")
            for etf in results['available'][:10]:
                latest_str = etf['latest_date'].strftime('%Y-%m-%d') if etf['latest_date'] else 'N/A'
                print(f"  {etf['ticker']:<12} {etf['name'][:50]:<50} {etf['days_available']:>4} days (latest: {latest_str})")
        
        # ETFs with no data
        if results['no_data']:
            print(f"\n❌ ETFs WITH NO DATA (first 10):")
            for etf in results['no_data'][:10]:
                print(f"  {etf['ticker']:<12} {etf['name'][:50]:<50} Error: {etf.get('error', 'Unknown')}")
        
        # ETFs with insufficient data
        if results['insufficient']:
            print(f"\n⚠️  ETFs WITH INSUFFICIENT DATA (first 10):")
            for etf in results['insufficient'][:10]:
                print(f"  {etf['ticker']:<12} {etf['name'][:50]:<50} {etf['days_available']} days")
        
        print("\n" + "="*80)
        
        # Export recommendations
        available_tickers = [etf['ticker'] for etf in results['available']]
        print(f"\n💡 RECOMMENDATION:")
        print(f"  Use {len(available_tickers)} ETFs with available data for analysis")
        print(f"  This represents {summary['availability_rate']:.1f}% of the database")
        
        return available_tickers


def main():
    """Analyze entire ETF database for data availability"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ETF database for data availability')
    parser.add_argument('--type', choices=['holdings', 'price'], default='holdings',
                       help='Type of data to analyze: holdings or price (default: holdings)')
    parser.add_argument('--min-days', type=int, default=30, 
                       help='Minimum days of price data required (default: 30, only for --type price)')
    parser.add_argument('--min-holdings', type=int, default=5,
                       help='Minimum holdings required (default: 5, only for --type holdings)')
    parser.add_argument('--rate-limit', type=float, default=0.1, 
                       help='Seconds between API calls (default: 0.1)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Initialize filter
    etf_filter = ETFDataFilter(debug=args.debug)
    
    if args.type == 'holdings':
        # Analyze holdings data availability
        print("🔍 Analyzing HOLDINGS data availability...\n")
        results = etf_filter.analyze_database_holdings_availability(
            min_holdings=args.min_holdings,
            rate_limit=args.rate_limit
        )
        
        # Print comprehensive report
        available_tickers = etf_filter.print_holdings_analysis_report(results)
        
    else:
        # Analyze price data availability
        print("🔍 Analyzing PRICE data availability...\n")
        results = etf_filter.analyze_database_data_availability(
            min_days=args.min_days,
            rate_limit=args.rate_limit
        )
        
        # Print comprehensive report
        available_tickers = etf_filter.print_analysis_report(results)
    
    # Optionally save results
    print(f"\n💾 To save results, uncomment the save section in the code")
    # import json
    # with open('data_availability_results.json', 'w') as f:
    #     json.dump(results, f, indent=2, default=str)
    # print(f"✅ Results saved to data_availability_results.json")


if __name__ == "__main__":
    main()
