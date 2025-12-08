#!/usr/bin/env python3
"""
ETF Data Filtration Layer - R&D Prototype
Filters ETFs based on data availability, risk class, and dynamic classification

Purpose: 
- Eliminate need for EODHD subscription
- Focus on ETFs with rich holdings data from yfinance/yahooquery
- Exclude low risk ETFs for growth focus
- Dynamic region/sector identification from actual holdings
"""

import yahooquery as yq
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import warnings
warnings.filterwarnings('ignore')


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
    
    def check_data_availability(self, ticker: str) -> bool:
        """
        Check if ETF has sufficient holdings data available
        
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
        Filter entire ETF universe
        
        Args:
            etf_list: List of ETF tickers to filter
            risk_categories: Dictionary of ticker -> risk_category mapping
            
        Returns:
            dict: Filtering results and statistics
        """
        print(f"🔍 Starting filtration of {len(etf_list)} ETFs...")
        
        filtered_results = []
        excluded_count = 0
        data_unavailable = 0
        risk_excluded = 0
        
        for i, ticker in enumerate(etf_list):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(etf_list)} ({i/len(etf_list)*100:.1f}%)")
            
            risk_category = risk_categories.get(ticker) if risk_categories else None
            result = self.filter_single_etf(ticker, risk_category)
            
            if result:
                filtered_results.append(result)
            else:
                excluded_count += 1
                if risk_category and self.should_exclude_risk_class(risk_category):
                    risk_excluded += 1
                else:
                    data_unavailable += 1
            
            # Rate limiting to avoid overwhelming Yahoo
            time.sleep(0.1)
        
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


def main():
    """Test the ETF filtration system"""
    
    # Example usage with a few ETFs
    test_etfs = ['VAS.AX', 'VTS.AX', 'IOZ.AX', 'ASIA.AX', 'GOLD.AX']
    
    # Mock risk categories for testing
    risk_categories = {
        'VAS.AX': 'LOW',
        'VTS.AX': 'HIGH', 
        'IOZ.AX': 'LOW',
        'ASIA.AX': 'HIGH',
        'GOLD.AX': 'MEDIUM'
    }
    
    # Initialize filter
    etf_filter = ETFDataFilter(debug=True)
    
    # Filter the universe
    results = etf_filter.filter_etf_universe(test_etfs, risk_categories)
    
    # Display results
    print(f"\n🎯 Filtered ETFs:")
    for etf in results['filtered_etfs']:
        print(f"  {etf['ticker']}: {etf['name']}")
        print(f"    Asset Class: {etf['asset_class']}")
        print(f"    Sector: {etf['primary_sector']} ({etf['sector_weight']:.1%})")
        print(f"    Region: {etf['primary_region']} ({etf['region_weight']:.1%})")
        print(f"    Holdings: {etf['holdings_count']} ({etf['data_quality']} data)")
        print()


if __name__ == "__main__":
    main()
