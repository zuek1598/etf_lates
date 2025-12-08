#!/usr/bin/env python3
"""
ETF Name Manager Module
Extends enhance_etf_database.py with additional ETF name management functionality
Provides utilities for adding, updating, and managing ETF names in the database
"""

import sys
import os
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.enhance_etf_database import ETFDatabaseEnhancer
from data_manager.etf_database import ETFDatabase


class ETFNameManager:
    """
    Advanced ETF name management with additional functionality
    Extends the base enhancer with specialized name operations
    """
    
    def __init__(self, cache_file: str = "data/etf_names_cache.json"):
        """
        Initialize ETF name manager
        
        Args:
            cache_file: Path to cache file for ETF names
        """
        self.base_enhancer = ETFDatabaseEnhancer(cache_file)
        self.etf_db = ETFDatabase()
        
    def add_single_etf_name(self, ticker: str, force_update: bool = False) -> Dict:
        """
        Add or update a single ETF name in the cache
        
        Args:
            ticker: ETF ticker symbol
            force_update: Force update even if already cached
            
        Returns:
            Dict with operation result
        """
        print(f"🔍 Processing ETF: {ticker}")
        
        # Check if already exists and not forcing update
        if not force_update and ticker in self.base_enhancer.name_cache:
            cached_info = self.base_enhancer.name_cache[ticker]
            print(f"✅ Already cached: {cached_info.get('longName', 'Unknown')}")
            return {
                'ticker': ticker,
                'status': 'cached',
                'name': cached_info.get('longName', 'Unknown'),
                'action': 'none'
            }
        
        # Fetch new information
        try:
            info = self.base_enhancer.fetch_etf_info(ticker)
            
            if 'fetch_error' in info:
                print(f"❌ Failed to fetch: {info['fetch_error']}")
                return {
                    'ticker': ticker,
                    'status': 'error',
                    'error': info['fetch_error'],
                    'action': 'failed'
                }
            
            print(f"✅ Added: {info.get('longName', 'Unknown')}")
            self.base_enhancer._save_cache()
            
            return {
                'ticker': ticker,
                'status': 'success',
                'name': info.get('longName', 'Unknown'),
                'action': 'added'
            }
            
        except Exception as e:
            print(f"❌ Exception: {e}")
            return {
                'ticker': ticker,
                'status': 'error',
                'error': str(e),
                'action': 'failed'
            }
    
    def add_multiple_etf_names(self, tickers: List[str], force_update: bool = False) -> Dict:
        """
        Add multiple ETF names to the cache
        
        Args:
            tickers: List of ETF ticker symbols
            force_update: Force update even if already cached
            
        Returns:
            Dict with batch operation results
        """
        print(f"🔄 Processing {len(tickers)} ETFs...")
        
        results = {
            'total': len(tickers),
            'added': 0,
            'updated': 0,
            'cached': 0,
            'failed': 0,
            'details': []
        }
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] ", end="")
            
            result = self.add_single_etf_name(ticker, force_update)
            results['details'].append(result)
            
            # Update counters
            if result['status'] == 'success':
                if result['action'] == 'added':
                    results['added'] += 1
                else:
                    results['updated'] += 1
            elif result['status'] == 'cached':
                results['cached'] += 1
            else:
                results['failed'] += 1
        
        # Print summary
        print(f"\n📊 Batch Operation Summary:")
        print(f"  Total processed: {results['total']}")
        print(f"  Added: {results['added']}")
        print(f"  Updated: {results['updated']}")
        print(f"  Already cached: {results['cached']}")
        print(f"  Failed: {results['failed']}")
        
        return results
    
    def search_etfs_by_name(self, search_term: str, limit: int = 10) -> List[Dict]:
        """
        Search ETFs by name (partial match)
        
        Args:
            search_term: Search term to match in ETF names
            limit: Maximum number of results to return
            
        Returns:
            List of matching ETFs with ticker and name
        """
        print(f"🔍 Searching for ETFs containing: '{search_term}'")
        
        matches = []
        search_lower = search_term.lower()
        
        # Search in cache
        for ticker, info in self.base_enhancer.name_cache.items():
            name = info.get('longName', '').lower()
            short_name = info.get('shortName', '').lower()
            
            if search_term in name or search_term in short_name:
                matches.append({
                    'ticker': ticker,
                    'name': info.get('longName', 'Unknown'),
                    'short_name': info.get('shortName', ''),
                    'category': info.get('category', 'Unknown'),
                    'source': 'cache'
                })
        
        # Also search in original database
        for ticker, info in self.etf_db.etf_data.items():
            if ticker in self.base_enhancer.name_cache:
                continue  # Already checked in cache
            
            name = info.get('name', '').lower()
            if search_term in name:
                matches.append({
                    'ticker': ticker,
                    'name': info.get('name', 'Unknown'),
                    'short_name': '',
                    'category': 'Unknown',
                    'source': 'database'
                })
        
        # Limit results
        matches = matches[:limit]
        
        print(f"✅ Found {len(matches)} matching ETFs:")
        for match in matches:
            print(f"  {match['ticker']}: {match['name'][:60]}{'...' if len(match['name']) > 60 else ''}")
        
        return matches
    
    def get_missing_names(self) -> List[str]:
        """
        Get list of ETF tickers that don't have cached names
        
        Returns:
            List of ticker symbols without cached names
        """
        all_tickers = set(self.etf_db.etf_data.keys())
        cached_tickers = set(self.base_enhancer.name_cache.keys())
        missing = list(all_tickers - cached_tickers)
        
        print(f"📊 ETF Name Coverage Analysis:")
        print(f"  Total ETFs in database: {len(all_tickers)}")
        print(f"  ETFs with cached names: {len(cached_tickers)}")
        print(f"  ETFs missing names: {len(missing)}")
        print(f"  Coverage: {len(cached_tickers)/len(all_tickers)*100:.1f}%")
        
        return missing
    
    def update_missing_names(self, batch_size: int = 50) -> Dict:
        """
        Update all missing ETF names in batches
        
        Args:
            batch_size: Number of ETFs to process in each batch
            
        Returns:
            Dict with update results
        """
        missing_tickers = self.get_missing_names()
        
        if not missing_tickers:
            print("✅ All ETFs already have cached names!")
            return {'status': 'complete', 'updated': 0}
        
        print(f"🔄 Updating {len(missing_tickers)} missing ETF names...")
        
        results = {
            'total_missing': len(missing_tickers),
            'updated': 0,
            'failed': 0,
            'batches_processed': 0
        }
        
        for i in range(0, len(missing_tickers), batch_size):
            batch = missing_tickers[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(missing_tickers) - 1) // batch_size + 1
            
            print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} ETFs)...")
            
            batch_results = self.add_multiple_etf_names(batch, force_update=False)
            
            results['updated'] += batch_results['added'] + batch_results['updated']
            results['failed'] += batch_results['failed']
            results['batches_processed'] += 1
            
            # Brief pause between batches
            if i + batch_size < len(missing_tickers):
                print("⏸️ Pausing between batches...")
                time.sleep(2)
        
        print(f"\n✅ Missing names update complete:")
        print(f"  Total processed: {results['total_missing']}")
        print(f"  Successfully updated: {results['updated']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Batches processed: {results['batches_processed']}")
        
        return results
    
    def export_names_to_csv(self, output_file: str = "data/etf_names_export.csv") -> str:
        """
        Export all cached ETF names to CSV file
        
        Args:
            output_file: Output CSV file path
            
        Returns:
            Path to exported file
        """
        print(f"📤 Exporting ETF names to CSV: {output_file}")
        
        export_data = []
        
        for ticker, info in self.base_enhancer.name_cache.items():
            export_data.append({
                'ticker': ticker,
                'long_name': info.get('longName', ''),
                'short_name': info.get('shortName', ''),
                'category': info.get('category', ''),
                'fund_family': info.get('fundFamily', ''),
                'expense_ratio': info.get('expenseRatio', 0.0),
                'currency': info.get('currency', 'AUD'),
                'exchange': info.get('exchange', 'ASX'),
                'last_updated': info.get('last_updated', '')
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        df = df.sort_values('ticker')
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"✅ Exported {len(export_data)} ETF names to {output_file}")
        return output_file
    
    def import_names_from_csv(self, csv_file: str, merge_strategy: str = 'update') -> Dict:
        """
        Import ETF names from CSV file
        
        Args:
            csv_file: Path to CSV file with ETF names
            merge_strategy: How to handle existing entries ('update', 'skip', 'replace')
            
        Returns:
            Dict with import results
        """
        print(f"📥 Importing ETF names from CSV: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"📊 Found {len(df)} ETFs in CSV file")
            
            results = {
                'total_in_file': len(df),
                'imported': 0,
                'skipped': 0,
                'updated': 0,
                'errors': 0
            }
            
            for _, row in df.iterrows():
                ticker = row.get('ticker', '')
                if not ticker:
                    results['errors'] += 1
                    continue
                
                # Create info dict from CSV
                info = {
                    'ticker': ticker,
                    'longName': row.get('long_name', ''),
                    'shortName': row.get('short_name', ''),
                    'category': row.get('category', ''),
                    'fundFamily': row.get('fund_family', ''),
                    'expenseRatio': row.get('expense_ratio', 0.0),
                    'currency': row.get('currency', 'AUD'),
                    'exchange': row.get('exchange', 'ASX'),
                    'last_updated': pd.Timestamp.now().isoformat(),
                    'source': 'csv_import'
                }
                
                # Handle merge strategy
                if ticker in self.base_enhancer.name_cache:
                    if merge_strategy == 'skip':
                        results['skipped'] += 1
                        continue
                    elif merge_strategy == 'update':
                        results['updated'] += 1
                    elif merge_strategy == 'replace':
                        results['updated'] += 1
                else:
                    results['imported'] += 1
                
                # Add to cache
                self.base_enhancer.name_cache[ticker] = info
            
            # Save updated cache
            self.base_enhancer._save_cache()
            
            print(f"✅ Import complete:")
            print(f"  Total in file: {results['total_in_file']}")
            print(f"  New imports: {results['imported']}")
            print(f"  Updated: {results['updated']}")
            print(f"  Skipped: {results['skipped']}")
            print(f"  Errors: {results['errors']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            return {'error': str(e)}
    
    def validate_names(self) -> Dict:
        """
        Validate cached ETF names for consistency and completeness
        
        Returns:
            Dict with validation results
        """
        print("🔍 Validating cached ETF names...")
        
        validation_results = {
            'total_cached': len(self.base_enhancer.name_cache),
            'valid_names': 0,
            'missing_names': 0,
            'duplicate_names': 0,
            'invalid_categories': 0,
            'issues': []
        }
        
        name_counts = {}
        
        for ticker, info in self.base_enhancer.name_cache.items():
            long_name = info.get('longName', '').strip()
            short_name = info.get('shortName', '').strip()
            category = info.get('category', '').strip()
            
            # Check for missing names
            if not long_name and not short_name:
                validation_results['missing_names'] += 1
                validation_results['issues'].append(f"Missing name: {ticker}")
                continue
            
            # Count names for duplicate detection
            display_name = long_name or short_name
            if display_name:
                name_counts[display_name] = name_counts.get(display_name, 0) + 1
            
            # Validate category
            if not category or category == 'Unknown':
                validation_results['invalid_categories'] += 1
            
            validation_results['valid_names'] += 1
        
        # Find duplicates
        duplicates = {name: count for name, count in name_counts.items() if count > 1}
        validation_results['duplicate_names'] = len(duplicates)
        
        if duplicates:
            for name, count in duplicates.items():
                validation_results['issues'].append(f"Duplicate name: '{name}' ({count} ETFs)")
        
        # Print results
        print(f"📊 Validation Results:")
        print(f"  Total cached: {validation_results['total_cached']}")
        print(f"  Valid names: {validation_results['valid_names']}")
        print(f"  Missing names: {validation_results['missing_names']}")
        print(f"  Duplicate names: {validation_results['duplicate_names']}")
        print(f"  Invalid categories: {validation_results['invalid_categories']}")
        
        if validation_results['issues']:
            print(f"\n⚠️ Issues found ({len(validation_results['issues'])}):")
            for issue in validation_results['issues'][:10]:  # Show first 10
                print(f"  - {issue}")
            if len(validation_results['issues']) > 10:
                print(f"  ... and {len(validation_results['issues']) - 10} more")
        else:
            print("✅ No issues found!")
        
        return validation_results


def main():
    """
    Command-line interface for ETF Name Manager
    """
    print("🚀 ETF Name Manager")
    print("=" * 50)
    
    manager = ETFNameManager()
    
    while True:
        print("\n📋 Available Operations:")
        print("1. Add single ETF name")
        print("2. Add multiple ETF names")
        print("3. Search ETFs by name")
        print("4. Show missing names")
        print("5. Update all missing names")
        print("6. Export names to CSV")
        print("7. Import names from CSV")
        print("8. Validate names")
        print("9. Show cache summary")
        print("0. Exit")
        
        try:
            choice = input("\nSelect operation (0-9): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                ticker = input("Enter ETF ticker (e.g., VAS.AX): ").strip().upper()
                if ticker:
                    manager.add_single_etf_name(ticker, force_update=True)
            elif choice == '2':
                tickers_input = input("Enter tickers (comma-separated): ").strip().upper()
                if tickers_input:
                    tickers = [t.strip() for t in tickers_input.split(',')]
                    manager.add_multiple_etf_names(tickers, force_update=True)
            elif choice == '3':
                search_term = input("Enter search term: ").strip()
                if search_term:
                    manager.search_etfs_by_name(search_term)
            elif choice == '4':
                manager.get_missing_names()
            elif choice == '5':
                manager.update_missing_names()
            elif choice == '6':
                filename = input("Enter output filename (default: data/etf_names_export.csv): ").strip()
                if not filename:
                    filename = "data/etf_names_export.csv"
                manager.export_names_to_csv(filename)
            elif choice == '7':
                filename = input("Enter CSV filename: ").strip()
                if filename:
                    strategy = input("Merge strategy (update/skip/replace, default: update): ").strip()
                    if not strategy:
                        strategy = 'update'
                    manager.import_names_from_csv(filename, strategy)
            elif choice == '8':
                manager.validate_names()
            elif choice == '9':
                manager.base_enhancer.print_cache_summary()
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
