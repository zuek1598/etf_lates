#!/usr/bin/env python3
"""
ETF Name Populator
Systematically fetches and adds ETF names from Yahoo Finance to etf_database.py

This script:
1. Loads the current ETF database
2. Identifies ETFs missing names
3. Fetches names from Yahoo Finance API
4. Updates etf_database.py with all names
5. Creates a backup before making changes

Usage:
    python data_manager/populate_etf_names.py
"""

import yfinance as yf
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_manager.etf_database import ETFDatabase


class ETFNamePopulator:
    """Fetches and populates ETF names from Yahoo Finance"""
    
    def __init__(self):
        self.db = ETFDatabase()
        self.rate_limit_delay = 0.5  # Delay between API calls (seconds)
        self.max_retries = 3
        self.fetched_names = {}
        self.failed_tickers = []
        
    def fetch_name_from_yahoo(self, ticker: str) -> Tuple[str, bool]:
        """
        Fetch ETF name from Yahoo Finance
        
        Args:
            ticker: ETF ticker symbol
            
        Returns:
            Tuple of (name, success)
        """
        for attempt in range(self.max_retries):
            try:
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info
                
                # Try multiple name fields
                name = (
                    info.get('longName') or
                    info.get('shortName') or
                    info.get('name') or
                    None
                )
                
                if name:
                    return name, True
                    
                # If no name found, return ticker as fallback
                return ticker, False
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * (attempt + 1))
                    continue
                else:
                    print(f"  ❌ Failed to fetch {ticker}: {str(e)[:50]}")
                    return ticker, False
        
        return ticker, False
    
    def identify_missing_names(self) -> List[str]:
        """Identify ETFs that are missing names"""
        missing = []
        for ticker, info in self.db.etf_data.items():
            if 'name' not in info or not info.get('name') or info.get('name') == ticker:
                missing.append(ticker)
        return missing
    
    def fetch_all_names(self, tickers: List[str]) -> Dict[str, str]:
        """
        Fetch names for all specified tickers
        
        Args:
            tickers: List of ticker symbols to fetch
            
        Returns:
            Dictionary mapping ticker -> name
        """
        print(f"\n{'='*70}")
        print(f"FETCHING ETF NAMES FROM YAHOO FINANCE")
        print(f"{'='*70}")
        print(f"Total ETFs to fetch: {len(tickers)}")
        print(f"Rate limit: {self.rate_limit_delay}s between requests")
        print(f"{'='*70}\n")
        
        names = {}
        total = len(tickers)
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{total}] Fetching {ticker}...", end=' ', flush=True)
            
            name, success = self.fetch_name_from_yahoo(ticker)
            names[ticker] = name
            
            if success:
                print(f"✅ {name[:50]}")
            else:
                print(f"⚠️  Using ticker as fallback")
                self.failed_tickers.append(ticker)
            
            # Rate limiting
            if i < total:
                time.sleep(self.rate_limit_delay)
        
        return names
    
    def update_database_file(self, new_names: Dict[str, str]) -> bool:
        """
        Update etf_database.py file with new names
        
        Args:
            new_names: Dictionary mapping ticker -> name
            
        Returns:
            True if successful, False otherwise
        """
        db_file = Path(__file__).parent / 'etf_database.py'
        
        if not db_file.exists():
            print(f"❌ Database file not found: {db_file}")
            return False
        
        # Create backup
        backup_file = db_file.parent / f'etf_database_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'
        print(f"\n📋 Creating backup: {backup_file.name}")
        
        try:
            with open(db_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Save backup
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update content with new names using regex
            updated_content = content
            updates_count = 0
            
            for ticker, name in new_names.items():
                # Escape single quotes and newlines in name
                escaped_name = name.replace("'", "\\'").replace("\n", " ").strip()
                
                # Pattern to find the ETF entry: 'TICKER': { ... }
                # Match the entire entry including the closing brace
                pattern = rf"('{re.escape(ticker)}':\s*{{)([^}}]*?)(}})"
                
                def replace_match(match):
                    nonlocal updates_count
                    entry_start = match.group(1)  # "'TICKER': {"
                    entry_content = match.group(2)  # Content inside braces
                    entry_end = match.group(3)  # "}"
                    
                    # Check if name already exists
                    if "'name':" in entry_content:
                        # Replace existing name
                        entry_content = re.sub(
                            r"'name':\s*'[^']*'",
                            f"'name': '{escaped_name}'",
                            entry_content
                        )
                        updates_count += 1
                    else:
                        # Add name before the closing brace
                        entry_content = entry_content.rstrip()
                        # Ensure there's a comma before adding name
                        if entry_content and not entry_content.endswith(','):
                            entry_content += ','
                        entry_content += f" 'name': '{escaped_name}'"
                        updates_count += 1
                    
                    return entry_start + entry_content + entry_end
                
                # Use DOTALL flag to match across newlines if needed
                new_content = re.sub(pattern, replace_match, updated_content, flags=re.DOTALL)
                if new_content != updated_content:
                    updated_content = new_content
            
            # Write updated content
            with open(db_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✅ Updated {updates_count} ETF entries in database")
            print(f"💾 Backup saved: {backup_file.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating database file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self, fetch_missing_only: bool = True):
        """
        Run the name population process
        
        Args:
            fetch_missing_only: If True, only fetch names for ETFs missing them
        """
        print(f"\n{'='*70}")
        print(f"ETF NAME POPULATOR")
        print(f"{'='*70}")
        print(f"Database: {len(self.db.etf_data)} ETFs")
        
        # Identify ETFs needing names
        if fetch_missing_only:
            tickers_to_fetch = self.identify_missing_names()
            print(f"ETFs missing names: {len(tickers_to_fetch)}")
        else:
            tickers_to_fetch = list(self.db.etf_data.keys())
            print(f"Fetching names for all ETFs: {len(tickers_to_fetch)}")
        
        if not tickers_to_fetch:
            print("\n✅ All ETFs already have names!")
            return
        
        # Confirm before proceeding
        print(f"\n⚠️  This will make {len(tickers_to_fetch)} API calls to Yahoo Finance")
        print(f"   Estimated time: ~{len(tickers_to_fetch) * self.rate_limit_delay / 60:.1f} minutes")
        
        try:
            response = input("\nProceed? (y/n): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return
        except KeyboardInterrupt:
            print("\nCancelled.")
            return
        
        # Fetch names
        new_names = self.fetch_all_names(tickers_to_fetch)
        
        # Summary
        print(f"\n{'='*70}")
        print(f"FETCH SUMMARY")
        print(f"{'='*70}")
        print(f"Total fetched: {len(new_names)}")
        print(f"Successful: {len(new_names) - len(self.failed_tickers)}")
        print(f"Failed: {len(self.failed_tickers)}")
        
        if self.failed_tickers:
            print(f"\n⚠️  Failed tickers:")
            for ticker in self.failed_tickers[:10]:
                print(f"   - {ticker}")
            if len(self.failed_tickers) > 10:
                print(f"   ... and {len(self.failed_tickers) - 10} more")
        
        # Update database file
        print(f"\n{'='*70}")
        print(f"UPDATING DATABASE FILE")
        print(f"{'='*70}")
        
        success = self.update_database_file(new_names)
        
        if success:
            print(f"\n✅ SUCCESS!")
            print(f"   Database updated with {len(new_names)} ETF names")
            print(f"   Backup created before changes")
        else:
            print(f"\n❌ FAILED to update database file")
            print(f"   Backup was created, but update failed")


def main():
    """Main entry point"""
    populator = ETFNamePopulator()
    populator.run(fetch_missing_only=True)


if __name__ == "__main__":
    main()

