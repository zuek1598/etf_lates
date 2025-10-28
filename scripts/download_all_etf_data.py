#!/usr/bin/env python3
"""
Non-interactive Historical Data Downloader
Downloads all missing ETF data automatically (no prompts)

Usage:
    python3 download_all_etf_data.py
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from data_manager.etf_database import ETFDatabase


def download_etf_data(ticker: str, period: str = 'max') -> pd.DataFrame:
    """Download historical data for a single ETF"""
    try:
        data = yf.download(ticker, period=period, interval='1d', 
                         progress=False, prepost=False, auto_adjust=False)
        
        if data.empty:
            return None
        
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Standardize column names
        data.columns = [str(col).title() for col in data.columns]
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            return None
        
        # Keep only required columns
        keep_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        data = data[[col for col in keep_cols if col in data.columns]]
        
        # Reset index
        data = data.reset_index()
        
        return data
        
    except Exception as e:
        return None


def main():
    """Download all missing ETF data"""
    print("="*80)
    print("ETF HISTORICAL DATA DOWNLOADER - AUTO MODE")
    print("="*80)
    print()
    
    # Initialize
    db = ETFDatabase()
    all_tickers = list(db.etf_data.keys())
    historical_dir = Path(__file__).parent / 'data' / 'historical'
    historical_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Total ETFs in database: {len(all_tickers)}")
    
    # Check existing
    existing_files = {f.stem.replace('_', '.') for f in historical_dir.glob('*.parquet')}
    missing = [t for t in all_tickers if t not in existing_files]
    
    print(f"✅ Already have data for: {len(existing_files)} ETFs")
    print(f"📥 Need to download: {len(missing)} ETFs")
    
    if not missing:
        print("\n✨ All ETFs already have data!")
        return
    
    print(f"\n⏱️  Estimated time: ~{len(missing) * 2 // 60} minutes")
    print(f"💾 Saving to: {historical_dir.absolute()}")
    
    print("\n" + "="*80)
    print(f"📥 DOWNLOADING {len(missing)} ETFs (5 year history)")
    print("="*80)
    print()
    
    success = 0
    failed = 0
    failed_list = []
    
    for i, ticker in enumerate(missing, 1):
        print(f"  [{i:3d}/{len(missing)}] {ticker:12s} ", end='', flush=True)
        
        data = download_etf_data(ticker, period='max')
        
        if data is not None and len(data) > 0:
            # Save
            filename = ticker.replace('.', '_') + '.parquet'
            filepath = historical_dir / filename
            data.to_parquet(filepath, compression='snappy', index=False)
            
            # Date range
            date_col = pd.to_datetime(data['Date'])
            start = date_col.min().strftime('%Y-%m-%d')
            end = date_col.max().strftime('%Y-%m-%d')
            
            print(f"✅ {len(data):4d} days ({start} to {end})")
            success += 1
        else:
            print(f"❌ No data")
            failed += 1
            failed_list.append(ticker)
    
    # Summary
    print("\n" + "="*80)
    print("✅ DOWNLOAD COMPLETE")
    print("="*80)
    print(f"   Success: {success}")
    print(f"   Failed: {failed}")
    
    total_files = len(list(historical_dir.glob('*.parquet')))
    total_size = sum(f.stat().st_size for f in historical_dir.glob('*.parquet'))
    
    print(f"\n   Total files: {total_files}")
    print(f"   Total size: {total_size / (1024*1024):.1f} MB")
    
    if failed_list:
        print(f"\n   Failed tickers: {', '.join(failed_list[:10])}")
        if len(failed_list) > 10:
            print(f"   ... and {len(failed_list) - 10} more")
    
    print("\n" + "="*80)
    print("✅ Ready to run full universe backtest!")
    print("   Run: python3 run_analysis.py")
    print("   Then select option 2 (Full universe backtest)")
    print("="*80)
    print()


if __name__ == "__main__":
    main()

