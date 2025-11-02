#!/usr/bin/env python3
"""
Historical Data Validation & Classification
Categorizes ETFs by data maturity and finds peer proxies for immature ETFs

Classification:
- Mature: 312+ days (can use standard 252-day training windows)
- Immature: 60-311 days (needs expanding windows + peer proxy)
- Insufficient: <60 days (exclude from backtesting)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analyzers.etf_risk_classifier import ETFRiskClassifier


def load_historical_data(file_path: str) -> pd.DataFrame:
    """Load parquet file with error handling"""
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None


def analyze_etf_data(ticker: str, data: pd.DataFrame) -> Dict:
    """
    Analyze ETF historical data

    Returns:
        Dict with: total_days, years, first_date, last_date
    """
    if data is None or data.empty:
        return None

    try:
        # Get date index
        if isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        else:
            return None

        total_days = len(data)
        years = total_days / 252
        first_date = dates[0]
        last_date = dates[-1]

        return {
            'ticker': ticker,
            'total_days': total_days,
            'years': years,
            'first_date': first_date,
            'last_date': last_date
        }
    except Exception as e:
        print(f"  Error analyzing {ticker}: {e}")
        return None


def categorize_etf(total_days: int) -> str:
    """Categorize ETF by data maturity"""
    if total_days >= 312:
        return 'mature'
    elif total_days >= 60:
        return 'immature'
    else:
        return 'insufficient'


def calculate_similarity(ticker1: str, ticker2: str,
                        data_dict: Dict[str, pd.DataFrame]) -> float:
    """
    Calculate similarity between two ETFs

    Metrics:
    1. Risk category match (if available)
    2. Price correlation (if overlapping period exists)
    3. Recent momentum correlation

    Returns: Similarity score 0-1 (higher = more similar)
    """
    data1 = data_dict.get(ticker1)
    data2 = data_dict.get(ticker2)

    if data1 is None or data2 is None or data1.empty or data2.empty:
        return 0.0

    try:
        # Extract close prices
        price1 = data1['Close'] if 'Close' in data1.columns else data1.iloc[:, 0]
        price2 = data2['Close'] if 'Close' in data2.columns else data2.iloc[:, 0]

        # Find overlapping dates
        common_dates = price1.index.intersection(price2.index)
        if len(common_dates) < 30:  # Need minimum 30 days overlap
            return 0.0

        # Calculate correlation on overlapping period
        price1_overlap = price1.loc[common_dates]
        price2_overlap = price2.loc[common_dates]

        # Price change correlation
        change1 = price1_overlap.pct_change().dropna()
        change2 = price2_overlap.pct_change().dropna()

        common_changes = set(change1.index).intersection(set(change2.index))
        if len(common_changes) < 20:
            return 0.0

        correlation = change1.loc[list(common_changes)].corr(change2.loc[list(common_changes)])

        # Volatility similarity
        vol1 = change1.std() * np.sqrt(252)
        vol2 = change2.std() * np.sqrt(252)
        vol_similarity = 1.0 - abs(vol1 - vol2) / max(vol1, vol2, 0.01)

        # Combined score: correlation (0.6 weight) + volatility (0.4 weight)
        similarity = 0.6 * max(correlation, 0) + 0.4 * vol_similarity

        return max(0.0, min(1.0, similarity))

    except Exception as e:
        return 0.0


def find_peer_proxy(immature_ticker: str, immature_data: pd.DataFrame,
                   mature_tickers: List[str], data_dict: Dict[str, pd.DataFrame],
                   risk_classifier: ETFRiskClassifier) -> Tuple[str, float]:
    """
    Find best peer proxy (mature ETF) for immature ETF

    Matching criteria:
    1. Same risk category (if available)
    2. Highest price/momentum correlation
    3. Similar volatility profile

    Returns:
        (best_ticker, similarity_score)
    """
    if not mature_tickers:
        return None, 0.0

    # Get risk category for immature ETF (if available)
    immature_risk = None
    try:
        # Try to infer from name or use classifier
        if immature_ticker in risk_classifier.etf_data:
            immature_risk = risk_classifier.etf_data[immature_ticker].get('risk_category')
    except:
        pass

    # Score candidates
    candidates = []
    for mature_ticker in mature_tickers:
        # Prefer same risk category if available
        try:
            mature_risk = risk_classifier.etf_data.get(mature_ticker, {}).get('risk_category')
            risk_match = 1.0 if (immature_risk and mature_risk == immature_risk) else 0.7
        except:
            risk_match = 0.7

        # Calculate similarity
        similarity = calculate_similarity(immature_ticker, mature_ticker, data_dict)

        # Combined score
        combined_score = 0.3 * risk_match + 0.7 * similarity
        candidates.append((mature_ticker, combined_score))

    if not candidates:
        return None, 0.0

    # Return best match
    best_ticker, best_score = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
    return best_ticker if best_score > 0.3 else None, best_score  # Minimum threshold 0.3


def main():
    """Validate and classify all historical data"""

    print("\n" + "="*80)
    print("HISTORICAL DATA VALIDATION & CLASSIFICATION")
    print("="*80)

    # Load all historical data files
    historical_dir = project_root / 'data' / 'historical'

    if not historical_dir.exists():
        print(f"ERROR: Historical data directory not found: {historical_dir}")
        return

    # Get all parquet files
    parquet_files = list(historical_dir.glob('*.parquet'))
    print(f"\nFound {len(parquet_files)} historical data files")

    if not parquet_files:
        print("ERROR: No parquet files found!")
        return

    # Load all data and analyze
    print("\nLoading and analyzing data...")
    results = []
    data_dict = {}  # Keep loaded data for similarity calculations

    for i, file_path in enumerate(parquet_files, 1):
        ticker = file_path.stem.replace('_', '.')

        if i % 50 == 0:
            print(f"  [{i}/{len(parquet_files)}] Processed {i} files...")

        # Load data
        data = load_historical_data(str(file_path))
        if data is None:
            continue

        data_dict[ticker] = data  # Store for later similarity calculations

        # Analyze
        analysis = analyze_etf_data(ticker, data)
        if analysis is None:
            continue

        results.append(analysis)

    print(f"  Successfully analyzed {len(results)} ETFs")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('total_days', ascending=False).reset_index(drop=True)

    # Categorize
    df['category'] = df['total_days'].apply(categorize_etf)

    # Statistics
    print("\n" + "="*80)
    print("DATA CLASSIFICATION RESULTS")
    print("="*80)
    print(f"\nTotal ETFs analyzed: {len(df)}")
    print(f"\nCategory breakdown:")
    print(f"  Mature (312+ days):       {(df['category'] == 'mature').sum():>4} ETFs")
    print(f"  Immature (60-311 days):   {(df['category'] == 'immature').sum():>4} ETFs")
    print(f"  Insufficient (<60 days):  {(df['category'] == 'insufficient').sum():>4} ETFs")

    print(f"\nData coverage:")
    print(f"  Mean years of data:       {df['years'].mean():.2f} years")
    print(f"  Median years of data:     {df['years'].median():.2f} years")
    print(f"  Min years of data:        {df['years'].min():.2f} years")
    print(f"  Max years of data:        {df['years'].max():.2f} years")

    # Find peer proxies for immature ETFs
    print("\nFinding peer proxies for immature ETFs...")

    mature_df = df[df['category'] == 'mature']
    immature_df = df[df['category'] == 'immature']
    mature_tickers = mature_df['ticker'].tolist()

    try:
        risk_classifier = ETFRiskClassifier()
    except:
        risk_classifier = None
        print("  WARNING: Could not load risk classifier, using correlation only")

    peer_proxies = {}
    peer_scores = {}

    for idx, row in immature_df.iterrows():
        immature_ticker = row['ticker']
        immature_data = data_dict.get(immature_ticker)

        if immature_data is not None:
            best_ticker, best_score = find_peer_proxy(
                immature_ticker, immature_data,
                mature_tickers, data_dict, risk_classifier
            )

            if best_ticker:
                peer_proxies[immature_ticker] = best_ticker
                peer_scores[immature_ticker] = best_score

    print(f"  Found {len(peer_proxies)} peer proxies for immature ETFs")

    # Add peer proxy info to DataFrame
    df['peer_proxy_ticker'] = df['ticker'].apply(lambda x: peer_proxies.get(x))
    df['peer_proxy_score'] = df['ticker'].apply(lambda x: peer_scores.get(x))

    # Can backtest if not insufficient
    df['can_backtest'] = df['category'] != 'insufficient'

    print(f"  ETFs eligible for backtesting: {df['can_backtest'].sum()}")

    # Show sample of immature ETFs with their proxies
    if not immature_df.empty:
        print(f"\nSample immature ETFs with peer proxies:")
        sample_immature = immature_df.head(10)
        for idx, row in sample_immature.iterrows():
            ticker = row['ticker']
            proxy = peer_proxies.get(ticker)
            score = peer_scores.get(ticker, 0)
            print(f"  {ticker:12} (days: {row['total_days']:3}) -> {proxy:12} (match: {score:.2f})")

    # Save to parquet
    output_path = project_root / 'data' / 'etf_data_classification.parquet'
    df.to_parquet(output_path, compression='snappy', index=False)

    print(f"\n[OK] Saved classification to: {output_path}")

    # Show detailed statistics
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)

    print("\nMature ETFs (top 10 by data coverage):")
    for idx, row in mature_df.head(10).iterrows():
        print(f"  {row['ticker']:12} {row['total_days']:4} days ({row['years']:.2f} years)")

    print("\nImmature ETFs (top 10, most recent first):")
    for idx, row in immature_df.sort_values('total_days', ascending=False).head(10).iterrows():
        proxy = peer_proxies.get(row['ticker'], 'N/A')
        score = peer_scores.get(row['ticker'], 0)
        print(f"  {row['ticker']:12} {row['total_days']:3} days -> {proxy:12} (match: {score:.2f})")

    print("\n" + "="*80)
    print("READY FOR BACKTESTING")
    print("="*80)
    print(f"\nYou can now run:")
    print(f"  python scripts/run_professional_backtest.py")
    print(f"\nThis will backtest {df['can_backtest'].sum()} eligible ETFs")
    print()


if __name__ == "__main__":
    main()
