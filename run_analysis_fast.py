#!/usr/bin/env python3
"""
Run ETF Analysis with Batch Fetching - Fast Version
Same functionality as run_analysis.py but 10x faster downloads
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import warnings
import pandas as pd
import os
from datetime import datetime

from system.orchestrator import ETFAnalysisSystem
from analyzers.etf_risk_classifier import ETFRiskClassifier


class ETFAnalysisSystemFast(ETFAnalysisSystem):
    """
    Fast version of ETF Analysis System with batch fetching
    Inherits all original functionality but adds optimized downloads
    """
    
    def __init__(self):
        """Initialize fast system with batch-enabled risk classifier"""
        super().__init__()
        
        # Replace the risk classifier with batch-enabled version
        self.risk_classifier = ETFRiskClassifier(enable_cache=True)
        
        # Download benchmark data once
        print("📊 Pre-loading benchmark data...")
        self.risk_classifier.download_benchmark_data()
        print(f"   Loaded {len(self.risk_classifier.benchmark_data)} benchmarks")
    
    def classify_etfs_by_risk(self, etf_tickers: list) -> dict:
        """
        Override classification to use batch fetching
        """
        print(f"🚀 Classifying {len(etf_tickers)} ETFs by risk (FAST MODE)...")
        
        # Use batch downloading instead of individual downloads
        batch_results = self.risk_classifier.download_etf_data_batch(etf_tickers)
        
        # Process results using original logic
        low_risk_etfs = {}
        medium_risk_etfs = {}
        high_risk_etfs = {}
        failed_downloads = []
        
        for ticker, result in batch_results.items():
            try:
                data, quality_tier, quality_score = result
                
                if data is None or quality_tier == "insufficient" or quality_tier == "error":
                    failed_downloads.append(ticker)
                    continue
                
                # Calculate volatility and beta using original methods
                volatility = self.risk_classifier.calculate_enhanced_volatility(data, ticker)
                
                if pd.isna(volatility):
                    failed_downloads.append(ticker)
                    continue
                
                beta, best_benchmark = self.risk_classifier.calculate_volatility_beta(data, ticker)
                
                # Use fallback beta if needed
                beta_confidence = 'normal'
                if pd.isna(beta):
                    if len(data) >= 90:
                        beta = 1.0
                        best_benchmark = list(self.risk_classifier.benchmark_data.keys())[0] if self.risk_classifier.benchmark_data else 'VTS.AX'
                        beta_confidence = 'low'
                        print(f"  ⚠️  Using fallback beta=1.0 for {ticker}")
                    else:
                        failed_downloads.append(ticker)
                        continue
                
                # Classify risk
                risk_category, risk_score = self.risk_classifier.classify_risk(volatility, beta)
                
                # Store in appropriate risk category
                etf_data = {
                    'data': data,
                    'volatility': volatility,
                    'beta': beta,
                    'best_benchmark': best_benchmark,
                    'quality_tier': quality_tier,
                    'quality_score': quality_score,
                    'risk_score': risk_score,
                    'etf_info': self.etf_database.etf_data.get(ticker, {})
                }
                
                if risk_category == 'LOW':
                    low_risk_etfs[ticker] = etf_data
                elif risk_category == 'MEDIUM':
                    medium_risk_etfs[ticker] = etf_data
                elif risk_category == 'HIGH':
                    high_risk_etfs[ticker] = etf_data
                
                print(f"  ✅ {ticker}: {risk_category} risk (Vol: {volatility:.1%}, Beta: {beta:.2f})")
                
            except Exception as e:
                print(f"  ❌ Error processing {ticker}: {str(e)[:50]}")
                failed_downloads.append(ticker)
        
        # Create summary
        summary = {
            'total_processed': len(etf_tickers),
            'low_risk_count': len(low_risk_etfs),
            'medium_risk_count': len(medium_risk_etfs),
            'high_risk_count': len(high_risk_etfs),
            'failed_count': len(failed_downloads),
            'failed_downloads': failed_downloads
        }
        
        results = {
            'low_risk_etfs': low_risk_etfs,
            'medium_risk_etfs': medium_risk_etfs,
            'high_risk_etfs': high_risk_etfs
        }
        
        print(f"\n{'='*60}")
        print("FAST CLASSIFICATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total processed: {summary['total_processed']}")
        print(f"Low risk: {summary['low_risk_count']}")
        print(f"Medium risk: {summary['medium_risk_count']}")
        print(f"High risk: {summary['high_risk_count']}")
        print(f"Failed downloads: {summary['failed_count']}")
        print(f"💰 Time saved: ~{len(etf_tickers) * 2.0:.0f} seconds vs individual downloads")
        
        return results


def print_analysis_summary(results, etf_db):
    """Print comprehensive analysis summary"""
    print("\n" + "="*80)
    print(" " * 25 + "FAST ANALYSIS SUMMARY REPORT")
    print("="*80)

    analysis_results = results.get('analysis_results', {})
    risk_classifications = results.get('risk_classifications', {})

    universe_data = []
    for ticker, analysis in analysis_results.items():
        etf_info = etf_db.etf_data.get(ticker, {})
        risk_raw = risk_classifications.get(ticker, analysis.get('risk_category', 'MEDIUM'))
        risk_map = {'low_risk_etfs': 'LOW', 'medium_risk_etfs': 'MEDIUM', 'high_risk_etfs': 'HIGH'}
        risk = risk_map.get(risk_raw, risk_raw.upper() if isinstance(risk_raw, str) else 'MEDIUM')
        
        # Get full ETF name
        try:
            import yfinance as yf
            yf_ticker = yf.Ticker(ticker)
            full_name = yf_ticker.info.get('longName', etf_info.get('name', ticker))
        except:
            full_name = etf_info.get('name', ticker)

        universe_data.append({
            'ticker': ticker,
            'name': full_name,
            'risk_category': risk,
            'composite_score': analysis.get('composite_score', 0.0),
            'cvar': analysis.get('cvar', 0.0),
            'ml_forecast': analysis.get('ml_forecast', 0.0),
            'hit_rate': analysis.get('hit_rate', 0.0),
            'volatility': analysis.get('volatility', 0.0),
            'beta': analysis.get('beta', 0.0)
        })

    universe_df = pd.DataFrame(universe_data)

    # Overall Statistics
    print(f"\n📊 OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total ETFs Analyzed:         {len(universe_df)}")
    print(f"Average Composite Score:     {universe_df['composite_score'].mean():.3f}")
    print(f"Average CVaR:                {universe_df['cvar'].mean():.4f}")
    print(f"Average ML Forecast:         {universe_df['ml_forecast'].mean():+.2f}%")
    print(f"Average Hit Rate:            {universe_df['hit_rate'].mean():.2f}")
    print(f"Average Volatility:          {universe_df['volatility'].mean():.1%}")
    print(f"Average Beta:                {universe_df['beta'].mean():.2f}")

    # Risk Distribution
    print(f"\n🎯 RISK CATEGORY DISTRIBUTION")
    print("-" * 80)
    risk_counts = universe_df['risk_category'].value_counts()
    for risk in ['LOW', 'MEDIUM', 'HIGH']:
        if risk in risk_counts.index:
            count = risk_counts[risk]
            pct = (count / len(universe_df)) * 100
            print(f"[{risk[0]}] {risk:10} Risk: {count:4} ETFs ({pct:5.1f}%)")
    
    # Top performers
    print(f"\n🏆 TOP PERFORMERS")
    print("-" * 80)
    
    top_composite = universe_df.nlargest(3, 'composite_score')
    print(f"Top 3 by Composite Score:")
    for i, (_, row) in enumerate(top_composite.iterrows(), 1):
        print(f"  {i}. {row['ticker']:8} ({row['name'][:30]:30}) - Score: {row['composite_score']:.3f}")
    
    top_ml = universe_df.nlargest(3, 'ml_forecast')
    print(f"\nTop 3 by ML Forecast:")
    for i, (_, row) in enumerate(top_ml.iterrows(), 1):
        print(f"  {i}. {row['ticker']:8} - Forecast: {row['ml_forecast']:+.2f}% (Hit Rate: {row['hit_rate']:.2f})")

    return universe_df


def main():
    """Main function to run fast analysis"""
    parser = argparse.ArgumentParser(description='Run Fast ETF Analysis with Batch Fetching')
    parser.add_argument('--etfs', nargs='+', help='Specific ETF tickers to analyze')
    parser.add_argument('--sample', type=int, help='Analyze a sample of N random ETFs')
    parser.add_argument('--save', action='store_true', help='Save results to Parquet files')
    
    args = parser.parse_args()
    
    # Setup
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Initialize fast system
    print("🚀 Initializing FAST ETF Analysis System...")
    system = ETFAnalysisSystemFast()
    
    # Determine which ETFs to analyze
    etf_db = system.etf_database
    
    if args.etfs:
        etf_tickers = args.etfs
        print(f"Analyzing specified ETFs: {etf_tickers}")
    elif args.sample:
        all_tickers = list(etf_db.etf_data.keys())
        import random
        random.seed(42)
        etf_tickers = random.sample(all_tickers, min(args.sample, len(all_tickers)))
        print(f"Analyzing random sample of {len(etf_tickers)} ETFs")
    else:
        etf_tickers = list(etf_db.etf_data.keys())
        print(f"Analyzing all {len(etf_tickers)} ETFs in database")
    
    # Run analysis
    print(f"\n🎯 Starting FAST analysis...")
    start_time = datetime.now()
    
    results = system.run_full_analysis(etf_tickers)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Print summary
    universe_df = print_analysis_summary(results, etf_db)
    
    # Save results if requested
    if args.save:
        print(f"\n💾 Saving results...")
        universe_file = data_dir / 'etf_universe_fast.parquet'
        universe_df.to_parquet(universe_file)
        print(f"  ✅ Saved {len(universe_df)} ETFs to {universe_file.name}")
    
    print(f"\n🎉 FAST analysis complete!")
    print(f"⚡ Processing time: {processing_time:.1f} seconds")
    print(f"💰 Estimated time saved: ~{len(etf_tickers) * 2.0 - processing_time:.0f} seconds")
    
    return results


if __name__ == "__main__":
    results = main()
