"""
ETF Analysis Runner - Modified System
Runs comprehensive ETF analysis and saves results for interactive dashboard

Author: ETF Analysis System - Modified
Date: October 22, 2025
Version: 3.0

This script performs full analysis of ETFs through the modified system with:
- Risk Component (CVaR, Ulcer, Beta, IR) with 30/30/20/20 weighting
- ML Ensemble (raw forecasts + confidence, NO bias correction)
- Adaptive Kalman Hull Supertrend (momentum indicator)
- Volume Intelligence (spike, correlation, A/D)
- Composite scoring with 40/30/30 weighting

Results are saved in Parquet format for Dash dashboard.
"""

# Fix imports so this works from anywhere
import sys
from pathlib import Path
# Add project root to path if not already there
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import warnings
import pandas as pd
import os
from datetime import datetime

# Import analysis system
from system.orchestrator import ETFAnalysisSystem
from data_manager.data_manager import ETFDataManager as ETFDatabase


def setup_data_directory():
    """Create data directory if it doesn't exist"""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    return data_dir


def print_analysis_summary(results, etf_db):
    """
    Print comprehensive analysis summary with key forecasts and metrics
    before saving to Parquet files
    """
    print("\n" + "="*80)
    print(" " * 25 + "ANALYSIS SUMMARY REPORT")
    print("="*80)
    
    analysis_results = results.get('analysis_results', {})
    rankings = results.get('rankings', {})
    
    # Get actual risk classifications from orchestrator (not from rankings)
    risk_classifications = results.get('risk_classifications', {})
    
    universe_data = []
    for ticker, analysis in analysis_results.items():
        etf_info = etf_db.etf_data.get(ticker, {})
        
        # Use actual risk classification and normalize the name
        risk_raw = risk_classifications.get(ticker, analysis.get('risk_category', 'MEDIUM'))
        # Normalize risk category names: low_risk_etfs -> LOW, medium_risk_etfs -> MEDIUM, high_risk_etfs -> HIGH
        risk_map = {'low_risk_etfs': 'LOW', 'medium_risk_etfs': 'MEDIUM', 'high_risk_etfs': 'HIGH'}
        risk = risk_map.get(risk_raw, risk_raw.upper() if isinstance(risk_raw, str) else 'MEDIUM')
        
        universe_data.append({
            'ticker': ticker,
            'name': etf_info.get('name', ticker),
            'risk_category': risk,
            'composite_score': analysis.get('composite_percentile', analysis.get('composite_score', 0.0)),
            
            # Risk Component (NEW)
            'cvar': analysis.get('cvar', 0.0),
            'ulcer_index': analysis.get('ulcer_index', 0.0),
            'beta': analysis.get('beta', 0.0),
            'information_ratio': analysis.get('information_ratio', 0.0),
            'risk_score': analysis.get('risk_score', 0.0),
            
            # Kalman Hull (NEW)
            'kalman_trend': analysis.get('kalman_trend', 0),
            'kalman_signal_strength': analysis.get('kalman_signal_strength', 0.0),
            'kalman_efficiency_ratio': analysis.get('kalman_efficiency_ratio', 0.5),
            
            # Volume Intelligence (NEW)
            'volume_spike_score': analysis.get('volume_spike_score', 0.0),
            'volume_correlation': analysis.get('volume_correlation', 0.0),
            'volume_confidence': analysis.get('volume_confidence', 0.0),
            
            # ML Ensemble (MODIFIED - no bias correction)
            'ml_forecast': analysis.get('ml_forecast', 0.0),
            'ml_confidence': analysis.get('ml_confidence', 0.5),
            'mae_score': analysis.get('mae_score', 0.0),
            'hit_rate': analysis.get('hit_rate', 0.0),
            
            # Returns
            'ytd_return': analysis.get('ytd_return', 0.0),
            'volatility': analysis.get('volatility', 0.0)
        })
    
    universe_df = pd.DataFrame(universe_data)
    
    # Overall Statistics
    print(f"\nOVERALL STATISTICS")
    print("-" * 80)
    print(f"Total ETFs Analyzed:         {len(universe_df)}")
    print(f"Average Composite Score:     {universe_df['composite_score'].mean():.2f}")
    print(f"Average Risk Score:          {universe_df['risk_score'].mean():.3f}")
    print(f"Average CVaR:                {universe_df['cvar'].mean():.4f}")
    print(f"Average Ulcer Index:         {universe_df['ulcer_index'].mean():.4f}")
    print(f"Average YTD Return:          {universe_df['ytd_return'].mean()*100:+.2f}%")
    print(f"Average ML Forecast:         {universe_df['ml_forecast'].mean():+.2f}%")
    
    # Risk Distribution
    print(f"\nRISK CATEGORY DISTRIBUTION")
    print("-" * 80)
    risk_counts = universe_df['risk_category'].value_counts()
    for risk in ['LOW', 'MEDIUM', 'HIGH']:
        if risk in risk_counts.index:
            count = risk_counts[risk]
            pct = (count / len(universe_df)) * 100
            risk_label = f"[{risk[0]}]" if risk != "MEDIUM" else "[M]"
            print(f"{risk_label} {risk:10} Risk: {count:4} ETFs ({pct:5.1f}%)")
    
    # ML Forecast Summary
    print(f"\nML FORECAST SUMMARY (60-Day Outlook)")
    print("-" * 80)
    positive_forecasts = len(universe_df[universe_df['ml_forecast'] > 0])
    negative_forecasts = len(universe_df[universe_df['ml_forecast'] < 0])
    avg_confidence = universe_df['ml_confidence'].mean()
    
    print(f"Positive Forecasts:    {positive_forecasts} ETFs ({positive_forecasts/len(universe_df)*100:.1f}%)")
    print(f"Negative Forecasts:    {negative_forecasts} ETFs ({negative_forecasts/len(universe_df)*100:.1f}%)")
    print(f"Average Confidence:    {avg_confidence:.2f}")
    print(f"Best Forecast:         {universe_df['ml_forecast'].max():+.2f}% "
          f"({universe_df.nlargest(1, 'ml_forecast')['ticker'].iloc[0]})")
    print(f"Worst Forecast:        {universe_df['ml_forecast'].min():+.2f}% "
          f"({universe_df.nsmallest(1, 'ml_forecast')['ticker'].iloc[0]})")
    
    # Kalman Hull Trend Summary
    print(f"\nKALMAN HULL TREND SUMMARY")
    print("-" * 80)
    uptrend = len(universe_df[universe_df['kalman_trend'] == 1])
    downtrend = len(universe_df[universe_df['kalman_trend'] == -1])
    neutral = len(universe_df[universe_df['kalman_trend'] == 0])
    
    print(f"Uptrend:     {uptrend} ETFs ({uptrend/len(universe_df)*100:.1f}%)")
    print(f"Downtrend:   {downtrend} ETFs ({downtrend/len(universe_df)*100:.1f}%)")
    print(f"Neutral:     {neutral} ETFs ({neutral/len(universe_df)*100:.1f}%)")
    print(f"Avg Signal Strength:   {universe_df['kalman_signal_strength'].mean():.3f}")
    print(f"Avg Efficiency Ratio:  {universe_df['kalman_efficiency_ratio'].mean():.3f}")
    
    # Volume Intelligence Summary
    print(f"\nVOLUME INTELLIGENCE SUMMARY")
    print("-" * 80)
    high_volume_spike = len(universe_df[universe_df['volume_spike_score'] > 70])
    positive_correlation = len(universe_df[universe_df['volume_correlation'] > 0.5])
    
    print(f"High Volume Spikes:          {high_volume_spike} ETFs")
    print(f"Strong Price-Vol Correlation: {positive_correlation} ETFs")
    print(f"Avg Volume Confidence:       {universe_df['volume_confidence'].mean():.3f}")
    
    # Detailed analysis by risk category
    for risk_cat in ["LOW", "MEDIUM", "HIGH"]:
        risk_data = universe_df[universe_df['risk_category'] == risk_cat]

        if len(risk_data) == 0:
            continue

        print(f"\n" + "="*80)
        print(f"{risk_cat} RISK CATEGORY ANALYSIS ({len(risk_data)} ETFs)")
        print("="*80)
        
        # Category Statistics
        print(f"\nCategory Statistics:")
        print(f"  Average Composite Score:     {risk_data['composite_score'].mean():.2f}")
        print(f"  Average Risk Score:          {risk_data['risk_score'].mean():.3f}")
        print(f"  Average CVaR:                {risk_data['cvar'].mean():.4f}")
        print(f"  Average Ulcer Index:         {risk_data['ulcer_index'].mean():.4f}")
        print(f"  Average Beta:                {risk_data['beta'].mean():.2f}")
        print(f"  Average Information Ratio:   {risk_data['information_ratio'].mean():.2f}")
        print(f"  Average Volatility:          {risk_data['volatility'].mean():.2%}")
        print(f"  Average YTD Return:          {risk_data['ytd_return'].mean()*100:+.2f}%")
        
        # Forecast Metrics
        print(f"\nForecast Metrics:")
        print(f"  Average ML Forecast:         {risk_data['ml_forecast'].mean():+.2f}%")
        print(f"  Average Confidence:          {risk_data['ml_confidence'].mean():.2f}")
        
        pos_forecasts = len(risk_data[risk_data['ml_forecast'] > 0])
        print(f"  Positive Forecasts:          {pos_forecasts}/{len(risk_data)} ({pos_forecasts/len(risk_data)*100:.1f}%)")
        
        # Trend Distribution
        uptrend_cat = len(risk_data[risk_data['kalman_trend'] == 1])
        print(f"  Uptrend ETFs:                {uptrend_cat}/{len(risk_data)} ({uptrend_cat/len(risk_data)*100:.1f}%)")
        
        # Top 10 ETFs in this category
        print(f"\nTop 10 ETFs:")
        print(f"{'Rank':<6}{'Ticker':<12}{'Name':<40}{'Score':<8}{'Trend':<8}{'Forecast':<10}{'YTD %':<8}")
        print("-" * 80)
        
        top_10 = risk_data.nlargest(10, 'composite_score')
        for idx, row in enumerate(top_10.itertuples(), 1):
            name_short = row.name[:37] + "..." if len(row.name) > 40 else row.name
            trend_str = '↑' if row.kalman_trend == 1 else '↓' if row.kalman_trend == -1 else '→'
            print(f"{idx:<6}{row.ticker:<12}{name_short:<40}"
                  f"{row.composite_score:>6.1f}  {trend_str:>6}  "
                  f"{row.ml_forecast:>+6.1f}%   {row.ytd_return*100:>6.1f}")
        
        # Best forecasts in category
        print(f"\nBest ML Forecasts:")
        best_forecasts = risk_data.nlargest(5, 'ml_forecast')
        for idx, row in enumerate(best_forecasts.itertuples(), 1):
            print(f"  {idx}. {row.ticker:<10} {row.ml_forecast:>+6.2f}%  "
                  f"(Confidence: {row.ml_confidence:.2f}, Score: {row.composite_score:.1f})")
    
    # Overall Top Performers
    print(f"\n" + "="*80)
    print(f"TOP 15 ETFS (ALL CATEGORIES)")
    print("="*80)
    print(f"{'Rank':<6}{'Ticker':<12}{'Name':<30}{'Risk':<10}{'Score':<8}{'Trend':<8}{'Forecast':<10}{'YTD %':<8}")
    print("-" * 80)
    
    top_15 = universe_df.nlargest(15, 'composite_score')
    for idx, row in enumerate(top_15.itertuples(), 1):
        name_short = row.name[:27] + "..." if len(row.name) > 30 else row.name
        trend_str = '↑' if row.kalman_trend == 1 else '↓' if row.kalman_trend == -1 else '→'
        print(f"{idx:<6}{row.ticker:<12}{name_short:<30}{row.risk_category:<9}"
              f"{row.composite_score:>6.1f}  {trend_str:>6}  "
              f"{row.ml_forecast:>+6.1f}%   {row.ytd_return*100:>6.1f}")
    
    # Key Insights
    print(f"\n" + "="*80)
    print(f"KEY INSIGHTS")
    print("="*80)
    
    best_score = universe_df.nlargest(1, 'composite_score').iloc[0]
    best_risk = universe_df.nsmallest(1, 'risk_score').iloc[0]
    best_return = universe_df.nlargest(1, 'ytd_return').iloc[0]
    best_forecast = universe_df.nlargest(1, 'ml_forecast').iloc[0]
    
    print(f"\n• Best Overall Score:      {best_score['ticker']} ({best_score['composite_score']:.1f}) - {best_score['name']}")
    print(f"• Best Risk Score:         {best_risk['ticker']} ({best_risk['risk_score']:.3f}) - {best_risk['name']}")
    print(f"• Best YTD Return:         {best_return['ticker']} ({best_return['ytd_return']*100:+.2f}%) - {best_return['name']}")
    print(f"• Best ML Forecast:        {best_forecast['ticker']} ({best_forecast['ml_forecast']:+.2f}%) - {best_forecast['name']}")
    
    print("\n" + "="*80 + "\n")


def save_analysis_to_parquet(results, etf_db, data_dir='data'):
    """
    Save analysis results to Parquet files optimized for Dash dashboard
    
    Creates 4 files:
    1. etf_universe.parquet - All ETFs with key metrics
    2. rankings_low_risk.parquet - Top low-risk ETFs
    3. rankings_medium_risk.parquet - Top medium-risk ETFs
    4. rankings_high_risk.parquet - Top high-risk ETFs
    
    Args:
        results: Dict from ETFAnalysisSystem.run_full_analysis()
        etf_db: ETFDatabase instance
        data_dir: Directory to save files
        
    Returns:
        dict: File paths and statistics
    """
    print("\n" + "="*60)
    print("SAVING ANALYSIS RESULTS TO PARQUET FORMAT")
    print("="*60)
    
    data_path = Path(data_dir)
    analysis_results = results.get('analysis_results', {})
    rankings = results.get('rankings', {})
    
    saved_files = {}
    
    # 1. Create Universe DataFrame (all ETFs)
    print("\n1. Creating universe table...")
    universe_data = []
    
    # Get actual risk classifications from orchestrator results
    risk_classifications = results.get('risk_classifications', {})
    
    for ticker, analysis in analysis_results.items():
        etf_info = etf_db.etf_data.get(ticker, {})
        
        # Use actual risk classification and normalize the name
        risk_raw = risk_classifications.get(ticker, analysis.get('risk_category', 'MEDIUM'))
        # Normalize risk category names: low_risk_etfs -> LOW, medium_risk_etfs -> MEDIUM, high_risk_etfs -> HIGH
        risk_map = {'low_risk_etfs': 'LOW', 'medium_risk_etfs': 'MEDIUM', 'high_risk_etfs': 'HIGH'}
        risk = risk_map.get(risk_raw, risk_raw.upper() if isinstance(risk_raw, str) else 'MEDIUM')
        
        universe_data.append({
            # Identifiers
            'ticker': ticker,
            'name': etf_info.get('name', ticker),
            'subcategory': etf_info.get('subcategory', 'Unknown'),

            # Risk metrics (NEW STRUCTURE)
            'risk_category': risk,
            'risk_score': analysis.get('risk_score', 0.0),
            'cvar': analysis.get('cvar', 0.0),
            'ulcer_index': analysis.get('ulcer_index', 0.0),
            'beta': analysis.get('beta', 0.0),
            'information_ratio': analysis.get('information_ratio', 0.0),
            'best_benchmark': analysis.get('best_benchmark', 'N/A'),
            'volatility': analysis.get('volatility', 0.0),

            # Kalman Hull Supertrend (NEW)
            'kalman_trend': analysis.get('kalman_trend', 0),
            'kalman_price': analysis.get('kalman_price', 0.0),
            'kalman_upper_band': analysis.get('kalman_upper_band', 0.0),
            'kalman_lower_band': analysis.get('kalman_lower_band', 0.0),
            'kalman_efficiency_ratio': analysis.get('kalman_efficiency_ratio', 0.5),
            'kalman_divergence': analysis.get('kalman_divergence', 'none'),
            'kalman_consistency': analysis.get('kalman_consistency', False),
            'kalman_signal_strength': analysis.get('kalman_signal_strength', 0.0),

            # Volume Intelligence (NEW)
            'volume_spike_score': analysis.get('volume_spike_score', 0.0),
            'volume_correlation': analysis.get('volume_correlation', 0.0),
            'volume_ad_signal': analysis.get('volume_ad_signal', 'neutral'),
            'volume_confidence': analysis.get('volume_confidence', 0.0),

            # ML Ensemble (MODIFIED - no bias correction)
            'ml_forecast': analysis.get('ml_forecast', 0.0),
            'ml_confidence': analysis.get('ml_confidence', 0.5),
            'mae_score': analysis.get('mae_score', 0.0),
            'hit_rate': analysis.get('hit_rate', 0.0),

            # Returns
            'ytd_return': analysis.get('ytd_return', 0.0),
            'one_year_return': analysis.get('one_year_return', 0.0),
            'latest_price': analysis.get('latest_price', 0.0),

            # Composite score (fallback to percentile for new system)
            'composite_score': analysis.get('composite_percentile', analysis.get('composite_score', 0.0)),
            
            # Metadata
            'group_classification': analysis.get('group_classification', 'Unknown'),
            'quality_flag': analysis.get('quality_flag', 'TIER_3')
        })
    
    universe_df = pd.DataFrame(universe_data)
    universe_df = universe_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    
    universe_file = data_path / 'etf_universe.parquet'
    universe_df.to_parquet(universe_file, compression='snappy', index=False)
    saved_files['universe'] = str(universe_file)
    print(f"   Saved: {universe_file} ({len(universe_df)} ETFs, {universe_file.stat().st_size / 1024:.1f} KB)")
    
    # 2. Create Rankings DataFrames (by risk category)
    print("\n2. Creating risk category rankings...")
    
    for risk_type, label in [
        ('LOW', 'Low Risk'),
        ('MEDIUM', 'Medium Risk'),
        ('HIGH', 'High Risk')
    ]:
        category_data = rankings.get(risk_type, {})

        if not category_data:
            print(f"   No {label} ETFs found")
            continue

        ranking_data = []

        # Handle both old format (list of tuples) and new format (dict with 'rankings' key)
        if isinstance(category_data, dict) and 'rankings' in category_data:
            # New format: dict with 'rankings' key from PercentileRanker
            category_etfs = category_data.get('rankings', [])
            items_to_rank = [(idx + 1, item.get('ticker'), item.get('composite_percentile', 0.0))
                            for idx, item in enumerate(category_etfs)]
        elif isinstance(category_data, list):
            # Old format: list of (ticker, score) tuples or dicts
            if len(category_data) > 0 and isinstance(category_data[0], tuple):
                items_to_rank = [(rank, ticker, score) for rank, (ticker, score) in enumerate(category_data, 1)]
            else:
                items_to_rank = [(idx + 1, item.get('ticker'), item.get('composite_percentile', 0.0))
                                for idx, item in enumerate(category_data)]
        else:
            items_to_rank = []

        for rank, ticker, score in items_to_rank:
            analysis = analysis_results.get(ticker, {})
            etf_info = etf_db.etf_data.get(ticker, {})

            ranking_data.append({
                'rank': rank,
                'ticker': ticker,
                'name': etf_info.get('name', ticker),
                'subcategory': etf_info.get('subcategory', 'Unknown'),
                'composite_score': score,
                
                # Risk Component
                'risk_score': analysis.get('risk_score', 0.0),
                'cvar': analysis.get('cvar', 0.0),
                'ulcer_index': analysis.get('ulcer_index', 0.0),
                'beta': analysis.get('beta', 0.0),
                'information_ratio': analysis.get('information_ratio', 0.0),
                
                # Kalman Hull
                'kalman_trend': analysis.get('kalman_trend', 0),
                'kalman_signal_strength': analysis.get('kalman_signal_strength', 0.0),
                'kalman_divergence': analysis.get('kalman_divergence', 'none'),
                
                # Volume Intelligence
                'volume_spike_score': analysis.get('volume_spike_score', 0.0),
                'volume_correlation': analysis.get('volume_correlation', 0.0),
                'volume_ad_signal': analysis.get('volume_ad_signal', 'neutral'),
                
                # ML Ensemble
                'ml_forecast': analysis.get('ml_forecast', 0.0),
                'ml_confidence': analysis.get('ml_confidence', 0.5),
                'mae_score': analysis.get('mae_score', 0.0),
                'hit_rate': analysis.get('hit_rate', 0.0),
                
                # Returns
                'ytd_return': analysis.get('ytd_return', 0.0),
                'one_year_return': analysis.get('one_year_return', 0.0),
                
                # Other
                'volatility': analysis.get('volatility', 0.0),
                'best_benchmark': analysis.get('best_benchmark', 'N/A')
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        risk_type_lower = risk_type.lower() + '_risk'
        ranking_file = data_path / f'rankings_{risk_type_lower}.parquet'
        ranking_df.to_parquet(ranking_file, compression='snappy', index=False)
        saved_files[risk_type_lower] = str(ranking_file)
        print(f"   Saved: {ranking_file} ({len(ranking_df)} ETFs, {ranking_file.stat().st_size / 1024:.1f} KB)")
    
    # 3. Create metadata file
    print("\n3. Creating metadata...")
    metadata = {
        'analysis_date': datetime.now().isoformat(),
        'total_etfs': len(analysis_results),
        'processing_time': results.get('summary', {}).get('processing_time', 0),
        'system_version': '3.0_modified',
        'risk_breakdown': {
            'low': len(rankings.get('LOW', [])),
            'medium': len(rankings.get('MEDIUM', [])),
            'high': len(rankings.get('HIGH', []))
        },
        'components': {
            'risk_component': 'CVaR, Ulcer, Beta, IR (30/30/20/20)',
            'ml_ensemble': 'Raw forecast + confidence (NO bias correction)',
            'kalman_hull': 'Adaptive momentum indicator',
            'volume_intelligence': 'Spike, correlation, A/D'
        }
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_file = data_path / 'analysis_metadata.parquet'
    metadata_df.to_parquet(metadata_file, compression='snappy', index=False)
    saved_files['metadata'] = str(metadata_file)
    print(f"   Saved: {metadata_file} ({metadata_file.stat().st_size / 1024:.1f} KB)")
    
    # Summary
    total_size = sum(Path(f).stat().st_size for f in saved_files.values())
    print("\n" + "="*60)
    print(f"SUCCESSFULLY SAVED {len(saved_files)} FILES")
    print(f"   Total size: {total_size / 1024:.1f} KB")
    print(f"   Location: {data_path.absolute()}")
    print("="*60)
    
    return saved_files


def run_full_etf_analysis(save_results=True):
    """
    Run comprehensive ETF analysis on all tickers in the database.
    
    Args:
        save_results: Whether to save results to Parquet files
        
    Returns:
        dict: Complete analysis results containing:
            - analysis_results: Dict of per-ETF analysis
            - rankings: Rankings by risk category
            - summary: Overall summary stats
            - saved_files: Paths to saved Parquet files (if save_results=True)
    """
    warnings.filterwarnings('ignore')
    
    print("="*60)
    print("ETF ANALYSIS SYSTEM - MODIFIED VERSION")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize systems
    system = ETFAnalysisSystem()
    etf_db = ETFDatabase()
    all_tickers = list(etf_db.etf_data.keys())
    
    print(f"Analyzing {len(all_tickers)} ETFs...")
    print()
    
    # Run analysis
    results = system.run_full_analysis(all_tickers)
    
    # Print comprehensive analysis summary
    print_analysis_summary(results, etf_db)
    
    # Save results if requested
    if save_results:
        data_dir = setup_data_directory()
        saved_files = save_analysis_to_parquet(results, etf_db, data_dir)
        results['saved_files'] = saved_files
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total ETFs Analyzed: {len(results.get('analysis_results', {}))}")
    print(f"{'='*60}\n")
    
    return results


def main():
    """
    Main entry point for ETF analysis.
    Runs analysis and saves results to Parquet files for dashboard.
    Optionally runs backtesting validation.
    """
    # Run main analysis
    results = run_full_etf_analysis(save_results=True)
    
    # Ask user if they want to run backtesting
    print("\n" + "="*80)
    print("BACKTESTING VALIDATION")
    print("="*80)
    print("\nWould you like to run backtesting to validate the strategy?")
    print("\nOptions:")
    print("  1. Quick test (11 sample ETFs, ~1-2 min)")
    print("  2. Full universe (all ETFs with data, ~30-60 min)")
    print("  3. Skip backtesting")
    
    try:
        choice = input("\nYour choice (1/2/3): ").strip()
        
        if choice == '1':
            # Quick sample backtest
            print("\n" + "="*80)
            print("RUNNING QUICK BACKTEST (11 Sample ETFs)")
            print("="*80)
            
            # Import backtest function
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utilities.backtest_engine import run_backtest_on_universe
            
            # Sample tickers across risk categories
            sample_tickers = [
                'VAS.AX', 'VGS.AX', 'NDQ.AX',  # LOW risk
                'MNRS.AX', 'HETH.AX', 'FOOD.AX', 'ROBO.AX',  # MEDIUM risk
                'CRYP.AX', 'URNM.AX', 'GDX.AX', 'ATEC.AX'  # HIGH risk
            ]
            
            run_backtest_on_universe(tickers=sample_tickers)
            print("\nQuick backtest complete! View results in dashboard (Backtest Results page)")
        
        elif choice == '2':
            # Full universe backtest
            print("\n" + "="*80)
            print("RUNNING FULL UNIVERSE BACKTEST")
            print("="*80)
            print("\n This will take 30-60 minutes for all ETFs with historical data")
            confirm = input("Continue? (y/n): ").strip().lower()
            
            if confirm == 'y':
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from utilities.backtest_engine import run_backtest_on_universe
                
                # Backtest all available ETFs
                run_backtest_on_universe()
                print("\nFull universe backtest complete! View results in dashboard")
            else:
                print("\n⏭ Skipping backtest")
        else:
            print("\n⏭ Skipping backtest")
    
    except KeyboardInterrupt:
        print("\n\n⏭ Skipping backtest")
    except Exception as e:
        print(f"\n Backtest skipped due to error: {e}")

    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. View dashboard: python3 run_dashboard.py")
    print("  2. Open: http://127.0.0.1:8050")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    main()

