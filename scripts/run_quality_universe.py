#!/usr/bin/env python3
"""
Run QualityRanker on the full universe of ETFs
Optimized for batch processing with caching and progress saving
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from analyzers.quality_ranker import QualityRanker
from data_manager.data_manager import ETFDataManager

def run_full_universe():
    """Run QualityRanker on all ETFs in the universe"""
    
    print("="*80)
    print("QUALITY RANKER - FULL UNIVERSE ANALYSIS")
    print("="*80)
    
    # Initialize
    ranker = QualityRanker()
    data_manager = ETFDataManager()
    
    # Get all ETF tickers
    # Load from etf_database.py directly
    import sys
    sys.path.append(str(Path(__file__).parent / 'data_manager'))
    from etf_database import ETFDatabase
    db = ETFDatabase()
    all_etfs = list(db.etf_data.keys())
    
    print(f"\nProcessing {len(all_etfs)} ETFs...")
    
    # Check for cached results
    cache_file = ranker.cache_dir / 'full_universe_scores.pkl'
    if cache_file.exists():
        print("\nLoading cached results...")
        with open(cache_file, 'rb') as f:
            results = pickle.load(f)
        quality_scores = results['quality_scores']
        print(f"Loaded {len(quality_scores)} cached scores")
    else:
        # Process in batches
        batch_size = 50
        all_scores = []
        
        for i in range(0, len(all_etfs), batch_size):
            batch = all_etfs[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(all_etfs)-1)//batch_size + 1}")
            print(f"ETFs {i+1}-{min(i+batch_size, len(all_etfs))}")
            
            start_time = time.time()
            
            # Calculate quality scores for batch
            batch_scores = ranker.calculate_quality_scores(batch)
            
            if len(batch_scores) > 0:
                all_scores.append(batch_scores)
                print(f"  ✓ {len(batch_scores)} ETFs scored in {time.time()-start_time:.1f}s")
                
                # Save progress after each batch
                progress_file = ranker.cache_dir / 'progress_batch.pkl'
                with open(progress_file, 'wb') as f:
                    pickle.dump(all_scores, f)
            else:
                print(f"  ✗ No ETFs met minimum criteria")
        
        # Combine all batches
        if all_scores:
            quality_scores = pd.concat(all_scores, ignore_index=True)
            quality_scores = quality_scores.sort_values('score', ascending=False).reset_index(drop=True)
            quality_scores['rank'] = range(1, len(quality_scores) + 1)
            
            # Cache final results
            with open(cache_file, 'wb') as f:
                pickle.dump({'quality_scores': quality_scores, 'timestamp': datetime.now()}, f)
        else:
            print("\n❌ No ETFs met minimum criteria")
            return
    
    # Display top results
    print(f"\n{'='*80}")
    print("TOP 20 ETFs BY QUALITY SCORE")
    print("="*80)
    
    print(f"\n{'Rank':<4} {'ETF':<8} {'Score':<7} {'Hit Rate':<9} {'Conviction':<10} {'Stability':<9} {'Forecast':<8} {'Risk'}")
    print("-"*80)
    
    for i, row in quality_scores.head(20).iterrows():
        # Get risk classification
        risk_data = ranker.risk_classifier.classify_etfs([row['etf']])
        # classify_etfs returns tuple (classifications_dict, summary_dict)
        classifications = risk_data[0] if isinstance(risk_data, tuple) else risk_data
        # classify_etfs returns dict with low/medium/high risk ETF lists
        risk_cat = 'unknown'
        if row['etf'] in classifications.get('low_risk_etfs', {}):
            risk_cat = 'low'
        elif row['etf'] in classifications.get('medium_risk_etfs', {}):
            risk_cat = 'medium'
        elif row['etf'] in classifications.get('high_risk_etfs', {}):
            risk_cat = 'high'
        
        print(f"{row['rank']:<4} {row['etf']:<8} {row['score']:<7.2f} "
              f"{row['hit_rate']*100:<9.1f}% {row['conviction']:<10.2f} "
              f"{row['stability']:<9.2f} {row['forecast']:<8.2f}% {risk_cat}")
    
    # Select portfolio
    portfolio_result = ranker.select_portfolio(quality_scores)
    top_10_etfs = portfolio_result['top_10_scores']
    
    print(f"\n{'='*80}")
    print("TOP 10 ETFs FOR MAX GROWTH STRATEGY")
    print("="*80)
    print(f"\nSelect your 3-4 ETFs from these top 10 based on your personal conviction:")
    print(f"\n{'Rank':<4} {'ETF':<8} {'Score':<7} {'Hit Rate':<9} {'Conviction':<10} {'Stability':<9} {'Forecast':<8} {'Risk'}")
    print("-"*80)
    
    for i, etf_data in enumerate(top_10_etfs, 1):
        # Get risk classification
        risk_data = ranker.risk_classifier.classify_etfs([etf_data['etf']])
        # classify_etfs returns tuple (classifications_dict, summary_dict)
        classifications = risk_data[0] if isinstance(risk_data, tuple) else risk_data
        # classify_etfs returns dict with low/medium/high risk ETF lists
        risk_cat = 'unknown'
        if etf_data['etf'] in classifications.get('low_risk_etfs', {}):
            risk_cat = 'low'
        elif etf_data['etf'] in classifications.get('medium_risk_etfs', {}):
            risk_cat = 'medium'
        elif etf_data['etf'] in classifications.get('high_risk_etfs', {}):
            risk_cat = 'high'
        
        print(f"{i:<4} {etf_data['etf']:<8} {etf_data['score']:<7.2f} "
              f"{etf_data['hit_rate']*100:<9.1f}% {etf_data['conviction']:<10.2f} "
              f"{etf_data['stability']:<9.2f} {etf_data['forecast']:<8.2f}% {risk_cat}")
    
    # Save top 10 to file
    top_10_df = pd.DataFrame(top_10_etfs)
    top_10_file = ranker.cache_dir / 'top_10_etfs.pkl'
    with open(top_10_file, 'wb') as f:
        pickle.dump(top_10_etfs, f)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"\n✅ Top 10 ETFs saved to: {top_10_file}")
    print(f"\nStrategy: Max Growth - No diversification constraints")
    print(f"Focus: Hit rate and conviction for maximum returns")
    print(f"\nRecommendation: Choose 3-4 ETFs from the list above based on:")
    print(f"  - Your conviction in the sector/theme")
    print(f"  - Higher hit rates (>60%)")
    print(f"  - Strong conviction scores")
    print(f"\nNote: All 10 ETFs have passed the minimum hit rate threshold of 50%")
    
    return portfolio_result['portfolio'], quality_scores

if __name__ == "__main__":
    portfolio, scores = run_full_universe()
