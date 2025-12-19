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
    portfolio = portfolio_result['portfolio']
    
    print(f"\n{'='*80}")
    print("SELECTED PORTFOLIO (3-4 ETFs)")
    print("="*80)
    
    print(f"\nSelected Portfolio ({len(portfolio)} ETFs):")
    for i, etf in enumerate(portfolio, 1):
        row = quality_scores[quality_scores['etf'] == etf].iloc[0]
        # Get risk classification
        risk_data = ranker.risk_classifier.classify_etfs([etf])
        # classify_etfs returns tuple (classifications_dict, summary_dict)
        classifications = risk_data[0] if isinstance(risk_data, tuple) else risk_data
        # classify_etfs returns dict with low/medium/high risk ETF lists
        risk_cat = 'unknown'
        if etf in classifications.get('low_risk_etfs', {}):
            risk_cat = 'low'
        elif etf in classifications.get('medium_risk_etfs', {}):
            risk_cat = 'medium'
        elif etf in classifications.get('high_risk_etfs', {}):
            risk_cat = 'high'
        
        print(f"\n{i}. {etf}")
        print(f"   Quality Score: {row['score']:.2f}")
        print(f"   Hit Rate: {row['hit_rate']*100:.1f}%")
        print(f"   Conviction: {row['conviction']:.2f}")
        print(f"   Stability: {row['stability']:.2f}")
        print(f"   Risk Category: {risk_cat}")
    
    # Portfolio metrics
    portfolio_data = quality_scores[quality_scores['etf'].isin(portfolio)]
    
    print(f"\nPortfolio Metrics:")
    print(f"  Average Hit Rate: {portfolio_data['hit_rate'].mean()*100:.1f}%")
    print(f"  Average Conviction: {portfolio_data['conviction'].mean():.2f}")
    print(f"  Average Stability: {portfolio_data['stability'].mean():.2f}")
    print(f"  Average Forecast: {portfolio_data['forecast'].mean():.2f}%")
    
    # Risk distribution
    risk_counts = {}
    for etf in portfolio:
        risk_data = ranker.risk_classifier.classify_etfs([etf])
        # classify_etfs returns tuple (classifications_dict, summary_dict)
        classifications = risk_data[0] if isinstance(risk_data, tuple) else risk_data
        risk_cat = 'unknown'
        if etf in classifications.get('low_risk_etfs', {}):
            risk_cat = 'low'
        elif etf in classifications.get('medium_risk_etfs', {}):
            risk_cat = 'medium'
        elif etf in classifications.get('high_risk_etfs', {}):
            risk_cat = 'high'
        risk_counts[risk_cat] = risk_counts.get(risk_cat, 0) + 1
    
    print(f"\nRisk Distribution:")
    for risk, count in risk_counts.items():
        print(f"  {risk}: {count} ETF(s)")
    
    # Check correlations
    if len(portfolio) > 1:
        print(f"\nPortfolio Correlations:")
        print("-"*40)
        
        correlation_data = {}
        for etf in portfolio:
            try:
                import yfinance as yf
                data = yf.download(etf, period='3mo', progress=False)
                if len(data) > 20:
                    correlation_data[etf] = data['Close'].squeeze()
            except:
                pass
        
        if len(correlation_data) > 1:
            df = pd.DataFrame(correlation_data)
            corr_matrix = df.corr()
            print(corr_matrix.round(2))
            
            max_corr = corr_matrix.values[corr_matrix.values != 1].max()
            print(f"\nMax Correlation: {max_corr:.2f}")
            
            if max_corr > 0.85:
                print("⚠️  Warning: High correlation between holdings")
            else:
                print("✅ Good diversification")
    
    # Save final portfolio
    portfolio_file = ranker.cache_dir / 'selected_portfolio.pkl'
    with open(portfolio_file, 'wb') as f:
        pickle.dump({
            'portfolio': portfolio,
            'quality_scores': quality_scores,
            'timestamp': datetime.now(),
            'metrics': {
                'avg_hit_rate': portfolio_data['hit_rate'].mean(),
                'avg_conviction': portfolio_data['conviction'].mean(),
                'avg_stability': portfolio_data['stability'].mean(),
                'avg_forecast': portfolio_data['forecast'].mean()
            }
        }, f)
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print("="*80)
    print(f"""
1. Portfolio saved to: {portfolio_file}
2. Run weekly to check for rebalance signals
3. Hold positions for minimum 45 days
4. Only rebalance if:
   - ETF drops >2 ranks in quality score
   - New ETF is >3% better in score
5. Consider backtesting this strategy vs buy-and-hold

To re-run with fresh data:
   rm {cache_file}
   python3 run_quality_universe.py
""")
    
    return portfolio, quality_scores

if __name__ == "__main__":
    portfolio, scores = run_full_universe()
