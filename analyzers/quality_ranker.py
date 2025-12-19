#!/usr/bin/env python3
"""
QualityRanker: Selects 3-4 ETFs based on quality metrics
Focus on hit rate, stability, conviction, and diversification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from data_manager.data_manager import ETFDataManager
from analyzers.ml_ensemble_production import MLEnsembleProduction
from analyzers.etf_risk_classifier import ETFRiskClassifier
from analyzers.percentile_ranker import PercentileRanker

class QualityRanker:
    """Ranks ETFs by quality metrics for 3-4 ETF portfolios"""
    
    def __init__(self, cache_dir='data/quality_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.data_manager = ETFDataManager()
        self.ml_ensemble = MLEnsembleProduction()
        self.risk_classifier = ETFRiskClassifier()
        
        # Quality metric weights
        self.weights = {
            'conviction': 0.40,    # forecast × confidence
            'hit_rate': 0.35,      # quarterly outperformance rate
            'stability': 0.15,     # signal stability
            'diversification': 0.10 # anti-correlation bonus
        }
        
        # Minimum thresholds
        self.min_hit_rate = 0.50   # Temporarily lowered to see more results
        self.min_hold_days = 45    # Minimum days between rebalances
        self.min_score_drop = 2    # Positions must drop >2 ranks to replace
        self.min_improvement = 0.03 # New pick must be 3% better
        
        # Sector keywords for diversification
        self.commodity_keywords = ['GOLD', 'SILV', 'COPP', 'PLAT', 'PALL', 'MIN', 'METAL', 'COMMOD', 'WIRE', 'ETPM']
        self.tech_keywords = ['TECH', 'NDQ', 'HNDQ', 'ATEC', 'ESPO', 'GAME', 'FANG']
        self.finance_keywords = ['BANK', 'FIN', 'CRED', 'BOND', 'AAA', 'VAP']
        
    def calculate_quality_scores(self, etf_list: List[str]) -> pd.DataFrame:
        """Calculate quality scores for all ETFs"""
        
        print("Calculating quality scores...")
        print(f"  Processing {len(etf_list)} ETFs")
        
        # Get historical data for hit rate calculation
        print("  Fetching 3-year historical data...")
        historical_data = self._fetch_historical_data(etf_list)
        
        # Calculate hit rates
        print("  Calculating hit rates...")
        hit_rates = self._calculate_hit_rates(historical_data)
        
        # Get current ML forecasts
        print("  Getting ML forecasts...")
        forecasts = self._get_current_forecasts(etf_list)
        
        # Calculate stability scores
        print("  Calculating stability...")
        stability_scores = self._calculate_stability(etf_list)
        
        # Combine into quality scores
        quality_scores = []
        
        for etf in etf_list:
            # Get metrics
            hit_rate = hit_rates.get(etf, 0.0)
            forecast = forecasts.get(etf, {}).get('forecast_return', 0.0)
            confidence = forecasts.get(etf, {}).get('confidence_score', 0.0)
            stability = stability_scores.get(etf, 0.0)
            
            # Calculate conviction (forecast × confidence)
            conviction = forecast * confidence
            
            # Skip ETFs with low hit rate
            if hit_rate < self.min_hit_rate:
                continue
            
            # Calculate weighted score
            score = (
                conviction * self.weights['conviction'] +
                hit_rate * self.weights['hit_rate'] +
                stability * self.weights['stability']
            )
            
            quality_scores.append({
                'etf': etf,
                'score': score,
                'hit_rate': hit_rate,
                'conviction': conviction,
                'stability': stability,
                'forecast': forecast,
                'confidence': confidence
            })
        
        # Convert to DataFrame and rank
        df = pd.DataFrame(quality_scores)
        if len(df) > 0:
            df = df.sort_values('score', ascending=False).reset_index(drop=True)
            df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def select_portfolio(self, quality_scores: pd.DataFrame) -> Dict:
        """Select top 10 ETFs based purely on quality score (no diversification constraints)"""
        
        # Filter by minimum hit rate
        qualified = quality_scores[quality_scores['hit_rate'] >= self.min_hit_rate]
        
        if len(qualified) == 0:
            return {'portfolio': [], 'reason': 'No ETFs meet minimum hit rate'}
        
        # Sort by score and return top 10
        top_etfs = qualified.sort_values('score', ascending=False).head(10)
        
        print(f"\nTop 10 ETFs by Quality Score:")
        print("-" * 80)
        for i, (_, row) in enumerate(top_etfs.iterrows(), 1):
            print(f"{i:2d}. {row['etf']:8s} - Score: {row['score']:6.2f} | "
                  f"Hit Rate: {row['hit_rate']*100:5.1f}% | "
                  f"Conviction: {row['conviction']:5.2f} | "
                  f"Stability: {row['stability']:5.2f} | "
                  f"Forecast: {row['forecast']:5.2f}%")
        
        return {
            'portfolio': top_etfs['etf'].tolist(),
            'top_10_scores': top_etfs.to_dict('records'),
            'reason': 'Top 10 ETFs by quality score (no diversification constraints)'
        }
    
    def _fetch_historical_data(self, etf_list: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch max historical data for hit rate calculation using batch downloads"""
        
        historical_data = {}
        cache_file = self.cache_dir / 'historical_max.pkl'
        
        # Try to load from cache
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                historical_data = pickle.load(f)
        
        # Fetch missing data in batches
        missing_etfs = [etf for etf in etf_list if etf not in historical_data]
        
        if missing_etfs:
            print(f"    Downloading max history for {len(missing_etfs)} ETFs...")
            
            # Process in batches of 50 (yfinance limit)
            batch_size = 50
            for i in range(0, len(missing_etfs), batch_size):
                batch = missing_etfs[i:i+batch_size]
                print(f"      Batch {i//batch_size + 1}/{(len(missing_etfs)-1)//batch_size + 1}: {len(batch)} ETFs")
                
                try:
                    # Batch download
                    import yfinance as yf
                    data = yf.download(batch, period='max', progress=False, group_by='ticker')
                    
                    # Process each ticker in the batch
                    for ticker in batch:
                        if ticker in data.columns:
                            # Extract data for this ticker
                            ticker_data = data[ticker].dropna()
                            if len(ticker_data) > 500:  # Only keep if sufficient data
                                historical_data[ticker] = ticker_data
                            else:
                                print(f"        {ticker}: Insufficient data ({len(ticker_data)} days)")
                        else:
                            print(f"        {ticker}: No data found")
                            
                except Exception as e:
                    print(f"      Batch download failed: {e}")
                    # Fall back to individual downloads for this batch
                    for ticker in batch:
                        try:
                            individual_data = yf.download(ticker, period='max', progress=False)
                            if len(individual_data) > 500:
                                historical_data[ticker] = individual_data
                        except:
                            print(f"        Failed to download {ticker}")
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(historical_data, f)
        
        return historical_data
    
    def _calculate_hit_rates(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate quarterly hit rate vs sector median"""
        
        hit_rates = {}
        
        # Group ETFs by sector for comparison
        etf_sectors = self._get_etf_sectors(list(historical_data.keys()))
        
        for etf, data in historical_data.items():
            if len(data) < 500:
                continue
            
            # Calculate quarterly returns
            quarterly_returns = self._get_quarterly_returns(data)
            
            # Get sector median for comparison
            sector = etf_sectors.get(etf, 'unknown')
            sector_returns = self._get_sector_median_returns(sector, historical_data, etf_sectors)
            
            # Calculate hit rate
            wins = 0
            total = 0
            
            for quarter, etf_return in quarterly_returns.items():
                if quarter in sector_returns:
                    sector_median = sector_returns[quarter]
                    # Debug print
                    # print(f"Debug: etf_return type: {type(etf_return)}, value: {etf_return}")
                    # print(f"Debug: sector_median type: {type(sector_median)}, value: {sector_median}")
                    
                    # Ensure we're comparing scalars
                    if hasattr(sector_median, 'iloc'):
                        sector_median = sector_median.iloc[0]
                    if hasattr(sector_median, 'item'):
                        sector_median = sector_median.item()
                    
                    # Convert to float to be safe
                    sector_median = float(sector_median)
                    etf_return = float(etf_return)
                    
                    if etf_return > sector_median:
                        wins += 1
                    total += 1
            
            hit_rate = wins / total if total > 0 else 0.0
            hit_rates[etf] = hit_rate
        
        return hit_rates
    
    def _get_quarterly_returns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate quarterly returns from price data"""
        
        quarterly_returns = {}
        
        # Resample to quarter end prices
        quarterly_prices = data['Close'].resample('Q').last()
        
        # Calculate returns
        for i in range(1, len(quarterly_prices)):
            prev_price = quarterly_prices.iloc[i-1]
            curr_price = quarterly_prices.iloc[i]
            return_pct = (curr_price / prev_price - 1) * 100
            
            quarter_key = f"{quarterly_prices.index[i].year}-Q{(quarterly_prices.index[i].month-1)//3+1}"
            quarterly_returns[quarter_key] = return_pct
        
        return quarterly_returns
    
    def _get_etf_sectors(self, etf_list: List[str]) -> Dict[str, str]:
        """Get sector classification for ETFs"""
        
        sectors = {}
        
        # Simple sector mapping based on ETF patterns
        for etf in etf_list:
            if 'GOLD' in etf or 'MNRS' in etf:
                sectors[etf] = 'gold'
            elif 'TECH' in etf or 'HNDQ' in etf or 'ATEC' in etf:
                sectors[etf] = 'technology'
            elif 'BOND' in etf or 'VAP' in etf or 'AAA' in etf:
                sectors[etf] = 'fixed_income'
            elif 'CRYP' in etf or 'BTC' in etf:
                sectors[etf] = 'crypto'
            elif 'GEAR' in etf or 'BBOZ' in etf:
                sectors[etf] = 'leveraged'
            else:
                sectors[etf] = 'diversified'
        
        return sectors
    
    def _get_sector_median_returns(self, sector: str, 
                                 historical_data: Dict[str, pd.DataFrame],
                                 etf_sectors: Dict[str, str]) -> Dict[str, float]:
        """Get median quarterly returns for a sector"""
        
        sector_returns = []
        
        # Get all ETFs in this sector
        sector_etfs = [etf for etf, s in etf_sectors.items() if s == sector]
        
        # Calculate quarterly returns for each
        for etf in sector_etfs:
            if etf in historical_data:
                returns = self._get_quarterly_returns(historical_data[etf])
                sector_returns.append(returns)
        
        # Calculate median across sector
        if sector_returns:
            all_quarters = set()
            for returns in sector_returns:
                all_quarters.update(returns.keys())
            
            median_returns = {}
            for quarter in all_quarters:
                quarter_returns = []
                for r in sector_returns:
                    if quarter in r:
                        val = r[quarter]
                        # Convert to float, skip if not numeric
                        try:
                            float_val = float(val)
                            if not np.isnan(float_val):
                                quarter_returns.append(float_val)
                        except (ValueError, TypeError):
                            continue
                
                if quarter_returns:
                    median_val = np.median(quarter_returns)
                    median_returns[quarter] = float(median_val)
            
            return median_returns
        
        return {}
    
    def _get_current_forecasts(self, etf_list: List[str]) -> Dict[str, Dict]:
        """Get current ML forecasts for all ETFs using batch downloads"""
        
        forecasts = {}
        
        # Process in batches for efficiency
        batch_size = 50
        
        for i in range(0, len(etf_list), batch_size):
            batch = etf_list[i:i+batch_size]
            
            # Batch download recent data
            try:
                import yfinance as yf
                data = yf.download(batch, period='2y', progress=False, group_by='ticker')
                
                for etf in batch:
                    try:
                        if etf in data.columns:
                            etf_data = data[etf].dropna()
                            if len(etf_data) > 100:
                                result = self.ml_ensemble.forecast_etf(etf_data)
                                forecasts[etf] = {
                                    'forecast_return': result.get('forecast_return', 0.0),
                                    'confidence_score': result.get('confidence_score', 0.0)
                                }
                            else:
                                forecasts[etf] = {'forecast_return': 0.0, 'confidence_score': 0.0}
                        else:
                            forecasts[etf] = {'forecast_return': 0.0, 'confidence_score': 0.0}
                    except Exception as e:
                        # print(f"    Failed to get forecast for {etf}: {e}")
                        forecasts[etf] = {'forecast_return': 0.0, 'confidence_score': 0.0}
                        
            except Exception as e:
                print(f"    Batch forecast failed: {e}")
                # Fall back to individual downloads
                for etf in batch:
                    try:
                        individual_data = yf.download(etf, period='2y', progress=False)
                        if len(individual_data) > 100:
                            result = self.ml_ensemble.forecast_etf(individual_data)
                            forecasts[etf] = {
                                'forecast_return': result.get('forecast_return', 0.0),
                                'confidence_score': result.get('confidence_score', 0.0)
                            }
                    except:
                        forecasts[etf] = {'forecast_return': 0.0, 'confidence_score': 0.0}
        
        return forecasts
    
    def _calculate_stability(self, etf_list: List[str]) -> Dict[str, float]:
        """Calculate stability score based on 4-week feature volatility using batch downloads"""
        
        stability_scores = {}
        
        # Process in batches for efficiency
        batch_size = 50
        
        for i in range(0, len(etf_list), batch_size):
            batch = etf_list[i:i+batch_size]
            
            # Batch download 1 month data
            try:
                import yfinance as yf
                data = yf.download(batch, period='1mo', progress=False, group_by='ticker')
                
                for etf in batch:
                    try:
                        if etf in data.columns:
                            etf_data = data[etf].dropna()
                            
                            if len(etf_data) < 20:
                                stability_scores[etf] = 0.0
                                continue
                            
                            # Extract Close price as Series
                            if 'Close' in etf_data.columns:
                                close_prices = etf_data['Close'].squeeze()
                            else:
                                close_prices = etf_data.squeeze()
                            
                            # Ensure we have a Series
                            if hasattr(close_prices, 'columns'):
                                close_prices = close_prices.iloc[:, 0]
                            
                            # Calculate stability metrics
                            returns = close_prices.pct_change().dropna()
                            
                            # 1. Return stability
                            return_volatility = float(returns.std() * np.sqrt(252))
                            return_stability = max(0, 1 - return_volatility / 0.5)
                            
                            # 2. Trend stability
                            sma_short = close_prices.rolling(5).mean()
                            sma_long = close_prices.rolling(20).mean()
                            trend_signals = (sma_short > sma_long).astype(int)
                            trend_changes = float(trend_signals.diff().abs().sum())
                            trend_stability = max(0, 1 - trend_changes / float(len(trend_signals.dropna())))
                            
                            # 3. Price stability
                            price_changes = close_prices.diff().abs()
                            avg_price_change = float(price_changes.mean())
                            price_stability = max(0, 1 - avg_price_change / float(close_prices.iloc[0]) * 0.05)
                            
                            # Combine stability metrics
                            overall_stability = float(
                                return_stability * 0.4 +
                                trend_stability * 0.4 +
                                price_stability * 0.2
                            )
                            
                            stability_scores[etf] = overall_stability
                        else:
                            stability_scores[etf] = 0.0
                            
                    except Exception as e:
                        print(f"    Failed to calculate stability for {etf}: {e}")
                        stability_scores[etf] = 0.0
                        
            except Exception as e:
                print(f"    Batch stability failed: {e}")
                # Set all to 0.0 for this batch
                for etf in batch:
                    stability_scores[etf] = 0.0
        
        return stability_scores
    
    def _check_correlations(self, etf: str, portfolio: List[str]) -> Dict[str, float]:
        """Check correlation of ETF with existing portfolio"""
        
        correlations = []
        
        try:
            # Get recent price data for correlation calculation
            etf_data = yf.download(etf, period='3mo', progress=False)
            
            for existing_etf in portfolio:
                existing_data = yf.download(existing_etf, period='3mo', progress=False)
                
                # Align dates and calculate correlation
                aligned_etf = etf_data['Close'].reindex(existing_data.index).dropna()
                aligned_existing = existing_data['Close'].reindex(etf_data.index).dropna()
                
                if len(aligned_etf) > 20:
                    corr = aligned_etf.corr(aligned_existing)
                    correlations.append(corr)
            
            return {
                'max_correlation': max(correlations) if correlations else 0.0,
                'avg_correlation': np.mean(correlations) if correlations else 0.0
            }
            
        except:
            return {'max_correlation': 0.0, 'avg_correlation': 0.0}
    
    def _should_rebalance(self, current_portfolio: List[str], 
                         new_portfolio: List[str], 
                         quality_scores: pd.DataFrame) -> bool:
        """Determine if portfolio should be rebalanced"""
        
        # Check if any current ETF dropped significantly in rank
        current_ranks = {}
        for etf in current_portfolio:
            rank_row = quality_scores[quality_scores['etf'] == etf]
            if len(rank_row) > 0:
                current_ranks[etf] = rank_row.iloc[0]['rank']
        
        # Check if any ETF dropped >2 positions
        for etf, rank in current_ranks.items():
            if rank > 3:  # Dropped out of top 3
                return True
            if etf not in new_portfolio and rank > 3:
                return True
        
        # Check if new ETFs are significantly better
        if len(new_portfolio) >= 3:
            # Get scores
            current_scores = []
            new_scores = []
            
            for etf in current_portfolio[:3]:
                row = quality_scores[quality_scores['etf'] == etf]
                if len(row) > 0:
                    current_scores.append(row.iloc[0]['score'])
            
            for etf in new_portfolio[:3]:
                row = quality_scores[quality_scores['etf'] == etf]
                if len(row) > 0:
                    new_scores.append(row.iloc[0]['score'])
            
            # Compare average scores
            if current_scores and new_scores:
                avg_current = np.mean(current_scores)
                avg_new = np.mean(new_scores)
                
                improvement = (avg_new - avg_current) / avg_current
                if improvement > self.min_improvement:
                    return True
        
        return False
