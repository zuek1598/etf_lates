#!/usr/bin/env python3
"""
Single Ticker Analyzer
Analyzes individual stocks or crypto coins using the same validated system components
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from analyzers.ml_ensemble_production import MLEnsembleProduction
from analyzers.risk_component import RiskComponent
from analyzers.kalman_hull import calculate_adaptive_kalman_hull
from utilities.shared_utils import extract_column, transform_to_returns
from data_manager.external_data import fetch_external_data


class SingleTickerAnalyzer:
    """
    Analyzes a single stock or crypto coin using validated system components
    """
    
    def __init__(self):
        """Initialize analyzer components"""
        self.ml_ensemble = MLEnsembleProduction()
        self.risk_component = RiskComponent()
        self.risk_component.validated_only = True  # Use validated factors only
        self.external_data = None
        self.vix_data = None
        
        # Load external market data
        self._load_market_data()
    
    def _load_market_data(self):
        """Load external market data for analysis"""
        try:
            external = fetch_external_data()
            self.external_data = external
            
            # Extract VIX data if available
            if 'vix' in external:
                vix_df = external['vix']
                if vix_df is not None and not vix_df.empty:
                    self.vix_data = extract_column(vix_df, 'Close')
        except Exception as e:
            print(f"⚠️  Warning: Could not load external market data: {e}")
            self.external_data = {}
            self.vix_data = None
    
    def download_ticker_data(self, ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """
        Download historical data for a ticker
        
        Args:
            ticker: Stock or crypto ticker symbol (e.g., 'AAPL', 'BTC-USD')
            period: Data period ('1y', '2y', '5y', etc.)
            
        Returns:
            DataFrame with OHLCV data or None if download fails
        """
        try:
            print(f"📥 Downloading data for {ticker}...")
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=period)
            
            if data.empty:
                print(f"❌ No data found for {ticker}")
                return None
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                print(f"⚠️  Missing columns: {missing_cols}")
                # Fill missing columns with Close price
                for col in missing_cols:
                    if col == 'Volume':
                        data[col] = 0
                    else:
                        data[col] = data['Close']
            
            print(f"✅ Downloaded {len(data)} days of data for {ticker}")
            return data
            
        except Exception as e:
            print(f"❌ Error downloading data for {ticker}: {e}")
            return None
    
    def analyze(self, ticker: str, period: str = "2y") -> Optional[Dict]:
        """
        Run complete analysis on a single ticker
        
        Args:
            ticker: Stock or crypto ticker symbol
            period: Historical data period
            
        Returns:
            Dictionary with analysis results or None if analysis fails
        """
        print(f"\n{'='*70}")
        print(f"ANALYZING: {ticker}")
        print(f"{'='*70}\n")
        
        # Download data
        data = self.download_ticker_data(ticker, period)
        if data is None:
            return None
        
        # Extract price and volume
        prices = extract_column(data, 'Close')
        volume = extract_column(data, 'Volume') if 'Volume' in data.columns else None
        
        if len(prices) < 30:
            print(f"❌ Insufficient data: {len(prices)} days (minimum 30 required)")
            return None
        
        # Get ticker info
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            ticker_name = info.get('longName', info.get('shortName', ticker))
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
        except:
            ticker_name = ticker
            sector = 'Unknown'
            industry = 'Unknown'
        
        print(f"📊 {ticker_name}")
        print(f"   Sector: {sector} | Industry: {industry}\n")
        
        # Determine risk category (for Kalman Hull parameters)
        # Use volatility-based classification
        returns = transform_to_returns(prices)
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        if volatility < 0.15:
            risk_category = 'LOW'
        elif volatility < 0.30:
            risk_category = 'MEDIUM'
        else:
            risk_category = 'HIGH'
        
        print(f"📈 Risk Category: {risk_category} (Volatility: {volatility*100:.1f}%)\n")
        
        # Run analysis components
        results = {
            'ticker': ticker,
            'name': ticker_name,
            'sector': sector,
            'industry': industry,
            'risk_category': risk_category,
            'volatility': volatility,
            'latest_price': float(prices.iloc[-1]),
            'data_points': len(prices),
        }
        
        # 1. ML Ensemble Analysis
        print("🤖 Running ML Ensemble Analysis...")
        print(f"   ℹ️  Note: Model retrains on latest data - forecast updates with new market data")
        try:
            ml_result = self.ml_ensemble.forecast_etf(data)
            
            # Calculate hit rate if we have enough data
            hit_rate = 0.0
            if len(prices) >= 312:  # Need enough data for validation
                try:
                    validation = self.ml_ensemble.walk_forward_validate(prices)
                    hit_rate = validation.get('hit_rate', 0.0)
                except:
                    pass
            
            results['ml_forecast'] = -ml_result.get('forecast_return', 0.0)  # Direction corrected
            results['ml_confidence'] = ml_result.get('confidence_score', ml_result.get('confidence', 0.0))
            results['hit_rate'] = ml_result.get('hit_rate', hit_rate)
            print(f"   ✅ ML Forecast: {results['ml_forecast']:+.2f}%")
            print(f"   ✅ Hit Rate: {results['hit_rate']:.2%}")
            print(f"   ✅ Confidence: {results['ml_confidence']:.2%}")
        except Exception as e:
            print(f"   ⚠️  ML Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            results['ml_forecast'] = 0.0
            results['ml_confidence'] = 0.0
            results['hit_rate'] = 0.0
        
        # 2. Risk Component Analysis
        print("\n📉 Running Risk Analysis...")
        try:
            etf_info = {'region': 'UNKNOWN', 'subcategory': 'Stock/Crypto', 'type': 'equity'}
            risk_result = self.risk_component.analyze_etf(
                data, 
                etf_info, 
                vix_data=self.vix_data,
                benchmark_data=None,
                beta=np.nan
            )
            results['cvar'] = risk_result.get('cvar', np.nan)
            results['beta'] = risk_result.get('beta', np.nan)
            results['risk_score'] = risk_result.get('risk_score', np.nan)
            print(f"   ✅ CVaR: {results['cvar']:.4f}")
            if not np.isnan(results['beta']):
                print(f"   ✅ Beta: {results['beta']:.2f}")
            print(f"   ✅ Risk Score: {results['risk_score']:.2f}")
        except Exception as e:
            print(f"   ⚠️  Risk Analysis failed: {e}")
            results['cvar'] = np.nan
            results['beta'] = np.nan
            results['risk_score'] = np.nan
        
        # 3. Kalman Hull Analysis
        print("\n📊 Running Kalman Hull Momentum Analysis...")
        try:
            kalman_result = calculate_adaptive_kalman_hull(
                prices, 
                volume=volume,
                risk_category=risk_category,
                ohlc_data=data
            )
            signal_strength = kalman_result.get('signal_strength', 0.0)
            results['kalman_signal_strength'] = signal_strength
            
            # Calculate trend from signal strength (signal > 0.5 = bullish, else bearish)
            # Also check price position relative to recent average
            recent_avg = prices.tail(20).mean()
            current_price = prices.iloc[-1]
            price_above_avg = current_price > recent_avg
            
            # Trend: 1 = bullish, 0 = bearish
            if signal_strength > 0.5 and price_above_avg:
                results['kalman_trend'] = 1
            elif signal_strength < 0.5 and not price_above_avg:
                results['kalman_trend'] = 0
            else:
                # Mixed signals - use signal strength as tiebreaker
                results['kalman_trend'] = 1 if signal_strength > 0.5 else 0
            
            # Calculate efficiency ratio separately (Kaufman Efficiency Ratio)
            try:
                from analyzers.kalman_hull import _calculate_efficiency_ratio
                efficiency_raw = _calculate_efficiency_ratio(prices)
            except:
                # Fallback calculation
                if len(prices) >= 11:
                    price_change = abs(prices.iloc[-1] - prices.iloc[-11])
                    volatility = prices.diff().abs().tail(10).sum()
                    efficiency_raw = price_change / volatility if volatility > 0 else 0.0
                    efficiency_raw = min(1.0, max(0.0, efficiency_raw))
                else:
                    efficiency_raw = 0.0
            
            results['kalman_efficiency'] = efficiency_raw
            
            print(f"   ✅ Trend: {'🟢 BULLISH' if results['kalman_trend'] == 1 else '🔴 BEARISH'}")
            print(f"   ✅ Signal Strength: {results['kalman_signal_strength']:.3f}")
            # Efficiency ratio is already 0-1, format as percentage
            if efficiency_raw > 0.0001:  # Only show if meaningful
                print(f"   ✅ Efficiency Ratio: {efficiency_raw*100:.2f}%")
            else:
                print(f"   ✅ Efficiency Ratio: {efficiency_raw*100:.4f}% (very low)")
        except Exception as e:
            print(f"   ⚠️  Kalman Hull Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            results['kalman_trend'] = 0
            results['kalman_signal_strength'] = 0.0
            results['kalman_efficiency'] = 0.0
        
        # 4. Calculate Returns
        print("\n📈 Calculating Returns...")
        try:
            # Normalize timezone for comparison - convert to timezone-naive
            prices_copy = prices.copy()
            if hasattr(prices_copy.index, 'tz') and prices_copy.index.tz is not None:
                prices_copy.index = prices_copy.index.tz_localize(None)
            
            # YTD Return
            current_year_start = pd.Timestamp(datetime.now().replace(month=1, day=1))
            ytd_mask = prices_copy.index >= current_year_start
            ytd_prices = prices_copy[ytd_mask]
            if len(ytd_prices) > 1:
                ytd_return = (ytd_prices.iloc[-1] / ytd_prices.iloc[0] - 1) * 100
            else:
                ytd_return = 0.0
            
            # 1-Year Return
            one_year_ago = pd.Timestamp(datetime.now() - timedelta(days=365))
            one_year_mask = prices_copy.index >= one_year_ago
            one_year_prices = prices_copy[one_year_mask]
            if len(one_year_prices) > 1:
                one_year_return = (one_year_prices.iloc[-1] / one_year_prices.iloc[0] - 1) * 100
            else:
                one_year_return = 0.0
            
            results['ytd_return'] = ytd_return
            results['one_year_return'] = one_year_return
            print(f"   ✅ YTD Return: {ytd_return:+.2f}%")
            print(f"   ✅ 1-Year Return: {one_year_return:+.2f}%")
        except Exception as e:
            print(f"   ⚠️  Return calculation failed: {e}")
            import traceback
            traceback.print_exc()
            results['ytd_return'] = 0.0
            results['one_year_return'] = 0.0
        
        # 5. Calculate Composite Score
        print("\n🎯 Calculating Composite Score...")
        try:
            composite_score = self._calculate_composite_score(results)
            results['composite_score'] = composite_score
            print(f"   ✅ Composite Score: {composite_score:.1f}/100")
        except Exception as e:
            print(f"   ⚠️  Score calculation failed: {e}")
            results['composite_score'] = 0.0
        
        return results
    
    def _calculate_composite_score(self, results: Dict) -> float:
        """
        Calculate composite score from validated factors
        
        Uses the same scoring methodology as ETF analysis system
        """
        score = 0.0
        
        # ML Forecast (40% weight) - normalized to 0-40
        # For negative forecasts, we still give partial credit based on magnitude
        ml_forecast = results.get('ml_forecast', 0.0)
        if ml_forecast > 0:
            ml_score = np.clip(ml_forecast / 10.0 * 40, 0, 40)  # Positive: 10% forecast = 40 points
        else:
            # Negative forecasts get negative score (penalty)
            ml_score = np.clip(ml_forecast / 10.0 * 40, -40, 0)  # Negative: -10% forecast = -40 points
        score += ml_score
        
        # Hit Rate (20% weight) - normalized to 0-20
        hit_rate = results.get('hit_rate', 0.0)
        hit_score = hit_rate * 20  # 100% hit rate = 20 points
        score += hit_score
        
        # Kalman Signal Strength (20% weight) - normalized to 0-20
        kalman_signal = results.get('kalman_signal_strength', 0.0)
        kalman_score = np.clip(kalman_signal * 20, 0, 20)  # Signal strength 0-1 = 0-20 points
        score += kalman_score
        
        # CVaR (20% weight) - inverted (lower CVaR = better, but cap extreme values)
        cvar = results.get('cvar', 0.0)
        if not np.isnan(cvar) and cvar < 0:
            # Cap CVaR at -0.5 for scoring (extreme values like -0.95 are too penalizing)
            capped_cvar = max(cvar, -0.5)
            cvar_score = np.clip(-capped_cvar * 200, 0, 20)  # CVaR of -0.1 = 20 points
            score += cvar_score
        
        # Ensure score is between 0-100 (can go negative for very poor assets)
        return max(0.0, min(100.0, score))
    
    def print_summary(self, results: Dict):
        """
        Print formatted analysis summary
        
        Args:
            results: Analysis results dictionary
        """
        if results is None:
            print("❌ No results to display")
            return
        
        print(f"\n{'='*70}")
        print(f"ANALYSIS SUMMARY: {results['ticker']}")
        print(f"{'='*70}")
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Data Points: {results['data_points']} days\n")
        
        print(f"📊 {results['name']}")
        print(f"   Sector: {results['sector']} | Industry: {results['industry']}")
        print(f"   Current Price: ${results['latest_price']:.2f}")
        print(f"   Risk Category: {results['risk_category']} (Volatility: {results['volatility']*100:.1f}%)\n")
        
        print(f"{'='*70}")
        print("VALIDATED FACTORS")
        print(f"{'='*70}\n")
        
        print(f"🤖 ML ENSEMBLE:")
        print(f"   Forecast Return: {results['ml_forecast']:+.2f}%")
        print(f"   Hit Rate: {results['hit_rate']:.2%}")
        print(f"   Confidence: {results['ml_confidence']:.2%}\n")
        
        print(f"📉 RISK METRICS:")
        print(f"   CVaR: {results['cvar']:.4f}")
        if not np.isnan(results['beta']):
            print(f"   Beta: {results['beta']:.2f}")
        print(f"   Risk Score: {results['risk_score']:.2f}\n")
        
        print(f"📊 KALMAN HULL MOMENTUM:")
        trend_emoji = "🟢 BULLISH" if results['kalman_trend'] == 1 else "🔴 BEARISH"
        print(f"   Trend: {trend_emoji}")
        print(f"   Signal Strength: {results['kalman_signal_strength']:.3f}")
        print(f"   Efficiency Ratio: {results['kalman_efficiency']:.2%}\n")
        
        print(f"{'='*70}")
        print("PERFORMANCE METRICS")
        print(f"{'='*70}\n")
        
        print(f"📈 Returns:")
        print(f"   YTD Return: {results['ytd_return']:+.2f}%")
        print(f"   1-Year Return: {results['one_year_return']:+.2f}%\n")
        
        print(f"{'='*70}")
        print(f"🎯 COMPOSITE SCORE: {results['composite_score']:.1f}/100")
        print(f"{'='*70}\n")
        
        # Interpretation
        score = results['composite_score']
        if score >= 80:
            interpretation = "🌟 EXCELLENT - Strong buy signal"
        elif score >= 60:
            interpretation = "✅ GOOD - Positive outlook"
        elif score >= 40:
            interpretation = "⚠️  MODERATE - Mixed signals"
        elif score >= 20:
            interpretation = "❌ POOR - Weak outlook"
        else:
            interpretation = "🚫 VERY POOR - Strong sell signal"
        
        print(f"💡 Interpretation: {interpretation}\n")
        
        # Important note about ETF ranking vs single ticker analysis
        print(f"{'='*70}")
        print("ℹ️  NOTE: ETF Ranking vs Single Ticker Analysis")
        print(f"{'='*70}")
        print("""
The ETF analysis system uses PERCENTILE RANKING - ETFs are ranked relative 
to other ETFs in their risk category. A Bitcoin ETF might rank high even with 
a negative forecast if:
  • Other HIGH risk ETFs have worse forecasts
  • It has better hit rates, signal strength, or risk metrics vs peers
  • Percentile ranking compares to historical distribution

This single ticker analysis shows ABSOLUTE forecasts and metrics, not relative 
rankings. For relative comparison, check the ETF ranking tables.
        """)

