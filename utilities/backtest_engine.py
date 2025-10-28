#!/usr/bin/env python3
"""
Backtesting Engine for ETF Analysis System
Tests historical performance of composite scores and signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from indicators.kalman_hull import calculate_adaptive_kalman_hull
from analyzers.volume_intelligence import VolumeIntelligence
from analyzers.risk_component import RiskComponent
from analyzers.ml_ensemble import MLEnsemble
from analyzers.scoring_system_growth import GrowthScoringSystem


class BacktestEngine:
    """
    Backtest historical performance of the analysis system
    """
    
    def __init__(self, commission_pct: float = 0.001, slippage_pct: float = 0.0005):
        """
        Initialize backtest engine
        
        Args:
            commission_pct: Trading commission (0.1% = 0.001)
            slippage_pct: Slippage (0.05% = 0.0005)
        """
        self.commission = commission_pct
        self.slippage = slippage_pct
        self.total_cost = commission_pct + slippage_pct
        
        # Initialize components
        self.volume_intelligence = VolumeIntelligence()
        self.risk_component = RiskComponent()
        self.ml_ensemble = MLEnsemble()
        self.scoring = GrowthScoringSystem()
    
    def backtest_single_etf(self, prices: pd.Series, ohlc_data: pd.DataFrame, 
                           risk_category: str, rebalance_days: int = 60,
                           score_threshold: float = 27.0) -> Dict:
        """
        Backtest a single ETF using rolling analysis
        
        Args:
            prices: Price series
            ohlc_data: DataFrame with OHLC data
            risk_category: 'LOW', 'MEDIUM', 'HIGH'
            rebalance_days: Days between rebalancing (60 = every 2 months)
            score_threshold: Minimum score to hold position
        
        Returns:
            Dict with backtest results
        """
        if len(prices) < 150 + rebalance_days:  # Reduced from 252 for testing with 1yr data
            return self._empty_backtest_result(f"Insufficient data (need {150 + rebalance_days}, have {len(prices)})")
        
        # Backtest parameters
        lookback = 150  # Reduced from 252 for testing (6 months training)
        positions = []  # List of (date, position, score, signals)
        trades = []     # List of trades
        equity_curve = []
        
        # Starting capital
        capital = 10000.0
        position = 0  # 0 = no position, 1 = long
        entry_price = 0.0
        
        # Walk forward through history
        for i in range(lookback, len(prices), rebalance_days):
            date = prices.index[i]
            current_price = prices.iloc[i]
            
            # Analysis window
            window_prices = prices.iloc[max(0, i-lookback):i]
            window_ohlc = ohlc_data.iloc[max(0, i-lookback):i]
            window_volume = ohlc_data['Volume'].iloc[max(0, i-lookback):i] if 'Volume' in ohlc_data.columns else None
            
            if len(window_prices) < 100:
                continue
            
            # Run analysis
            try:
                analysis = self._run_analysis(window_prices, window_ohlc, window_volume, risk_category)
                result = self.scoring.calculate_composite_score(analysis, risk_category)
                score = result['composite_score']
                position_size = result['position_size']
                
                # Trading decision (RELAXED for realistic score distribution 25-45)
                should_hold = (
                    score >= score_threshold and
                    (analysis.get('kalman_trend', 0) == 1 or analysis.get('ml_forecast', 0) > 0) and
                    analysis.get('kalman_signal_strength', 0) > 0.2  # Lowered to 0.2
                )
                
                # Execute trades
                if should_hold and position == 0:
                    # ENTRY
                    position = 1
                    entry_price = current_price * (1 + self.total_cost)
                    shares = (capital * position_size) / entry_price
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': entry_price,
                        'shares': shares,
                        'score': score,
                        'signal_strength': analysis.get('kalman_signal_strength', 0),
                        'position_size': position_size
                    })
                
                elif not should_hold and position == 1:
                    # EXIT
                    exit_price = current_price * (1 - self.total_cost)
                    pnl = (exit_price - entry_price) / entry_price
                    capital *= (1 + pnl * position_size)
                    
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': exit_price,
                        'pnl': pnl,
                        'score': score
                    })
                    position = 0
                    entry_price = 0.0
                
                # Record state
                if position == 1:
                    current_value = capital * (1 + (current_price - entry_price) / entry_price * position_size)
                else:
                    current_value = capital
                
                equity_curve.append({
                    'date': date,
                    'equity': current_value,
                    'position': position,
                    'score': score
                })
                
            except Exception as e:
                print(f"Error at {date}: {e}")
                continue
        
        # Close any open position at end
        if position == 1:
            final_price = prices.iloc[-1] * (1 - self.total_cost)
            pnl = (final_price - entry_price) / entry_price
            capital *= (1 + pnl * position_size)
            trades.append({
                'date': prices.index[-1],
                'action': 'SELL',
                'price': final_price,
                'pnl': pnl,
                'score': 0
            })
        
        # Calculate metrics
        return self._calculate_backtest_metrics(
            pd.DataFrame(equity_curve),
            pd.DataFrame(trades),
            prices.iloc[lookback:],
            capital
        )
    
    def _run_analysis(self, prices: pd.Series, ohlc_data: pd.DataFrame, 
                     volume: Optional[pd.Series], risk_category: str) -> Dict:
        """Run full analysis on a data window"""
        analysis = {}
        
        # Kalman Hull
        kalman = calculate_adaptive_kalman_hull(prices, volume, risk_category, ohlc_data)
        analysis.update({
            'kalman_trend': kalman['trend'],
            'kalman_signal_strength': kalman['signal_strength'],
            'kalman_efficiency_ratio': kalman['efficiency_ratio'],
            'kalman_divergence': kalman['divergence']
        })
        
        # Volume Intelligence
        if volume is not None and len(volume) > 60:
            vol_intel = self.volume_intelligence.analyze_volume(prices, volume, ohlc_data)
            analysis.update({
                'volume_spike_score': vol_intel.get('spike_score', 50.0),
                'volume_correlation': vol_intel.get('price_volume_correlation', 0.0),
                'volume_ad_signal': vol_intel.get('ad_signal', 'neutral')
            })
        else:
            analysis.update({
                'volume_spike_score': 50.0,
                'volume_correlation': 0.0,
                'volume_ad_signal': 'neutral'
            })
        
        # Risk Component (simplified for backtesting)
        returns = prices.pct_change().dropna()
        if len(returns) > 60:
            volatility = returns.std() * np.sqrt(252)
            risk_score = min(1.0, volatility / 0.5)  # Normalize to 0-1
        else:
            risk_score = 0.5
        analysis['risk_score'] = risk_score
        analysis['cvar'] = -volatility * 2.5  # Approximation
        analysis['avg_daily_volume'] = volume.tail(60).mean() if volume is not None else 1_000_000
        
        # ML Forecast (simplified - use recent momentum as proxy)
        if len(prices) > 30:
            recent_return = (prices.iloc[-1] / prices.iloc[-30] - 1) * 100
            analysis['ml_forecast'] = np.clip(recent_return * 2, -15, 15)  # Scale to ±15%
            analysis['ml_confidence'] = 0.5  # Neutral confidence
            analysis['hit_rate'] = 0.55  # Assume average hit rate
        else:
            analysis['ml_forecast'] = 0.0
            analysis['ml_confidence'] = 0.5
            analysis['hit_rate'] = 0.5
        
        return analysis
    
    def _calculate_backtest_metrics(self, equity_curve: pd.DataFrame, trades: pd.DataFrame,
                                    prices: pd.Series, final_capital: float) -> Dict:
        """Calculate comprehensive backtest metrics"""
        if len(equity_curve) == 0:
            return self._empty_backtest_result("No trades executed")
        
        # Returns
        equity_curve = equity_curve.set_index('date')
        equity_returns = equity_curve['equity'].pct_change().dropna()
        
        # Buy & Hold benchmark
        bh_return = (prices.iloc[-1] / prices.iloc[0] - 1)
        
        # Strategy return
        strategy_return = (final_capital / 10000.0 - 1)
        
        # Win rate
        if len(trades) > 0:
            winning_trades = trades[trades['action'] == 'SELL']
            if len(winning_trades) > 0:
                win_rate = (winning_trades['pnl'] > 0).sum() / len(winning_trades)
                avg_win = winning_trades[winning_trades['pnl'] > 0]['pnl'].mean() if (winning_trades['pnl'] > 0).any() else 0
                avg_loss = winning_trades[winning_trades['pnl'] < 0]['pnl'].mean() if (winning_trades['pnl'] < 0).any() else 0
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        # Sharpe ratio
        if len(equity_returns) > 1 and equity_returns.std() > 0:
            sharpe = equity_returns.mean() / equity_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Max drawdown
        cumulative = (1 + equity_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'status': 'success',
            'total_return': strategy_return,
            'benchmark_return': bh_return,
            'excess_return': strategy_return - bh_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_capital': final_capital,
            'trades': trades.to_dict('records') if len(trades) > 0 else [],
            'equity_curve': equity_curve.reset_index().to_dict('records')
        }
    
    def _empty_backtest_result(self, reason: str) -> Dict:
        """Return empty result with reason"""
        return {
            'status': 'failed',
            'reason': reason,
            'total_return': 0.0,
            'benchmark_return': 0.0,
            'excess_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'final_capital': 10000.0,
            'trades': [],
            'equity_curve': []
        }
    
    def backtest_portfolio(self, etf_data: Dict[str, Tuple[pd.Series, pd.DataFrame]], 
                          risk_classifications: Dict[str, str],
                          top_n: int = 10, rebalance_days: int = 60) -> Dict:
        """
        Backtest a portfolio strategy across multiple ETFs
        
        Args:
            etf_data: Dict of {ticker: (prices, ohlc_data)}
            risk_classifications: Dict of {ticker: risk_category}
            top_n: Number of top-scored ETFs to hold
            rebalance_days: Days between rebalancing
        
        Returns:
            Portfolio backtest results
        """
        print(f"Backtesting portfolio with {len(etf_data)} ETFs, holding top {top_n}...")
        
        individual_results = {}
        for ticker, (prices, ohlc_data) in etf_data.items():
            risk_cat = risk_classifications.get(ticker, 'MEDIUM')
            print(f"  Backtesting {ticker} ({risk_cat})...")
            result = self.backtest_single_etf(prices, ohlc_data, risk_cat, rebalance_days)
            individual_results[ticker] = result
        
        # Calculate portfolio metrics
        successful = {k: v for k, v in individual_results.items() if v['status'] == 'success'}
        
        if len(successful) == 0:
            return {'status': 'failed', 'reason': 'No successful backtests'}
        
        # Average metrics
        avg_return = np.mean([v['total_return'] for v in successful.values()])
        avg_sharpe = np.mean([v['sharpe_ratio'] for v in successful.values()])
        avg_max_dd = np.mean([v['max_drawdown'] for v in successful.values()])
        avg_win_rate = np.mean([v['win_rate'] for v in successful.values()])
        
        return {
            'status': 'success',
            'num_etfs': len(successful),
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_max_drawdown': avg_max_dd,
            'avg_win_rate': avg_win_rate,
            'individual_results': individual_results,
            'best_performers': sorted(
                [(k, v['total_return']) for k, v in successful.items()],
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
        }


def main():
    """Example usage"""
    print("Backtest Engine - Example")
    print("=" * 60)
    print("To use: Load historical data and call backtest_single_etf()")
    print("\nExample:")
    print("  engine = BacktestEngine()")
    print("  result = engine.backtest_single_etf(prices, ohlc_data, 'MEDIUM')")
    print("  print(f'Return: {result[\"total_return\"]:.2%}')")
    print("  print(f'Sharpe: {result[\"sharpe_ratio\"]:.2f}')")


def run_backtest_on_universe(tickers: List[str] = None, sample_size: int = None) -> pd.DataFrame:
    """
    Run backtest on ETF universe or sample
    
    Args:
        tickers: Specific tickers to backtest. If None, uses all available.
        sample_size: If provided, randomly samples this many ETFs
    
    Returns:
        DataFrame with backtest results
    """
    from analyzers.etf_risk_classifier import ETFRiskClassifier
    from utilities.shared_utils import extract_column
    
    classifier = ETFRiskClassifier()
    engine = BacktestEngine()
    
    # Get all historical data files
    historical_dir = Path(__file__).parent.parent / 'data' / 'historical'
    if not historical_dir.exists():
        print(f"❌ No historical data found in {historical_dir}")
        return pd.DataFrame()
    
    # Get available tickers
    available_tickers = [f.stem.replace('_', '.') for f in historical_dir.glob('*.parquet')]
    
    if tickers:
        test_tickers = [t for t in tickers if t in available_tickers]
    elif sample_size:
        test_tickers = np.random.choice(available_tickers, min(sample_size, len(available_tickers)), replace=False).tolist()
    else:
        test_tickers = available_tickers
    
    print(f"\n{'='*80}")
    print(f"🧪 BACKTESTING {len(test_tickers)} ETFs")
    print(f"{'='*80}\n")
    
    results = []
    for i, ticker in enumerate(test_tickers, 1):
        try:
            # Load data
            file_path = historical_dir / f"{ticker.replace('.', '_')}.parquet"
            if not file_path.exists():
                continue
            
            data = pd.read_parquet(file_path)
            prices = extract_column(data, 'Close')
            
            if prices is None or len(prices) < 210:  # Min 150 train + 60 test
                print(f"  [{i}/{len(test_tickers)}] ❌ {ticker}: Insufficient data ({len(prices) if prices is not None else 0} days)")
                continue
            
            # Classify risk category
            returns = prices.pct_change().dropna()
            if len(returns) < 60:
                risk_category = 'MEDIUM'
            else:
                # Simple classification based on volatility
                vol_annual = returns.std() * np.sqrt(252)
                if vol_annual < 0.12:
                    risk_category = 'LOW'
                elif vol_annual < 0.25:
                    risk_category = 'MEDIUM'
                else:
                    risk_category = 'HIGH'
            
            # Run backtest
            result = engine.backtest_single_etf(prices, data, risk_category)
            
            if result['status'] == 'success':
                print(f"  [{i}/{len(test_tickers)}] ✅ {ticker}: Return {result['total_return']:+.1f}%, Sharpe {result['sharpe_ratio']:.2f}, Win Rate {result['win_rate']:.1f}%")
                results.append({
                    'ticker': ticker,
                    'risk_category': risk_category,
                    **result
                })
            else:
                print(f"  [{i}/{len(test_tickers)}] ❌ {ticker}: {result['reason']}")
        
        except Exception as e:
            print(f"  [{i}/{len(test_tickers)}] ❌ {ticker}: Error - {str(e)}")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        
        # Convert trades and equity_curve to JSON strings for Parquet compatibility
        if 'trades' in results_df.columns:
            results_df['trades'] = results_df['trades'].apply(lambda x: str(x) if x else '[]')
        if 'equity_curve' in results_df.columns:
            results_df['equity_curve'] = results_df['equity_curve'].apply(lambda x: str(x) if x else '[]')
        
        output_path = Path(__file__).parent.parent / 'data' / 'backtest_results.parquet'
        results_df.to_parquet(output_path, compression='snappy', index=False)
        
        print(f"\n{'='*80}")
        print(f"✅ BACKTEST COMPLETE")
        print(f"{'='*80}")
        print(f"Tested: {len(results)} ETFs")
        print(f"Avg Return: {results_df['total_return'].mean():+.2f}%")
        print(f"Avg Sharpe: {results_df['sharpe_ratio'].mean():.2f}")
        print(f"Avg Win Rate: {results_df['win_rate'].mean():.1f}%")
        print(f"\nResults saved to: {output_path}")
        print(f"{'='*80}\n")
        
        return results_df
    else:
        print("\n❌ No successful backtests")
        return pd.DataFrame()


if __name__ == "__main__":
    main()

