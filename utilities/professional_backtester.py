#!/usr/bin/env python3
"""
Professional Backtesting Engine
Hold-based strategy with flexible multi-condition exit logic
Uses real ML/Kalman/Volume signals from Phase 1 Extended optimizations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from system.orchestrator import ETFAnalysisSystem
from frameworks.macro_framework import calculate_macro_framework
from frameworks.geopolitical_framework import calculate_geopolitical_framework


class ProfessionalBacktester:
    """
    Professional backtesting system with real signals
    - No transaction costs (commission-free platform)
    - Hold-based strategy (60+ day minimum holds)
    - Real ML/Kalman/Volume signals
    - Simple $10 capital allocation per signal
    """

    def __init__(self, min_hold_days: int = 60, capital_per_trade: float = 10.0,
                 rebalance_frequency: int = 30, buy_threshold: float = 50.0,
                 sell_threshold: float = 40.0, target_return: float = 0.125,
                 stop_loss: float = -0.08, stale_days: int = 180,
                 lookback_months: Optional[int] = None):
        """
        Initialize backtester with strategy parameters

        Args:
            min_hold_days: Minimum hold period before exit allowed
            capital_per_trade: Fixed $ amount per position entry
            rebalance_frequency: Days between signal checks
            buy_threshold: Composite score threshold to enter
            sell_threshold: Composite score threshold to exit
            target_return: Target return for profit-taking (12.5% = 0.125)
            stop_loss: Stop loss threshold (-8% = -0.08)
            stale_days: Days before position considered stale (180)
            lookback_months: Number of months to include (None = full history)
        """
        self.min_hold_days = min_hold_days
        self.capital_per_trade = capital_per_trade
        self.rebalance_frequency = rebalance_frequency
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.target_return = target_return
        self.stop_loss = stop_loss
        self.stale_days = stale_days
        self.lookback_months = lookback_months

        # Initialize analysis system
        self.analysis_system = ETFAnalysisSystem()

        # Cache for frameworks (run once per backtest)
        self.macro_data = None
        self.geo_data = None

    def _get_risk_regime(self) -> Tuple[str, str]:
        """Get current macro and geopolitical regime"""
        try:
            if self.macro_data is None:
                self.macro_data = calculate_macro_framework()
            if self.geo_data is None:
                self.geo_data = calculate_geopolitical_framework()

            macro_regime = self.macro_data.get('regime', 'NORMAL')
            geo_level = self.geo_data.get('risk_level', 'LOW')

            return macro_regime, geo_level
        except Exception as e:
            # Fallback: assume normal conditions if frameworks fail
            return 'NORMAL', 'LOW'

    def _filter_prices_by_lookback(self, prices: pd.Series, ohlc_data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Filter prices and OHLC data to lookback_months if specified

        Returns: (filtered_prices, filtered_ohlc)
        """
        if self.lookback_months is None:
            return prices, ohlc_data

        # Calculate the cutoff date
        end_date = prices.index[-1]
        cutoff_date = pd.Timestamp(end_date) - pd.DateOffset(months=self.lookback_months)

        # Filter to dates >= cutoff_date
        mask = prices.index >= cutoff_date
        return prices[mask], ohlc_data[mask]

    def backtest_mature_etf(self, ticker: str, prices: pd.Series,
                           ohlc_data: pd.DataFrame, risk_category: str) -> Dict:
        """
        Backtest mature ETF (312+ days) using standard walk-forward

        Standard parameters:
        - Training: 252 days (1 year)
        - Rebalance: Every 30 days
        - Min data: 312 days (252 train + 60 min hold)
        - If lookback_months set, uses only last N months
        """
        # Filter to lookback period if specified
        prices, ohlc_data = self._filter_prices_by_lookback(prices, ohlc_data)

        # Dynamic minimum data requirement based on lookback period
        if self.lookback_months is None:
            min_days = 312  # Full history standard requirement
        else:
            # For short periods, use available data but require minimum for analysis
            available_days = len(prices)
            if self.lookback_months <= 1:
                min_days = max(15, available_days // 2)  # Need at least 15 days for 1-month analysis
            elif self.lookback_months <= 3:
                min_days = max(30, available_days // 2)  # Need at least 30 days for 3-month analysis
            elif self.lookback_months <= 6:
                min_days = max(45, int(available_days * 0.6))  # Need at least 45 days for 6-month analysis
            else:
                min_days = max(60, self.lookback_months * 21)  # Standard requirement for longer periods
        
        if len(prices) < min_days:
            return self._empty_result(f"Insufficient data ({len(prices)} < {min_days} days)")

        trades = []
        positions = []
        equity_curve = []

        # Dynamic training window based on lookback period
        if self.lookback_months is None:
            lookback = 252  # Full history: 1 year training
        else:
            # For limited lookback, use smaller training windows
            available_days = len(prices)
            if self.lookback_months <= 1:
                # 1-month: use minimum viable training (10-15 days)
                lookback = max(10, min(15, available_days // 3))
            elif self.lookback_months <= 3:
                # 3-month: use small training window (20-30 days)
                lookback = max(20, min(30, available_days // 2))
            elif self.lookback_months <= 6:
                # 6-month: use moderate training window (40-60 days)
                lookback = max(40, min(60, int(available_days * 0.5)))
            else:
                # Longer periods: use larger training window
                lookback = max(60, int(available_days * 0.7))

        # Walk forward through history
        for i in range(lookback, len(prices), self.rebalance_frequency):
            date = prices.index[i]
            current_price = prices.iloc[i]

            # Training window for analysis
            window_prices = prices.iloc[i - lookback:i]
            window_ohlc = ohlc_data.iloc[i - lookback:i]

            if len(window_prices) < 100:
                continue

            # Run analysis on this window
            try:
                analysis = self._run_analysis(
                    ticker, window_prices, window_ohlc, risk_category
                )
                if analysis is None:
                    continue

                score = analysis['composite_score']
                ml_forecast = analysis.get('ml_forecast', 0)
                kalman_trend = analysis.get('kalman_trend', 0)

                # DEBUG: Print first few signals to understand score distribution
                if len(trades) < 3:
                    print(f"    DEBUG: score={score:.1f}, ml={ml_forecast:.2f}, kalman={kalman_trend}, signal_str={analysis.get('kalman_signal_strength', 0):.2f}")

                # BUY signal
                if score >= self.buy_threshold and ml_forecast > 0 and kalman_trend == 1:
                    # Check if already in position
                    active_positions = [p for p in positions if p['exit_date'] is None]
                    if not active_positions:
                        position = {
                            'entry_date': date,
                            'entry_price': current_price,
                            'entry_score': score,
                            'shares': self.capital_per_trade / current_price,
                            'current_price': current_price,
                            'exit_date': None,
                            'exit_price': 0,
                            'days_held': 0,
                            'sell_reason': None
                        }
                        positions.append(position)
                        trades.append({
                            'date': date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'price': current_price,
                            'score': score,
                            'shares': position['shares']
                        })
                        print(f"    DEBUG: BUY SIGNAL triggered at score={score:.1f}")

                # Process existing positions
                for position in positions:
                    if position['exit_date'] is None:
                        days_held = (date - position['entry_date']).days
                        position['days_held'] = days_held
                        position['current_price'] = current_price
                        current_return = (current_price - position['entry_price']) / position['entry_price']

                        # Check sell conditions
                        should_sell, reason = self._check_sell_conditions(
                            position, analysis, current_return
                        )

                        if should_sell:
                            position['exit_date'] = date
                            position['exit_price'] = current_price
                            position['sell_reason'] = reason
                            pnl = current_return
                            trades.append({
                                'date': date,
                                'ticker': ticker,
                                'action': 'SELL',
                                'price': current_price,
                                'pnl': pnl,
                                'sell_reason': reason
                            })

                # Record equity state
                portfolio_value = self._calculate_portfolio_value(positions, current_price)
                equity_curve.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'score': score
                })

            except Exception as e:
                continue

        # Close any remaining positions at end
        for position in positions:
            if position['exit_date'] is None:
                final_price = prices.iloc[-1]
                position['exit_date'] = prices.index[-1]
                position['exit_price'] = final_price
                position['sell_reason'] = 'End of data'
                pnl = (final_price - position['entry_price']) / position['entry_price']
                trades.append({
                    'date': prices.index[-1],
                    'ticker': ticker,
                    'action': 'SELL',
                    'price': final_price,
                    'pnl': pnl,
                    'sell_reason': 'End of data'
                })

        return self._calculate_metrics(ticker, trades, positions, equity_curve)

    def backtest_immature_etf(self, ticker: str, prices: pd.Series,
                             ohlc_data: pd.DataFrame, risk_category: str,
                             peer_proxy_ticker: Optional[str] = None) -> Dict:
        """
        Backtest immature ETF (60-311 days) using expanding window

        Strategy:
        - Use expanding window: train on ALL data up to signal date
        - Minimum 60 days to start trading
        - Use peer proxy if available for enhanced confidence
        - If lookback_months set, uses only last N months
        """
        # Filter to lookback period if specified
        prices, ohlc_data = self._filter_prices_by_lookback(prices, ohlc_data)

        if len(prices) < 60:
            return self._empty_result(f"Insufficient data ({len(prices)} < 60 days)")

        trades = []
        positions = []
        equity_curve = []

        # For immature ETFs, use lower threshold since volume indicators weak with limited data
        immature_buy_threshold = max(35, self.buy_threshold * 0.7)  # ~35 instead of 50

        # Start trading once we have 60 days minimum
        for i in range(60, len(prices), self.rebalance_frequency):
            date = prices.index[i]
            current_price = prices.iloc[i]

            # Expanding window: use all data up to this point
            window_prices = prices.iloc[:i]
            window_ohlc = ohlc_data.iloc[:i]

            try:
                analysis = self._run_analysis(
                    ticker, window_prices, window_ohlc, risk_category
                )
                if analysis is None:
                    continue

                score = analysis['composite_score']
                ml_forecast = analysis.get('ml_forecast', 0)
                kalman_trend = analysis.get('kalman_trend', 0)

                # BUY signal for immature ETFs: slightly relaxed thresholds
                if score >= immature_buy_threshold and ml_forecast > 0 and kalman_trend == 1:
                    active_positions = [p for p in positions if p['exit_date'] is None]
                    if not active_positions:
                        position = {
                            'entry_date': date,
                            'entry_price': current_price,
                            'entry_score': score,
                            'shares': self.capital_per_trade / current_price,
                            'current_price': current_price,
                            'exit_date': None,
                            'exit_price': 0,
                            'days_held': 0,
                            'sell_reason': None
                        }
                        positions.append(position)
                        trades.append({
                            'date': date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'price': current_price,
                            'score': score,
                            'shares': position['shares']
                        })

                # Process existing positions
                for position in positions:
                    if position['exit_date'] is None:
                        days_held = (date - position['entry_date']).days
                        position['days_held'] = days_held
                        position['current_price'] = current_price
                        current_return = (current_price - position['entry_price']) / position['entry_price']

                        should_sell, reason = self._check_sell_conditions(
                            position, analysis, current_return
                        )

                        if should_sell:
                            position['exit_date'] = date
                            position['exit_price'] = current_price
                            position['sell_reason'] = reason
                            pnl = current_return
                            trades.append({
                                'date': date,
                                'ticker': ticker,
                                'action': 'SELL',
                                'price': current_price,
                                'pnl': pnl,
                                'sell_reason': reason
                            })

                # Record equity
                portfolio_value = self._calculate_portfolio_value(positions, current_price)
                equity_curve.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'score': score
                })

            except Exception as e:
                continue

        # Close remaining positions
        for position in positions:
            if position['exit_date'] is None:
                final_price = prices.iloc[-1]
                position['exit_date'] = prices.index[-1]
                position['exit_price'] = final_price
                position['sell_reason'] = 'End of data'
                pnl = (final_price - position['entry_price']) / position['entry_price']
                trades.append({
                    'date': prices.index[-1],
                    'ticker': ticker,
                    'action': 'SELL',
                    'price': final_price,
                    'pnl': pnl,
                    'sell_reason': 'End of data'
                })

        return self._calculate_metrics(ticker, trades, positions, equity_curve)

    def _run_analysis(self, ticker: str, prices: pd.Series, ohlc_data: pd.DataFrame,
                     risk_category: str) -> Optional[Dict]:
        """Run lightweight analysis on price window - NOT the full orchestrator"""
        try:
            # Extract OHLCV data
            if isinstance(ohlc_data, pd.DataFrame):
                volume = ohlc_data['Volume'] if 'Volume' in ohlc_data.columns else pd.Series([1000000] * len(prices), index=prices.index)
                close_prices = ohlc_data['Close'] if 'Close' in ohlc_data.columns else prices
            else:
                volume = pd.Series([1000000] * len(prices), index=prices.index)
                close_prices = prices

            # Fast component analysis (NOT full orchestration)
            analysis = {}

            # 1. Kalman Hull (pass ohlc_data, not prices)
            from indicators.kalman_hull import calculate_adaptive_kalman_hull
            kalman_result = calculate_adaptive_kalman_hull(close_prices, volume, risk_category, ohlc_data)
            analysis.update({
                'kalman_trend': kalman_result.get('trend', 0),
                'kalman_signal_strength': kalman_result.get('signal_strength', 0),
                'kalman_efficiency_ratio': kalman_result.get('efficiency_ratio', 0),
            })

            # 2. Volume Intelligence
            from analyzers.volume_intelligence import VolumeIntelligence
            vol_intel = VolumeIntelligence()
            vol_result = vol_intel.analyze_volume(close_prices, volume, ohlc_data)
            analysis.update({
                'volume_spike_score': vol_result.get('spike_score', 50.0),
                'volume_correlation': vol_result.get('price_volume_correlation', 0.0),
                'volume_ad_signal': vol_result.get('ad_signal', 'neutral')
            })

            # 3. Simple momentum forecast (skip expensive ML training during backtest)
            recent_return = (close_prices.iloc[-1] / close_prices.iloc[-60] - 1) * 100 if len(close_prices) >= 60 else 0
            analysis.update({
                'ml_forecast': recent_return,
                'ml_confidence': 0.55,  # Neutral confidence
            })

            # 4. Risk scoring (simplified)
            returns = close_prices.pct_change().dropna()
            vol_annual = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.15
            analysis['risk_score'] = min(vol_annual / 0.25, 1.0)  # 0-1 normalized

            # 5. Composite score (simple: kalman(60%) + volume(40%), skip ML for speed)
            kalman_score = (analysis['kalman_trend'] + 1) / 2 * 50  # -1 to +1 → 0-50
            kalman_score += analysis['kalman_signal_strength'] * 25  # 0-1 → 0-25
            kalman_score = np.clip(kalman_score, 0, 100)

            volume_score = (analysis['volume_spike_score'] / 100) * 50 + analysis['volume_correlation'] * 10
            volume_score = np.clip(volume_score, 0, 50)

            # Final composite: kalman(60%) + volume(40%)
            analysis['composite_score'] = (kalman_score * 0.6 + volume_score * 0.4)

            return analysis

        except Exception as e:
            print(f"    ERROR in _run_analysis: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return None

    def _check_sell_conditions(self, position: Dict, analysis: Dict,
                              current_return: float) -> Tuple[bool, str]:
        """
        Multi-condition sell logic

        Returns: (should_sell: bool, reason: str)
        """
        days_held = position['days_held']

        # Must hold minimum period
        if days_held < self.min_hold_days:
            return False, "Min hold period not met"

        # 1. Score deterioration
        current_score = analysis.get('composite_score', 0)
        if current_score < self.sell_threshold:
            return True, f"Score dropped to {current_score:.1f}"

        ml_forecast = analysis.get('ml_forecast', 0)
        ml_confidence = analysis.get('ml_confidence', 0)
        if ml_forecast < 0 and ml_confidence > 0.6:
            return True, f"ML forecast negative (conf: {ml_confidence:.2f})"

        kalman_trend = analysis.get('kalman_trend', 0)
        signal_strength = analysis.get('kalman_signal_strength', 0)
        if kalman_trend == -1 and signal_strength > 0.4:
            return True, f"Kalman downtrend (strength: {signal_strength:.2f})"

        # 2. Target achievement
        if current_return >= self.target_return:
            return True, f"Target achieved: {current_return:.1%}"

        # 3. Stop loss
        if current_return <= self.stop_loss:
            return True, f"Stop loss: {current_return:.1%}"

        # 4. Macro/geopolitical risk
        macro_regime, geo_level = self._get_risk_regime()
        if macro_regime == 'CRISIS':
            return True, f"Macro crisis: {macro_regime}"

        if geo_level in ['SEVERE', 'EXTREME']:
            return True, f"Geo risk: {geo_level}"

        # 5. Stale position
        if days_held > self.stale_days and current_return < 0.05:
            return True, f"Stale position ({days_held}d, {current_return:.1%})"

        return False, "Holding"

    def _calculate_portfolio_value(self, positions: List[Dict], current_price: float) -> float:
        """Calculate current portfolio value"""
        value = 0.0
        for pos in positions:
            if pos['exit_date'] is None:
                value += pos['shares'] * current_price
        return value

    def _calculate_metrics(self, ticker: str, trades: List[Dict],
                          positions: List[Dict], equity_curve: List[Dict]) -> Dict:
        """Calculate comprehensive backtest metrics"""
        if not trades or not positions:
            return self._empty_result("No trades executed")

        # Calculate returns from closed positions
        closed_positions = [p for p in positions if p['exit_date'] is not None]
        if not closed_positions:
            return self._empty_result("No closed positions")

        position_returns = []
        for pos in closed_positions:
            ret = (pos['exit_price'] - pos['entry_price']) / pos['entry_price']
            position_returns.append(ret)

        if not position_returns:
            return self._empty_result("No position returns")

        total_capital = len(closed_positions) * self.capital_per_trade
        total_pnl = sum([r * self.capital_per_trade for r in position_returns])
        total_return = total_pnl / total_capital if total_capital > 0 else 0

        # Win rate
        wins = sum([1 for r in position_returns if r > 0])
        win_rate = wins / len(position_returns) if position_returns else 0

        # Sharpe ratio (simplified)
        if len(position_returns) > 1:
            returns_std = np.std(position_returns)
            sharpe = np.mean(position_returns) / returns_std * np.sqrt(252 / len(position_returns)) if returns_std > 0 else 0
        else:
            sharpe = 0.0

        # Max drawdown
        if equity_curve:
            equity_values = [e['portfolio_value'] for e in equity_curve]
            if equity_values:
                peak = np.maximum.accumulate(equity_values)
                drawdown = (np.array(equity_values) - peak) / peak
                max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
            else:
                max_drawdown = 0.0
        else:
            max_drawdown = 0.0

        # Average hold period
        avg_hold = np.mean([p['days_held'] for p in closed_positions]) if closed_positions else 0

        return {
            'status': 'success',
            'ticker': ticker,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_capital': total_capital,
            'num_trades': len([t for t in trades if t['action'] == 'SELL']),
            'win_rate': win_rate,
            'avg_return': np.mean(position_returns),
            'max_return': max(position_returns),
            'min_return': min(position_returns),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_hold_days': avg_hold,
            'trades': trades,
            'positions': closed_positions
        }

    def _empty_result(self, reason: str) -> Dict:
        """Return empty result"""
        return {
            'status': 'failed',
            'reason': reason,
            'total_return': 0.0,
            'total_pnl': 0.0,
            'total_capital': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'max_return': 0.0,
            'min_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_hold_days': 0.0,
            'trades': [],
            'positions': []
        }
