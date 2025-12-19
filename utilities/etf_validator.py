#!/usr/bin/env python3
"""
ETF Activity Validation Layer
Checks if ETFs are still active based on recent price data
Handles weekend/Monday edge cases for trading calendars
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ETFActivityValidator:
    """
    Validates ETF activity based on price data freshness
    Filters out delisted, suspended, or inactive ETFs
    """
    
    def __init__(self, max_hours: int = 24, debug: bool = False):
        """
        Initialize ETF activity validator
        
        Args:
            max_hours: Maximum hours since last price (default: 24)
            debug: Enable debug logging
        """
        self.max_hours = max_hours
        self.debug = debug
        
        # Trading calendar considerations
        self.weekend_hours = 48  # Saturday + Sunday
        self.holiday_buffer = 4   # Extra hours for holidays
        
    def validate_etf_activity(self, prices: pd.Series, ticker: str = None) -> Dict:
        """
        Check if ETF has recent price data within allowed timeframe
        
        Args:
            prices: Price series with DatetimeIndex
            ticker: ETF ticker for logging (optional)
            
        Returns:
            Dict with validation results
        """
        if prices.empty:
            result = {
                'is_active': False,
                'reason': 'No price data available',
                'hours_since_last': np.nan,
                'last_price_date': None,
                'threshold_hours': self.max_hours
            }
            if self.debug and ticker:
                print(f"❌ {ticker}: {result['reason']}")
            return result
        
        # Get latest price date
        latest_date = prices.index[-1]
        now = pd.Timestamp.now(tz=latest_date.tz) if latest_date.tz else pd.Timestamp.now()
        
        # Calculate time difference
        time_diff = (now - latest_date).total_seconds() / 3600
        
        # Determine threshold based on day of week
        threshold = self._get_threshold_for_day(now, latest_date)
        
        # Check if within threshold
        is_active = time_diff <= threshold
        
        # Determine reason if inactive
        if not is_active:
            if time_diff > 168:  # More than 1 week
                reason = 'Likely delisted or suspended'
            elif time_diff > 72:  # More than 3 days
                reason = 'Inactive - no recent trading'
            else:
                reason = f'Stale data ({time_diff:.1f}h old > {threshold}h threshold)'
        else:
            reason = 'Active - recent price data available'
        
        result = {
            'is_active': is_active,
            'reason': reason,
            'hours_since_last': time_diff,
            'last_price_date': latest_date,
            'threshold_hours': threshold
        }
        
        if self.debug and ticker:
            status = "✅" if is_active else "❌"
            print(f"{status} {ticker}: {reason} ({time_diff:.1f}h old)")
        
        return result
    
    def _get_threshold_for_day(self, now: pd.Timestamp, last_price: pd.Timestamp) -> int:
        """
        Get appropriate threshold based on current day and last price day
        
        Args:
            now: Current timestamp
            last_price: Last price timestamp
            
        Returns:
            Threshold in hours
        """
        now_weekday = now.weekday()  # 0=Monday, 6=Sunday
        last_weekday = last_price.weekday()
        
        # Calculate days difference
        days_diff = (now.date() - last_price.date()).days
        
        # Monday special case: allow weekend time
        if now_weekday == 0:  # Monday
            # If last price was Friday, allow entire weekend
            if last_weekday == 5:  # Friday
                return self.max_hours + self.weekend_hours + self.holiday_buffer
            # If last price was Thursday, allow weekend + Friday
            elif last_weekday == 4:  # Thursday
                return self.max_hours + self.weekend_hours + 24 + self.holiday_buffer
            else:
                return self.max_hours + 48  # Conservative weekend allowance
        
        # Sunday: allow since markets closed
        elif now_weekday == 6:
            # If last price was Friday, allow entire weekend (Friday close to Sunday)
            if last_weekday == 5:  # Friday
                return self.max_hours + self.weekend_hours + self.holiday_buffer
            return self.max_hours + 24
        
        # Saturday: allow since markets closed
        # FIX: If last price was Friday, always allow it on Saturday (markets closed)
        elif now_weekday == 5:
            # If last price was Friday, allow it regardless of hours (markets closed)
            if last_weekday == 4:  # Friday
                return 72  # Allow up to 72 hours (Friday close to Monday open)
            # If last price was Thursday or earlier, use normal threshold
            return self.max_hours + 12
        
        # Friday: allow extra time for data updates
        elif now_weekday == 4:
            # If data is from previous day, allow extra buffer
            if days_diff >= 1:
                return self.max_hours + 24 + self.holiday_buffer
            else:
                return self.max_hours + 8  # Same day buffer
        
        # Thursday: allow some buffer
        elif now_weekday == 3:
            if days_diff >= 1:
                return self.max_hours + 16
            else:
                return self.max_hours + 4
        
        # Tuesday, Wednesday: normal weekday with small buffer
        else:  # 1=Tuesday, 2=Wednesday
            if days_diff >= 1:
                return self.max_hours + 12
            else:
                return self.max_hours
    
    def validate_etf_universe(self, price_data: Dict[str, pd.Series]) -> Dict:
        """
        Validate entire ETF universe
        
        Args:
            price_data: Dict of {ticker: price_series}
            
        Returns:
            Dict with validation summary and filtered ETFs
        """
        results = {}
        active_etfs = {}
        inactive_etfs = {}
        
        print(f"🔍 Validating {len(price_data)} ETFs for activity...")
        
        for ticker, prices in price_data.items():
            validation = self.validate_etf_activity(prices, ticker)
            results[ticker] = validation
            
            if validation['is_active']:
                active_etfs[ticker] = prices
            else:
                inactive_etfs[ticker] = validation
        
        # Summary statistics
        total_count = len(price_data)
        active_count = len(active_etfs)
        inactive_count = len(inactive_etfs)
        active_pct = (active_count / total_count) * 100 if total_count > 0 else 0
        
        # Categorize inactive reasons
        inactive_reasons = {}
        for ticker, validation in inactive_etfs.items():
            reason = validation['reason']
            if reason not in inactive_reasons:
                inactive_reasons[reason] = []
            inactive_reasons[reason].append(ticker)
        
        print(f"\n📊 ETF Activity Validation Summary:")
        print(f"  Total ETFs: {total_count}")
        print(f"  ✅ Active: {active_count} ({active_pct:.1f}%)")
        print(f"  ❌ Inactive: {inactive_count} ({100-active_pct:.1f}%)")
        
        if inactive_reasons:
            print(f"\n🔍 Inactive ETF Breakdown:")
            for reason, tickers in inactive_reasons.items():
                print(f"  • {reason}: {len(tickers)} ETFs")
                if len(tickers) <= 5:
                    print(f"    {', '.join(tickers)}")
                else:
                    print(f"    {', '.join(tickers[:3])}... (+{len(tickers)-3} more)")
        
        return {
            'validation_results': results,
            'active_etfs': active_etfs,
            'inactive_etfs': inactive_etfs,
            'summary': {
                'total_count': total_count,
                'active_count': active_count,
                'inactive_count': inactive_count,
                'active_percentage': active_pct,
                'inactive_reasons': inactive_reasons
            }
        }
    
    def get_activity_report(self, validation_results: Dict) -> str:
        """
        Generate human-readable activity report
        
        Args:
            validation_results: Results from validate_etf_universe
            
        Returns:
            Formatted report string
        """
        summary = validation_results['summary']
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    ETF ACTIVITY VALIDATION REPORT           ║
╚══════════════════════════════════════════════════════════════╝

📊 SUMMARY:
• Total ETFs Analyzed: {summary['total_count']}
• Active ETFs: {summary['active_count']} ({summary['active_percentage']:.1f}%)
• Inactive ETFs: {summary['inactive_count']} ({100-summary['active_percentage']:.1f}%)

⏰ TIME THRESHOLD: {self.max_hours} hours (adjusted for weekends)

🔍 INACTIVE BREAKDOWN:
"""
        
        for reason, tickers in summary['inactive_reasons'].items():
            report += f"• {reason}: {len(tickers)} ETFs\n"
            if len(tickers) <= 3:
                report += f"  {', '.join(tickers)}\n"
            else:
                report += f"  {', '.join(tickers[:2])}... (+{len(tickers)-2} more)\n"
        
        report += f"\n✅ RECOMMENDATION: Proceed with {summary['active_count']} active ETFs\n"
        
        return report


def validate_etf_data_simple(prices: pd.Series, max_hours: int = 24) -> bool:
    """
    Simple validation function for quick checks
    
    Args:
        prices: Price series
        max_hours: Maximum hours since last price
        
    Returns:
        True if ETF appears active, False otherwise
    """
    validator = ETFActivityValidator(max_hours=max_hours)
    result = validator.validate_etf_activity(prices)
    return result['is_active']


if __name__ == "__main__":
    # Test the validator
    validator = ETFActivityValidator(debug=True)
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = pd.Series(100 + np.random.randn(100).cumsum() * 0.5, index=dates)
    
    # Test recent data
    recent_prices = prices.copy()
    recent_prices.index = pd.date_range('2024-12-04', periods=100, freq='D')
    
    print("=== ETF Activity Validator Test ===")
    result = validator.validate_etf_activity(recent_prices, "TEST.AX")
    print(f"Result: {result}")
    
    # Test with old data
    old_prices = prices.copy()
    old_prices.index = pd.date_range('2024-01-01', periods=100, freq='D')
    
    print("\n=== Old Data Test ===")
    result = validator.validate_etf_activity(old_prices, "OLD.AX")
    print(f"Result: {result}")
