"""
Volume Intelligence Component
Analyzes volume patterns for spike detection, price-volume correlation, and A/D signals
"""

import numpy as np
import pandas as pd
from typing import Dict


class VolumeIntelligence:
    """Volume analysis component with 3 sub-indicators"""
    
    def __init__(self):
        """Initialize Volume Intelligence"""
        self.lookback = 20  # Standard lookback period
    
    def analyze_volume(self, prices: pd.Series, volume: pd.Series, ohlc_data: pd.DataFrame = None) -> Dict:
        """
        Main function - Complete volume analysis
        
        Args:
            prices: Price series (Close prices)
            volume: Volume series
            ohlc_data: DataFrame with Open, High, Low, Close columns (optional, for proper A/D Line)
        
        Returns:
            Dict with 4 fields: spike_score, price_volume_correlation,
                               accumulation_distribution, volume_confidence
        """
        if len(prices) < self.lookback or len(volume) < self.lookback:
            return self._empty_result()
        
        # Remove NaN values
        valid_idx = ~(prices.isna() | volume.isna())
        prices = prices[valid_idx]
        volume = volume[valid_idx]
        
        if len(prices) < self.lookback:
            return self._empty_result()
        
        # 1. Calculate Volume Spike Score
        spike_score = self._calculate_spike_score(volume)
        
        # 2. Calculate Price-Volume Correlation
        pv_correlation = self._calculate_price_volume_correlation(prices, volume)
        
        # 3. Detect Accumulation/Distribution
        ad_signal = self._detect_accumulation_distribution(prices, volume, ohlc_data)
        
        # 4. Calculate Volume Confidence
        vol_confidence = self._calculate_volume_confidence(volume)
        
        return {
            'spike_score': float(spike_score),
            'price_volume_correlation': float(pv_correlation),
            'accumulation_distribution': ad_signal,
            'volume_confidence': float(vol_confidence)
        }
    
    def _calculate_spike_score(self, volume: pd.Series) -> float:
        """
        Calculate volume spike score (0-100)
        Combines Relative Volume Ratio and Z-Score
        
        Formula:
        - RVR = current_volume / mean(volume[t-20:t])
        - z_score = (current_volume - μ_20d) / σ_20d
        - rvr_component = min(100, (RVR - 1.0) × 50)
        - z_component = min(100, z_score × 25)
        - spike_score = 0.6 × rvr_component + 0.4 × z_component
        """
        if len(volume) < self.lookback + 1:
            return 0.0
        
        current_vol = volume.iloc[-1]
        recent_vol = volume.iloc[-self.lookback-1:-1]
        
        # Relative Volume Ratio
        mean_vol = recent_vol.mean()
        if mean_vol == 0 or np.isnan(mean_vol):
            rvr = 1.0
        else:
            rvr = current_vol / mean_vol
        
        # Z-Score
        std_vol = recent_vol.std()
        if std_vol == 0 or np.isnan(std_vol):
            z_score = 0.0
        else:
            z_score = (current_vol - mean_vol) / std_vol
        
        # Combined spike score
        rvr_component = min(100, max(0, (rvr - 1.0) * 50))
        z_component = min(100, max(0, z_score * 25))
        
        spike_score = 0.6 * rvr_component + 0.4 * z_component
        
        return max(0.0, min(100.0, spike_score))
    
    def _calculate_price_volume_correlation(self, prices: pd.Series, 
                                           volume: pd.Series) -> float:
        """
        Calculate 20-day rolling correlation between price changes and volume
        
        Returns: Correlation coefficient [-1, 1]
        """
        if len(prices) < self.lookback + 1:
            return 0.0
        
        # Get recent data
        recent_prices = prices.iloc[-self.lookback-1:]
        recent_volume = volume.iloc[-self.lookback-1:]
        
        # Calculate price changes (absolute)
        price_changes = recent_prices.pct_change().abs().iloc[1:]
        volume_series = recent_volume.iloc[1:]
        
        # Calculate correlation
        if len(price_changes) < 2:
            return 0.0
        
        correlation = price_changes.corr(volume_series)
        
        if np.isnan(correlation):
            return 0.0
        
        return max(-1.0, min(1.0, correlation))
    
    def _detect_accumulation_distribution(self, prices: pd.Series, 
                                         volume: pd.Series, ohlc_data: pd.DataFrame = None) -> str:
        """
        Detect accumulation/distribution based on Money Flow Multiplier
        
        Logic:
        - Accumulation: price < 2% change AND A/D rising > 5%
        - Distribution: price > 2% change AND A/D falling < -2%
        - Neutral: otherwise
        
        Returns: 'accumulation', 'distribution', or 'neutral'
        """
        if len(prices) < self.lookback + 1:
            return 'neutral'
        
        # Calculate A/D Line
        ad_line = self._calculate_ad_line(prices, volume, ohlc_data)
        
        if len(ad_line) < self.lookback + 1:
            return 'neutral'
        
        # 20-day comparison
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-self.lookback-1]
        current_ad = ad_line.iloc[-1]
        past_ad = ad_line.iloc[-self.lookback-1]
        
        # Price change percentage
        price_change = (current_price - past_price) / past_price if past_price != 0 else 0
        
        # A/D change percentage
        if abs(past_ad) < 1e-10:  # Avoid division by zero
            ad_change = 0.0
        else:
            ad_change = (current_ad - past_ad) / abs(past_ad)
        
        # Determine signal
        if price_change < 0.02 and ad_change > 0.05:
            return 'accumulation'
        elif price_change > 0.02 and ad_change < -0.02:
            return 'distribution'
        else:
            return 'neutral'
    
    def _calculate_ad_line(self, prices: pd.Series, volume: pd.Series, ohlc_data: pd.DataFrame = None) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line with proper OHLC data if available
        
        Formula:
        - mfm = [(close - low) - (high - close)] / (high - low)
        - money_flow_volume = mfm × volume
        - ad_line[t] = ad_line[t-1] + money_flow_volume
        """
        n = len(prices)
        ad_line = np.zeros(n)
        
        # Try to use OHLC data if provided
        if ohlc_data is not None and not ohlc_data.empty:
            try:
                high = ohlc_data['High'] if 'High' in ohlc_data.columns else prices
                low = ohlc_data['Low'] if 'Low' in ohlc_data.columns else prices
                close = ohlc_data['Close'] if 'Close' in ohlc_data.columns else prices
                
                # Handle MultiIndex columns from yfinance
                if isinstance(high, pd.DataFrame):
                    high = high.iloc[:, 0]
                if isinstance(low, pd.DataFrame):
                    low = low.iloc[:, 0]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
            except:
                # Fallback to Close only
                high = low = close = prices
        else:
            # Fallback: Use close as proxy
            high = low = close = prices
        
        for i in range(n):
            # Money Flow Multiplier
            if high.iloc[i] == low.iloc[i]:
                mfm = 0
            else:
                mfm = ((close.iloc[i] - low.iloc[i]) - (high.iloc[i] - close.iloc[i])) / (high.iloc[i] - low.iloc[i])
            
            # Money Flow Volume
            mfv = mfm * volume.iloc[i]
            
            # Cumulative A/D Line
            ad_line[i] = ad_line[i-1] + mfv if i > 0 else mfv
        
        return pd.Series(ad_line, index=prices.index)
    
    def _calculate_volume_confidence(self, volume: pd.Series) -> float:
        """
        Calculate confidence in volume signals based on data quality
        
        Factors:
        - Zero volume days (lower confidence)
        - Volume consistency (higher confidence)
        
        Returns: Confidence score [0, 1]
        """
        if len(volume) < self.lookback:
            return 0.5
        
        recent_volume = volume.iloc[-self.lookback:]
        
        # Count zero volume days
        zero_days = (recent_volume == 0).sum()
        zero_penalty = zero_days / self.lookback
        
        # Calculate volume consistency (inverse of coefficient of variation)
        mean_vol = recent_volume.mean()
        std_vol = recent_volume.std()
        
        if mean_vol == 0 or np.isnan(mean_vol) or np.isnan(std_vol):
            consistency = 0.5
        else:
            cv = std_vol / mean_vol  # Coefficient of variation
            consistency = 1.0 / (1.0 + cv)  # Inverse relationship
        
        # Combined confidence
        confidence = (1.0 - zero_penalty) * 0.6 + consistency * 0.4
        
        return max(0.0, min(1.0, confidence))
    
    def _empty_result(self) -> Dict:
        """Return empty result for insufficient data"""
        return {
            'spike_score': 0.0,
            'price_volume_correlation': 0.0,
            'accumulation_distribution': 'neutral',
            'volume_confidence': 0.0
        }


# Test function
if __name__ == "__main__":
    print("Testing Volume Intelligence...")
    
    # Generate test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252)
    prices = pd.Series(50 + np.random.randn(252).cumsum() * 0.5, index=dates)
    volume = pd.Series(1000000 + np.random.randint(-200000, 200000, 252), index=dates)
    
    # Test Volume Intelligence
    vi = VolumeIntelligence()
    result = vi.analyze_volume(prices, volume)
    
    print(f"\nVolume Intelligence Results:")
    print(f"  Spike Score: {result['spike_score']:.1f}/100")
    print(f"  Price-Volume Correlation: {result['price_volume_correlation']:.3f}")
    print(f"  Accumulation/Distribution: {result['accumulation_distribution']}")
    print(f"  Volume Confidence: {result['volume_confidence']:.3f}")
    
    # Test with volume spike
    print(f"\nTesting with volume spike...")
    volume_spike = volume.copy()
    volume_spike.iloc[-1] = volume.mean() * 3  # 3x average volume
    
    result_spike = vi.analyze_volume(prices, volume_spike)
    print(f"  Spike Score: {result_spike['spike_score']:.1f}/100 (should be higher)")
    
    print("\n✅ Volume Intelligence implementation complete!")

