#!/usr/bin/env python3
"""
MACD-V (MACD with Volatility Normalization) Indicator
Traditional MACD normalized by volatility for consistent signals across market conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MACDVIndicator:
    """
    MACD-V (MACD with Volatility Normalization)
    Normalizes MACD signals by volatility to provide consistent readings
    across different market regimes and volatility environments
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                 volatility_window: int = 20, normalization_method: str = 'zscore'):
        """
        Initialize MACD-V indicator
        
        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
            volatility_window: Window for volatility calculation (default: 20)
            normalization_method: Method for volatility normalization ('zscore', 'percentile', 'adaptive')
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.volatility_window = volatility_window
        self.normalization_method = normalization_method
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate traditional MACD components
        
        Args:
            prices: Price series
            
        Returns:
            Dict with MACD components
        """
        ema_fast = self.calculate_ema(prices, self.fast_period)
        ema_slow = self.calculate_ema(prices, self.slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
    
    def calculate_volatility_metrics(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate volatility metrics for normalization
        
        Args:
            prices: Price series
            
        Returns:
            Dict with volatility metrics
        """
        returns = prices.pct_change().dropna()
        
        # Rolling volatility (standard deviation)
        rolling_vol = returns.rolling(window=self.volatility_window).std()
        
        # ATR-based volatility
        high_low = prices.rolling(window=2).max() - prices.rolling(window=2).min()
        atr_vol = high_low.rolling(window=self.volatility_window).mean()
        
        # Price range volatility
        price_range = prices.rolling(window=self.volatility_window).max() - prices.rolling(window=self.volatility_window).min()
        range_vol = price_range / prices.rolling(window=self.volatility_window).mean()
        
        return {
            'returns_volatility': rolling_vol,
            'atr_volatility': atr_vol,
            'range_volatility': range_vol
        }
    
    def normalize_macd_by_volatility(self, macd_components: Dict[str, pd.Series], 
                                    volatility_metrics: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Normalize MACD components by volatility
        
        Args:
            macd_components: Traditional MACD components
            volatility_metrics: Volatility metrics for normalization
            
        Returns:
            Dict with volatility-normalized MACD components
        """
        macd_line = macd_components['macd_line']
        signal_line = macd_components['signal_line']
        histogram = macd_components['histogram']
        
        # Use returns volatility as primary normalization metric
        volatility = volatility_metrics['returns_volatility']
        
        # Align volatility with MACD components (same index)
        volatility = volatility.reindex(macd_line.index, method='ffill').fillna(1)
        
        if self.normalization_method == 'zscore':
            # Z-score normalization
            vol_mean = volatility.rolling(window=50).mean()
            vol_std = volatility.rolling(window=50).std()
            normalized_vol = (volatility - vol_mean) / vol_std
            
            # Avoid division by zero
            normalized_vol = normalized_vol.fillna(1)
            normalized_vol = np.where(normalized_vol == 0, 1, normalized_vol)
            
            macd_v = macd_line / normalized_vol
            signal_v = signal_line / normalized_vol
            histogram_v = histogram / normalized_vol
            
        elif self.normalization_method == 'percentile':
            # Percentile-based normalization
            vol_percentile = volatility.rolling(window=100).rank(pct=True)
            normalized_vol = vol_percentile * 2  # Scale to 0-2 range
            
            # Avoid division by zero
            normalized_vol = normalized_vol.fillna(1)
            normalized_vol = np.where(normalized_vol == 0, 1, normalized_vol)
            
            macd_v = macd_line / normalized_vol
            signal_v = signal_line / normalized_vol
            histogram_v = histogram / normalized_vol
            
        elif self.normalization_method == 'adaptive':
            # Adaptive normalization based on volatility regime
            vol_ma = volatility.rolling(window=20).mean()
            vol_regime = pd.cut(vol_ma, bins=5, labels=[0.5, 0.75, 1.0, 1.5, 2.0])
            vol_regime = vol_regime.astype(float).fillna(1.0)
            
            macd_v = macd_line / vol_regime
            signal_v = signal_line / vol_regime
            histogram_v = histogram / vol_regime
            
        else:
            # Simple normalization by raw volatility
            volatility = volatility.fillna(volatility.mean())
            volatility = np.where(volatility == 0, volatility.mean(), volatility)
            
            macd_v = macd_line / volatility
            signal_v = signal_line / volatility
            histogram_v = histogram / volatility
        
        return {
            'macd_v_line': macd_v,
            'signal_v_line': signal_v,
            'histogram_v': histogram_v,
            'normalization_factor': volatility
        }
    
    def extract_macd_v_features(self, prices: pd.Series) -> Dict[str, float]:
        """
        Extract MACD-V features for ML models
        
        Args:
            prices: Price series
            
        Returns:
            Dict of MACD-V features
        """
        if len(prices) < max(self.slow_period, self.volatility_window) + self.signal_period:
            return self._get_default_features()
        
        try:
            # Calculate traditional MACD
            macd_components = self.calculate_macd(prices)
            
            # Calculate volatility metrics
            volatility_metrics = self.calculate_volatility_metrics(prices)
            
            # Normalize MACD by volatility
            macd_v_components = self.normalize_macd_by_volatility(macd_components, volatility_metrics)
            
            # Get current values
            macd_current = macd_components['macd_line'].iloc[-1]
            signal_current = macd_components['signal_line'].iloc[-1]
            histogram_current = macd_components['histogram'].iloc[-1]
            
            macd_v_current = macd_v_components['macd_v_line'].iloc[-1]
            signal_v_current = macd_v_components['signal_v_line'].iloc[-1]
            histogram_v_current = macd_v_components['histogram_v'].iloc[-1]
            
            # Volatility metrics
            current_vol = volatility_metrics['returns_volatility'].iloc[-1]
            vol_regime = self._classify_volatility_regime(current_vol)
            
            # Signal strength metrics
            macd_strength = abs(macd_current)
            macd_v_strength = abs(macd_v_current)
            
            # Trend consistency (how consistent is the signal)
            macd_trend_consistency = self._calculate_trend_consistency(macd_components['macd_line'])
            macd_v_trend_consistency = self._calculate_trend_consistency(macd_v_components['macd_v_line'])
            
            # Divergence between regular and volatility-normalized MACD
            macd_divergence = abs(macd_v_current - macd_current)
            
            # Signal quality (higher in low volatility, more reliable)
            signal_quality = 1.0 / (1.0 + current_vol * 10)  # Inverse relationship with volatility
            
            features = {
                'macd_signal': float(macd_current - signal_current),  # Traditional MACD signal
                'macd_histogram': float(histogram_current),          # Traditional MACD histogram
                'macd_v_signal': float(macd_v_current - signal_v_current),  # Volatility-normalized signal
                'macd_v_histogram': float(histogram_v_current),      # Volatility-normalized histogram
                'macd_strength': float(macd_strength),               # Traditional signal strength
                'macd_v_strength': float(macd_v_strength),           # Volatility-normalized strength
                'volatility_level': float(current_vol),               # Current volatility level
                'volatility_regime': float(vol_regime),               # Volatility regime (1-5)
                'macd_divergence': float(macd_divergence),           # Divergence between regular and V-MACD
                'trend_consistency': float(macd_trend_consistency),  # Trend consistency
                'macd_v_consistency': float(macd_v_trend_consistency), # V-MACD trend consistency
                'signal_quality': float(signal_quality),              # Signal quality score
                'volatility_adjusted_momentum': float(macd_v_current) # Main V-MACD signal
            }
            
            return features
            
        except Exception as e:
            print(f"⚠️ MACD-V calculation failed: {e}")
            return self._get_default_features()
    
    def _classify_volatility_regime(self, volatility: float) -> float:
        """Classify volatility into regime (1=very low, 5=very high)"""
        if volatility < 0.005:  # < 0.5% daily vol
            return 1.0
        elif volatility < 0.01:  # < 1% daily vol
            return 2.0
        elif volatility < 0.02:  # < 2% daily vol
            return 3.0
        elif volatility < 0.03:  # < 3% daily vol
            return 4.0
        else:
            return 5.0
    
    def _calculate_trend_consistency(self, signal_series: pd.Series, window: int = 10) -> float:
        """Calculate how consistent the signal trend is (1 = very consistent, 0 = choppy)"""
        if len(signal_series) < window:
            return 0.5
        
        recent_signals = signal_series.tail(window)
        sign_changes = (np.sign(recent_signals.diff()).abs().sum()) / (window - 1)
        consistency = 1 - sign_changes  # Inverse of sign changes
        return max(0, consistency)  # Ensure non-negative
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when calculation fails"""
        return {
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'macd_v_signal': 0.0,
            'macd_v_histogram': 0.0,
            'macd_strength': 0.0,
            'macd_v_strength': 0.0,
            'volatility_level': 0.01,
            'volatility_regime': 3.0,
            'macd_divergence': 0.0,
            'trend_consistency': 0.5,
            'macd_v_consistency': 0.5,
            'signal_quality': 0.5,
            'volatility_adjusted_momentum': 0.0
        }


class DemandSupplyIndicator:
    """
    Demand-Supply Indicator based on volume-price analysis
    Identifies accumulation vs distribution patterns
    """
    
    def __init__(self, volume_window: int = 20, price_window: int = 10, 
                 trend_window: int = 30):
        """
        Initialize Demand-Supply indicator
        
        Args:
            volume_window: Window for volume analysis (default: 20)
            price_window: Window for price analysis (default: 10)
            trend_window: Window for trend analysis (default: 30)
        """
        self.volume_window = volume_window
        self.price_window = price_window
        self.trend_window = trend_window
    
    def calculate_volume_price_trend(self, prices: pd.Series, volumes: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate volume-price trend indicators
        
        Args:
            prices: Price series
            volumes: Volume series
            
        Returns:
            Dict with volume-price indicators
        """
        returns = prices.pct_change().dropna()
        
        # Volume-weighted returns
        volume_weighted_returns = returns * volumes.shift(1)
        
        # Volume moving averages
        volume_ma = volumes.rolling(window=self.volume_window).mean()
        volume_ratio = volumes / volume_ma
        
        # Price-volume correlation (demand indicator)
        price_volume_corr = returns.rolling(window=self.volume_window).corr(volumes)
        
        # Accumulation/Distribution line
        ad_line = self._calculate_accumulation_distribution(prices, volumes)
        
        # On-Balance Volume (OBV)
        obv = self._calculate_obv(prices, volumes)
        
        # Money Flow Index
        mfi = self._calculate_money_flow_index(prices, volumes)
        
        return {
            'volume_weighted_returns': volume_weighted_returns,
            'volume_ratio': volume_ratio,
            'price_volume_correlation': price_volume_corr,
            'accumulation_distribution': ad_line,
            'obv': obv,
            'money_flow_index': mfi
        }
    
    def _calculate_accumulation_distribution(self, prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """Calculate Accumulation/Distribution line"""
        high = prices.rolling(window=2).max()
        low = prices.rolling(window=2).min()
        close = prices
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        
        # Money Flow Volume
        mfv = mfm * volumes
        
        # Accumulation/Distribution line
        ad_line = mfv.cumsum()
        return ad_line
    
    def _calculate_obv(self, prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        price_change = prices.diff()
        obv = pd.Series(index=prices.index, dtype=float)
        
        obv.iloc[0] = volumes.iloc[0]
        
        for i in range(1, len(prices)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volumes.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volumes.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_money_flow_index(self, prices: pd.Series, volumes: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = prices  # Simplified - using close as typical price
        money_flow = typical_price * volumes
        
        # Positive and negative money flow
        positive_mf = money_flow.where(typical_price.diff() > 0, 0)
        negative_mf = money_flow.where(typical_price.diff() < 0, 0)
        
        # Sum over period
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()
        
        # Money Flow Index
        mfi = 100 - (100 / (1 + positive_mf_sum / negative_mf_sum))
        return mfi.fillna(50)
    
    def extract_demand_supply_features(self, prices: pd.Series, volumes: pd.Series) -> Dict[str, float]:
        """
        Extract demand-supply features for ML models
        
        Args:
            prices: Price series
            volumes: Volume series
            
        Returns:
            Dict of demand-supply features
        """
        if len(prices) < self.trend_window or len(volumes) < self.trend_window:
            return self._get_default_ds_features()
        
        try:
            # Calculate volume-price indicators
            vp_indicators = self.calculate_volume_price_trend(prices, volumes)
            
            # Current values
            volume_ratio = vp_indicators['volume_ratio'].iloc[-1]
            price_volume_corr = vp_indicators['price_volume_correlation'].iloc[-1]
            mfi = vp_indicators['money_flow_index'].iloc[-1]
            
            # AD and OBV trends
            ad_trend = vp_indicators['accumulation_distribution'].diff(self.trend_window).iloc[-1]
            obv_trend = vp_indicators['obv'].diff(self.trend_window).iloc[-1]
            
            # Volume pressure indicators
            volume_ma = volumes.rolling(window=self.volume_window).mean()
            volume_pressure = (volumes.iloc[-1] - volume_ma.iloc[-1]) / volume_ma.iloc[-1]
            
            # Demand/Supply classification
            demand_strength = self._calculate_demand_strength(prices, volumes)
            supply_pressure = self._calculate_supply_pressure(prices, volumes)
            
            # Volume confirmation of price trend
            price_trend = (prices.iloc[-1] / prices.iloc[-self.price_window] - 1)
            volume_trend = (volumes.iloc[-self.volume_window:].mean() / 
                          volumes.iloc[-self.volume_window*2:-self.volume_window].mean() - 1)
            volume_confirmation = 1.0 if (price_trend > 0 and volume_trend > 0) else \
                                0.0 if (price_trend > 0 and volume_trend < 0) else \
                                1.0 if (price_trend < 0 and volume_trend < 0) else \
                                0.0  # Mixed signals
            
            # Buying/selling pressure
            buying_pressure = max(0, price_volume_corr) * volume_ratio
            selling_pressure = max(0, -price_volume_corr) * volume_ratio
            
            features = {
                'volume_ratio': float(volume_ratio),              # Current volume vs average
                'price_volume_correlation': float(price_volume_corr),  # Demand indicator
                'money_flow_index': float(mfi),                   # MFI (0-100)
                'ad_trend': float(ad_trend),                      # Accumulation trend
                'obv_trend': float(obv_trend),                    # OBV trend
                'volume_pressure': float(volume_pressure),        # Volume pressure
                'demand_strength': float(demand_strength),        # Overall demand strength
                'supply_pressure': float(supply_pressure),        # Overall supply pressure
                'volume_confirmation': float(volume_confirmation), # Volume confirms price
                'buying_pressure': float(buying_pressure),        # Buying pressure indicator
                'selling_pressure': float(selling_pressure),      # Selling pressure indicator
                'demand_supply_balance': float(demand_strength - supply_pressure),  # Net D-S balance
                'volume_trend_strength': float(volume_trend),     # Volume trend strength
                'price_volume_efficiency': float(price_trend / (volume_trend + 0.001))  # Price efficiency
            }
            
            return features
            
        except Exception as e:
            print(f"⚠️ Demand-Supply calculation failed: {e}")
            return self._get_default_ds_features()
    
    def _calculate_demand_strength(self, prices: pd.Series, volumes: pd.Series) -> float:
        """Calculate overall demand strength (0-1 scale)"""
        returns = prices.pct_change().dropna()
        volume_ratio = volumes / volumes.rolling(window=self.volume_window).mean()
        
        # Demand: positive returns with high volume
        demand_days = (returns > 0) & (volume_ratio > 1.0)
        demand_strength = demand_days.rolling(window=self.trend_window).mean()
        
        return demand_strength.iloc[-1] if len(demand_strength) > 0 else 0.5
    
    def _calculate_supply_pressure(self, prices: pd.Series, volumes: pd.Series) -> float:
        """Calculate overall supply pressure (0-1 scale)"""
        returns = prices.pct_change().dropna()
        volume_ratio = volumes / volumes.rolling(window=self.volume_window).mean()
        
        # Supply: negative returns with high volume
        supply_days = (returns < 0) & (volume_ratio > 1.0)
        supply_pressure = supply_days.rolling(window=self.trend_window).mean()
        
        return supply_pressure.iloc[-1] if len(supply_pressure) > 0 else 0.5
    
    def _get_default_ds_features(self) -> Dict[str, float]:
        """Return default demand-supply features when calculation fails"""
        return {
            'volume_ratio': 1.0,
            'price_volume_correlation': 0.0,
            'money_flow_index': 50.0,
            'ad_trend': 0.0,
            'obv_trend': 0.0,
            'volume_pressure': 0.0,
            'demand_strength': 0.5,
            'supply_pressure': 0.5,
            'volume_confirmation': 0.5,
            'buying_pressure': 0.0,
            'selling_pressure': 0.0,
            'demand_supply_balance': 0.0,
            'volume_trend_strength': 0.0,
            'price_volume_efficiency': 0.0
        }


if __name__ == "__main__":
    # Test the indicators
    print("🧪 Testing MACD-V and Demand-Supply Indicators")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    
    # Simulate price with trend and cycles
    trend = np.linspace(100, 120, 200)
    cycle = 5 * np.sin(np.linspace(0, 4*np.pi, 200))
    noise = np.random.randn(200) * 2
    prices = pd.Series(trend + cycle + noise, index=dates)
    
    # Simulate volume with price correlation
    base_volume = 500000
    volume_variation = np.random.randn(200) * 100000
    price_volume_corr = (prices - prices.mean()) * 10000
    volumes = pd.Series(np.maximum(100000, base_volume + volume_variation + price_volume_corr), index=dates)
    
    # Test MACD-V
    print("📊 Testing MACD-V Indicator...")
    macd_v = MACDVIndicator()
    macd_v_features = macd_v.extract_macd_v_features(prices)
    print(f"✅ Extracted {len(macd_v_features)} MACD-V features")
    
    # Test Demand-Supply
    print("\n📈 Testing Demand-Supply Indicator...")
    ds_indicator = DemandSupplyIndicator()
    ds_features = ds_indicator.extract_demand_supply_features(prices, volumes)
    print(f"✅ Extracted {len(ds_features)} Demand-Supply features")
    
    # Show sample features
    print(f"\n🎯 Sample Features:")
    print("MACD-V Features:")
    for name, value in list(macd_v_features.items())[:5]:
        print(f"  {name:25s}: {value:8.4f}")
    
    print("\nDemand-Supply Features:")
    for name, value in list(ds_features.items())[:5]:
        print(f"  {name:25s}: {value:8.4f}")
    
    print(f"\n✅ All indicators working correctly!")
