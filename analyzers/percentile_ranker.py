"""
Percentile-Based ETF Ranking System
Ranks ETFs within risk categories using 252-day rolling percentiles
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import json
from pathlib import Path


class PercentileRanker:
    """
    Historical percentile-based ranking system for ETF analysis.

    Key Features:
    - 252-day rolling percentile calculation (stability across time)
    - Each ETF compared to its own historical distribution
    - Automatic metric inversion for "lower is better" metrics
    - Risk category isolation (LOW/MEDIUM/HIGH ranked separately)
    - Equal weighting initially (weights_config.json for future customization)
    """

    def __init__(self, lookback_days: int = 252, weights_config_path: str = None):
        """
        Initialize percentile ranker.

        Args:
            lookback_days: Rolling window for percentile calculation (default: 252 = 1 year)
            weights_config_path: Path to weights configuration JSON
        """
        self.lookback_days = lookback_days
        self.minimum_history = 60  # Minimum data points required

        # Metrics where LOWER values are better (will be inverted)
        # Only cvar remains from validated factors
        self.inverted_metrics = {
            'cvar': True,               # Lower CVaR (risk) is better - VALIDATED
        }
        
        # Validated factors (statistically significant p < 0.05, positive IC)
        self.validated_factors = [
            'ml_forecast',              # IC=+0.229, p=0.027
            'hit_rate',                 # IC=+0.344, p=0.001
            'kalman_signal_strength',   # IC=+0.234, p=0.023
            'cvar',                     # IC=+0.261, p=0.011 (inverted)
        ]

        # Load weights configuration
        self.weights = self._load_weights_config(weights_config_path)
        self.weighting_mode = self.weights.get('weighting_mode', 'equal')

    def _load_weights_config(self, config_path: str = None) -> Dict:
        """Load weights configuration from JSON file"""
        if config_path is None:
            config_path = 'config/weights_config.json'

        config_file = Path(config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load weights config: {e}. Using defaults.")

        # Default equal weighting
        return {
            'weighting_mode': 'equal',
            'description': 'Equal weighting (all factors 1.0)',
            'factor_weights': {}
        }

    def calculate_percentile(self, current_value: float, historical_values: pd.Series,
                            metric_name: str) -> float:
        """
        Calculate percentile rank within historical distribution.

        Args:
            current_value: Current metric value
            historical_values: Series of historical values (252-day window)
            metric_name: Name of metric (used to determine inversion)

        Returns:
            Percentile rank (0-100)
        """
        if len(historical_values) < self.minimum_history:
            return np.nan

        # Calculate raw percentile (0-100)
        try:
            # Handle NaN values in historical data
            clean_historical = historical_values.dropna()
            if len(clean_historical) < self.minimum_history:
                return np.nan

            # Check if current value is valid
            if pd.isna(current_value):
                return np.nan

            # Calculate percentile (0-100)
            percentile = stats.percentileofscore(clean_historical.values, current_value, kind='rank')

            # Invert if metric is "lower is better"
            if self._should_invert(metric_name):
                percentile = 100 - percentile

            return np.clip(percentile, 0, 100)

        except Exception as e:
            print(f"Error calculating percentile for {metric_name}: {e}")
            return np.nan

    def _calculate_cross_sectional_percentile(
        self,
        current_value: float,
        all_category_values: list,
        metric_name: str
    ) -> float:
        """
        Calculate percentile by comparing to other ETFs in same risk category.

        Used when historical data is unavailable. Compares current ETF's metric
        value against all other ETFs in the same risk category.

        Args:
            current_value: ETF's current metric value
            all_category_values: All values for this metric across risk category
            metric_name: Name of metric (for inversion logic)

        Returns:
            Percentile rank (0-100), or NaN if calculation fails
        """
        try:
            if pd.isna(current_value):
                return np.nan

            # Filter out NaN values
            valid_values = [v for v in all_category_values if not pd.isna(v)]

            if len(valid_values) == 0:
                return np.nan

            # Calculate percentile using scipy
            from scipy import stats
            percentile = stats.percentileofscore(valid_values, current_value, kind='rank')

            # Apply inversion for "lower is better" metrics
            if self._should_invert(metric_name):
                percentile = 100 - percentile

            return np.clip(percentile, 0, 100)

        except Exception as e:
            print(f"Error calculating cross-sectional percentile for {metric_name}: {e}")
            return np.nan

    def _should_invert(self, metric_name: str) -> bool:
        """Determine if metric should be inverted (lower = better)"""
        return self.inverted_metrics.get(metric_name, False)

    def rank_etf(self, etf_metrics: Dict, metric_names: List[str],
                 etf_history: Dict[str, pd.Series] = None,
                 category_metrics: Dict[str, list] = None) -> Dict:
        """
        Calculate average percentile score for a single ETF.

        Tries historical percentiles first (preferred), then falls back to
        cross-sectional percentiles if historical data unavailable.

        Args:
            etf_metrics: Dict of current metric values for this ETF
            metric_names: List of metric names to include in ranking
            etf_history: Dict of {metric_name: pd.Series} with historical values (optional)
            category_metrics: Dict of {metric_name: [all values in category]} for cross-sectional (optional)

        Returns:
            Dict with composite_percentile, individual percentiles, and metadata
        """
        percentiles = {}
        valid_metrics = 0

        for metric_name in metric_names:
            if metric_name not in etf_metrics:
                continue

            current_value = etf_metrics[metric_name]

            if pd.isna(current_value):
                continue

            percentile = np.nan

            # Try historical percentile first (preferred)
            if etf_history is not None and metric_name in etf_history:
                historical_values = etf_history[metric_name]

                # Get recent 252-day window
                if len(historical_values) >= self.minimum_history:
                    recent_window = historical_values.tail(self.lookback_days)
                    percentile = self.calculate_percentile(current_value, recent_window, metric_name)

            # Fallback to cross-sectional percentile if historical unavailable
            if pd.isna(percentile) and category_metrics is not None and metric_name in category_metrics:
                percentile = self._calculate_cross_sectional_percentile(
                    current_value,
                    category_metrics[metric_name],
                    metric_name
                )

            # Store percentile if valid
            if not pd.isna(percentile):
                percentiles[metric_name] = percentile
                valid_metrics += 1

        # Calculate composite score (simple average or weighted)
        if valid_metrics > 0:
            if self.weighting_mode == 'equal':
                composite_percentile = np.mean(list(percentiles.values()))
            else:
                # Weighted average
                composite_percentile = self._weighted_average(percentiles, metric_names)
        else:
            composite_percentile = np.nan

        return {
            'composite_percentile': composite_percentile,
            'individual_percentiles': percentiles,
            'num_factors_used': valid_metrics,
            'num_factors_available': len(percentiles)
        }

    def _weighted_average(self, percentiles: Dict[str, float], metric_names: List[str]) -> float:
        """Calculate weighted average of percentiles"""
        weights_dict = self.weights.get('factor_weights', {})

        total_weight = 0
        weighted_sum = 0

        for metric_name, percentile in percentiles.items():
            weight = weights_dict.get(metric_name, 1.0)
            if not pd.isna(percentile):
                weighted_sum += percentile * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        return np.nan

    def rank_etf_universe(self, etf_analysis_results: Dict[str, Dict],
                         risk_categories: Dict[str, List[str]],
                         metric_names: List[str],
                         etf_history_data: Dict[str, Dict[str, pd.Series]] = None) -> Dict:
        """
        Rank entire ETF universe within risk categories.

        Args:
            etf_analysis_results: Dict of {ticker: {metric_name: value}}
            risk_categories: Dict of {risk_level: [list of tickers]}
            metric_names: List of metric names to use in ranking
            etf_history_data: Dict of {ticker: {metric_name: pd.Series}}

        Returns:
            Dict of {risk_category: ranked_results}
        """
        ranked_results = {}

        for risk_level, tickers in risk_categories.items():
            print(f"Ranking {len(tickers)} ETFs in {risk_level} category...")

            # Build cross-sectional metric data for this risk category
            category_metrics = {}
            for metric_name in metric_names:
                category_metrics[metric_name] = [
                    etf_analysis_results[ticker].get(metric_name)
                    for ticker in tickers
                    if ticker in etf_analysis_results
                ]

            category_rankings = []

            for ticker in tickers:
                if ticker not in etf_analysis_results:
                    continue

                etf_metrics = etf_analysis_results[ticker]
                etf_hist = None
                if etf_history_data and ticker in etf_history_data:
                    etf_hist = etf_history_data[ticker]

                # Calculate percentile scores (with fallback to cross-sectional)
                ranking = self.rank_etf(etf_metrics, metric_names, etf_hist, category_metrics)

                if not pd.isna(ranking['composite_percentile']):
                    category_rankings.append({
                        'ticker': ticker,
                        'composite_percentile': ranking['composite_percentile'],
                        'individual_percentiles': ranking['individual_percentiles'],
                        'num_factors': ranking['num_factors_used']
                    })

            # Sort by composite percentile (highest first)
            category_rankings.sort(key=lambda x: x['composite_percentile'], reverse=True)

            ranked_results[risk_level] = {
                'rankings': category_rankings,
                'count': len(category_rankings),
                'top_3': category_rankings[:3] if len(category_rankings) >= 3 else category_rankings
            }

            # Print summary
            if category_rankings:
                print(f"  Top 3 {risk_level} ETFs:")
                for i, rank in enumerate(category_rankings[:3], 1):
                    print(f"    {i}. {rank['ticker']}: {rank['composite_percentile']:.1f}th percentile")

        return ranked_results

    def apply_risk_filters(self, ranked_results: Dict, risk_filters: Dict = None) -> Dict:
        """
        Apply risk-based filtering to remove untradeable ETFs.

        Args:
            ranked_results: Output from rank_etf_universe
            risk_filters: Dict of {metric_name: {threshold, action}}
                Example: {'cvar': {'threshold': -20, 'action': 'remove'}}

        Returns:
            Filtered ranked results
        """
        if risk_filters is None:
            return ranked_results

        filtered_results = {}

        for risk_level, category_data in ranked_results.items():
            filtered_rankings = []
            removed_count = 0

            for rank in category_data['rankings']:
                ticker = rank['ticker']
                percentiles = rank['individual_percentiles']

                # Check if any filter threshold breached
                should_remove = False
                for metric_name, filter_config in risk_filters.items():
                    if metric_name in percentiles:
                        value = percentiles[metric_name]
                        threshold = filter_config.get('threshold', 10)

                        if value < threshold:
                            should_remove = True
                            removed_count += 1
                            break

                if not should_remove:
                    filtered_rankings.append(rank)

            filtered_results[risk_level] = {
                'rankings': filtered_rankings,
                'count': len(filtered_rankings),
                'removed': removed_count,
                'top_3': filtered_rankings[:3] if len(filtered_rankings) >= 3 else filtered_rankings
            }

            print(f"  {risk_level}: Removed {removed_count} ETFs (risk filters)")

        return filtered_results

    def export_rankings_to_csv(self, ranked_results: Dict, output_path: str = 'data/rankings_percentile.csv'):
        """
        Export rankings to CSV for analysis.

        Args:
            ranked_results: Output from rank_etf_universe
            output_path: CSV output path
        """
        all_rankings = []

        for risk_level, category_data in ranked_results.items():
            for rank_idx, rank in enumerate(category_data['rankings'], 1):
                row = {
                    'rank': rank_idx,
                    'risk_category': risk_level,
                    'ticker': rank['ticker'],
                    'composite_percentile': rank['composite_percentile'],
                    'num_factors': rank['num_factors'],
                    **{f'percentile_{k}': v for k, v in rank['individual_percentiles'].items()}
                }
                all_rankings.append(row)

        df = pd.DataFrame(all_rankings)
        df.to_csv(output_path, index=False)
        print(f"Rankings exported to {output_path}")
        return df

    @staticmethod
    def create_default_weights_config(output_path: str = 'config/weights_config.json'):
        """
        Create default weights configuration file.

        Args:
            output_path: Path to save config file
        """
        config = {
            "weighting_mode": "equal",
            "description": "Factor weights for percentile ranking. 'equal' = simple average, 'custom' = weighted average",
            "factor_weights": {
                "ml_forecast": 1.0,
                "ml_confidence": 1.0,
                "hit_rate": 1.0,
                "kalman_signal_strength": 1.0,
                "kalman_efficiency_ratio": 1.0,
                "volume_correlation": 1.0,
                "volume_spike_score": 1.0
            },
            "notes": [
                "All factors equally weighted (1.0)",
                "To use custom weights: set weighting_mode to 'custom' and adjust factor_weights",
                "Factors not listed will use default weight of 1.0",
                "Remove factors from list to exclude them from ranking"
            ]
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)

        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Default weights config created at {output_path}")
        return config
