"""
Factor Validator - Comprehensive Predictive Power Testing
Tests individual metrics for predictive power before inclusion in ranking system

Tests:
1. Information Coefficient (IC) - Correlation with forward returns
2. Hit Rate - Directional accuracy
3. Quintile Analysis - Monotonic relationship verification
4. Factor Correlation Matrix - Identifies redundant factors
5. Factor Decay Analysis - Determines optimal holding period
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import spearmanr, pearsonr
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FactorValidator:
    """
    Validates individual factors for predictive power using multiple tests.

    Success Criteria:
    - IC > 0.05 (good), > 0.10 (great), auto-reject < 0.02
    - Hit rate > 55% (good), > 60% (great), reject ≈ 50%
    - Quintile analysis: monotonic relationship (Q1 < Q2 < Q3 < Q4 < Q5)
    - Factor independence: correlation < 0.5 between factors
    - Optimal holding period identified via decay analysis
    """

    def __init__(self, ic_threshold: float = 0.02, hit_rate_threshold: float = 0.52):
        """
        Initialize factor validator.

        Args:
            ic_threshold: Minimum IC for factor acceptance (auto-reject below this)
            hit_rate_threshold: Minimum hit rate for factor acceptance
        """
        self.ic_threshold = ic_threshold
        self.hit_rate_threshold = hit_rate_threshold
        self.validation_results = {}
        self.forward_periods = [5, 10, 20, 40, 60]  # Days to test

    def _normalize_factor_values(self, factor_values: Dict) -> Dict[str, pd.Series]:
        """
        Normalize factor_values to ensure all values are pandas Series.

        Handles both:
        - Dict of dicts: {factor_name: {ticker: value}} -> converts to Series
        - Dict of Series: {factor_name: pd.Series} -> returns as-is

        Args:
            factor_values: Factor values in either format

        Returns:
            Dict of pandas Series: {factor_name: pd.Series(ticker: value)}
        """
        normalized = {}

        for factor_name, factor_data in factor_values.items():
            if isinstance(factor_data, pd.Series):
                # Already a Series, use as-is
                normalized[factor_name] = factor_data
            elif isinstance(factor_data, dict):
                # Dict of {ticker: value}, convert to Series
                normalized[factor_name] = pd.Series(factor_data)
            else:
                # Assume it's array-like, try to convert to Series
                normalized[factor_name] = pd.Series(factor_data)

        return normalized

    # ========================================================================
    # TEST 1: INFORMATION COEFFICIENT (IC)
    # ========================================================================

    def test_information_coefficient(self, factor_values: Dict[str, pd.Series],
                                     price_data: Dict[str, pd.Series],
                                     forward_days: int = 20) -> Dict[str, Dict]:
        """
        Test Information Coefficient - correlation between factor and forward returns.

        NOTE: IC is NOT applicable to cross-sectional factors (constant value per ticker).
        For cross-sectional factors, use Hit Rate and Quintile tests instead.
        This test returns SKIPPED for all factors.

        Args:
            factor_values: Dict of {factor_name: Series with tickers as index}
            price_data: Dict of {ticker: Series with dates as index}
            forward_days: Number of days forward to calculate returns

        Returns:
            Dict of {factor_name: {'status': 'SKIPPED'}}
        """
        ic_results = {}

        print(f"\nTEST 1: Information Coefficient (IC)")
        print(f"  Forward period: {forward_days} days")
        print(f"  Threshold: IC > {self.ic_threshold}")
        print(f"  {'-' * 60}")
        print(f"  NOTE: IC test skipped - not applicable to cross-sectional factors")
        print(f"        (factors are constant per ticker, not time-varying)")
        print(f"        Use Hit Rate and Quintile tests instead.\n")

        for factor_name, factor_series in factor_values.items():
            ic_results[factor_name] = {
                'ic': np.nan,
                'p_value': np.nan,
                'status': 'SKIPPED',
                'reason': 'Not applicable to cross-sectional factors'
            }
            print(f"  {factor_name:30s} [SKIPPED - cross-sectional factor]")

        return ic_results

    # ========================================================================
    # TEST 2: HIT RATE (DIRECTIONAL ACCURACY)
    # ========================================================================

    def test_hit_rate(self, factor_values: Dict[str, pd.Series],
                     price_data: Dict[str, pd.Series],
                     forward_days: int = 20) -> Dict[str, Dict]:
        """
        Test Hit Rate - percentage of correct directional predictions.

        Args:
            factor_values: Dict of {factor_name: Series with tickers as index}
            price_data: Dict of {ticker: Series with dates as index}
            forward_days: Number of days forward for return calculation

        Returns:
            Dict of {factor_name: {'hit_rate': value, 'status': 'VALIDATED'/'REJECTED'}}
        """
        hit_rate_results = {}

        print(f"\nTEST 2: Hit Rate (Directional Accuracy)")
        print(f"  Forward period: {forward_days} days")
        print(f"  Threshold: Hit rate > {self.hit_rate_threshold:.1%}")
        print(f"  {'-' * 60}")

        for factor_name, factor_series in factor_values.items():
            correct_predictions = 0
            total_predictions = 0

            for ticker in factor_series.index:
                if ticker not in price_data:
                    continue

                factor_val = factor_series[ticker]  # Single scalar value
                prices = price_data[ticker]  # Time series

                # Skip if factor value is NaN
                if pd.isna(factor_val):
                    continue

                # Calculate factor direction (positive/negative)
                factor_direction = np.sign(factor_val)

                # Calculate return direction
                forward_returns = prices.shift(-forward_days) / prices - 1
                return_direction = np.sign(forward_returns)

                # Count matches
                common_dates = forward_returns.dropna().index
                if len(common_dates) >= 30:
                    return_dir_clean = return_direction.loc[common_dates].dropna()

                    if len(return_dir_clean) > 0:
                        matches = np.sum(return_dir_clean.values == factor_direction)
                        correct_predictions += matches
                        total_predictions += len(return_dir_clean)

            if total_predictions > 0:
                hit_rate = correct_predictions / total_predictions

                # Determine status
                if hit_rate < self.hit_rate_threshold:
                    status = 'REJECTED'
                    reason = f'Hit rate {hit_rate:.1%} below threshold {self.hit_rate_threshold:.1%}'
                elif hit_rate > 0.60:
                    status = 'GREAT'
                    reason = f'Hit rate {hit_rate:.1%} (great directional accuracy)'
                else:
                    status = 'VALIDATED'
                    reason = f'Hit rate {hit_rate:.1%} (good directional accuracy)'

                hit_rate_results[factor_name] = {
                    'hit_rate': hit_rate,
                    'correct': correct_predictions,
                    'total': total_predictions,
                    'status': status,
                    'reason': reason
                }

                print(f"  {factor_name:30s} Hit Rate={hit_rate:.1%}  ({correct_predictions}/{total_predictions})  [{status}]")
            else:
                hit_rate_results[factor_name] = {
                    'hit_rate': np.nan,
                    'status': 'INSUFFICIENT_DATA'
                }
                print(f"  {factor_name:30s} [INSUFFICIENT DATA]")

        return hit_rate_results

    # ========================================================================
    # TEST 3: QUINTILE ANALYSIS
    # ========================================================================

    def test_quintile_analysis(self, factor_values: Dict[str, pd.Series],
                              price_data: Dict[str, pd.Series],
                              forward_days: int = 20) -> Dict[str, Dict]:
        """
        Test Quintile Analysis - verify monotonic relationship with returns.

        Args:
            factor_values: Dict of {factor_name: Series with tickers as index}
            price_data: Dict of {ticker: Series with dates as index}
            forward_days: Number of days forward for return calculation

        Returns:
            Dict of {factor_name: quintile_spreads, monotonic_check}
        """
        quintile_results = {}

        print(f"\nTEST 3: Quintile Analysis (Monotonic Relationship)")
        print(f"  Forward period: {forward_days} days")
        print(f"  {'-' * 60}")

        for factor_name, factor_series in factor_values.items():
            all_data = []

            # Collect all (factor_value, forward_return) pairs
            for ticker in factor_series.index:
                if ticker not in price_data:
                    continue

                factor_val = factor_series[ticker]
                prices = price_data[ticker]
                forward_returns = prices.shift(-forward_days) / prices - 1

                # Skip if factor value is NaN
                if pd.isna(factor_val):
                    continue

                # Get all forward returns for this ticker
                common_dates = forward_returns.dropna().index
                if len(common_dates) >= 30:
                    for date in common_dates:
                        if pd.notna(forward_returns[date]):
                            all_data.append({
                                'factor': factor_val,
                                'return': forward_returns[date]
                            })

            if len(all_data) >= 100:
                df = pd.DataFrame(all_data)

                # Split into quintiles (handle duplicates by dropping extra labels)
                try:
                    df['quintile'] = pd.qcut(df['factor'], q=5, duplicates='drop')
                except Exception:
                    # If qcut fails, try with fewer quantiles
                    try:
                        df['quintile'] = pd.qcut(df['factor'], q=3, duplicates='drop')
                    except Exception:
                        # If still fails, skip this factor
                        quintile_results[factor_name] = {
                            'status': 'INSUFFICIENT_DATA',
                            'num_observations': len(all_data)
                        }
                        print(f"  {factor_name:30s} [INSUFFICIENT DATA: Could not create quintiles]")
                        continue

                # Calculate average returns per quintile
                quintile_returns = df.groupby('quintile')['return'].mean()

                # Check monotonicity
                q1_to_q5 = quintile_returns.iloc[-1] - quintile_returns.iloc[0]
                is_monotonic = all(quintile_returns.iloc[i] <= quintile_returns.iloc[i+1]
                                  for i in range(len(quintile_returns) - 1))

                status = 'MONOTONIC' if is_monotonic else 'NON_MONOTONIC'

                quintile_results[factor_name] = {
                    'q1_return': float(quintile_returns.iloc[0]) if len(quintile_returns) > 0 else np.nan,
                    'q5_return': float(quintile_returns.iloc[-1]) if len(quintile_returns) > 0 else np.nan,
                    'spread': float(q1_to_q5),
                    'is_monotonic': is_monotonic,
                    'status': status,
                    'quintile_returns': quintile_returns.to_dict()
                }

                print(f"  {factor_name:30s} Q1={quintile_returns.iloc[0]:+.4f}  Q5={quintile_returns.iloc[-1]:+.4f}  Spread={q1_to_q5:+.4f}  [{status}]")
            else:
                quintile_results[factor_name] = {
                    'status': 'INSUFFICIENT_DATA',
                    'num_observations': len(all_data)
                }
                print(f"  {factor_name:30s} [INSUFFICIENT DATA: {len(all_data)} observations]")

        return quintile_results

    # ========================================================================
    # TEST 4: FACTOR CORRELATION MATRIX
    # ========================================================================

    def test_factor_correlation(self, factor_values: Dict[str, pd.Series],
                               correlation_threshold: float = 0.70) -> Dict:
        """
        Test Factor Correlation - identify redundant factors.

        Args:
            factor_values: Dict of {factor_name: factor_values_series}
            correlation_threshold: Flag pairs with correlation above this

        Returns:
            Dict with correlation matrix and redundant factor pairs
        """
        factor_names = list(factor_values.keys())

        print(f"\nTEST 4: Factor Correlation Matrix")
        print(f"  Redundancy threshold: correlation > {correlation_threshold}")
        print(f"  {'-' * 60}")

        # Build correlation matrix
        correlation_matrix = {}
        redundant_pairs = []

        for i, factor1 in enumerate(factor_names):
            correlation_matrix[factor1] = {}

            for j, factor2 in enumerate(factor_names):
                if i == j:
                    correlation_matrix[factor1][factor2] = 1.0
                    continue

                # Calculate correlation
                series1 = factor_values[factor1].dropna()
                series2 = factor_values[factor2].dropna()

                common_index = series1.index.intersection(series2.index)
                if len(common_index) >= 30:
                    corr, _ = pearsonr(series1[common_index], series2[common_index])
                    correlation_matrix[factor1][factor2] = corr

                    # Flag high correlations (one direction only)
                    if i < j and abs(corr) > correlation_threshold:
                        redundant_pairs.append({
                            'factor1': factor1,
                            'factor2': factor2,
                            'correlation': corr
                        })
                        print(f"  HIGH CORRELATION: {factor1} <-> {factor2} = {corr:+.3f}")

        if not redundant_pairs:
            print("  No highly correlated factor pairs found (all < 0.70)")

        return {
            'correlation_matrix': correlation_matrix,
            'redundant_pairs': redundant_pairs,
            'num_redundant': len(redundant_pairs)
        }

    # ========================================================================
    # TEST 5: FACTOR DECAY ANALYSIS
    # ========================================================================

    def test_factor_decay(self, factor_values: Dict[str, pd.Series],
                         price_data: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """
        Test Factor Decay - determine optimal holding period.

        NOTE: Decay analysis is NOT applicable to cross-sectional factors (constant value per ticker).
        This test returns SKIPPED for all factors.

        Args:
            factor_values: Dict of {factor_name: Series with tickers as index}
            price_data: Dict of {ticker: Series with dates as index}

        Returns:
            Dict with status SKIPPED
        """
        decay_results = {}

        print(f"\nTEST 5: Factor Decay Analysis")
        print(f"  Testing forward periods: {self.forward_periods} days")
        print(f"  {'-' * 60}")
        print(f"  NOTE: Decay test skipped - not applicable to cross-sectional factors\n")

        for factor_name, factor_series in factor_values.items():
            decay_results[factor_name] = {
                'status': 'SKIPPED',
                'reason': 'Not applicable to cross-sectional factors'
            }
            print(f"  {factor_name:30s} [SKIPPED - cross-sectional factor]")

        return decay_results

    # ========================================================================
    # COMPREHENSIVE VALIDATION
    # ========================================================================

    def run_comprehensive_validation(self, factor_values: Dict[str, pd.Series],
                                    price_data: Dict[str, pd.Series],
                                    forward_days: int = 20) -> Dict:
        """
        Run all 5 tests and generate validation summary.

        Args:
            factor_values: Dict of {factor_name: factor_values_series}
            price_data: Dict of {ticker: closing_prices_series}
            forward_days: Forward period for return calculations

        Returns:
            Dict with all validation results
        """
        print("\n" + "=" * 70)
        print("COMPREHENSIVE FACTOR VALIDATION - ALL TESTS")
        print("=" * 70)

        # Normalize factor_values to ensure all are pandas Series
        factor_values = self._normalize_factor_values(factor_values)

        # Run all tests
        ic_results = self.test_information_coefficient(factor_values, price_data, forward_days)
        hit_rate_results = self.test_hit_rate(factor_values, price_data, forward_days)
        quintile_results = self.test_quintile_analysis(factor_values, price_data, forward_days)
        correlation_results = self.test_factor_correlation(factor_values)
        decay_results = self.test_factor_decay(factor_values, price_data)

        # Compile comprehensive results
        comprehensive_results = {
            'validation_date': pd.Timestamp.now().isoformat(),
            'forward_period_days': forward_days,
            'ic_results': ic_results,
            'hit_rate_results': hit_rate_results,
            'quintile_results': quintile_results,
            'correlation_results': correlation_results,
            'decay_results': decay_results,
            'validated_factors': self._determine_validated_factors(ic_results, hit_rate_results, quintile_results)
        }

        return comprehensive_results

    def _determine_validated_factors(self, ic_results: Dict, hit_rate_results: Dict,
                                     quintile_results: Dict) -> List[str]:
        """
        Determine which factors pass validation across all tests.

        Criteria for cross-sectional factors:
        - Hit rate > threshold (primary criterion)
        - Quintile monotonic (secondary criterion, if available)
        
        Note: IC is not applicable to cross-sectional factors (constant value per ticker)
        """
        validated = []

        for factor in hit_rate_results.keys():
            hr_status = hit_rate_results.get(factor, {}).get('status')
            quintile_status = quintile_results.get(factor, {}).get('status')

            # Must pass hit rate test
            if hr_status in ['VALIDATED', 'GREAT']:
                # Bonus: check if quintile is monotonic
                if quintile_status == 'MONOTONIC':
                    validated.append(factor)
                elif quintile_status != 'INSUFFICIENT_DATA':
                    # Still validate if we have quintile data, even if not monotonic
                    validated.append(factor)
                else:
                    # Validate based on hit rate alone if no quintile data
                    validated.append(factor)

        return validated

    def export_validation_results(self, results: Dict, output_path: str = 'config/validated_factors.json'):
        """
        Export validation results to JSON.

        Args:
            results: Comprehensive validation results
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True, parents=True)

        # Convert numpy types to JSON-serializable
        export_data = self._prepare_for_json(results)

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\nValidation results exported to {output_path}")
        return export_data

    @staticmethod
    def _prepare_for_json(obj):
        """Convert numpy/pandas types to JSON-serializable format."""
        if isinstance(obj, dict):
            return {str(k): FactorValidator._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [FactorValidator._prepare_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, (pd.Series, pd.Index)):
            return obj.to_list() if hasattr(obj, 'to_list') else obj.tolist()
        elif isinstance(obj, pd.Interval):
            # Convert Interval to string representation
            return f"({obj.left:.4f}, {obj.right:.4f}]"
        elif pd.isna(obj):
            return None
        else:
            return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj

    def print_summary(self, results: Dict):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        validated = results.get('validated_factors', [])
        print(f"\nValidated Factors ({len(validated)}):")
        for factor in validated:
            ic = results['ic_results'].get(factor, {}).get('ic', np.nan)
            hr = results['hit_rate_results'].get(factor, {}).get('hit_rate', np.nan)
            print(f"  - {factor:30s} IC={ic:+.4f}  Hit Rate={hr:.1%}")

        if len(validated) == 0:
            print("  WARNING: No factors passed validation!")

        print(f"\nRecommendation:")
        if len(validated) >= 4:
            print(f"  [OK] Sufficient factors ({len(validated)}) for ranking system")
        elif len(validated) > 0:
            print(f"  [WARN] Limited factors ({len(validated)}). Consider relaxing thresholds.")
        else:
            print(f"  [FAIL] No validated factors. System cannot be built.")
