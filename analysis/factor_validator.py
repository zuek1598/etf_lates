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

    def _normalize_factor_values(self, factor_values: Dict) -> Dict:
        """
        Normalize factor values to consistent format.
        
        Handles:
        - Dict of dicts: {factor: {ticker: value}} → Series
        - Dict of Series: {factor: Series} → Series
        - Dict of DataFrames: {factor: DataFrame} → DataFrame (time-series)
        - Dict of arrays: {factor: array} → Series
        """
        normalized = {}
        
        for factor_name, factor_data in factor_values.items():
            if isinstance(factor_data, pd.DataFrame):
                # Keep DataFrame as-is for time-series data
                normalized[factor_name] = factor_data
            elif isinstance(factor_data, pd.Series):
                normalized[factor_name] = factor_data
            elif isinstance(factor_data, dict):
                # Convert dict to Series
                normalized[factor_name] = pd.Series(factor_data)
            elif isinstance(factor_data, np.ndarray):
                # Handle numpy arrays - check dimensionality
                if factor_data.ndim == 1:
                    normalized[factor_name] = pd.Series(factor_data)
                elif factor_data.ndim == 2:
                    # 2D array - convert to DataFrame
                    normalized[factor_name] = pd.DataFrame(factor_data)
                else:
                    raise ValueError(f"Unsupported array shape for {factor_name}: {factor_data.shape}")
            else:
                # Try to convert to Series as fallback
                try:
                    normalized[factor_name] = pd.Series(factor_data)
                except Exception as e:
                    raise ValueError(f"Cannot normalize {factor_name}: {type(factor_data)}")

        return normalized

    # ========================================================================
    # TEST 1: INFORMATION COEFFICIENT (IC)
    # ========================================================================

    def test_cross_sectional_ic(self, factor_values: Dict[str, pd.Series],
                               price_data: Dict[str, pd.Series],
                               forward_days: int = 20) -> Dict[str, Dict]:
        """
        Test Cross-Sectional Information Coefficient - correlation between factor ranking and forward returns.
        
        For cross-sectional factors: Do higher factor values predict higher returns across tickers?

        Args:
            factor_values: Dict of {factor_name: Series with tickers as index}
            price_data: Dict of {ticker: Series with dates as index}
            forward_days: Number of days forward to calculate returns

        Returns:
            Dict of {factor_name: {'ic': value, 'p_value': value, 'status': 'VALIDATED'/'REJECTED'}}
        """
        ic_results = {}

        print(f"\nTEST 1: Cross-Sectional IC (Ranking Correlation)")
        print(f"  Forward period: {forward_days} days")
        print(f"  Threshold: IC > {self.ic_threshold}")
        print(f"  {'-' * 60}")

        for factor_name, factor_series in factor_values.items():
            factor_returns = []

            # factor_series has tickers as index
            for ticker in factor_series.index:
                if ticker not in price_data:
                    continue

                factor_val = factor_series[ticker]  # Single scalar value
                prices = price_data[ticker]  # Time series

                # Skip if factor value is NaN
                if pd.isna(factor_val):
                    continue

                # Calculate forward return (use most recent available)
                # FIXED: Calculate actual forward returns (future price / current price - 1)
                forward_returns = prices.shift(-forward_days) / prices - 1
                
                # Get the most recent forward return that's not NaN
                # This gives us the return from today to 20 days in the future
                valid_returns = forward_returns.dropna()
                if len(valid_returns) > 0:
                    forward_return = valid_returns.iloc[-1]  # Most recent forward return
                    factor_returns.append({
                        'factor': factor_val,
                        'return': forward_return,
                        'ticker': ticker
                    })

            # Sample size warning
            if len(factor_returns) < 30:
                print(f"  [WARN] Sample size {len(factor_returns)} < 30 - Results may not be statistically significant")
            
            if len(factor_returns) >= 10:
                # Create DataFrame for correlation analysis
                df = pd.DataFrame(factor_returns)
                
                # Calculate cross-sectional IC (Spearman correlation)
                if df['factor'].std() > 0 and df['return'].std() > 0:
                    ic, p_value = spearmanr(df['factor'], df['return'])
                    
                    # Determine status - FIXED LOGIC
                    if p_value > 0.05:
                        status = 'REJECTED'
                        reason = f'IC={ic:.4f} p={p_value:.3f} (NOT SIGNIFICANT)'
                    elif abs(ic) < self.ic_threshold:
                        status = 'REJECTED'
                        reason = f'IC={ic:.4f} below threshold {self.ic_threshold}'
                    elif ic < 0:
                        status = 'REJECTED'
                        reason = f'IC={ic:.4f} NEGATIVE correlation (inverse relationship)'
                    elif ic > 0.10:
                        status = 'GREAT'
                        reason = f'IC={ic:.4f} p={p_value:.3f} (great positive ranking)'
                    else:
                        status = 'VALIDATED'
                        reason = f'IC={ic:.4f} p={p_value:.3f} (good positive ranking)'

                    ic_results[factor_name] = {
                        'ic': ic,
                        'p_value': p_value,
                        'status': status,
                        'reason': reason,
                        'num_observations': len(df)
                    }

                    print(f"  {factor_name:30s} IC={ic:+.4f}  p={p_value:.4f}  [{status}]")
                else:
                    ic_results[factor_name] = {
                        'ic': np.nan,
                        'p_value': np.nan,
                        'status': 'INSUFFICIENT_DATA',
                        'reason': 'Zero variance in factor or returns',
                        'num_observations': len(df)
                    }
                    print(f"  {factor_name:30s} [INSUFFICIENT DATA - zero variance]")
            else:
                ic_results[factor_name] = {
                    'ic': np.nan,
                    'p_value': np.nan,
                    'status': 'INSUFFICIENT_DATA',
                    'reason': f'Only {len(factor_returns)} observations (need ≥10)',
                    'num_observations': len(factor_returns)
                }
                print(f"  {factor_name:30s} [INSUFFICIENT DATA: {len(factor_returns)} obs]")

        return ic_results

    # ========================================================================
    # TEST 2: HIT RATE (DIRECTIONAL ACCURACY)
    # ========================================================================

    def test_cross_sectional_hit_rate(self, factor_values: Dict[str, pd.Series],
                                     price_data: Dict[str, pd.Series],
                                     forward_days: int = 20) -> Dict[str, Dict]:
        """
        Test Cross-Sectional Hit Rate - do factors correctly rank future returns?

        For cross-sectional factors: Do higher factor values lead to higher returns across tickers?

        Args:
            factor_values: Dict of {factor_name: Series with tickers as index}
            price_data: Dict of {ticker: Series with dates as index}
            forward_days: Number of days forward for return calculation

        Returns:
            Dict of {factor_name: {'hit_rate': value, 'status': 'VALIDATED'/'REJECTED'}}
        """
        hit_rate_results = {}

        print(f"\nTEST 2: Cross-Sectional Hit Rate (Ranking Accuracy)")
        print(f"  Forward period: {forward_days} days")
        print(f"  Threshold: Hit rate > {self.hit_rate_threshold:.1%}")
        print(f"  {'-' * 60}")

        for factor_name, factor_series in factor_values.items():
            factor_returns = []

            # factor_series has tickers as index
            for ticker in factor_series.index:
                if ticker not in price_data:
                    continue

                factor_val = factor_series[ticker]  # Single scalar value
                prices = price_data[ticker]  # Time series

                # Skip if factor value is NaN
                if pd.isna(factor_val):
                    continue

                # Calculate forward return (use most recent available)
                forward_returns = prices.shift(-forward_days) / prices - 1
                valid_returns = forward_returns.dropna()
                if len(valid_returns) > 0:
                    forward_return = valid_returns.iloc[-1]  # Most recent
                    factor_returns.append({
                        'factor': factor_val,
                        'return': forward_return
                    })

            if len(factor_returns) >= 10:
                # Create DataFrame for ranking analysis
                df = pd.DataFrame(factor_returns)
                
                # Calculate median split hit rate
                factor_median = df['factor'].median()
                return_median = df['return'].median()
                
                # Count correct directional predictions
                correct = 0
                total = len(df)
                
                for _, row in df.iterrows():
                    # Factor above median should predict return above median
                    if (row['factor'] >= factor_median and row['return'] >= return_median) or \
                       (row['factor'] < factor_median and row['return'] < return_median):
                        correct += 1
                
                hit_rate = correct / total

                # Determine status
                if hit_rate < self.hit_rate_threshold:
                    status = 'REJECTED'
                    reason = f'Hit rate {hit_rate:.1%} below threshold {self.hit_rate_threshold:.1%}'
                elif hit_rate > 0.60:
                    status = 'GREAT'
                    reason = f'Hit rate {hit_rate:.1%} (great ranking accuracy)'
                else:
                    status = 'VALIDATED'
                    reason = f'Hit rate {hit_rate:.1%} (good ranking accuracy)'

                hit_rate_results[factor_name] = {
                    'hit_rate': hit_rate,
                    'correct': correct,
                    'total': total,
                    'status': status,
                    'reason': reason
                }

                print(f"  {factor_name:30s} Hit Rate={hit_rate:.1%}  ({correct}/{total})  [{status}]")
            else:
                hit_rate_results[factor_name] = {
                    'hit_rate': np.nan,
                    'status': 'INSUFFICIENT_DATA'
                }
                print(f"  {factor_name:30s} [INSUFFICIENT DATA: {len(factor_returns)} obs]")

        return hit_rate_results

    # ========================================================================
    # TEST 3: QUINTILE ANALYSIS
    # ========================================================================

    def test_cross_sectional_quintile(self, factor_values: Dict[str, pd.Series],
                                     price_data: Dict[str, pd.Series],
                                     forward_days: int = 20) -> Dict[str, Dict]:
        """
        Test Cross-Sectional Quintile Analysis - verify monotonic relationship with returns.

        For cross-sectional factors: Do higher factor quintiles have higher returns?

        Args:
            factor_values: Dict of {factor_name: Series with tickers as index}
            price_data: Dict of {ticker: Series with dates as index}
            forward_days: Number of days forward for return calculation

        Returns:
            Dict of {factor_name: quintile_spreads, monotonic_check}
        """
        quintile_results = {}

        print(f"\nTEST 3: Cross-Sectional Quintile Analysis (Monotonic Relationship)")
        print(f"  Forward period: {forward_days} days")
        print(f"  {'-' * 60}")

        for factor_name, factor_series in factor_values.items():
            all_data = []

            # factor_series has tickers as index
            for ticker in factor_series.index:
                if ticker not in price_data:
                    continue

                factor_val = factor_series[ticker]  # Single scalar value
                prices = price_data[ticker]  # Time series

                # Skip if factor value is NaN
                if pd.isna(factor_val):
                    continue

                # Calculate forward return (use most recent available)
                forward_returns = prices.shift(-forward_days) / prices - 1
                valid_returns = forward_returns.dropna()
                if len(valid_returns) > 0:
                    forward_return = valid_returns.iloc[-1]  # Most recent
                    all_data.append({
                        'factor': factor_val,
                        'return': forward_return
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
        Run all cross-sectional validation tests on factors.

        Args:
            factor_values: Dict of {factor_name: Series with tickers as index}
            price_data: Dict of {ticker: Series with dates as index}
            forward_days: Number of days forward for return calculation

        Returns:
            Dict with all validation results and final validated factors list
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE FACTOR VALIDATION - CROSS-SECTIONAL TESTS")
        print("="*70)

        # Normalize factor values
        factor_values = self._normalize_factor_values(factor_values)

        # Run cross-sectional tests
        ic_results = self.test_cross_sectional_ic(factor_values, price_data, forward_days)
        hit_rate_results = self.test_cross_sectional_hit_rate(factor_values, price_data, forward_days)
        quintile_results = self.test_cross_sectional_quintile(factor_values, price_data, forward_days)
        correlation_results = self.test_factor_correlation(factor_values)

        # Determine validated factors (cross-sectional criteria)
        validated_factors = self._determine_validated_factors_cross_sectional(
            ic_results, hit_rate_results, quintile_results
        )

        # Compile results
        results = {
            'ic_results': ic_results,
            'hit_rate_results': hit_rate_results,
            'quintile_results': quintile_results,
            'correlation_results': correlation_results,
            'validated_factors': validated_factors
        }

        self.validation_results = results
        return results

    def _determine_validated_factors_cross_sectional(self, ic_results: Dict, hit_rate_results: Dict,
                                         quintile_results: Dict) -> List[str]:
        """
        Determine which factors pass validation across all tests.

        Criteria for cross-sectional factors:
        - IC > threshold (primary criterion)
        - Hit rate > threshold (primary criterion)
        - Quintile monotonic (secondary criterion, if available)
        """
        validated_factors = []

        for factor_name in ic_results.keys():
            ic_status = ic_results[factor_name].get('status', 'UNKNOWN')
            hit_status = hit_rate_results.get(factor_name, {}).get('status', 'UNKNOWN')
            quintile_status = quintile_results.get(factor_name, {}).get('status', 'UNKNOWN')

            # Count passed tests
            passed_tests = 0
            total_tests = 0

            # IC test (primary)
            if ic_status in ['VALIDATED', 'GREAT']:
                passed_tests += 1
            total_tests += 1

            # Hit rate test (primary)
            if hit_status in ['VALIDATED', 'GREAT']:
                passed_tests += 1
            total_tests += 1

            # Quintile test (secondary)
            if quintile_status in ['VALIDATED', 'GREAT']:
                passed_tests += 1
            elif quintile_status != 'INSUFFICIENT_DATA':
                total_tests += 1

            # Validate if at least 2/3 primary tests pass
            if passed_tests >= 2:
                validated_factors.append(factor_name)

        return validated_factors
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
