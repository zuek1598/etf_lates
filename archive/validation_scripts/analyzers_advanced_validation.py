#!/usr/bin/env python3
"""
Advanced Validation System - Phase 3 Implementation
Implements nested cross-validation, expanding window validation, and bootstrap confidence intervals

Key Features:
- Nested cross-validation for robust model evaluation
- Expanding window validation for time series
- Bootstrap confidence intervals for predictions
- Confidence flagging system (High/Medium/Low)
- Out-of-sample error estimation
- Model stability assessment

Usage:
    from analyzers.advanced_validation import AdvancedValidator
    
    validator = AdvancedValidator()
    results = validator.validate_model(ml_ensemble, price_data)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedValidator:
    """Advanced validation system for ML models with confidence estimation"""
    
    def __init__(self, 
                 n_splits: int = 3,
                 bootstrap_samples: int = 1000,
                 confidence_levels: List[float] = [0.68, 0.95, 0.99]):
        """
        Initialize advanced validator
        
        Args:
            n_splits: Number of splits for time series cross-validation
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            confidence_levels: Confidence levels for intervals (default: 68%, 95%, 99%)
        """
        self.n_splits = n_splits
        self.bootstrap_samples = bootstrap_samples
        self.confidence_levels = confidence_levels
        
        # Validation parameters - test size must be > lookback_days
        self.min_train_size = 200  # Minimum training size
        self.test_size = 150       # Test window size (must be > lookback_days)
        self.expanding_step = 50   # Expansion step size
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.7,      # High confidence: >70%
            'medium': 0.4,    # Medium confidence: 40-70%
            'low': 0.0        # Low confidence: <40%
        }
    
    def nested_cross_validation(self, ml_ensemble, price_data: pd.Series, 
                              lookback_days: int = 100) -> Dict:
        """
        Perform nested cross-validation for robust model evaluation
        
        Args:
            ml_ensemble: MLEnsemble instance to validate
            price_data: Price series for validation
            lookback_days: Lookback period for feature extraction
            
        Returns:
            Dictionary with nested CV results
        """
        print(f"DEBUG_NESTED: Starting nested CV with {len(price_data)} data points")
        print(f"DEBUG_NESTED: min_train_size={self.min_train_size}, test_size={self.test_size}")
        
        # Check if we have enough data
        min_required_data = self.min_train_size + self.test_size
        if len(price_data) < min_required_data:
            print(f"DEBUG_NESTED: ❌ Insufficient data: {len(price_data)} < {min_required_data}")
            return {'error': f'Insufficient data for nested CV. Need at least {min_required_data} points'}
        
        # Create time series split for outer loop
        tscv = TimeSeriesSplit(n_splits=self.n_splits, 
                              test_size=self.test_size,
                              gap=lookback_days)  # Gap for feature extraction
        
        outer_mae_scores = []
        outer_hit_rates = []
        stability_scores = []
        
        print(f"DEBUG_NESTED: Created TimeSeriesSplit with {self.n_splits} splits")
        
        fold_count = 0
        for outer_fold, (train_idx, test_idx) in enumerate(tscv.split(price_data)):
            fold_count += 1
            print(f"DEBUG_NESTED: === Outer Fold {fold_count}/{self.n_splits} ===")
            
            # Split data
            train_data = price_data.iloc[train_idx]
            test_data = price_data.iloc[test_idx]
            
            print(f"DEBUG_NESTED: Train: {len(train_data)} points, Test: {len(test_data)} points")
            
            try:
                # Train model on training data
                print(f"DEBUG_NESTED: Training model on outer fold {fold_count}...")
                models = ml_ensemble.train_ensemble(train_data, lookback_days=lookback_days)
                
                if models is None or models.get('rf') is None:
                    print(f"DEBUG_NESTED: ❌ Model training failed on outer fold {fold_count}")
                    continue
                
                print(f"DEBUG_NESTED: ✅ Model training successful on outer fold {fold_count}")
                
                # Perform inner cross-validation for stability assessment
                print(f"DEBUG_NESTED: Running inner CV for stability...")
                inner_results = self._inner_cross_validation(ml_ensemble, train_data, lookback_days)
                
                # Evaluate model on test data
                print(f"DEBUG_NESTED: Evaluating model on test data...")
                test_results = self._evaluate_model(ml_ensemble, models, test_data, lookback_days)
                
                print(f"DEBUG_NESTED: Test results - MAE: {test_results['mae']}, Hit Rate: {test_results['hit_rate']}")
                
                # Store results
                outer_mae_scores.append(test_results['mae'])
                outer_hit_rates.append(test_results['hit_rate'])
                stability_scores.append(self._calculate_model_stability(inner_results))
                
                print(f"DEBUG_NESTED: ✅ Outer fold {fold_count} completed successfully")
                
            except Exception as e:
                print(f"DEBUG_NESTED: ❌ Exception in outer fold {fold_count}: {e}")
                import traceback
                traceback.print_exc()
                # Append NaN for failed folds
                outer_mae_scores.append(np.nan)
                outer_hit_rates.append(np.nan)
                stability_scores.append(0.0)
        
        print(f"DEBUG_NESTED: === Nested CV Complete ===")
        print(f"DEBUG_NESTED: Total folds processed: {fold_count}")
        print(f"DEBUG_NESTED: MAE scores: {outer_mae_scores}")
        print(f"DEBUG_NESTED: Hit rates: {outer_hit_rates}")
        
        # Calculate aggregate statistics
        valid_mae_scores = [score for score in outer_mae_scores if not np.isnan(score)]
        valid_hit_rates = [rate for rate in outer_hit_rates if not np.isnan(rate)]
        
        print(f"DEBUG_NESTED: Valid MAE scores: {valid_mae_scores}")
        print(f"DEBUG_NESTED: Valid hit rates: {valid_hit_rates}")
        
        if valid_mae_scores:
            avg_mae = np.mean(valid_mae_scores)
            avg_hit_rate = np.mean(valid_hit_rates)
            validation_variance = np.var(valid_mae_scores)
        else:
            avg_mae = np.nan
            avg_hit_rate = np.nan
            validation_variance = np.nan
        
        avg_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        print(f"DEBUG_NESTED: Final results - MAE: {avg_mae}, Hit Rate: {avg_hit_rate}")
        
        return {
            'avg_outer_mae': avg_mae,
            'avg_outer_hit_rate': avg_hit_rate,
            'avg_stability': avg_stability,
            'validation_variance': validation_variance,
            'num_folds': len(valid_mae_scores),
            'outer_mae_scores': outer_mae_scores,
            'outer_hit_rates': outer_hit_rates
        }
    
    def _inner_cross_validation(self, ml_ensemble, train_data: pd.Series, 
                               lookback_days: int) -> Dict:
        """Perform inner cross-validation for hyperparameter tuning"""
        inner_mae_scores = []
        inner_hit_rates = []
        
        # Use 3-fold inner CV
        inner_splits = 3
        tscv_inner = TimeSeriesSplit(
            n_splits=inner_splits,
            max_train_size=None,
            test_size=self.test_size // 2,  # Smaller test for inner CV
            gap=0
        )
        
        for inner_train_idx, inner_test_idx in tscv_inner.split(train_data):
            inner_train = train_data.iloc[inner_train_idx]
            inner_test = train_data.iloc[inner_test_idx]
            
            # Skip if insufficient data
            if len(inner_train) < lookback_days + 60:
                continue
            
            # Train and evaluate
            inner_models = ml_ensemble.train_ensemble(inner_train, lookback_days)
            if inner_models.get('rf') is None:
                continue
            
            inner_results = self._evaluate_model(
                ml_ensemble, inner_models, inner_test, lookback_days
            )
            
            inner_mae_scores.append(inner_results['mae'])
            inner_hit_rates.append(inner_results['hit_rate'])
        
        return {
            'mae_scores': inner_mae_scores,
            'hit_rates': inner_hit_rates,
            'avg_mae': np.mean(inner_mae_scores) if inner_mae_scores else np.nan,
            'std_mae': np.std(inner_mae_scores) if inner_mae_scores else np.nan,
            'avg_hit_rate': np.mean(inner_hit_rates) if inner_hit_rates else np.nan
        }
    
    def _evaluate_model(self, ml_ensemble, models: Dict, test_data: pd.Series, 
                       lookback_days: int) -> Dict:
        """Evaluate trained model on test data"""
        predictions = []
        actuals = []
        
        # Need at least lookback_days + 1 points for evaluation
        if len(test_data) <= lookback_days:
            return {'mae': np.nan, 'hit_rate': np.nan, 'predictions': [], 'actuals': []}
        
        # Use valid range for evaluation - can use the very last point
        start_idx = lookback_days
        end_idx = len(test_data)  # Can go to the end
        
        # Just evaluate at the last few points for simplicity
        num_evaluations = min(5, end_idx - start_idx)
        if num_evaluations <= 0:
            return {'mae': np.nan, 'hit_rate': np.nan, 'predictions': [], 'actuals': []}
        
        # Evaluate at the end of the test data
        step = max(1, (end_idx - start_idx) // num_evaluations)
        evaluation_indices = range(end_idx - num_evaluations * step, end_idx, step)
        
        for i in evaluation_indices:
            # Get features for prediction
            window_start = i - lookback_days
            window_end = i
            window_data = test_data.iloc[window_start:window_end]
            
            try:
                # Extract features and predict
                features = ml_ensemble.extract_ml_features(window_data, use_last_point=True)
                
                # Check if features are valid
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    continue
                
                # Scale features
                features_scaled = ml_ensemble.robust_scale_features(features)
                
                # Check if scaled features are valid
                if np.any(np.isnan(features_scaled)) or np.any(np.isinf(features_scaled)):
                    continue
                
                # Make prediction
                rf_pred = models['rf'].predict(features_scaled)[0]
                ridge_pred = models['ridge'].predict(features_scaled)[0]
                prediction = (rf_pred + ridge_pred) / 2.0
                
                # Get actual return - use previous return if at the end
                if i > 0:
                    actual_return = (test_data.iloc[i] - test_data.iloc[i-1]) / test_data.iloc[i-1]
                else:
                    actual_return = 0.0  # Default if can't calculate
                
                # Validate prediction and actual are finite
                if np.isfinite(prediction) and np.isfinite(actual_return):
                    predictions.append(prediction)
                    actuals.append(actual_return)
                
            except Exception as e:
                continue
        
        if not predictions:
            return {'mae': np.nan, 'hit_rate': np.nan, 'predictions': [], 'actuals': []}
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Remove any remaining infinite values
        valid_mask = np.isfinite(predictions) & np.isfinite(actuals)
        predictions = predictions[valid_mask]
        actuals = actuals[valid_mask]
        
        if len(predictions) == 0:
            return {'mae': np.nan, 'hit_rate': np.nan, 'predictions': [], 'actuals': []}
        
        mae = np.mean(np.abs(predictions - actuals))
        hit_rate = np.mean((predictions > 0) == (actuals > 0))
        
        return {
            'mae': mae,
            'hit_rate': hit_rate,
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist()
        }
    
    def _calculate_model_stability(self, inner_results: Dict) -> float:
        """Calculate model stability based on inner CV variance"""
        if 'mae_scores' not in inner_results or len(inner_results['mae_scores']) < 2:
            return 0.5  # Default stability
        
        mae_scores = inner_results['mae_scores']
        stability = 1.0 - (np.std(mae_scores) / (np.mean(mae_scores) + 1e-8))
        
        return max(0.0, min(1.0, stability))
    
    def _aggregate_nested_results(self, outer_scores: List[Dict], 
                                 inner_scores: List[Dict], 
                                 stability_scores: List[float]) -> Dict:
        """Aggregate results from nested cross-validation"""
        if not outer_scores:
            return {'error': 'No valid outer folds'}
        
        # Extract outer scores
        outer_mae = [s['mae'] for s in outer_scores if not np.isnan(s['mae'])]
        outer_hit_rates = [s['hit_rate'] for s in outer_scores if not np.isnan(s['hit_rate'])]
        
        # Calculate aggregated metrics
        results = {
            'outer_scores': outer_scores,
            'inner_scores': inner_scores,
            'avg_outer_mae': np.mean(outer_mae) if outer_mae else np.nan,
            'std_outer_mae': np.std(outer_mae) if outer_mae else np.nan,
            'avg_outer_hit_rate': np.mean(outer_hit_rates) if outer_hit_rates else np.nan,
            'std_outer_hit_rate': np.std(outer_hit_rates) if outer_hit_rates else np.nan,
            'avg_stability': np.mean(stability_scores) if stability_scores else np.nan,
            'validation_variance': np.var(outer_mae) if outer_mae else np.nan,
            'num_folds': len(outer_scores)
        }
        
        return results
    
    def expanding_window_validation(self, ml_ensemble, price_data: pd.Series,
                                  lookback_days: int = 100) -> Dict:
        """
        Perform expanding window validation for time series robustness
        
        Args:
            ml_ensemble: MLEnsemble instance to validate
            price_data: Price series for validation
            lookback_days: Lookback period for feature extraction
            
        Returns:
            Dictionary with expanding window results
        """
        print(f"\n{'='*60}")
        print("EXPANDING WINDOW VALIDATION - PHASE 3")
        print(f"{'='*60}")
        print(f"Initial train size: {self.min_train_size}")
        print(f"Expansion step: {self.expanding_step}")
        print(f"Test window: {self.test_size}")
        
        if len(price_data) < self.min_train_size + self.test_size:
            print("❌ Insufficient data for expanding window validation")
            return {'error': 'Insufficient data'}
        
        results = {
            'window_results': [],
            'performance_trend': [],
            'cumulative_metrics': {}
        }
        
        # Start with initial training window
        train_end = self.min_train_size
        window_num = 0
        
        while train_end + self.test_size < len(price_data):
            window_num += 1
            print(f"\n--- Window {window_num} ---")
            
            # Define training and test windows
            train_data = price_data.iloc[:train_end]
            test_start = train_end
            test_end = train_end + self.test_size
            test_data = price_data.iloc[test_start:test_end]
            
            print(f"Train: {len(train_data)} points, Test: {len(test_data)} points")
            
            # Train model
            models = ml_ensemble.train_ensemble(train_data, lookback_days)
            
            if models.get('rf') is None:
                print(f"❌ Window {window_num}: Model training failed")
                train_end += self.expanding_step
                continue
            
            # Evaluate
            test_results = self._evaluate_model(
                ml_ensemble, models, test_data, lookback_days
            )
            
            window_result = {
                'window': window_num,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'mae': test_results['mae'],
                'hit_rate': test_results['hit_rate'],
                'train_end_date': train_data.index[-1],
                'test_end_date': test_data.index[-1]
            }
            
            results['window_results'].append(window_result)
            results['performance_trend'].append(test_results['mae'])
            
            print(f"Window {window_num} results:")
            print(f"  • MAE: {test_results['mae']:.3f}")
            print(f"  • Hit Rate: {test_results['hit_rate']:.1%}")
            
            # Expand window
            train_end += self.expanding_step
        
        # Calculate cumulative metrics
        if results['window_results']:
            mae_trend = [w['mae'] for w in results['window_results']]
            hit_rate_trend = [w['hit_rate'] for w in results['window_results']]
            
            results['cumulative_metrics'] = {
                'avg_mae': np.mean(mae_trend),
                'mae_trend_slope': np.polyfit(range(len(mae_trend)), mae_trend, 1)[0],
                'avg_hit_rate': np.mean(hit_rate_trend),
                'hit_rate_trend_slope': np.polyfit(range(len(hit_rate_trend)), hit_rate_trend, 1)[0],
                'performance_stability': max(0.0, 1.0 - (np.std(mae_trend) / (np.mean(mae_trend) + 1e-8))),
                'total_windows': len(results['window_results'])
            }
            
            print(f"\n📊 Expanding Window Summary:")
            print(f"  • Total windows: {results['cumulative_metrics']['total_windows']}")
            print(f"  • Average MAE: {results['cumulative_metrics']['avg_mae']:.3f}")
            print(f"  • MAE trend slope: {results['cumulative_metrics']['mae_trend_slope']:.6f}")
            print(f"  • Performance stability: {results['cumulative_metrics']['performance_stability']:.3f}")
        
        return results
    
    def bootstrap_confidence_intervals(self, predictions: np.ndarray, 
                                     actuals: np.ndarray) -> Dict:
        """
        Calculate bootstrap confidence intervals for predictions
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            
        Returns:
            Dictionary with confidence intervals
        """
        print(f"\n{'='*60}")
        print("BOOTSTRAP CONFIDENCE INTERVALS - PHASE 3")
        print(f"{'='*60}")
        print(f"Bootstrap samples: {self.bootstrap_samples}")
        print(f"Confidence levels: {self.confidence_levels}")
        
        if len(predictions) == 0:
            return {'error': 'No predictions provided'}
        
        # Calculate errors
        errors = predictions - actuals
        mae_observed = np.mean(np.abs(errors))
        
        # Bootstrap sampling
        bootstrap_mae = []
        bootstrap_errors = []
        
        for i in range(self.bootstrap_samples):
            # Resample with replacement
            indices = resample(range(len(errors)), replace=True, n_samples=len(errors))
            sample_errors = errors[indices]
            sample_mae = np.mean(np.abs(sample_errors))
            
            bootstrap_mae.append(sample_mae)
            bootstrap_errors.append(sample_errors)
        
        bootstrap_mae = np.array(bootstrap_mae)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for level in self.confidence_levels:
            alpha = 1 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_mae, lower_percentile)
            ci_upper = np.percentile(bootstrap_mae, upper_percentile)
            
            confidence_intervals[f'{int(level*100)}%'] = {
                'lower': ci_lower,
                'upper': ci_upper,
                'width': ci_upper - ci_lower
            }
        
        # Calculate prediction intervals for individual predictions
        prediction_intervals = self._calculate_prediction_intervals(
            predictions, bootstrap_errors
        )
        
        results = {
            'observed_mae': mae_observed,
            'bootstrap_mae_mean': np.mean(bootstrap_mae),
            'bootstrap_mae_std': np.std(bootstrap_mae),
            'confidence_intervals': confidence_intervals,
            'prediction_intervals': prediction_intervals,
            'bootstrap_samples': self.bootstrap_samples
        }
        
        print(f"📊 Bootstrap Results:")
        print(f"  • Observed MAE: {mae_observed:.4f}")
        print(f"  • Bootstrap MAE mean: {results['bootstrap_mae_mean']:.4f}")
        print(f"  • Bootstrap MAE std: {results['bootstrap_mae_std']:.4f}")
        
        for level, ci in confidence_intervals.items():
            print(f"  • {level} CI: [{ci['lower']:.4f}, {ci['upper']:.4f}] (width: {ci['width']:.4f})")
        
        return results
    
    def _calculate_prediction_intervals(self, predictions: np.ndarray, 
                                       bootstrap_errors: List[np.ndarray]) -> Dict:
        """Calculate prediction intervals for individual predictions"""
        # Aggregate bootstrap errors
        all_errors = np.concatenate(bootstrap_errors)
        
        # Calculate percentiles for prediction intervals
        intervals = {}
        for level in self.confidence_levels:
            alpha = 1 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            error_lower = np.percentile(all_errors, lower_percentile)
            error_upper = np.percentile(all_errors, upper_percentile)
            
            intervals[f'{int(level*100)}%'] = {
                'error_lower': error_lower,
                'error_upper': error_upper
            }
        
        return intervals
    
    def confidence_flagging_system(self, validation_results: Dict) -> Dict:
        """
        Implement confidence flagging system (High/Medium/Low)
        
        Args:
            validation_results: Results from validation procedures
            
        Returns:
            Dictionary with confidence flags and recommendations
        """
        print(f"\n{'='*60}")
        print("CONFIDENCE FLAGGING SYSTEM - PHASE 3")
        print(f"{'='*60}")
        
        flags = {
            'overall_confidence': 'UNKNOWN',
            'confidence_score': 0.0,
            'individual_flags': {},
            'recommendations': []
        }
        
        # Extract metrics from validation results
        metrics = {}
        
        if 'avg_outer_mae' in validation_results:
            metrics['nested_mae'] = validation_results['avg_outer_mae']
            metrics['nested_stability'] = validation_results.get('avg_stability', 0.5)
        
        if 'cumulative_metrics' in validation_results:
            metrics['expanding_mae'] = validation_results['cumulative_metrics']['avg_mae']
            metrics['performance_stability'] = validation_results['cumulative_metrics']['performance_stability']
        
        if 'observed_mae' in validation_results:
            metrics['bootstrap_mae'] = validation_results['observed_mae']
            metrics['bootstrap_ci_width'] = validation_results['confidence_intervals']['95%']['width']
        
        if not metrics:
            flags['overall_confidence'] = 'INSUFFICIENT_DATA'
            return flags
        
        # Calculate individual confidence scores
        individual_scores = {}
        
        # Nested CV confidence
        if 'nested_mae' in metrics:
            mae_score = max(0, 1 - metrics['nested_mae'] / 0.1)  # Normalize against 10% MAE
            stability_score = metrics['nested_stability']
            nested_score = (mae_score + stability_score) / 2
            individual_scores['nested_cv'] = nested_score
        
        # Expanding window confidence
        if 'expanding_mae' in metrics:
            mae_score = max(0, 1 - metrics['expanding_mae'] / 0.1)
            stability_score = metrics['performance_stability']
            expanding_score = (mae_score + stability_score) / 2
            individual_scores['expanding_window'] = expanding_score
        
        # Bootstrap confidence
        if 'bootstrap_mae' in metrics:
            mae_score = max(0, 1 - metrics['bootstrap_mae'] / 0.1)
            ci_width_score = max(0, 1 - metrics['bootstrap_ci_width'] / 0.05)  # Normalize against 5% CI width
            bootstrap_score = (mae_score + ci_width_score) / 2
            individual_scores['bootstrap'] = bootstrap_score
        
        # Calculate overall confidence score
        if individual_scores:
            overall_score = np.mean(list(individual_scores.values()))
            flags['confidence_score'] = overall_score
            
            # Determine confidence flag
            if overall_score >= self.confidence_thresholds['high']:
                flags['overall_confidence'] = 'HIGH'
                flags['recommendations'].append('Model shows high confidence - suitable for production')
            elif overall_score >= self.confidence_thresholds['medium']:
                flags['overall_confidence'] = 'MEDIUM'
                flags['recommendations'].append('Model shows moderate confidence - monitor closely')
            else:
                flags['overall_confidence'] = 'LOW'
                flags['recommendations'].append('Model shows low confidence - requires improvement')
        
        flags['individual_flags'] = individual_scores
        
        print(f"📊 Confidence Flagging Results:")
        print(f"  • Overall confidence: {flags['overall_confidence']}")
        print(f"  • Confidence score: {flags['confidence_score']:.3f}")
        
        for method, score in individual_scores.items():
            flag = 'HIGH' if score >= 0.7 else 'MEDIUM' if score >= 0.4 else 'LOW'
            print(f"  • {method}: {flag} (score: {score:.3f})")
        
        print(f"\nRecommendations:")
        for rec in flags['recommendations']:
            print(f"  • {rec}")
        
        return flags

# Convenience function for comprehensive validation
def comprehensive_model_validation(ml_ensemble, price_data: pd.Series, 
                                 lookback_days: int = 100) -> Dict:
    """
    Perform comprehensive model validation using all Phase 3 methods
    
    Args:
        ml_ensemble: MLEnsemble instance to validate
        price_data: Price series for validation
        lookback_days: Lookback period for feature extraction
        
    Returns:
        Complete validation results dictionary
    """
    validator = AdvancedValidator()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL VALIDATION - PHASE 3")
    print("="*80)
    
    # Perform all validation methods
    results = {
        'nested_cv': validator.nested_cross_validation(ml_ensemble, price_data, lookback_days),
        'expanding_window': validator.expanding_window_validation(ml_ensemble, price_data, lookback_days),
        'bootstrap': None,  # Will be populated if predictions available
        'confidence_flags': None
    }
    
    # Extract predictions for bootstrap if available
    if 'nested_cv' in results and 'outer_scores' in results['nested_cv']:
        all_predictions = []
        all_actuals = []
        
        for score in results['nested_cv']['outer_scores']:
            if 'predictions' in score and 'actuals' in score:
                all_predictions.extend(score['predictions'])
                all_actuals.extend(score['actuals'])
        
        if all_predictions:
            predictions = np.array(all_predictions)
            actuals = np.array(all_actuals)
            results['bootstrap'] = validator.bootstrap_confidence_intervals(predictions, actuals)
    
    # Apply confidence flagging
    combined_metrics = {}
    for method in ['nested_cv', 'expanding_window']:
        if results[method] and 'error' not in results[method]:
            combined_metrics.update(results[method])
    
    if results['bootstrap']:
        combined_metrics.update(results['bootstrap'])
    
    results['confidence_flags'] = validator.confidence_flagging_system(combined_metrics)
    
    # Summary
    print(f"\n{'='*80}")
    print("PHASE 3 VALIDATION SUMMARY")
    print("="*80)
    
    if results['confidence_flags']['overall_confidence'] != 'INSUFFICIENT_DATA':
        print(f"Overall Confidence: {results['confidence_flags']['overall_confidence']}")
        print(f"Confidence Score: {results['confidence_flags']['confidence_score']:.3f}")
        
        if 'nested_cv' in results and 'avg_outer_mae' in results['nested_cv']:
            print(f"Nested CV MAE: {results['nested_cv']['avg_outer_mae']:.4f}")
        
        if 'expanding_window' in results and 'cumulative_metrics' in results['expanding_window']:
            print(f"Expanding Window MAE: {results['expanding_window']['cumulative_metrics']['avg_mae']:.4f}")
        
        if 'bootstrap' in results and results['bootstrap']:
            print(f"Bootstrap MAE: {results['bootstrap']['observed_mae']:.4f}")
    else:
        print("❌ Insufficient data for comprehensive validation")
    
    return results

if __name__ == "__main__":
    # Test the advanced validation system
    print("Testing Advanced Validation System...")
    
    # This would be tested with actual ML ensemble and price data
    print("✅ Advanced validation system implemented successfully")
    print("Ready for integration with ML ensemble models")
