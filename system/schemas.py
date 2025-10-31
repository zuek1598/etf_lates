#!/usr/bin/env python3
"""
Data Schemas and Interfaces - Modified
Defines structure for new components: Risk Component, ML Ensemble, Kalman Hull, Volume Intelligence

Last Updated: October 2025
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# ============================================================
# OUTPUT SCHEMAS - What Each Module Returns
# ============================================================

RISK_COMPONENT_SCHEMA = {
    # Risk Component (30/30/20/20 weighting)
    'cvar': {'type': float, 'range': (-1.0, 0.0), 'required': True, 'description': 'Parametric CVaR (95%)'},
    'ulcer_index': {'type': float, 'range': (0.0, 50.0), 'required': True, 'description': 'Ulcer Index (drawdown pain)'},
    'beta': {'type': float, 'range': (-2.0, 3.0), 'required': True, 'description': 'Market beta'},
    'information_ratio': {'type': float, 'range': (-5.0, 5.0), 'required': True, 'description': 'Information ratio vs benchmark'},
    'risk_score': {'type': float, 'range': (0.0, 1.0), 'required': True, 'description': 'Weighted risk score (0=best, 1=worst)'},
    'risk_category': {'type': str, 'values': ['LOW', 'MEDIUM', 'HIGH'], 'required': True, 'description': 'Risk category'},
    'quality_flag': {'type': str, 'values': ['[EMOJI]', '~', '[EMOJI]', '[EMOJI]'], 'required': True, 'description': 'Quality flag'},
    't_distribution_params': {'type': dict, 'required': False, 'description': 'T-distribution fit parameters'},
    'volatility': {'type': float, 'range': (0.0, 1.0), 'required': False, 'description': 'Annualized volatility'},
    
    # Liquidity metrics
    'amihud': {'type': float, 'range': (0.0, 100.0), 'required': False, 'description': 'Amihud illiquidity ratio'},
    'avg_daily_volume': {'type': float, 'range': (0.0, 1e12), 'required': False, 'description': 'Average daily dollar volume'},
    'zero_volume_days': {'type': int, 'range': (0, 60), 'required': False, 'description': 'Days with zero volume in last 60'},
}

KALMAN_HULL_SCHEMA = {
    # Kalman Hull Supertrend
    'trend': {'type': int, 'values': [-1, 0, 1], 'required': True, 'description': 'Trend direction'},
    'kalman_price': {'type': float, 'required': True, 'description': 'Kalman-filtered price'},
    'upper_band': {'type': float, 'required': True, 'description': 'Supertrend upper band'},
    'lower_band': {'type': float, 'required': True, 'description': 'Supertrend lower band'},
    'efficiency_ratio': {'type': float, 'range': (0.0, 1.0), 'required': True, 'description': 'Efficiency ratio (0-1)'},
    'divergence': {'type': str, 'values': ['bullish', 'bearish', 'none'], 'required': True, 'description': 'Divergence signal'},
    'trend_consistency': {'type': bool, 'required': True, 'description': 'Trend is stable'},
    'signal_strength': {'type': float, 'range': (0.0, 1.0), 'required': True, 'description': 'Signal confidence (0-1)'},
}

VOLUME_INTELLIGENCE_SCHEMA = {
    # Volume Intelligence
    'spike_score': {'type': float, 'range': (0.0, 100.0), 'required': True, 'description': 'Volume spike score (0-100)'},
    'price_volume_correlation': {'type': float, 'range': (-1.0, 1.0), 'required': True, 'description': 'Price-volume correlation'},
    'accumulation_distribution': {'type': str, 'values': ['accumulation', 'distribution', 'neutral'], 'required': True, 'description': 'A/D signal'},
    'volume_confidence': {'type': float, 'range': (0.0, 1.0), 'required': True, 'description': 'Volume signal confidence'},
}

ML_ENSEMBLE_SCHEMA = {
    # ML Ensemble (NO bias correction)
    'forecast_return': {'type': float, 'range': (-30.0, 30.0), 'required': True, 'description': '60-day return forecast (%)'},
    'confidence_score': {'type': float, 'range': (0.0, 1.0), 'required': True, 'description': 'Forecast confidence (0-1)'},
    'features_used': {'type': dict, 'required': True, 'description': 'Features used in model'},
    'model_ensemble_output': {'type': float, 'required': True, 'description': 'Raw ensemble output'},
    'feature_importance': {'type': dict, 'required': False, 'description': 'Feature importance scores'},
}

SCORING_OUTPUT_SCHEMA = {
    'composite_score': {'type': float, 'range': (0.0, 100.0), 'required': True, 'description': 'Final composite score'},
    'component_scores': {'type': dict, 'required': False, 'description': 'Individual component scores'},
    'penalties_applied': {'type': dict, 'required': False, 'description': 'Penalties applied to score'},
}

RISK_CLASSIFICATION_SCHEMA = {
    'risk_category': {'type': str, 'values': ['LOW', 'MEDIUM', 'HIGH', 'UNKNOWN'], 'required': True, 'description': 'Risk category'},
    'risk_score': {'type': float, 'range': (0.0, 1.0), 'required': True, 'description': 'Calculated risk score'},
    'volatility': {'type': float, 'range': (0.0, 1.0), 'required': True, 'description': 'Annualized volatility'},
    'beta': {'type': float, 'range': (-2.0, 3.0), 'required': True, 'description': 'Market beta'},
}

# ============================================================
# INPUT SCHEMAS - What Each Module Expects
# ============================================================

PRICE_DATA_SCHEMA = {
    'required_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
    'index': 'datetime',
    'min_rows': 90,
    'data_types': {
        'Open': float,
        'High': float,
        'Low': float,
        'Close': float,
        'Volume': float
    }
}

ETF_METADATA_SCHEMA = {
    'ticker': {'type': str, 'required': True, 'example': 'VAS.AX'},
    'name': {'type': str, 'required': False, 'example': 'Vanguard Australian Shares Index ETF'},
    'region': {'type': str, 'required': True, 'example': 'AUSTRALIA'},
    'subcategory': {'type': str, 'required': True, 'example': 'Index Tracking'},
    'benchmark': {'type': str, 'required': False, 'example': 'S&P/ASX 300'},
    'type': {'type': str, 'required': True, 'example': 'broad_market'},
}

FUNDAMENTAL_DATA_SCHEMA = {
    'expense_ratio': {'type': float, 'range': (0.0, 0.05), 'required': True, 'description': 'Annual expense ratio'},
    'aum_aud': {'type': float, 'range': (0.0, 1e12), 'required': True, 'description': 'Assets under management (AUD)'},
    'inception_date': {'type': str, 'required': False, 'description': 'ETF inception date (YYYY-MM-DD)'},
}

# ============================================================
# VALIDATION SCHEMAS
# ============================================================

VALIDATION_RANGES = {
    'volatility': (0.0, 1.0),
    'beta': (-2.0, 3.0),
    'ulcer_index': (0.0, 50.0),
    'information_ratio': (-5.0, 5.0),
    'cvar': (-1.0, 0.0),
    'amihud': (0.0, 100.0),
    'forecast': (-30.0, 30.0),
    'confidence': (0.0, 1.0),
    'composite_score': (0.0, 100.0),
    'spike_score': (0.0, 100.0),
    'correlation': (-1.0, 1.0),
    'efficiency_ratio': (0.0, 1.0),
    'signal_strength': (0.0, 1.0),
}

# ============================================================
# SCHEMA VALIDATION FUNCTIONS
# ============================================================

def validate_field(value: Any, schema: Dict) -> Tuple[bool, str]:
    """
    Validate a single field against its schema
    
    Args:
        value: The value to validate
        schema: The schema definition
        
    Returns:
        (is_valid, error_message)
    """
    # Check required
    if value is None or (isinstance(value, float) and np.isnan(value)):
        if schema.get('required', False):
            return False, "Required field is missing"
        return True, ""
    
    # Check type
    expected_type = schema.get('type')
    if expected_type and not isinstance(value, expected_type):
        return False, f"Expected type {expected_type.__name__}, got {type(value).__name__}"
    
    # Check range
    if 'range' in schema and isinstance(value, (int, float)):
        min_val, max_val = schema['range']
        if not (min_val <= value <= max_val):
            return False, f"Value {value} outside valid range [{min_val}, {max_val}]"
    
    # Check values (for string/int fields)
    if 'values' in schema:
        if value not in schema['values']:
            return False, f"Value '{value}' not in allowed values: {schema['values']}"
    
    return True, ""


def validate_output(data: Dict, schema: Dict, soft: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate output data against schema
    
    Args:
        data: Dictionary of output data
        schema: Schema to validate against
        soft: If True, log warnings instead of failing
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    for field_name, field_schema in schema.items():
        value = data.get(field_name)
        is_valid, error_msg = validate_field(value, field_schema)
        
        if not is_valid:
            error_msg = f"{field_name}: {error_msg}"
            if soft:
                print(f"[EMOJI]  Validation warning: {error_msg}")
            else:
                errors.append(error_msg)
    
    return len(errors) == 0, errors


def validate_risk_component(data: Dict) -> bool:
    """Validate Risk Component output"""
    is_valid, errors = validate_output(data, RISK_COMPONENT_SCHEMA, soft=False)
    if not is_valid:
        print(f"[EMOJI] Risk Component validation failed: {errors}")
    return is_valid


def validate_kalman_hull(data: Dict) -> bool:
    """Validate Kalman Hull output"""
    is_valid, errors = validate_output(data, KALMAN_HULL_SCHEMA, soft=False)
    if not is_valid:
        print(f"[EMOJI] Kalman Hull validation failed: {errors}")
    return is_valid


def validate_volume_intelligence(data: Dict) -> bool:
    """Validate Volume Intelligence output"""
    is_valid, errors = validate_output(data, VOLUME_INTELLIGENCE_SCHEMA, soft=False)
    if not is_valid:
        print(f"[EMOJI] Volume Intelligence validation failed: {errors}")
    return is_valid


def validate_ml_ensemble(data: Dict) -> bool:
    """Validate ML Ensemble output (CRITICAL: NO bias_correction field)"""
    # CRITICAL CHECK: Ensure NO bias correction field exists
    if 'bias_correction' in data or 'bias_corrected' in data:
        print(f"[EMOJI] CRITICAL: Bias correction field must NOT exist (raw output required)")
        return False
    
    is_valid, errors = validate_output(data, ML_ENSEMBLE_SCHEMA, soft=False)
    if not is_valid:
        print(f"[EMOJI] ML Ensemble validation failed: {errors}")
    return is_valid


# Example usage
if __name__ == "__main__":
    print("Schema Validation Examples")
    print("=" * 60)
    
    # Test Risk Component
    risk_data = {
        'cvar': -0.05,
        'ulcer_index': 3.2,
        'beta': 1.1,
        'information_ratio': 0.8,
        'risk_score': 0.35,
        'risk_category': 'LOW',
        'quality_flag': '[EMOJI]'
    }
    print("\n1. Risk Component:")
    print(f"   Valid: {validate_risk_component(risk_data)}")
    
    # Test Kalman Hull
    kalman_data = {
        'trend': 1,
        'kalman_price': 50.5,
        'upper_band': 52.0,
        'lower_band': 49.0,
        'efficiency_ratio': 0.7,
        'divergence': 'bullish',
        'trend_consistency': True,
        'signal_strength': 0.8
    }
    print("\n2. Kalman Hull:")
    print(f"   Valid: {validate_kalman_hull(kalman_data)}")
    
    # Test Volume Intelligence
    volume_data = {
        'spike_score': 65.0,
        'price_volume_correlation': 0.6,
        'accumulation_distribution': 'accumulation',
        'volume_confidence': 0.75
    }
    print("\n3. Volume Intelligence:")
    print(f"   Valid: {validate_volume_intelligence(volume_data)}")
    
    # Test ML Ensemble
    ml_data = {
        'forecast_return': 2.5,
        'confidence_score': 0.7,
        'features_used': {'momentum': 0.5},
        'model_ensemble_output': 2.5,
        'feature_importance': {'f1': 0.3}
    }
    print("\n4. ML Ensemble:")
    print(f"   Valid: {validate_ml_ensemble(ml_data)}")
    
    print("\n" + "=" * 60)

