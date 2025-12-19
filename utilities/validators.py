"""
Validation Module - Updated
Validates outputs from all components with specific range checks
"""

import numpy as np
import pandas as pd
from typing import Dict


# ============================================================================
# NEW VALIDATORS (ADDED)
# ============================================================================

def validate_kalman_hull_output(output: Dict) -> bool:
    """
    Validate Kalman Hull Supertrend output
    Ensures all fields are within spec ranges
    """
    required_fields = ['trend', 'kalman_price', 'upper_band', 'lower_band', 
                      'efficiency_ratio', 'divergence', 'trend_consistency', 'signal_strength']
    
    # Check all required fields exist
    for field in required_fields:
        if field not in output:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate trend
    trend = output['trend']
    if trend not in [-1, 0, 1]:
        raise ValueError(f"Trend must be -1, 0, or 1, got {trend}")
    
    # Validate numeric fields
    numeric_fields = ['kalman_price', 'upper_band', 'lower_band']
    for field in numeric_fields:
        if not isinstance(output[field], (int, float)):
            raise ValueError(f"{field} must be numeric, got {type(output[field])}")
        if np.isnan(output[field]) or np.isinf(output[field]):
            raise ValueError(f"{field} is NaN or Inf")
    
    # Validate efficiency ratio [0, 1]
    if not (0 <= output['efficiency_ratio'] <= 1):
        raise ValueError(f"efficiency_ratio must be [0,1], got {output['efficiency_ratio']}")
    
    # Validate divergence
    if output['divergence'] not in ['bullish', 'bearish', 'none']:
        raise ValueError(f"divergence must be bullish/bearish/none, got {output['divergence']}")
    
    # Validate trend consistency
    if not isinstance(output['trend_consistency'], bool):
        raise ValueError(f"trend_consistency must be bool, got {type(output['trend_consistency'])}")
    
    # Validate signal strength [0, 1]
    if not (0 <= output['signal_strength'] <= 1):
        raise ValueError(f"signal_strength must be [0,1], got {output['signal_strength']}")
    
    return True


def validate_ml_ensemble_output(output: Dict) -> bool:
    """
    Validate ML Ensemble output
    CRITICAL: Ensures NO bias_correction field exists (raw output)
    """
    required_fields = ['forecast_return', 'confidence_score', 'features_used', 
                      'model_ensemble_output', 'feature_importance']
    
    # Check all required fields exist
    for field in required_fields:
        if field not in output:
            raise ValueError(f"Missing required field: {field}")
    
    # CRITICAL: Check NO bias correction field exists
    if 'bias_correction' in output or 'bias_corrected' in output:
        raise ValueError("CRITICAL: Bias correction field must NOT exist (raw output required)")
    
    # Validate forecast return is numeric
    forecast = output['forecast_return']
    if not isinstance(forecast, (int, float)):
        raise ValueError(f"forecast_return must be numeric, got {type(forecast)}")
    if np.isinf(forecast):
        raise ValueError(f"forecast_return is Inf")
    # Allow NaN only if no data available
    
    # Validate confidence score [0, 1]
    conf = output['confidence_score']
    if not isinstance(conf, (int, float)):
        raise ValueError(f"confidence_score must be numeric, got {type(conf)}")
    if not (0 <= conf <= 1):
        raise ValueError(f"confidence_score must be [0,1], got {conf}")
    
    # Validate features_used is dict
    if not isinstance(output['features_used'], dict):
        raise ValueError(f"features_used must be dict, got {type(output['features_used'])}")
    
    # Validate model ensemble output is numeric
    ensemble = output['model_ensemble_output']
    if not isinstance(ensemble, (int, float)):
        raise ValueError(f"model_ensemble_output must be numeric, got {type(ensemble)}")
    
    # Validate feature_importance is dict
    if not isinstance(output['feature_importance'], dict):
        raise ValueError(f"feature_importance must be dict, got {type(output['feature_importance'])}")
    
    return True


def validate_risk_component_output(output: Dict) -> bool:
    """
    Validate Risk Component output (CVaR only - validated mode)
    """
    required_fields = ['cvar', 'risk_score', 'risk_category', 'quality_flag']
    
    # Check all required fields exist
    for field in required_fields:
        if field not in output:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate CVaR [-1, 0]
    cvar = output['cvar']
    if not isinstance(cvar, (int, float)):
        raise ValueError(f"cvar must be numeric, got {type(cvar)}")
    if not (-1 <= cvar <= 0):
        raise ValueError(f"cvar must be [-1, 0], got {cvar}")
    
    # Validate risk_score [0, 1]
    risk_score = output['risk_score']
    if not isinstance(risk_score, (int, float)):
        raise ValueError(f"risk_score must be numeric, got {type(risk_score)}")
    if not (0 <= risk_score <= 1):
        raise ValueError(f"risk_score must be [0, 1], got {risk_score}")
    
    # Validate risk_category
    if output['risk_category'] not in ['LOW', 'MEDIUM', 'HIGH']:
        raise ValueError(f"risk_category must be LOW/MEDIUM/HIGH, got {output['risk_category']}")
    
    # Validate quality_flag
    if output['quality_flag'] not in ['[EMOJI]', '~', '[EMOJI]', '[EMOJI]']:
        raise ValueError(f"quality_flag must be [EMOJI]/~/[EMOJI]/[EMOJI], got {output['quality_flag']}")
    
    return True


# ============================================================================
# EXISTING VALIDATORS (KEPT)
# ============================================================================

def validate_risk_category(category: str) -> bool:
    """Validate risk category"""
    if category not in ['LOW', 'MEDIUM', 'HIGH', 'UNKNOWN']:
        raise ValueError(f"Invalid risk category: {category}")
    return True


def validate_quality_flag(flag: str) -> bool:
    """Validate quality flag"""
    if flag not in ['[EMOJI]', '~', '[EMOJI]', '[EMOJI]']:
        raise ValueError(f"Invalid quality flag: {flag}")
    return True


def validate_data_quality_tier(tier: str) -> bool:
    """Validate data quality tier"""
    if tier not in ['tier_1', 'tier_2', 'tier_3', 'tier_4']:
        raise ValueError(f"Invalid data quality tier: {tier}")
    return True


def validate_risk_score(score: float) -> bool:
    """Validate risk score is between 0 and 1"""
    if not isinstance(score, (int, float)):
        raise ValueError(f"Risk score must be numeric, got {type(score)}")
    if not (0 <= score <= 1):
        raise ValueError(f"Risk score must be [0,1], got {score}")
    return True


# ============================================================================
# UTILITY VALIDATION FUNCTION
# ============================================================================

def validate_output(component_name: str, output: Dict) -> bool:
    """
    Dispatch validation based on component type
    """
    validators = {
        'risk_component': validate_risk_component_output,
        'kalman_hull': validate_kalman_hull_output,
        'ml_ensemble': validate_ml_ensemble_output,
    }
    
    if component_name in validators:
        return validators[component_name](output)
    else:
        raise ValueError(f"Unknown component: {component_name}")
