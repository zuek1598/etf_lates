#!/usr/bin/env python3
"""
PRODUCTION CONFIGURATION - Final Validated Settings
Implements the optimized 10-feature set with balanced scoring methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json

class ProductionConfig:
    """
    Production configuration with validated ML features and settings
    Based on comprehensive statistical validation and balanced scoring
    """
    
    # FINAL PRODUCTION FEATURES - Validated through rigorous testing
    PRODUCTION_FEATURES = [
        'volatility',              # Basic Technical - Highest balanced score (0.744)
        'gold_equity_corr',        # Regime - Strong cross-asset correlation (0.673)
        'volatility_level',        # MACD-V - Robust volatility normalization (0.660)
        'signal_quality',          # MACD-V - Consistent signal strength (0.659)
        'vix_rates_corr',          # Regime - VIX-rates correlation (0.602)
        'cross_asset_dispersion',  # Regime - Cross-asset risk dispersion (0.583)
        'macd_histogram',          # MACD-V - Momentum divergence (0.569)
        'macd_signal',             # MACD-V - Standard MACD signal (0.569)
        'momentum',                # Basic Technical - Highest temporal importance (0.558)
        'equity_bonds_corr'        # Regime - Equity-bonds correlation (0.453)
    ]
    
    # Feature categories for analysis and documentation
    FEATURE_CATEGORIES = {
        'volatility': 'basic_technical',
        'momentum': 'basic_technical',
        'volatility_level': 'macd_v',
        'signal_quality': 'macd_v',
        'macd_histogram': 'macd_v',
        'macd_signal': 'macd_v',
        'gold_equity_corr': 'regime',
        'vix_rates_corr': 'regime',
        'cross_asset_dispersion': 'regime',
        'equity_bonds_corr': 'regime'
    }
    
    # Balanced scoring weights used for feature selection
    BALANCED_SCORING_WEIGHTS = {
        'cv_improvement': 0.4,      # 40% weight on cross-validation performance
        'temporal_importance': 0.3,  # 30% weight on temporal validation importance
        'correlation': 0.3          # 30% weight on correlation significance
    }
    
    # Validation thresholds and requirements
    VALIDATION_THRESHOLDS = {
        'significance_level': 0.05,      # Statistical significance threshold
        'min_cv_improvement': 0.02,      # Minimum 2% CV improvement
        'min_temporal_importance': 0.02, # Minimum temporal importance
        'max_correlation': 0.7,          # Maximum allowed correlation between features
        'min_samples_per_feature': 15    # Minimum samples per feature for reliability
    }
    
    # Model hyperparameters (optimized through validation)
    MODEL_CONFIG = {
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        'ridge_regression': {
            'alpha': 1.0,
            'random_state': 42
        }
    }
    
    # Performance expectations based on validation
    PERFORMANCE_EXPECTATIONS = {
        'expected_improvement_range': '20-30%',
        'conservative_estimate': '20%',
        'realistic_target': '25%',
        'optimistic_scenario': '30%',
        'risk_level': 'LOW',
        'temporal_robustness': 'HIGH',
        'generalization_confidence': 'HIGH'
    }
    
    # COVID-19 bias adjustments
    COVID_BIAS_ADJUSTMENTS = {
        'volatility_bias_detected': True,
        'bias_magnitude': 0.455,  # 45.5% overestimation during COVID
        'adjustment_required': True,
        'adjustment_factor': 0.545,  # Reduce volatility expectations by 45.5%
        'monitoring_required': True
    }
    
    # Data requirements for production
    DATA_REQUIREMENTS = {
        'min_history_length': 120,      # Minimum 120 days of price data
        'optimal_history_length': 252,  # Optimal 1 year of data
        'required_columns': ['Close'],
        'optional_columns': ['Volume'],
        'data_quality_threshold': 0.95   # 95% data quality required
    }
    
    # Monitoring and recalibration settings
    MONITORING_CONFIG = {
        'performance_monitoring_frequency': 'monthly',
        'recalibration_frequency': 'quarterly',
        'min_samples_for_recalibration': 100,
        'performance_degradation_threshold': 0.1,  # 10% degradation triggers alert
        'feature_drift_threshold': 0.2            # 20% feature drift triggers alert
    }
    
    @classmethod
    def get_production_config(cls) -> Dict:
        """
        Get complete production configuration
        
        Returns:
            Dictionary with all production settings
        """
        return {
            'features': cls.PRODUCTION_FEATURES,
            'feature_categories': cls.FEATURE_CATEGORIES,
            'balanced_scoring_weights': cls.BALANCED_SCORING_WEIGHTS,
            'validation_thresholds': cls.VALIDATION_THRESHOLDS,
            'model_config': cls.MODEL_CONFIG,
            'performance_expectations': cls.PERFORMANCE_EXPECTATIONS,
            'covid_bias_adjustments': cls.COVID_BIAS_ADJUSTMENTS,
            'data_requirements': cls.DATA_REQUIREMENTS,
            'monitoring_config': cls.MONITORING_CONFIG,
            'version': '1.0.0',
            'last_updated': pd.Timestamp.now().isoformat()
        }
    
    @classmethod
    def save_config(cls, filepath: str = 'config/production_config.json'):
        """
        Save production configuration to file
        
        Args:
            filepath: Path to save configuration
        """
        config = cls.get_production_config()
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"✅ Production configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str = 'config/production_config.json') -> Dict:
        """
        Load production configuration from file
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            print(f"✅ Production configuration loaded from {filepath}")
            return config
            
        except FileNotFoundError:
            print(f"⚠️ Configuration file not found: {filepath}")
            print(f"   Using default configuration")
            return cls.get_production_config()
    
    @classmethod
    def validate_config(cls, config: Dict) -> bool:
        """
        Validate configuration completeness
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_sections = [
            'features', 'feature_categories', 'balanced_scoring_weights',
            'validation_thresholds', 'model_config', 'performance_expectations'
        ]
        
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required configuration section: {section}")
                return False
        
        # Validate feature count
        if len(config['features']) != 10:
            print(f"❌ Expected 10 production features, found {len(config['features'])}")
            return False
        
        # Validate feature categories
        for feature in config['features']:
            if feature not in config['feature_categories']:
                print(f"❌ Missing category for feature: {feature}")
                return False
        
        print("✅ Configuration validation passed")
        return True


def main():
    """Initialize and save production configuration"""
    print("🔧 PRODUCTION CONFIGURATION - FINAL VALIDATED SETTINGS")
    print("=" * 60)
    
    # Create and save configuration
    config = ProductionConfig.get_production_config()
    
    print("📊 PRODUCTION CONFIGURATION SUMMARY:")
    print(f"   Features: {len(config['features'])}")
    print(f"   Feature categories: {len(set(config['feature_categories'].values()))}")
    print(f"   Expected performance: {config['performance_expectations']['expected_improvement_range']}")
    print(f"   Risk level: {config['performance_expectations']['risk_level']}")
    print(f"   COVID bias adjustment: {'Required' if config['covid_bias_adjustments']['adjustment_required'] else 'Not required'}")
    
    print(f"\n🎯 PRODUCTION FEATURES:")
    category_counts = {}
    for feature in config['features']:
        category = config['feature_categories'][feature]
        category_counts[category] = category_counts.get(category, 0) + 1
        print(f"   • {feature:<25} ({category})")
    
    print(f"\n📈 FEATURE BREAKDOWN:")
    for category, count in category_counts.items():
        print(f"   • {category.replace('_', ' ').title()}: {count} features")
    
    # Save configuration
    ProductionConfig.save_config()
    
    # Validate configuration
    is_valid = ProductionConfig.validate_config(config)
    
    if is_valid:
        print(f"\n✅ Production configuration ready for deployment!")
    else:
        print(f"\n❌ Configuration validation failed!")
    
    return config if is_valid else None


if __name__ == "__main__":
    config = main()
