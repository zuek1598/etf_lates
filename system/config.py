#!/usr/bin/env python3
"""
System Configuration
ALL CONFIGURABLE PARAMETERS IN ONE PLACE

Change values here to modify system behavior without touching code!
Update this file quarterly for rates and thresholds.

Last Updated: October 2025
"""

# ============================================================
# SYSTEM PARAMETERS
# ============================================================
RISK_FREE_RATE = 0.0435  # RBA rate Oct 2024, update quarterly
LOOKBACK_DAYS = 252      # 1 trading year for analysis
MIN_OBSERVATIONS = 100   # Minimum data points required for analysis

# ============================================================
# RISK CLASSIFICATION THRESHOLDS
# ============================================================
RISK_THRESHOLDS = {
    'volatility': {
        'low': 0.12,      # < 12% = low volatility (bonds, defensive)
        'medium': 0.22    # 12-22% = medium, >22% = high (growth, emerging)
    },
    'beta': {
        'low': 0.8,       # < 0.8 = low market sensitivity
        'high': 1.2       # > 1.2 = high market sensitivity
    }
}

# Risk classification comments
RISK_BANDS_NOTES = """
Volatility Bands:
- < 12%: Low volatility (bonds, defensive)
- 12-22%: Medium volatility (standard equity)
- > 22%: High volatility (growth, emerging)

Beta Bands:
- < 0.8: Low market sensitivity
- 0.8-1.2: Market-aligned  
- > 1.2: High market sensitivity
"""

# ============================================================
# VAR/CVAR PARAMETERS
# ============================================================
VAR_CONFIDENCE = 0.95     # 95% confidence level for VaR/CVaR calculations

VAR_THRESHOLDS = {
    'good': -0.05,        # Better than -5% (green)
    'warning': -0.10      # -5% to -10% (yellow), worse = red
}

# ============================================================
# SCORING WEIGHTS
# ============================================================
SCORING_WEIGHTS = {
    'statistical': 0.30,    # Weight for statistical metrics
    'technical': 0.25,      # Weight for technical indicators
    'forecast': 0.25,       # Weight for ML forecast
    'risk_adjusted': 0.20   # Weight for risk-adjusted performance
}

# ============================================================
# SCORING PENALTIES
# ============================================================
PENALTIES = {
    'cvar': {
        'high': -15,         # CVaR < -10% (severe tail risk)
        'medium': -5         # CVaR -10% to -5% (moderate tail risk)
    },
    'liquidity': {
        'very_low': -10,     # < $500k average daily volume
        'low': -5            # $500k - $1M average daily volume
    },
    'expense_ratio': {
        'very_high': -15,    # > 0.75% expense ratio
        'high': -10,         # 0.50% - 0.75%
        'medium': -5         # 0.25% - 0.50%
    },
    'aum': {
        'very_low': -10,     # < $50M AUM
        'low': -5            # $50M - $100M AUM
    },
    'amihud': -5,           # Amihud ratio > 1.0 (poor liquidity)
    'zero_volume': -5       # > 5 days with no volume in last 60
}

# ============================================================
# LIQUIDITY THRESHOLDS
# ============================================================
LIQUIDITY = {
    'volume_thresholds': {
        'high': 5_000_000,      # > $5M = highly liquid
        'medium': 1_000_000,    # $1M - $5M = medium liquidity
        'low': 500_000          # < $500k = illiquid
    },
    'amihud': {
        'liquid': 0.5,          # < 0.5 = good liquidity
        'warning': 1.0          # > 1.0 = poor liquidity
    },
    'zero_days_max': 5          # Max acceptable zero-volume days in 60d
}

# ============================================================
# TECHNICAL INDICATOR PARAMETERS
# ============================================================
TECHNICAL = {
    # Validated technical indicators only
    # KAMA, Stochastic, VWAP, ROC, RSI removed - not statistically validated
}

# Technical indicator scoring thresholds removed with unused indicators

# ============================================================
# STATISTICAL PARAMETERS
# ============================================================
STATISTICAL = {
    't_distribution': {
        'min_df': 2.1,              # Minimum degrees of freedom
        'small_sample_threshold': 100,  # Apply bias correction if n < 100
        'fallback_df': 5.0          # Fallback df if fitting fails
    },
    'sharpe_bounds': {
        'min': -5.0,                # Minimum reasonable Sharpe ratio
        'max': 5.0                  # Maximum reasonable Sharpe ratio
    },
    'volatility_bounds': {
        'min': 0.0,                 # Minimum volatility
        'max': 1.0                  # Maximum volatility (100%)
    },
    'beta_bounds': {
        'min': -2.0,                # Minimum reasonable beta
        'max': 3.0                  # Maximum reasonable beta
    }
}

# ============================================================
# MACHINE LEARNING PARAMETERS
# ============================================================
ML = {
    'training': {
        'train_days': 252,          # Training window (1 year)
        'test_days': 60,            # Test/forecast window (3 months)
        'lookback_window': 60,      # Feature lookback window
        'min_train_samples': 100    # Minimum samples for training
    },
    'models': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 5,
            'random_state': 42
        },
        'ridge': {
            'alpha': 1.0,
            'random_state': 42
        }
    },
    'features': {
        'use_enhanced_features': True,  # Enable enhanced feature set
        'vix_percentile_window': 252,   # Window for VIX percentile
        'correlation_window': 60,       # SPY correlation window
        'volume_ratio_short': 5,        # Short period for volume ratio
        'volume_ratio_long': 20         # Long period for volume ratio
    },
    'ensemble_weights': {
        'random_forest': 0.6,       # Weight for Random Forest
        'ridge': 0.4                # Weight for Ridge Regression
    }
}

# ============================================================
# WALK-FORWARD VALIDATION PARAMETERS
# ============================================================
VALIDATION = {
    'train_days': 252,              # Training window
    'test_days': 60,                # Testing window
    'min_windows': 3,               # Minimum number of test windows
    'thresholds': {
        'min_hit_rate': 0.52        # 52% directional accuracy minimum
    }
}

# Forecast quality thresholds
FORECAST_QUALITY = {
    'hit_rate_thresholds': {
        'excellent': 0.65,          # > 65% directional accuracy
        'good': 0.55,               # > 55%
        'acceptable': 0.50          # > 50% (coin flip)
    },
    'confidence_thresholds': {
        'high': 0.7,                # High confidence
        'medium': 0.4,              # Medium confidence
        'low': 0.0                  # Low confidence
    }
}

# ============================================================
# DATA QUALITY PARAMETERS
# ============================================================
DATA_QUALITY = {
    'tier_thresholds': {
        'tier_1': 756,              # 3+ years of data
        'tier_2': 126,              # 6+ months of data
        'tier_3': 90,               # 3+ months (minimum for analysis)
        'min_days': 90              # Absolute minimum
    },
    'performance': {
        'premium_percentile': 75,   # Top 25% for premium classification
        'exceptional_percentile': 90,  # Top 10% for exceptional
        'aum_threshold': 100_000_000   # $100M minimum AUM for premium
    },
    'penalties': {
        'standard_new': -3,         # Penalty for standard new ETFs
        'very_new': -5              # Penalty for very new ETFs
    }
}

# ============================================================
# DASHBOARD STYLING
# ============================================================
DASHBOARD = {
    'colors': {
        'primary': '#3498db',       # Blue (primary actions)
        'success': '#27ae60',       # Green (positive, success)
        'warning': '#e67e22',       # Orange (warnings, medium risk)
        'danger': '#e74c3c',        # Red (danger, high risk)
        'neutral': '#95a5a6',       # Gray (neutral, unknown)
        'background': '#ecf0f1',    # Light gray background
        'text': '#2c3e50',          # Dark blue-gray text
        'accent': '#d35400'         # Orange accent
    },
    'risk_colors': {
        'LOW': '#27ae60',           # Green for low risk
        'MEDIUM': '#e67e22',        # Orange for medium risk
        'HIGH': '#e74c3c',          # Red for high risk
        'UNKNOWN': '#95a5a6'        # Gray for unknown
    },
    'server': {
        'host': '127.0.0.1',        # Dashboard host
        'port': 8051,               # Dashboard port
        'debug_mode': True          # Enable debug mode
    },
    'cache': {
        'timeout': 3600             # 1 hour cache timeout (seconds)
    }
}

# ============================================================
# DATA PATHS
# ============================================================
DATA_PATHS = {
    'data_dir': 'data',
    'universe': 'data/etf_universe.parquet',
    'rankings': {
        'low': 'data/rankings_low_risk.parquet',
        'medium': 'data/rankings_medium_risk.parquet',
        'high': 'data/rankings_high_risk.parquet'
    },
    'metadata': 'data/analysis_metadata.parquet',
    'fundamentals': 'data/fundamental_data.json',
    'historical': 'data/historical/',
    'validation': 'data/validation_results.json'
}

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
LOGGING = {
    'level': 'INFO',                # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'files': {
        'system': 'logs/system.log',
        'errors': 'logs/errors.log',
        'dashboard': 'logs/dashboard.log'
    },
    'rotation': {
        'max_bytes': 10_000_000,    # 10MB per log file
        'backup_count': 3           # Keep 3 backup files
    }
}

# ============================================================
# SYSTEM BEHAVIOR FLAGS
# ============================================================
SYSTEM_FLAGS = {
    'enable_caching': True,             # Enable data caching
    'enable_validation': True,          # Enable walk-forward validation
    'enable_macro_scoring': True,       # Enable macro/geo scoring
    'enable_enhanced_ml': True,         # Enable enhanced ML features
    'enable_fundamental_penalties': True,  # Apply fundamental penalties
    'enable_liquidity_penalties': True,    # Apply liquidity penalties
    'parallel_processing': False,       # Enable parallel ETF processing
    'save_intermediate_results': False  # Save intermediate calculations
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_category_color(category: str) -> str:
    """Get color for risk category"""
    return DASHBOARD['risk_colors'].get(category.upper(), DASHBOARD['colors']['neutral'])


def get_data_path(key: str) -> str:
    """Get data file path by key"""
    if key in DATA_PATHS['rankings']:
        return DATA_PATHS['rankings'][key]
    return DATA_PATHS.get(key, '')


def is_feature_enabled(feature: str) -> bool:
    """Check if a system feature is enabled"""
    return SYSTEM_FLAGS.get(feature, False)


def get_penalty(category: str, subcategory: str = None) -> float:
    """Get penalty value for a category"""
    if subcategory and category in PENALTIES and isinstance(PENALTIES[category], dict):
        return PENALTIES[category].get(subcategory, 0)
    return PENALTIES.get(category, 0)


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

def print_config_summary():
    """Print configuration summary"""
    print("="*60)
    print("SYSTEM CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Risk-Free Rate: {RISK_FREE_RATE:.4f} ({RISK_FREE_RATE*100:.2f}%)")
    print(f"Lookback Days: {LOOKBACK_DAYS}")
    print(f"VaR Confidence: {VAR_CONFIDENCE:.2%}")
    print(f"ML Train/Test: {ML['training']['train_days']}/{ML['training']['test_days']} days")
    print(f"Enhanced Features: {'Enabled' if ML['features']['use_enhanced_features'] else 'Disabled'}")
    print(f"Dashboard Port: {DASHBOARD['server']['port']}")
    print("="*60)


if __name__ == "__main__":
    # Display configuration when run directly
    print_config_summary()
    
    # Show some key parameters
    print("\nKey Thresholds:")
    print(f"  Low Vol Threshold: {RISK_THRESHOLDS['volatility']['low']:.1%}")
    print(f"  Medium Vol Threshold: {RISK_THRESHOLDS['volatility']['medium']:.1%}")
    print(f"  High Liquidity: ${LIQUIDITY['volume_thresholds']['high']:,.0f}")
    print(f"  Acceptable MAE: {VALIDATION['thresholds']['acceptable_mae']:.1%}")
    print(f"  Min Hit Rate: {VALIDATION['thresholds']['min_hit_rate']:.1%}")

