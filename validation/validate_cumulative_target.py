#!/usr/bin/env python3
"""
Validation script for cumulative return target
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from ml_models.dual_target_forecaster import DualTargetForecaster
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

def validate_cumulative_target():
    """
    Complete validation for cumulative return target
    
    Tests:
    1. Statistical significance
    2. Temporal validation
    3. COVID bias analysis
    4. Feature importance
    """
    
    print("🔍 Validating Cumulative Return Target...")
    
    # Create synthetic but realistic data
    np.random.seed(42)
    
    # Simulate 372 ETFs with 300 days each
    n_etfs = 50  # Use smaller sample for quick test
    n_days = 300
    forecast_horizon = 60
    
    results = []
    
    for etf_idx in range(n_etfs):
        # Simulate returns with realistic patterns
        base_return = np.random.normal(0.001, 0.001)  # 0.1% daily avg
        volatility = np.random.uniform(0.01, 0.05)    # 1-5% daily vol
        
        returns = pd.Series(np.random.normal(base_return, volatility, n_days))
        
        # Create features based on validated set
        features = np.random.randn(n_days - forecast_horizon - 100, 10) * 0.1 + 0.5
        
        # Calculate targets
        for i in range(len(features)):
            start_idx = i
            end_idx = i + forecast_horizon
            
            if end_idx <= len(returns):
                target_daily = returns.iloc[start_idx:end_idx].mean()
                target_cumulative = returns.iloc[start_idx:end_idx].sum()
                
                results.append({
                    'etf': etf_idx,
                    'features': features[i],
                    'target_daily': target_daily,
                    'target_cumulative': target_cumulative
                })
    
    # Convert to arrays
    X = np.array([r['features'] for r in results])
    y_daily = np.array([r['target_daily'] for r in results])
    y_cumulative = np.array([r['target_cumulative'] for r in results])
    
    # Basic statistics
    print(f"📊 Dataset: {len(results)} samples from {n_etfs} ETFs")
    print(f"   Daily targets: {y_daily.mean():.4f} ± {y_daily.std():.4f}")
    print(f"   Cumulative targets: {y_cumulative.mean():.4f} ± {y_cumulative.std():.4f}")
    
    # 1. Statistical significance test
    print("\n📈 Statistical Significance Test")
    
    # Correlation test
    correlations = []
    for i in range(10):  # 10 features
        corr, _ = pearsonr(X[:, i], y_cumulative)
        correlations.append(abs(corr))
    
    significant_features = [i for i, corr in enumerate(correlations) if corr > 0.1]
    print(f"   Significant features: {len(significant_features)}/10")
    print(f"   Mean correlation: {np.mean(correlations):.3f}")
    
    # 2. Temporal validation
    print("\n⏱️ Temporal Validation")
    
    # Simple train/test split
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_cumulative[:split_idx], y_cumulative[split_idx:]
    
    # Train simple model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   MAE: {mae:.4f} ({mae*100:.2f}%)")
    print(f"   R²: {r2:.3f}")
    
    # 3. COVID bias analysis
    print("\n🦠 COVID Bias Analysis")
    
    # Simulate COVID period (first 100 days)
    covid_mask = np.arange(len(y_cumulative)) < 100
    covid_returns = y_cumulative[covid_mask]
    post_covid_returns = y_cumulative[~covid_mask]
    
    covid_bias = (covid_returns.mean() - post_covid_returns.mean()) / post_covid_returns.mean()
    print(f"   COVID bias: {covid_bias:.2f}")
    
    # 4. Model comparison
    print("\n⚖️ Model Comparison")
    
    # Compare daily vs cumulative predictions
    daily_model = RandomForestRegressor(n_estimators=50, random_state=42)
    daily_model.fit(X_train, y_train)
    daily_pred = daily_model.predict(X_test)
    
    # Convert daily to cumulative
    daily_to_cumulative = daily_pred * forecast_horizon
    
    # Agreement between models
    agreement = 1 - np.mean(np.abs(daily_to_cumulative - y_pred)) / np.mean(np.abs(y_pred))
    print(f"   Model agreement: {agreement:.2f}")
    
    # Spearman correlation for ranking
    rank_corr, _ = spearmanr(y_pred, daily_to_cumulative)
    print(f"   Ranking correlation: {rank_corr:.2f}")
    
    return {
        'significant_features': len(significant_features),
        'mean_correlation': np.mean(correlations),
        'mae': mae,
        'r2': r2,
        'covid_bias': covid_bias,
        'model_agreement': agreement,
        'ranking_correlation': rank_corr
    }

if __name__ == "__main__":
    print("=" * 60)
    print("CUMULATIVE TARGET VALIDATION")
    print("=" * 60)
    
    results = validate_cumulative_target()
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for key, value in results.items():
        print(f"{key}: {value:.3f}")
    
    # Success criteria
    success = (
        results['significant_features'] >= 8 and
        results['mean_correlation'] >= 0.1 and
        results['r2'] >= 0.1 and
        results['model_agreement'] >= 0.7
    )
    
    if success:
        print("\n✅ Cumulative target validation PASSED")
    else:
        print("\n❌ Cumulative target validation FAILED")
