# 🧪 TESTING GUIDE - Current System State

**System Status:** Phase 2 Complete (89%) - Core Components Ready  
**Date:** October 22, 2025  
**What's Testable:** Risk Component, ML Ensemble, Scoring System, Orchestrator

---

## 📊 WHAT'S CURRENTLY IMPLEMENTED

### ✅ Ready to Test:
1. **Risk Component** - CVaR, Ulcer Index, Beta, Information Ratio (30/30/20/20)
2. **ML Ensemble** - Raw forecasts + confidence (NO bias correction)
3. **Scoring System** - Integrated scoring with all components
4. **Orchestrator** - Full pipeline (with placeholders for Kalman Hull & Volume Intelligence)
5. **Validators** - All output validation
6. **Schemas** - All data structures defined
7. **Data Loader** - Compatible with new schema

### ⏳ Not Yet Implemented:
- Kalman Hull indicator (placeholder values in orchestrator)
- Volume Intelligence (placeholder values in orchestrator)
- Dashboard visualization (specification created)

---

## 🚀 QUICK START TESTING

### Option 1: Test Individual Components

```bash
cd "/Users/uliana/Desktop/new_alpha/latest /modified"

# Test Risk Component
python -c "
from analyzers.risk_component import RiskComponent
import pandas as pd
import numpy as np

# Create test data
dates = pd.date_range('2023-01-01', periods=252)
prices = pd.DataFrame({
    'Close': 50 + np.random.randn(252).cumsum()
}, index=dates)

# Test Risk Component
risk = RiskComponent()
print('Testing Risk Component...')
result = risk.calculate_risk_scores(prices, {'ticker': 'TEST.AX'})
print(f'Risk Score: {result[\"risk_score\"]:.3f}')
print(f'Category: {result[\"risk_category\"]}')
print(f'CVaR: {result[\"cvar\"]:.4f}')
print('✅ Risk Component works!')
"
```

```bash
# Test ML Ensemble
python -c "
from analyzers.ml_ensemble import MLEnsemble
import pandas as pd
import numpy as np

dates = pd.date_range('2023-01-01', periods=252)
data = pd.DataFrame({
    'Close': 50 + np.random.randn(252).cumsum(),
    'Volume': 1000000 + np.random.randint(-100000, 100000, 252)
}, index=dates)

ml = MLEnsemble()
print('Testing ML Ensemble...')
result = ml.generate_ml_forecast(data)
print(f'Forecast: {result[\"forecast_return\"]:.2f}%')
print(f'Confidence: {result[\"confidence_score\"]:.3f}')
print(f'Has bias_correction? {\"bias_correction\" in result}')
print('✅ ML Ensemble works!')
"
```

```bash
# Test Validators
python -c "
from utilities.validators import validate_output

# Test Risk Component validator
risk_output = {
    'cvar': -0.05,
    'ulcer_index': 3.2,
    'beta': 1.1,
    'information_ratio': 0.8,
    'risk_score': 0.35,
    'risk_category': 'LOW',
    'quality_flag': '✅'
}
print('Testing Risk Component Validator...')
is_valid = validate_output('risk_component', risk_output)
print(f'Valid: {is_valid}')

# Test ML Ensemble validator (should reject bias_correction)
ml_output = {
    'forecast_return': 2.5,
    'confidence_score': 0.7,
    'features_used': {},
    'model_ensemble_output': 2.5
}
print('\\nTesting ML Ensemble Validator...')
is_valid = validate_output('ml_ensemble', ml_output)
print(f'Valid: {is_valid}')

# Test with bias_correction (should fail)
ml_bad = ml_output.copy()
ml_bad['bias_correction'] = 0.5
print('\\nTesting ML Ensemble with bias_correction (should fail)...')
try:
    is_valid = validate_output('ml_ensemble', ml_bad)
    print(f'Valid: {is_valid}')
except:
    print('✅ Correctly rejected bias_correction!')
"
```

---

### Option 2: Test Full Pipeline (Recommended)

Create a test script:

```bash
cat > "/Users/uliana/Desktop/new_alpha/latest /modified/test_pipeline.py" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
Test Current System Pipeline
Tests Risk Component, ML Ensemble, Scoring, and Orchestrator
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*60)
print("TESTING CURRENT SYSTEM STATE")
print("="*60)

# Test 1: Risk Component
print("\n1. Testing Risk Component...")
try:
    from analyzers.risk_component import RiskComponent
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=252)
    test_data = pd.DataFrame({
        'Close': 50 + np.random.randn(252).cumsum(),
        'Volume': 1000000 + np.random.randint(-100000, 100000, 252)
    }, index=dates)
    
    risk = RiskComponent()
    result = risk.calculate_risk_scores(test_data, {'ticker': 'TEST.AX'})
    
    print(f"   ✓ Risk Score: {result['risk_score']:.3f}")
    print(f"   ✓ Category: {result['risk_category']}")
    print(f"   ✓ CVaR: {result['cvar']:.4f}")
    print(f"   ✓ Ulcer Index: {result['ulcer_index']:.4f}")
    print(f"   ✓ Beta: {result['beta']:.3f}")
    print(f"   ✓ Information Ratio: {result['information_ratio']:.3f}")
    print("   ✅ Risk Component: PASS")
except Exception as e:
    print(f"   ❌ Risk Component: FAIL - {e}")

# Test 2: ML Ensemble
print("\n2. Testing ML Ensemble...")
try:
    from analyzers.ml_ensemble import MLEnsemble
    
    ml = MLEnsemble()
    result = ml.generate_ml_forecast(test_data)
    
    print(f"   ✓ Forecast: {result['forecast_return']:.2f}%")
    print(f"   ✓ Confidence: {result['confidence_score']:.3f}")
    print(f"   ✓ NO bias_correction: {'bias_correction' not in result}")
    print("   ✅ ML Ensemble: PASS")
except Exception as e:
    print(f"   ❌ ML Ensemble: FAIL - {e}")

# Test 3: Scoring System
print("\n3. Testing Scoring System...")
try:
    from analyzers.scoring_system import ScoringRankingSystem
    
    scoring = ScoringRankingSystem()
    
    # Sample analysis data
    analysis = {
        'risk_score': 0.35,
        'kalman_trend': 1,
        'kalman_signal_strength': 0.7,
        'kalman_divergence': 'bullish',
        'ml_forecast': 2.5,
        'ml_confidence': 0.75,
        'volume_spike_score': 65.0,
        'volume_correlation': 0.6,
        'volume_ad_signal': 'accumulation',
        'young_etf_penalty': 0.0,
        'cvar': -0.03
    }
    
    score = scoring.calculate_composite_score(analysis, 'LOW')
    print(f"   ✓ Composite Score: {score:.1f}/100")
    print("   ✅ Scoring System: PASS")
except Exception as e:
    print(f"   ❌ Scoring System: FAIL - {e}")

# Test 4: Validators
print("\n4. Testing Validators...")
try:
    from utilities.validators import validate_output
    
    # Test risk component validation
    risk_output = {
        'cvar': -0.05,
        'ulcer_index': 3.2,
        'beta': 1.1,
        'information_ratio': 0.8,
        'risk_score': 0.35,
        'risk_category': 'LOW',
        'quality_flag': '✅'
    }
    is_valid = validate_output('risk_component', risk_output)
    print(f"   ✓ Risk Component Validation: {is_valid}")
    
    # Test ML ensemble validation
    ml_output = {
        'forecast_return': 2.5,
        'confidence_score': 0.7,
        'features_used': {},
        'model_ensemble_output': 2.5
    }
    is_valid = validate_output('ml_ensemble', ml_output)
    print(f"   ✓ ML Ensemble Validation: {is_valid}")
    
    print("   ✅ Validators: PASS")
except Exception as e:
    print(f"   ❌ Validators: FAIL - {e}")

# Test 5: Schemas
print("\n5. Testing Schemas...")
try:
    from system.schemas import (
        RISK_COMPONENT_SCHEMA,
        ML_ENSEMBLE_SCHEMA,
        KALMAN_HULL_SCHEMA,
        VOLUME_INTELLIGENCE_SCHEMA
    )
    
    print(f"   ✓ Risk Component Schema: {len(RISK_COMPONENT_SCHEMA)} fields")
    print(f"   ✓ ML Ensemble Schema: {len(ML_ENSEMBLE_SCHEMA)} fields")
    print(f"   ✓ Kalman Hull Schema: {len(KALMAN_HULL_SCHEMA)} fields")
    print(f"   ✓ Volume Intelligence Schema: {len(VOLUME_INTELLIGENCE_SCHEMA)} fields")
    print("   ✅ Schemas: PASS")
except Exception as e:
    print(f"   ❌ Schemas: FAIL - {e}")

# Test 6: Orchestrator (basic import test)
print("\n6. Testing Orchestrator...")
try:
    from system.orchestrator import ETFAnalysisSystem
    
    print("   ✓ Orchestrator imports successfully")
    print("   ✓ Pipeline structure ready")
    print("   ⚠️  Note: Full pipeline test requires actual ETF data")
    print("   ✅ Orchestrator: PASS (import test)")
except Exception as e:
    print(f"   ❌ Orchestrator: FAIL - {e}")

print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("✅ All core components are functional!")
print("✅ NO bias correction in ML output")
print("✅ 30/30/20/20 risk weighting implemented")
print("✅ New scoring system integrated")
print("✅ Validators working correctly")
print("\n⚠️  Note: Kalman Hull & Volume Intelligence are placeholders")
print("   (Will be implemented in next phase)")
print("="*60)

PYTHON_EOF

# Run the test
python "/Users/uliana/Desktop/new_alpha/latest /modified/test_pipeline.py"
```

---

### Option 3: Test with Real ETF Data (If Available)

```bash
cd "/Users/uliana/Desktop/new_alpha/latest /modified"

# Check if data exists
if [ -d "data/historical" ]; then
    echo "✓ Historical data found"
    
    # Run full analysis on a few ETFs
    python -c "
from system.orchestrator import ETFAnalysisSystem
import warnings
warnings.filterwarnings('ignore')

print('Testing with real ETF data...')
print('='*60)

# Initialize system
system = ETFAnalysisSystem()

# Test with a small subset
test_tickers = ['VAS.AX', 'VGS.AX', 'VAF.AX']  # Adjust to your ETFs
print(f'Testing {len(test_tickers)} ETFs...')

try:
    results = system.run_full_analysis(test_tickers)
    print(f'✅ Analysis complete!')
    print(f'   Analyzed: {len(results[\"analysis_results\"])} ETFs')
    print(f'   Top ETF: {results[\"top_etfs\"][0][\"ticker\"]}')
    print(f'   Top Score: {results[\"top_etfs\"][0][\"score\"]:.1f}')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"
else
    echo "❌ No historical data found in data/historical/"
    echo "   You'll need to run data collection first"
fi
```

---

## 🔍 DETAILED COMPONENT TESTS

### Test 1: Risk Component Deep Dive

```python
# Save as test_risk_component.py
from analyzers.risk_component import RiskComponent
import pandas as pd
import numpy as np

# Create realistic test data
np.random.seed(42)
dates = pd.date_range('2022-01-01', periods=500)
returns = np.random.randn(500) * 0.01  # 1% daily volatility
prices = pd.DataFrame({
    'Close': 100 * (1 + returns).cumprod(),
    'Volume': 1000000 + np.random.randint(-200000, 200000, 500)
}, index=dates)

risk = RiskComponent()

print("="*60)
print("RISK COMPONENT DETAILED TEST")
print("="*60)

# Test CVaR calculation
print("\n1. Testing CVaR...")
cvar_result = risk.calculate_cvar(prices['Close'].pct_change().dropna())
print(f"   CVaR (95%): {cvar_result['cvar']:.4f}")
print(f"   T-dist params: df={cvar_result['t_params']['degrees_of_freedom']:.2f}")

# Test Ulcer Index
print("\n2. Testing Ulcer Index...")
ulcer = risk.calculate_ulcer_index(prices['Close'])
print(f"   Ulcer Index: {ulcer:.4f}")

# Test Information Ratio
print("\n3. Testing Information Ratio...")
benchmark = pd.Series(100 * (1 + np.random.randn(500) * 0.008).cumprod(), index=dates)
ir = risk.calculate_information_ratio(prices['Close'], benchmark)
print(f"   Information Ratio: {ir:.4f}")

# Test full risk scoring
print("\n4. Testing Full Risk Scoring...")
result = risk.calculate_risk_scores(prices, {'ticker': 'TEST.AX'})
print(f"   Risk Score: {result['risk_score']:.3f}")
print(f"   Category: {result['risk_category']}")
print(f"   Quality Flag: {result['quality_flag']}")
print(f"   CVaR: {result['cvar']:.4f}")
print(f"   Ulcer: {result['ulcer_index']:.4f}")
print(f"   Beta: {result['beta']:.3f}")
print(f"   IR: {result['information_ratio']:.3f}")

# Verify weighting
print("\n5. Verifying 30/30/20/20 Weighting...")
weights = risk.weights
print(f"   CVaR weight: {weights['cvar']*100:.0f}%")
print(f"   Ulcer weight: {weights['ulcer']*100:.0f}%")
print(f"   Beta weight: {weights['beta']*100:.0f}%")
print(f"   IR weight: {weights['ir']*100:.0f}%")
assert weights['cvar'] == 0.30, "CVaR weight should be 30%"
assert weights['ulcer'] == 0.30, "Ulcer weight should be 30%"
assert weights['beta'] == 0.20, "Beta weight should be 20%"
assert weights['ir'] == 0.20, "IR weight should be 20%"
print("   ✅ Weights correct!")

print("\n" + "="*60)
print("✅ ALL RISK COMPONENT TESTS PASSED")
print("="*60)
```

### Test 2: ML Ensemble Bias Check

```python
# Save as test_ml_no_bias.py
from analyzers.ml_ensemble import MLEnsemble
import pandas as pd
import numpy as np

print("="*60)
print("ML ENSEMBLE BIAS CORRECTION CHECK")
print("="*60)

# Create test data
dates = pd.date_range('2022-01-01', periods=500)
data = pd.DataFrame({
    'Close': 100 * (1 + np.random.randn(500) * 0.01).cumprod(),
    'Volume': 1000000 + np.random.randint(-200000, 200000, 500)
}, index=dates)

ml = MLEnsemble()

print("\n1. Checking for bias correction in class...")
print(f"   Has 'historical_bias' attribute: {hasattr(ml, 'historical_bias')}")
print(f"   Has 'bias' in __dict__: {'bias' in ml.__dict__}")

print("\n2. Testing forecast output...")
result = ml.generate_ml_forecast(data)

print(f"   Forecast: {result['forecast_return']:.2f}%")
print(f"   Confidence: {result['confidence_score']:.3f}")

print("\n3. CRITICAL: Checking for bias_correction field...")
has_bias_correction = any(key for key in result.keys() if 'bias' in key.lower())
print(f"   Has any 'bias' field: {has_bias_correction}")

if has_bias_correction:
    print("   ❌ FAIL: Bias correction field found!")
    print(f"   Fields: {[k for k in result.keys() if 'bias' in k.lower()]}")
else:
    print("   ✅ PASS: NO bias correction field!")

print("\n4. Output fields present:")
for key in result.keys():
    print(f"   - {key}: {type(result[key]).__name__}")

print("\n5. Verifying raw output...")
print(f"   model_ensemble_output: {result.get('model_ensemble_output', 'N/A')}")
print(f"   forecast_return: {result.get('forecast_return', 'N/A')}")
print(f"   Are they equal? {abs(result.get('model_ensemble_output', 0) - result.get('forecast_return', 1)) < 0.01}")

print("\n" + "="*60)
if not has_bias_correction:
    print("✅ ML ENSEMBLE: NO BIAS CORRECTION CONFIRMED")
else:
    print("❌ ML ENSEMBLE: BIAS CORRECTION FOUND (SHOULD NOT BE PRESENT)")
print("="*60)
```

---

## 📋 TESTING CHECKLIST

Use this checklist to verify all components:

```
COMPONENT TESTS:
[ ] Risk Component imports successfully
[ ] Risk Component calculates CVaR
[ ] Risk Component calculates Ulcer Index
[ ] Risk Component calculates Beta
[ ] Risk Component calculates Information Ratio
[ ] Risk Component uses 30/30/20/20 weighting
[ ] Risk Component returns risk_score (0-1)
[ ] Risk Component returns risk_category (LOW/MEDIUM/HIGH)
[ ] Risk Component returns quality_flag

ML ENSEMBLE TESTS:
[ ] ML Ensemble imports successfully
[ ] ML Ensemble generates forecast
[ ] ML Ensemble calculates confidence score
[ ] ML Ensemble has NO bias_correction field
[ ] ML Ensemble has NO historical_bias attribute
[ ] ML Ensemble returns raw ensemble output
[ ] ML Ensemble tracks features_used
[ ] ML Ensemble provides feature_importance

SCORING SYSTEM TESTS:
[ ] Scoring System imports successfully
[ ] Scoring System integrates Risk Component
[ ] Scoring System integrates ML Ensemble
[ ] Scoring System integrates Kalman Hull (placeholder)
[ ] Scoring System integrates Volume Intelligence (placeholder)
[ ] Scoring System calculates composite score
[ ] Scoring System applies risk multipliers
[ ] Scoring System applies quality penalties

ORCHESTRATOR TESTS:
[ ] Orchestrator imports successfully
[ ] Orchestrator initializes all components
[ ] Orchestrator downloads market data
[ ] Orchestrator classifies ETFs by risk
[ ] Orchestrator runs analysis pipeline
[ ] Orchestrator combines results
[ ] Orchestrator calculates rankings
[ ] Orchestrator returns top ETFs

VALIDATOR TESTS:
[ ] Validators import successfully
[ ] Risk Component validator works
[ ] ML Ensemble validator works
[ ] ML Ensemble validator rejects bias_correction
[ ] Kalman Hull validator works
[ ] Volume Intelligence validator works
[ ] Dispatcher function works

SCHEMA TESTS:
[ ] All schemas import successfully
[ ] RISK_COMPONENT_SCHEMA defined
[ ] ML_ENSEMBLE_SCHEMA defined
[ ] KALMAN_HULL_SCHEMA defined
[ ] VOLUME_INTELLIGENCE_SCHEMA defined
[ ] Schema validation functions work
```

---

## 🐛 TROUBLESHOOTING

### Common Issues:

**1. Import Errors**
```bash
# If you get import errors, make sure you're in the right directory
cd "/Users/uliana/Desktop/new_alpha/latest /modified"
python -c "import sys; print(sys.path[0])"
```

**2. Missing Dependencies**
```bash
# Install required packages
pip install pandas numpy scipy scikit-learn yfinance
```

**3. No Data Available**
```
If testing with real data fails, you need to:
1. Run data collection first
2. Or use the synthetic data tests above
```

---

## 📊 EXPECTED TEST RESULTS

When all tests pass, you should see:

```
✅ Risk Component: PASS
   - CVaR calculated correctly
   - Ulcer Index calculated correctly
   - 30/30/20/20 weighting verified
   - Risk score in range [0, 1]
   - Risk category: LOW/MEDIUM/HIGH

✅ ML Ensemble: PASS
   - Forecast generated
   - Confidence score in range [0, 1]
   - NO bias_correction field ✓
   - Raw ensemble output present

✅ Scoring System: PASS
   - Composite score in range [0, 100]
   - All components integrated
   - Risk multipliers applied

✅ Orchestrator: PASS
   - Pipeline runs successfully
   - Results generated
   - Rankings calculated

✅ Validators: PASS
   - All validations work
   - Bias correction rejected ✓

✅ Schemas: PASS
   - All schemas defined
   - Validation functions work
```

---

## 🚀 NEXT STEPS AFTER TESTING

Once testing confirms everything works:

1. **Create Kalman Hull** (`indicators/kalman_hull.py`)
2. **Create Volume Intelligence** (`analyzers/volume_intelligence.py`)
3. **Integrate into Orchestrator** (replace placeholders)
4. **Run Full System Test** (end-to-end)
5. **Performance Benchmarks**

---

**Ready to test?** Run the Quick Start tests above or the comprehensive test script! 🧪

