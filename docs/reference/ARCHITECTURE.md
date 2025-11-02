# ARCHITECTURE & ACTION PLAN - GEOMACRO ETF ANALYSIS SYSTEM v2

**Last Updated:** October 22, 2025  
**Status:** PLANNING PHASE - READY FOR IMPLEMENTATION

---

## 📋 CORE INSTRUCTION FOR ALL DEVELOPMENT

> **PRIMARY DIRECTIVE:** Always write the most efficient and shortest code possible while meeting all functionality. When working on a hard part - ultrathink. Do not create or invent new methodologies other than what is stated.

---

## 🎯 SECTION 1: SYSTEM EVOLUTION SUMMARY

### What Stays (No Changes)
- ✅ Conditional Sharpe Ratio calculations
- ✅ CVaR (Parametric) with T-distribution fitting
- ✅ Ulcer Index with risk-adaptive lookback periods
- ✅ Beta calculations (vs benchmark)
- ✅ Information Ratio calculations
- ✅ Walk-Forward Validation system (full validation infrastructure)
- ✅ MAE Quality Flag system with visual badges (✅/~/⚠️/🔴)
- ✅ Data Quality Tiers (4-tier system with penalties)
- ✅ Risk Classification Matrix (LOW/MEDIUM/HIGH based on volatility & beta)
- ✅ ETF Database structure and data loading
- ✅ Macro & Geopolitical Frameworks

### What's Removed
- ❌ VaR (Parametric/Historical) - Keep ONLY CVaR
- ❌ Holdings Momentum component - Don't collect holdings data
- ❌ Bias Correction logic - Use raw ML output with confidence scores
- ❌ KAMA, RSI, Stochastic, VWAP - Replace entirely with Kalman Hull
- ❌ Unused utility functions after refactor

### What Changes (Major Redesign)
- 🔄 **Momentum Layer** → Adaptive Kalman Hull Supertrend (NEW FILE: `indicators/kalman_hull.py`)
- 🔄 **Volume Analysis** → Volume Intelligence component (NEW APPROACH)
- 🔄 **Statistical Component** → **Risk Component** (RENAME + NEW WEIGHTS)
  - CVaR: 30% (was implicit)
  - Ulcer Index: 30% (was implicit)
  - Beta: 20% (was implicit)
  - Information Ratio: 20% (was implicit)
- 🔄 **Forecasting Engine** → **ML Ensemble** (RENAME + OUTPUT CHANGE)
  - Remove bias correction
  - Add confidence scores
  - Output: Features → ML Ensemble → Confidence
- 🔄 **Output Schemas** - Complete redesign of technical indicators section
- 🔄 **Validation Rules** - Update validators.py for new indicator ranges

---

## 🏗️ SECTION 2: NEW DIRECTORY STRUCTURE

```
modified/
├── indicators/                          [NEW DIRECTORY]
│   ├── __init__.py
│   └── kalman_hull.py                  [NEW FILE - Kalman Hull Supertrend]
│
├── analyzers/                           [UPDATED]
│   ├── risk_component.py                [RENAMED from statistical_analyzer.py]
│   ├── volume_intelligence.py           [NEW FILE]
│   ├── ml_ensemble.py                   [RENAMED from forecasting_engine.py]
│   ├── etf_risk_classifier.py           [KEEP AS IS]
│   ├── scoring_system.py                [UPDATED - WEIGHTS CHANGE]
│   └── __init__.py
│
├── data_manager/                        [NO CHANGES]
│   ├── data_manager.py
│   ├── etf_database.py
│   └── __init__.py
│
├── dashboard/                           [UPDATED]
│   ├── app.py                           [UPDATE: Output schema redesign]
│   ├── data_loader.py                   [KEEP AS IS]
│   └── __init__.py
│
├── frameworks/                          [NO CHANGES]
│   ├── macro_framework.py
│   ├── geopolitical_framework.py
│   ├── integrated_framework.py
│   └── __init__.py
│
├── utilities/                           [UPDATED]
│   ├── validators.py                    [UPDATE: New validation rules]
│   ├── shared_utils.py                  [UPDATE: New helpers, remove unused]
│   ├── walk_forward_validator.py        [KEEP AS IS]
│   └── __init__.py
│
├── system/                              [UPDATED]
│   ├── orchestrator.py                  [UPDATE: Component wiring]
│   ├── config.py                        [KEEP AS IS]
│   ├── schemas.py                       [UPDATE: Output schema]
│   ├── requirements.txt                 [KEEP - Add new deps if needed]
│   └── __init__.py
│
├── ARCHITECTURE_AND_ACTION_PLAN.md      [THIS FILE]
└── IMPLEMENTATION_CHECKLIST.md          [WILL CREATE]
```

---

## 📊 SECTION 3: DETAILED COMPONENT BREAKDOWN

### 3.1 INDICATORS LAYER (indicators/)

#### **indicators/kalman_hull.py** [NEW FILE - 200-250 lines]
**Purpose:** Replace all momentum indicators with single unified indicator

**Required Functions:**
```python
def apply_kalman_filter(prices, measurement_noise=0.1)
    → Returns: kalman_prices (smoothed price series)

def calculate_hull_ma(prices, period=9)
    → Returns: hull_ma_values (lag-reduced moving average)

def calculate_supertrend_bands(prices, atr_multiplier=1.0, atr_period=10)
    → Returns: upper_band, lower_band, basic_ub, basic_lb

def calculate_efficiency_ratio(prices, period=10)
    → Returns: efficiency_ratio (0-1, adaptability metric)

def detect_divergence(prices, supertrend, period=10)
    → Returns: divergence_signal (bullish/bearish/none)

def apply_consistency_check(trend_values, min_consecutive=3)
    → Returns: filtered_trend (stable signals only)

def calculate_adaptive_kalman_hull(prices, volume=None)
    → MAIN FUNCTION
    → Returns: {
        'trend': [-1, 0, 1],           # Direction
        'kalman_price': float,          # Smoothed price
        'upper_band': float,            # Supertrend upper
        'lower_band': float,            # Supertrend lower
        'efficiency_ratio': float,      # Adaptability (0-1)
        'divergence': 'bullish'/'bearish'/'none',
        'trend_consistency': bool,      # Trend is stable
        'signal_strength': float        # Confidence (0-1)
    }
```

**Input Requirements:**
- 2+ years historical daily prices
- Optional: volume data

**Output Validation:**
- trend ∈ {-1, 0, 1}
- upper_band, lower_band numeric
- divergence ∈ {'bullish', 'bearish', 'none'}
- efficiency_ratio ∈ [0, 1]
- signal_strength ∈ [0, 1]

---

### 3.2 ANALYZERS LAYER (analyzers/)

#### **analyzers/risk_component.py** [RENAMED from statistical_analyzer.py - UPDATE]

**Purpose:** Calculate risk metrics with new weighting system

**Metrics (NEW WEIGHTS):**
1. **CVaR (30%)** - Conditional Value at Risk
   - T-distribution fitting (same as before)
   - Parametric calculation
   
2. **Ulcer Index (30%)** - Drawdown risk metric
   - Risk-adaptive lookback periods
   - Same calculation methodology
   
3. **Beta (20%)** - Systematic risk
   - Correlation with best-matched benchmark
   - Same calculation as before
   
4. **Information Ratio (20%)** - Risk-adjusted performance
   - Tracking error vs benchmark
   - Same calculation as before

**Function Signature:**
```python
def calculate_risk_scores(returns_data, benchmark_data, quality_tier)
    → Returns: {
        'cvar': float,
        'ulcer_index': float,
        'beta': float,
        'information_ratio': float,
        'risk_score': float,  # Weighted combination
        'risk_category': 'LOW'/'MEDIUM'/'HIGH',
        'quality_flag': '✅'/'~'/'⚠️'/'🔴'
    }
```

**Weights Application:**
```
risk_score = (0.30 * cvar_scaled + 
              0.30 * ulcer_scaled + 
              0.20 * beta_scaled + 
              0.20 * ir_scaled)
```

---

#### **analyzers/volume_intelligence.py** [NEW FILE - 150-200 lines]

**Purpose:** Replace old volume indicators with structured intelligence

**Section 1.3 Implementation (from spec):**

1. **Spike Score** (Section 1.3.1)
   - Calculate relative volume ratio: current_vol / (20-day MA vol)
   - Apply Z-score normalization: (z-score - mean) / std
   - Scale to 0-100: `spike_score = max(0, min(100, (z_score + 3) * 10))`
   
2. **Price-Volume Correlation** (Section 1.3.2)
   - Calculate correlation between price changes and volume changes
   - Correlation score ∈ [-1, 1]
   - Interpretation: positive = volume confirms price moves
   
3. **Accumulation-Distribution Divergence** (Section 1.3.3)
   - Money Flow Multiplier: (close - low) - (high - close) / (high - low)
   - Cumulative A/D: sum of (MFM × volume)
   - Divergence detection: Compare A/D trend vs price trend
   - Signal: 'accumulation' / 'distribution' / 'neutral'

**Function Signatures:**
```python
def calculate_spike_score(volumes, period=20)
    → Returns: spike_score ∈ [0, 100]

def calculate_price_volume_correlation(prices, volumes, period=20)
    → Returns: correlation ∈ [-1, 1]

def detect_accumulation_distribution(prices, volumes)
    → Returns: signal ∈ {'accumulation', 'distribution', 'neutral'}

def analyze_volume_intelligence(prices, volumes)
    → MAIN FUNCTION
    → Returns: {
        'spike_score': float,           # 0-100
        'price_volume_correlation': float,  # -1 to 1
        'accumulation_distribution': str,   # accumulation/distribution/neutral
        'volume_confidence': float      # 0-1
    }
```

**Output Validation:**
- spike_score ∈ [0, 100]
- price_volume_correlation ∈ [-1, 1]
- accumulation_distribution ∈ {'accumulation', 'distribution', 'neutral'}

---

#### **analyzers/ml_ensemble.py** [RENAMED from forecasting_engine.py - UPDATE]

**Purpose:** ML-based forecasting with confidence scores (NO BIAS CORRECTION)

**Key Changes from Original:**
- Remove bias correction entirely
- Use raw ML output
- Add confidence score for uncertainty communication
- Output includes: Features → ML Ensemble → Confidence

**Function Signature:**
```python
def generate_ml_forecast(features_dict, model_ensemble)
    → Returns: {
        'forecast_return': float,       # Raw ML prediction
        'confidence_score': float,      # 0-1, inverse of uncertainty
        'features_used': dict,          # Input features
        'model_ensemble_output': float, # Raw ensemble output
        'feature_importance': dict      # Which features drove decision
    }
```

**CRITICAL:** No bias correction. Confidence score replaces bias adjustment:
- Users see both forecast AND confidence level
- Low-confidence predictions can be discounted by users themselves
- This aligns with specification Section 2.1

---

#### **analyzers/scoring_system.py** [UPDATE]

**Changes:**
- Update component weights to reflect new Risk Component weights
- Integrate new Kalman Hull signals
- Integrate Volume Intelligence signals
- Output: Keep same structure, add new indicator outputs

**Output Schema Update:**
```python
def calculate_comprehensive_score(risk_metrics, kalman_hull, volume_intel, ml_forecast)
    → Returns: {
        'risk_component': {
            'cvar': float,
            'ulcer_index': float,
            'beta': float,
            'information_ratio': float,
            'score': float (weighted)
        },
        'momentum_component': {
            'kalman_hull': {
                'trend': int,  # -1, 0, 1
                'bands': [upper, lower],
                'efficiency_ratio': float,
                'divergence': str,
                'signal_strength': float
            }
        },
        'volume_component': {
            'spike_score': float,
            'price_volume_correlation': float,
            'accumulation_distribution': str,
            'volume_confidence': float
        },
        'ml_component': {
            'forecast': float,
            'confidence': float,
            'features_used': dict
        },
        'quality_metrics': {
            'quality_flag': str,
            'data_tier': int
        }
    }
```

---

### 3.3 UTILITIES LAYER (utilities/)

#### **utilities/validators.py** [UPDATE]

**New Validation Rules (Section 4.6 from spec):**

1. **Kalman Hull Validator**
```python
def validate_kalman_hull_output(output_dict)
    Checks:
    - trend ∈ {-1, 0, 1}
    - upper_band is numeric
    - lower_band is numeric
    - divergence ∈ {'bullish', 'bearish', 'none'}
    - efficiency_ratio ∈ [0, 1]
    - signal_strength ∈ [0, 1]
```

2. **Volume Intelligence Validator**
```python
def validate_volume_intelligence(output_dict)
    Checks:
    - spike_score ∈ [0, 100]
    - price_volume_correlation ∈ [-1, 1]
    - accumulation_distribution ∈ {'accumulation', 'distribution', 'neutral'}
    - volume_confidence ∈ [0, 1]
```

3. **ML Ensemble Validator**
```python
def validate_ml_ensemble_output(output_dict)
    Checks:
    - forecast_return is numeric (can be negative)
    - confidence_score ∈ [0, 1]
    - features_used is dict
    - NO bias correction field exists
```

**Existing Validators (KEEP):**
- Risk classification validation
- Data quality tier validation
- Quality flag validation

---

#### **utilities/shared_utils.py** [UPDATE]

**Functions to ADD (from Section 1.3 spec):**

1. **Spike Score Helper**
```python
def calculate_relative_volume_ratio(volumes, period=20)
    → Returns: ratio_array
    Used by: volume_intelligence.py
```

2. **Correlation Helper**
```python
def calculate_rolling_correlation(series1, series2, period=20)
    → Returns: correlation_array
    Used by: volume_intelligence.py
```

3. **Divergence Detection Helper**
```python
def detect_trend_divergence(price_trend, indicator_trend)
    → Returns: 'bullish' / 'bearish' / 'none'
    Used by: kalman_hull.py and volume_intelligence.py
```

4. **Z-score Normalizer**
```python
def z_score_normalize(series, period=20)
    → Returns: normalized_array
    Used by: volume_intelligence.py (spike score)
```

**Functions to REMOVE:**
- Any KAMA calculation helpers
- Any RSI helpers
- Any Stochastic helpers
- Any VWAP helpers
- Holdings-related utilities
- Bias correction helpers
- Any VaR-only functions (keep CVaR)

---

### 3.4 SYSTEM LAYER (system/)

#### **system/orchestrator.py** [UPDATE]

**Component Wiring Changes:**
- Replace `technical_analyzer` import with `indicators/kalman_hull`
- Add `analyzers/volume_intelligence`
- Rename `forecasting_engine` to `ml_ensemble`
- Rename `statistical_analyzer` to `risk_component`
- Update pipeline to: Features → Kalman Hull → Volume Intelligence → ML Ensemble → Scoring
- Remove any holdings data collection

**Pipeline Flow:**
```
1. Data Loading (ETF price/volume)
2. Feature Extraction (macro, geopolitical, fundamental)
3. Risk Calculation (CVaR, Ulcer, Beta, IR)
4. Momentum Analysis (Kalman Hull Supertrend)
5. Volume Analysis (Volume Intelligence)
6. ML Ensemble Forecast (Raw output + Confidence)
7. Integrated Scoring (Combine all)
8. Risk Classification (Same as before)
9. Validation (Walk-forward)
10. Dashboard Output
```

---

#### **system/schemas.py** [UPDATE]

**ETF_ANALYSIS_OUTPUT Schema - COMPLETE REDESIGN**

**Section to Redesign: "Technical Indicators"**

```python
{
    'etf_code': str,
    'analysis_date': str,
    
    # RISK COMPONENT (renamed from Statistical)
    'risk_metrics': {
        'cvar_99': float,
        'ulcer_index': float,
        'beta': float,
        'information_ratio': float,
        'risk_score': float,        # 0-1, weighted
        'risk_category': str,       # LOW/MEDIUM/HIGH
    },
    
    # TECHNICAL INDICATORS (REDESIGNED)
    'technical_indicators': {
        'kalman_hull_supertrend': {
            'trend': int,           # -1, 0, 1
            'kalman_price': float,
            'upper_band': float,
            'lower_band': float,
            'efficiency_ratio': float,  # 0-1
            'divergence': str,      # bullish/bearish/none
            'trend_consistency': bool,
            'signal_strength': float
        },
        'volume_intelligence': {
            'spike_score': float,   # 0-100
            'price_volume_correlation': float,  # -1 to 1
            'accumulation_distribution': str,  # acc/dist/neutral
            'volume_confidence': float
        }
    },
    
    # ML ENSEMBLE (renamed from Forecasting)
    'ml_ensemble': {
        'features_used': dict,
        'forecast_return': float,   # Raw output
        'confidence_score': float,  # 0-1
        'feature_importance': dict
    },
    
    # QUALITY & VALIDATION
    'quality_metrics': {
        'quality_flag': str,        # ✅/~/⚠️/🔴
        'data_tier': int,           # 1-4
    },
    
    'validation': {
        'walk_forward_mae': float,
        'validation_passed': bool,
        'confidence_level': str     # HIGH/MEDIUM/LOW
    },
    
    # EXISTING (KEEP)
    'macro_framework': {...},
    'geopolitical_framework': {...},
    'integration_score': float
}
```

**Remove Entirely from Schema:**
- VaR (Parametric, Historical) - Keep ONLY CVaR
- KAMA, RSI, Stochastic, VWAP details
- Holdings momentum data
- Any bias correction fields

---

### 3.5 DASHBOARD LAYER (dashboard/)

#### **dashboard/app.py** [UPDATE]

**Changes:**
- Update data loading to use new schema
- Visualize: Risk Component metrics (new weights)
- Visualize: Kalman Hull Supertrend with bands and signals
- Visualize: Volume Intelligence spikes and A/D divergence
- Visualize: ML Ensemble forecast with confidence bar
- Keep validation comparison display
- Update color schemes to match new components

**Display Structure:**
```
1. Risk Overview (CVaR, Ulcer, Beta, IR with 30/30/20/20 weights)
2. Momentum & Volume (Kalman Hull + Volume Intelligence)
3. ML Forecast (Forecast with Confidence Level)
4. Quality & Validation (Flags + Walk-Forward Performance)
5. Macro & Geopolitical Context (Keep as is)
```

---

## 📋 SECTION 4: IMPLEMENTATION PHASE SEQUENCE

### Phase 1: Foundation (Weeks 1-2)
1. Create `indicators/` directory structure
2. Implement `indicators/kalman_hull.py` (Ultrathink this - complex math)
3. Implement `analyzers/volume_intelligence.py`
4. Update `utilities/validators.py` with new rules
5. Update `utilities/shared_utils.py` (add new, remove old)

### Phase 2: Core Components (Weeks 3-4)
1. Rename & Update `analyzers/statistical_analyzer.py` → `analyzers/risk_component.py`
2. Rename & Update `analyzers/forecasting_engine.py` → `analyzers/ml_ensemble.py` (remove bias correction)
3. Update `analyzers/scoring_system.py` with new weights
4. Verify `analyzers/etf_risk_classifier.py` works with new schemas (no changes needed)

### Phase 3: System Integration (Week 5)
1. Update `system/orchestrator.py` - wire new components
2. Update `system/schemas.py` - new ETF_ANALYSIS_OUTPUT
3. Update `dashboard/app.py` - visualize new outputs
4. Update `config.py` if needed for new parameters

### Phase 4: Validation & Testing (Week 6)
1. Test each component independently
2. Test full pipeline integration
3. Compare outputs against specification
4. Validate walk-forward performance
5. Generate validation_results.json

---

## 🔍 SECTION 5: DETAILED CODE ORGANIZATION RULES

### Import Organization Pattern
```python
# Each file should follow this pattern:
# 1. Standard library imports
# 2. Third-party imports (numpy, pandas, scipy)
# 3. Local imports
# 4. Type hints at module level
```

### Function Naming Convention
- Calculation functions: `calculate_*` (e.g., `calculate_spike_score`)
- Detection functions: `detect_*` (e.g., `detect_divergence`)
- Validation functions: `validate_*` (e.g., `validate_kalman_hull_output`)
- Main pipeline functions: `*_pipeline` or just function name (e.g., `apply_kalman_filter`)

### Variable Naming Convention
- Raw metrics: `metric_name` (e.g., `cvar`, `ulcer_index`)
- Scaled/normalized: `metric_name_scaled` or `metric_name_ratio`
- Arrays/Series: `plural_form` (e.g., `prices`, `volumes`, `returns`)
- Dictionaries with outputs: `output_*` (e.g., `output_dict`)

### Code Length Guidelines
- Kalman Hull: 200-250 lines max (complex, but efficient)
- Volume Intelligence: 150-200 lines max
- Risk Component: 200 lines max (mostly metrics reuse)
- ML Ensemble: 150 lines max (simplified without bias correction)
- Validators: 100 lines max
- Each function: 20-40 lines average (break up if longer)

---

## 📊 SECTION 6: DATA FLOW DIAGRAMS

### Input → Processing → Output

```
ETF Data (Daily Prices, Volume)
    ↓
[Data Validation & Loading]
    ↓
Risk Metrics ← [Risk Component: CVaR, Ulcer, Beta, IR] ← Historical Returns
Momentum Signal ← [Kalman Hull Supertrend] ← Price Data
Volume Signal ← [Volume Intelligence] ← Price + Volume
ML Forecast ← [ML Ensemble] ← Features (Risk + Momentum + Volume)
    ↓
[Comprehensive Score]
    ↓
Risk Classification (LOW/MEDIUM/HIGH)
    ↓
[Walk-Forward Validation]
    ↓
Dashboard Output
```

---

## ✅ SECTION 7: CRITICAL SUCCESS CRITERIA

### Must-Haves
- ✅ No bias correction in ML output (use raw + confidence)
- ✅ Kalman Hull replaces ALL old momentum indicators
- ✅ Risk Component has 30/30/20/20 weights
- ✅ All new indicators validate within specified ranges
- ✅ Walk-forward validation continues seamlessly
- ✅ Quality flags remain visual (✅/~/⚠️/🔴)
- ✅ No new methodologies invented
- ✅ Code is efficient and minimal

### Validation Checkpoints
1. **Per-Component:** Each analyzer outputs valid data per spec
2. **Schema:** All outputs match updated schemas
3. **Integration:** Pipeline runs without errors
4. **Performance:** Validation metrics align with old system baseline
5. **Dashboard:** All components display correctly

---

## 🚫 SECTION 8: COMMON PITFALLS TO AVOID

1. **Over-complicating Kalman Hull** - Use standard formulas, don't invent variants
2. **Adding methodologies not in spec** - Stick exactly to Section 1.1-1.3
3. **Including VaR when only CVaR needed** - Remove entirely
4. **Retaining bias correction** - Delete ALL bias correction logic
5. **Creating unnecessary helper functions** - Only add what's used
6. **Breaking existing validated components** - Risk classifier, macro/geo frameworks unchanged
7. **Forgetting validation updates** - All new outputs must be validated
8. **Ignoring efficiency** - Each function should be as short as possible while complete

---

## 📝 SECTION 9: FILE CREATION CHECKLIST

**Create These New Files:**
- [ ] /modified/indicators/__init__.py
- [ ] /modified/indicators/kalman_hull.py (200-250 lines)
- [ ] /modified/analyzers/volume_intelligence.py (150-200 lines)
- [ ] /modified/analyzers/ml_ensemble.py (renamed with updates)
- [ ] /modified/analyzers/risk_component.py (renamed with updates)
- [ ] /modified/IMPLEMENTATION_CHECKLIST.md (detailed task tracker)

**Update These Files:**
- [ ] /modified/analyzers/scoring_system.py
- [ ] /modified/utilities/validators.py (add new rules)
- [ ] /modified/utilities/shared_utils.py (add new helpers, remove old)
- [ ] /modified/system/orchestrator.py
- [ ] /modified/system/schemas.py
- [ ] /modified/dashboard/app.py

**Keep Unchanged:**
- /modified/data_manager/ (all files)
- /modified/frameworks/ (all files)
- /modified/utilities/walk_forward_validator.py
- /modified/system/config.py
- /modified/system/requirements.txt
- /modified/data_manager/ (all files)

---

## 🎓 SECTION 10: REFERENCE LOOKUP TABLE

**For Specific Kalman Hull Details** → Check Section 1.1 of modification spec
**For Volume Intelligence Details** → Check Section 1.3 of modification spec
**For ML Ensemble Changes** → Check Section 2.1 of modification spec
**For Risk Component Weights** → Check Section 1.1 breakdown of metrics
**For Validation Rules** → Check Section 4.6 of modification spec
**For Output Schema** → Check Sections 3.x of modification spec

---

**END OF ARCHITECTURE & ACTION PLAN**
**READY FOR IMPLEMENTATION PHASE**
