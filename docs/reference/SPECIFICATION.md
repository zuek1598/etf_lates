# new

# ETF Analysis System - Technical Specification

## SYSTEM OVERVIEW

**Purpose**: Multi-dimensional ETF analysis combining risk assessment, momentum analysis, volume intelligence, and ML forecasting.

**Architecture**: 15 systems (10 core indicators + 1 ML model + 2 context frameworks + 2 support metrics)

**Output**: Ranked ETF list with composite scores (0-100) and detailed analytics per ETF.

---

## 1. INDICATOR SPECIFICATIONS

### 1.1 RISK LAYER (4 Indicators)

### 1.1.1 Parametric CVaR (Conditional Value at Risk)

**Purpose**: Measure expected loss in worst 5% of scenarios (tail risk).

**Method**:

- Fit t-distribution to daily returns using [`scipy.stats.t.fit](http://scipy.stats.t.fit)()`
- Extract parameters: degrees_of_freedom (ν), location (μ), scale (σ)
- Calculate 95% CVaR using correct analytical formula:

```jsx
# Step 1: Calculate VaR at 5% level
t_quantile_005 = t.ppf(0.05, df=ν, loc=μ, scale=σ)

# Step 2: Calculate CVaR using proper t-distribution formula
CVaR_95 = μ + σ × sqrt((ν-2)/ν) × t.pdf(t_quantile_005, df=ν) / (1-0.05) × (ν + t_quantile_005²)/(ν-1)

where:
- t_quantile_005 = standardized t-distribution quantile at 5%
- t.pdf() = probability density function of t-distribution
- ν = degrees of freedom from fitted t-distribution
```

**Edge Cases**:

- If ν ≤ 2.0, set ν = 2.1 (avoid variance calculation issues)
- If sample size < 100, apply small-sample bias correction: ν × (n-1) / max(n-3, 1)
- Minimum 30 days required; otherwise return NaN

**Output**: Percentage (e.g., -8.5% means 5% worst scenarios average 8.5% loss)

**Scoring**:

- CVaR > -2%: 90-100 points
- CVaR -2% to -5%: 70-90 points
- CVaR -5% to -10%: 40-70 points
- CVaR < -10%: 0-40 points

### 1.1.2 Ulcer Index

**Purpose**: Measure drawdown severity and duration (downside pain metric).

**Method**:

- Calculate running maximum: `running_max = prices.expanding().max()`
- Calculate drawdown percentage: `drawdown_pct = (prices - running_max) / running_max × 100`
- Ulcer Index: `sqrt(mean(drawdown_pct²))`
- Lookback period varies by ETF group:
    - Australian Equity: 60 days
    - International Equity: 60 days
    - Bonds: 90 days
    - Commodities: 30 days

**Output**: Percentage (lower = better; e.g., 3.5%)

**Scoring**:

- Ulcer < 2%: 100 points
- Ulcer 2-5%: 80-100 points
- Ulcer 5-10%: 60-80 points
- Ulcer 10-20%: 40-60 points
- Ulcer > 20%: 0-40 points

### 1.1.3 Beta (Market Sensitivity)

**Purpose**: Measure systematic risk relative to best-correlated benchmark.

**Method**:

1. Identify best-correlated benchmark from pool:
    - ASX200 (^AXJO)
    - MSCI World (URTH)
    - S&P500 (^GSPC)
    - MSCI EM (EEM)
    - Global Bonds (BND)
    - NASDAQ (^IXIC)
    - VIX (^VIX)
    - DXY (DX-Y.NYB)
    - Gold (GLD)
2. Calculate correlation for each benchmark over maximum available period
3. Select benchmark with highest correlation (minimum 30 overlapping days required)
4. **Benchmark Correlation Threshold**:
    - **Minimum correlation threshold**: 0.30
    - If max_correlation < 0.30: use ASX200 as default fallback with LOW_CORRELATION flag
    - This ensures benchmark relevance and prevents spurious beta calculations
5. Calculate 1-year rolling beta:

```
beta = covariance(etf_returns, benchmark_returns) / variance(benchmark_returns)
```

1. Use up to 252 days (1 year), minimum 30 days

**Fallback**: If insufficient data, beta = 1.0 with low confidence flag

**Output**: Ratio (e.g., 0.95, 1.2)

**Scoring**: Risk category dependent (see Section 4.5)

### 1.1.4 Information Ratio

**Purpose**: Measure benchmark-relative skill (active return per unit tracking error).

**Method**:

1. Align ETF and benchmark returns by date
2. Calculate excess returns: `excess = etf_returns - benchmark_returns`
3. Use rolling window:
    - Minimum 126 days (6 months)
    - Use actual available data if > 126 days
4. Calculate:

```
mean_excess = mean(excess returns over window)
tracking_error = std(excess returns over window)
IR = (mean_excess / tracking_error) × sqrt(252)
```

**Output**: Ratio (e.g., 0.5, -0.2)

**Scoring**:

- IR ≥ 1.0: 100 points
- IR 0.5-1.0: 70-100 points
- IR 0.0-0.5: 40-70 points
- IR < 0.0: 0-40 points

### 1.2 MOMENTUM LAYER (1 Unified Indicator)

### 1.2.1 Adaptive Kalman Hull Supertrend

**Purpose**: Unified trend indicator combining optimal estimation (Kalman), lag reduction (Hull), and volatility awareness (Supertrend).

**Components**:

**A. Efficiency Ratio (Adaptability Mechanism)**

```
ER = |price[t] - price[t-10]| / sum(|price[i] - price[i-1]|, i=t-10 to t)
```

- High ER (trending): filter follows price closely
- Low ER (choppy): filter ignores noise

**B. Volatility Regime**

```
vol_regime = ATR(14) / current_price
```

**C. Adaptive Parameters**

```
base_measurement_noise = risk_category_base
base_process_noise = risk_category_base
adaptive_measurement_noise = base_measurement × (1 - ER) × (1 + vol_regime)
adaptive_process_noise = base_process × (1 + vol_regime)
```

**Risk Category Parameters**:

| Category | Base Measurement | Base Process | ATR Factor | ATR Period |
| --- | --- | --- | --- | --- |
| Low Risk | 4.0 | 0.005 | 1.5 | 14 |
| Medium Risk | 3.0 | 0.01 | 1.7 | 12 |
| High Risk | 2.0 | 0.02 | 2.0 | 10 |

**D. Kalman Filter**

State Initialization:

```
state_estimate = first_price
error_covariance = 100.0
```

For each price point:

```
# Prediction
predicted_state = state_estimate
predicted_error = error_covariance + adaptive_process_noise

# Update
kalman_gain = predicted_error / (predicted_error + adaptive_measurement_noise)
state_estimate = predicted_state + kalman_gain × (price - predicted_state)
error_covariance = (1 - kalman_gain) × predicted_error
kalman_filtered_price = state_estimate
```

**E. Hull MA Application**

```
n = measurement_noise (rounded to integer)
half_n = n / 2
sqrt_n = sqrt(n)

# Apply Kalman to different periods
kalman_half = kalman_filter(prices, period=half_n)
kalman_full = kalman_filter(prices, period=n)

# Hull formula
hull_raw = 2 × kalman_half - kalman_full
hull_final = kalman_filter(hull_raw, period=sqrt_n)
```

**F. Supertrend Bands**

```
atr = ATR(atr_period)
upper_band = hull_final + factor × atr
lower_band = hull_final - factor × atr

# Band adjustment (prevent whipsaws)
if close[t-1] < lower_band[t-1]:
    lower_band[t] = max(lower_band[t], lower_band[t-1])
if close[t-1] > upper_band[t-1]:
    upper_band[t] = min(upper_band[t], upper_band[t-1])

# Trend determination
if price > upper_band:
    trend = 1 (uptrend)
elif price < lower_band:
    trend = -1 (downtrend)
else:
    trend = 0 (neutral)
```

**G. Divergence Detection**

```
# Bullish divergence: price makes lower low, indicator makes higher low
price_ll = price[t] < min(price[t-20:t])
indicator_hl = hull_final[t] > min(hull_final[t-20:t])
bullish_divergence = price_ll AND indicator_hl

# Bearish divergence: price makes higher high, indicator makes lower high
price_hh = price[t] > max(price[t-20:t])
indicator_lh = hull_final[t] < max(hull_final[t-20:t])
bearish_divergence = price_hh AND indicator_lh
```

**Outputs**:

- `trend_value`: Smoothed price level (float)
- `trend_direction`: 1 (up), -1 (down), 0 (neutral)
- `upper_band`: Upper volatility band (float)
- `lower_band`: Lower volatility band (float)
- `divergence`: 'bullish', 'bearish', or 'none'

**Scoring**:

```
momentum_score = weighted_average([
    40% × direction_score:
        - Uptrend (1): 80 points
        - Downtrend (-1): 20 points
        - Neutral (0): 50 points
    
    30% × strength_score:
        - distance_to_band = min(|price - upper|, |price - lower|)
        - normalized = distance_to_band / (upper - lower)
        - strength_score = 50 + normalized × 50
    
    20% × divergence_score:
        - Bullish divergence: +20 points
        - Bearish divergence: -20 points
        - None: 0 points (added to base 50)
    
    10% × consistency_score:
        - Count trend changes in last 20 periods
        - consistency = 100 - (changes × 5)
])
```

### 1.3 VOLUME INTELLIGENCE LAYER (3 Indicators)

### 1.3.1 Volume Spike Index

**Purpose**: Detect unusual volume activity indicating institutional interest or news events.

**Method**:

**Relative Volume Ratio (RVR)**:

```
RVR = current_volume / mean(volume[t-20:t])
```

**Volume Z-Score**:

```
z_score = (current_volume - μ_20d) / σ_20d
```

**Combined Spike Score**:

```
rvr_component = min(100, (RVR - 1.0) × 50)
z_component = min(100, z_score × 25)
spike_score = 0.6 × rvr_component + 0.4 × z_component
```

**Thresholds**:

- RVR > 2.0: High interest
- RVR > 3.0: Exceptional activity
- Z-Score > 2σ: Institutional activity
- Z-Score > 3σ: Major event

**Output**: Score 0-100 (higher = more unusual activity)

**Scoring**: Direct use of spike_score

### 1.3.2 Price-Volume Correlation

**Purpose**: Assess whether volume confirms price direction (move conviction).

**Method**:

```
# 20-day rolling window
price_changes = abs(prices.pct_change())
volume_series = volumes
correlation = pearson_correlation(price_changes[t-20:t], volume_series[t-20:t])
```

**Interpretation**:

- r > +0.5: Strong confirmation (high conviction)
- r = 0.0 to +0.5: Moderate confirmation
- r = -0.3 to 0.0: Weak/no confirmation
- r < -0.3: Divergence warning (volume contradicts price)

**Output**: Correlation coefficient -1.0 to +1.0

**Scoring**:

```
if correlation > 0.5:
    score = 85 + (correlation - 0.5) × 30
elif correlation > 0.0:
    score = 50 + correlation × 70
elif correlation > -0.3:
    score = 35 + (correlation + 0.3) × 50
else:
    score = max(0, 35 + correlation × 100)
```

### 1.3.3 Accumulation/Distribution Line (A/D Line)

**Purpose**: Track smart money positioning (institutional buying on dips vs selling on rallies).

**Method**:

```
# Money Flow Multiplier
for each day:
    if high == low:
        mfm = 0
    else:
        mfm = [(close - low) - (high - close)] / (high - low)
    
    money_flow_volume = mfm × volume
    
    # Cumulative A/D Line
    ad_line[t] = ad_line[t-1] + money_flow_volume
```

**Divergence Detection**:

```
# 20-day comparison
price_change_20d = (price[t] - price[t-20]) / price[t-20]
ad_change_20d = (ad_line[t] - ad_line[t-20]) / abs(ad_line[t-20])

# Accumulation: price down/flat, A/D rising
if price_change_20d < 0.02 AND ad_change_20d > 0.05:
    signal = 'accumulation'
# Distribution: price up, A/D flat/falling
elif price_change_20d > 0.02 AND ad_change_20d < -0.02:
    signal = 'distribution'
else:
    signal = 'neutral'
```

**Output**: Signal ('accumulation', 'distribution', 'neutral')

**Scoring**:

- Accumulation: 70 points
- Neutral: 50 points
- Distribution: 30 points

### 1.4 LIQUIDITY LAYER (2 Indicators)

### 1.4.1 Amihud Illiquidity Ratio

**Purpose**: Estimate price impact per dollar traded (execution cost proxy).

**Method**:

```
for each day in last 20 days:
    daily_return = abs(close[t] - close[t-1]) / close[t-1]
    dollar_volume = close[t] × volume[t]
    
    if dollar_volume > 0:
        daily_illiquidity = daily_return / dollar_volume
    else:
        daily_illiquidity = NaN

amihud = mean(daily_illiquidity[valid])
```

**Output**: Ratio (lower = better liquidity; typical range: 0.001 - 10.0)

**Usage**: Penalty application only (not scored directly)

**Penalties**:

- Amihud > 1.0: -5 points

### 1.4.2 Average Daily Volume

**Purpose**: Liquidity baseline assessment.

**Method**:

```
avg_volume = mean(volume[t-20:t])
dollar_volume = avg_volume × current_price
```

**Output**: Dollar amount (e.g., $2,500,000)

**Usage**: Penalty application only

**Penalties**:

- < $500,000: -10 points
- $500,000 - $1,000,000: -5 points
- ≥ $1,000,000: No penalty

### 1.5 SUPPORT METRICS (4 Systems)

### 1.5.1 Basic Return Metrics

**Purpose**: Provide standard performance benchmarks for analysis.

**Method**:

```jsx
# YTD Return calculation
year_start_price = get_price_at_year_start(current_date)
ytd_return = (current_price - year_start_price) / year_start_price

# 1-Year Return calculation
price_252_days_ago = get_price_n_days_ago(252)  # 252 trading days ≈ 1 year
one_year_return = (current_price - price_252_days_ago) / price_252_days_ago

# Handle missing data
if year_start_price is None:
    ytd_return = NaN
if price_252_days_ago is None:
    one_year_return = NaN
```

**Output**:

- `ytd_return`: Percentage (e.g., 0.125 = 12.5% YTD gain)
- `one_year_return`: Percentage (e.g., -0.08 = 8% loss over past year)

**Usage**: Display only (not scored directly)

### 1.5.3 Zero Volume Days

**Purpose**: Detect trading interruptions.

**Method**:

```jsx
zero_days = count(volume[t-60:t] == 0)
```

**Output**: Integer count

**Penalty**: If > 5 days: -5 points

### 1.5.4 Data Quality Tiers

**Purpose**: Adjust confidence and scores based on data availability.

**Classification**:

```jsx
data_length = number of trading days available

if data_length >= 756:  # 3+ years
    tier = 'tier_1'
    score_penalty = 0
    confidence_adjustment = 0.0
elif data_length >= 504:  # 2-3 years
    tier = 'tier_2'
    score_penalty = -2
    confidence_adjustment = -0.05
elif data_length >= 252:  # 1-2 years
    tier = 'tier_3'
    score_penalty = -5
    confidence_adjustment = -0.10
else:  # < 1 year
    tier = 'tier_4'
    score_penalty = -10
    confidence_adjustment = -0.15
```

**Output**: Tier classification with penalties

---

## 2. FORECASTING ENGINE

### 2.1 ML Ensemble Model

**Architecture**: Random Forest (20 trees, max_depth=3) + Ridge Regression (α=1.0)

**Horizon**: 60 trading days (~3 months)

**Features** (10 inputs):

1. CVaR (tail risk measure)
2. Ulcer Index (drawdown measure)
3. Beta (systematic risk)
4. Information Ratio (alpha skill)
5. Kalman Hull trend value (smoothed price level)
6. Kalman Hull direction (1/-1/0)
7. Volume Spike score (0-100)
8. Price-Volume correlation (-1 to +1)
9. A/D Line 20-day momentum (percentage change)
10. Data Quality score (derived from tier)

**Training Process**:

1. **Historical Backtest**:

```
FOR each 60-day period in history (step by 2 days):
    - Extract features at time t
    - Record actual 60-day return from t to t+60
    - Store (features, actual_return) pair
```

1. **Train/Test Split**:
    - Training: First 70% of samples (minimum)
    - Testing: Last 30% of samples (minimum 10 samples)
2. **Model Training**:
    - Random Forest: fit(X_train, y_train)
    - Ridge: fit(X_train, y_train)
3. **Prediction**:
    - RF_pred = random_forest.predict(X_test)
    - Ridge_pred = ridge.predict(X_test)
    - Ensemble_pred = 0.5 × RF_pred + 0.5 × Ridge_pred
4. **Validation**:
    - MAE = mean(|Ensemble_pred - y_test|)
    - Hit_Rate = mean((Ensemble_pred > 0) == (y_test > 0))
    - Bias = mean(Ensemble_pred - y_test)

**Confidence Calculation**:

```
# Base confidence from MAE (non-linear mapping)
scale_factor = sqrt(forecast_horizon / 60.0)
excellent_threshold = 3.0 × scale_factor
good_threshold = 6.0 × scale_factor
fair_threshold = 12.0 × scale_factor
poor_threshold = 25.0 × scale_factor

if MAE <= excellent_threshold:
    base_confidence = 0.85 + 0.10 × (1 - MAE / excellent_threshold)
elif MAE <= good_threshold:
    ratio = (MAE - excellent_threshold) / (good_threshold - excellent_threshold)
    base_confidence = 0.85 - 0.20 × ratio
elif MAE <= fair_threshold:
    ratio = (MAE - good_threshold) / (fair_threshold - good_threshold)
    base_confidence = 0.65 - 0.30 × ratio
elif MAE <= poor_threshold:
    ratio = (MAE - fair_threshold) / (poor_threshold - fair_threshold)
    base_confidence = 0.35 - 0.20 × ratio
else:
    base_confidence = 0.10

# Sample size adjustment
min_reliable_samples = 60
optimal_samples = 252

if sample_size >= optimal_samples:
    final_confidence = base_confidence
elif sample_size >= min_reliable_samples:
    ratio = (sample_size - min_reliable_samples) / (optimal_samples - min_reliable_samples)
    penalty = 0.1 × (1 - ratio)
    final_confidence = max(0.1, base_confidence - penalty)
elif sample_size >= 30:
    ratio = sample_size / min_reliable_samples
    penalty = 0.3 × (1 - ratio)²
    final_confidence = max(0.1, base_confidence - penalty)
else:
    ratio = sample_size / 30.0
    penalty = 0.3 + 0.5 × (1 - ratio)
    final_confidence = max(0.1, base_confidence - penalty)
```

**Final Forecast Output**:

```python
{
    'forecast': raw_ml_prediction,     # Percentage return (e.g., 0.042 = 4.2%)
    'confidence': final_confidence,    # 0.0 - 1.0
    'mae': mae_score,                  # Mean Absolute Error (%)
    'hit_rate': directional_accuracy,  # Percentage (0.0 - 1.0)
    'bias': systematic_bias,           # Mean error (%)
    'sample_size': validation_samples, # Integer
    'method': 'ML'                     # Fixed value
}
```

**Critical Notes**:

- **NO bias correction applied** (use raw ML output)
- **NO fallback mechanism** (always use ML forecast)
- Confidence reflects reliability; low confidence = speculative forecast

---

## 3. CONTEXT FRAMEWORKS (Metadata Only)

### 3.1 Macro Economic Framework

**Purpose**: Provide economic regime context (NOT auto-applied to forecasts).

**Calculation**: See `frameworks/macro_[framework.py](http://framework.py)` (unchanged)

**Output**:

```python
{
    'multiplier': 0.75 - 1.25,  # Economic regime adjustment
    'composite_score': 0-100,    # Overall macro health
    'regime': 'GOLDILOCKS' | 'TRANSITIONAL' | 'CRISIS',
    'factors': {
        'systematic_risk': 0-100,
        'growth_momentum': 0-100,
        'monetary_policy': 0-100,
        'regime_classification': 0-100
    }
}
```

**Storage**: Attached to each ETF result as metadata under `context.macro`

**User Application** (Optional):

```
adjusted_forecast = forecast × macro_multiplier
```

### 3.2 Geopolitical Risk Framework

**Purpose**: Provide geopolitical tail risk context (NOT auto-applied to confidence).

**Calculation**: See `frameworks/geopolitical_[framework.py](http://framework.py)` (unchanged)

**Output**:

```python
{
    'risk_score': 0-100,  # Composite geopolitical risk
    'risk_level': 'LOW' | 'MODERATE' | 'HIGH' | 'SEVERE' | 'EXTREME',
    'pillars': {
        'us_china_taiwan': 0-100,
        'military_conflict': 0-100,
        'trade_war': 0-100,
        'financial_stress': 0-100,
        'energy_security': 0-100
    }
}
```

**Storage**: Attached to each ETF result as metadata under `context.geopolitical`

**User Application** (Optional):

```
# For risk assets
adjusted_confidence = confidence × (1 - geo_score/100 × 0.5)

# For safe havens
adjusted_confidence = confidence × (1 + geo_score/100 × 0.3)
```

---

## 4. SYSTEM LOGIC & PROCESS

### 4.1 Phase 1: Data Collection & Classification

**Input**: List of ETF tickers (e.g., ['[VAS.AX](http://VAS.AX)', '[NDQ.AX](http://NDQ.AX)', '[VAF.AX](http://VAF.AX)', ...])

**Process**:

**STEP 1.1: Download Market Data**

```
├─ VIX: ^VIX (max history)
├─ Benchmarks: [^AXJO, URTH, ^GSPC, EEM, BND, ^IXIC, DX-Y.NYB, GLD]
├─ Macro inputs: [Treasury yields, credit spreads, sector ETFs]
└─ Geopolitical inputs: [Defense stocks, Taiwan/China ETFs, commodities]
```

**STEP 1.2: Download ETF Data**

```python
FOR EACH ticker:
    download_data = [yf.download](http://yf.download)(ticker, period='max')
    
    IF data.empty OR len(data) < 30:
        mark as failed
        continue
    
    # Quality assessment
    years_available = len(data) / 252
    completeness = 1 - (data['Close'].isna().sum() / len(data))
    
    IF years_available >= 3:
        quality_tier = 'tier_1'
    ELIF years_available >= 2:
        quality_tier = 'tier_2'
    ELIF years_available >= 1:
        quality_tier = 'tier_3'
    ELSE:
        quality_tier = 'tier_4'
    
    quality_score = completeness × min(years_available / 3, 1.0)
    
    store: {
        'ticker': ticker,
        'data': data,
        'quality_tier': quality_tier,
        'quality_score': quality_score,
        'data_length': len(data)
    }
```

**STEP 1.3: Risk Classification**

```python
FOR EACH ETF with valid data:
    # Calculate annualized volatility (252-day)
    returns = data['Close'].pct_change().dropna()
    volatility = returns.std() × sqrt(252)
    
    # Calculate beta vs best benchmark
    best_benchmark, correlation = identify_highest_correlation(returns, benchmarks)
    beta = calculate_beta(returns, benchmark_returns, period=252)
    
    # Classify risk
    IF volatility < 0.12:
        risk_category = 'low_risk'
    ELIF volatility > 0.22 AND beta > 1.2:
        risk_category = 'high_risk'
    ELSE:
        risk_category = 'medium_risk'
    
    # Apply matrix adjustment
    IF volatility < 0.12 AND beta > 1.2:
        risk_category = 'medium_risk'  # Upgrade due to market sensitivity
    ELIF volatility > 0.22 AND beta < 0.8:
        risk_category = 'medium_risk'  # Downgrade due to market independence
    
    store: {
        'risk_category': risk_category,
        'volatility': volatility,
        'beta': beta,
        'best_benchmark': best_benchmark
    }
```

**OUTPUT**:

```python
{
    'low_risk_etfs': {...},
    'medium_risk_etfs': {...},
    'high_risk_etfs': {...}
}
```

### 4.2 Phase 2: Multi-Layer Analysis

**Process per ETF**:

```python
FOR EACH risk_category IN ['low_risk', 'medium_risk', 'high_risk']:
    FOR EACH etf IN risk_category:
        
        # === LAYER 1: RISK ANALYSIS ===
        
        # 1.1 CVaR
        returns = calculate_returns([etf.data](http://etf.data))
        t_params = fit_t_distribution(returns)
        cvar = calculate_parametric_cvar(t_params, confidence=0.95)
        
        # 1.2 Ulcer Index
        etf_group = classify_etf_group([etf.info](http://etf.info))  # Australian_Equity, etc.
        ulcer_lookback = get_lookback_period(etf_group)
        ulcer = calculate_ulcer_index(returns, ulcer_lookback)
        
        # 1.3 Beta
        # (already calculated in Phase 1)
        beta = etf.beta
        
        # 1.4 Information Ratio
        benchmark_data = get_benchmark_data([etf.best](http://etf.best)_benchmark)
        benchmark_returns = calculate_returns(benchmark_data)
        info_ratio = calculate_information_ratio(returns, benchmark_returns)
        
        # === LAYER 2: MOMENTUM ANALYSIS ===
        
        kalman_params = get_adaptive_params(risk_category)
        
        kalman_hull = AdaptiveKalmanHullSupertrend(
            base_measurement=kalman_params['measurement'],
            base_process=kalman_params['process'],
            atr_factor=kalman_params['atr_factor'],
            atr_period=kalman_params['atr_period']
        )
        
        momentum_result = kalman_hull.analyze([etf.data](http://etf.data)['Close'])
        # Returns: {trend_value, trend_direction, upper_band, lower_band, divergence}
        
        # === LAYER 3: VOLUME INTELLIGENCE ===
        
        # 3.1 Volume Spike
        current_volume = [etf.data](http://etf.data)['Volume'].iloc[-1]
        avg_volume_20d = [etf.data](http://etf.data)['Volume'].rolling(20).mean().iloc[-1]
        rvr = current_volume / avg_volume_20d
        z_score = (current_volume - avg_volume_20d) / [etf.data](http://etf.data)['Volume'].rolling(20).std().iloc[-1]
        spike_score = calculate_spike_score(rvr, z_score)
        
        # 3.2 Price-Volume Correlation
        price_changes = abs([etf.data](http://etf.data)['Close'].pct_change())
        pv_correlation = price_changes.rolling(20).corr([etf.data](http://etf.data)['Volume']).iloc[-1]
        
        # 3.3 A/D Line
        ad_line = calculate_ad_line([etf.data](http://etf.data))
        ad_signal = detect_ad_divergence([etf.data](http://etf.data)['Close'], ad_line)
        
        # === LAYER 4: LIQUIDITY ===
        
        amihud = calculate_amihud_ratio([etf.data](http://etf.data))
        avg_volume = [etf.data](http://etf.data)['Volume'].rolling(20).mean().iloc[-1] × [etf.data](http://etf.data)['Close'].iloc[-1]
        zero_volume_days = ([etf.data](http://etf.data)['Volume'].tail(60) == 0).sum()
        
        # === STORE ANALYSIS ===
        
        analysis_results[ticker] = {
            # Risk metrics
            'cvar': cvar,
            'ulcer_index': ulcer,
            'beta': beta,
            'information_ratio': info_ratio,
            't_distribution_params': t_params,
            
            # Momentum metrics
            'kalman_hull_value': momentum_result['trend_value'],
            'kalman_hull_direction': momentum_result['trend_direction'],
            'kalman_hull_strength': momentum_result['trend_strength'],
            'divergence': momentum_result['divergence'],
            
            # Volume metrics
            'volume_spike_score': spike_score,
            'pv_correlation': pv_correlation,
            'ad_signal': ad_signal,
            
            # Liquidity metrics
            'amihud': amihud,
            'avg_daily_volume': avg_volume,
            'zero_volume_days': zero_volume_days,
            
            # Metadata
            'risk_category': risk_category,
            'quality_tier': etf.quality_tier,
            'data_length': [etf.data](http://etf.data)_length
        }
```

### 4.3 Phase 3: Forecasting

**Process per ETF**:

```python
FOR EACH etf IN all_analyzed_etfs:
    
    # === PREPARE FEATURES ===
    
    features = [
        analysis[ticker]['cvar'],
        analysis[ticker]['ulcer_index'],
        analysis[ticker]['beta'],
        analysis[ticker]['information_ratio'],
        analysis[ticker]['kalman_hull_value'],
        analysis[ticker]['kalman_hull_direction'],
        analysis[ticker]['volume_spike_score'],
        analysis[ticker]['pv_correlation'],
        calculate_ad_momentum(analysis[ticker]['ad_line']),  # 20-day % change
        get_quality_score(analysis[ticker]['quality_tier'])
    ]
    
    # === HISTORICAL VALIDATION ===
    
    validation_results = []
    
    FOR t IN range(120, len([etf.data](http://etf.data)) - 60, 2):  # Step by 2 days
        # Extract features at time t
        historical_features = extract_features_at_time([etf.data](http://etf.data)[:t])
        
        # Calculate actual 60-day return
        price_t = [etf.data](http://etf.data)['Close'].iloc[t]
        price_t_plus_60 = [etf.data](http://etf.data)['Close'].iloc[t + 60]
        actual_return = (price_t_plus_60 / price_t) - 1
        
        validation_results.append({
            'features': historical_features,
            'actual_return': actual_return,
            'timestamp': [etf.data](http://etf.data).index[t]
        })
    
    IF len(validation_results) < 20:
        # Insufficient data for validation
        forecast_output = {
            'forecast': 0.0,
            'confidence': 0.1,
            'mae': NaN,
            'method': 'insufficient_data'
        }
        continue
    
    # === TRAIN/TEST SPLIT ===
    
    split_point = int(len(validation_results) × 0.7)
    split_point = max(split_point, len(validation_results) - 10)  # Ensure min 10 test samples
    
    X_train = [v['features'] for v in validation_results[:split_point]]
    y_train = [v['actual_return'] for v in validation_results[:split_point]]
    X_test = [v['features'] for v in validation_results[split_point:]]
    y_test = [v['actual_return'] for v in validation_results[split_point:]]
    
    # === MODEL TRAINING ===
    
    rf_model = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)
    rf_[model.fit](http://model.fit)(X_train, y_train)
    
    ridge_model = Ridge(alpha=1.0)
    ridge_[model.fit](http://model.fit)(X_train, y_train)
    
    # === PREDICTIONS ===
    
    rf_pred = rf_model.predict(X_test)
    ridge_pred = ridge_model.predict(X_test)
    ensemble_pred = 0.5 × rf_pred + 0.5 × ridge_pred
    
    # === VALIDATION METRICS ===
    
    mae = mean_absolute_error(y_test, ensemble_pred)
    hit_rate = mean((ensemble_pred > 0) == (y_test > 0))
    bias = mean(ensemble_pred - y_test)
    
    # === CONFIDENCE CALCULATION ===
    
    base_confidence = calculate_mae_confidence(mae, forecast_horizon=60)
    final_confidence = adjust_confidence_for_sample_size(base_confidence, len(X_test))
    
    # === GENERATE FINAL FORECAST ===
    
    rf_model_final = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)
    rf_model_[final.fit](http://final.fit)(X_train + X_test, y_train + y_test)
    
    ridge_model_final = Ridge(alpha=1.0)
    ridge_model_[final.fit](http://final.fit)(X_train + X_test, y_train + y_test)
    
    rf_forecast = rf_model_final.predict([features])[0]
    ridge_forecast = ridge_model_final.predict([features])[0]
    final_forecast = 0.5 × rf_forecast + 0.5 × ridge_forecast
    
    # === OUTPUT (NO ADJUSTMENTS) ===
    
    forecast_output = {
        'forecast': final_forecast,              # Raw ML prediction (e.g., 0.042 = 4.2%)
        'confidence': final_confidence,          # 0.0 - 1.0
        'mae': mae,                              # Mean Absolute Error (%)
        'hit_rate': hit_rate,                    # Directional accuracy
        'bias': bias,                            # Systematic error
        'sample_size': len(X_test),
        'method': 'ML'
    }
    
    analysis[ticker]['forecast'] = forecast_output
```

### 4.4 Phase 4: Context Calculation (Metadata Only)

```python
# === MACRO FRAMEWORK ===
macro_result = calculate_macro_framework()
# Returns: {multiplier, composite_score, regime, factors}

# === GEOPOLITICAL FRAMEWORK ===
geo_result = calculate_geopolitical_framework()
# Returns: {risk_score, risk_level, pillars}

# === ATTACH TO ALL ETFS ===
FOR EACH ticker IN analysis:
    analysis[ticker]['context'] = {
        'macro': macro_result,
        'geopolitical': geo_result
    }
```

### 4.5 Phase 5: Composite Scoring

**Process per ETF**:

```python
FOR EACH ticker IN analysis:
    
    # === COMPONENT SCORING (0-100 scale) ===
    
    # 1. RISK SCORE (30% weight)
    cvar_score = score_cvar(analysis[ticker]['cvar'])
    ulcer_score = score_ulcer(analysis[ticker]['ulcer_index'])
    beta_score = score_beta(analysis[ticker]['beta'], analysis[ticker]['risk_category'])
    ir_score = score_information_ratio(analysis[ticker]['information_ratio'])
    
    risk_score = (
        0.30 × cvar_score +
        0.30 × ulcer_score +
        0.20 × beta_score +
        0.20 × ir_score
    )
    
    # 2. MOMENTUM SCORE (25% weight)
    direction_score = score_direction(analysis[ticker]['kalman_hull_direction'])
    strength_score = score_strength(analysis[ticker]['kalman_hull_strength'])
    divergence_score = score_divergence(analysis[ticker]['divergence'])
    consistency_score = calculate_consistency_score(analysis[ticker])
    
    momentum_score = (
        0.40 × direction_score +
        0.30 × strength_score +
        0.20 × divergence_score +
        0.10 × consistency_score
    )
    
    # 3. VOLUME SCORE (20% weight)
    spike_score = analysis[ticker]['volume_spike_score']  # Already 0-100
    conviction_score = score_pv_correlation(analysis[ticker]['pv_correlation'])
    smart_money_score = score_ad_signal(analysis[ticker]['ad_signal'])
    
    volume_score = (
        0.40 × spike_score +
        0.35 × conviction_score +
        0.25 × smart_money_score
    )
    
    # 4. FORECAST SCORE (20% weight)
    forecast_value = analysis[ticker]['forecast']['forecast']
    forecast_confidence = analysis[ticker]['forecast']['confidence']
    
    forecast_score = score_forecast(forecast_value, forecast_confidence)
    
    # 5. CONTEXT SCORE (5% weight)
    macro_score = analysis[ticker]['context']['macro']['composite_score']
    geo_score = 100 - analysis[ticker]['context']['geopolitical']['risk_score']
    
    context_score = 0.5 × macro_score + 0.5 × geo_score
    
    # === APPLY RISK CATEGORY MULTIPLIERS ===
    
    multipliers = get_risk_multipliers(analysis[ticker]['risk_category'])
    # Low Risk:    [1.1, 0.9, 0.8, 1.0, 1.0]
    # Medium Risk: [1.0, 1.0, 1.0, 1.0, 1.0]
    # High Risk:   [0.9, 1.1, 1.2, 1.0, 1.0]
    
    risk_score *= multipliers[0]
    momentum_score *= multipliers[1]
    volume_score *= multipliers[2]
    # forecast_score and context_score not multiplied
    
    # === WEIGHTED COMPOSITE ===
    
    base_score = (
        risk_score × 0.30 +
        momentum_score × 0.25 +
        volume_score × 0.20 +
        forecast_score × 0.20 +
        context_score × 0.05
    )
    
    # === APPLY PENALTIES ===
    
    penalties = 0
    
    # CVaR penalties
    IF analysis[ticker]['cvar'] < -0.10:
        penalties += 15
    ELIF analysis[ticker]['cvar'] < -0.05:
        penalties += 5
    
    # Liquidity penalties
    IF analysis[ticker]['amihud'] > 1.0:
        penalties += 5
    
    IF analysis[ticker]['avg_daily_volume'] < 500000:
        penalties += 10
    ELIF analysis[ticker]['avg_daily_volume'] < 1000000:
        penalties += 5
    
    IF analysis[ticker]['zero_volume_days'] > 5:
        penalties += 5
    
    # Fundamental penalties (from ETF database)
    expense_ratio = get_expense_ratio(ticker)
    aum = get_aum(ticker)
    
    IF expense_ratio > 0.0075:
        penalties += 15
    ELIF expense_ratio > 0.0050:
        penalties += 10
    ELIF expense_ratio > 0.0025:
        penalties += 5
    
    IF aum < 50000000:
        penalties += 10
    ELIF aum < 100000000:
        penalties += 5
    
    # Quality tier penalty
    quality_penalty = get_quality_penalty(analysis[ticker]['quality_tier'])
    
    # === FINAL SCORE ===
    
    composite_score = base_score - penalties - quality_penalty
    composite_score = CLAMP(composite_score, 0, 100)
    
    analysis[ticker]['composite_score'] = composite_score
    analysis[ticker]['penalties'] = {
        'total': penalties + quality_penalty,
        'cvar': cvar_penalty,
        'liquidity': liquidity_penalty,
        'fundamental': fundamental_penalty,
        'quality': quality_penalty
    }
```

### 4.6 Phase 6: Ranking & Validation

```python
# === RANK WITHIN CATEGORIES ===
FOR EACH risk_category IN ['low_risk', 'medium_risk', 'high_risk']:
    etfs_in_category = filter_by_category(analysis, risk_category)
    sorted_etfs = sort_by_score(etfs_in_category, descending=True)
    
    FOR rank, (ticker, score) IN enumerate(sorted_etfs, start=1):
        analysis[ticker]['rank_in_category'] = rank
    
    category_rankings[risk_category] = sorted_etfs

# === CROSS-CATEGORY RANKINGS ===
all_etfs = combine_all_categories(analysis)
overall_rankings = sort_by_score(all_etfs, descending=True)

FOR rank, (ticker, score) IN enumerate(overall_rankings, start=1):
    analysis[ticker]['overall_rank'] = rank

# === VALIDATION CHECKS ===
validation_report = {
    'data_quality': assess_quality_distribution(analysis),
    'score_distribution': analyze_score_distribution(analysis),
    'component_correlation': check_component_independence(analysis),
    'forecast_validation': validate_forecast_metrics(analysis),
    'sanity_checks': run_sanity_checks(analysis, category_rankings)
}

# === GENERATE OUTPUTS ===
OUTPUT = {
    'analysis_results': analysis,  # Full per-ETF data
    'category_rankings': category_rankings,  # Ranked lists per category
    'overall_rankings': overall_rankings,  # All ETFs ranked
    'validation_report': validation_report,  # Quality checks
    'summary': {
        'total_analyzed': len(analysis),
        'low_risk_count': len(category_rankings['low_risk']),
        'medium_risk_count': len(category_rankings['medium_risk']),
        'high_risk_count': len(category_rankings['high_risk']),
        'processing_time': elapsed_time
    }
}
```

---

## 5. RETURN SPECIFICATIONS

### 5.1 Analysis Results Structure

**Per-ETF Dictionary**:

```python
analysis_results[ticker] = {
    # === IDENTIFIERS ===
    'ticker': str,  # e.g., '[VAS.AX](http://VAS.AX)'
    'name': str,    # ETF full name
    'risk_category': str,  # 'low_risk', 'medium_risk', 'high_risk'
    
    # === CORE SCORES ===
    'composite_score': float,        # 0-100 final score
    'rank_in_category': int,         # 1, 2, 3, ...
    'overall_rank': int,             # 1, 2, 3, ...
    
    # === RISK LAYER ===
    'cvar': float,                   # e.g., -0.085 = -8.5%
    'ulcer_index': float,            # e.g., 0.035 = 3.5%
    'beta': float,                   # e.g., 0.98
    'information_ratio': float,      # e.g., 0.45
    'risk_score': float,             # 0-100 component score
    
    # === MOMENTUM LAYER ===
    'kalman_hull_value': float,      # Smoothed price level
    'kalman_hull_direction': int,    # 1, 0, or -1
    'kalman_hull_upper_band': float, # Upper volatility band
    'kalman_hull_lower_band': float, # Lower volatility band
    'divergence': str,               # 'bullish', 'bearish', 'none'
    'momentum_score': float,         # 0-100 component score
    
    # === VOLUME LAYER ===
    'volume_spike_score': float,     # 0-100
    'pv_correlation': float,         # -1.0 to +1.0
    'ad_signal': str,                # 'accumulation', 'distribution', 'neutral'
    'volume_score': float,           # 0-100 component score
    
    # === FORECAST ===
    'forecast': float,               # 60-day return prediction (e.g., 0.042 = 4.2%)
    'forecast_confidence': float,    # 0.0 - 1.0
    'mae': float,                    # Mean Absolute Error (%)
    'hit_rate': float,               # Directional accuracy (0.0 - 1.0)
    'bias': float,                   # Systematic error (%)
    'forecast_score': float,         # 0-100 component score
    
    # === LIQUIDITY ===
    'amihud': float,                 # Illiquidity ratio
    'avg_daily_volume': float,       # Dollar volume
    'zero_volume_days': int,         # Count in last 60 days
    
    # === FUNDAMENTALS (from database) ===
    'expense_ratio': float,          # e.g., 0.0010 = 0.10%
    'aum_aud': float,                # Assets under management
    'inception_date': str,           # YYYY-MM-DD
    'type': str,                     # 'equity', 'bond', 'commodity', etc.
    'region': str,                   # 'AUSTRALIA', 'INTERNATIONAL', etc.
    'subcategory': str,              # 'Broad Market', 'Technology', etc.
    
    # === PERFORMANCE (calculated) ===
    'ytd_return': float,             # Year-to-date return
    'one_year_return': float,        # Trailing 1-year return
    'volatility': float,             # Annualized volatility
    'latest_price': float,           # Current price
    
    # === QUALITY ===
    'quality_tier': str,             # 'tier_1', 'tier_2', 'tier_3', 'tier_4'
    'data_length': int,              # Number of trading days available
    'quality_score': float,          # 0.0 - 1.0
    
    # === PENALTIES ===
    'penalties': {
        'total': float,              # Sum of all penalties
        'cvar': float,               # CVaR-based penalty
        'liquidity': float,          # Volume/Amihud penalties
        'fundamental': float,        # Expense/AUM penalties
        'quality': float             # Data quality penalty
    },
    
    # === CONTEXT (METADATA) ===
    'context': {
        'macro': {
            'multiplier': float,     # 0.75 - 1.25
            'regime': str,           # 'GOLDILOCKS', 'TRANSITIONAL', 'CRISIS'
            'composite_score': float # 0-100
        },
        'geopolitical': {
            'risk_score': float,     # 0-100
            'risk_level': str        # 'LOW', 'MODERATE', 'HIGH', 'SEVERE', 'EXTREME'
        }
    },
    
    # === T-DISTRIBUTION PARAMS (for transparency) ===
    't_distribution_params': {
        'degrees_of_freedom': float,
        'location': float,
        'scale': float
    }
}
```

### 5.2 Category Rankings Structure

```python
category_rankings = {
    'low_risk': [
        ('[VAF.AX](http://VAF.AX)', 78.5),
        ('[VGB.AX](http://VGB.AX)', 76.2),
        ('[AAA.AX](http://AAA.AX)', 72.1),
        ...
    ],
    'medium_risk': [
        ('[VAS.AX](http://VAS.AX)', 67.8),
        ('[IOZ.AX](http://IOZ.AX)', 65.4),
        ...
    ],
    'high_risk': [
        ('[NDQ.AX](http://NDQ.AX)', 71.2),
        ('[HACK.AX](http://HACK.AX)', 68.9),
        ...
    ]
}
```

### 5.3 Dashboard Output Format

**Based on current `dashboard/[app.py](http://app.py)` structure:**

### 5.3.1 Main Summary Table

**Columns**:

1. Rank (overall)
2. Ticker
3. Name
4. Risk Category
5. Composite Score
6. Forecast (60-day %)
7. Confidence
8. YTD Return (%)
9. 1Y Return (%)
10. Volatility (%)
11. Beta
12. Expense Ratio (%)

**Formatting**:

- Scores: 1 decimal place
- Returns/Forecasts: 1 decimal place with +/- sign
- Confidence: Percentage (0-100%)
- Color coding: Green (score >70), Yellow (50-70), Red (<50)

### 5.3.2 Detailed ETF Cards (per ETF)

**Section 1: Overview**

- Ticker, Name, Category
- Composite Score (large, prominent)
- Overall Rank / Category Rank
- Quality Tier badge

**Section 2: Risk Metrics**

- CVaR (bar chart, color-coded)
- Ulcer Index (bar chart)
- Beta (gauge)
- Information Ratio (bar chart)
- Component Score: X/100

**Section 3: Momentum Signals**

- Kalman Hull trend direction (arrow: ↑↓→)
- Current price vs bands (visual band chart)
- Divergence signal (badge if present)
- Component Score: X/100

**Section 4: Volume Intelligence**

- Spike Score (gauge: 0-100)
- P-V Correlation (bar: -1 to +1)
- A/D Signal (badge: Accumulation/Distribution/Neutral)
- Component Score: X/100

**Section 5: Forecast**

- 60-day prediction (large, color-coded)
- Confidence level (percentage bar)
- MAE (small text)
- Hit Rate (small text)
- Component Score: X/100

**Section 6: Fundamentals**

- Expense Ratio
- AUM
- Avg Daily Volume
- Inception Date
- ETF Type/Region/Subcategory

**Section 7: Performance**

- **YTD Return** (bar chart with percentage, color-coded: green if positive, red if negative)
- **1-Year Return** (bar chart with percentage, color-coded: green if positive, red if negative)
- Latest Price
- Volatility (annualized %)
- **Data Note**: Returns displayed as "N/A" if insufficient historical data available

**Section 8: Context (Collapsible)**

- Macro Regime: [CRISIS/TRANSITIONAL/GOLDILOCKS]
- Macro Multiplier: X.XX
- Geopolitical Risk: [LOW/MODERATE/HIGH/SEVERE/EXTREME]
- Geo Risk Score: X/100
- Note: "Context not auto-applied. User can apply manually if desired."

**Section 9: Penalties Applied**

- Total Penalties: -X points
- Breakdown: CVaR (-X), Liquidity (-X), Fundamental (-X), Quality (-X)

---

## 6. IMPLEMENTATION CONSTRAINTS

### 6.1 Critical Constraints

1. **NO bias correction** applied to ML forecasts
2. **NO automatic application** of macro/geo multipliers
3. **NO fallback mechanism** in forecasting (always use ML)
4. **Information Ratio must be included** (unique dimension)
5. **Quality tiers transparent** (never hide data limitations)
6. **All penalties explicit** (user sees exactly what reduces score)
7. **Context as metadata only** (user controls application)

### 6.2 Required Libraries

**Core**:

- pandas >= 1.5.0
- numpy >= 1.23.0
- scipy >= 1.9.0

**Data**:

- yfinance >= 0.2.0

**ML**:

- scikit-learn >= 1.2.0

**Utilities**:

- warnings (built-in)
- time (built-in)
- typing (built-in)

### 6.3 Execution Flow

1. Initialize system: `ETFAnalysisSystem()`
2. Load ETF universe: read from database or user input
3. Execute: `system.analyze_etfs(etf_tickers, enable_validation=True)`
4. Generate dashboard: `DashboardGenerator(results).create_output()`
5. Export: Save results to files/database

### 6.4 Performance Optimization

**Expected Runtime**:

- 100 ETFs: 2-3 minutes
- 300 ETFs: 5-10 minutes
- 367 ETFs: 8-15 minutes

**Caching**:

- Benchmark data (reuse across ETFs)
- VIX data (single download)
- Macro/Geo framework (calculate once)

**Error Handling**:

- Missing Data: Use last valid value (forward fill)
- Failed Calculations: Return neutral score (50) for that component
- Insufficient Samples: Apply quality penalties transparently

---

**END OF SPECIFICATION**