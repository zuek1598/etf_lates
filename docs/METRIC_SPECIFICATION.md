# ETF Metric Calculation Specification

## Overview
This document defines the exact formulas and parameters for all ETF quality metrics. These specifications MUST be followed identically when building confidence intervals AND during backtesting to ensure consistency.

## Core Metrics

### 1. Hit Rate
**Definition**: Percentage of periods with positive returns.

**Formula**:
```
hit_rate = count(returns > 0) / N
```

**Parameters**:
- Window: Last 100 trading days
- Range: 0 to 1 (0% = all negative returns, 100% = all positive returns)

**Calculation Steps**:
1. Calculate daily returns: `returns = prices.pct_change().dropna()`
2. Take last 100 days: `recent_returns = returns.tail(100)`
3. Count positive returns: `positive = (recent_returns > 0).sum()`
4. Calculate hit rate: `hit_rate = positive / 100`

**Example**: If 54 out of 100 days have positive returns → hit_rate = 0.54

### 2. Conviction
**Definition**: Average strength of recent momentum relative to historical momentum.

**Formula**:
```
conviction = recent_momentum / abs(avg_momentum)
```

**Parameters**:
- Recent momentum: Mean of last 10 days returns
- Average momentum: Mean of last 40 days returns
- Range: Unbounded (can be negative or positive)

**Calculation Steps**:
1. Calculate daily returns: `returns = prices.pct_change().dropna()`
2. Recent momentum: `recent_momentum = returns.tail(10).mean()`
3. Average momentum: `avg_momentum = returns.tail(40).mean()`
4. Conviction: `recent_momentum / abs(avg_momentum)`
5. If avg_momentum = 0, conviction = 0

**Interpretation**:
- Positive conviction: Recent momentum in same direction as historical
- Negative conviction: Recent momentum opposite to historical
- Higher absolute value: Stronger conviction

### 3. Stability
**Definition**: Inverted volatility, bounded between 0 and 1. Higher values indicate more stable (less volatile) ETFs.

**Formula**:
```
stability = 1.0 / (1.0 + volatility_30d * 10)
```

**Parameters**:
- volatility_30d: Standard deviation of last 30 days daily returns
- Range: 0 to 1 (0 = infinite volatility, 1 = zero volatility)

**Calculation Steps**:
1. Calculate daily returns: `returns = prices.pct_change().dropna()`
2. Get volatility: `volatility = returns.tail(30).std()`
3. Apply bounded inverse: `stability = 1.0 / (1.0 + volatility * 10)`

**Examples**:
- Daily volatility 1.5% → stability = 1/(1+0.15) = 0.87
- Daily volatility 3.0% → stability = 1/(1+0.30) = 0.77
- Daily volatility 0.5% → stability = 1/(1+0.05) = 0.95

## Composite Metrics

### Ranking Score
**Formula**:
```
ranking_score = (hit_rate * 0.35) + (conviction_normalized * 0.40) + (stability * 0.25)
```

**Note**: Conviction is normalized using sigmoid for ranking, but raw conviction is used for CI calculations.

## Confidence Intervals

### Calculation Method
- Type: Percentile-based (non-parametric)
- Confidence Level: 95%
- Percentiles: 2.5th and 97.5th
- Lookback Period: Last 100 trading days of metric values
- Update Frequency: Daily (rolling window)

### Outlier Handling
- Conviction values are winsorized at 1st and 99th percentiles before CI calculation
- This prevents extreme values from skewing the confidence intervals

### Example Calculation
For HACK conviction over 100 days: [2.8, 3.1, 3.2, 2.9, ..., 3.0]

1. Sort values: [2.4, 2.5, 2.6, ..., 3.5, 3.6, 3.7]
2. Apply winsorization: Clip values below 2.5th percentile and above 99th percentile
3. Calculate bounds:
   - Lower bound: 2.5th percentile ≈ 2.55
   - Upper bound: 97.5th percentile ≈ 3.45

## Noise Detection Logic

### Weighted Distance Approach
For each metric outside its confidence interval:

1. Calculate distance beyond CI bound:
   ```
   distance = abs(current_value - nearest_bound)
   percentage_distance = distance / abs(nearest_bound)
   ```

2. Assign weight based on distance:
   - 0% to 10% beyond CI: weight = 0.3 (borderline noise)
   - 10% to 20% beyond CI: weight = 0.7 (minor signal)
   - 20%+ beyond CI: weight = 1.0 (strong signal)

3. Sum weights across all three metrics:
   ```
   total_weight = sum(weights_of_metrics_outside_CI)
   ```

4. Decision rule:
   - total_weight < 0.5: HOLD (likely noise)
   - 0.5 ≤ total_weight ≤ 1.5: WAIT (minor signal, confirm next period)
   - total_weight > 1.5: ROTATE (strong signal)

### Market-Wide Stress Detection
- Trigger when >30% of ETFs in top 20 show strong signals (total_weight > 1.5)
- When in stress regime, ALL positions are held regardless of individual signals

## Implementation Requirements

### Critical Rules
1. **Single Source of Truth**: All metric calculations must use the same code path
2. **Identical Parameters**: Window sizes, benchmarks, and formulas must match exactly
3. **Audit Trail**: Log all calculated metrics with timestamps for verification
4. **Regular Verification**: Compare CI bounds to actual metric distributions quarterly

### Verification Procedure
After each backtest:
1. Extract all metric values from backtest logs
2. For each ETF, recalculate percentiles on actual values
3. Compare to pre-calculated CI bounds
4. Discrepancies >2% require investigation

### File Locations
- Specification: `/docs/METRIC_SPECIFICATION.md` (this file)
- Implementation: `/analyzers/metric_calculation.py`
- Verification: `/scripts/verify_standardization.py`
- CI Database: `/data/confidence_intervals.pkl`

## Version History
- v1.0 (2025-01-26): Initial specification with bounded stability formula
- v0.9 (2025-01-25): Used unbounded stability (1/volatility) - DEPRECATED
