# Phase 2: Professional Backtesting - Fix Implementation

## Issue Discovered

During initial testing of the professional backtester, immature ETFs (60-311 days of data) were failing with "No trades executed" while mature ETFs worked correctly.

### Root Cause Analysis

**Immature ETF Signal Problem:**
1. Volume intelligence indicators require substantial data to establish baselines
2. With only 60-305 days of data, volume indicators return 0.0 score
3. Composite score formula: `60% × Kalman + 40% × Volume`
4. Example FHNG.AX (305 days, HIGH volatility):
   - Kalman score: ~40-60 (reasonable)
   - Volume score: 0.0 (weak on limited data)
   - Composite: 24-36 (below mature threshold of 50)
5. Buy condition requires: `score >= 50 AND ml_forecast > 0 AND kalman_trend == 1`
6. Result: No buy signals generated, no trades executed

## Solution Implemented

**Lower Buy Threshold for Immature ETFs**

Modified `utilities/professional_backtester.py` method `backtest_immature_etf()`:

```python
# For immature ETFs, use lower threshold since volume indicators weak with limited data
immature_buy_threshold = max(35, self.buy_threshold * 0.7)  # ~35 instead of 50

# BUY signal for immature ETFs: slightly relaxed thresholds
if score >= immature_buy_threshold and ml_forecast > 0 and kalman_trend == 1:
```

**Rationale:**
- Immature ETFs have less data to establish volume patterns
- Volume score becomes unreliable/weak
- Kalman trend signal is more reliable with trend-following indicator
- Lowering threshold from 50 to 35 allows signals while maintaining control
- Other conditions (ml_forecast > 0, kalman_trend == 1) remain strict

## Test Results

**Before Fix:**
- FCAP.AX (295 days): FAILED - No trades executed
- FHNG.AX (305 days): FAILED - No trades executed
- IEM.AX (4512 days): SUCCESS
- IVV.AX (4511 days): SUCCESS
- SFY.AX (4511 days): SUCCESS
- **Result: 3/5 passed (60%)**

**After Fix:**
- FCAP.AX (295 days): SUCCESS - +1.38% return, 3 trades, 33.3% win rate
- FHNG.AX (305 days): SUCCESS - +1.81% return, 3 trades, 66.7% win rate
- IEM.AX (4512 days): SUCCESS - +0.05% return, 4 trades, 50.0% win rate
- IVV.AX (4511 days): SUCCESS - Results completed
- SFY.AX (4511 days): SUCCESS - -0.27% return, 9 trades, 44.4% win rate
- **Result: 5/5 passed (100%)**

## Signal Quality Check

Diagnostic on FHNG.AX showing when signals trigger with new threshold:

| Day | Window | Score | ML Forecast | Kalman | Mature (50+) | Immature (35+) |
|-----|--------|-------|-------------|--------|--------------|-----------------|
| 60  | 61 days | 41.9 | -1.95 | 1 | NO | FAIL (ML) |
| 90  | 91 days | 40.7 | +23.06 | 1 | NO | **YES** |
| 120 | 121 days | 34.9 | +15.09 | 1 | NO | NO (score) |
| 150 | 151 days | 7.0 | -6.94 | -1 | NO | NO |
| 180 | 181 days | 42.0 | -7.76 | 1 | NO | FAIL (ML) |
| 210 | 211 days | 35.3 | +13.64 | 1 | NO | **YES** |
| 240 | 241 days | 34.8 | +14.66 | 1 | NO | NO (score) |
| 270 | 271 days | 40.0 | +11.83 | 1 | NO | **YES** |

Shows that immature threshold of 35 enables 3 buy signals (days 90, 210, 270) that mature threshold of 50 would miss.

## Impact

- Immature ETFs now backtest successfully with appropriate signal generation
- 30 immature ETFs (out of 349 eligible) can now be evaluated
- Risk-adjusted signals still applied (ml_forecast > 0, kalman_trend == 1)
- Solution is data-aware: threshold automatically scales based on backtester configuration

## Files Modified

1. `utilities/professional_backtester.py` - Line 231: Added immature_buy_threshold logic
2. Test validation: `test_backtester_small.py` - Now passes 5/5 tests

## Next Steps

- Run full universe backtest on all 349 eligible ETFs
- Generate comprehensive performance metrics
- Validate signal effectiveness across all categories
- Compare mature vs immature ETF performance
