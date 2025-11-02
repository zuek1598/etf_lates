# ETF Scoring System Fix Action Plan

## Executive Summary
The current scoring system has critical flaws causing poor ETF rankings and score clustering. The orchestrator uses the old `ScoringRankingSystem` instead of the improved `GrowthScoringSystem`, resulting in:
- 95.5% of ETFs scoring below 50 (clustered 20-40 range)
- SNAS.AX (with corporate action) getting highest score (48.0)
- Component scores not saved or accessible
- Overly aggressive penalty stacking

**Timeline:** 2-3 weeks  
**Impact:** Transform scoring from broken to robust, enabling proper ETF ranking

---

## PHASE 1: IMMEDIATE FIXES (Week 1)
*Fix the core system architecture*

### 1.1 Switch to Growth Scoring System
**Status:** ❌ Critical - Wrong system being used  
**Impact:** High - Fixes scaling, penalties, and component storage  
**Time:** 1-2 days

**Tasks:**
- [ ] **Update orchestrator import**
  ```python
  # orchestrator.py line 19
  from analyzers.scoring_system_growth import GrowthScoringSystem
  ```
- [ ] **Update orchestrator initialization**
  ```python
  # orchestrator.py line 33
  self.scoring_system = GrowthScoringSystem()
  ```
- [ ] **Update method calls** (ensure compatibility)
- [ ] **Test system startup** and basic functionality

**Dependencies:**
- Verify GrowthScoringSystem has all required methods

**Testing:**
- System starts without errors
- Composite scores are calculated
- Component scores are accessible

---

### 1.2 Fix Component Score Storage
**Status:** ❌ Component scores show as "N/A" in data  
**Impact:** High - Cannot analyze individual signal contributions  
**Time:** 1-2 days

**Tasks:**
- [ ] **Update orchestrator to store component scores**
  ```python
  # In orchestrator.py around line 378
  if ticker in all_results:
      all_results[ticker]['composite_score'] = score
      all_results[ticker]['component_scores'] = result.get('components', {})  # NEW
      all_results[ticker]['adjusted_components'] = result.get('adjusted_components', {})  # NEW
  ```
- [ ] **Update data schema** to include component fields
- [ ] **Verify component scores are saved** to parquet files

**Dependencies:**
- GrowthScoringSystem returns component data correctly

**Testing:**
- Component scores appear in data files (not "N/A")
- Risk, momentum, forecast, volume scores are numeric
- Adjusted component scores reflect risk category multipliers

---

### 1.3 Reduce Penalty Aggressiveness
**Status:** ❌ Penalties stack to destroy scores (60% → 21.6)  
**Impact:** High - Good ETFs get crushed by minor issues  
**Time:** 2-3 days

**Current Problem (GrowthScoringSystem):**
```python
# Multiple 20-40% penalties stack multiplicatively
composite *= 0.60  # Liquidity
composite *= 0.80  # Amihud  
composite *= 0.75  # CVaR
Result: 60 × 0.60 × 0.80 × 0.75 = 21.6 (-78.4%)
```

**Solution - Additive Penalties:**
```python
def apply_growth_penalties_fixed(self, composite: float, analysis: Dict, risk_category: str) -> float:
    """Additive penalties with maximum caps"""
    penalty_points = 0.0
    
    # CVaR penalties (additive, max 20 points)
    cvar = analysis.get('cvar', 0.0)
    if cvar < -0.50: penalty_points += 20  # Extreme tail risk
    elif cvar < -0.30: penalty_points += 15  # High tail risk
    elif cvar < -0.20: penalty_points += 10  # Moderate tail risk
    
    # Liquidity penalties (additive, max 15 points)
    volume = analysis.get('avg_daily_volume', np.nan)
    if volume < 50_000: penalty_points += 15   # Very low liquidity
    elif volume < 200_000: penalty_points += 10  # Low liquidity
    elif volume < 500_000: penalty_points += 5   # Moderate liquidity
    
    # Amihud penalty (additive, max 10 points)
    amihud = analysis.get('amihud', np.nan)
    if amihud > 10.0: penalty_points += 10
    elif amihud > 5.0: penalty_points += 5
    
    # Cap total penalties at 30 points (maximum -30% reduction)
    penalty_points = min(penalty_points, 30.0)
    
    # Apply as percentage reduction
    penalty_factor = 1.0 - (penalty_points / 100.0)
    return composite * penalty_factor
```

**Tasks:**
- [ ] **Replace multiplicative penalties** with additive system
- [ ] **Add penalty caps** (maximum 30 point deduction)
- [ ] **Test penalty impact** on various ETF profiles

**Dependencies:**
- GrowthScoringSystem penalty method update

**Testing:**
- ETFs with minor issues don't get destroyed
- Severe issues still penalized appropriately
- Score distribution spreads out (not clustered at bottom)

---

## PHASE 2: SCALING & RELIABILITY FIXES (Week 2)
*Ensure proper score distribution and reliability*

### 2.1 Fix Component Score Scaling
**Status:** ⚠️ Components don't use full 0-100 range  
**Impact:** Medium - Contributes to score clustering  
**Time:** 2-3 days

**Current Issues:**
- Momentum scores cluster 50-70 (not using full range)
- Forecast scores may exceed 100 or not reach 0
- Volume scores may not use full scale

**Solution - Normalize All Components:**
```python
def score_momentum_fixed(self, trend: int, signal_strength: float, efficiency_ratio: float, divergence: str) -> float:
    """Ensure full 0-100 utilization"""
    # Current logic but ensure full range usage
    momentum_quality = (signal_strength * 0.6) + (efficiency_ratio * 0.4)
    
    # Map quality 0-1 to score 0-100, not just 25-85
    if momentum_quality > 0.8: return 95.0  # Excellent
    elif momentum_quality > 0.7: return 85.0
    elif momentum_quality > 0.6: return 75.0
    elif momentum_quality > 0.5: return 65.0
    elif momentum_quality > 0.4: return 55.0
    elif momentum_quality > 0.3: return 45.0
    elif momentum_quality > 0.2: return 35.0
    else: return 25.0  # Poor
```

**Tasks:**
- [ ] **Audit each component scoring function** for range utilization
- [ ] **Add explicit range mapping** where needed
- [ ] **Ensure extreme cases hit 0-100 bounds**
- [ ] **Test score distributions** for each component

**Dependencies:**
- Access to individual component scoring functions

**Testing:**
- Each component shows full 0-100 range utilization
- Extreme cases (best/worst) hit appropriate bounds
- Component scores correlate properly with underlying signals

---

### 2.2 Implement Corporate Action Detection
**Status:** ❌ SNAS gets high score despite 1957% jump  
**Impact:** High - Invalid rankings from structural changes  
**Time:** 2-3 days

**Solution - Corporate Action Flagging:**
```python
def detect_corporate_action(self, prices: pd.Series) -> bool:
    """Flag ETFs with extreme price jumps (corporate actions)"""
    if len(prices) < 10:
        return False
    
    # Check for single-day jumps >300% or <-50%
    daily_returns = prices.pct_change().dropna()
    max_jump = daily_returns.max()
    min_jump = daily_returns.min()
    
    return max_jump > 3.0 or min_jump < -0.5

def calculate_composite_score_with_ca_check(self, analysis: Dict, risk_category: str) -> Dict:
    """Flag corporate action ETFs as unreliable"""
    # Check for corporate actions in price data
    prices = analysis.get('price_data', pd.Series())
    has_corporate_action = self.detect_corporate_action(prices)
    
    if has_corporate_action:
        return {
            'composite_score': 0.0,  # Flag as unreliable
            'corporate_action_detected': True,
            'components': {},  # No valid component scores
            'warning': 'Corporate action detected - ranking unreliable'
        }
    
    # Normal calculation
    return self.calculate_composite_score(analysis, risk_category)
```

**Tasks:**
- [ ] **Add corporate action detection function**
- [ ] **Integrate into scoring pipeline**
- [ ] **Flag affected ETFs** with zero score and warning
- [ ] **Update dashboard** to show corporate action warnings

**Dependencies:**
- Access to price data in scoring function

**Testing:**
- SNAS.AX gets flagged as corporate action ETF
- Other ETFs without corporate actions score normally
- Dashboard shows appropriate warnings

---

### 2.3 Improve Score Distribution Analysis
**Status:** ⚠️ No validation of score distribution health  
**Impact:** Medium - Cannot detect when scoring is broken  
**Time:** 1-2 days

**Solution - Score Distribution Monitoring:**
```python
def analyze_score_distribution(self, all_scores: List[float]) -> Dict:
    """Monitor score distribution health"""
    scores = np.array(all_scores)
    
    return {
        'mean': np.mean(scores),
        'median': np.median(scores),
        'std': np.std(scores),
        'range': np.max(scores) - np.min(scores),
        'percentiles': {
            '10th': np.percentile(scores, 10),
            '25th': np.percentile(scores, 25),
            '75th': np.percentile(scores, 75),
            '90th': np.percentile(scores, 90)
        },
        'distribution_health': self._check_distribution_health(scores)
    }

def _check_distribution_health(self, scores: np.array) -> str:
    """Check if distribution indicates healthy scoring"""
    mean = np.mean(scores)
    std = np.std(scores)
    
    # Warning signs of broken scoring
    if std < 5:  # Too clustered
        return "WARNING: Scores too clustered (std < 5)"
    elif mean < 30 or mean > 70:  # Biased distribution
        return "WARNING: Distribution biased (mean not 40-60)"
    elif np.max(scores) - np.min(scores) < 40:  # Not using full range
        return "WARNING: Not using full score range"
    else:
        return "HEALTHY: Good distribution spread"
```

**Tasks:**
- [ ] **Add distribution analysis functions**
- [ ] **Integrate into post-scoring analysis**
- [ ] **Log distribution health warnings**
- [ ] **Create score distribution dashboard**

**Dependencies:**
- Access to all calculated scores

**Testing:**
- Distribution analysis identifies current clustering problem
- After fixes, shows healthy distribution
- Warnings alert when scoring becomes unhealthy

---

## PHASE 3: VALIDATION & OPTIMIZATION (Week 3)
*Ensure fixes work and optimize performance*

### 3.1 Backtest Validation
**Status:** ⚠️ No validation that fixes improve rankings  
**Impact:** High - Cannot confirm fixes work  
**Time:** 3-4 days

**Tasks:**
- [ ] **Run backtests** before and after fixes
- [ ] **Compare score distributions** (before: clustered 20-40, after: spread 0-100)
- [ ] **Validate top ETF rankings** make sense qualitatively
- [ ] **Check component contribution** to final scores
- [ ] **Test edge cases** (very risky, very safe, corporate action ETFs)

**Dependencies:**
- Working backtest engine
- Historical data availability

**Testing:**
- Before/after comparison shows clear improvement
- Top-ranked ETFs have strong fundamentals
- Component scores correlate with underlying signals

---

### 3.2 Performance Benchmarking
**Status:** ❌ No comparison against reasonable benchmarks  
**Impact:** Medium - Cannot assess if scoring is skillful  
**Time:** 2-3 days

**Solution - Multi-Benchmark Comparison:**
```python
def benchmark_scoring_system(self) -> Dict:
    """Compare scoring against multiple benchmarks"""
    
    benchmarks = {
        'equal_weight': 'Equal weight all ETFs',
        'market_cap_weighted': 'Weighted by AUM',
        'risk_weighted': 'Weighted by inverse volatility',
        'momentum_weighted': 'Weighted by 6-month momentum',
        'random': 'Random selection'
    }
    
    results = {}
    for benchmark_name, description in benchmarks.items():
        # Calculate benchmark performance
        benchmark_returns = self.calculate_benchmark_returns(benchmark_name)
        results[benchmark_name] = {
            'description': description,
            'sharpe_ratio': benchmark_returns.sharpe(),
            'max_drawdown': benchmark_returns.max_drawdown(),
            'total_return': benchmark_returns.total_return()
        }
    
    # Compare scoring system performance
    scoring_returns = self.calculate_scoring_system_returns()
    results['scoring_system'] = {
        'sharpe_ratio': scoring_returns.sharpe(),
        'max_drawdown': scoring_returns.max_drawdown(),
        'total_return': scoring_returns.total_return()
    }
    
    return results
```

**Tasks:**
- [ ] **Implement benchmark comparisons**
- [ ] **Calculate scoring system performance**
- [ ] **Generate performance reports**
- [ ] **Identify where scoring adds value**

**Dependencies:**
- Historical return data
- Backtest framework

**Testing:**
- Scoring system outperforms random selection
- Clear differentiation from naive benchmarks
- Performance attribution shows which components add value

---

### 3.3 Documentation & Maintenance
**Status:** ⚠️ Scoring system changes not documented  
**Impact:** Low - Future maintenance harder  
**Time:** 1-2 days

**Tasks:**
- [ ] **Document scoring methodology** in detail
- [ ] **Create scoring system validation tests**
- [ ] **Add monitoring alerts** for score distribution issues
- [ ] **Document penalty calibration** and rationale
- [ ] **Create troubleshooting guide** for scoring issues

**Dependencies:**
- All fixes completed and tested

**Testing:**
- Documentation covers all edge cases
- Validation tests catch regressions
- Alerts work when scoring becomes unhealthy

---

## SUCCESS METRICS

### Phase 1 Success Criteria
- [ ] Growth Scoring System successfully integrated
- [ ] Component scores saved and accessible in data
- [ ] Penalties reduced from multiplicative stacking to additive caps
- [ ] System runs without errors

### Phase 2 Success Criteria  
- [ ] Score distribution spreads across 0-100 range (not clustered 20-40)
- [ ] Corporate action ETFs (like SNAS) flagged as unreliable
- [ ] Each component utilizes full 0-100 scale appropriately
- [ ] Score distribution health monitoring implemented

### Phase 3 Success Criteria
- [ ] Backtests show improved ranking quality
- [ ] Scoring system outperforms random benchmark
- [ ] Top-ranked ETFs have strong qualitative fundamentals
- [ ] Comprehensive documentation and monitoring in place

---

## DEPENDENCY MANAGEMENT

### Critical Dependencies
- **Growth Scoring System**: Must have all required methods and return component data
- **Historical Price Data**: For corporate action detection and backtesting
- **Component Data Pipeline**: Risk, ML, Kalman, Volume data must be available
- **Backtest Framework**: For validation and benchmarking

### Risk Mitigation
- **Incremental Changes**: Test each fix individually before combining
- **Rollback Plan**: Keep old scoring system as backup
- **Data Validation**: Extensive testing of score calculations
- **Performance Monitoring**: Track system health after deployment

---

## IMPLEMENTATION SEQUENCE

**Week 1:**
1. Switch to Growth Scoring System
2. Fix component score storage  
3. Implement additive penalties

**Week 2:**
4. Fix component scaling issues
5. Add corporate action detection
6. Implement distribution monitoring

**Week 3:**
7. Backtest validation
8. Performance benchmarking
9. Documentation and maintenance setup

---

## CONCLUSION

This action plan systematically addresses all identified scoring system issues:

- **Architecture**: Switch to correct scoring system
- **Scaling**: Fix component ranges and penalty stacking  
- **Reliability**: Add corporate action detection
- **Validation**: Comprehensive testing and benchmarking

**Result:** Scoring system transforms from broken (clustered, invalid rankings) to robust (distributed, reliable rankings) enabling proper ETF selection and portfolio construction.

**Ready to proceed with implementation when authorized.**
