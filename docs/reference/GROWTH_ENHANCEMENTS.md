# Growth-Optimized System Enhancements

**Date:** October 25, 2025  
**Version:** 2.1 (Growth Focus)

---

## 🎯 Overview

This document describes the growth-focused enhancements to the ETF Analysis System, including reweighted scoring, backtesting capabilities, and enhanced dashboard visualizations.

---

## 📊 **1. Growth-Optimized Scoring System**

### Location
`analyzers/scoring_system_growth.py`

### Key Changes from Original System

#### Weight Redistribution
```
ORIGINAL:  Risk(40%) + Technical(30%) + ML+Volume(30%)
GROWTH:    Momentum(35%) + Forecast(25%) + Risk(25%) + Volume(15%)
```

**Rationale:**
- **↑ Momentum (30% → 35%)**: Primary signal for growth trading
- **↑ Forecast (18% → 25%)**: Directional bias from ML predictions
- **↓ Risk (40% → 25%)**: Accept volatility for growth potential
- **Volume (12% → 15%)**: Confirmation signal

#### Risk Category Multipliers (HIGH Risk ETFs)

| Component | LOW Risk | MEDIUM Risk | HIGH Risk |
|-----------|----------|-------------|-----------|
| Momentum  | 0.8×     | 1.0×        | **1.3×** ↑ |
| Forecast  | 0.9×     | 1.0×        | **1.2×** ↑ |
| Risk      | 1.3×     | 1.0×        | **0.7×** ↓ |
| Volume    | 1.0×     | 1.0×        | **1.1×** ↑ |

**Effect**: HIGH risk ETFs are boosted for strong momentum/forecast, penalized less for volatility.

---

### Momentum Scoring Enhancement

**Now includes:**
1. **Signal Strength** (60%): Kalman Hull's position within bands
2. **Efficiency Ratio** (40%): Trend quality (directional movement vs. noise)

**Thresholds:**
```python
Momentum Score = (signal_strength × 0.6) + (efficiency_ratio × 0.4)

> 0.75:  Strong momentum → Full position size (100%)
0.60-0.75:  Good momentum → 85% position
0.45-0.60:  Moderate → 65% position
0.30-0.45:  Weak → 40% position
< 0.30:  Very weak → 20% position (exit consideration)
```

---

### Position Sizing Logic

**Integrated into scoring system:**

| Risk Category | Base Size | Strong Momentum | Good Momentum | Weak Momentum |
|---------------|-----------|-----------------|---------------|---------------|
| **LOW**       | 15%       | 15.0%           | 12.8%         | 6.0%          |
| **MEDIUM**    | 12%       | 12.0%           | 10.2%         | 4.8%          |
| **HIGH**      | 8%        | 8.0%            | 6.8%          | 3.2%          |

**Example:**
```python
# HACK.AX (HIGH risk, strong momentum = 0.82)
base_size = 8%
multiplier = 1.0  # Strong momentum
position_size = 8% × 1.0 = 8.0%
```

---

### Penalty Scaling for Growth

**CVaR penalties reduced for HIGH risk:**
```python
penalty_scale = {
    'LOW': 1.0,      # Full penalties
    'MEDIUM': 0.7,   # 70% of penalties
    'HIGH': 0.4      # 40% of penalties (accept risk)
}

# Example: CVaR < -50%
LOW:    25% penalty
MEDIUM: 17.5% penalty
HIGH:   10% penalty  ← Accepts extreme tail risk
```

**Liquidity penalties NOT scaled** (critical for all categories).

---

## 🧪 **2. Backtesting Engine**

### Location
`utilities/backtest_engine.py`

### Features

#### Single ETF Backtest
```python
engine = BacktestEngine(commission_pct=0.001, slippage_pct=0.0005)
result = engine.backtest_single_etf(
    prices, 
    ohlc_data, 
    risk_category='MEDIUM',
    rebalance_days=60,        # Rebalance every 2 months
    score_threshold=60.0       # Min score to hold position
)
```

#### Trading Logic
**ENTRY:**
- Score ≥ 60
- Kalman trend = +1 (bullish)
- Signal strength > 0.4

**EXIT:**
- Score < 60
- Kalman trend ≠ +1
- Signal strength ≤ 0.4

**Costs:**
- Commission: 0.10% per trade
- Slippage: 0.05% per trade
- **Total: 0.15% per trade**

#### Metrics Calculated
1. **Total Return**: Strategy P&L
2. **Benchmark Return**: Buy & Hold P&L
3. **Excess Return**: Strategy - Benchmark
4. **Sharpe Ratio**: Risk-adjusted return
5. **Max Drawdown**: Peak-to-trough decline
6. **Win Rate**: % of profitable trades
7. **Avg Win/Loss**: Average P&L of winners vs. losers

#### Portfolio Backtest
```python
results = engine.backtest_portfolio(
    etf_data={'VAS.AX': (prices, ohlc_data), ...},
    risk_classifications={'VAS.AX': 'LOW', ...},
    top_n=10,
    rebalance_days=60
)
```

---

### Running Backtests

**Quick Test (12 ETFs):**
```bash
cd /Users/uliana/Desktop/new_alpha/latest\ /modified
python3 run_backtest.py
```

**Custom Tickers:**
```python
# Edit run_backtest.py
run_quick_backtest(['VAS.AX', 'NDQ.AX', 'HACK.AX', 'CRYP.AX'])
```

**Output:**
- Console summary (returns, Sharpe, win rate)
- `data/backtest_results.parquet` (detailed results)

---

## 📊 **3. Dashboard Enhancements**

### New Pages

#### 🚀 **Growth Opportunities** (`growth`)
**Purpose:** Find top-scored MEDIUM/HIGH risk ETFs for growth strategy

**Features:**
1. **Summary Cards**
   - Total opportunities (score > 60)
   - Breakdown by risk category
   - Average momentum score

2. **Top 10 Table**
   - Ticker, Score, Risk Category
   - Momentum Score (signal + efficiency)
   - ML Forecast, Volume Signal
   - Action (BUY/WAIT based on Kalman trend)

3. **Score vs Momentum Scatter Plot**
   - Visualize score vs momentum strength
   - Identify high-score + high-momentum opportunities

4. **Position Sizing Guide**
   - Reference table for position sizes by risk + momentum

#### 📊 **Backtest Results** (`backtest`)
**Purpose:** View historical performance validation

**Features:**
1. **Summary Cards**
   - ETFs that beat Buy & Hold
   - Average excess return
   - Average Sharpe ratio
   - Average max drawdown

2. **Detailed Results Table**
   - Per-ETF strategy vs. benchmark returns
   - Sharpe, max drawdown, trades, win rate

**Note:** Requires `run_backtest.py` to be run first.

---

### Updated Navigation

```
📈 Summary              ← System overview
🚀 Growth Opportunities ← NEW: Top growth picks
📊 Backtest Results     ← NEW: Performance validation
🌍 Macro & Geo          ← Real-time macro/geo risk
🔍 ETF Explorer         ← Filter/search all ETFs
📊 ETF Details          ← Individual ETF deep dive
```

---

## 🔄 **4. Usage Workflow**

### Weekly Routine (Growth Strategy)

#### **Sunday Night:**
```bash
# 1. Run analysis (15-20 min for 385 ETFs)
python3 run_analysis.py

# 2. Start dashboard
python3 run_dashboard.py
# Open: http://127.0.0.1:8050
```

#### **In Dashboard:**
1. **Growth Opportunities Tab**
   - Review top 10-20 opportunities
   - Note: Score > 75 (strong), 65-75 (good), 60-65 (fair)

2. **For Each Candidate:**
   - Click ticker → ETF Details page
   - Check: **ALL components must agree**
     - ✅ Kalman Hull: BULLISH (+1)
     - ✅ Divergence: BULLISH or NONE
     - ✅ Volume Signal: ACCUMULATION
     - ✅ ML Forecast: Positive
     - ✅ ML Confidence: > 40/100
   - If all agree → BUY signal
   - If mixed → WAIT for confirmation

3. **Position Sizing:**
   - Use momentum score to determine size:
     - Strong (>75): Full position
     - Good (60-75): 85% position
     - Weak (<60): Reduce or wait

4. **Existing Positions:**
   - Check ETF Details pages
   - **EXIT if:**
     - Kalman Hull flips BEARISH
     - Divergence turns BEARISH
     - Volume shows DISTRIBUTION
     - Score drops < 50

---

### Monthly Routine

```bash
# Run backtest to validate system performance
python3 run_backtest.py

# Review in dashboard:
# - Backtest Results tab
# - Check: Excess return, Sharpe, win rate
# - Identify: Which ETFs/risk categories perform best
```

---

## 📈 **5. Performance Expectations**

### Growth Strategy (MEDIUM/HIGH Risk Focus)

**Target Metrics:**
- **Excess Return:** +5-10% vs. Buy & Hold (annual)
- **Sharpe Ratio:** 1.0-1.4 (risk-adjusted)
- **Max Drawdown:** -15% to -25% (accept volatility)
- **Win Rate:** 55-65% (directional accuracy)

**Risk Profile:**
- **Volatility:** 25-40% annualized (HIGH risk ETFs)
- **Position Sizing:** 3-12% per position
- **Portfolio:** 8-12 positions (diversification)

---

## ⚠️ **6. Risk Management**

### Hard Rules

**NEVER:**
1. Over-allocate (>12% MEDIUM, >8% HIGH per position)
2. Hold BEARISH trends (Kalman = -1)
3. Trade illiquid ETFs (avg volume < $500k/day)
4. Ignore DISTRIBUTION signals (smart money exit)

**ALWAYS:**
1. Use stop losses (-8 to -12% depending on risk)
2. Size down for weak momentum (<60 score)
3. Rebalance regularly (every 60 days)
4. Monitor CVaR (exit if < -50% for MEDIUM risk)

---

## 📝 **7. Files Modified/Created**

### New Files
- `analyzers/scoring_system_growth.py` (growth-optimized scoring)
- `utilities/backtest_engine.py` (backtesting framework)
- `dashboard/growth_components.py` (growth dashboard pages)
- `run_backtest.py` (backtest runner script)
- `docs/GROWTH_ENHANCEMENTS.md` (this file)

### Modified Files
- `dashboard/app.py` (added growth/backtest tabs)

### Original Files (Unchanged)
- `analyzers/scoring_system.py` (original conservative scoring)
- All other core modules remain unchanged

---

## 🔄 **8. Switching Between Strategies**

### Use Growth System:
The dashboard automatically uses the **growth system** if you're filtering/viewing MEDIUM/HIGH risk ETFs on the Growth Opportunities page.

### Use Original (Conservative) System:
The original scoring system is still available in `analyzers/scoring_system.py` if you want to compare or revert.

**Key Difference:**
- **Original:** Risk(40%), Technical(30%), ML+Volume(30%) — conservative
- **Growth:** Momentum(35%), Forecast(25%), Risk(25%), Volume(15%) — aggressive

---

## 📊 **9. Example Trade Decision**

### HACK.AX (Cybersecurity ETF, HIGH Risk)

**Growth System Output:**
```
Composite Score: 83.1/100
Risk Category: HIGH
Position Size: 6.8%  ← 85% of base 8% (good momentum)

Components:
- Momentum: 100.0/100  ✅ (signal 0.78 + efficiency 0.62)
- Forecast: 69.2/100   ✅ (+8.5% ML forecast, 65% confidence)
- Risk: 40.0/100       ⚠️  (High volatility, CVaR -28%)
- Volume: 76.2/100     ✅ (Accumulation signal)

Kalman Hull:
- Trend: BULLISH (+1)  ✅
- Divergence: BULLISH  ✅
- Signal Strength: 0.78  ✅

ML Forecast:
- 60-day: +8.5%  ✅
- Confidence: 65/100  ✅
- Hit Rate: 72%  ✅ (excellent)
```

**Decision:**
✅ **BUY** - All components agree, high momentum, strong forecast
📊 **Position:** 6.8% of portfolio
🛑 **Stop Loss:** -8% (HIGH risk = tighter stop)
🎯 **Target:** +15-20% (take profits)

---

## 🎯 **10. Key Takeaways**

1. **Growth System emphasizes momentum + forecast over risk aversion**
   - Boosted weights for MEDIUM/HIGH risk ETFs
   - Reduced penalties for volatility/CVaR

2. **Position sizing is now dynamic**
   - Based on momentum strength (40-100% of base)
   - Prevents overexposure to weak signals

3. **Backtesting validates strategy performance**
   - Real transaction costs included
   - Out-of-sample testing (walk-forward)
   - Sharpe/drawdown metrics for risk assessment

4. **Dashboard provides actionable insights**
   - Growth Opportunities: Top picks at a glance
   - Backtest Results: Historical validation
   - All components on one screen for quick decisions

5. **Risk management is built-in**
   - Automatic position sizing
   - Exit signals from divergence/distribution
   - Liquidity checks prevent illiquid trades

---

## 📞 **Support**

For questions or issues:
1. Check `README.md` for system overview
2. Review `docs/00_DOCUMENTATION_INDEX.md` for all docs
3. Run `python3 <script>.py` for any errors and investigate

---

**End of Growth Enhancements Documentation**

