# Strategy Comparator - Quick Start Guide

## Installation

Everything is ready to use. No additional setup needed.

```bash
# Verify backtester data exists
ls -l data/professional_backtest_results.parquet
# Should show: 28K file, created recently
```

## Most Common Use Cases

### 1. Analyze Your Best Trades

```bash
# See top 10 best performing signals
python scripts/compare_strategy.py --top 10
```

**Output:**
```
ATOM.AX      +39.37% vs +16.59% buy-hold   Alpha: +22.77% [YES OUTPERFORMS]
GAME.AX      +28.13% vs -4.57% buy-hold    Alpha: +32.70% [YES OUTPERFORMS]
ETPMAG.AX    +22.74% vs +4.76% buy-hold    Alpha: +17.98% [YES OUTPERFORMS]
...
```

**Insight:** Your signals perfectly timed entries on high-momentum stocks

---

### 2. Analyze Your Worst Trades

```bash
# See bottom 10 worst performing signals
python scripts/compare_strategy.py --bottom 10
```

**Output:**
```
HGEN.AX      -22.99% vs +31.36% buy-hold   Alpha: -54.35% [NO - UNDERPERFORMS]
BBOZ.AX      -22.37% vs -0.14% buy-hold    Alpha: -22.23% [NO - UNDERPERFORMS]
BBUS.AX      -16.99% vs -6.95% buy-hold    Alpha: -10.04% [NO - UNDERPERFORMS]
...
```

**Insight:** Signals triggered on false breakouts; avoid-entire-approach would have been better

---

### 3. Deep Dive on Specific ETF

```bash
# Detailed analysis of single ETF
python scripts/compare_strategy.py ATOM.AX
```

**Output includes:**
- Strategy performance metrics (trades, win rate, Sharpe)
- Buy-hold return over same period
- Two comparison approaches explaining the numbers
- Benchmark performance (ASX200, S&P500)
- Summary table

**Example:**
```
Strategy Return:                +39.37%
Buy-Hold Return:                +16.59%
Alpha (Strategy vs Buy-Hold):   +22.77%
Result: [YES] STRATEGY OUTPERFORMS

ASX200:                         +0.41%
Alpha vs ASX200:                +38.95%
Result: [YES] STRATEGY OUTPERFORMS ASX200

S&P500:                         +2.00%
Alpha vs S&P500:                +37.37%
Result: [YES] STRATEGY OUTPERFORMS S&P500
```

---

### 4. Compare Your Favorite ETFs

```bash
# Compare multiple ETFs side-by-side
python scripts/compare_strategy.py VAS.AX VGS.AX IOZ.AX
```

**Output:**
```
Ticker    Strategy Return  Buy-Hold Return  Alpha       Outperforms
VAS.AX              1.96%            0.51%     +1.45%        Yes
VGS.AX             -10.07%           3.68%    -13.74%        No
IOZ.AX              2.34%            1.12%     +1.22%        Yes
```

---

### 5. Batch Analysis (Export to Excel)

```bash
# Test all 313 ETFs, save to CSV for Excel analysis
python scripts/compare_strategy.py --all --output strategy_results.csv
```

**Output file:** `strategy_results.csv`

**Columns:**
- Ticker
- Strategy Return
- Buy-Hold Return
- Alpha
- Outperforms (Yes/No)
- Trades
- Win Rate
- Sharpe Ratio

**Excel pivot tables:**
- Count by Outperforms (Yes/No)
- Average return by category
- Win rate distribution
- Sharpe ratio analysis

---

### 6. Save Detailed Report

```bash
# Create formatted text report for sharing
python scripts/compare_strategy.py ATOM.AX --save-report atom_analysis.txt
```

**Output file:** `atom_analysis.txt`

Contains full analysis with formatting, suitable for email/sharing.

---

## Understanding the Results

### Green Light (Outperforms = Yes)

✓ Strategy beat buy-hold
✓ Strategy beat benchmarks
✓ Signal was well-timed

**Action:** These are your wins. Understand what triggered the signal.

---

### Red Light (Outperforms = No)

✗ Strategy underperformed buy-hold
✗ Strategy underperformed benchmarks
✗ Signal was poorly-timed

**Action:** These are your losses. Avoid similar setups in future.

---

## Two Ways to Interpret Results

### Approach A: Capital-Deployed

"Did my signal timing beat simple buy-hold?"

- Strategy: Return on $10 deployed when signal triggered
- Buy-Hold: Return on $10 held from day 1
- **Fair comparison of**: Signal entry quality

**Example:**
- Signal deployed $10 on ATOM.AX → +39.37%
- Buy-hold had $10 from day 1 → +16.59%
- Signal was better! (+22.77% alpha)

---

### Approach B: Full-Period

"How much capital was I actually using?"

- Strategy: Used only 3.33% of available capital per day (very selective)
- Buy-Hold: Used 100% of capital every day (fully invested)

**Fair comparison of**: Capital efficiency vs opportunity cost

**Example:**
- Strategy deployed total $10 across 30 days
- Buy-hold would have deployed $10 for entire 30 days
- Strategy is selective by design

---

## Key Metrics Quick Reference

| Metric | Interpretation |
|--------|-----------------|
| **Alpha > 0** | Strategy outperformed benchmark ✓ |
| **Alpha < 0** | Strategy underperformed ✗ |
| **Win Rate > 50%** | More trades won than lost ✓ |
| **Sharpe > 0.5** | Good risk-adjusted returns ✓ |
| **Max Drawdown < -10%** | Large losing streak ✗ |

---

## Decision Making

### For Portfolio Construction (Phase 3)

**Include in portfolio:**
- Outperforms = Yes
- Alpha > 5%
- Win Rate > 40%
- Sharpe > 0

**Avoid in portfolio:**
- Outperforms = No
- Alpha < -5%
- Win Rate < 20%

**Monitor carefully:**
- Close calls (Alpha between -2% and +2%)

---

### Example Phase 3 Portfolio

```
Top 20 performers from Phase 2 backtest
├─ Allocate capital proportional to backtest return
├─ Apply position sizing based on volatility
└─ Monitor performance vs this analysis

10 best performers: 40% of portfolio
10 middle performers: 40% of portfolio
Remaining 293 ETFs: 20% or exclude
```

---

## Files You'll Use

```
scripts/
├─ compare_strategy.py           ← Run this file
└─ run_professional_backtest.py  ← Already ran this

utilities/
└─ strategy_comparator.py        ← Engine (don't edit)

data/
├─ professional_backtest_results.parquet   ← Input data
└─ strategy_results.csv                     ← Your output (if exported)

STRATEGY_COMPARATOR_GUIDE.md  ← Full documentation
QUICK_START_COMPARISONS.md    ← This file
```

---

## Troubleshooting

**"Error: ETF not found"**
```bash
# Check available ETFs
python scripts/compare_strategy.py --all | head -20
```

**"No data for benchmark"**
- Network issue, try again
- Benchmark may be temporarily unavailable
- Continues with available data

**"Period only shows last month"**
- Normal! Comparator uses current data
- Backtest results cover 2017-2024
- Buy-hold calculated on available recent data

---

## Next Steps

1. **Identify Winners**: `--top 10`
2. **Understand Losers**: `--bottom 10`
3. **Export All**: `--all --output results.csv`
4. **Plan Portfolio**: Based on outperformance analysis
5. **Phase 3**: Portfolio construction with best performers

---

## Example Workflow

```bash
# 1. See what worked best
python scripts/compare_strategy.py --top 10

# 2. See what failed
python scripts/compare_strategy.py --bottom 10

# 3. Export all for analysis
python scripts/compare_strategy.py --all --output results.csv

# 4. Deep dive on top performer
python scripts/compare_strategy.py ATOM.AX

# 5. Compare your favorites
python scripts/compare_strategy.py VAS.AX VGS.AX IOZ.AX

# 6. Save detailed report
python scripts/compare_strategy.py ATOM.AX --save-report ATOM_report.txt
```

**Time: 5 minutes**
**Output: Complete understanding of strategy effectiveness**

---

## Questions?

See **STRATEGY_COMPARATOR_GUIDE.md** for:
- Full documentation
- Detailed metric explanations
- Advanced analysis ideas
- Interpretation examples
