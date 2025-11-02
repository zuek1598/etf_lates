# Strategy Comparator Guide

## Overview

The Strategy Comparator module evaluates your backtested trading strategy against buy-and-hold and benchmark performance. It provides two different comparison approaches to understand strategy effectiveness from different angles.

## What It Does

For any selected ETF, the comparator shows:

1. **Strategy Performance** - Return on capital deployed per buy signal
2. **Buy-and-Hold Return** - If you just bought and held the ETF for the period
3. **Benchmark Returns** - ASX200 (^AXJO) and S&P500 (^GSPC) performance
4. **Alpha Metrics** - Outperformance vs benchmarks and buy-hold
5. **Two Comparison Approaches** - Different perspectives on strategy effectiveness

## Installation

No additional installation needed. Uses existing modules:
- `utilities/strategy_comparator.py` - Core comparison engine
- `scripts/compare_strategy.py` - Command-line interface

## Usage

### Single ETF Comparison

```bash
python scripts/compare_strategy.py VAS.AX
python scripts/compare_strategy.py ATOM.AX
```

Output includes:
- Strategy metrics (return, trades, win rate, Sharpe)
- Buy-hold return
- Both comparison approaches (Capital-Deployed and Full-Period)
- Benchmark comparisons
- Summary table

### Save Detailed Report

```bash
python scripts/compare_strategy.py ATOM.AX --save-report atom_report.txt
```

### Compare Top/Bottom Performers

```bash
# Top 10 best performers from backtest
python scripts/compare_strategy.py --top 10

# Bottom 10 worst performers
python scripts/compare_strategy.py --bottom 10
```

### Compare Multiple ETFs

```bash
python scripts/compare_strategy.py VAS.AX VGS.AX IOZ.AX
```

Generates comparison table for all three.

### Compare All ETFs (Large Dataset)

```bash
python scripts/compare_strategy.py --all --output results.csv
```

Creates CSV file with all 313 backtested ETFs.

## Understanding the Output

### Example Output

```
==========================================================================================
STRATEGY COMPARISON: ATOM.AX
==========================================================================================
Period: 2025-09-30 to 2025-10-31 (30 days)

STRATEGY PERFORMANCE
------------------------------------------------------------------------------------------
  Return (Deployed Capital):      +39.37%
  Trades Executed:                     1
  Win Rate:                        1.0%
  Sharpe Ratio:                    0.00
  Avg Hold Period:                30.0 days
  Total Capital Deployed:         $10.00
```

**Interpretation:**
- Strategy deployed $10 (1 trade × $10) when signal triggered
- That $10 returned +39.37%
- Held for 30 days average
- Win rate of 1.0% (1 out of 1 trades won)

### APPROACH A: Capital-Deployed Comparison

```
APPROACH A: CAPITAL-DEPLOYED COMPARISON
------------------------------------------------------------------------------------------
  Strategy Return:                +39.37%
  Buy-Hold Return:                +16.59%
  Alpha (Strategy vs Buy-Hold):   +22.77%
  Result: [YES] STRATEGY OUTPERFORMS
```

**What it measures:**
- Compares return on capital deployed per signal vs buy-hold
- Strategy deployed $10 when signal triggered
- Buy-hold had $10 invested from day 1
- Strategy's +39.37% > Buy-hold's +16.59%
- Alpha of +22.77% (excess return)

**When to use this:**
- Understanding if your signals identify good entry points
- Comparing signal effectiveness in isolation
- "Did my signal timing beat simple buy-hold?"

**Important caveat:**
- Strategy capital sits idle when no signals
- Buy-hold is fully invested 100% of the time
- Not directly comparable due to different capital utilization

### APPROACH B: Full-Period Comparison

```
APPROACH B: FULL-PERIOD COMPARISON
------------------------------------------------------------------------------------------
  Strategy Capital Utilization:    3.33%
  (Average $0.333/day deployed)
  Buy-Hold Capital:              100.00%

  Interpretation: Strategy deployed $10 total across 30 days
  Buy-hold had $10.00 deployed for entire period (100% utilization).
```

**What it measures:**
- How much capital strategy deployed vs could have deployed
- 3.33% utilization = only using 3.33% of available capital per day
- Buy-hold uses 100% of capital continuously

**Interpretation:**
- Strategy is selective: only deploys capital when signals trigger
- 96.67% of time, capital is idle earning nothing
- This is by design (conservative, signal-based approach)

**When to use this:**
- Understanding opportunity cost
- "If I had allocated more capital when signals trigger, what would happen?"
- "Am I leaving money on the table?"

### Benchmark Comparisons

```
BENCHMARK COMPARISONS
------------------------------------------------------------------------------------------
  ASX200 (^AXJO):                 +0.41%
  Alpha vs ASX200:                +38.95%
  Result: [YES] STRATEGY OUTPERFORMS ASX200

  S&P500 (^GSPC):                 +2.00%
  Alpha vs S&P500:                +37.37%
  Result: [YES] STRATEGY OUTPERFORMS S&P500
```

**Interpretation:**
- Strategy +39.37% vs ASX200 +0.41% = +38.95% alpha
- Strategy +39.37% vs S&P500 +2.00% = +37.37% alpha
- Strategy significantly outperforms both benchmarks

## Key Metrics Explained

| Metric | Meaning |
|--------|---------|
| **Strategy Return** | Total return on capital deployed per signal (sum of all signal profits) |
| **Buy-Hold Return** | Return if you bought at start and held until end of period |
| **Alpha** | Excess return (Strategy - Benchmark) |
| **Win Rate** | % of trades that were profitable |
| **Sharpe Ratio** | Risk-adjusted returns (higher = better risk/reward) |
| **Capital Utilization** | % of available capital deployed per day |

## Interpretation Guide

### Top Performer Example (ATOM.AX: +39.37%)

**Story:** Signal perfectly timed entry into ATOM.AX which rallied +39.37%.
- Strategy: +39.37% (1 signal × $10)
- Buy-Hold: +16.59% (same period)
- Alpha: +22.77% (strategy beat buy-hold by this amount)
- **Insight:** Signal identified strong trending stock better than simple buy-hold

### Bottom Performer Example (HGEN.AX: -22.99%)

**Story:** Signal triggered on HGEN.AX which then declined sharply.
- Strategy: -22.99% (1 signal × $10 lost)
- Buy-Hold: +31.36% (buy-hold went UP!)
- Alpha: -54.35% (strategy massively underperformed)
- **Insight:** Signal was terrible; avoided entry entirely would have been better

## Strategy Insights from Comparisons

### When Strategy Outperforms (+Alpha)

- ✓ Signals identify good entry points (timing matters)
- ✓ Signals avoid big downturns
- ✓ Capital deployed efficiently on winning trades

### When Strategy Underperforms (-Alpha)

- ✗ Signals trigger on false breakouts
- ✗ Signals are late to identify trends
- ✗ Capital deployed on losing positions

### Capital Utilization Insights

**High Utilization (>50%):**
- Strategy generates many signals
- More aggressive/frequent trading
- Higher opportunity for gains, higher risk

**Low Utilization (<20%):**
- Strategy generates few signals
- Very selective/conservative
- Waits for high-confidence setups
- Leaves capital idle but avoids noise

## CSV Output

When using `--all --output results.csv`:

```csv
Ticker,Strategy Return,Buy-Hold Return,Alpha,Outperforms,Trades,Win Rate,Sharpe
ATOM.AX,39.37%,16.59%,+22.77%,Yes,1,1.0%,0.00
VAS.AX,1.96%,0.51%,+1.45%,Yes,10,0.8%,2.59
HGEN.AX,-22.99%,31.36%,-54.35%,No,1,0.0%,0.00
...
```

Can be imported into Excel for further analysis.

## Advanced Analysis Ideas

1. **Filter by Alpha**: Find ETFs where strategy significantly beats buy-hold
   ```
   Strategy Return > 10% AND Alpha > 5%
   ```

2. **Compare by Category**: Group by ETF type (growth, bonds, etc.)

3. **Win Rate Analysis**: Find ETFs where signal quality is highest
   ```
   Win Rate > 50%
   ```

4. **Risk Adjustment**: Use Sharpe ratio for risk-adjusted comparisons

## Limitations

1. **Recent Data Only**: Comparator uses recent price data; backtests cover longer periods
   - Backtest period: 2017-2024
   - Buy-hold calculated from available data (usually last few months)
   - Dates won't match historical backtest period

2. **No Transaction Costs**: Comparator assumes commission-free trading (matches backtest)

3. **No Slippage**: Assumes exact prices; real execution may vary

4. **Benchmark Selection**: Uses ASX200 and S&P500; may not match your actual opportunity set

## Troubleshooting

### "Error: ETF not found in backtest results"
- Check spelling (case-insensitive, with or without .AX suffix)
- Ensure backtest was run on that ETF
- Run: `python scripts/compare_strategy.py --all` to see all available

### "Insufficient data for [ticker]"
- ETF may be too new (immature)
- Try with `--top 10` for well-covered ETFs

### Character encoding errors
- Ensure terminal/console supports UTF-8
- Output format is ASCII-compatible

## Next Steps

1. **Identify Best Opportunities**: `python scripts/compare_strategy.py --top 20`
2. **Understand Losses**: `python scripts/compare_strategy.py --bottom 10`
3. **Batch Analysis**: `python scripts/compare_strategy.py --all --output analysis.csv`
4. **Deep Dive**: `python scripts/compare_strategy.py TICKER --save-report report.txt`
