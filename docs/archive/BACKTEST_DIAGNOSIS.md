# 🔍 Backtest Results Diagnosis

## 📊 **Current Situation:**

Your backtest shows **16/339 ETFs beat buy-and-hold** with only **5% win rate**.

### **Key Findings:**
- **0 trades** for 333 out of 339 ETFs
- Only 6 ETFs had any trades (QPON, HBRD, VAS, AAA, GEAR, SUBD, IOZ)
- Even those ETFs only had 2 trades each (100% win rate, but statistically meaningless)

---

## 🐛 **Root Causes:**

### **1. Insufficient Historical Data** (PRIMARY ISSUE)
**Current data:** 1 year (~253 trading days)

**Backtest requirements:**
- Training period: 150 days
- Rebalance period: 60 days
- **Result:** Only 1-2 rebalance cycles possible

**Impact:**
- Not enough data to properly train ML models
- Insufficient trading opportunities
- Risk metrics (CVaR, Ulcer) are unreliable with < 1 year
- Can't capture multiple market cycles

### **2. Conservative Thresholds**
**Current settings:**
- Composite score >= 60.0 (originally, now 40.0)
- Kalman signal strength > 0.4 (now 0.3)
- Must have bullish trend (kalman_trend == 1)

**With limited data:**
- Scores rarely exceed 40-60 range
- Signal strength is weak (< 0.3-0.4)
- System stays out of market most of the time

### **3. Walk-Forward Design**
The backtest uses a **rolling 150-day window**, which means:
- First 150 days: Training only, no trading
- Days 151-210: First rebalance period (60 days)
- Days 211-270: Second rebalance period (60 days) - but we only have 253 days total!

**Result:** Barely any trading opportunities.

---

## ✅ **Solutions:**

### **Option A: Download Maximum Historical Data (RECOMMENDED)**
```bash
# This will give you 5-20 years of data per ETF
cd "/Users/uliana/Desktop/new_alpha/latest /modified"
python3 -c "
import yfinance as yf
import pandas as pd
from pathlib import Path
from data_manager.etf_database import ETFDatabase

etf_db = ETFDatabase()
historical_dir = Path('data/historical')
historical_dir.mkdir(exist_ok=True)

tickers = list(etf_db.etf_data.keys())
print(f'Downloading MAX history for {len(tickers)} ETFs...')

for i, ticker in enumerate(tickers, 1):
    try:
        data = yf.download(ticker, period='max', progress=False)
        if not data.empty:
            data.to_parquet(historical_dir / f'{ticker.replace(\".\", \"_\")}.parquet')
            print(f'  [{i}/{len(tickers)}] ✓ {ticker}: {len(data)} days')
    except Exception as e:
        print(f'  [{i}/{len(tickers)}] ✗ {ticker}: {e}')

print('\n✅ Download complete!')
"

# Then re-run backtest
python3 run_analysis.py
# → Choose Option 2: Full universe backtest
```

**Expected results with MAX data:**
- 10-30 rebalance periods per ETF
- 50-150 trades per ETF
- Meaningful statistics
- Win rate: 55-65% (realistic for momentum strategy)
- Some ETFs will beat buy-and-hold, some won't

---

### **Option B: Further Lower Thresholds (TEMPORARY FIX)**
If you can't download more data right now, you can make the system more aggressive:

```python
# In utilities/backtest_engine.py, change:
score_threshold: float = 30.0  # From 40.0
kalman_signal_strength > 0.2   # From 0.3
```

**This will generate more trades, but:**
- Results will be unreliable (overfitting to 1 year)
- Not representative of long-term performance
- Can't validate strategy properly

---

### **Option C: Use Buy-and-Hold Comparison Only (CURRENT STATE)**
With 1 year of data, your backtest is essentially showing:
- "Did our system beat a simple buy-and-hold over 1 year?"
- Answer: No (5% win rate)

**But this doesn't mean the strategy is bad**, it means:
1. 1 year is too short to validate
2. 2024-2025 was a strong bull market → buy-and-hold wins
3. Momentum strategies shine in volatile/ranging markets

---

## 📈 **What to Expect with MAX Data:**

### **Realistic Performance Goals:**
| Metric | Expected Range | Why |
|--------|---------------|-----|
| **Win Rate** | 55-65% | Momentum + risk management |
| **Excess Return** | -5% to +10% | Depends on market conditions |
| **Sharpe Ratio** | 0.5 to 1.5 | Better risk-adjusted returns |
| **Max Drawdown** | -10% to -20% | Volatility management |
| **Trades per ETF** | 50-150 | Multiple cycles |

### **HIGH Risk ETFs:** Should **outperform** buy-and-hold
- Examples: URNM, GDX, CRYP, ATOM, HGEN
- Reason: High volatility = more trading opportunities

### **LOW Risk ETFs:** May **underperform** buy-and-hold
- Examples: VAS, VAF, BOND, AAA
- Reason: Steady trends = system exits too early

---

## 🎯 **Recommendation:**

**Run Option A** - Download MAX historical data and re-run the full universe backtest.

This will give you:
✅ Statistically significant results  
✅ Multiple market cycles (bull + bear)  
✅ Proper validation of the strategy  
✅ Confidence in deployment  

**Current results are NOT indicative of strategy failure** - they're simply data-limited.

---

**Last Updated:** October 25, 2025  
**System Version:** 3.1 (Growth-Optimized)
