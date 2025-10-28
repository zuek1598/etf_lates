# 🚀 Complete Backtest Workflow: MAX History + Full Universe

## Step-by-Step Guide

### ⏰ **Total Time: 45-90 minutes**
- Download data: 15-30 min
- Run backtest: 30-60 min

---

## 📥 **Step 1: Download MAX Historical Data**

```bash
cd "/Users/uliana/Desktop/new_alpha/latest /modified"
python3 download_max_history.py
```

**What this does:**
- Downloads ALL available historical data for 385 ETFs
- Uses `period='max'` (typically 5-20 years depending on ETF)
- Saves to `data/historical/*.parquet`
- Takes ~15-30 minutes

**Expected output:**
```
✅ Successful: 350-370 ETFs
❌ Failed: 10-30 ETFs (new ETFs, delisted, etc.)
```

---

## 🧪 **Step 2: Run Full Universe Backtest**

```bash
python3 run_backtest_all.py
```

**What this does:**
- Backtests ALL ETFs with historical data
- Uses 252-day training, 60-day testing
- Multiple walk-forward windows
- Saves results to `data/backtest_results.parquet`
- Takes ~30-60 minutes

---

## 📊 **Step 3: View Results in Dashboard**

```bash
python3 run_dashboard.py
# Open: http://127.0.0.1:8050
# Click: 📊 Backtest Results tab
```

---

## 🎯 **What to Expect**

### **With MAX History (2-20 years):**

**Good Results:**
- More trading opportunities (20-50 trades per ETF)
- Multiple market cycles (bull + bear)
- Meaningful Sharpe ratios
- Strategy will likely beat buy-and-hold on HIGH risk ETFs (volatility trading)
- Strategy may underperform on LOW risk ETFs (trend-following in steady uptrends)

**Expected Performance:**
- **Excess Return:** -5% to +10% vs buy-and-hold (depends on ETF category)
- **Sharpe Ratio:** 0.5 to 1.5 (risk-adjusted returns)
- **Win Rate:** 55-65% (directional accuracy)
- **Max Drawdown:** -10% to -20% (volatility management)

**Why some ETFs will underperform:**
- Strong trending markets favor buy-and-hold
- The system exits on bearish signals = misses some upside
- But reduces drawdowns significantly

---

## 💡 **Tips**

### **While Downloading (Step 1):**
- Don't interrupt (Ctrl+C will stop)
- Progress updates every 50 ETFs
- Failed tickers are normal (10-30 expected)

### **While Backtesting (Step 2):**
- Takes 30-60 minutes for 350+ ETFs
- You'll see progress for each ETF
- Can interrupt and restart if needed

### **If Download Fails:**
```bash
# Just re-run, it will skip existing files
python3 download_max_history.py
```

---

## 📈 **Expected Timeline**

```
Now                → +20 min  → +60 min  → +65 min
│                    │           │           │
Download starts   Download    Backtest    View results
                  complete    complete    in dashboard
```

---

## 🎯 **Quick Commands**

```bash
# Complete workflow (run in sequence):
cd "/Users/uliana/Desktop/new_alpha/latest /modified"

# 1. Download MAX data (~20 min)
python3 download_max_history.py

# 2. Run full backtest (~45 min)
python3 run_backtest_all.py

# 3. View in dashboard
python3 run_dashboard.py
```

---

**Ready to start! Run the first command to download MAX historical data.** 📥

