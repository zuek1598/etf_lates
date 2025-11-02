# 🚀 ETF Analysis System - Quick Start

## 📊 Main Commands (Just 2!)

### **1. Run Analysis**
```bash
python3 run_analysis.py
```
**This does everything:**
- Downloads historical data automatically
- Analyzes all 385 ETFs
- **Optionally runs backtesting:**
  - **Option 1:** Quick test (11 ETFs, ~2 min)
  - **Option 2:** Full universe (all ETFs, ~30-60 min)
  - **Option 3:** Skip
- Saves results to `data/etf_universe.parquet`
- Takes 15-20 minutes (+ backtest time)
- **Start here** - this is all you need!

---

### **2. View Dashboard**
```bash
python3 run_dashboard.py
```
- Opens: http://127.0.0.1:8050
- 6 pages: Summary, Growth Opportunities, Backtest Results, Macro/Geo, Explorer, Details
- **Must run analysis first**

---

## 🎯 Complete Workflow

### **First Time Setup:**
```bash
# 1. Run analysis (it will ask about backtesting)
python3 run_analysis.py

# When prompted:
#   Option 1: Quick test → Fast validation (recommended for first run)
#   Option 2: Full universe → Comprehensive validation (for production)
#   Option 3: Skip → Just analysis, no backtest

# 2. View dashboard
python3 run_dashboard.py
# → Open: http://127.0.0.1:8050
```

### **Weekly Routine:**
```bash
# Every Sunday: Generate fresh analysis
python3 run_analysis.py
# → Choose "1" for quick backtest or "3" to skip

# View updated opportunities
python3 run_dashboard.py
```

---

## 📁 Folder Structure

```
modified/
├── run_analysis.py          ← Run this! (does everything)
├── run_dashboard.py         ← View results
├── QUICK_START.md           ← This file
├── README.md                ← Full documentation
│
├── analyzers/               ← Core analysis components
├── indicators/              ← Kalman Hull momentum
├── dashboard/               ← Dash app
├── system/                  ← Orchestrator, scoring
├── data/                    ← Results & historical data
├── utilities/               ← Backtesting engine
└── docs/                    ← Documentation
```

---

## 💡 What Happens When You Run Analysis?

```
python3 run_analysis.py

Step 1: Downloads historical data (1 year by default)
  ├─ 385 ETFs from Yahoo Finance
  ├─ Saves to data/historical/*.parquet
  └─ Takes ~5-10 minutes

Step 2: Analyzes all ETFs
  ├─ Risk metrics (CVaR, Ulcer, Beta, IR)
  ├─ ML forecasts with validation
  ├─ Kalman Hull momentum
  ├─ Volume intelligence
  └─ Takes ~10-15 minutes

Step 3: Asks about backtesting (YOU CHOOSE)
  ├─ Option 1: Quick test (11 ETFs, 1-2 min)
  ├─ Option 2: Full universe (all ETFs, 30-60 min) ← Production validation
  ├─ Option 3: Skip
  └─ Saves to data/backtest_results.parquet

Step 4: Done!
  └─ Ready for dashboard
```

---

## 🆘 Troubleshooting

**Dashboard shows no data:**
```bash
# Run analysis first
python3 run_analysis.py
```

**Dashboard won't start:**
```bash
# Kill existing dashboard
pkill -f run_dashboard
# Then restart
python3 run_dashboard.py
```

**Want to re-run just the backtest:**
```bash
# Just re-run analysis and choose backtest option
python3 run_analysis.py
```

---

## 📖 Documentation

- **QUICK_START.md** - This file (quick reference)
- **README.md** - Complete system documentation
- **docs/GROWTH_ENHANCEMENTS.md** - Growth strategy details

---

**Version:** 3.1 (Growth-Optimized)  
**Last Updated:** October 2025

---

**That's it! Just run `python3 run_analysis.py` and you're ready to trade.** 🚀
