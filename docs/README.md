# 📚 Documentation Index

**Version:** 2.0 (Production)  
**Last Updated:** October 25, 2025  
**Status:** ✅ Production Ready

---

## 🚀 Start Here

**New users:**
1. **[QUICK_START.md](QUICK_START.md)** - Get running in 5 minutes
2. **[guides/USER_GUIDE.md](guides/USER_GUIDE.md)** - Complete usage guide

**System operators:**
- **[READY_TO_RUN.md](READY_TO_RUN.md)** - Pre-flight checklist
- **[guides/BACKTEST_GUIDE.md](guides/BACKTEST_GUIDE.md)** - Backtesting workflow

**Developers:**
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design
- **[guides/DEVELOPMENT_GUIDE.md](guides/DEVELOPMENT_GUIDE.md)** - Development guide

---

## 📖 Documentation Structure

### **Core Documentation** (This Directory)

| File | Purpose |
|------|---------|
| **QUICK_START.md** | Fast start guide (5 min) |
| **READY_TO_RUN.md** | System readiness checklist |
| **CLEANUP_SUMMARY.md** | Recent cleanup record |
| **ARCHITECTURE.md** | System architecture & design |
| **SPECIFICATION.md** | Technical specifications |
| **SYSTEM_OVERVIEW.md** | High-level system overview |

### **User Guides** (`guides/`)

| File | Purpose |
|------|---------|
| **USER_GUIDE.md** | Complete user manual |
| **BACKTEST_GUIDE.md** | Backtesting workflow |
| **DEVELOPMENT_GUIDE.md** | Developer setup & practices |
| **TESTING_GUIDE.md** | Testing procedures |

### **Technical Reference** (`reference/`)

| File | Purpose |
|------|---------|
| **DASHBOARD_FEATURES.md** | Dashboard capabilities |
| **GROWTH_ENHANCEMENTS.md** | Growth optimization features |
| **MACRO_GEO_CACHING.md** | Caching implementation |
| **VALIDATION_IMPLEMENTATION.md** | Walk-forward validation |
| **RETURN_CALCULATIONS_ANALYSIS.md** | Return calculation methods |
| **1Y_RETURN_METHODOLOGY.md** | 1-year return methodology |

### **Historical Archive** (`archive/`)

- Bug diagnosis documents
- Code review history
- Changelog records
- Deprecated documentation
- Old README versions

---

## 🎯 Quick Navigation

### **I want to...**

**Use the system:**
- Run analysis → [QUICK_START.md](QUICK_START.md)
- View dashboard → [guides/USER_GUIDE.md](guides/USER_GUIDE.md) → Dashboard section
- Run backtests → [guides/BACKTEST_GUIDE.md](guides/BACKTEST_GUIDE.md)

**Understand the system:**
- High-level overview → [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
- Technical details → [SPECIFICATION.md](SPECIFICATION.md)
- Architecture → [ARCHITECTURE.md](ARCHITECTURE.md)

**Modify the system:**
- Development setup → [guides/DEVELOPMENT_GUIDE.md](guides/DEVELOPMENT_GUIDE.md)
- Test changes → [guides/TESTING_GUIDE.md](guides/TESTING_GUIDE.md)
- Understand features → [reference/](reference/)

**Troubleshoot:**
- Pre-flight check → [READY_TO_RUN.md](READY_TO_RUN.md)
- Testing guide → [guides/TESTING_GUIDE.md](guides/TESTING_GUIDE.md)
- Historical fixes → [archive/](archive/)

---

## 📊 System Summary

### **What it does:**
Analyzes 385 Australian ETFs using:
- Risk metrics (CVaR, Ulcer Index, Beta)
- Technical indicators (Kalman Hull Supertrend)
- ML forecasting (Random Forest + Ridge Regression)
- Volume intelligence
- Macro/geopolitical risk assessment

### **Key Features:**
- ✅ Interactive Dashboard (6 pages)
- ✅ Growth Opportunities Ranking
- ✅ Full Universe Backtesting (363 ETFs)
- ✅ Real-time Macro/Geo Analysis
- ✅ Risk-adjusted Scoring

### **Usage:**
```bash
# Run full analysis
python3 run_analysis.py

# Launch dashboard
python3 run_dashboard.py

# Update historical data
python3 scripts/download_all_etf_data.py

# Run backtest
python3 scripts/run_backtest.py
```

---

## 📁 Complete File Tree

```
docs/
├── README.md                           # ← You are here
├── QUICK_START.md                      # Fast start guide
├── READY_TO_RUN.md                     # Pre-flight checklist
├── CLEANUP_SUMMARY.md                  # Recent cleanup
├── ARCHITECTURE.md                     # System design
├── SPECIFICATION.md                    # Technical specs
├── SYSTEM_OVERVIEW.md                  # Overview
│
├── guides/                             # User guides
│   ├── USER_GUIDE.md                  # Complete manual
│   ├── BACKTEST_GUIDE.md              # Backtesting
│   ├── DEVELOPMENT_GUIDE.md           # Development
│   └── TESTING_GUIDE.md               # Testing
│
├── reference/                          # Technical docs
│   ├── DASHBOARD_FEATURES.md
│   ├── GROWTH_ENHANCEMENTS.md
│   ├── MACRO_GEO_CACHING.md
│   ├── VALIDATION_IMPLEMENTATION.md
│   ├── RETURN_CALCULATIONS_ANALYSIS.md
│   └── 1Y_RETURN_METHODOLOGY.md
│
└── archive/                            # Historical
    ├── BACKTEST_DIAGNOSIS.md
    ├── BUG_FIXED_MULTI_LEVEL_COLUMNS.md
    ├── BACKTEST_FIX.md
    ├── CHANGELOG.md
    ├── CODE_REVIEW_HISTORY.md
    ├── ORGANIZATION_COMPLETE.md
    ├── SYSTEM_READY.md
    └── [3 more old docs]
```

---

## 🔄 Version History

**v2.0 (October 25, 2025):**
- Cleaned & organized workspace
- All docs in `docs/` directory
- Updated main README
- Scripts moved to `scripts/`
- Complete documentation index

**v3.1 (October 25, 2025):**
- Growth-optimized scoring
- Integrated backtesting
- Full universe capability
- Dashboard enhancements

**v3.0 (October 23, 2025):**
- Macro/Geo caching
- Bug fixes
- Documentation reorganization

---

**Need help?** Start with [QUICK_START.md](QUICK_START.md) or check the [guides/](guides/) folder.
