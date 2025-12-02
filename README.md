# ETF Analysis System - Validated Factor Model

A comprehensive ETF analysis system using **4 statistically validated factors** for ranking.

---

## Validated Factors (p < 0.05)

The ranking system uses only factors that passed rigorous statistical validation:

| Factor | IC | p-value | Hit Rate | Description |
|--------|-----|---------|----------|-------------|
| **ml_forecast** | +0.229 | 0.027 | 61.7% | ML Ensemble forecast |
| **hit_rate** | +0.344 | 0.001 | 65.1% | Directional accuracy |
| **kalman_signal_strength** | +0.234 | 0.023 | 63.8% | Momentum strength |
| **cvar** | +0.261 | 0.011 | 61.7% | Conditional VaR (risk) |

**Validation:** Cross-sectional testing with 100 ETFs, 20-day forward returns.

---

## Quick Start

### 1. Run Analysis
```bash
python run_analysis.py
```

### 2. Launch Dashboard
```bash
python run_dashboard.py
```
Open http://127.0.0.1:8050 in your browser.

## Dashboard Features

**6 Interactive Pages:**
1. **Summary** - Overview & Top ETFs by validated factors
2. **Growth Opportunities** - High-potential ETFs
3. **Backtest Results** - Strategy validation
4. **Macro & Geo** - Market context analysis
5. **ETF Explorer** - Search & filter ETFs
6. **ETF Details** - Individual ETF deep-dive

## 📁 Project Structure

```
modified/
├── run_analysis.py          # Main analysis entry point
├── run_dashboard.py         # Dashboard launcher
├── README.md                # This file
│
├── analyzers/              # Analysis components
│   ├── etf_risk_classifier.py
│   ├── ml_ensemble.py
│   ├── risk_component.py
│   ├── scoring_system.py
│   ├── scoring_system_growth.py
│   └── volume_intelligence.py
│
├── dashboard/              # Dashboard application
│   ├── app.py             # Main Dash app
│   ├── data_loader.py     # Data loading utilities
│   └── growth_components.py
│
├── data/                   # Data storage
│   ├── historical/        # ETF price data (377 files)
│   ├── backtest_results.parquet
│   ├── etf_universe.parquet
│   ├── analysis_metadata.parquet
│   └── rankings_*.parquet
│
├── data_manager/          # Data management
│   ├── data_manager.py
│   └── etf_database.py
│
├── frameworks/            # Analysis frameworks
│   ├── integrated_framework.py
│   ├── macro_framework.py
│   └── geopolitical_framework.py
│
├── indicators/            # Technical indicators
│   └── kalman_hull.py
│
├── system/               # Core system
│   ├── config.py
│   ├── orchestrator.py
│   ├── schemas.py
│   ├── requirements.txt
│   └── run_analysis.py
│
├── utilities/            # Helper utilities
│   ├── backtest_engine.py
│   ├── shared_utils.py
│   ├── validators.py
│   └── walk_forward_validator.py
│
├── scripts/              # Utility scripts
│   ├── download_all_etf_data.py
│   └── run_backtest.py
│
└── docs/                 # Documentation
    ├── QUICK_START.md
    ├── READY_TO_RUN.md
    ├── CLEANUP_SUMMARY.md
    ├── README.md
    ├── ARCHITECTURE.md
    ├── SPECIFICATION.md
    ├── SYSTEM_OVERVIEW.md
    ├── guides/
    ├── reference/
    └── archive/
```

## 🔧 Dependencies

Install requirements:
```bash
pip install -r system/requirements.txt
```

Key dependencies:
- pandas, numpy - Data processing
- dash, plotly - Dashboard & visualization
- yfinance - Market data
- scikit-learn - ML models
- pyarrow - Parquet file handling

## 📈 Data

**Historical Data:**
- 377 Australian ETFs
- OHLCV (Open, High, Low, Close, Volume)
- Stored in Parquet format for fast access

**Why Local Storage?**
- Avoids API rate limits
- Faster analysis (no network delays)
- Consistent historical data
- Can be updated periodically

**Update Historical Data:**
```bash
python3 scripts/download_all_etf_data.py
```

## 🧪 Run Backtest

**Full Universe Backtest:**
```bash
python3 scripts/run_backtest.py
```

Results saved to `data/backtest_results.parquet` (363 ETFs).

## 📚 Documentation

- **Quick Start**: `docs/QUICK_START.md`
- **System Ready**: `docs/READY_TO_RUN.md`
- **Cleanup Summary**: `docs/CLEANUP_SUMMARY.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **User Guide**: `docs/guides/USER_GUIDE.md`
- **Backtest Guide**: `docs/guides/BACKTEST_GUIDE.md`
- **Development**: `docs/guides/DEVELOPMENT_GUIDE.md`

## Current Status

**Fully Operational with Validated Factor Model**
- 4 statistically validated factors (p < 0.05)
- 377 ETFs with historical data
- Cross-sectional percentile ranking
- Dashboard with 6 interactive pages

### Factor Validation Summary
- **Tested:** 12 factors across ML, Kalman, Volume, Risk categories
- **Validated:** 4 factors with statistical significance
- **Rejected:** 8 factors (negative IC or p > 0.05)
- **Sample:** 100 ETFs, 20-day forward returns

## 🔍 Troubleshooting

**Dashboard not showing backtest results:**
1. Ensure you're in the `modified/` directory
2. Hard refresh browser (Cmd+Shift+R / Ctrl+Shift+R)
3. Click the "📊 Backtest Results" tab (3rd tab)
4. Restart dashboard if needed

**Missing historical data:**
```bash
python3 scripts/download_all_etf_data.py
```

**Python errors:**
- Check you're using Python 3.8+
- Reinstall requirements: `pip install -r system/requirements.txt`

## 📝 Notes

- System tested with Python 3.13
- Dashboard runs on http://127.0.0.1:8050
- Backtest uses MACD-V strategy
- Results compare strategy vs buy-and-hold

---

**Version:** 3.0 (November 2025)  
**Status:** Production Ready - Validated Factor Model  
**Validation Date:** 2025-11-29
