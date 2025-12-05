# Architecture Update - Production System

## Current System Architecture (December 2025)

### Core Components

#### 1. Entry Points
- `run_analysis.py` - Main production analysis with 10 validated features
- `run_analysis_fast.py` - Batch processing optimized version  
- `run_dashboard.py` - Interactive dashboard

#### 2. Production ML Ensemble
- **File**: `analyzers/ml_ensemble_production.py`
- **Features**: 10 statistically validated indicators
- **Models**: RandomForest + Ridge ensemble
- **External Data**: VIX, Gold, Rates, Bonds for regime features

#### 3. System Orchestrator
- **File**: `system/orchestrator.py`
- **Integration**: Combines risk, ML, and technical analysis
- **Optimization**: Removed CSV exports, optimized name lookups

### Key Improvements Made

1. **Feature Optimization**: 40 → 10 validated features (75% reduction)
2. **Performance**: Removed slow Yahoo Finance API calls in summary
3. **Output**: CSV exports removed, Parquet only
4. **Validation**: Temporal out-of-sample testing implemented

### Data Flow

```
ETF Data → Risk Classification → ML Features (10) → Ensemble Forecast → Composite Score → Rankings
```

### Production Features

| Category | Features | Count |
|----------|----------|-------|
| Basic Technical | volatility, momentum | 2 |
| MACD-V | volatility_level, signal_quality, macd_histogram, macd_signal | 4 |
| Regime | gold_equity_corr, vix_rates_corr, cross_asset_dispersion, equity_bonds_corr | 4 |

### Performance Metrics

- **Processing Time**: ~15 minutes for 377 ETFs
- **Expected Improvement**: 20-30% over baseline
- **Risk Level**: LOW (statistically validated)

## File Structure

```
etf_lates/
├── run_analysis.py          # Main entry point
├── run_analysis_fast.py     # Optimized version
├── run_dashboard.py         # Dashboard
├── analyzers/
│   └── ml_ensemble_production.py  # Production ML model
├── system/
│   └── orchestrator.py      # Main controller
├── data_manager/            # Data handling
├── config/                  # Configuration
├── data/                    # Data storage
└── docs/                    # Documentation
```

## Deployment Status

✅ **Production Ready**
- 10 validated features implemented
- Batch processing optimized
- CSV exports removed
- Performance improvements applied
- Documentation updated
