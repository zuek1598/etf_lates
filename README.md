# ETF Analysis System - Advanced ML & Regime Detection

A sophisticated ETF analysis system featuring **regime-aware ML models**, **advanced validation**, and **risk management** capabilities.

---

## 🎯 System Overview

### Core Features
- **Regime Detection**: 5-year historical analysis with cross-asset correlations
- **ML Ensemble**: RandomForest + Ridge with regime-aware features
- **Advanced Validation**: Nested cross-validation + expanding windows + bootstrap CI
- **Risk Management**: Confidence flagging (High/Medium/Low) with stability assessment
- **Interactive Dashboard**: Real-time visualization and analysis

### Phases Completed ✅
- **Phase 1**: Data Pipeline & Regime Framework
- **Phase 2**: ML Model Improvements with Enhanced Features
- **Phase 3**: Advanced Validation & Risk Management

---

## 🚀 Quick Start

### 1. Run Analysis
```bash
python run_analysis.py
```

### 2. Launch Dashboard
```bash
python run_dashboard.py
```
Open http://127.0.0.1:8050 in your browser.

---

## 📊 System Architecture

### Data Pipeline
- **External Data**: VIX, AUD/USD, AU/US 10Y yields, Gold (5-year history)
- **Cross-Asset Correlations**: 5 key pairs for regime refinement
- **Regime Windows**: 252-day rolling with 63-day correlation windows

### ML Components
- **Feature Engineering**: Regime indicators, relative strength, liquidity metrics
- **Ensemble Models**: RandomForest + Ridge with robust scaling
- **Target Processing**: ±15% clipping to prevent overfitting

### Validation Framework
- **Nested Cross-Validation**: 3 splits with time series awareness
- **Expanding Window**: Progressive validation for temporal stability
- **Bootstrap CI**: 1000 samples for confidence intervals
- **Performance Metrics**: MAE, Hit Rate, Stability scores

---

## 📁 Project Structure

```
etf_lates_optimisation/
├── README.md                # This file
├── ACTION_PLAN.md           # Current status and next steps
├── run_analysis.py          # Main analysis entry point
├── run_dashboard.py         # Dashboard launcher
│
├── analyzers/              # Core analysis components
│   ├── advanced_validation.py  # Phase 3 validation framework
│   ├── ml_ensemble.py          # ML models with regime features
│   ├── regime_detector.py      # Regime detection system
│   ├── etf_risk_classifier.py  # Risk classification
│   └── [other analyzers...]
│
├── system/                 # Core system infrastructure
│   ├── orchestrator.py        # Main system controller
│   └── [system components...]
│
├── data_manager/           # Data management & external feeds
├── frameworks/             # Analysis frameworks
├── indicators/             # Technical indicators
├── utilities/              # Helper utilities
├── dashboard/              # Interactive dashboard
├── config/                 # Configuration files
├── data/                   # Data storage
└── docs/                   # Documentation
```

---

## 🔧 Dependencies

```bash
pip install -r system/requirements.txt
```

Key dependencies:
- **Data**: pandas, numpy, pyarrow
- **ML**: scikit-learn, scipy
- **Visualization**: dash, plotly
- **Data Sources**: yfinance, yahooquery
- **Statistics**: statsmodels

---

## 📈 Performance Metrics

### Validation Results
- **Nested CV MAE**: 0.035 (target: <0.05)
- **Hit Rate**: 40% (target: >50%)
- **Confidence Score**: Medium (0.534)
- **Model Stability**: High (1.0)

### Data Coverage
- **ETF Universe**: 377+ Australian ETFs
- **Historical Data**: 5 years (2020-2025)
- **Update Frequency**: Daily
- **External Indicators**: VIX, rates, FX, gold

---

## 📚 Documentation

### Essential Reading
- **[ACTION_PLAN.md](./ACTION_PLAN.md)**: Current status and roadmap
- **[Phase Reports](./docs/)**: Detailed implementation documentation
- **[Technical Guides](./docs/guides/)**: Development and user guides
- **[System Reference](./docs/reference/)**: Architecture and specifications

### Key Documents
- **Phase 1 Completion**: Data pipeline and regime framework
- **Phase 2 Completion**: ML model enhancements
- **Phase 3 Completion**: Advanced validation system
- **Development Guide**: System architecture and coding standards

---

## 🎯 Current Status

**Phase 3 Complete - Production Ready**
- ✅ Regime-aware ML models operational
- ✅ Advanced validation framework working
- ✅ Confidence flagging system implemented
- ✅ Dashboard with real-time visualizations
- ✅ External data integration complete

### Next Steps
- **Phase 4**: Production deployment and monitoring
- **Phase 5**: Enhanced features and portfolio optimization

---

## 🔍 Troubleshooting

**Common Issues:**
1. **Dashboard not loading**: Check dependencies and restart
2. **Missing external data**: Run data refresh script
3. **ML model errors**: Verify data quality and feature extraction
4. **Validation failures**: Check data sufficiency and parameters

**Support:**
- Check documentation in `docs/` directory
- Review phase completion reports
- Consult development guides

---

## 📝 System Information

- **Version**: Phase 3 Complete (December 2025)
- **Python**: 3.8+ required
- **Data**: 5-year historical coverage
- **Validation**: Statistical rigor with bootstrap methods
- **Status**: Production Ready with Advanced ML

---

**Last Updated**: December 2025  
**System Version**: Phase 3 Complete  
**Next Milestone**: Production Deployment (Phase 4)
