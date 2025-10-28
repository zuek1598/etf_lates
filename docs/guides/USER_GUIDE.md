# 📖 User Guide - ETF Analysis System

**Version:** 3.1 (Growth-Optimized)  
**For:** Investors, Portfolio Managers, Analysts

---

## 🎯 What This System Does

The ETF Analysis System helps you:
1. **Identify growth opportunities** among 385 Australian ETFs
2. **Assess risk** using institutional-grade metrics
3. **Forecast returns** using ML and momentum indicators
4. **Backtest strategies** to validate performance
5. **Monitor macro/geopolitical risks** affecting markets

---

## 🚀 Getting Started

### **Step 1: Run Analysis**
```bash
cd "/Users/uliana/Desktop/new_alpha/latest /modified"
python3 run_analysis.py
```

**What happens:**
- Downloads historical data (5-20 years per ETF)
- Analyzes all 385 ETFs
- Optionally runs backtest
- Saves results

**Time:** 15-30 minutes (data download) + 10-15 minutes (analysis)

---

### **Step 2: View Dashboard**
```bash
python3 run_dashboard.py
```

Open: http://127.0.0.1:8050

---

## 📊 Dashboard Pages

### **1. Summary** 📈
- Portfolio overview
- Risk category breakdown
- Top performers
- Key statistics

### **2. Growth Opportunities** 🚀
- Top growth candidates (score > 60)
- Position sizing recommendations
- Momentum vs. composite score scatter plot
- Filtered by MEDIUM/HIGH risk (growth-focused)

### **3. Backtest Results** 📊
- Historical strategy performance
- Win rates, Sharpe ratios
- Comparison vs. buy-and-hold
- Trade statistics

### **4. Macro & Geopolitical Risk** 🌍
- Real-time economic indicators
- Geopolitical threat assessment
- Market regime analysis
- Cached for 4 hours (fast loading)

### **5. ETF Explorer** 🔍
- Search and filter all 385 ETFs
- Sort by any metric
- Filter by risk category
- Quick comparison

### **6. ETF Details** 📊
- Deep dive into individual ETFs
- Price charts (candlestick/line toggle)
- Risk breakdown
- ML forecast quality
- Volume analysis

---

## 🎯 How to Use the System

### **For Growth Investors:**
1. Go to **Growth Opportunities** page
2. Focus on ETFs with score > 60
3. Check position sizing recommendations
4. Review momentum indicators (Kalman Hull)
5. Verify with backtest results

### **For Risk-Conscious Investors:**
1. Go to **Summary** page
2. Check LOW/MEDIUM risk categories
3. Review CVaR and Ulcer Index
4. Check Information Ratio
5. Monitor Macro/Geo risks

### **For Active Traders:**
1. Go to **ETF Details** page
2. Check Kalman Hull trend (1 = bullish, -1 = bearish)
3. Monitor signal strength (> 0.4 = strong)
4. Watch volume spike score (> 70 = unusual activity)
5. Use ML confidence for entry timing

---

## 📈 Understanding the Scores

### **Composite Score (0-100)**
**Calculation:**
- Momentum: 35%
- Forecast: 25%
- Risk: 25%
- Volume: 15%

**Interpretation:**
- **> 70:** Excellent opportunity
- **60-70:** Good opportunity
- **50-60:** Moderate
- **< 50:** Avoid

### **Risk Score (0-100)**
**Components:**
- CVaR (30%) - Tail risk
- Ulcer Index (30%) - Drawdown pain
- Beta (20%) - Market sensitivity
- Information Ratio (20%) - Risk-adjusted alpha

**Interpretation:**
- **> 70:** Low risk
- **50-70:** Medium risk
- **< 50:** High risk

### **Momentum Score (0-100)**
**Based on Kalman Hull:**
- Signal strength (60%)
- Efficiency ratio (40%)

**Interpretation:**
- **> 75:** Strong momentum
- **60-75:** Good momentum
- **45-60:** Moderate
- **< 45:** Weak

---

## 🧪 Backtesting

### **What It Tests:**
- Rolling 150-day training window
- 60-day rebalance periods
- Entry: Score > 40, Bullish trend, Signal strength > 0.3
- Exit: Score < 40 OR Bearish trend
- Transaction costs: 0.15% (0.1% commission + 0.05% slippage)

### **Interpreting Results:**
**Win Rate:**
- 55-65% = Good (realistic for momentum strategy)
- < 50% = Strategy not working for this ETF

**Sharpe Ratio:**
- > 1.0 = Excellent risk-adjusted returns
- 0.5-1.0 = Good
- < 0.5 = Poor

**Excess Return:**
- Positive = Beat buy-and-hold
- Negative = Underperformed

**Expected:**
- HIGH risk ETFs: Often beat buy-and-hold (volatility = opportunity)
- LOW risk ETFs: May underperform (steady trends favor buy-and-hold)

---

## ⚠️ Important Notes

### **Data Freshness:**
- Historical data: Downloaded once, reused
- Analysis results: Regenerated each run
- Macro/Geo: Cached for 4 hours

**To refresh:**
```bash
# Delete old data
rm -f data/historical/*.parquet

# Re-run analysis
python3 run_analysis.py
```

### **System Limitations:**
1. **No real-time data** - analysis is point-in-time
2. **ML forecasts** - 60-day horizon, use with caution
3. **Backtest** - past performance ≠ future results
4. **Transaction costs** - assumes 0.15%, may vary

### **Best Practices:**
1. **Run weekly** - Update analysis every Sunday
2. **Compare trends** - Don't rely on single data point
3. **Diversify** - Don't put all capital in top 1-2 ETFs
4. **Monitor macro** - Check Macro/Geo page regularly
5. **Validate** - Cross-reference with your own research

---

## 🆘 Troubleshooting

**Dashboard shows no data:**
```bash
# Re-run analysis
python3 run_analysis.py
```

**Backtest shows 0 trades:**
```bash
# You need more historical data
rm -f data/historical/*.parquet
python3 run_analysis.py
# → Choose Option 2: Full universe backtest
```

**Macro/Geo page slow:**
```bash
# Wait 5-10 seconds on first load (downloading real-time data)
# Subsequent loads: Fast (cached for 4 hours)
# Or click "Refresh Now" button to force update
```

**Dashboard won't start:**
```bash
# Kill existing dashboard
pkill -f run_dashboard
# Restart
python3 run_dashboard.py
```

---

## 📚 Next Steps

- **Deep dive:** Read [SPECIFICATION.md](../SPECIFICATION.md)
- **Backtest:** See [BACKTEST_GUIDE.md](BACKTEST_GUIDE.md)
- **Development:** See [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)

---

**Questions?** See [README.md](../README.md) for more resources.
