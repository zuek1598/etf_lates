#!/usr/bin/env python3
"""
Dashboard Launcher for Modified ETF Analysis System
Runs the Dash web application on port 8051

Usage:
    python3 run_dashboard.py
    
Then open: http://127.0.0.1:8051/
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and run the dashboard
from dashboard.app import app

if __name__ == '__main__':
    print("\n" + "="*70)
    print("LAUNCHING MODIFIED ETF ANALYSIS DASHBOARD")
    print("="*70)
    print("\nFeatures:")
    print("   • 6 Interactive Pages:")
    print("     - Summary: Overview & Top ETFs")
    print("     - Growth Opportunities: High-potential ETFs")
    print("     - Backtest Results: Strategy validation (363 ETFs)")
    print("     - Macro & Geo: Market context analysis")
    print("     - Explorer: Search & filter ETFs")
    print("     - Details: Individual ETF deep-dive")
    print("   • Kalman Hull Supertrend Indicators")
    print("   • Volume Intelligence Metrics")
    print("   • ML Ensemble Forecasts (Raw, No Bias Correction)")
    print("   • Risk Component Analysis (CVaR 30%, Ulcer 30%, Beta 20%, IR 20%)")
    print("   • Candlestick/Line Chart Toggle")
    print("\nDashboard URL: http://127.0.0.1:8050/")
    print("="*70)
    print("\n⏳ Starting server...\n")
    
    app.run(debug=True, port=8050, host='127.0.0.1')

