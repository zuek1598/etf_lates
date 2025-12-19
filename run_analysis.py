#!/usr/bin/env python3
"""
Entry Point: Run ETF Analysis - Modified System
Simple wrapper to run the full analysis pipeline

Usage:
    python3 run_analysis.py

This will:
1. Analyze all ETFs in the database
2. Generate comprehensive rankings by risk category
3. Save results to data/ directory in Parquet format
4. Display detailed analysis summary

Modified System Features:
- Risk Component: CVaR, Ulcer, Beta, IR (30/30/20/20)
- ML Ensemble: Raw forecasts + confidence (NO bias correction)
- Kalman Hull: Adaptive momentum indicator
- Volume Intelligence: Spike, correlation, A/D
- Scoring: Risk(40%), Technical(30%), ML+Volume(30%)
"""

if __name__ == "__main__":
    # Import and run the system module
    from system.run_analysis import main
    main()