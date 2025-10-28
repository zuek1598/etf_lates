# Utility Scripts

Standalone utility scripts for system maintenance and data management.

## Scripts

### download_all_etf_data.py
Downloads historical data for all ETFs in the universe.

**Usage:**
```bash
python3 scripts/download_all_etf_data.py
```

**What it does:**
- Reads ETF universe from `data/etf_universe.parquet`
- Downloads max available history from Yahoo Finance
- Saves to `data/historical/*.parquet`
- Shows progress and success/failure counts

**When to use:**
- Initial setup
- Updating historical data
- Adding new ETFs

### run_backtest.py
Runs backtest on the full ETF universe (non-interactive).

**Usage:**
```bash
python3 scripts/run_backtest.py
```

**What it does:**
- Loads all historical data
- Runs MACD-V strategy backtest on 363 ETFs
- Saves results to `data/backtest_results.parquet`
- Takes ~30-60 seconds

**When to use:**
- After updating historical data
- Testing strategy changes
- Refreshing dashboard backtest data

## Notes

- All scripts must be run from the `modified/` directory
- Requires all dependencies from `system/requirements.txt`
- Logs output to console
