import sys
sys.path.append('.')
from data_manager.data_manager import ETFDataManager

print('🔍 Debugging ETF Data Loading')
print('=' * 40)

data_manager = ETFDataManager()

# Test loading a few specific ETFs
test_tickers = ['VAS.AX', 'IOZ.AX', 'MVB.AX']

for ticker in test_tickers:
    try:
        print(f'\n📊 Testing {ticker}:')
        hist_data = data_manager.load_historical_prices(ticker)
        if hist_data is not None:
            print(f'  ✅ Loaded: {len(hist_data)} rows')
            print(f'  Columns: {list(hist_data.columns)}')
            print(f'  Date range: {hist_data.index[0]} to {hist_data.index[-1]}')
        else:
            print(f'  ❌ No data returned')
    except Exception as e:
        print(f'  ❌ Error: {e}')

# Test universe loading
print(f'\n📈 Testing universe loading:')
universe_df = data_manager.load_universe()
print(f'  ✅ Universe: {len(universe_df)} ETFs')
sample_tickers = universe_df['ticker'].head(5).tolist()
print(f'  Sample tickers: {sample_tickers}')
