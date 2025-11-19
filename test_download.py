import yfinance as yf
import pandas as pd

# Test downloading a few tickers
tickers = ['DBBF.AX', 'DHOF.AX', 'IBAL.AX', 'MHHT.AX']
price_data = {}

for ticker in tickers:
    try:
        hist = yf.download(ticker, period="2y", progress=False)
        if isinstance(hist, pd.DataFrame) and len(hist) > 0:
            # Handle MultiIndex columns from yfinance
            if isinstance(hist.columns, pd.MultiIndex):
                close_col = [col for col in hist.columns if col[0] == 'Close']
                if close_col:
                    price_data[ticker] = hist[close_col[0]]
                    print(f"✓ {ticker}: Downloaded {len(hist)} rows")
            elif 'Close' in hist.columns:
                price_data[ticker] = hist['Close']
                print(f"✓ {ticker}: Downloaded {len(hist)} rows")
    except Exception as e:
        print(f"✗ {ticker}: {type(e).__name__}: {str(e)[:80]}")

print(f"\nTotal downloaded: {len(price_data)} ETFs")
