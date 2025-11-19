import yfinance as yf
import pandas as pd

# Test downloading a single ticker
ticker = 'DBBF.AX'
print(f"Testing yfinance download for {ticker}...")

hist = yf.download(ticker, period='2y', progress=False)

print(f"Type: {type(hist)}")
print(f"Shape: {hist.shape if hasattr(hist, 'shape') else 'N/A'}")
print(f"Columns: {list(hist.columns) if hasattr(hist, 'columns') else 'N/A'}")
print(f"Empty: {len(hist) == 0}")
print(f"\nFirst few rows:")
print(hist.head())
