import yfinance as yf
from datetime import datetime, timedelta

# Get ETF ticker from user
ticker = input("Enter ETF ticker (e.g., SPY, QQQ, VTI): ").strip().upper()

# Download data
etf = yf.Ticker(ticker)

# Get today's date and date from 1 year ago
today = datetime.now()
one_year_ago = today - timedelta(days=365)

# Fetch historical data
data = etf.history(start=one_year_ago, end=today)

# Get prices
price_one_year_ago = data['Close'].iloc[0]
current_price = data['Close'].iloc[-1]
difference = current_price - price_one_year_ago
percent_change = (difference / price_one_year_ago) * 100

# Display results
print(f"\n{'='*50}")
print(f"ETF: {ticker}")
print(f"{'='*50}")
print(f"Price 1 Year Ago:  ${price_one_year_ago:.2f}")
print(f"Current Price:     ${current_price:.2f}")
print(f"Difference:        ${difference:.2f}")
print(f"Percent Change:    {percent_change:.2f}%")
print(f"{'='*50}\n")