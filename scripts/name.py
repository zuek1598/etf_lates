import yfinance as yf
name = 'MCCL.AX'
print(yf.Ticker(name).info)
print(yf.Ticker(name).info['longName'])
print(yf.Ticker(name).info['shortName'])
print(yf.Ticker(name).info['region'])

import yahooquery as yq
print(yq.Ticker(name).fund_sector_weightings)
print(yq.Ticker(name).asset_profile)
for i in yq.Ticker(name).fund_top_holdings:
    print(i)
