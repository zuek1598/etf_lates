#!/usr/bin/env python3
import pandas as pd

# Load ETF universe data
df = pd.read_parquet('data/etf_universe.parquet')

print('ETF Universe - Score Analysis')
print(f'Total ETFs: {len(df)}')
print(f'Composite scores range: {df["composite_score"].min():.1f} - {df["composite_score"].max():.1f}')
print(f'Average composite score: {df["composite_score"].mean():.1f}')
print(f'Median composite score: {df["composite_score"].median():.1f}')
print()

print('Top 10 ETFs by composite score:')
top10 = df.nlargest(10, 'composite_score')[['ticker', 'composite_score', 'risk_category', 'name']]
for _, row in top10.iterrows():
    print(f'  {row["ticker"]:8} {row["composite_score"]:5.1f} {row["risk_category"]:8} {row["name"][:30]}')
print()

print('Score distribution:')
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
dist = pd.cut(df['composite_score'], bins=bins).value_counts().sort_index()
for bin_range, count in dist.items():
    print(f'  {bin_range}: {count} ETFs ({count/len(df)*100:.1f}%)')

# Check SNAS specifically
snas_data = df[df['ticker'] == 'SNAS.AX']
if len(snas_data) > 0:
    snas = snas_data.iloc[0]
    print()
    print('SNAS.AX Analysis:')
    print(f'  Composite Score: {snas["composite_score"]:.1f}')
    print(f'  Risk Category: {snas["risk_category"]}')
    print(f'  Name: {snas["name"]}')

# Check component scores for top ETFs
print()
print('Component Scores for Top 5 ETFs:')
for _, row in df.nlargest(5, 'composite_score').iterrows():
    print(f'  {row["ticker"]}: Risk={row.get("risk_score", "N/A")}, Momentum={row.get("momentum_score", "N/A")}, Forecast={row.get("forecast_score", "N/A")}, Volume={row.get("volume_score", "N/A")}')
