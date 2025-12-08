import yahooquery as yq

name = "ASIA.AX"
ticker = yq.Ticker(name)

# Get holdings and asset allocation
holdings_info = ticker.fund_holding_info[name]
holdings_data = holdings_info['holdings']

# Determine primary asset class from fund allocation
stock_position = holdings_info.get('stockPosition', 0)
bond_position = holdings_info.get('bondPosition', 0)
other_position = holdings_info.get('otherPosition', 0)

# Classify primary asset type
if stock_position > 0.5:
    primary_asset_class = "Equity"
elif bond_position > 0.5:
    primary_asset_class = "Bond"
elif other_position > 0.5:
    fund_name = ticker.quote_type[name].get('longName', '').lower()
    crypto_keywords = ['bitcoin', 'ethereum', 'crypto', 'blockchain']
    commodity_keywords = ['gold', 'silver', 'oil', 'commodity', 'metal']
    
    if any(keyword in fund_name for keyword in crypto_keywords):
        primary_asset_class = "Crypto"
    elif any(keyword in fund_name for keyword in commodity_keywords):
        primary_asset_class = "Commodity"
    else:
        primary_asset_class = "Other"
else:
    primary_asset_class = "Mixed"

# Get primary sector from sector weightings
sector_weightings = holdings_info.get('sectorWeightings', [])
if sector_weightings:
    sector_dict = {list(s.keys())[0]: list(s.values())[0] for s in sector_weightings}
    primary_sector = max(sector_dict, key=sector_dict.get)
    primary_sector_weight = sector_dict[primary_sector]
else:
    if primary_asset_class in ["Commodity", "Crypto"]:
        primary_sector = primary_asset_class
    else:
        primary_sector = "N/A"
    primary_sector_weight = 0

# Enhanced exchange-to-region mapping based on your table
exchange_region_map = {
    '-HK': 'Hong Kong',
    '-US': 'United States',
    '-CA': 'Canada',
    '-GB': 'United Kingdom',
    '-IN': 'India',
    '-JP': 'Japan',
    '-JKT': 'Indonesia',
    '-PH': 'Philippines',
    '-AU': 'Australia',
    '-TH': 'Thailand',
    '-NZ': 'New Zealand',
    '-ZA': 'South Africa',
    '-SG': 'Singapore',
    '-SE': 'Sweden',
    '-CN': 'China',
    '-BM': 'Bermuda',
    '-NE': 'Canada',  # NEO Exchange
    '-BR': 'Brazil',
    '-DE': 'Germany',
    '-FR': 'France',
    '.AX': 'Australia',  # ASX format
    '.L': 'United Kingdom',
    '.TO': 'Canada',
    '.HK': 'Hong Kong',
    '.T': 'Japan',
    '.DE': 'Germany',
    '.PA': 'France',
    '.SI': 'Singapore',
    '.NZ': 'New Zealand',
    '': 'United States'  # No suffix = US
}

# Determine primary region from holdings
regions = []
for holding in holdings_data:
    symbol = holding['symbol']
    weight = holding.get('holdingPercent', 0)
    
    # Check for dash suffix first (e.g., AAPL-US)
    if '-' in symbol:
        suffix = '-' + symbol.split('-')[-1]
    # Then check for dot suffix (e.g., BHP.AX)
    elif '.' in symbol:
        suffix = '.' + symbol.split('.')[-1]
    else:
        suffix = ''
    
    region = exchange_region_map.get(suffix, 'United States')  # Default to US
    regions.append((region, weight))

# Calculate primary region by weight
if regions:
    region_weights = {}
    for region, weight in regions:
        region_weights[region] = region_weights.get(region, 0) + weight
    primary_region = max(region_weights, key=region_weights.get)
    primary_region_weight = region_weights[primary_region]
else:
    primary_region = "N/A"
    primary_region_weight = 0

print(f"Primary Asset Class: {primary_asset_class}")
print(f"Stock Position: {stock_position*100:.1f}%")
print(f"Bond Position: {bond_position*100:.1f}%")
print(f"Other Position: {other_position*100:.1f}%")
print(f"\nPrimary Sector: {primary_sector}")
if primary_sector_weight > 0:
    print(f"Sector Weight: {primary_sector_weight*100:.1f}%")
print(f"\nPrimary Region: {primary_region}")
print(f"Region Weight: {primary_region_weight*100:.1f}%")
