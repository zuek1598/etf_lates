import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from system.orchestrator import ETFAnalysisSystem
from data_manager.data_manager import ETFDataManager as ETFDatabase

# Load database and get sample tickers
etf_db = ETFDatabase()
all_tickers = list(etf_db.etf_data.keys())
sample_tickers = all_tickers[:5]

print(f"Sample tickers: {sample_tickers}\n")

# Run analysis on sample
orchestrator = ETFAnalysisSystem()
sample_results = orchestrator.run_full_analysis(sample_tickers)
analysis_results = sample_results.get('analysis_results', {})

# Extract factor values
factor_names = ['ml_forecast', 'ml_confidence', 'hit_rate']
factor_values = {}

for factor in factor_names:
    factor_values[factor] = {}
    for ticker in sample_tickers:
        if ticker in analysis_results:
            value = analysis_results[ticker].get(factor)
            if value is not None:
                factor_values[factor][ticker] = value

# Show structure
print("Factor values structure:")
for factor_name, values_dict in factor_values.items():
    print(f"\n{factor_name}:")
    print(f"  Type: {type(values_dict)}")
    print(f"  Keys: {list(values_dict.keys())}")
    print(f"  Values: {values_dict}")
