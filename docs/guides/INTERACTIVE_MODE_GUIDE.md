# Interactive Mode Guide - Strategy Comparator

## Updated Feature

The `compare_strategy.py` script now supports **interactive mode** - you can run it without command-line arguments and choose what you want to compare!

## How to Use

Simply run the script with no arguments:

```bash
python scripts/compare_strategy.py
```

You'll see an interactive menu:

```
================================================================================
STRATEGY COMPARATOR - INTERACTIVE MODE
================================================================================

Options:
  1. Compare single ETF (e.g., VAS.AX)
  2. Compare multiple ETFs (e.g., VAS.AX VGS.AX IOZ.AX)
  3. Top 10 performers
  4. Bottom 10 performers
  5. All ETFs (export to CSV)
  6. Exit

Enter choice (1-6): _
```

## Menu Options

### Option 1: Compare Single ETF

```
Enter choice (1-6): 1
Enter ETF ticker (e.g., VAS.AX): ATOM.AX
```

**Output:**
- Full detailed analysis for ATOM.AX
- Strategy performance metrics
- Buy-and-hold comparison (both approaches)
- Benchmark comparisons (ASX200, S&P500)
- Summary table

### Option 2: Compare Multiple ETFs

```
Enter choice (1-6): 2
Enter ETF tickers separated by spaces (e.g., VAS.AX VGS.AX IOZ.AX): VAS.AX VGS.AX IOZ.AX
```

**Output:**
- Quick comparison table with all three ETFs side-by-side
- Columns: Ticker, Strategy Return, Buy-Hold Return, Alpha, Outperforms, Trades, Win Rate, Sharpe

### Option 3: Top 10 Performers

```
Enter choice (1-6): 3
```

**Output:**
- Automatically shows top 10 ETFs by strategy return
- Ranks them by performance
- Shows which ones outperform buy-hold
- No additional input needed

### Option 4: Bottom 10 Performers

```
Enter choice (1-6): 4
```

**Output:**
- Automatically shows bottom 10 ETFs by strategy return
- Identifies your worst trades
- Useful for understanding what NOT to do
- No additional input needed

### Option 5: All ETFs (Export to CSV)

```
Enter choice (1-6): 5
Enter output filename (default: strategy_results.csv): my_results.csv
```

**Output:**
- Analyzes all 313 backtested ETFs
- Exports results to CSV file
- Can be opened in Excel for further analysis
- Default filename: `strategy_results.csv`

### Option 6: Exit

```
Enter choice (1-6): 6
```

Closes the program.

## Command-Line Arguments Still Work

The script still supports command-line arguments for scripting/automation:

```bash
# Single ETF (command-line)
python scripts/compare_strategy.py VAS.AX

# Multiple ETFs (command-line)
python scripts/compare_strategy.py VAS.AX VGS.AX IOZ.AX

# Top 10 (command-line)
python scripts/compare_strategy.py --top 10

# Bottom 10 (command-line)
python scripts/compare_strategy.py --bottom 10

# All ETFs (command-line)
python scripts/compare_strategy.py --all --output results.csv

# Save report (command-line)
python scripts/compare_strategy.py ATOM.AX --save-report atom_report.txt
```

## Examples

### Example 1: Analyze Your Best Trade

```
$ python scripts/compare_strategy.py

Enter choice (1-6): 1
Enter ETF ticker: ATOM.AX

[Detailed analysis showing ATOM.AX returned +39.37% vs buy-hold +16.59%]
```

### Example 2: Batch Compare Top Performers

```
$ python scripts/compare_strategy.py

Enter choice (1-6): 3

[Shows top 10 ETFs with quick comparison table]
```

### Example 3: Identify Problem Areas

```
$ python scripts/compare_strategy.py

Enter choice (1-6): 4

[Shows bottom 10 ETFs - strategies that failed]
```

### Example 4: Full Analysis Export

```
$ python scripts/compare_strategy.py

Enter choice (1-6): 5
Enter output filename: my_analysis.csv

[Analyzes all 313 ETFs and saves to my_analysis.csv]
```

## Input Validation

The script includes input validation:
- Empty inputs are rejected with "Invalid ticker. Try again."
- Invalid menu choices (7, 8, abc, etc.) are rejected with "Invalid choice. Try again."
- You can keep trying until you enter valid input

## Tips

1. **Ticker format** - Works with or without .AX suffix
   - `ATOM.AX` ✓
   - `ATOM` ✓
   - Both are converted to uppercase automatically

2. **Multiple tickers** - Space-separated, no commas
   - `VAS.AX VGS.AX IOZ.AX` ✓
   - `VAS.AX, VGS.AX, IOZ.AX` ✗ (commas not supported)

3. **CSV export** - Useful for Excel analysis
   - Choose option 5 for all ETFs
   - Open the CSV in Excel
   - Use pivot tables to analyze by category
   - Sort by Alpha to find best opportunities

## File Output

### CSV Export Structure

When you choose option 5, the CSV file includes:

```
Ticker,Strategy Return,Buy-Hold Return,Alpha,Outperforms,Trades,Win Rate,Sharpe
VAS.AX,1.96%,0.51%,+1.45%,Yes,10,0.8%,2.59
VGS.AX,-10.07%,3.68%,-13.74%,No,1,0.0%,0.00
ATOM.AX,39.37%,16.59%,+22.77%,Yes,1,1.0%,0.00
```

Can be analyzed in Excel with:
- Filtering by Outperforms = Yes
- Sorting by Alpha (descending)
- Creating pivot tables by category
- Charting return distribution

## Troubleshooting

**"Invalid ticker" message**
- Check spelling
- Ensure no extra spaces
- Try with .AX suffix

**"No data for benchmark"**
- Network issue (try again later)
- Benchmark temporarily unavailable
- Analysis continues with available data

**Option doesn't work**
- Make sure you're entering 1-6
- Try again after the error message

## Comparison with Command-Line Mode

| Use Case | Interactive Mode | Command-Line Mode |
|----------|------------------|-------------------|
| Quick single ETF check | ✓ Easier | ✓ Faster if you know ticker |
| Compare multiple ETFs | ✓ Good | ✓ Good |
| Top/bottom performers | ✓ Easiest | ✓ Requires remembering --top flag |
| Full batch analysis | ✓ Works | ✓ Better for scripts |
| Automation/scripting | ✗ No | ✓ Yes |

## Next Steps

1. **Run interactive mode**: `python scripts/compare_strategy.py`
2. **Choose option**: Pick from 1-6
3. **Provide input**: Enter ticker or choose from menu
4. **Analyze output**: Review strategy performance
5. **Export if needed**: Use option 5 for Excel analysis

## Quick Workflow

```bash
# Step 1: See what worked
python scripts/compare_strategy.py
# Choose: 3 (Top 10)

# Step 2: See what failed
python scripts/compare_strategy.py
# Choose: 4 (Bottom 10)

# Step 3: Deep dive on winner
python scripts/compare_strategy.py
# Choose: 1, enter ATOM.AX

# Step 4: Deep dive on loser
python scripts/compare_strategy.py
# Choose: 1, enter HGEN.AX

# Step 5: Export all for Excel
python scripts/compare_strategy.py
# Choose: 5, name the file
```

Total time: ~5 minutes to understand your entire strategy performance across 313 ETFs.
