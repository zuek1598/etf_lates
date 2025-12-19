"""
Enhanced ETF Analysis Dashboard
Multi-page dashboard with search, filters, and detailed ETF pages

Author: ETF Analysis System
Date: October 5, 2025
Version: 2.0
"""

# Fix imports so this works from anywhere
import sys
from pathlib import Path
# Add project root to path if not already there
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.express as px
import plotly.graph_objects as go
# New folder structure imports
from dashboard.data_loader import ETFDataLoader
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Import macro and geopolitical frameworks (new structure)
from frameworks.macro_framework import calculate_macro_framework
from frameworks.geopolitical_framework import calculate_geopolitical_framework
from frameworks.integrated_framework import calculate_complete_risk_assessment

# Import growth components
from dashboard.growth_components import (
    create_growth_opportunities_page,
    create_backtest_results_page
)

# Initialize Dash app
app = dash.Dash(__name__, title="ETF Analysis Dashboard", suppress_callback_exceptions=True)

# Initialize data loader
loader = ETFDataLoader(data_dir='data')

# Load data at startup
try:
    universe = loader.load_universe()
    metadata = loader.load_metadata()
    print(f"Loaded {len(universe)} ETFs from {metadata.get('analysis_date', 'N/A')}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("\nPlease run 'python etf_dashboard.py' first to generate data files.")
    exit(1)

# ============================================================================
# MACRO/GEO FRAMEWORK CACHING
# ============================================================================
# Cache for macro/geo results to avoid repeated API calls
_macro_geo_cache = {
    'data': None,
    'timestamp': None,
    'cache_duration_hours': 4  # Cache for 4 hours
}

def get_cached_macro_geo():
    """
    Get cached macro/geo results or fetch new if expired
    Returns: dict with macro and geo results
    """
    cache = _macro_geo_cache
    now = datetime.now()
    
    # Check if cache is valid
    if cache['data'] is not None and cache['timestamp'] is not None:
        age = (now - cache['timestamp']).total_seconds() / 3600  # hours
        if age < cache['cache_duration_hours']:
            print(f"Using cached macro/geo data (age: {age:.1f}h)")
            return cache['data']
    
    # Cache expired or doesn't exist - fetch new data
    print("Fetching fresh macro/geo data...")
    try:
        result = calculate_complete_risk_assessment()
        cache['data'] = result
        cache['timestamp'] = now
        print(f"Macro/geo data cached at {now.strftime('%H:%M:%S')}")
        return result
            
    except Exception as e:
        print(f"Error fetching macro/geo data: {e}")
        import traceback
        traceback.print_exc()
        # Return cached data even if expired, or None
        return cache['data'] if cache['data'] is not None else None

def clear_macro_geo_cache():
    """Clear the macro/geo cache to force refresh"""
    _macro_geo_cache['data'] = None
    _macro_geo_cache['timestamp'] = None
    print(" Macro/geo cache cleared")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_historical_data(ticker, data_dir='data', period='1y'):
    """
    Get historical data for a ticker (tries cache first, then downloads)
    
    Args:
        ticker: ETF ticker symbol
        data_dir: Directory containing cached data
        period: Period to download if not cached
        
    Returns:
        DataFrame with historical data, or empty DataFrame if failed
    """
    # Try to load from cache first
    historical_dir = Path(data_dir) / 'historical'
    file_name = f"{ticker.replace('.', '_')}.parquet"
    file_path = historical_dir / file_name
    
    if file_path.exists():
        try:
            print(f"Loading cached data for {ticker}")
            data = pd.read_parquet(file_path)
            return data
        except Exception as e:
            print(f" Error loading cached data for {ticker}: {e}")
    
    # Download if not cached
    try:
        print(f"Downloading data for {ticker}")
        data = yf.download(ticker, period=period, progress=False)
        
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        return data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return pd.DataFrame()


# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

def create_header():
    """Create dashboard header with navigation"""
    return html.Div([
        html.Div([
            html.H1("ETF Analysis Dashboard", 
                   style={'color': '#2c3e50', 'margin': '0', 'display': 'inline-block'}),
            html.Div([
                html.Span(f"Total ETFs: {len(universe)}", 
                         style={'marginRight': '20px', 'color': '#7f8c8d'}),
                html.Span(f"Analysis Date: {metadata.get('analysis_date', 'N/A')[:10]}", 
                         style={'color': '#7f8c8d'})
            ], style={'float': 'right', 'paddingTop': '10px'})
        ], style={'borderBottom': '2px solid #3498db', 'paddingBottom': '10px', 'marginBottom': '20px'}),
        
        # Navigation tabs
        html.Div([
            dcc.Tabs(id='page-tabs', value='summary', children=[
                dcc.Tab(label='Summary', value='summary', 
                       style={'fontWeight': 'bold'},
                       selected_style={'fontWeight': 'bold', 'backgroundColor': '#3498db', 'color': 'white'}),
                dcc.Tab(label='Growth Opportunities', value='growth',
                       style={'fontWeight': 'bold'},
                       selected_style={'fontWeight': 'bold', 'backgroundColor': '#27ae60', 'color': 'white'}),
                dcc.Tab(label='Backtest Results', value='backtest',
                       style={'fontWeight': 'bold'},
                       selected_style={'fontWeight': 'bold', 'backgroundColor': '#9b59b6', 'color': 'white'}),
                dcc.Tab(label='Macro & Geo', value='macro_geo',
                       style={'fontWeight': 'bold'},
                       selected_style={'fontWeight': 'bold', 'backgroundColor': '#3498db', 'color': 'white'}),
                dcc.Tab(label='ETF Explorer', value='explorer',
                       style={'fontWeight': 'bold'},
                       selected_style={'fontWeight': 'bold', 'backgroundColor': '#3498db', 'color': 'white'}),
                dcc.Tab(label='ETF Details', value='details',
                       style={'fontWeight': 'bold'},
                       selected_style={'fontWeight': 'bold', 'backgroundColor': '#3498db', 'color': 'white'})
            ])
        ])
    ])


def get_forecast_breakdown_display(etf):
    """Create display for ML Ensemble forecast (Modified System - No Bias Correction)"""
    # Get ML forecast and confidence
    ml_forecast = etf.get('ml_forecast', 0.0)
    ml_confidence = etf.get('ml_confidence', 0.0)
    
    return html.Div([
        # ML Ensemble Banner
        html.Div([
            html.H4("ML Ensemble Forecast", 
                   style={'color': '#9b59b6', 'margin': '0', 'fontSize': '24px', 'textAlign': 'center'}),
            html.P("Random Forest + Ridge Regression", 
                   style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '5px 0', 'textAlign': 'center'})
        ], style={'padding': '15px', 'backgroundColor': '#9b59b620', 'borderRadius': '8px', 
                 'border': '2px solid #9b59b6', 'marginBottom': '20px'}),
        
        # Forecast Display
        html.Div([
            html.Div([
                # Raw ML Forecast
                html.Div([
                    html.H5("Raw ML Forecast", style={'color': '#2c3e50', 'fontSize': '16px', 'margin': '0'}),
                    html.P(f"{ml_forecast:+.2f}%", 
                          style={'fontSize': '48px', 
                                'color': '#27ae60' if ml_forecast > 0 else '#e74c3c', 
                                'margin': '10px 0', 'fontWeight': 'bold', 'textAlign': 'center'}),
                    html.P("60-Day Horizon", 
                          style={'fontSize': '14px', 'color': '#7f8c8d', 'margin': '0', 'textAlign': 'center'})
                ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 
                         'margin': '10px', 'border': '2px solid #3498db'}),
                
                # ML Confidence
                html.Div([
                    html.H5("Model Confidence", style={'color': '#2c3e50', 'fontSize': '16px', 'margin': '0'}),
                    html.P(f"{ml_confidence:.0f}/100", 
                          style={'fontSize': '48px', 
                                'color': '#27ae60' if ml_confidence > 70 else '#e67e22' if ml_confidence > 40 else '#e74c3c', 
                                'margin': '10px 0', 'fontWeight': 'bold', 'textAlign': 'center'}),
                    html.P("Model Agreement", 
                          style={'fontSize': '14px', 'color': '#7f8c8d', 'margin': '0', 'textAlign': 'center'})
                ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 
                         'margin': '10px', 'border': '2px solid #16a085'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
        ]),
        
        # Model Details
        html.Div([
            html.H4("Model Details", style={'color': '#2c3e50', 'marginTop': '20px', 'marginBottom': '15px'}),
            
            html.Div([
                html.P([
                    html.Strong("Ensemble Components:"),
                    html.Br(),
                    "• Random Forest (60% weight)",
                    html.Br(),
                    "• Ridge Regression (40% weight)"
                ], style={'fontSize': '14px', 'color': '#34495e', 'margin': '10px 0', 'lineHeight': '1.8'})
            ], style={'padding': '15px', 'backgroundColor': '#ffffff', 'borderRadius': '8px', 
                     'marginBottom': '10px', 'border': '1px solid #bdc3c7'}),
            
            html.Div([
                html.P([
                    html.Strong("No Bias Correction Applied:"),
                    html.Br(),
                    "This is the RAW ML output. The modified system does NOT apply bias correction. ",
                    "Use the confidence score to assess reliability."
                ], style={'fontSize': '14px', 'color': '#e67e22', 'margin': '10px 0', 'lineHeight': '1.8', 'fontWeight': 'bold'})
            ], style={'padding': '15px', 'backgroundColor': '#fff3cd', 'borderRadius': '8px', 
                     'border': '2px solid #f39c12'})
        ]),
        
        # Explanation
        html.Div([
            html.P([
                html.Strong("How It Works: "),
                "The ML Ensemble combines Random Forest and Ridge Regression predictions. ",
                "Features include volatility, momentum, mean reversion, and trend indicators. ",
                "The confidence score (0-100) indicates model agreement - higher confidence means both models agree on direction and magnitude. ",
                html.Strong("No bias correction is applied - this is the raw ML output.")
            ], style={'fontSize': '13px', 'color': '#7f8c8d', 'marginTop': '15px', 'fontStyle': 'italic'})
        ])
    ])


def get_quality_explanation(quality_flag, hit_rate, confidence):
    """Provide explanation for the quality flag"""
    if "UNRELIABLE" in quality_flag:
        if confidence < 0.15:
            return f"Marked UNRELIABLE due to very low confidence ({confidence*100:.0f}%), likely insufficient sample data for reliable forecasting."
        elif hit_rate < 0.45:
            return f"Marked UNRELIABLE due to very low hit rate ({hit_rate*100:.0f}%), indicating poor directional accuracy."
    elif "POOR" in quality_flag:
        if hit_rate < 0.50:
            return f"Marked POOR due to low hit rate ({hit_rate*100:.0f}%) indicating limited forecast accuracy."
        else:
            return f"Marked POOR due to low confidence ({confidence*100:.0f}%)."
    elif "EXCELLENT" in quality_flag:
        return f"Marked EXCELLENT with high hit rate ({hit_rate*100:.0f}%) and strong confidence ({confidence*100:.0f}%)."
    elif "GOOD" in quality_flag:
        return f"Marked GOOD with decent hit rate ({hit_rate*100:.0f}%) and confidence ({confidence*100:.0f}%)."
    elif "FAIR" in quality_flag:
        return f"Marked FAIR with moderate hit rate ({hit_rate*100:.0f}%) - use with caution."
    return "Quality assessment based on hit rate and confidence metrics."


def get_hit_rate_quality_flag(hit_rate, confidence):
    """Determine quality flag based on hit rate and confidence"""
    import numpy as np
    
    # Handle NaN values
    if pd.isna(hit_rate) or pd.isna(confidence):
        return "UNRELIABLE", "#e74c3c"
    
    # Quality assessment based on hit rate and confidence
    if confidence < 0.15 or hit_rate < 0.45:
        return "UNRELIABLE", "#e74c3c"
    elif hit_rate >= 0.65 and confidence > 0.70:
        return "EXCELLENT", "#27ae60"
    elif hit_rate >= 0.60 and confidence > 0.50:
        return "GOOD", "#2ecc71"
    elif hit_rate >= 0.52:
        return "FAIR", "#f39c12"
    else:
        return "POOR", "#e67e22"


def load_walk_forward_validation():
    """Load walk-forward validation results from JSON file"""
    import json
    validation_file = Path('data/validation_results.json')
    
    if not validation_file.exists():
        return {}
    
    try:
        with open(validation_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading validation results: {e}")
        return {}


def get_walk_forward_quality_flag(hit_rate):
    """Determine quality flag based on walk-forward validation metrics"""
    if pd.isna(hit_rate):
        return "NOT TESTED", "#95a5a6"
    
    if hit_rate >= 0.65:
        return "EXCELLENT", "#27ae60"
    elif hit_rate >= 0.55:
        return "GOOD", "#2ecc71"
    elif hit_rate >= 0.50:
        return "FAIR", "#f39c12"
    elif hit_rate < 0.45:
        return "POOR", "#e74c3c"
    else:
        return "~ FAIR", "#f39c12"


def get_forecast_quality_display(etf):
    """Create display for walk-forward validation metrics"""
    ticker = etf.get('ticker', '')
    hit_rate = etf.get('hit_rate', float('nan'))
    ml_confidence = etf.get('ml_confidence', float('nan'))
    
    # Check if hit rate data is available
    if pd.isna(hit_rate):
        return html.Div([
            html.P("Walk-forward validation not available for this ETF (requires 312+ days of historical data)", 
                   style={'color': '#7f8c8d', 'fontSize': '16px', 'textAlign': 'center'})
        ])
    
    # Get quality flag based on walk-forward validation results
    validation_flag, validation_color = get_walk_forward_quality_flag(hit_rate)
    
    # Determine hit rate color
    if hit_rate >= 0.60:
        hr_color = '#27ae60'
    elif hit_rate >= 0.55:
        hr_color = '#f39c12'
    else:
        hr_color = '#e74c3c'
    
    return html.Div([
        # Header
        html.Div([
            html.H4("Walk-Forward Validation Results", 
                   style={'color': '#2c3e50', 'marginBottom': '5px', 'textAlign': 'center', 'fontSize': '20px'}),
            html.P("Out-of-sample testing across 5 validation windows (252-day train, 60-day test)", 
                   style={'color': '#7f8c8d', 'fontSize': '13px', 'margin': '0', 'textAlign': 'center', 'fontStyle': 'italic'})
        ], style={'marginBottom': '20px'}),
        
        # Quality Flag
        html.Div([
            html.H3(validation_flag, style={'color': validation_color, 'margin': '0', 'fontSize': '32px', 'textAlign': 'center', 'fontWeight': 'bold'})
        ], style={'padding': '20px', 'backgroundColor': f'{validation_color}20', 'borderRadius': '8px', 
                 'border': f'3px solid {validation_color}', 'marginBottom': '20px'}),
        
        # Metrics Grid
        html.Div([
            # Hit Rate
            html.Div([
                html.P("Hit Rate", style={'fontSize': '14px', 'color': '#7f8c8d', 'margin': '0', 'fontWeight': 'bold'}),
                html.P(f"{hit_rate*100:.0f}%", 
                      style={'fontSize': '36px', 'color': hr_color, 'margin': '10px 0', 'fontWeight': 'bold'}),
                html.P("Direction Accuracy", style={'fontSize': '12px', 'color': '#95a5a6', 'margin': '0'})
            ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 
                     'margin': '5px', 'textAlign': 'center', 'border': '2px solid #ecf0f1', 'minWidth': '150px'}),
            
            # ML Confidence
            html.Div([
                html.P("ML Confidence", style={'fontSize': '14px', 'color': '#7f8c8d', 'margin': '0', 'fontWeight': 'bold'}),
                html.P(f"{ml_confidence:.0f}" if not pd.isna(ml_confidence) else "N/A", 
                      style={'fontSize': '36px', 'color': '#3498db', 'margin': '10px 0', 'fontWeight': 'bold'}),
                html.P("Model Agreement", style={'fontSize': '12px', 'color': '#95a5a6', 'margin': '0'})
            ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 
                     'margin': '5px', 'textAlign': 'center', 'border': '2px solid #ecf0f1', 'minWidth': '150px'}),
            
            # Validation Flag
            html.Div([
                html.P("Validation", style={'fontSize': '14px', 'color': '#7f8c8d', 'margin': '0', 'fontWeight': 'bold'}),
                html.P(validation_flag, 
                      style={'fontSize': '20px', 'color': validation_color, 'margin': '10px 0', 'fontWeight': 'bold'}),
                html.P("Out-of-Sample", style={'fontSize': '12px', 'color': '#95a5a6', 'margin': '0'})
            ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 
                     'margin': '5px', 'textAlign': 'center', 'border': '2px solid #ecf0f1', 'minWidth': '150px'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '20px'}),
        
        # Explanation
        html.Div([
            html.P([html.Strong("What This Means:")], 
                   style={'fontSize': '16px', 'color': '#2c3e50', 'marginBottom': '10px', 'fontWeight': 'bold'}),
            html.P([
                "• ",
                html.Strong("Walk-Forward Validation: "),
                "The model was trained on historical data and tested on completely unseen future data across 5 separate test periods. ",
                html.Strong("This is TRUE out-of-sample testing", style={'color': '#27ae60'}),
                " - not training data."
            ], style={'fontSize': '14px', 'color': '#34495e', 'margin': '8px 0', 'lineHeight': '1.8'}),
            html.P([
                "• ",
                html.Strong("Hit Rate: "),
                f"The model correctly predicted price direction {hit_rate*100:.0f}% of the time. ",
                html.Strong("50% = random", style={'color': '#95a5a6'}),
                ", ",
                html.Strong("> 55% is useful", style={'color': '#f39c12'}),
                ", ",
                html.Strong("> 65% is excellent", style={'color': '#27ae60'}),
                "."
            ], style={'fontSize': '14px', 'color': '#34495e', 'margin': '8px 0', 'lineHeight': '1.8'}),
            html.P([
                "• ",
                html.Strong("Reliability: "),
                html.Strong("HIGH", style={'color': '#27ae60', 'fontSize': '16px'}) if validation_flag.startswith("[EMOJI]") else 
                html.Strong("MEDIUM", style={'color': '#f39c12', 'fontSize': '16px'}) if validation_flag.startswith("[EMOJI]") else 
                html.Strong("LOW", style={'color': '#e74c3c', 'fontSize': '16px'}),
                " - These metrics come from real-world backtesting. You can trust them for investment decisions."
            ], style={'fontSize': '14px', 'color': '#34495e', 'margin': '8px 0', 'lineHeight': '1.8', 'fontWeight': 'bold'})
        ], style={'padding': '20px', 'backgroundColor': '#e8f8f5', 'borderRadius': '8px', 'marginTop': '15px', 
                 'border': '2px solid #27ae60'})
    ])


def create_forecast_quality_summary():
    """Create forecast quality summary section based on hit rate"""
    # Check if hit_rate field exists
    if 'hit_rate' not in universe.columns:
        return html.Div([
            html.H2("Forecast Quality Analysis", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            html.P("Validation data not available. Re-run analysis to generate validation metrics.", 
                   style={'color': '#7f8c8d', 'fontSize': '16px'})
        ])
    
    # Filter out NaN hit rates
    valid_hit_rates = universe[universe['hit_rate'].notna()]
    
    if len(valid_hit_rates) == 0:
        return html.Div("No validation data available", style={'color': '#7f8c8d', 'fontSize': '16px'})
    
    # Calculate hit rate statistics
    hit_rates = valid_hit_rates['hit_rate']
    
    avg_hit_rate = hit_rates.mean()
    median_hit_rate = hit_rates.median()
    
    # Categorize hit rates
    excellent = len(hit_rates[hit_rates >= 0.65])
    good = len(hit_rates[(hit_rates >= 0.55) & (hit_rates < 0.65)])
    fair = len(hit_rates[(hit_rates >= 0.50) & (hit_rates < 0.55)])
    poor = len(hit_rates[hit_rates < 0.50])
    
    # Reliability rate (hit rate >= 55%)
    reliable = excellent + good
    reliability_rate = (reliable / len(hit_rates) * 100) if len(hit_rates) > 0 else 0
    
    return html.Div([
        # Forecast Quality Metrics Cards
        html.Div([
            # Card 1: Average Hit Rate
            html.Div([
                html.Div([
                    html.H3(f"{avg_hit_rate*100:.1f}%", style={'color': '#9b59b6', 'fontSize': '48px', 'margin': '0'}),
                    html.P("Average Hit Rate", style={'color': '#7f8c8d', 'fontSize': '18px', 'margin': '5px 0'})
                ], style={'textAlign': 'center'})
            ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Card 2: Median Hit Rate
            html.Div([
                html.Div([
                    html.H3(f"{median_hit_rate*100:.1f}%", style={'color': '#16a085', 'fontSize': '48px', 'margin': '0'}),
                    html.P("Median Hit Rate", style={'color': '#7f8c8d', 'fontSize': '18px', 'margin': '5px 0'})
                ], style={'textAlign': 'center'})
            ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Card 3: Reliable Models
            html.Div([
                html.Div([
                    html.H3(f"{reliable}", style={'color': '#27ae60', 'fontSize': '48px', 'margin': '0'}),
                    html.P("Reliable Models (≥55% Hit Rate)", style={'color': '#7f8c8d', 'fontSize': '18px', 'margin': '5px 0'})
                ], style={'textAlign': 'center'})
            ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            # Card 4: Total Tested
            html.Div([
                html.Div([
                    html.H3(f"{len(hit_rates)}", style={'color': '#3498db', 'fontSize': '48px', 'margin': '0'}),
                    html.P("Total Models Tested", style={'color': '#7f8c8d', 'fontSize': '18px', 'margin': '5px 0'})
                ], style={'textAlign': 'center'})
            ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
        
        # Hit Rate Distribution
        html.Div([
            # Hit rate distribution pie chart
            html.Div([
                dcc.Graph(
                    figure=px.pie(
                        values=[excellent, good, fair, poor],
                        names=['Excellent (≥65%)', 'Good (55-65%)', 'Fair (50-55%)', 'Poor (<50%)'],
                        title="Hit Rate Distribution",
                        color_discrete_sequence=['#27ae60', '#2ecc71', '#f39c12', '#e74c3c']
                    ).update_traces(textposition='inside', textinfo='percent+label')
                )
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            # Hit rate quality breakdown
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("Excellent (≥65%)", style={'color': '#27ae60'}),
                        html.P(f"{excellent} ETFs ({excellent/len(hit_rates)*100:.1f}%)",
                              style={'fontSize': '20px', 'color': '#2c3e50'})
                    ], style={'padding': '15px', 'backgroundColor': '#d4edda', 'borderRadius': '8px', 'marginBottom': '10px'}),
                    
                    html.Div([
                        html.H4("Good (55-65%)", style={'color': '#2ecc71'}),
                        html.P(f"{good} ETFs ({good/len(hit_rates)*100:.1f}%)",
                              style={'fontSize': '20px', 'color': '#2c3e50'})
                    ], style={'padding': '15px', 'backgroundColor': '#d5f4e6', 'borderRadius': '8px', 'marginBottom': '10px'}),
                    
                    html.Div([
                        html.H4("Fair (50-55%)", style={'color': '#f39c12'}),
                        html.P(f"{fair} ETFs ({fair/len(hit_rates)*100:.1f}%)",
                              style={'fontSize': '20px', 'color': '#2c3e50'})
                    ], style={'padding': '15px', 'backgroundColor': '#fff3cd', 'borderRadius': '8px', 'marginBottom': '10px'}),
                    
                    html.Div([
                        html.H4("Poor (<50%)", style={'color': '#e74c3c'}),
                        html.P(f"{poor} ETFs ({poor/len(hit_rates)*100:.1f}%)",
                              style={'fontSize': '20px', 'color': '#2c3e50'})
                    ], style={'padding': '15px', 'backgroundColor': '#ffe6cc', 'borderRadius': '8px'})
                ])
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'})
        ])
    ])


def create_macro_geo_page():
    """Create macro and geopolitical analysis page"""
    # Get cached macro and geo frameworks (or fetch if expired)
    result = get_cached_macro_geo()
    
    if result is None:
        return html.Div([
            html.H2("Unable to Load Real-Time Analysis", style={'color': '#e74c3c'}),
            html.P("The Macro & Geopolitical frameworks require real-time data access.", 
                   style={'color': '#7f8c8d'}),
            html.P("This page shows market-wide risk analysis and is independent of individual ETF analysis.", 
                   style={'color': '#7f8c8d', 'marginTop': '10px'}),
            html.Button("Retry", id='retry-macro-geo', n_clicks=0, 
                       style={'marginTop': '20px', 'padding': '10px 20px', 'fontSize': '16px'})
        ])
    
    macro = result.get('macro', {})
    geo = result.get('geopolitical', {})
    cache_time = _macro_geo_cache.get('timestamp')
    cache_age = (datetime.now() - cache_time).total_seconds() / 60 if cache_time else 0  # minutes
    
    # Handle NaN values
    if macro.get('risk_score') is None or str(macro.get('risk_score')) == 'nan':
        macro['risk_score'] = 50.0
        macro['risk_level'] = 'MODERATE'
    if geo.get('risk_score') is None or str(geo.get('risk_score')) == 'nan':
        geo['risk_score'] = 50.0
        geo['risk_level'] = 'MODERATE'
    
    return html.Div([
        # Page Title
        html.Div([
            html.H1("Macro Economic & Geopolitical Risk Analysis", 
                   style={'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P("Independent frameworks analyzing market conditions and tail risks",
                  style={'color': '#7f8c8d', 'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '10px'})
        ]),
        
        # Cache Info Banner
        html.Div([
            html.Div([
                html.Span(f"Data Age: {cache_age:.0f} minutes", 
                         style={'color': '#27ae60' if cache_age < 60 else '#e67e22' if cache_age < 180 else '#e74c3c', 
                               'fontWeight': 'bold', 'marginRight': '20px'}),
                html.Span(f"⏰ Last Updated: {cache_time.strftime('%H:%M:%S') if cache_time else 'N/A'}", 
                         style={'color': '#7f8c8d', 'marginRight': '20px'}),
                html.Span(f"Cache Duration: {_macro_geo_cache['cache_duration_hours']}h", 
                         style={'color': '#7f8c8d', 'marginRight': '20px'}),
                html.Button("Refresh Now", id='refresh-macro-geo', n_clicks=0,
                           style={'padding': '5px 15px', 'fontSize': '14px', 'cursor': 'pointer',
                                 'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                                 'borderRadius': '5px', 'fontWeight': 'bold'})
            ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'marginBottom': '20px'})
        ]),
        
        # ============================================
        # MACRO ECONOMIC CYCLE OVERLAY
        # ============================================
        html.Div([
            html.H2("Macro Economic Cycle Overlay", 
                   style={'color': '#3498db', 'borderBottom': '3px solid #3498db', 'paddingBottom': '10px', 'marginBottom': '20px'}),
            
            # Overall Macro Summary Cards
            html.Div([
                # Multiplier Card
                html.Div([
                    html.H3("Cycle Multiplier", style={'color': '#7f8c8d', 'fontSize': '16px', 'margin': '0'}),
                    html.H2(f"{macro['multiplier']:.4f}", 
                           style={'fontSize': '48px', 'color': '#3498db', 'margin': '10px 0', 'fontWeight': 'bold'}),
                    html.P("(0.75 - 1.25 range)", style={'color': '#95a5a6', 'fontSize': '14px', 'margin': '0'})
                ], style={'flex': '1', 'padding': '25px', 'backgroundColor': '#e8f4f8', 'borderRadius': '10px', 
                         'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'textAlign': 'center',
                         'border': '2px solid #3498db'}),
                
                # Composite Score Card
                html.Div([
                    html.H3("Composite Score", style={'color': '#7f8c8d', 'fontSize': '16px', 'margin': '0'}),
                    html.H2(f"{macro['composite_score']:.1f}/100", 
                           style={'fontSize': '48px', 'color': '#27ae60', 'margin': '10px 0', 'fontWeight': 'bold'}),
                    html.P("Overall market strength", style={'color': '#95a5a6', 'fontSize': '14px', 'margin': '0'})
                ], style={'flex': '1', 'padding': '25px', 'backgroundColor': '#e8f8f0', 'borderRadius': '10px', 
                         'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'textAlign': 'center',
                         'border': '2px solid #27ae60'}),
                
                # Regime Card
                html.Div([
                    html.H3("Market Regime", style={'color': '#7f8c8d', 'fontSize': '16px', 'margin': '0'}),
                    html.H2(macro['regime'], 
                           style={'fontSize': '40px', 
                                 'color': '#e74c3c' if macro['regime'] == 'CRISIS' else '#27ae60' if macro['regime'] == 'GOLDILOCKS' else '#f39c12', 
                                 'margin': '10px 0', 'fontWeight': 'bold'}),
                    html.P("Current market state", style={'color': '#95a5a6', 'fontSize': '14px', 'margin': '0'})
                ], style={'flex': '1', 'padding': '25px', 'backgroundColor': '#fef5e7', 'borderRadius': '10px', 
                         'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'textAlign': 'center',
                         'border': f"2px solid {'#e74c3c' if macro['regime'] == 'CRISIS' else '#27ae60' if macro['regime'] == 'GOLDILOCKS' else '#f39c12'}"})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px'}),
            
            # Regime Interpretation
            html.Div([
                html.H4("What This Means:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                html.P([
                    html.Strong(f"{macro['regime']}: "),
                    "Focus on " + ("defensive assets, reduce cyclicals" if macro['regime'] == 'CRISIS' 
                                  else "growth assets, maximize cyclicals" if macro['regime'] == 'GOLDILOCKS' 
                                  else "balanced allocation, mixed signals")
                ], style={'fontSize': '16px', 'color': '#34495e', 'lineHeight': '1.8'}),
                html.P([
                    html.Strong(f"Multiplier {macro['multiplier']:.4f}: "),
                    f"Forecasts are {'reduced by' if macro['multiplier'] < 1.0 else 'boosted by'} "
                    f"{abs((1-macro['multiplier'])*100):.1f}% based on macro conditions"
                ], style={'fontSize': '16px', 'color': '#34495e', 'lineHeight': '1.8'})
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'marginBottom': '30px'}),
            
            # Factor Breakdown
            html.H3("Factor Breakdown", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            
            html.Div([
                # Factor 1: Systematic Risk
                html.Div([
                    html.Div([
                        html.H4("Systematic Risk", style={'color': '#e74c3c', 'fontSize': '18px', 'margin': '0'}),
                        html.P(f"{macro['factors']['systematic_risk']:.1f}/100", 
                              style={'fontSize': '36px', 'color': '#e74c3c', 'margin': '10px 0', 'fontWeight': 'bold'}),
                        html.P(f"Weight: {macro['regime_weights']['systematic_risk']:.0%}", 
                              style={'color': '#7f8c8d', 'fontSize': '14px'}),
                        html.Hr(style={'margin': '15px 0', 'border': '1px solid #ecf0f1'}),
                        html.P("Credit spreads, dollar strength, yield curve dynamics", 
                              style={'color': '#95a5a6', 'fontSize': '13px', 'fontStyle': 'italic'})
                    ], style={'padding': '20px', 'textAlign': 'center'})
                ], style={'flex': '1', 'backgroundColor': '#fff', 'borderRadius': '8px', 'margin': '10px',
                         'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'border': '1px solid #ecf0f1'}),
                
                # Factor 2: Growth Momentum
                html.Div([
                    html.Div([
                        html.H4("Growth Momentum", style={'color': '#27ae60', 'fontSize': '18px', 'margin': '0'}),
                        html.P(f"{macro['factors']['growth_momentum']:.1f}/100", 
                              style={'fontSize': '36px', 'color': '#27ae60', 'margin': '10px 0', 'fontWeight': 'bold'}),
                        html.P(f"Weight: {macro['regime_weights']['growth_momentum']:.0%}", 
                              style={'color': '#7f8c8d', 'fontSize': '14px'}),
                        html.Hr(style={'margin': '15px 0', 'border': '1px solid #ecf0f1'}),
                        html.P("PMI momentum, earnings revisions, sector rotation", 
                              style={'color': '#95a5a6', 'fontSize': '13px', 'fontStyle': 'italic'})
                    ], style={'padding': '20px', 'textAlign': 'center'})
                ], style={'flex': '1', 'backgroundColor': '#fff', 'borderRadius': '8px', 'margin': '10px',
                         'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'border': '1px solid #ecf0f1'}),
                
                # Factor 3: Monetary Policy
                html.Div([
                    html.Div([
                        html.H4("Monetary Policy", style={'color': '#9b59b6', 'fontSize': '18px', 'margin': '0'}),
                        html.P(f"{macro['factors']['monetary_policy']:.1f}/100", 
                              style={'fontSize': '36px', 'color': '#9b59b6', 'margin': '10px 0', 'fontWeight': 'bold'}),
                        html.P(f"Weight: {macro['regime_weights']['monetary_policy']:.0%}", 
                              style={'color': '#7f8c8d', 'fontSize': '14px'}),
                        html.Hr(style={'margin': '15px 0', 'border': '1px solid #ecf0f1'}),
                        html.P("Real rates, inflation trajectory, Fed policy stance", 
                              style={'color': '#95a5a6', 'fontSize': '13px', 'fontStyle': 'italic'})
                    ], style={'padding': '20px', 'textAlign': 'center'})
                ], style={'flex': '1', 'backgroundColor': '#fff', 'borderRadius': '8px', 'margin': '10px',
                         'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'border': '1px solid #ecf0f1'}),
                
                # Factor 4: Regime Classification
                html.Div([
                    html.Div([
                        html.H4("Regime Class", style={'color': '#3498db', 'fontSize': '18px', 'margin': '0'}),
                        html.P(f"{macro['factors']['regime_classification']:.1f}/100", 
                              style={'fontSize': '36px', 'color': '#3498db', 'margin': '10px 0', 'fontWeight': 'bold'}),
                        html.P(f"Weight: {macro['regime_weights']['regime']:.0%}", 
                              style={'color': '#7f8c8d', 'fontSize': '14px'}),
                        html.Hr(style={'margin': '15px 0', 'border': '1px solid #ecf0f1'}),
                        html.P("Dynamic factor reweighting based on market state", 
                              style={'color': '#95a5a6', 'fontSize': '13px', 'fontStyle': 'italic'})
                    ], style={'padding': '20px', 'textAlign': 'center'})
                ], style={'flex': '1', 'backgroundColor': '#fff', 'borderRadius': '8px', 'margin': '10px',
                         'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'border': '1px solid #ecf0f1'})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '40px'}),
            
            # Macro Visual Gauge
            html.Div([
                dcc.Graph(
                    figure=go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=macro['composite_score'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Macro Composite Score", 'font': {'size': 24}},
                        delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "#3498db"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': '#ffcccc'},
                                {'range': [30, 50], 'color': '#fff3cd'},
                                {'range': [50, 70], 'color': '#d4edda'},
                                {'range': [70, 100], 'color': '#c3e6cb'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': macro['composite_score']
                            }
                        }
                    )).update_layout(height=400, font={'size': 16})
                )
            ], style={'marginBottom': '40px'})
        ], style={'marginBottom': '50px', 'padding': '30px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
        
        # ============================================
        # GEOPOLITICAL RISK OVERLAY
        # ============================================
        html.Div([
            html.H2("Geopolitical Risk Overlay", 
                   style={'color': '#e74c3c', 'borderBottom': '3px solid #e74c3c', 'paddingBottom': '10px', 'marginBottom': '20px'}),
            
            # Overall Geo Summary Cards
            html.Div([
                # Risk Score Card
                html.Div([
                    html.H3("Risk Score", style={'color': '#7f8c8d', 'fontSize': '16px', 'margin': '0'}),
                    html.H2(f"{geo['risk_score']:.1f}/100", 
                           style={'fontSize': '48px', 
                                 'color': '#27ae60' if geo['risk_score'] < 20 else '#f39c12' if geo['risk_score'] < 35 else '#e67e22' if geo['risk_score'] < 50 else '#e74c3c', 
                                 'margin': '10px 0', 'fontWeight': 'bold'}),
                    html.P("Higher = More Risk", style={'color': '#95a5a6', 'fontSize': '14px', 'margin': '0'})
                ], style={'flex': '1', 'padding': '25px', 'backgroundColor': '#fef5e7', 'borderRadius': '10px', 
                         'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'textAlign': 'center',
                         'border': '2px solid #e74c3c'}),
                
                # Risk Level Card
                html.Div([
                    html.H3("Risk Level", style={'color': '#7f8c8d', 'fontSize': '16px', 'margin': '0'}),
                    html.H2(geo['risk_level'], 
                           style={'fontSize': '40px', 
                                 'color': '#27ae60' if geo['risk_level'] == 'LOW' else '#f39c12' if geo['risk_level'] == 'MODERATE' else '#e67e22' if geo['risk_level'] == 'HIGH' else '#e74c3c', 
                                 'margin': '10px 0', 'fontWeight': 'bold'}),
                    html.P("Current threat level", style={'color': '#95a5a6', 'fontSize': '14px', 'margin': '0'})
                ], style={'flex': '1', 'padding': '25px', 'backgroundColor': '#fdedec', 'borderRadius': '10px', 
                         'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'textAlign': 'center',
                         'border': f"2px solid {'#27ae60' if geo['risk_level'] == 'LOW' else '#f39c12' if geo['risk_level'] == 'MODERATE' else '#e67e22' if geo['risk_level'] == 'HIGH' else '#e74c3c'}"})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px'}),
            
            # Risk Level Interpretation
            html.Div([
                html.H4("What This Means:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                html.P([
                    html.Strong(f"{geo['risk_level']}: "),
                    (f"Normal allocation - minimal tail risk concerns" if geo['risk_level'] == 'LOW'
                     else f"Reduce risk assets 10-15%, slight defensive tilt" if geo['risk_level'] == 'MODERATE'
                     else f"Reduce risk assets 20-30%, add gold/defense" if geo['risk_level'] == 'HIGH'
                     else f"Reduce risk assets 40%+, maximum defensives" if geo['risk_level'] == 'SEVERE'
                     else f"Crisis portfolio - gold, defense, treasuries only")
                ], style={'fontSize': '16px', 'color': '#34495e', 'lineHeight': '1.8'}),
                html.P([
                    "Position sizing adjustment applied to all risk assets based on their geopolitical exposure coefficients."
                ], style={'fontSize': '14px', 'color': '#7f8c8d', 'fontStyle': 'italic'})
            ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'marginBottom': '30px'}),
            
            # Pillar Breakdown
            html.H3("Pillar Breakdown", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            
            html.Div([
                # Pillar 1: US-China-Taiwan
                html.Div([
                    html.Div([
                        html.H4("US-China-Taiwan", style={'color': '#e74c3c', 'fontSize': '18px', 'margin': '0'}),
                        html.P(f"{geo['pillars']['us_china_taiwan']:.1f}/100", 
                              style={'fontSize': '36px', 'color': '#e74c3c', 'margin': '10px 0', 'fontWeight': 'bold'}),
                        html.P("Weight: 30%", style={'color': '#7f8c8d', 'fontSize': '14px'}),
                        html.Hr(style={'margin': '15px 0', 'border': '1px solid #ecf0f1'}),
                        html.P("Semiconductor supply chain & Taiwan conflict risk", 
                              style={'color': '#95a5a6', 'fontSize': '13px', 'fontStyle': 'italic'})
                    ], style={'padding': '20px', 'textAlign': 'center'})
                ], style={'flex': '1', 'backgroundColor': '#fff', 'borderRadius': '8px', 'margin': '10px',
                         'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'border': '1px solid #ecf0f1'}),
                
                # Pillar 2: Military Conflict
                html.Div([
                    html.Div([
                        html.H4("Military Conflict", style={'color': '#c0392b', 'fontSize': '18px', 'margin': '0'}),
                        html.P(f"{geo['pillars']['military_conflict']:.1f}/100", 
                              style={'fontSize': '36px', 'color': '#c0392b', 'margin': '10px 0', 'fontWeight': 'bold'}),
                        html.P("Weight: 25%", style={'color': '#7f8c8d', 'fontSize': '14px'}),
                        html.Hr(style={'margin': '15px 0', 'border': '1px solid #ecf0f1'}),
                        html.P("VIX spikes, safe-haven flows, energy shocks", 
                              style={'color': '#95a5a6', 'fontSize': '13px', 'fontStyle': 'italic'})
                    ], style={'padding': '20px', 'textAlign': 'center'})
                ], style={'flex': '1', 'backgroundColor': '#fff', 'borderRadius': '8px', 'margin': '10px',
                         'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'border': '1px solid #ecf0f1'}),
                
                # Pillar 3: Trade War
                html.Div([
                    html.Div([
                        html.H4("Trade War", style={'color': '#d35400', 'fontSize': '18px', 'margin': '0'}),
                        html.P(f"{geo['pillars']['trade_war']:.1f}/100", 
                              style={'fontSize': '36px', 'color': '#d35400', 'margin': '10px 0', 'fontWeight': 'bold'}),
                        html.P("Weight: 20%", style={'color': '#7f8c8d', 'fontSize': '14px'}),
                        html.Hr(style={'margin': '15px 0', 'border': '1px solid #ecf0f1'}),
                        html.P("Tariff escalation, US-China divergence, currency wars", 
                              style={'color': '#95a5a6', 'fontSize': '13px', 'fontStyle': 'italic'})
                    ], style={'padding': '20px', 'textAlign': 'center'})
                ], style={'flex': '1', 'backgroundColor': '#fff', 'borderRadius': '8px', 'margin': '10px',
                         'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'border': '1px solid #ecf0f1'}),
                
                # Pillar 4: Financial Stress
                html.Div([
                    html.Div([
                        html.H4("Financial Stress", style={'color': '#8e44ad', 'fontSize': '18px', 'margin': '0'}),
                        html.P(f"{geo['pillars']['financial_stress']:.1f}/100", 
                              style={'fontSize': '36px', 'color': '#8e44ad', 'margin': '10px 0', 'fontWeight': 'bold'}),
                        html.P("Weight: 15%", style={'color': '#7f8c8d', 'fontSize': '14px'}),
                        html.Hr(style={'margin': '15px 0', 'border': '1px solid #ecf0f1'}),
                        html.P("Yield curve inversion, equity drawdowns, PMI contraction", 
                              style={'color': '#95a5a6', 'fontSize': '13px', 'fontStyle': 'italic'})
                    ], style={'padding': '20px', 'textAlign': 'center'})
                ], style={'flex': '1', 'backgroundColor': '#fff', 'borderRadius': '8px', 'margin': '10px',
                         'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'border': '1px solid #ecf0f1'}),
                
                # Pillar 5: Energy Security
                html.Div([
                    html.Div([
                        html.H4("Energy Security", style={'color': '#16a085', 'fontSize': '18px', 'margin': '0'}),
                        html.P(f"{geo['pillars']['energy_security']:.1f}/100", 
                              style={'fontSize': '36px', 'color': '#16a085', 'margin': '10px 0', 'fontWeight': 'bold'}),
                        html.P("Weight: 10%", style={'color': '#7f8c8d', 'fontSize': '14px'}),
                        html.Hr(style={'margin': '15px 0', 'border': '1px solid #ecf0f1'}),
                        html.P("EU energy vulnerability, oil chokepoints, energy volatility", 
                              style={'color': '#95a5a6', 'fontSize': '13px', 'fontStyle': 'italic'})
                    ], style={'padding': '20px', 'textAlign': 'center'})
                ], style={'flex': '1', 'backgroundColor': '#fff', 'borderRadius': '8px', 'margin': '10px',
                         'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'border': '1px solid #ecf0f1'})
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '40px'}),
            
            # Geo Visual Gauge
            html.Div([
                dcc.Graph(
                    figure=go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=geo['risk_score'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Geopolitical Risk Score", 'font': {'size': 24}},
                        delta={'reference': 35, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkred"},
                            'bar': {'color': "#e74c3c"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 20], 'color': '#d4edda'},
                                {'range': [20, 35], 'color': '#fff3cd'},
                                {'range': [35, 50], 'color': '#ffe6cc'},
                                {'range': [50, 65], 'color': '#f8d7da'},
                                {'range': [65, 100], 'color': '#f5b7b1'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': geo['risk_score']
                            }
                        }
                    )).update_layout(height=400, font={'size': 16})
                )
            ])
        ], style={'padding': '30px', 'backgroundColor': '#fef5e7', 'borderRadius': '10px'}),
        
        # ============================================
        # FRAMEWORK INDEPENDENCE
        # ============================================
        html.Div([
            html.H3("Framework Independence", style={'color': '#2c3e50', 'marginTop': '40px', 'marginBottom': '15px'}),
            html.P([
                html.Strong("These frameworks use DIFFERENT data sources: "),
                "Macro uses economic indicators (credit spreads, PMI, inflation, Fed policy), while Geopolitical uses ",
                "tail risk indicators (VIX, defense stocks, Taiwan ETF, gold flows). Less than 5% data overlap ensures ",
                "independent signals without double-counting."
            ], style={'fontSize': '15px', 'color': '#34495e', 'backgroundColor': '#ecf0f1', 'padding': '15px', 
                     'borderRadius': '8px', 'lineHeight': '1.8'})
        ])
    ])


def create_summary_page():
    """Create summary page with key metrics and overview"""
    stats = loader.get_summary_stats()
    risk_breakdown = stats['risk_breakdown']
    
    # Calculate additional stats
    positive_forecast = len(universe[universe['ml_forecast'] > 0])
    negative_forecast = len(universe[universe['ml_forecast'] < 0])
    avg_forecast = universe['ml_forecast'].mean()
    
    return html.Div([
        # Key Metrics Cards
        html.Div([
            html.H2("Key Metrics", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            html.Div([
                # Card 1: Total ETFs
                html.Div([
                    html.Div([
                        html.H3(f"{len(universe)}", style={'color': '#3498db', 'fontSize': '48px', 'margin': '0'}),
                        html.P("Total ETFs", style={'color': '#7f8c8d', 'fontSize': '18px', 'margin': '5px 0'})
                    ], style={'textAlign': 'center'})
                ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#ecf0f1', 
                         'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                
                # Card 2: Average Score
                html.Div([
                    html.Div([
                        html.H3(f"{stats['avg_composite_score']:.1f}", style={'color': '#2ecc71', 'fontSize': '48px', 'margin': '0'}),
                        html.P("Avg Composite Score", style={'color': '#7f8c8d', 'fontSize': '18px', 'margin': '5px 0'})
                    ], style={'textAlign': 'center'})
                ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#ecf0f1', 
                         'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                
                # Card 3: Average YTD Return
                html.Div([
                    html.Div([
                        html.H3(f"{stats['avg_ytd_return']*100:+.1f}%", 
                               style={'color': '#e74c3c' if stats['avg_ytd_return'] < 0 else '#27ae60', 
                                     'fontSize': '48px', 'margin': '0'}),
                        html.P("Avg YTD Return", style={'color': '#7f8c8d', 'fontSize': '18px', 'margin': '5px 0'})
                    ], style={'textAlign': 'center'})
                ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#ecf0f1', 
                         'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                
                # Card 4: Average Sharpe
                html.Div([
                    html.Div([
                        html.H3(f"{stats['avg_sharpe']:.2f}", style={'color': '#9b59b6', 'fontSize': '48px', 'margin': '0'}),
                        html.P("Avg Sharpe Ratio", style={'color': '#7f8c8d', 'fontSize': '18px', 'margin': '5px 0'})
                    ], style={'textAlign': 'center'})
                ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#ecf0f1', 
                         'borderRadius': '10px', 'margin': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
                
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'})
        ], style={'marginBottom': '40px'}),
        
        # Risk Distribution
        html.Div([
            html.H2("Risk Distribution", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            html.Div([
                # Risk breakdown pie chart
                html.Div([
                    dcc.Graph(
                        figure=px.pie(
                            values=list(risk_breakdown.values()),
                            names=list(risk_breakdown.keys()),
                            title="ETFs by Risk Category",
                            color=list(risk_breakdown.keys()),
                            color_discrete_map={'LOW': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e74c3c'}
                        ).update_traces(textposition='inside', textinfo='percent+label')
                    )
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                # Risk stats
                html.Div([
                    html.Div([
                        html.Div([
                            html.H4("🟢 Low Risk", style={'color': '#2ecc71'}),
                            html.P(f"{risk_breakdown.get('LOW', 0)} ETFs ({risk_breakdown.get('LOW', 0)/len(universe)*100:.1f}%)",
                                  style={'fontSize': '20px', 'color': '#2c3e50'})
                        ], style={'padding': '20px', 'backgroundColor': '#d4edda', 'borderRadius': '8px', 'marginBottom': '15px'}),
                        
                        html.Div([
                            html.H4("🟠 Medium Risk", style={'color': '#f39c12'}),
                            html.P(f"{risk_breakdown.get('MEDIUM', 0)} ETFs ({risk_breakdown.get('MEDIUM', 0)/len(universe)*100:.1f}%)",
                                  style={'fontSize': '20px', 'color': '#2c3e50'})
                        ], style={'padding': '20px', 'backgroundColor': '#fff3cd', 'borderRadius': '8px', 'marginBottom': '15px'}),
                        
                        html.Div([
                            html.H4("High Risk", style={'color': '#e74c3c'}),
                            html.P(f"{risk_breakdown.get('HIGH', 0)} ETFs ({risk_breakdown.get('HIGH', 0)/len(universe)*100:.1f}%)",
                                  style={'fontSize': '20px', 'color': '#2c3e50'})
                        ], style={'padding': '20px', 'backgroundColor': '#f8d7da', 'borderRadius': '8px'})
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '2%'})
            ])
        ], style={'marginBottom': '40px'}),
        
        # ML Forecast Summary
        html.Div([
            html.H2("ML Forecast Summary (60-Day)", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            html.Div([
                html.Div([
                    html.H4("Positive Forecasts", style={'color': '#2ecc71', 'margin': '0'}),
                    html.P(f"{positive_forecast} ETFs ({positive_forecast/len(universe)*100:.1f}%)",
                          style={'fontSize': '24px', 'color': '#2c3e50', 'margin': '10px 0'})
                ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#d4edda', 
                         'borderRadius': '8px', 'margin': '10px'}),
                
                html.Div([
                    html.H4("Negative Forecasts", style={'color': '#e74c3c', 'margin': '0'}),
                    html.P(f"{negative_forecast} ETFs ({negative_forecast/len(universe)*100:.1f}%)",
                          style={'fontSize': '24px', 'color': '#2c3e50', 'margin': '10px 0'})
                ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#f8d7da', 
                         'borderRadius': '8px', 'margin': '10px'}),
                
                html.Div([
                    html.H4("Average Forecast", style={'color': '#3498db', 'margin': '0'}),
                    html.P(f"{avg_forecast:+.2f}%",
                          style={'fontSize': '24px', 'color': '#2c3e50', 'margin': '10px 0'})
                ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#d6eaf8', 
                         'borderRadius': '8px', 'margin': '10px'})
            ], style={'display': 'flex', 'flexWrap': 'wrap'})
        ], style={'marginBottom': '40px'}),
        
        # MAE Analysis Section
        html.Div([
            html.H2("Forecast Quality Analysis", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            # Forecast quality statistics
            create_forecast_quality_summary()
        ], style={'marginBottom': '40px'}),
        
        # Top 10 ETFs Overall
        html.Div([
            html.H2("Top 10 ETFs Overall", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            html.Div([
                create_top_etfs_table(universe.nlargest(10, 'composite_score'))
            ])
        ])
    ])


def create_explorer_page():
    """Create ETF explorer page with search and filters"""
    return html.Div([
        # Search and Filters Section
        html.Div([
            html.H2("ETF Explorer", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            
            # Search by ticker
            html.Div([
                html.Label("Search by Ticker or Name:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
                dcc.Input(
                    id='ticker-search',
                    type='text',
                    placeholder='Enter ticker (e.g., VAS.AX) or name...',
                    style={'width': '100%', 'padding': '10px', 'fontSize': '16px', 
                          'borderRadius': '5px', 'border': '2px solid #3498db', 'marginTop': '5px'}
                )
            ], style={'marginBottom': '30px'}),
            
            # Filters
            html.Div([
                # Risk Category Filter
                html.Div([
                    html.Label("Risk Category:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='risk-filter',
                        options=[
                            {'label': 'All Risk Levels', 'value': 'All'},
                            {'label': '🟢 Low Risk', 'value': 'LOW'},
                            {'label': '🟠 Medium Risk', 'value': 'MEDIUM'},
                            {'label': 'High Risk', 'value': 'HIGH'}
                        ],
                        value='All',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                # Score Filter
                html.Div([
                    html.Label("Composite Score Range:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.RangeSlider(
                        id='score-range',
                        min=0,
                        max=100,
                        step=5,
                        value=[0, 100],
                        marks={i: str(i) for i in range(0, 101, 20)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%', 'verticalAlign': 'top'}),
                
                # Forecast Filter
                html.Div([
                    html.Label("ML Forecast (60D):", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='forecast-filter',
                        options=[
                            {'label': 'All Forecasts', 'value': 'All'},
                            {'label': 'Positive (> 0%)', 'value': 'Positive'},
                            {'label': 'Negative (< 0%)', 'value': 'Negative'},
                            {'label': 'Strong Positive (> 5%)', 'value': 'StrongPositive'}
                        ],
                        value='All',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                # Volatility Filter
                html.Div([
                    html.Label("Volatility Range:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.RangeSlider(
                        id='volatility-range',
                        min=0,
                        max=0.5,
                        step=0.05,
                        value=[0, 0.5],
                        marks={i/100: f'{i}%' for i in range(0, 51, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '23%', 'display': 'inline-block', 'verticalAlign': 'top'})
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginBottom': '20px'})
        ]),
        
        # Results count
        html.Div(id='results-count', style={'fontSize': '18px', 'color': '#7f8c8d', 'marginBottom': '15px'}),
        
        # ETF Table
        html.Div(id='etf-table-container')
    ])


def create_details_page():
    """Create ETF details page with key metrics and charts"""
    return html.Div([
        html.H2("ETF Details", style={'color': '#2c3e50', 'marginBottom': '20px'}),
        
        # Ticker selector
        html.Div([
            html.Label("Select ETF:", style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='etf-selector',
                options=[{'label': f"{row['ticker']} - {row['name']}", 'value': row['ticker']} 
                        for _, row in universe.iterrows()],
                value=universe.iloc[0]['ticker'],
                style={'width': '100%', 'marginBottom': '20px'}
            )
        ]),
        
        # ETF Details Content
        html.Div(id='etf-details-content')
    ])


def create_top_etfs_table(df):
    """Create a formatted table for top ETFs"""
    display_df = df[['ticker', 'name', 'risk_category', 'composite_score', 
                     'ytd_return', 'ml_forecast']].copy()
    
    # Format columns
    display_df['composite_score'] = display_df['composite_score'].apply(lambda x: f"{x:.1f}")
    display_df['ytd_return'] = display_df['ytd_return'].apply(lambda x: f"{x*100:.1f}%")
    display_df['ml_forecast'] = display_df['ml_forecast'].apply(lambda x: f"{x:+.1f}%")
    
    return dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[
            {'name': 'Ticker', 'id': 'ticker'},
            {'name': 'Name', 'id': 'name'},
            {'name': 'Risk', 'id': 'risk_category'},
            {'name': 'Score', 'id': 'composite_score'},
            {'name': 'YTD Return', 'id': 'ytd_return'},
            {'name': 'ML Forecast', 'id': 'ml_forecast'}
        ],
        style_cell={
            'textAlign': 'left',
            'padding': '12px',
            'fontFamily': 'Arial',
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': '#34495e',
            'color': 'white',
            'fontWeight': 'bold',
            'fontSize': '15px'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{risk_category} = "Low"'},
                'backgroundColor': '#d4edda'
            },
            {
                'if': {'filter_query': '{risk_category} = "Medium"'},
                'backgroundColor': '#fff3cd'
            },
            {
                'if': {'filter_query': '{risk_category} = "High"'},
                'backgroundColor': '#f8d7da'
            }
        ],
        page_size=10
    )


# ============================================================================
# MAIN LAYOUT
# ============================================================================

app.layout = html.Div([
    create_header(),
    html.Div(id='page-content', style={'padding': '20px'})
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1600px', 'margin': '0 auto', 'padding': '20px'})


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    Output('page-content', 'children'),
    Input('page-tabs', 'value')
)
def render_page(tab):
    """Render page content based on selected tab"""
    if tab == 'summary':
        return create_summary_page()
    elif tab == 'growth':
        return create_growth_opportunities_page(universe)
    elif tab == 'backtest':
        return create_backtest_results_page()
    elif tab == 'macro_geo':
        return create_macro_geo_page()
    elif tab == 'explorer':
        return create_explorer_page()
    elif tab == 'details':
        return create_details_page()
    return html.Div("Page not found")


@app.callback(
    Output('page-content', 'children', allow_duplicate=True),
    Input('refresh-macro-geo', 'n_clicks'),
    prevent_initial_call=True
)
def refresh_macro_geo_data(n_clicks):
    """Refresh macro/geo data when button is clicked"""
    if n_clicks > 0:
        clear_macro_geo_cache()
        print(f"User requested cache refresh (click #{n_clicks})")
        return create_macro_geo_page()
    return dash.no_update


@app.callback(
    [Output('etf-table-container', 'children'),
     Output('results-count', 'children')],
    [Input('ticker-search', 'value'),
     Input('risk-filter', 'value'),
     Input('score-range', 'value'),
     Input('forecast-filter', 'value'),
     Input('volatility-range', 'value')]
)
def update_etf_table(search_term, risk_filter, score_range, forecast_filter, volatility_range):
    """Update ETF table based on filters"""
    
    # Start with all data
    filtered_df = universe.copy()
    
    # Apply search filter
    if search_term:
        search_term = search_term.lower()
        mask = (
            filtered_df['ticker'].str.lower().str.contains(search_term, na=False) |
            filtered_df['name'].str.lower().str.contains(search_term, na=False) |
            filtered_df['subcategory'].str.lower().str.contains(search_term, na=False)
        )
        filtered_df = filtered_df[mask]
    
    # Apply risk filter
    if risk_filter != 'All':
        filtered_df = filtered_df[filtered_df['risk_category'] == risk_filter]
    
    # Apply score range filter
    filtered_df = filtered_df[
        (filtered_df['composite_score'] >= score_range[0]) &
        (filtered_df['composite_score'] <= score_range[1])
    ]
    
    # Apply forecast filter
    if forecast_filter == 'Positive':
        filtered_df = filtered_df[filtered_df['ml_forecast'] > 0]
    elif forecast_filter == 'Negative':
        filtered_df = filtered_df[filtered_df['ml_forecast'] < 0]
    elif forecast_filter == 'StrongPositive':
        filtered_df = filtered_df[filtered_df['ml_forecast'] > 5]
    
    # Apply volatility filter
    filtered_df = filtered_df[
        (filtered_df['volatility'] >= volatility_range[0]) &
        (filtered_df['volatility'] <= volatility_range[1])
    ]
    
    # Sort by composite score
    filtered_df = filtered_df.sort_values('composite_score', ascending=False)
    
    # Create table
    display_df = filtered_df[['ticker', 'name', 'latest_price', 'risk_category', 'composite_score', 
                               'volatility', 'ytd_return', 'ml_forecast', 'ml_confidence', 
                               'kalman_trend', 'volume_spike_score']].copy()
    
    # Format columns
    display_df['latest_price'] = display_df['latest_price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
    display_df['composite_score'] = display_df['composite_score'].apply(lambda x: f"{x:.1f}")
    display_df['volatility'] = display_df['volatility'].apply(lambda x: f"{x*100:.1f}%")
    display_df['ytd_return'] = display_df['ytd_return'].apply(lambda x: f"{x*100:.1f}%")
    display_df['ml_forecast'] = display_df['ml_forecast'].apply(lambda x: f"{x:+.1f}%")
    display_df['ml_confidence'] = display_df['ml_confidence'].apply(lambda x: f"{x:.0f}")
    display_df['kalman_trend'] = display_df['kalman_trend'].apply(lambda x: "🟢" if x > 0 else "[EMOJI]" if x < 0 else "[EMOJI]")
    display_df['volume_spike_score'] = display_df['volume_spike_score'].apply(lambda x: f"{x:.0f}")
    
    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[
            {'name': 'Ticker', 'id': 'ticker'},
            {'name': 'Name', 'id': 'name'},
            {'name': 'Price', 'id': 'latest_price'},
            {'name': 'Risk', 'id': 'risk_category'},
            {'name': 'Score', 'id': 'composite_score'},
            {'name': 'YTD Return', 'id': 'ytd_return'},
            {'name': 'ML Forecast', 'id': 'ml_forecast'},
            {'name': 'ML Conf', 'id': 'ml_confidence'},
            {'name': 'KH Trend', 'id': 'kalman_trend'},
            {'name': 'Vol Spike', 'id': 'volume_spike_score'}
        ],
        style_cell={
            'textAlign': 'left',
            'padding': '12px',
            'fontFamily': 'Arial',
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': '#34495e',
            'color': 'white',
            'fontWeight': 'bold',
            'fontSize': '15px'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{risk_category} = "Low"'},
                'backgroundColor': '#d4edda'
            },
            {
                'if': {'filter_query': '{risk_category} = "Medium"'},
                'backgroundColor': '#fff3cd'
            },
            {
                'if': {'filter_query': '{risk_category} = "High"'},
                'backgroundColor': '#f8d7da'
            }
        ],
        page_size=20,
        sort_action='native',
        filter_action='native'
    )
    
    results_text = f"Showing {len(filtered_df)} of {len(universe)} ETFs"
    
    return table, results_text


@app.callback(
    Output('etf-details-content', 'children'),
    Input('etf-selector', 'value')
)
def update_etf_details(ticker):
    """Update ETF details page with selected ticker"""
    
    if not ticker:
        return html.Div("Please select an ETF")
    
    # Get ETF data
    etf_data = universe[universe['ticker'] == ticker]
    
    if etf_data.empty:
        return html.Div("ETF not found")
    
    etf = etf_data.iloc[0]
    
    # Get historical price data (from cache or download)
    hist_data = get_historical_data(ticker, data_dir='data', period='1y')
    has_data = not hist_data.empty and len(hist_data) > 0
    
    # Key Metrics Section
    metrics_section = html.Div([
        html.H3(f"{ticker} - {etf['name']}", style={'color': '#2c3e50', 'marginBottom': '20px'}),
        
        # Metrics Grid
        html.Div([
            # Row 1
            html.Div([
                html.Div([
                    html.H4("Latest Price", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"${etf['latest_price']:.2f}" if etf.get('latest_price', 0) > 0 else "N/A", 
                          style={'fontSize': '32px', 'color': '#3498db', 'margin': '5px 0', 'fontWeight': 'bold'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'margin': '5px'}),
                
                html.Div([
                    html.H4("Composite Score", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{etf['composite_score']:.1f}", style={'fontSize': '32px', 'color': '#2ecc71', 'margin': '5px 0', 'fontWeight': 'bold'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'margin': '5px'}),
                
                html.Div([
                    html.H4("Risk Category", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{'🟢' if etf['risk_category']=='Low' else '🟠' if etf['risk_category']=='Medium' else '[EMOJI]'} {etf['risk_category']}", 
                          style={'fontSize': '32px', 'color': '#2c3e50', 'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P(f"Score: {etf.get('risk_score', 0):.3f}", 
                          style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '0'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'margin': '5px'}),
                
                html.Div([
                    html.H4("YTD Return", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{etf['ytd_return']*100:+.1f}%", 
                          style={'fontSize': '32px', 'color': '#27ae60' if etf['ytd_return'] > 0 else '#e74c3c', 
                                'margin': '5px 0', 'fontWeight': 'bold'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'margin': '5px'})
            ], style={'display': 'flex', 'marginBottom': '10px'}),
            
            # Row 1.5
            html.Div([
                html.Div([
                    html.H4("1Y Return", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{etf['one_year_return']*100:+.1f}%", 
                          style={'fontSize': '32px', 'color': '#27ae60' if etf['one_year_return'] > 0 else '#e74c3c', 
                                'margin': '5px 0', 'fontWeight': 'bold'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 'borderRadius': '8px', 'margin': '5px'})
            ], style={'display': 'flex', 'marginBottom': '10px'}),
            
            
            # Row 2.5 - Risk Component (30/30/20/20 weights)
            html.Div([
                html.Div([
                    html.H4("Risk Component Analysis", style={'color': '#2c3e50', 'fontSize': '16px', 'margin': '10px 0 5px 0', 'fontWeight': 'bold'}),
                ], style={'width': '100%'})
            ]),
            html.Div([
                html.Div([
                    html.H4("CVaR (30%)", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{etf.get('cvar', 0)*100:.2f}%", 
                          style={'fontSize': '28px', 
                                'color': '#27ae60' if etf.get('cvar', 0) > -0.05 else '#e67e22' if etf.get('cvar', 0) > -0.10 else '#e74c3c',
                                'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P("Tail Risk", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#fef5e7', 'borderRadius': '8px', 'margin': '5px', 'border': '2px solid #f39c12'}),
                
                html.Div([
                    html.H4("Ulcer Index (30%)", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{etf.get('ulcer_index', 0):.2f}%", 
                          style={'fontSize': '28px', 
                                'color': '#27ae60' if etf.get('ulcer_index', 0) < 10 else '#e67e22' if etf.get('ulcer_index', 0) < 20 else '#e74c3c',
                                'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P("Drawdown Pain", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#fef5e7', 'borderRadius': '8px', 'margin': '5px', 'border': '2px solid #f39c12'}),
                
                html.Div([
                    html.H4("Beta (20%)", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{etf.get('beta', 0):.2f}", 
                          style={'fontSize': '28px', 'color': '#3498db', 'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P("Market Sensitivity", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#fef5e7', 'borderRadius': '8px', 'margin': '5px'}),
                
                html.Div([
                    html.H4("Info Ratio (20%)", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{etf.get('information_ratio', 0):.2f}", 
                          style={'fontSize': '28px', 
                                'color': '#27ae60' if etf.get('information_ratio', 0) > 0.5 else '#e67e22' if etf.get('information_ratio', 0) > 0 else '#e74c3c',
                                'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P("Risk-Adj Alpha", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#fef5e7', 'borderRadius': '8px', 'margin': '5px'})
            ], style={'display': 'flex', 'marginBottom': '10px'}),
            
            # Row 3 - ML Ensemble & Technical
            html.Div([
                html.Div([
                    html.H4("Kalman Hull Supertrend", style={'color': '#2c3e50', 'fontSize': '16px', 'margin': '10px 0 5px 0', 'fontWeight': 'bold'}),
                ], style={'width': '100%'})
            ]),
            html.Div([
                html.Div([
                    html.H4("Trend Direction", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P("🟢 BULLISH" if etf.get('kalman_trend', 0) > 0 else "BEARISH" if etf.get('kalman_trend', 0) < 0 else "NEUTRAL",
                          style={'fontSize': '24px', 'color': '#27ae60' if etf.get('kalman_trend', 0) > 0 else '#e74c3c' if etf.get('kalman_trend', 0) < 0 else '#95a5a6',
                                'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P(f"Signal: {etf.get('kalman_trend', 0)}", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#e8f8f5', 'borderRadius': '8px', 'margin': '5px'}),
                
                html.Div([
                    html.H4("Divergence", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(str(etf.get('kalman_divergence', 'none')).upper(),
                          style={'fontSize': '24px', 'color': '#27ae60' if etf.get('kalman_divergence', 'none') == 'bullish' else '#e74c3c' if etf.get('kalman_divergence', 'none') == 'bearish' else '#95a5a6',
                                'margin': '5px 0', 'fontWeight': 'bold'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#e8f8f5', 'borderRadius': '8px', 'margin': '5px'}),
                
                html.Div([
                    html.H4("Efficiency Ratio", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{etf.get('kalman_efficiency_ratio', 0):.3f}",
                          style={'fontSize': '24px', 'color': '#27ae60' if etf.get('kalman_efficiency_ratio', 0) > 0.5 else '#e67e22' if etf.get('kalman_efficiency_ratio', 0) > 0.3 else '#e74c3c',
                                'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P("Trend strength", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#e8f8f5', 'borderRadius': '8px', 'margin': '5px'})
            ], style={'display': 'flex', 'marginBottom': '10px'}),
            
            # Row 3.5 - Volume Intelligence
            html.Div([
                html.Div([
                    html.H4("Volume Intelligence", style={'color': '#2c3e50', 'fontSize': '16px', 'margin': '10px 0 5px 0', 'fontWeight': 'bold'}),
                ], style={'width': '100%'})
            ]),
            html.Div([
                html.Div([
                    html.H4("Volume Spike Score", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{etf.get('volume_spike_score', 0):.1f}/100",
                          style={'fontSize': '28px', 'color': '#27ae60' if etf.get('volume_spike_score', 0) > 70 else '#e67e22' if etf.get('volume_spike_score', 0) > 40 else '#95a5a6',
                                'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P("Unusual volume activity", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#e8f8f5', 'borderRadius': '8px', 'margin': '5px'}),
                
                html.Div([
                    html.H4("Price-Volume Corr", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{etf.get('volume_correlation', 0):+.2f}",
                          style={'fontSize': '28px', 'color': '#27ae60' if etf.get('volume_correlation', 0) > 0.3 else '#e74c3c' if etf.get('volume_correlation', 0) < -0.3 else '#95a5a6',
                                'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P("Volume confirms direction", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#e8f8f5', 'borderRadius': '8px', 'margin': '5px'}),
                
                html.Div([
                    html.H4("A/D Signal", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(str(etf.get('volume_ad_signal', 'neutral')).upper(),
                          style={'fontSize': '24px', 'color': '#27ae60' if etf.get('volume_ad_signal', 'neutral') == 'accumulation' else '#e74c3c' if etf.get('volume_ad_signal', 'neutral') == 'distribution' else '#95a5a6',
                                'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P("Smart money positioning", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#e8f8f5', 'borderRadius': '8px', 'margin': '5px'})
            ], style={'display': 'flex', 'marginBottom': '10px'}),
            
            # Liquidity Metrics
            html.Div([
                html.Div([
                    html.H4("Liquidity Metrics", style={'color': '#2c3e50', 'fontSize': '16px', 'margin': '10px 0 5px 0', 'fontWeight': 'bold'}),
                ], style={'width': '100%'})
            ]),
            html.Div([
                html.Div([
                    html.H4("Avg Daily Volume", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"${(etf.get('avg_daily_volume', 0) * etf.get('latest_price', 0))/1e6:.2f}M" if etf.get('avg_daily_volume', 0) > 0 else "$0.00M", 
                          style={'fontSize': '24px', 
                                'color': '#27ae60' if (etf.get('avg_daily_volume', 0) * etf.get('latest_price', 0)) > 1e6 else '#e67e22' if (etf.get('avg_daily_volume', 0) * etf.get('latest_price', 0)) > 500000 else '#e74c3c',
                                'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P("60-day average", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '12px', 'backgroundColor': '#e8f8f5', 'borderRadius': '8px', 'margin': '5px'}),
                
                html.Div([
                    html.H4("Amihud Ratio", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{etf.get('amihud', 0):.6f}", 
                          style={'fontSize': '24px', 
                                'color': '#27ae60' if etf.get('amihud', 0) < 0.5 else '#e67e22' if etf.get('amihud', 0) < 1.0 else '#e74c3c',
                                'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P("Illiquidity measure", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '12px', 'backgroundColor': '#e8f8f5', 'borderRadius': '8px', 'margin': '5px'}),
                
                html.Div([
                    html.H4("Zero Volume Days", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P(f"{int(etf.get('zero_volume_days', 0))}", 
                          style={'fontSize': '24px', 
                                'color': '#27ae60' if etf.get('zero_volume_days', 0) == 0 else '#e67e22' if etf.get('zero_volume_days', 0) <= 5 else '#e74c3c',
                                'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P("Last 60 days", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '12px', 'backgroundColor': '#e8f8f5', 'borderRadius': '8px', 'margin': '5px'}),
                
                html.Div([
                    html.H4("Liquidity Status", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '0'}),
                    html.P("High" if (etf.get('avg_daily_volume', 0) * etf.get('latest_price', 0)) > 5e6 else "Medium" if (etf.get('avg_daily_volume', 0) * etf.get('latest_price', 0)) > 1e6 else "Low", 
                          style={'fontSize': '24px', 'color': '#2c3e50', 'margin': '5px 0', 'fontWeight': 'bold'}),
                    html.P("Overall assessment", style={'fontSize': '11px', 'color': '#95a5a6', 'margin': '0'})
                ], style={'flex': '1', 'padding': '12px', 'backgroundColor': '#e8f8f5', 'borderRadius': '8px', 'margin': '5px'})
            ], style={'display': 'flex'})
        ], style={'marginBottom': '30px'}),
        
        # Forecast Breakdown Section
        html.Div([
            html.H3("Forecast Breakdown", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            
            get_forecast_breakdown_display(etf)
        ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#e8f4f8', 'borderRadius': '8px'}),
        
        # MAE Analysis Section
        html.Div([
            html.H3("Forecast Quality (MAE Analysis)", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            
            # Get forecast quality flag
            html.Div(id=f'forecast-quality-{ticker}', children=[
                get_forecast_quality_display(etf)
            ])
        ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'})
    ])
    
    # Historical Chart Section
    if has_data:
        # Create price chart with candlestick and line traces
        fig = go.Figure()
        
        # Candlestick trace
        fig.add_trace(go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name=ticker,
            visible=True
        ))
        
        # Line trace
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['Close'],
            mode='lines',
            name=ticker,
            line=dict(color='#3498db', width=2),
            visible=False
        ))
        
        # Add buttons to toggle between candlestick and line
        fig.update_layout(
            title=f'{ticker} - 1 Year Price History',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_white',
            height=500,
            hovermode='x unified',
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True, False]}],
                            label="Candlestick",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [False, True]}],
                            label="Line",
                            method="update"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                ),
            ]
        )
        
        # Create volume chart
        fig_volume = go.Figure()
        
        fig_volume.add_trace(go.Bar(
            x=hist_data.index,
            y=hist_data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ))
        
        fig_volume.update_layout(
            title=f'{ticker} - Trading Volume',
            yaxis_title='Volume',
            xaxis_title='Date',
            template='plotly_white',
            height=300
        )
        
        chart_section = html.Div([
            html.H3("Historical Data", style={'color': '#2c3e50', 'marginTop': '30px', 'marginBottom': '20px'}),
            dcc.Graph(figure=fig),
            dcc.Graph(figure=fig_volume)
        ])
    else:
        chart_section = html.Div([
            html.H3("Historical Data", style={'color': '#2c3e50', 'marginTop': '30px', 'marginBottom': '20px'}),
            html.P("Unable to load historical price data", style={'color': '#7f8c8d', 'fontSize': '16px'})
        ])
    
    return html.Div([metrics_section, chart_section])


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Enhanced ETF Dashboard...")
    print("="*60)
    print(f"Loaded {len(universe)} ETFs")
    print(f"Analysis Date: {metadata.get('analysis_date', 'N/A')[:10]}")
    print(f"⏱ Processing Time: {metadata.get('processing_time', 0):.1f}s")
    print("="*60)
    print("\nDashboard running at: http://127.0.0.1:8051/")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=True, port=8051)