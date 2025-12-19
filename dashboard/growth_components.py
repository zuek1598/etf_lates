"""
Growth Strategy Dashboard Components
Additional pages and visualizations for growth-focused trading
"""

import dash
from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_growth_opportunities_page(universe: pd.DataFrame) -> html.Div:
    """
    Create a page showing top growth opportunities
    Focuses on MEDIUM/HIGH risk ETFs with strong momentum
    """
    # Filter for growth opportunities (MEDIUM/HIGH risk, score > 35)
    growth_candidates = universe[
        (universe['risk_category'].isin(['MEDIUM', 'HIGH'])) &
        (universe['composite_score'] > 35)
    ].copy()
    
    if len(growth_candidates) == 0:
        return html.Div([
            html.H2("Growth Opportunities", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            html.P("No opportunities found (score > 35, MEDIUM/HIGH risk)", 
                   style={'color': '#7f8c8d', 'fontSize': '16px'})
        ])
    
    # Sort by composite score
    growth_candidates = growth_candidates.sort_values('composite_score', ascending=False)
    
    # Calculate momentum score
    growth_candidates['momentum_score'] = (
        growth_candidates['kalman_signal_strength'] * 0.6 +
        growth_candidates['kalman_efficiency_ratio'] * 0.4
    ) * 100
    
    # Top opportunities
    top_10 = growth_candidates.head(10)
    
    return html.Div([
        # Header
        html.Div([
            html.H2("Top Growth Opportunities", style={'color': '#2c3e50', 'display': 'inline-block', 'marginRight': '20px'}),
            html.Span(f"{len(growth_candidates)} opportunities found", 
                     style={'color': '#7f8c8d', 'fontSize': '16px', 'verticalAlign': 'middle'})
        ], style={'marginBottom': '20px'}),
        
        # Strategy Info Banner
        html.Div([
            html.H4("Growth Strategy Focus", style={'color': '#2c3e50', 'marginBottom': '10px'}),
            html.P([
                "Targeting ",
                html.Strong("MEDIUM/HIGH risk ETFs"),
                " with ",
                html.Strong("Score > 35"),
                ", strong momentum, and positive forecasts. ",
                "Position sizing based on momentum strength (40-100% of base allocation)."
            ], style={'color': '#34495e', 'fontSize': '14px', 'lineHeight': '1.6'})
        ], style={'backgroundColor': '#e8f8f5', 'padding': '15px', 'borderRadius': '8px', 
                 'marginBottom': '20px', 'border': '2px solid #27ae60'}),
        
        # Summary Cards
        html.Div([
            # Card 1: Total Opportunities
            html.Div([
                html.H3(f"{len(growth_candidates)}", 
                       style={'color': '#27ae60', 'fontSize': '42px', 'margin': '0'}),
                html.P("Opportunities", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '5px 0'}),
                html.P("Score > 35", style={'color': '#95a5a6', 'fontSize': '11px', 'margin': '0'})
            ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'margin': '5px', 'textAlign': 'center'}),
            
            # Card 2: MEDIUM Risk
            html.Div([
                html.H3(f"{len(growth_candidates[growth_candidates['risk_category'] == 'MEDIUM'])}", 
                       style={'color': '#f39c12', 'fontSize': '42px', 'margin': '0'}),
                html.P("MEDIUM Risk", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '5px 0'}),
                html.P(f"Avg Score: {growth_candidates[growth_candidates['risk_category']=='MEDIUM']['composite_score'].mean():.1f}" if len(growth_candidates[growth_candidates['risk_category']=='MEDIUM']) > 0 else "N/A", 
                      style={'color': '#95a5a6', 'fontSize': '11px', 'margin': '0'})
            ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'margin': '5px', 'textAlign': 'center'}),
            
            # Card 3: HIGH Risk
            html.Div([
                html.H3(f"{len(growth_candidates[growth_candidates['risk_category'] == 'HIGH'])}", 
                       style={'color': '#e74c3c', 'fontSize': '42px', 'margin': '0'}),
                html.P("HIGH Risk", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '5px 0'}),
                html.P(f"Avg Score: {growth_candidates[growth_candidates['risk_category']=='HIGH']['composite_score'].mean():.1f}" if len(growth_candidates[growth_candidates['risk_category']=='HIGH']) > 0 else "N/A", 
                      style={'color': '#95a5a6', 'fontSize': '11px', 'margin': '0'})
            ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'margin': '5px', 'textAlign': 'center'}),
            
            # Card 4: Avg Momentum
            html.Div([
                html.H3(f"{growth_candidates['momentum_score'].mean():.0f}/100", 
                       style={'color': '#9b59b6', 'fontSize': '42px', 'margin': '0'}),
                html.P("Avg Momentum", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '5px 0'}),
                html.P("Signal + Efficiency", style={'color': '#95a5a6', 'fontSize': '11px', 'margin': '0'})
            ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'margin': '5px', 'textAlign': 'center'})
        ], style={'display': 'flex', 'marginBottom': '30px'}),
        
        # Top 10 Table
        html.Div([
            html.H3("Top 10 Opportunities", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Table([
                # Header
                html.Thead(html.Tr([
                    html.Th("#", style={'padding': '12px', 'textAlign': 'left', 'backgroundColor': '#34495e', 'color': 'white'}),
                    html.Th("Ticker", style={'padding': '12px', 'textAlign': 'left', 'backgroundColor': '#34495e', 'color': 'white'}),
                    html.Th("Score", style={'padding': '12px', 'textAlign': 'center', 'backgroundColor': '#34495e', 'color': 'white'}),
                    html.Th("Risk", style={'padding': '12px', 'textAlign': 'center', 'backgroundColor': '#34495e', 'color': 'white'}),
                    html.Th("Momentum", style={'padding': '12px', 'textAlign': 'center', 'backgroundColor': '#34495e', 'color': 'white'}),
                    html.Th("Forecast", style={'padding': '12px', 'textAlign': 'center', 'backgroundColor': '#34495e', 'color': 'white'}),
                    html.Th("Volume", style={'padding': '12px', 'textAlign': 'center', 'backgroundColor': '#34495e', 'color': 'white'}),
                    html.Th("Action", style={'padding': '12px', 'textAlign': 'center', 'backgroundColor': '#34495e', 'color': 'white'})
                ])),
                # Body
                html.Tbody([
                    html.Tr([
                        html.Td(i+1, style={'padding': '10px', 'fontWeight': 'bold', 'color': '#7f8c8d'}),
                        html.Td(
                            html.A(row['ticker'], href=f"#", 
                                  style={'color': '#3498db', 'textDecoration': 'none', 'fontWeight': 'bold'}),
                            style={'padding': '10px'}
                        ),
                        html.Td(f"{row['composite_score']:.1f}", 
                               style={'padding': '10px', 'textAlign': 'center', 'fontWeight': 'bold',
                                     'color': '#27ae60' if row['composite_score'] > 75 else '#f39c12'}),
                        html.Td(
                            html.Span(row['risk_category'], 
                                     style={'padding': '4px 8px', 'borderRadius': '4px', 'fontSize': '11px',
                                           'backgroundColor': '#f39c12' if row['risk_category']=='MEDIUM' else '#e74c3c',
                                           'color': 'white'}),
                            style={'padding': '10px', 'textAlign': 'center'}
                        ),
                        html.Td(f"{row['momentum_score']:.0f}/100", 
                               style={'padding': '10px', 'textAlign': 'center'}),
                        html.Td(f"{row['ml_forecast']:+.1f}%", 
                               style={'padding': '10px', 'textAlign': 'center',
                                     'color': '#27ae60' if row['ml_forecast'] > 0 else '#e74c3c'}),
                        html.Td(
                            get_volume_signal_icon(row['volume_ad_signal']),
                            style={'padding': '10px', 'textAlign': 'center'}
                        ),
                        html.Td(
                            html.Span("BUY" if row['kalman_trend'] == 1 else "⏸WAIT",
                                     style={'padding': '4px 8px', 'borderRadius': '4px', 'fontSize': '11px',
                                           'backgroundColor': '#27ae60' if row['kalman_trend']==1 else '#95a5a6',
                                           'color': 'white', 'fontWeight': 'bold'}),
                            style={'padding': '10px', 'textAlign': 'center'}
                        )
                    ], style={'backgroundColor': '#f8f9fa' if i % 2 == 0 else 'white',
                             'borderBottom': '1px solid #ecf0f1'})
                    for i, (_, row) in enumerate(top_10.iterrows())
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ], style={'marginBottom': '30px'}),
        
        # Scatter Plot: Score vs Momentum
        html.Div([
            html.H3("Score vs Momentum Strength", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            dcc.Graph(
                figure=create_score_momentum_scatter(growth_candidates),
                config={'displayModeBar': False}
            )
        ], style={'marginBottom': '30px'}),
        
        # Position Sizing Guide
        html.Div([
            html.H3("Position Sizing Guide", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Risk Category", style={'padding': '12px', 'backgroundColor': '#34495e', 'color': 'white'}),
                    html.Th("Base Size", style={'padding': '12px', 'backgroundColor': '#34495e', 'color': 'white'}),
                    html.Th("Strong Momentum (>75)", style={'padding': '12px', 'backgroundColor': '#34495e', 'color': 'white'}),
                    html.Th("Good Momentum (60-75)", style={'padding': '12px', 'backgroundColor': '#34495e', 'color': 'white'}),
                    html.Th("Weak Momentum (<45)", style={'padding': '12px', 'backgroundColor': '#34495e', 'color': 'white'})
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td("🟢 MEDIUM", style={'padding': '10px', 'fontWeight': 'bold'}),
                        html.Td("12%", style={'padding': '10px'}),
                        html.Td("12% (full)", style={'padding': '10px', 'color': '#27ae60', 'fontWeight': 'bold'}),
                        html.Td("10% (85%)", style={'padding': '10px'}),
                        html.Td("5% (40%)", style={'padding': '10px', 'color': '#e67e22'})
                    ]),
                    html.Tr([
                        html.Td("HIGH", style={'padding': '10px', 'fontWeight': 'bold'}),
                        html.Td("8%", style={'padding': '10px'}),
                        html.Td("8% (full)", style={'padding': '10px', 'color': '#27ae60', 'fontWeight': 'bold'}),
                        html.Td("7% (85%)", style={'padding': '10px'}),
                        html.Td("3% (40%)", style={'padding': '10px', 'color': '#e67e22'})
                    ], style={'backgroundColor': '#f8f9fa'})
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ])
    ])


def get_volume_signal_icon(signal: str) -> str:
    """Get icon for volume signal"""
    if signal == 'accumulation':
        return "ACC"
    elif signal == 'distribution':
        return "DIST"
    else:
        return "NEUT"


def create_score_momentum_scatter(data: pd.DataFrame) -> go.Figure:
    """Create scatter plot of score vs momentum"""
    fig = px.scatter(
        data,
        x='momentum_score',
        y='composite_score',
        color='risk_category',
        hover_data=['ticker', 'ml_forecast', 'volume_ad_signal'],
        labels={'momentum_score': 'Momentum Score', 'composite_score': 'Composite Score'},
        color_discrete_map={'MEDIUM': '#f39c12', 'HIGH': '#e74c3c'}
    )
    
    # Add threshold lines
    fig.add_hline(y=60, line_dash="dash", line_color="gray", annotation_text="Min Score (60)")
    fig.add_vline(x=60, line_dash="dash", line_color="gray", annotation_text="Good Momentum (60)")
    fig.add_vline(x=75, line_dash="dash", line_color="green", annotation_text="Strong Momentum (75)")
    
    fig.update_layout(
        height=500,
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='#f8f9fa'
    )
    
    return fig


def create_backtest_results_page(results_file: str = 'data/backtest_results.parquet') -> html.Div:
    """
    Display backtest results if available
    """
    from pathlib import Path
    
    if not Path(results_file).exists():
        return html.Div([
            html.H2("Backtest Results", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            html.P("No backtest results available. Run: python3 run_backtest.py", 
                   style={'color': '#7f8c8d', 'fontSize': '16px'})
        ])
    
    try:
        results = pd.read_parquet(results_file)
    except:
        return html.Div([
            html.H2("Backtest Results", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            html.P("Error loading backtest results", 
                   style={'color': '#e74c3c', 'fontSize': '16px'})
        ])
    
    # Calculate summary stats
    avg_excess = results['excess_return'].mean()
    avg_sharpe = results['sharpe_ratio'].mean()
    avg_max_dd = results['max_drawdown'].mean()
    avg_win_rate = results['win_rate'].mean()
    
    # Count winners
    winners = len(results[results['excess_return'] > 0])
    total = len(results)
    
    return html.Div([
        html.H2("Backtest Results", style={'color': '#2c3e50', 'marginBottom': '20px'}),
        
        # Summary Cards
        html.Div([
            html.Div([
                html.H3(f"{winners}/{total}", 
                       style={'color': '#27ae60', 'fontSize': '42px', 'margin': '0'}),
                html.P("Beat Buy & Hold", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '5px 0'}),
                html.P(f"{winners/total*100:.0f}% win rate", 
                      style={'color': '#95a5a6', 'fontSize': '11px', 'margin': '0'})
            ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'margin': '5px', 'textAlign': 'center'}),
            
            html.Div([
                html.H3(f"{avg_excess:+.1%}", 
                       style={'color': '#3498db', 'fontSize': '42px', 'margin': '0'}),
                html.P("Avg Excess Return", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '5px 0'}),
                html.P("vs Buy & Hold", style={'color': '#95a5a6', 'fontSize': '11px', 'margin': '0'})
            ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'margin': '5px', 'textAlign': 'center'}),
            
            html.Div([
                html.H3(f"{avg_sharpe:.2f}", 
                       style={'color': '#9b59b6', 'fontSize': '42px', 'margin': '0'}),
                html.P("Avg Sharpe Ratio", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '5px 0'}),
                html.P("Risk-adjusted", style={'color': '#95a5a6', 'fontSize': '11px', 'margin': '0'})
            ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'margin': '5px', 'textAlign': 'center'}),
            
            html.Div([
                html.H3(f"{avg_max_dd:.1%}", 
                       style={'color': '#e74c3c', 'fontSize': '42px', 'margin': '0'}),
                html.P("Avg Max Drawdown", style={'color': '#7f8c8d', 'fontSize': '14px', 'margin': '5px 0'}),
                html.P("Peak to trough", style={'color': '#95a5a6', 'fontSize': '11px', 'margin': '0'})
            ], style={'flex': '1', 'padding': '15px', 'backgroundColor': '#ecf0f1', 
                     'borderRadius': '8px', 'margin': '5px', 'textAlign': 'center'})
        ], style={'display': 'flex', 'marginBottom': '30px'}),
        
        # Results table
        html.Div([
            html.H3("Detailed Results", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th(col, style={'padding': '12px', 'backgroundColor': '#34495e', 'color': 'white'})
                    for col in ['Ticker', 'Risk', 'Strategy', 'Buy&Hold', 'Excess', 'Sharpe', 'Max DD', 'Trades', 'Win%']
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(row['ticker'], style={'padding': '10px', 'fontWeight': 'bold'}),
                        html.Td(row['risk_category'], style={'padding': '10px'}),
                        html.Td(f"{row['total_return']:+.1%}", 
                               style={'padding': '10px', 'color': '#27ae60' if row['total_return']>0 else '#e74c3c'}),
                        html.Td(f"{row['benchmark_return']:+.1%}", style={'padding': '10px'}),
                        html.Td(f"{row['excess_return']:+.1%}", 
                               style={'padding': '10px', 'fontWeight': 'bold',
                                     'color': '#27ae60' if row['excess_return']>0 else '#e74c3c'}),
                        html.Td(f"{row['sharpe_ratio']:.2f}", style={'padding': '10px'}),
                        html.Td(f"{row['max_drawdown']:.1%}", style={'padding': '10px', 'color': '#e74c3c'}),
                        html.Td(f"{row['num_trades']:.0f}", style={'padding': '10px'}),
                        html.Td(f"{row['win_rate']:.0%}", style={'padding': '10px'})
                    ], style={'backgroundColor': '#f8f9fa' if i%2==0 else 'white'})
                    for i, (_, row) in enumerate(results.iterrows())
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ])
    ])

