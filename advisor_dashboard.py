#!/usr/bin/env python3
"""
Advisor Dashboard

This script creates a simple dashboard for monitoring trading signals and performance.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger("advisor_dashboard")

def load_signals(signals_file):
    """Load signals from JSON file."""
    with open(signals_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'signals' in data:
        signals = data['signals']
    else:
        signals = data
    
    return signals

def load_backtest_results(results_file):
    """Load backtest results from JSON file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def create_dashboard(signals, backtest_results, port=8050, host='0.0.0.0'):
    """Create and run the dashboard."""
    # Convert signals to DataFrame
    signals_df = pd.DataFrame(signals)
    
    # Create app
    app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
    
    # Define layout
    app.layout = html.Div([
        html.H1("Forex Signal Trading System - Advisor Dashboard"),
        
        html.Div([
            html.Div([
                html.H3("Performance Metrics"),
                html.Div([
                    html.Div([
                        html.H4("Total Return"),
                        html.H2(f"{backtest_results['total_return'] * 100:.2f}%", 
                               style={'color': 'green' if backtest_results['total_return'] > 0 else 'red'})
                    ], className="three columns"),
                    
                    html.Div([
                        html.H4("Win Rate"),
                        html.H2(f"{backtest_results['win_rate'] * 100:.2f}%")
                    ], className="three columns"),
                    
                    html.Div([
                        html.H4("Profit Factor"),
                        html.H2(f"{backtest_results['profit_factor']:.2f}")
                    ], className="three columns"),
                    
                    html.Div([
                        html.H4("Max Drawdown"),
                        html.H2(f"{backtest_results['max_drawdown'] * 100:.2f}%",
                               style={'color': 'red'})
                    ], className="three columns"),
                ], className="row"),
                
                html.Div([
                    html.Div([
                        html.H4("TFT Compliance"),
                        html.Div([
                            html.P(f"Daily Loss Limit (1.5%): {'PASS' if backtest_results['max_drawdown'] <= 0.015 else 'FAIL'}",
                                  style={'color': 'green' if backtest_results['max_drawdown'] <= 0.015 else 'red'}),
                            html.P(f"Max Drawdown Limit (4%): {'PASS' if backtest_results['max_drawdown'] <= 0.04 else 'FAIL'}",
                                  style={'color': 'green' if backtest_results['max_drawdown'] <= 0.04 else 'red'}),
                            html.P(f"Profit Target (8%): {'PASS' if backtest_results['total_return'] >= 0.08 else 'FAIL'}",
                                  style={'color': 'green' if backtest_results['total_return'] >= 0.08 else 'red'})
                        ])
                    ], className="six columns"),
                    
                    html.Div([
                        html.H4("Trading Statistics"),
                        html.Div([
                            html.P(f"Total Trades: {backtest_results['total_trades']}"),
                            html.P(f"Winning Trades: {backtest_results['winning_trades']}"),
                            html.P(f"Losing Trades: {backtest_results['losing_trades']}")
                        ])
                    ], className="six columns")
                ], className="row")
            ], className="six columns"),
            
            html.Div([
                html.H3("Equity Curve"),
                dcc.Graph(
                    id='equity-curve',
                    figure={
                        'data': [
                            {'x': list(range(len(backtest_results['equity_curve']))), 
                             'y': backtest_results['equity_curve'], 
                             'type': 'line', 
                             'name': 'Equity'}
                        ],
                        'layout': {
                            'title': 'Equity Curve',
                            'xaxis': {'title': 'Trade Number'},
                            'yaxis': {'title': 'Equity'}
                        }
                    }
                )
            ], className="six columns")
        ], className="row"),
        
        html.Div([
            html.Div([
                html.H3("Drawdown Curve"),
                dcc.Graph(
                    id='drawdown-curve',
                    figure={
                        'data': [
                            {'x': list(range(len(backtest_results['drawdown_curve']))), 
                             'y': [d * 100 for d in backtest_results['drawdown_curve']], 
                             'type': 'line', 
                             'name': 'Drawdown',
                             'fill': 'tozeroy',
                             'line': {'color': 'red'}}
                        ],
                        'layout': {
                            'title': 'Drawdown Curve',
                            'xaxis': {'title': 'Trade Number'},
                            'yaxis': {'title': 'Drawdown (%)'}
                        }
                    }
                )
            ], className="six columns"),
            
            html.Div([
                html.H3("Exit Reasons"),
                dcc.Graph(
                    id='exit-reasons',
                    figure={
                        'data': [
                            {'x': list(backtest_results['exit_reasons'].keys()), 
                             'y': list(backtest_results['exit_reasons'].values()), 
                             'type': 'bar', 
                             'name': 'Exit Reasons'}
                        ],
                        'layout': {
                            'title': 'Exit Reasons',
                            'xaxis': {'title': 'Reason'},
                            'yaxis': {'title': 'Count'}
                        }
                    }
                )
            ], className="six columns")
        ], className="row"),
        
        html.Div([
            html.H3("Signal Quality Distribution"),
            dcc.Graph(
                id='quality-distribution',
                figure={
                    'data': [
                        {'x': signals_df['quality_score'], 
                         'type': 'histogram', 
                         'nbinsx': 20,
                         'name': 'Quality Score'}
                    ],
                    'layout': {
                        'title': 'Signal Quality Distribution',
                        'xaxis': {'title': 'Quality Score'},
                        'yaxis': {'title': 'Count'}
                    }
                }
            )
        ], className="row"),
        
        html.Div([
            html.H3("Recent Signals"),
            dash_table.DataTable(
                id='signals-table',
                columns=[
                    {'name': 'ID', 'id': 'id'},
                    {'name': 'Asset', 'id': 'asset'},
                    {'name': 'Direction', 'id': 'direction'},
                    {'name': 'Entry', 'id': 'entry'},
                    {'name': 'Stop Loss', 'id': 'stop_loss'},
                    {'name': 'Take Profit', 'id': 'take_profit'},
                    {'name': 'Quality Score', 'id': 'quality_score'},
                    {'name': 'Timestamp', 'id': 'timestamp'}
                ],
                data=signals_df.sort_values('timestamp', ascending=False).head(10).to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
        ], className="row"),
        
        html.Div([
            html.H3("Currency Pair Performance"),
            dcc.Dropdown(
                id='pair-dropdown',
                options=[{'label': pair, 'value': pair} for pair in signals_df['asset'].unique()],
                value=signals_df['asset'].iloc[0] if not signals_df.empty else None,
                style={'width': '50%'}
            ),
            dcc.Graph(id='pair-performance')
        ], className="row"),
        
        html.Footer([
            html.P("Forex Signal Trading System - Advisor Dashboard"),
            html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        ], style={'textAlign': 'center', 'marginTop': '50px'})
    ])
    
    @app.callback(
        Output('pair-performance', 'figure'),
        [Input('pair-dropdown', 'value')]
    )
    def update_pair_performance(selected_pair):
        if not selected_pair:
            return {}
        
        # Filter signals for selected pair
        pair_signals = signals_df[signals_df['asset'] == selected_pair]
        
        # Create figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add entry prices
        fig.add_trace(
            go.Scatter(
                x=pair_signals['timestamp'],
                y=pair_signals['entry'],
                mode='markers',
                name='Entry Price',
                marker=dict(
                    size=10,
                    color='blue'
                )
            ),
            secondary_y=False
        )
        
        # Add stop loss prices
        fig.add_trace(
            go.Scatter(
                x=pair_signals['timestamp'],
                y=pair_signals['stop_loss'],
                mode='markers',
                name='Stop Loss',
                marker=dict(
                    size=8,
                    color='red'
                )
            ),
            secondary_y=False
        )
        
        # Add take profit prices
        fig.add_trace(
            go.Scatter(
                x=pair_signals['timestamp'],
                y=pair_signals['take_profit'],
                mode='markers',
                name='Take Profit',
                marker=dict(
                    size=8,
                    color='green'
                )
            ),
            secondary_y=False
        )
        
        # Add quality scores
        fig.add_trace(
            go.Scatter(
                x=pair_signals['timestamp'],
                y=pair_signals['quality_score'],
                mode='lines+markers',
                name='Quality Score',
                marker=dict(
                    size=6,
                    color='purple'
                ),
                line=dict(
                    width=1,
                    dash='dot'
                )
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=f"Performance for {selected_pair}",
            xaxis_title="Timestamp",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="Quality Score", secondary_y=True)
        
        return fig
    
    # Run server
    app.run(debug=True, port=port, host=host)

def main():
    parser = argparse.ArgumentParser(description='Run advisor dashboard for Forex Signal Trading System')
    parser.add_argument('--signals-file', type=str, default='data/pipeline_output_real/filtered/filtered_signals.json',
                        help='Path to the filtered and enriched signals JSON file')
    parser.add_argument('--backtest-results', type=str, default='data/backtest_results/backtest_results.json',
                        help='Path to the backtest results JSON file')
    parser.add_argument('--port', type=int, default=12000,
                        help='Port to run the dashboard on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the dashboard on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Load signals
    logger.info(f"Loading signals from {args.signals_file}")
    try:
        signals = load_signals(args.signals_file)
        logger.info(f"Loaded {len(signals)} signals")
    except FileNotFoundError:
        logger.error(f"Signals file not found: {args.signals_file}")
        signals = []
    
    # Load backtest results
    logger.info(f"Loading backtest results from {args.backtest_results}")
    try:
        backtest_results = load_backtest_results(args.backtest_results)
        logger.info("Loaded backtest results")
        
        # Map new keys to old keys for compatibility
        if 'net_profit_pct' in backtest_results and 'total_return' not in backtest_results:
            backtest_results['total_return'] = backtest_results['net_profit_pct']
        
        if 'final_capital' in backtest_results and 'final_equity' not in backtest_results:
            backtest_results['final_equity'] = backtest_results['final_capital']
        
        # Create exit_reasons if not present
        if 'exit_reasons' not in backtest_results:
            exit_reasons = {}
            for trade in backtest_results.get('trades', []):
                reason = trade.get('exit_reason', 'unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            backtest_results['exit_reasons'] = exit_reasons
        
        # Create equity_curve and drawdown_curve if not in expected format
        if 'equity_curve' in backtest_results and isinstance(backtest_results['equity_curve'], list) and isinstance(backtest_results['equity_curve'][0], dict):
            equity_values = [float(point['capital']) for point in backtest_results['equity_curve']]
            backtest_results['equity_curve'] = equity_values
            
            # Calculate drawdown curve
            peak = backtest_results['initial_capital']
            drawdown_curve = []
            for equity in equity_values:
                peak = max(peak, equity)
                drawdown = (peak - equity) / peak if peak > 0 else 0
                drawdown_curve.append(drawdown)
            backtest_results['drawdown_curve'] = drawdown_curve
        
    except FileNotFoundError:
        logger.error(f"Backtest results file not found: {args.backtest_results}")
        backtest_results = {
            'initial_capital': 10000,
            'final_equity': 10000,
            'total_return': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'equity_curve': [10000],
            'drawdown_curve': [0],
            'exit_reasons': {}
        }
    
    # Create dashboard
    logger.info(f"Creating dashboard on {args.host}:{args.port}")
    create_dashboard(signals, backtest_results, args.port, args.host)

if __name__ == "__main__":
    main()