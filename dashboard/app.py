"""
Lightweight dashboard for the AI-Powered API Testing System.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

# Initialize Dash app
app = dash.Dash(__name__, title="AI-Powered API Testing Dashboard")

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Layout
app.layout = html.Div([
    html.H1("AI-Powered API Testing System Dashboard", className="header"),
    
    html.Div([
        html.Div([
            html.H3("System Overview"),
            html.Div(id="system-stats")
        ], className="card"),
        
        html.Div([
            html.H3("Coverage Metrics"),
            dcc.Graph(id="coverage-chart")
        ], className="card"),
        
        html.Div([
            html.H3("Test Execution Trends"),
            dcc.Graph(id="execution-chart")
        ], className="card"),
        
        html.Div([
            html.H3("RL Performance"),
            dcc.Graph(id="rl-chart")
        ], className="card")
    ], className="dashboard-container"),
    
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # Update every 30 seconds
        n_intervals=0
    )
])

@callback(
    Output('system-stats', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_system_stats(n):
    try:
        response = requests.get(f"{API_BASE_URL}/../status")
        if response.status_code == 200:
            data = response.json()
            stats = data.get('statistics', {})
            
            return html.Div([
                html.P(f"API Specifications: {stats.get('api_specifications', 0)}"),
                html.P(f"Test Cases: {stats.get('test_cases', 0)}"),
                html.P(f"Execution Sessions: {stats.get('execution_sessions', 0)}"),
                html.P(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
            ])
    except:
        return html.P("Unable to fetch system statistics")

@callback(
    Output('coverage-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_coverage_chart(n):
    # Mock data for demonstration
    data = {
        'Metric': ['Endpoint Coverage', 'Method Coverage', 'Response Code Coverage'],
        'Percentage': [75, 80, 65]
    }
    
    fig = px.bar(data, x='Metric', y='Percentage', 
                 title="API Coverage Metrics",
                 color='Percentage',
                 color_continuous_scale='viridis')
    
    fig.update_layout(
        xaxis_title="Coverage Type",
        yaxis_title="Percentage (%)",
        showlegend=False
    )
    
    return fig

@callback(
    Output('execution-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_execution_chart(n):
    # Mock time series data
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                         end=datetime.now(), freq='D')
    
    data = pd.DataFrame({
        'Date': dates,
        'Passed Tests': [45, 52, 48, 61, 58, 67, 72],
        'Failed Tests': [5, 8, 12, 9, 7, 3, 8]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Passed Tests'],
        mode='lines+markers', name='Passed Tests',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Failed Tests'],
        mode='lines+markers', name='Failed Tests',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="Test Execution Trends (Last 7 Days)",
        xaxis_title="Date",
        yaxis_title="Number of Tests",
        hovermode='x unified'
    )
    
    return fig

@callback(
    Output('rl-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_rl_chart(n):
    # Mock RL performance data
    episodes = list(range(0, 1001, 100))
    rewards = [0.2, 0.35, 0.48, 0.52, 0.61, 0.68, 0.72, 0.75, 0.78, 0.82, 0.85]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=episodes, y=rewards,
        mode='lines+markers',
        name='Average Reward',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title="RL Model Learning Curve",
        xaxis_title="Training Episodes",
        yaxis_title="Average Reward",
        showlegend=True
    )
    
    return fig

# Add CSS styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                text-align: center;
                color: #2c3e50;
                margin-bottom: 30px;
            }
            .dashboard-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
            }
            .card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .card h3 {
                margin-top: 0;
                color: #34495e;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
