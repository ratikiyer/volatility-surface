import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import warnings
import re

from data_fetch import get_options_data
from volatility_calc import (
    implied_volatility,
    calculate_implied_volatility_with_market_data,
    validate_implied_volatility,
)
from arbitrage import detect_arbitrage

app = dash.Dash(__name__)
app.title = "Options Volatility Surface"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-color: #6366f1;
                --primary-hover: #4f46e5;
                --primary-light: #818cf8;
                --success-color: #10b981;
                --warning-color: #f59e0b;
                --danger-color: #ef4444;
                --bg-primary: #ffffff;
                --bg-secondary: #f8fafc;
                --bg-tertiary: #f1f5f9;
                --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --text-primary: #1e293b;
                --text-secondary: #64748b;
                --text-muted: #94a3b8;
                --border-color: #e2e8f0;
                --shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
                --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
                --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
                --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 10px 10px -5px rgb(0 0 0 / 0.04);
                --radius: 16px;
                --radius-sm: 12px;
                --radius-lg: 24px;
            }

            [data-theme="dark"] {
                --primary-color: #818cf8;
                --primary-hover: #6366f1;
                --primary-light: #a5b4fc;
                --bg-primary: #0f172a;
                --bg-secondary: #1e293b;
                --bg-tertiary: #334155;
                --bg-gradient: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                --text-primary: #f8fafc;
                --text-secondary: #cbd5e1;
                --text-muted: #64748b;
                --border-color: #334155;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: var(--bg-secondary);
                color: var(--text-primary);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                line-height: 1.6;
            }

            .app-container {
                min-height: 100vh;
                background: var(--bg-secondary);
            }

            .header {
                background: var(--bg-primary);
                border-bottom: 1px solid var(--border-color);
                padding: 1.5rem 2rem;
                box-shadow: var(--shadow-lg);
                position: sticky;
                top: 0;
                z-index: 100;
                backdrop-filter: blur(10px);
                background: rgba(255, 255, 255, 0.95);
            }

            [data-theme="dark"] .header {
                background: rgba(15, 23, 42, 0.95);
            }

            .header-content {
                max-width: 1400px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .logo {
                display: flex;
                align-items: center;
                gap: 1rem;
                font-size: 1.75rem;
                font-weight: 800;
                color: var(--text-primary);
                text-decoration: none;
                transition: all 0.3s ease;
            }

            .logo:hover {
                transform: translateY(-2px);
            }

            .logo-icon {
                width: 48px;
                height: 48px;
                background: var(--bg-gradient);
                border-radius: var(--radius);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 1.5rem;
                box-shadow: var(--shadow-lg);
                transition: all 0.3s ease;
            }

            .logo-icon:hover {
                transform: scale(1.05) rotate(5deg);
                box-shadow: var(--shadow-xl);
            }

            .theme-toggle {
                background: var(--bg-tertiary);
                border: 2px solid var(--border-color);
                border-radius: var(--radius);
                padding: 0.75rem;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                color: var(--text-secondary);
                font-size: 1.25rem;
                width: 56px;
                height: 56px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .theme-toggle:hover {
                background: var(--primary-color);
                border-color: var(--primary-color);
                color: white;
                transform: translateY(-2px) scale(1.05);
                box-shadow: var(--shadow-lg);
            }

            .main-content {
                max-width: 1400px;
                margin: 0 auto;
                padding: 2rem;
            }

            .controls-card {
                background: var(--bg-primary);
                border-radius: var(--radius-lg);
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: var(--shadow-xl);
                border: 1px solid var(--border-color);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }

            .controls-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 25px 50px -12px rgb(0 0 0 / 0.25);
            }

            .controls-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 2rem;
                align-items: end;
            }

            .input-group {
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
            }

            .input-label {
                font-size: 0.875rem;
                font-weight: 700;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-bottom: 0.25rem;
            }

            .input-field {
                padding: 1rem 1.25rem;
                border: 2px solid var(--border-color);
                border-radius: var(--radius-sm);
                background: var(--bg-primary);
                color: var(--text-primary);
                font-size: 1rem;
                font-weight: 500;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                outline: none;
                box-shadow: var(--shadow-sm);
            }

            .input-field:focus {
                border-color: var(--primary-color);
                box-shadow: 0 0 0 4px rgb(99 102 241 / 0.1);
                transform: translateY(-1px);
            }

            .input-field:hover {
                border-color: var(--primary-light);
                transform: translateY(-1px);
            }

            .radio-group {
                display: flex;
                gap: 1.5rem;
                align-items: center;
                flex-wrap: wrap;
            }

            .radio-item {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                cursor: pointer;
                padding: 0.5rem 1rem;
                border-radius: var(--radius-sm);
                transition: all 0.2s ease;
                font-weight: 500;
            }

            .radio-item:hover {
                background: var(--bg-tertiary);
            }

            .radio-input {
                accent-color: var(--primary-color);
                width: 1.25rem;
                height: 1.25rem;
            }

            .btn-primary {
                background: var(--bg-gradient);
                color: white;
                border: none;
                padding: 1rem 2.5rem;
                border-radius: var(--radius-sm);
                font-weight: 700;
                font-size: 1rem;
                cursor: pointer;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: var(--shadow-lg);
                text-transform: uppercase;
                letter-spacing: 0.05em;
                position: relative;
                overflow: hidden;
            }

            .btn-primary::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }

            .btn-primary:hover::before {
                left: 100%;
            }

            .btn-primary:hover {
                transform: translateY(-3px);
                box-shadow: var(--shadow-xl);
            }

            .btn-primary:active {
                transform: translateY(-1px);
            }

            .graph-card {
                background: var(--bg-primary);
                border-radius: var(--radius-lg);
                padding: 2rem;
                box-shadow: var(--shadow-xl);
                border: 1px solid var(--border-color);
                margin-bottom: 2rem;
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }

            .graph-card:hover {
                transform: translateY(-2px);
            }

            .graph-title {
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding-bottom: 1rem;
                border-bottom: 2px solid var(--border-color);
            }

            .arbitrage-card {
                background: var(--bg-primary);
                border-radius: var(--radius-lg);
                padding: 2rem;
                box-shadow: var(--shadow-xl);
                border: 1px solid var(--border-color);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
            }

            .arbitrage-card:hover {
                transform: translateY(-2px);
            }

            .arbitrage-title {
                font-size: 1.25rem;
                font-weight: 700;
                color: var(--text-primary);
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding-bottom: 1rem;
                border-bottom: 2px solid var(--border-color);
            }

            .arbitrage-content {
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
                font-size: 0.875rem;
                line-height: 1.7;
                color: var(--text-secondary);
                background: var(--bg-tertiary);
                padding: 1.5rem;
                border-radius: var(--radius-sm);
                border: 1px solid var(--border-color);
                box-shadow: var(--shadow-sm);
            }

            .status-indicator {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.5rem 1rem;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                box-shadow: var(--shadow-sm);
                animation: pulse 2s infinite;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.8; }
            }

            .status-success {
                background: linear-gradient(135deg, var(--success-color), #059669);
                color: white;
            }

            .status-warning {
                background: linear-gradient(135deg, var(--warning-color), #d97706);
                color: white;
            }

            .loading-spinner {
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 3rem;
            }

            .spinner {
                width: 48px;
                height: 48px;
                border: 4px solid var(--border-color);
                border-top: 4px solid var(--primary-color);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            /* Enhanced animations */
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .controls-card,
            .graph-card,
            .arbitrage-card {
                animation: fadeInUp 0.6s ease-out;
            }

            .graph-card {
                animation-delay: 0.1s;
            }

            .arbitrage-card {
                animation-delay: 0.2s;
            }

            /* Responsive design */
            @media (max-width: 768px) {
                .header-content {
                    flex-direction: column;
                    gap: 1rem;
                }
                
                .controls-grid {
                    grid-template-columns: 1fr;
                }
                
                .main-content {
                    padding: 1rem;
                }

                .logo {
                    font-size: 1.5rem;
                }

                .logo-icon {
                    width: 40px;
                    height: 40px;
                    font-size: 1.25rem;
                }

                .controls-card,
                .graph-card,
                .arbitrage-card {
                    padding: 1.5rem;
                }
            }

            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }

            ::-webkit-scrollbar-track {
                background: var(--bg-tertiary);
            }

            ::-webkit-scrollbar-thumb {
                background: var(--primary-color);
                border-radius: 4px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: var(--primary-hover);
            }
        </style>
        <script>
            // Theme management
            function toggleTheme() {
                const html = document.documentElement;
                const currentTheme = html.getAttribute('data-theme');
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                html.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
            }

            // Initialize theme from localStorage
            document.addEventListener('DOMContentLoaded', function() {
                const savedTheme = localStorage.getItem('theme') || 'light';
                document.documentElement.setAttribute('data-theme', savedTheme);
                
                // Listen for theme toggle clicks
                document.addEventListener('click', function(e) {
                    if (e.target && e.target.id === 'theme-toggle') {
                        toggleTheme();
                    }
                });
            });
        </script>
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

app.layout = html.Div([
    html.Header(className="header", children=[
        html.Div(className="header-content", children=[
            html.Div(className="logo", children=[
                html.Div(className="logo-icon", children="📊"),
                "Volatility Surface Visualizer"
            ]),
            html.Button(
                id="theme-toggle",
                className="theme-toggle",
                children="🌙",
                title="Toggle dark mode"
            )
        ])
    ]),
    dcc.Store(id="theme-store", data=True),
    html.Main(className="main-content", children=[
        html.Div(className="controls-card", children=[
            html.H2(style={
                'fontSize': '1.5rem',
                'fontWeight': '700',
                'color': 'var(--text-primary)',
                'marginBottom': '1.5rem',
                'display': 'flex',
                'alignItems': 'center',
                'gap': '0.75rem'
            }, children=[
                "⚙️ Configuration Panel"
            ]),
            html.Div(className="controls-grid", children=[
                html.Div(className="input-group", children=[
                    html.Label("STOCK TICKER", className="input-label"),
                    dcc.Input(
                        id='input-ticker',
                        type='text',
                        value='SPY',
                        placeholder='Enter ticker symbol...',
                        className="input-field"
                    )
                ]),
                html.Div(className="input-group", children=[
                    html.Label("RISK-FREE RATE (%)", className="input-label"),
                    dcc.Input(
                        id='input-rfr',
                        type='number',
                        value=4.725,
                        step=0.025,
                        placeholder='4.725',
                        className="input-field"
                    )
                ]),
                html.Div(className="input-group", children=[
                    html.Label("Y-AXIS DISPLAY", className="input-label"),
                    html.Div(
                        id="y-axis-toggle-container",
                        style={"display": "flex", "alignItems": "center", "gap": "1rem"},
                        children=[
                            html.Button(
                                id="y-axis-toggle",
                                n_clicks=0,
                                children="Strike",
                                className="toggle-btn toggle-btn-strike",
                                style={
                                    "padding": "0.5rem 1.5rem",
                                    "borderRadius": "999px",
                                    "border": "2px solid var(--primary-color)",
                                    "background": "var(--primary-color)",
                                    "color": "#fff",
                                    "fontWeight": "700",
                                    "fontSize": "1rem",
                                    "cursor": "pointer",
                                    "transition": "all 0.2s",
                                }
                            )
                        ]
                    ),
                    dcc.Store(id="y-axis-toggle-store", data="strike")
                ]),
                html.Div(className="input-group", children=[
                    html.Label("", className="input-label"),
                    html.Button(
                        'UPDATE SURFACE',
                        id='update-button',
                        n_clicks=0,
                        className="btn-primary"
                    )
                ])
            ])
        ]),
        html.Div(className="graph-card", children=[
            html.H3(className="graph-title", children=[
                "📊 Volatility Surface",
                html.Span(id="status-indicator", className="status-indicator status-success", children="Ready")
            ]),
            dcc.Loading(
                id="loading-graph",
                type="default",
                children=[
                    dcc.Graph(
                        id='vol-surface-plot', 
                        style={'height': '700px'},
                        config={
                            "scrollZoom": False,
                            "doubleClick": False,
                            "displayModeBar": False,
                            "editable": False,
                            "responsive": False
                        }
                    )
                ]
            )
        ]),
        html.Div(className="arbitrage-card", children=[
            html.H3(className="arbitrage-title", children=[
                "🔍 Arbitrage Detection",
                html.Span(id="arbitrage-status", className="status-indicator status-success", children="Clean")
            ]),
            html.Div(
                id='arbitrage-messages',
                className="arbitrage-content",
                children=[]
            )
        ])
    ])
])

@app.callback(
    Output("theme-store", "data"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True
)
def toggle_theme_store(n_clicks, current_theme):
    return not current_theme if current_theme is not None else True

@app.callback(
    Output("theme-toggle", "children"),
    Input("theme-store", "data")
)
def update_theme_emoji(is_dark):
    return "☀️" if is_dark else "🌙"

def get_toggle_label(value):
    return "Strike" if value == "strike" else "Moneyness"

@app.callback(
    Output("y-axis-toggle", "children"),
    Output("y-axis-toggle-store", "data"),
    Input("y-axis-toggle", "n_clicks"),
    State("y-axis-toggle-store", "data"),
    prevent_initial_call=True
)
def toggle_y_axis(n_clicks, current):
    if n_clicks is None:
        return get_toggle_label(current), current
    new_value = "moneyness" if current == "strike" else "strike"
    return get_toggle_label(new_value), new_value

@app.callback(
    Output('vol-surface-plot', 'figure'),
    Output('arbitrage-messages', 'children'),
    Output('status-indicator', 'children'),
    Output('status-indicator', 'className'),
    Output('arbitrage-status', 'children'),
    Output('arbitrage-status', 'className'),
    Input('update-button', 'n_clicks'),
    Input('theme-store', 'data'),
    State('input-ticker', 'value'),
    State('input-rfr', 'value'),
    State('y-axis-toggle-store', 'data')
)
def update_surface(n_clicks, is_dark, ticker, rfr_percentage, axis_scale):
    """Fetch data, compute IVs, build the surface, detect arbitrage."""
    if not ticker:
        return (
            dash.no_update,
            "Please enter a valid ticker symbol.",
            "Error",
            "status-indicator status-warning",
            "Error",
            "status-indicator status-warning",
        )


    rfr = 4.725
    if rfr_percentage is not None:
        rfr = float(rfr_percentage) / 100.0


    try:
        options_df, spot_price = get_options_data(ticker)
    except Exception as e:
        return (
            dash.no_update,
            f"Error fetching data for {ticker}: {e}",
            "Error",
            "status-indicator status-warning",
            "Error",
            "status-indicator status-warning",
        )

    if options_df.empty:
        return (
            dash.no_update,
            f"No options data available for {ticker}.",
            "No Data",
            "status-indicator status-warning",
            "No Data",
            "status-indicator status-warning",
        )


    calls = options_df.copy()
    calls = calculate_implied_volatility_with_market_data(calls, ticker)

    iv_issues = validate_implied_volatility(calls)
    if iv_issues:
        print("IV Calculation Issues:", iv_issues)


    lower_strike = 0.5 * spot_price
    upper_strike = 1.5 * spot_price
    calls = calls[(calls['strike'] >= lower_strike) & (calls['strike'] <= upper_strike)]

    calls = calls.dropna(subset=['imp_vol'])
    if calls.empty:
        return (
            dash.no_update,
            "Implied volatility calculation failed for all options.",
            "Error",
            "status-indicator status-warning",
            "Error",
            "status-indicator status-warning",
        )


    unique_expiries = np.sort(calls['days_to_expiry'].unique())
    min_strike = calls.groupby('days_to_expiry')['strike'].min().max()
    max_strike = calls.groupby('days_to_expiry')['strike'].max().min()
    strike_values = np.linspace(min_strike, max_strike, num=120)

    y_axis_label = "Strike Price"
    y_vals = strike_values
    if axis_scale == 'moneyness':
        y_axis_label = "Moneyness (K/S)"
        y_vals = strike_values / spot_price


    points = calls[['days_to_expiry', 'strike']].values
    values = calls['imp_vol'].values
    grid_x, grid_y = np.meshgrid(unique_expiries, strike_values)


    try:
        surface_matrix = griddata(points, values, (grid_x, grid_y), method='cubic')
    except Exception:
        surface_matrix = griddata(points, values, (grid_x, grid_y), method='linear')

    if np.isnan(surface_matrix).any():
        min_vol = np.nanmin(surface_matrix)
        surface_matrix = np.where(np.isnan(surface_matrix), min_vol, surface_matrix)


    surface_matrix = gaussian_filter(surface_matrix, sigma=2.0)


    if is_dark:
        text_color = "#fff"
        grid_color = "rgba(255,255,255,0.15)"
        zeroline_color = "rgba(255,255,255,0.25)"
        colorbar_bg = "rgba(30,41,59,0.85)"
        colorscale = "Viridis"
    else:
        text_color = "#000"
        grid_color = "rgba(128,128,128,0.15)"
        zeroline_color = "rgba(128,128,128,0.2)"
        colorbar_bg = "rgba(255,255,255,0.85)"
        colorscale = "Turbo"


    fig = go.Figure(
        data=[
            go.Surface(
                z=surface_matrix,
                x=unique_expiries,
                y=y_vals,
                colorscale=colorscale,
                colorbar=dict(
                    title=dict(text="Implied Volatility", font={"size": 16, "color": text_color}),
                    tickfont={"size": 14, "color": text_color},
                    outlinewidth=0,
                    bgcolor=colorbar_bg,
                    bordercolor=colorbar_bg,
                ),
                showscale=True,
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="#42f462", project_z=True, width=2),
                    x=dict(show=True, usecolormap=True, highlightcolor="#42f462", project_x=True, width=1),
                    y=dict(show=True, usecolormap=True, highlightcolor="#42f462", project_y=True, width=1),
                ),
                lighting=dict(ambient=0.8, diffuse=0.9, fresnel=0.1, roughness=0.1, specular=0.5),
                hoverinfo='z+text',
                hovertemplate=(
                    '<b>Days to Expiry:</b> %{x}<br>'
                    f'<b>{y_axis_label}:</b> %{{y:.2f}}<br>'
                    '<b>Implied Volatility:</b> %{z:.3f}<br>'
                    '<extra></extra>'
                ),
            )
        ]
    )

    fig.update_layout(
        paper_bgcolor="#181f2a",
        plot_bgcolor="#181f2a",
        scene=dict(
            bgcolor="#181f2a",
            xaxis_title=dict(text="Days to Expiration", font=dict(size=16, color=text_color)),
            yaxis_title=dict(text=y_axis_label, font=dict(size=16, color=text_color)),
            zaxis_title=dict(text="Implied Volatility", font=dict(size=16, color=text_color)),
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.5)),
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)",
                showbackground=False,
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)",
                showbackground=False,
            ),
            zaxis=dict(
                gridcolor="rgba(255,255,255,0.1)",
                zerolinecolor="rgba(255,255,255,0.2)",
                showbackground=False,
            ),
        ),
        scene_dragmode="orbit",
        margin=dict(l=0, r=0, b=0, t=40),
        dragmode=False,
        uirevision=True,
        font=dict(color=text_color, family="Arial, sans-serif"),
        showlegend=False,
        autosize=True,
        height=700,
    )


    arb_msgs = detect_arbitrage(calls, spot_price, r=rfr, q=0.0)
    if arb_msgs and (len(arb_msgs) > 0 and not (len(arb_msgs) == 1 and ('No significant' in arb_msgs[0] or not arb_msgs[0].strip()))):
        arb_children = []
        for msg in arb_msgs:
            if ':' in msg:
                summary, details = msg.split(':', 1)
                summary = summary.strip()
                details = details.strip()
            else:
                summary = msg
                details = ''

            trade_bullets = []
            expiration = None
            exp_match = re.search(r'expiry (\d+) days', summary)
            if exp_match:
                days = int(exp_match.group(1))
                exp_row = calls[calls['days_to_expiry'] == days]
                if not exp_row.empty:
                    expiration = str(exp_row['expiration'].iloc[0])
            for action, ticker, strike, expiration, side, price in re.findall(r'(Buy|Sell) 1x (\w+) Call, Strike ([\d.]+), Exp ([\d-]+), at (Ask|Bid) \$([\d.]+)', details):
                trade_bullets.append(
                    html.Li([
                        html.Strong(f"{action} 1x {ticker} Call"),
                        f", Strike {strike}, Exp {expiration if expiration else '?'}",
                        f", at {side} ${price}"
                    ], style={"marginBottom": "0.25rem", "fontSize": "0.97rem"})
                )
            arb_children.append(
                html.Details([
                    html.Summary(f"💡 {summary}", style={"fontWeight":700, "fontSize":"1.05rem", "color":"#f59e0b"}),
                    html.Ul(trade_bullets, style={"marginLeft":"1.5rem", "marginBottom":"1rem", "marginTop":"0.5rem"})
                ], style={"marginBottom":"1rem", "background":"#232946", "borderRadius":"10px", "padding":"0.5rem 1rem", "boxShadow":"0 2px 8px rgba(0,0,0,0.08)"})
            )
        arb_text = arb_children
        arb_status = "Alerts"
        arb_status_class = "status-indicator status-warning"
    else:
        arb_text = None
        arb_status = ""
        arb_status_class = ""

    return (
        fig,
        arb_text,
        "Ready",
        "status-indicator status-success",
        arb_status,
        arb_status_class,
    )

if __name__ == "__main__":
    app.run(debug=True)
