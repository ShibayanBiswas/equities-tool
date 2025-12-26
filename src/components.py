"""
Reusable Streamlit components with dark theme styling.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from typing import Optional, List, Dict
from millify import millify


def create_plotly_config() -> dict:
    """Create Plotly config for charts."""
    return {
        'displaylogo': False,
        'displayModeBar': True,
        'modeBarButtonsToRemove': [
            'zoom2d', 'pan2d', 'select2d', 'lasso2d',
            'hoverClosestCartesian', 'hoverCompareCartesian',
            'autoScale2d', 'toggleSpikelines', 'resetScale2d',
            'zoomIn2d', 'zoomOut2d', 'hoverClosest3d',
            'hoverClosestGeo', 'hoverClosestGl2d', 'hoverClosestPie',
            'toggleHover', 'resetViews', 'toggleSpikeLines',
            'resetViewMapbox', 'resetGeo', 'hoverClosestGeo',
            'sendDataToCloud', 'hoverClosestGl'
        ],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'chart',
            'height': 600,
            'width': 1200,
            'scale': 2
        }
    }


def get_dark_theme_layout(title: str, **kwargs) -> Dict:
    """Create dark theme layout for Plotly charts."""
    base_layout = {
        'title': {
            'text': title,
            'font': {
                'family': 'Inter, system-ui, -apple-system, sans-serif',
                'size': 20,
                'color': '#FAFAFA'
            },
            'x': 0.5,
            'xanchor': 'center'
        },
        'paper_bgcolor': '#0E1117',
        'plot_bgcolor': '#1E2229',
        'font': {
            'family': 'Inter, system-ui, -apple-system, sans-serif',
            'size': 12,
            'color': '#E0E0E0'
        },
        'xaxis': {
            'gridcolor': '#2E3338',
            'linecolor': '#3E4449',
            'zerolinecolor': '#2E3338',
            'title': {
                'font': {'color': '#E0E0E0', 'size': 13}
            },
            'tickfont': {'color': '#B0B0B0'}
        },
        'yaxis': {
            'gridcolor': '#2E3338',
            'linecolor': '#3E4449',
            'zerolinecolor': '#2E3338',
            'title': {
                'font': {'color': '#E0E0E0', 'size': 13}
            },
            'tickfont': {'color': '#B0B0B0'}
        },
        'legend': {
            'bgcolor': 'rgba(0, 0, 0, 0)',
            'bordercolor': '#3E4449',
            'borderwidth': 1,
            'font': {'color': '#E0E0E0', 'size': 11},
            'x': 1.02,
            'y': 1,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        'hovermode': 'x unified',
        'hoverlabel': {
            'bgcolor': '#1E2229',
            'bordercolor': '#00D4AA',
            'font': {'color': '#FAFAFA', 'size': 11}
        },
        'margin': {'l': 60, 'r': 40, 't': 80, 'b': 60}
    }
    base_layout.update(kwargs)
    return base_layout


def plot_stock_price(df: pd.DataFrame, symbol: str, years: int = 5):
    """Plot stock price over time with dark theme."""
    if df.empty:
        st.warning("No price data available")
        return
    
    # Filter to last N years if needed
    if len(df) > 0:
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years)
        df_filtered = df[df.index >= cutoff_date] if isinstance(df.index, pd.DatetimeIndex) else df
    else:
        df_filtered = df
    
    # Determine line color based on performance
    is_positive = len(df_filtered) > 0 and df_filtered['Price'].iloc[-1] > df_filtered['Price'].iloc[0]
    line_color = '#00D4AA' if is_positive else '#FF6B6B'
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_filtered.index,
            y=df_filtered['Price'],
            mode='lines',
            name='Price',
            line=dict(
                color=line_color,
                width=2.5,
                shape='spline',
                smoothing=1.0
            ),
            fill='tozeroy',
            fillcolor=f'rgba({0 if is_positive else 255}, {212 if is_positive else 107}, {170 if is_positive else 107}, 0.1)',
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
        )
    )
    
    layout = get_dark_theme_layout(
        f'{symbol} Market Performance',
        dragmode='pan',
        xaxis=dict(fixedrange=True),
        yaxis=dict(
            fixedrange=True,
            tickformat='$,.0f'
        )
    )
    fig.update_layout(layout)
    
    st.plotly_chart(fig, config=create_plotly_config(), width='stretch')


def plot_net_income(df: pd.DataFrame, symbol: str):
    """Plot net income over time with dark theme."""
    if df.empty or '= Net Income' not in df.columns:
        return
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['= Net Income'],
            mode='lines+markers',
            name='Net Income',
            line=dict(
                color='#9B59B6',
                width=3,
                shape='spline',
                smoothing=1.0
            ),
            marker=dict(
                size=8,
                color='#9B59B6',
                line=dict(width=1, color='#1E2229')
            ),
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.15)',
            hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Net Income: $%{y:,.0f}<extra></extra>'
        )
    )
    
    layout = get_dark_theme_layout(
        f'{symbol} Net Income',
        dragmode='pan',
        xaxis=dict(
            tickmode='array',
            tickvals=df.index,
            fixedrange=True
        ),
        yaxis=dict(
            fixedrange=True,
            tickformat='$,.0f'
        )
    )
    fig.update_layout(layout)
    
    st.plotly_chart(fig, config=create_plotly_config(), use_container_width=True)


def plot_profitability_margins(df: pd.DataFrame, symbol: str):
    """Plot profitability margins with dark theme."""
    if df.empty:
        return
    
    required_cols = ['Gross Profit Margin', 'Operating Profit Margin', 'Net Profit Margin']
    if not all(col in df.columns for col in required_cols):
        return
    
    colors = ['#00D4AA', '#4A90E2', '#7B68EE']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df.index,
        x=df['Gross Profit Margin'],
        name='Gross Profit Margin',
        marker=dict(
            color=colors[0],
            line=dict(color='#1E2229', width=1)
        ),
        orientation='h',
        hovertemplate='<b>%{fullData.name}</b><br>Year: %{y}<br>Margin: %{x:.1%}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        y=df.index,
        x=df['Operating Profit Margin'],
        name='Operating Profit Margin',
        marker=dict(
            color=colors[1],
            line=dict(color='#1E2229', width=1)
        ),
        orientation='h',
        hovertemplate='<b>%{fullData.name}</b><br>Year: %{y}<br>Margin: %{x:.1%}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        y=df.index,
        x=df['Net Profit Margin'],
        name='Net Profit Margin',
        marker=dict(
            color=colors[2],
            line=dict(color='#1E2229', width=1)
        ),
        orientation='h',
        hovertemplate='<b>%{fullData.name}</b><br>Year: %{y}<br>Margin: %{x:.1%}<extra></extra>'
    ))
    
    layout = get_dark_theme_layout(
        f'{symbol} Profitability Margins',
        bargap=0.15,
        barmode='group',
        dragmode='pan',
        xaxis=dict(
            fixedrange=True,
            tickformat='.0%'
        ),
        yaxis=dict(fixedrange=True)
    )
    fig.update_layout(layout)
    
    st.plotly_chart(fig, config=create_plotly_config(), use_container_width=True)


def plot_balance_sheet(df: pd.DataFrame, symbol: str):
    """Plot balance sheet with dark theme."""
    if df.empty:
        st.info("Balance sheet data is empty")
        return
    
    required_cols = ['Assets', 'Liabilities', 'Equity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Balance sheet missing columns: {missing_cols}. Available: {list(df.columns)[:10]}")
        return
    
    
    fig = go.Figure()
    # Add Assets as Scatter with fill (like ROE & ROA style)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Assets'],
        mode='lines+markers',
        name='Assets',
        line=dict(
            color='#00D4AA',  # Green
            width=3,
            shape='spline',
            smoothing=1.0
        ),
        marker=dict(
            size=8,
            color='#00D4AA',
            line=dict(width=1, color='#1E2229')
        ),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 170, 0.1)',
        hovertemplate='<b>Assets</b><br>Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))
    # Add Liabilities as Scatter with fill
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Liabilities'],
        mode='lines+markers',
        name='Liabilities',
        line=dict(
            color='#FF6B6B',  # Red
            width=3,
            shape='spline',
            smoothing=1.0
        ),
        marker=dict(
            size=8,
            color='#FF6B6B',
            line=dict(width=1, color='#1E2229')
        ),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.1)',
        hovertemplate='<b>Liabilities</b><br>Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))
    # Add Equity as Scatter with fill
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Equity'],
        mode='lines+markers',
        name='Equity',
        line=dict(
            color='#4A90E2',  # Blue
            width=3,
            shape='spline',
            smoothing=1.0
        ),
        marker=dict(
            size=8,
            color='#4A90E2',
            line=dict(width=1, color='#1E2229')
        ),
        fill='tozeroy',
        fillcolor='rgba(74, 144, 226, 0.1)',
        hovertemplate='<b>Equity</b><br>Year: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))
    
    layout = get_dark_theme_layout(
        f'{symbol} Balance Sheet',
        dragmode='pan',
        xaxis=dict(fixedrange=True),
        yaxis=dict(
            fixedrange=True,
            tickformat='$,.0f'
        )
    )
    fig.update_layout(layout)
    
    st.plotly_chart(fig, config=create_plotly_config(), use_container_width=True)


def plot_roe_roa(ratios_df: pd.DataFrame, symbol: str):
    """Plot ROE and ROA with dark theme."""
    if ratios_df.empty:
        return
    
    required_cols = ['Return on Equity', 'Return on Assets']
    if not all(col in ratios_df.columns for col in required_cols):
        return
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ratios_df.index,
        y=ratios_df['Return on Equity'],
        name='Return on Equity',
        line=dict(
            color='#00D4AA',
            width=3,
            shape='spline',
            smoothing=1.0
        ),
        marker=dict(
            size=8,
            color='#00D4AA',
            line=dict(width=1, color='#1E2229')
        ),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 170, 0.1)',
        hovertemplate='<b>Return on Equity</b><br>Year: %{x}<br>ROE: %{y:.1%}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=ratios_df.index,
        y=ratios_df['Return on Assets'],
        name='Return on Assets',
        line=dict(
            color='#4A90E2',
            width=3,
            shape='spline',
            smoothing=1.0
        ),
        marker=dict(
            size=8,
            color='#4A90E2',
            line=dict(width=1, color='#1E2229')
        ),
        fill='tozeroy',
        fillcolor='rgba(74, 144, 226, 0.1)',
        hovertemplate='<b>Return on Assets</b><br>Year: %{x}<br>ROA: %{y:.1%}<extra></extra>'
    ))
    
    layout = get_dark_theme_layout(
        f'{symbol} Return on Equity and Return on Assets',
        dragmode='pan',
        xaxis=dict(fixedrange=True),
        yaxis=dict(
            fixedrange=True,
            tickformat='.0%'
        )
    )
    fig.update_layout(layout)
    
    st.plotly_chart(fig, config=create_plotly_config(), use_container_width=True)


def plot_cash_flows(df: pd.DataFrame, symbol: str):
    """Plot cash flows with dark theme."""
    if df.empty:
        st.info("Cash flow data is empty")
        return
    
    required_cols = [
        'Cash flows from operating activities',
        'Cash flows from investing activities',
        'Cash flows from financing activities',
        'Free cash flow'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Cash flow missing columns: {missing_cols}. Available: {list(df.columns)[:10]}")
        return
    
    
    colors = ['#00D4AA', '#4A90E2', '#7B68EE', '#FFA500']
    
    fig = go.Figure()
    # All as Scatter plots with fill (like ROE & ROA style)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Cash flows from operating activities'],
        mode='lines+markers',
        name='Cash flows from operating activities',
        line=dict(
            color=colors[0],
            width=3,
            shape='spline',
            smoothing=1.0
        ),
        marker=dict(
            size=8,
            color=colors[0],
            line=dict(width=1, color='#1E2229')
        ),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 170, 0.1)',
        hovertemplate='<b>Cash flows from operating activities</b><br>Year: %{x}<br>Cash Flow: $%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Cash flows from investing activities'],
        mode='lines+markers',
        name='Cash flows from investing activities',
        line=dict(
            color=colors[1],
            width=3,
            shape='spline',
            smoothing=1.0
        ),
        marker=dict(
            size=8,
            color=colors[1],
            line=dict(width=1, color='#1E2229')
        ),
        fill='tozeroy',
        fillcolor='rgba(74, 144, 226, 0.1)',
        hovertemplate='<b>Cash flows from investing activities</b><br>Year: %{x}<br>Cash Flow: $%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Cash flows from financing activities'],
        mode='lines+markers',
        name='Cash flows from financing activities',
        line=dict(
            color=colors[2],
            width=3,
            shape='spline',
            smoothing=1.0
        ),
        marker=dict(
            size=8,
            color=colors[2],
            line=dict(width=1, color='#1E2229')
        ),
        fill='tozeroy',
        fillcolor='rgba(123, 104, 238, 0.1)',
        hovertemplate='<b>Cash flows from financing activities</b><br>Year: %{x}<br>Cash Flow: $%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Free cash flow'],
        mode='lines+markers',
        name='Free cash flow',
        line=dict(
            color=colors[3],
            width=3,
            shape='spline',
            smoothing=1.0
        ),
        marker=dict(
            symbol='diamond',
            size=12,
            color=colors[3],
            line=dict(width=2, color='#1E2229')
        ),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.1)',
        hovertemplate='<b>Free cash flow</b><br>Year: %{x}<br>FCF: $%{y:,.0f}<extra></extra>'
    ))
    
    layout = get_dark_theme_layout(
        f'{symbol} Cash Flows',
        dragmode='pan',
        xaxis=dict(fixedrange=True),
        yaxis=dict(
            fixedrange=True,
            tickformat='$,.0f'
        )
    )
    fig.update_layout(layout)
    
    st.plotly_chart(fig, config=create_plotly_config(), use_container_width=True)

