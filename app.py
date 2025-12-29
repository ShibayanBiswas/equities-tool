"""
Equities Analytics Dashboard - Main App
"""

import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

st.set_page_config(
    page_title='Equities Analytics Dashboard',
    page_icon='📈',
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.utils import config_menu_footer, get_available_indexes, load_index_cache

config_menu_footer()

# Custom CSS for beautiful homepage
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        height: 100%;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    .feature-card h3 {
        color: white;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .feature-card p {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        line-height: 1.6;
    }
    .feature-card ul {
        color: rgba(255,255,255,0.9);
        padding-left: 1.5rem;
        margin-top: 1rem;
    }
    .feature-card li {
        margin: 0.5rem 0;
    }
    .stats-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stats-card h4 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .stats-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }
    .welcome-section {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        color: white;
    }
    .welcome-section h2 {
        color: white;
        margin-bottom: 1rem;
    }
    .quick-link {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        margin: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .quick-link:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🚀 Equities Analytics Dashboard</h1>
    <p>Advanced Stock Analysis & Market Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Feature Cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Index Dashboard</h3>
        <p>Comprehensive market analysis with interactive visualizations and sector-wise rankings.</p>
        <ul>
            <li>📈 View top stocks across all sectors</li>
            <li>🥧 Interactive pie charts & bar charts</li>
            <li>🏢 Sector & industry breakdown</li>
            <li>🔍 Advanced filtering & sorting</li>
            <li>📋 Composite scoring system (0-5 scale)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📈 Stock Detail</h3>
        <p>Deep dive into individual stock analysis with comprehensive financial metrics.</p>
        <ul>
            <li>💰 Financial statements & ratios</li>
            <li>📊 Interactive charts & visualizations</li>
            <li>🎯 Analyst estimates & ratings</li>
            <li>💬 Sentiment analysis (News & Social)</li>
            <li>🏛️ Institutional holdings analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Welcome Section
st.markdown("""
<div class="welcome-section">
    <h2>✨ Getting Started</h2>
    <p style="font-size: 1.1rem; line-height: 1.8;">
        Welcome to your comprehensive equities analytics platform! This dashboard provides powerful tools for stock analysis, 
        market intelligence, and investment research.
    </p>
    <h3 style="color: white; margin-top: 1.5rem;">🎯 Quick Navigation</h3>
    <p style="font-size: 1rem; line-height: 1.8;">
        <strong>1. Index Dashboard:</strong> Start by exploring stocks across different indices and sectors. Use interactive 
        filters to find top-performing stocks.
    </p>
    <p style="font-size: 1rem; line-height: 1.8;">
        <strong>2. Stock Detail:</strong> Click on any stock symbol from the Index Dashboard to view detailed analysis including 
        financial statements, analyst ratings, sentiment analysis, and institutional holdings.
    </p>
    <h3 style="color: white; margin-top: 1.5rem;">💡 Pro Tips</h3>
    <ul style="font-size: 1rem; line-height: 1.8;">
        <li>Use the <strong>Index Dashboard</strong> to discover top stocks by composite score</li>
        <li>Filter by sector or industry to focus on specific market segments</li>
        <li>Sort stocks by different metrics to find the best opportunities</li>
        <li>Click on any stock symbol to dive deep into comprehensive analysis</li>
        <li>Explore interactive charts and visualizations for better insights</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Key Features
st.markdown("### 🌟 Key Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🎯 Smart Scoring
    - Composite score out of 5
    - Multi-factor analysis
    - Sector normalization
    - Real-time rankings
    """)

with col2:
    st.markdown("""
    #### 📊 Rich Visualizations
    - Interactive charts
    - Sector distribution
    - Industry breakdown
    - Trend analysis
    """)

with col3:
    st.markdown("""
    #### 🔍 Comprehensive Analysis
    - Financial ratios
    - Analyst ratings
    - Sentiment scores
    - Institutional data
    """)

st.markdown("---")

# Footer note
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p style="font-size: 0.9rem;">
        <strong>Equities Analytics Dashboard</strong> | Powered by Financial Modeling Prep API & Advanced Analytics
    </p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem;">
        Navigate using the sidebar to explore different sections of the dashboard
    </p>
</div>
""", unsafe_allow_html=True)
