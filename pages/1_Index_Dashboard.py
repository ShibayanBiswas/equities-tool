"""
Index Dashboard - Sector-first navigation and ranking with interactive visualizations
"""

import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from src.utils import get_available_indexes, load_index_cache, empty_lines
from src.indexes import get_index_universe_features, get_index_sector_groups
from src.scoring import rank_stocks
from src.fmp_client import FMPClient
from src.components import create_plotly_config

st.set_page_config(
    page_title='Index Dashboard',
    page_icon='📊',
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.utils import config_menu_footer
config_menu_footer()

# Helper functions for safe formatting
def safe_format_price(x):
    try:
        if pd.isna(x) or x is None:
            return "N/A"
        val = float(x)
        return f"${val:,.2f}"
    except (ValueError, TypeError):
        return "N/A"

def safe_format_marketcap(x):
    try:
        if pd.isna(x) or x is None:
            return "N/A"
        val = float(x)
        if val > 0:
            return f"${val/1e9:.2f}B"
        return "N/A"
    except (ValueError, TypeError):
        return "N/A"

def safe_format_ratio(x):
    try:
        if pd.isna(x) or x is None:
            return "N/A"
        val = float(x)
        return f"{val:.2f}"
    except (ValueError, TypeError):
        return "N/A"

def safe_format_percent(x):
    try:
        if pd.isna(x) or x is None:
            return "N/A"
        val = float(x)
        return f"{val*100:.2f}%"
    except (ValueError, TypeError):
        return "N/A"

def safe_format_score(x):
    try:
        if pd.isna(x) or x is None:
            return "N/A"
        val = float(x)
        return f"{val:.3f}"
    except (ValueError, TypeError):
        return "N/A"

# Custom CSS for enhanced sidebar
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1e1e2e 0%, #0e1117 100%);
    }
    .stSelectbox label, .stSlider label, .stCheckbox label {
        color: #FAFAFA !important;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    h1, h2, h3 {
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Index Dashboard")

# ==================== ENHANCED SIDEBAR ====================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Index Selection")

# Get available indexes
available_indexes = get_available_indexes()

if not available_indexes:
    st.warning("⚠️ No cached indexes found. Please run the index cache builder first:")
    st.code("python scripts/build_index_cache_agno.py --out data/index_cache --min-coverage 0.9")
    st.stop()

# Index selection
index_options = {name: slug for slug, name in available_indexes.items()}
selected_index_display = st.sidebar.selectbox(
    "📈 Select Index",
    options=list(index_options.keys()),
    index=0,
    help="Choose the index to analyze"
)

selected_index_slug = index_options[selected_index_display]
selected_index_name = selected_index_display

# Load index data
cache = load_index_cache(selected_index_slug)

if not cache or 'universe_features' not in cache:
    st.error(f"Failed to load data for {selected_index_name}")
    st.stop()

universe_features = cache['universe_features']
sector_groups = cache.get('sector_groups', {})
industry_groups = cache.get('industry_groups', {})

# Sidebar filters and options
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Display Options")

# Normalization toggle
normalize_within_sector = st.sidebar.checkbox(
    "📊 Normalize within sector",
    value=True,
    help="When enabled, scores are normalized within each sector. When disabled, scores are normalized across the entire index."
)

# Top N selector
top_n = st.sidebar.slider(
    "🔝 Top N Stocks",
    min_value=10,
    max_value=100,
    value=20,
    step=10,
    help="Number of top stocks to display"
)

# Sector filter with chart options (like Top Stocks Across All Sectors)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏢 Sector Filter")

sectors = ['All Sectors'] + sorted([s for s in sector_groups.keys() if s != 'Unknown'])
selected_sector = st.sidebar.selectbox(
    "🔍 Filter by Sector",
    sectors,
    help="Filter stocks by sector"
)

# Industry selection removed - no longer filtering by industry
selected_industry = None

# Visualization toggles (for pie charts)
show_pie_charts = True  # Always show pie charts
show_bar_charts = False  # Don't show the bar chart section

# ==================== MAIN CONTENT ====================

# Calculate composite scores if not already calculated or if all scores are zero
has_composite_score = 'composite_score' in universe_features.columns
all_zero_or_na = False
if has_composite_score:
    score_sum = universe_features['composite_score'].sum()
    score_mean = universe_features['composite_score'].mean()
    all_zero_or_na = (score_sum == 0) or (pd.isna(score_mean)) or (score_mean == 0)

if not has_composite_score or all_zero_or_na:
    from src.scoring import calculate_composite_score_v2
    universe_features = calculate_composite_score_v2(universe_features)

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Stocks", len(universe_features))
with col2:
    st.metric("Total Sectors", len(sector_groups))
with col3:
    total_market_cap = universe_features['marketCap'].sum() / 1e12 if 'marketCap' in universe_features.columns else 0
    st.metric("Total Market Cap", f"${total_market_cap:.2f}T")
with col4:
    avg_score = universe_features['composite_score'].mean() if 'composite_score' in universe_features.columns else 0
    st.metric("Avg Composite Score", f"{avg_score:.2f}/5")

st.markdown("---")

# Filter and rank stocks
if selected_sector == 'All Sectors':
    # ==================== ALL SECTORS VIEW ====================
    
    # Sector Summary Table (moved to top)
    st.header("📋 Sector Overview")
    sector_summary = []
    for sector, tickers in sector_groups.items():
        if sector != 'Unknown':
            sector_df = universe_features[universe_features['sector'] == sector]
            if not sector_df.empty:
                ranked = rank_stocks(sector_df, sector=sector, normalize_within_sector=normalize_within_sector, top_n=5)
            if not ranked.empty:
                sector_market_cap = sector_df['marketCap'].sum() / 1e12 if 'marketCap' in sector_df.columns else 0
                avg_score = sector_df['composite_score'].mean() if 'composite_score' in sector_df.columns else 0
                sector_summary.append({
                    'Sector': sector,
                    'Stock Count': len(sector_df),
                    'Top Stock': ranked.iloc[0]['symbol'] if len(ranked) > 0 else 'N/A',
                    'Top Score': f"{ranked.iloc[0]['composite_score']:.2f}/5" if len(ranked) > 0 else 'N/A',
                    'Avg Score': f"{avg_score:.2f}/5",
                    'Total Market Cap': f"${sector_market_cap:.2f}T"
                })
    
    if sector_summary:
        summary_df = pd.DataFrame(sector_summary)
        st.dataframe(
            summary_df,
            width='stretch',
            hide_index=True,
            use_container_width=True,
            column_config={
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "Stock Count": st.column_config.NumberColumn("Stock Count", format="%d"),
                "Top Stock": st.column_config.TextColumn("Top Stock", width="small"),
                "Top Score": st.column_config.TextColumn("Top Score", width="small"),
                "Avg Score": st.column_config.TextColumn("Avg Score", width="small"),
                "Total Market Cap": st.column_config.TextColumn("Market Cap", width="medium"),
            }
        )
    
    # Sector Distribution Pie Chart
    if show_pie_charts:
        st.header("📊 Sector Distribution")
        
        # Stock count by sector
        sector_counts = {sector: len(tickers) for sector, tickers in sector_groups.items() if sector != 'Unknown'}
        if sector_counts:
            # Calculate percentages for labels
            total_count = sum(sector_counts.values())
            # Create custom text labels - show only for major portions (>15% or top 2)
            sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
            custom_text = []
            for sector, count in sorted_sectors:
                percentage = (count / total_count * 100) if total_count > 0 else 0
                if percentage > 15 or len(custom_text) < 2:
                    custom_text.append(f"{sector}<br>{percentage:.1f}%")
                else:
                    custom_text.append("")
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(sector_counts.keys()),
                values=list(sector_counts.values()),
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3),
                textinfo='none',
                hovertemplate='<b>%{label}</b><br>Stocks: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            fig_pie.update_layout(
                title="Stock Count by Sector",
                template='plotly_dark',
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
            )
            st.plotly_chart(fig_pie, config=create_plotly_config(), width='stretch')
        
        # Market cap by sector (below stock count)
        sector_market_caps = {}
        for sector, tickers in sector_groups.items():
            if sector != 'Unknown':
                sector_df = universe_features[universe_features['sector'] == sector]
                if not sector_df.empty and 'marketCap' in sector_df.columns:
                    sector_market_caps[sector] = sector_df['marketCap'].sum() / 1e12  # Convert to trillions
        
        if sector_market_caps:
            # Calculate percentages for labels
            total_mcap = sum(sector_market_caps.values())
            # Create custom text labels - show only for major portions (>15% or top 2)
            sorted_sectors_mcap = sorted(sector_market_caps.items(), key=lambda x: x[1], reverse=True)
            custom_text_mcap = []
            for sector, mcap in sorted_sectors_mcap:
                percentage = (mcap / total_mcap * 100) if total_mcap > 0 else 0
                if percentage > 15 or len(custom_text_mcap) < 2:
                    custom_text_mcap.append(f"{sector}<br>{percentage:.1f}%")
                else:
                    custom_text_mcap.append("")
            
            fig_pie2 = go.Figure(data=[go.Pie(
                labels=list(sector_market_caps.keys()),
                values=list(sector_market_caps.values()),
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Pastel),
                textinfo='none',
                hovertemplate='<b>%{label}</b><br>Market Cap: $%{value:.2f}T<br>Percentage: %{percent}<extra></extra>'
            )])
            fig_pie2.update_layout(
                title="Market Cap by Sector",
                template='plotly_dark',
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
            )
            st.plotly_chart(fig_pie2, config=create_plotly_config(), width='stretch')
    
    # Top Stocks by Sector Table with Interactive Visualizations
    st.header("📋 Top Stocks by Sector")
    sector_top_stocks_table = []
    for sector, tickers in sector_groups.items():
        if sector != 'Unknown':
            sector_df = universe_features[universe_features['sector'] == sector]
            if not sector_df.empty:
                ranked = rank_stocks(sector_df, sector=sector, normalize_within_sector=normalize_within_sector, top_n=5)
                if not ranked.empty:
                    for idx, row in ranked.iterrows():
                        sector_top_stocks_table.append({
                            'Sector': sector,
                            'Industry': row.get('industry', 'N/A'),
                            'Rank': row.get('rank', 'N/A'),
                            'Symbol': row.get('symbol', 'N/A'),
                            'Company': row.get('companyName', 'N/A'),
                            'Price': safe_format_price(row.get('price', 0)),
                            'Market Cap': safe_format_marketcap(row.get('marketCap', 0)),
                            'Composite Score': f"{row.get('composite_score', 0):.2f}/5",
                            'Rating': row.get('rating', 'N/A'),
                            'Rating Recommendation': row.get('ratingRecommendation', 'N/A'),
                        })
    
    if sector_top_stocks_table:
        # Prepare both formatted and raw data
        top_stocks_table_df = pd.DataFrame(sector_top_stocks_table)
        
        # Prepare raw data for charts (before formatting)
        sector_top_stocks_raw = []
        for sector, tickers in sector_groups.items():
            if sector != 'Unknown':
                sector_df = universe_features[universe_features['sector'] == sector]
                if not sector_df.empty:
                    ranked = rank_stocks(sector_df, sector=sector, normalize_within_sector=normalize_within_sector, top_n=5)
                    if not ranked.empty:
                        for idx, row in ranked.iterrows():
                            sector_top_stocks_raw.append({
                                'Sector': sector,
                                'Industry': row.get('industry', 'N/A'),
                                'Symbol': row.get('symbol', 'N/A'),
                                'Company': row.get('companyName', 'N/A'),
                                'Price': row.get('price', 0),
                                'Market Cap': row.get('marketCap', 0),
                                'Composite Score': row.get('composite_score', 0),
                            })
        
        top_stocks_raw_df = pd.DataFrame(sector_top_stocks_raw)
        
        # Display table
        st.dataframe(
            top_stocks_table_df,
            width='stretch',
            hide_index=True,
            use_container_width=True,
            column_config={
                "Sector": st.column_config.TextColumn("Sector", width="medium"),
                "Industry": st.column_config.TextColumn("Industry", width="medium"),
                "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Company": st.column_config.TextColumn("Company", width="medium"),
                "Price": st.column_config.TextColumn("Price", width="small"),
                "Market Cap": st.column_config.TextColumn("Market Cap", width="medium"),
                "Composite Score": st.column_config.TextColumn("Score", width="small"),
                "Rating": st.column_config.TextColumn("Rating", width="small"),
                "Rating Recommendation": st.column_config.TextColumn("Rating Rec", width="medium"),
            }
        )
    
    # Top Stocks Across All Sectors
    st.header("🏆 Top Stocks Across All Sectors")
    ranked_all = rank_stocks(universe_features, sector=None, normalize_within_sector=normalize_within_sector, top_n=top_n)
    
    if ranked_all.empty:
        st.warning("No stocks found. Please check if universe_features data is loaded correctly.")
    else:
        # Sort by composite score by default
        if 'composite_score' in ranked_all.columns:
            ranked_all = ranked_all.sort_values('composite_score', ascending=False)
            ranked_all['rank'] = range(1, len(ranked_all) + 1)
        
        # Display ranked table - include rating columns (non-numeric) and composite score, add industry
        base_cols = ['rank', 'symbol', 'companyName', 'sector', 'industry', 'price', 'marketCap', 'composite_score']
        # Add rating columns (non-numeric for display)
        rating_cols = ['rating', 'ratingRecommendation', 'ratingDetailsDCFRecommendation', 
                      'ratingDetailsROERecommendation', 'ratingDetailsROARecommendation',
                      'ratingDetailsDERecommendation', 'ratingDetailsPERecommendation',
                      'ratingDetailsPBRecommendation']
        # Add optional columns if they exist
        optional_cols = ['sentimentScore', 'volume', 'beta', 'dividendYield']
        display_cols = base_cols + [col for col in rating_cols if col in ranked_all.columns] + [col for col in optional_cols if col in ranked_all.columns]
        
        available_cols = [col for col in display_cols if col in ranked_all.columns]
        if not available_cols:
            st.error(f"No expected columns found. Available columns: {list(ranked_all.columns)}")
        else:
            display_df = ranked_all[available_cols].copy()
            
            # Format numbers (keep original values for sorting, format for display)
            display_df_formatted = display_df.copy()
            
            if 'price' in display_df_formatted.columns:
                display_df_formatted['price'] = display_df_formatted['price'].apply(safe_format_price)
            if 'marketCap' in display_df_formatted.columns:
                display_df_formatted['marketCap'] = display_df_formatted['marketCap'].apply(safe_format_marketcap)
            if 'composite_score' in display_df_formatted.columns:
                display_df_formatted['composite_score'] = display_df_formatted['composite_score'].apply(
                    lambda x: f"{float(x):.2f}/5" if pd.notna(x) and x is not None else "N/A"
                )
            if 'sentimentScore' in display_df_formatted.columns:
                display_df_formatted['sentimentScore'] = display_df_formatted['sentimentScore'].apply(
                    lambda x: f"{float(x):.3f}" if pd.notna(x) and x is not None else "N/A"
                )
            if 'volume' in display_df_formatted.columns:
                display_df_formatted['volume'] = display_df_formatted['volume'].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) and x is not None else "N/A"
                )
            if 'beta' in display_df_formatted.columns:
                display_df_formatted['beta'] = display_df_formatted['beta'].apply(safe_format_ratio)
            if 'dividendYield' in display_df_formatted.columns:
                display_df_formatted['dividendYield'] = display_df_formatted['dividendYield'].apply(safe_format_percent)
            
            # Interactive controls for visualizations
            col1, col2 = st.columns(2)
            with col1:
                selected_sector_all = st.selectbox(
                    "🔍 Filter by Sector",
                    options=['All Sectors'] + sorted(display_df['sector'].unique().tolist()) if 'sector' in display_df.columns else ['All Sectors'],
                    key='sector_filter_all'
                )
            with col2:
                chart_type_all = st.selectbox(
                    "📊 Chart Type",
                    options=['Bar Chart', 'Treemap'],
                    key='chart_type_all'
                )
            
            # Filter data by sector
            if selected_sector_all != 'All Sectors' and 'sector' in display_df.columns:
                filtered_display_df = display_df[display_df['sector'] == selected_sector_all].copy()
                filtered_display_formatted = display_df_formatted[display_df_formatted['sector'] == selected_sector_all].copy()
            else:
                filtered_display_df = display_df.copy()
                filtered_display_formatted = display_df_formatted.copy()
            
            # Create visualizations (default to Composite Score)
            if not filtered_display_df.empty:
                metric_col = 'composite_score'
                metric_label = 'Score'
                metric_for_chart_all = 'Composite Score'
                
                if chart_type_all == 'Bar Chart':
                    # Top 20 for bar chart
                    top_n_chart = min(20, len(filtered_display_df))
                    chart_data = filtered_display_df.nlargest(top_n_chart, metric_col) if metric_col in filtered_display_df.columns else filtered_display_df.head(top_n_chart)
                    
                    fig = go.Figure(data=[go.Bar(
                        x=chart_data['symbol'],
                        y=chart_data[metric_col] if metric_col in chart_data.columns else [0] * len(chart_data),
                        marker=dict(
                            color=chart_data[metric_col] if metric_col in chart_data.columns else [0] * len(chart_data),
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title=metric_label)
                        ),
                        text=chart_data['companyName'] if 'companyName' in chart_data.columns else chart_data['symbol'],
                        hovertemplate='<b>%{text}</b><br>Symbol: %{x}<br>' + metric_label + ': %{y:,.2f}<extra></extra>'
                    )])
                    fig.update_layout(
                        title=f"Top {top_n_chart} Stocks by {metric_for_chart_all}",
                        xaxis_title="Stock Symbol",
                        yaxis_title=metric_label,
                        template='plotly_dark',
                        height=500,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, config=create_plotly_config(), width='stretch')
                
                elif chart_type_all == 'Treemap':
                    import plotly.express as px
                    # Prepare data for treemap
                    treemap_data = filtered_display_df.head(30).copy()  # Top 30 for treemap
                    if 'sector' in treemap_data.columns and metric_col in treemap_data.columns:
                        fig = px.treemap(
                            treemap_data,
                            path=['sector', 'symbol'],
                            values=metric_col,
                            color=metric_col,
                            color_continuous_scale='RdYlGn',
                            title=f"{metric_for_chart_all} Treemap by Sector",
                            hover_data=['companyName'] if 'companyName' in treemap_data.columns else []
                        )
                        fig.update_layout(template='plotly_dark', height=600)
                        st.plotly_chart(fig, config=create_plotly_config(), width='stretch')
            
            # Interactive dataframe with column configuration
            st.dataframe(
                filtered_display_formatted,
                width='stretch',
                hide_index=True,
                use_container_width=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "companyName": st.column_config.TextColumn("Company", width="medium"),
                    "sector": st.column_config.TextColumn("Sector", width="medium"),
                    "industry": st.column_config.TextColumn("Industry", width="medium"),
                    "composite_score": st.column_config.TextColumn("Score", width="small"),
                }
            )
            
            # Add clickable links using query params
            st.info(f"📊 Showing {len(display_df_formatted)} top stocks. Click on any symbol in the table to view detailed analysis.")
    
else:
    # ==================== SECTOR-SPECIFIC VIEW ====================
    st.header(f"🏢 {selected_sector} Stocks")
    
    # Filter by sector
    filtered_features = universe_features[universe_features['sector'] == selected_sector].copy()
    
    ranked = rank_stocks(
        filtered_features,
        sector=selected_sector,
        normalize_within_sector=normalize_within_sector,
        top_n=top_n
    )
    
    if ranked.empty:
        st.warning(f"No stocks found for sector: {selected_sector}")
    else:
        # Sort by composite score by default
        if 'composite_score' in ranked.columns:
            ranked = ranked.sort_values('composite_score', ascending=False)
            ranked['rank'] = range(1, len(ranked) + 1)
        
        # Display ranked table - include industry column
        base_cols = ['rank', 'symbol', 'companyName', 'sector', 'industry', 'price', 'marketCap', 'composite_score']
        # Add rating columns (non-numeric for display)
        rating_cols = ['rating', 'ratingRecommendation', 'ratingDetailsDCFRecommendation', 
                      'ratingDetailsROERecommendation', 'ratingDetailsROARecommendation',
                      'ratingDetailsDERecommendation', 'ratingDetailsPERecommendation',
                      'ratingDetailsPBRecommendation']
        # Add optional columns if they exist
        optional_cols = ['sentimentScore', 'volume', 'beta', 'dividendYield']
        display_cols = base_cols + [col for col in rating_cols if col in ranked.columns] + [col for col in optional_cols if col in ranked.columns]
        
        available_cols = [col for col in display_cols if col in ranked.columns]
        if not available_cols:
            st.error(f"No expected columns found. Available columns: {list(ranked.columns)}")
        else:
            display_df = ranked[available_cols].copy()
            
            # Format numbers
            display_df_formatted = display_df.copy()
            
            if 'price' in display_df_formatted.columns:
                display_df_formatted['price'] = display_df_formatted['price'].apply(safe_format_price)
            if 'marketCap' in display_df_formatted.columns:
                display_df_formatted['marketCap'] = display_df_formatted['marketCap'].apply(safe_format_marketcap)
            if 'composite_score' in display_df_formatted.columns:
                display_df_formatted['composite_score'] = display_df_formatted['composite_score'].apply(
                    lambda x: f"{float(x):.2f}/5" if pd.notna(x) and x is not None else "N/A"
                )
            if 'sentimentScore' in display_df_formatted.columns:
                display_df_formatted['sentimentScore'] = display_df_formatted['sentimentScore'].apply(
                    lambda x: f"{float(x):.3f}" if pd.notna(x) and x is not None else "N/A"
                )
            if 'volume' in display_df_formatted.columns:
                display_df_formatted['volume'] = display_df_formatted['volume'].apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) and x is not None else "N/A"
                )
            if 'beta' in display_df_formatted.columns:
                display_df_formatted['beta'] = display_df_formatted['beta'].apply(safe_format_ratio)
            if 'dividendYield' in display_df_formatted.columns:
                display_df_formatted['dividendYield'] = display_df_formatted['dividendYield'].apply(safe_format_percent)
            
            # Interactive controls for visualizations
            col1, col2 = st.columns(2)
            with col1:
                selected_industry_filter = st.selectbox(
                    "🔍 Filter by Industry",
                    options=['All Industries'] + sorted(display_df['industry'].unique().tolist()) if 'industry' in display_df.columns else ['All Industries'],
                    key='industry_filter_sector'
                )
            with col2:
                chart_type_sector = st.selectbox(
                    "📊 Chart Type",
                    options=['Bar Chart', 'Treemap'],
                    key='chart_type_sector_view'
                )
            
            # Filter data by industry
            if selected_industry_filter != 'All Industries' and 'industry' in display_df.columns:
                filtered_display_df = display_df[display_df['industry'] == selected_industry_filter].copy()
                filtered_display_formatted = display_df_formatted[display_df_formatted['industry'] == selected_industry_filter].copy()
            else:
                filtered_display_df = display_df.copy()
                filtered_display_formatted = display_df_formatted.copy()
            
            # Create visualizations (default to Composite Score)
            if not filtered_display_df.empty:
                metric_col = 'composite_score'
                metric_label = 'Score'
                metric_for_chart_sector = 'Composite Score'
                
                if chart_type_sector == 'Bar Chart':
                    # Top 20 for bar chart
                    top_n_chart = min(20, len(filtered_display_df))
                    chart_data = filtered_display_df.nlargest(top_n_chart, metric_col) if metric_col in filtered_display_df.columns else filtered_display_df.head(top_n_chart)
                    
                    fig = go.Figure(data=[go.Bar(
                        x=chart_data['symbol'],
                        y=chart_data[metric_col] if metric_col in chart_data.columns else [0] * len(chart_data),
                        marker=dict(
                            color=chart_data[metric_col] if metric_col in chart_data.columns else [0] * len(chart_data),
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title=metric_label)
                        ),
                        text=chart_data['companyName'] if 'companyName' in chart_data.columns else chart_data['symbol'],
                        hovertemplate='<b>%{text}</b><br>Symbol: %{x}<br>' + metric_label + ': %{y:,.2f}<extra></extra>'
                    )])
                    fig.update_layout(
                        title=f"Top {top_n_chart} Stocks by {metric_for_chart_sector}",
                        xaxis_title="Stock Symbol",
                        yaxis_title=metric_label,
                        template='plotly_dark',
                        height=500,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, config=create_plotly_config(), width='stretch')
                
                elif chart_type_sector == 'Treemap':
                    import plotly.express as px
                    # Prepare data for treemap
                    treemap_data = filtered_display_df.head(30).copy()  # Top 30 for treemap
                    if 'industry' in treemap_data.columns and metric_col in treemap_data.columns:
                        fig = px.treemap(
                            treemap_data,
                            path=['industry', 'symbol'],
                            values=metric_col,
                            color=metric_col,
                            color_continuous_scale='RdYlGn',
                            title=f"{metric_for_chart_sector} Treemap by Industry",
                            hover_data=['companyName'] if 'companyName' in treemap_data.columns else []
                        )
                        fig.update_layout(template='plotly_dark', height=600)
                        st.plotly_chart(fig, config=create_plotly_config(), width='stretch')
            
            # Interactive dataframe with column configuration
            st.dataframe(
                filtered_display_formatted,
                width='stretch',
                hide_index=True,
                use_container_width=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "companyName": st.column_config.TextColumn("Company", width="medium"),
                    "sector": st.column_config.TextColumn("Sector", width="medium"),
                    "industry": st.column_config.TextColumn("Industry", width="medium"),
                    "composite_score": st.column_config.TextColumn("Score", width="small"),
                }
            )
            
            st.info(f"📊 Showing {len(filtered_display_formatted)} stocks in {selected_sector}. Click on any symbol in the table to view detailed analysis.")
