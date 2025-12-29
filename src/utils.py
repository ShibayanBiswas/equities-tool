"""
Utility functions for the Equities Analytics Dashboard.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


def config_menu_footer() -> None:
    """Customize Streamlit interface with dark theme styling."""
    app_style = """
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            /* Hide default menu and footer */
            #MainMenu {
              visibility: hidden;
            }
            footer {
                visibility: hidden;
            }
            footer:before {
                content:"Equities Analytics Dashboard";
                visibility: visible;
                display: block;
                position: relative;
                text-align: center;
                color: #FAFAFA;
                padding: 10px;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
            }
            
            /* Custom dark theme styling */
            .stApp {
                background: linear-gradient(135deg, #0E1117 0%, #1E2229 100%);
            }
            
            /* Title styling */
            h1 {
                color: #FAFAFA !important;
                font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
                font-weight: 700 !important;
                font-size: 2.5rem !important;
                margin-bottom: 1rem !important;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            }
            
            h2 {
                color: #E0E0E0 !important;
                font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
                font-weight: 600 !important;
                font-size: 1.8rem !important;
                margin-top: 1.5rem !important;
                margin-bottom: 1rem !important;
            }
            
            h3 {
                color: #E0E0E0 !important;
                font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
                font-weight: 600 !important;
                font-size: 1.4rem !important;
            }
            
            /* Text styling */
            p, .stMarkdown {
                color: #E0E0E0 !important;
                font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
            }
            
            /* Sidebar styling */
            .css-1d391kg {
                background-color: #1E2229 !important;
            }
            
            /* Metric cards */
            [data-testid="stMetricValue"] {
                color: #00D4AA !important;
                font-weight: 700 !important;
                font-size: 2rem !important;
            }
            
            [data-testid="stMetricLabel"] {
                color: #B0B0B0 !important;
                font-weight: 500 !important;
            }
            
            /* Dataframe styling */
            .dataframe {
                background-color: #1E2229 !important;
                color: #E0E0E0 !important;
                border: 1px solid #3E4449 !important;
                border-radius: 8px !important;
            }
            
            .dataframe th {
                background-color: #262730 !important;
                color: #FAFAFA !important;
                font-weight: 600 !important;
                border: 1px solid #3E4449 !important;
            }
            
            .dataframe td {
                border: 1px solid #3E4449 !important;
                color: #E0E0E0 !important;
            }
            
            .dataframe tr:nth-child(even) {
                background-color: #262730 !important;
            }
            
            .dataframe tr:hover {
                background-color: #2E3338 !important;
            }
            
            /* Links */
            a {
                color: #00D4AA !important;
                text-decoration: none !important;
            }
            
            a:hover {
                color: #00B894 !important;
                text-decoration: underline !important;
            }
            
            /* Info boxes */
            .stInfo {
                background-color: #1E2229 !important;
                border-left: 4px solid #00D4AA !important;
                border-radius: 4px !important;
            }
            
            .stWarning {
                background-color: #2E1F0E !important;
                border-left: 4px solid #FFA500 !important;
                border-radius: 4px !important;
            }
            
            .stError {
                background-color: #2E1F0E !important;
                border-left: 4px solid #FF6B6B !important;
                border-radius: 4px !important;
            }
            
            .stSuccess {
                background-color: #0E2E1F !important;
                border-left: 4px solid #00D4AA !important;
                border-radius: 4px !important;
            }
            
            /* Input fields */
            .stTextInput > div > div > input {
                background-color: #1E2229 !important;
                color: #FAFAFA !important;
                border: 1px solid #3E4449 !important;
                border-radius: 6px !important;
            }
            
            .stSelectbox > div > div > select {
                background-color: #1E2229 !important;
                color: #FAFAFA !important;
                border: 1px solid #3E4449 !important;
            }
            
            /* Buttons */
            .stButton > button {
                background-color: #00D4AA !important;
                color: #0E1117 !important;
                font-weight: 600 !important;
                border-radius: 6px !important;
                border: none !important;
                padding: 0.5rem 1.5rem !important;
                transition: all 0.3s ease !important;
            }
            
            .stButton > button:hover {
                background-color: #00B894 !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 8px rgba(0, 212, 170, 0.3) !important;
            }
            
            /* Code blocks */
            code {
                background-color: #1E2229 !important;
                color: #00D4AA !important;
                border: 1px solid #3E4449 !important;
                border-radius: 4px !important;
                padding: 2px 6px !important;
            }
            
            /* Spinner */
            .stSpinner > div {
                border-color: #00D4AA !important;
            }
        </style>
    """
    st.markdown(app_style, unsafe_allow_html=True)


def empty_lines(n: int) -> None:
    """Insert n empty lines in Streamlit app."""
    for _ in range(n):
        st.write("")


def generate_card(text: str, icon: str = "📊") -> None:
    """Create a styled card with icon and title for dark theme."""
    card_html = f"""
        <div style='
            background: linear-gradient(135deg, #1E2229 0%, #262730 100%);
            border: 1px solid #3E4449;
            border-radius: 12px;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            margin: 10px 0;
        '>
            <span style='font-size: 32px; margin-right: 15px;'>{icon}</span>
            <h3 style='
                text-align: center;
                color: #FAFAFA;
                font-family: Inter, system-ui, -apple-system, sans-serif;
                font-weight: 600;
                margin: 0;
            '>{text}</h3>
        </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def get_delta(df: pd.DataFrame, key: str) -> str:
    """Calculate percentage change between two most recent values."""
    if key not in df.columns:
        return "N/A"
    
    if len(df) < 2:
        return "N/A"
    
    latest_value = df[key].iloc[0]
    previous_value = df[key].iloc[1]
    
    if latest_value <= 0 or previous_value <= 0:
        delta = (previous_value - latest_value) / abs(latest_value) * 100
    else:
        delta = (previous_value - latest_value) / latest_value * 100
    
    return f"{delta:.2f}%"


def color_highlighter(val: str) -> str:
    """Apply color styling to DataFrame cells based on value for dark theme."""
    if isinstance(val, str) and val.startswith('-'):
        return 'color: #FF6B6B;'
    return None


def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dark theme styling to a DataFrame."""
    styled = df.style.format({
        col: lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) and x > 1000 else f"{x:.2f}" 
        for col in df.select_dtypes(include=[np.number]).columns
    })
    return styled


def load_index_cache(index_slug: str, cache_dir: str = "data/index_cache") -> Optional[Dict[str, Any]]:
    """
    Load cached index data from disk.
    
    Args:
        index_slug: Index slug (e.g., 'gspc')
        cache_dir: Cache directory
    
    Returns:
        Dict with cached data or None if not found
    """
    cache_path = Path(cache_dir) / index_slug
    
    if not cache_path.exists():
        return None
    
    try:
        cache = {}
        
        # Load metadata
        meta_path = cache_path / 'meta.json'
        if meta_path.exists():
            import json
            with open(meta_path, 'r') as f:
                cache['meta'] = json.load(f)
        
        # Load sector groups
        sector_path = cache_path / 'sector_groups.json'
        if sector_path.exists():
            import json
            with open(sector_path, 'r') as f:
                cache['sector_groups'] = json.load(f)
        
        # Load industry groups
        industry_path = cache_path / 'industry_groups.json'
        if industry_path.exists():
            import json
            with open(industry_path, 'r') as f:
                cache['industry_groups'] = json.load(f)
        
        # Load universe features
        universe_path = cache_path / 'merged' / 'universe_features.parquet'
        if universe_path.exists():
            cache['universe_features'] = pd.read_parquet(universe_path)
        
        # Load constituents
        constituents_path = cache_path / 'constituents.parquet'
        if constituents_path.exists():
            cache['constituents'] = pd.read_parquet(constituents_path)
        
        # Load enriched data (for individual stock lookups)
        enriched_path = cache_path / 'enriched_data.json'
        if enriched_path.exists():
            import json
            try:
                with open(enriched_path, 'r', encoding='utf-8') as f:
                    enriched_data = json.load(f)
                    if isinstance(enriched_data, list):
                        cache['enriched_data'] = enriched_data
            except Exception as e:
                # If enriched_data fails to load, continue without it
                pass
        
        return cache if cache else None
    
    except Exception as e:
        st.error(f"Error loading cache: {e}")
        return None


def get_available_indexes(cache_dir: str = "data/index_cache") -> Dict[str, str]:
    """
    Get list of available cached indexes.
    
    Args:
        cache_dir: Cache directory
    
    Returns:
        Dict mapping index_slug -> index_name
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return {}
    
    indexes = {}
    
    for index_dir in cache_path.iterdir():
        if index_dir.is_dir():
            meta_path = index_dir / 'meta.json'
            if meta_path.exists():
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    indexes[index_dir.name] = meta.get('index_name', index_dir.name)
    
    return indexes

