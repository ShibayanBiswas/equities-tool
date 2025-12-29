"""
Stock Detail Page - Comprehensive stock analysis
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
import numpy as np
from millify import millify
import plotly.graph_objs as go
import requests
import re
from datetime import datetime, timedelta

from src.utils import config_menu_footer, empty_lines, get_delta, color_highlighter, generate_card, load_index_cache
from src.data import (
    get_company_info, get_stock_price, get_income_statement,
    get_balance_sheet, get_cash_flow, get_key_metrics, get_financial_ratios
)
import json
from pathlib import Path
from typing import Optional, Dict
from src.components import (
    plot_stock_price, plot_net_income, plot_profitability_margins,
    plot_balance_sheet, plot_roe_roa, plot_cash_flows, create_plotly_config
)
from src.valuation import get_dcf, calculate_intrinsic_value_metrics

# Import sentiment with error handling
try:
    from src.sentiment import get_company_news_sentiment, get_aggregate_sentiment
except ImportError as e:
    st.warning(f"Sentiment analysis module could not be loaded: {e}")
    # Create dummy functions
    def get_company_news_sentiment(*args, **kwargs):
        return pd.DataFrame()
    def get_aggregate_sentiment(*args, **kwargs):
        return {'avg_compound': 0.0, 'avg_pos': 0.0, 'avg_neu': 0.0, 'avg_neg': 0.0, 'news_count': 0}

from src.analyst import get_analyst_estimates, get_rating
from src.earnings import get_earnings_surprises
from src.institutional import (
    get_institutional_holders, get_institutional_holders_historical,
    get_institutional_summary
)
from src.fmp_client import FMPClient

st.set_page_config(
    page_title='Stock Analysis Dashboard',
    page_icon='📈',
    layout="wide",
)

config_menu_footer()

st.title("Stock Analysis Dashboard 📈")

# Get symbol from query params or input
symbol = st.query_params.get("symbol", "")
if not symbol:
    symbol = st.text_input("Enter a stock ticker", "").upper()

if not symbol:
    st.info("Enter a stock ticker symbol to view detailed analysis.")
    st.stop()

# Initialize client (reuse across calls)
# FMPClient will automatically try: provided key -> env var -> Streamlit secrets
# So we can just pass None and let it handle it
try:
    api_key = st.secrets.get("FMP_API_KEY")
except:
    api_key = None
client = FMPClient(api_key=api_key if api_key else None)

# Try to load from cache first
def load_ticker_from_cache(symbol: str) -> Optional[Dict]:
    """Load ticker data from index cache if available."""
    cache_dir = Path("data/index_cache")
    # Try all available indexes
    for index_dir in cache_dir.iterdir():
        if index_dir.is_dir():
            enriched_path = index_dir / 'enriched_data.json'
            if enriched_path.exists():
                try:
                    with open(enriched_path, 'r', encoding='utf-8') as f:
                        enriched_data = json.load(f)
                        if isinstance(enriched_data, list):
                            for ticker_data in enriched_data:
                                if isinstance(ticker_data, dict) and ticker_data.get('symbol') == symbol:
                                    return ticker_data
                except Exception:
                    continue
    return None

# Always fetch from API (skip cache)
try:
    # Fetch all data from API
    with st.spinner(f"🔄 Fetching {symbol} data from API..."):
        # Always fetch from API, skip cache
        cached_ticker_data = None
        if False:  # Disabled cache loading - always use API
            st.info(f"📦 Loading {symbol} from cache...")
            # Convert cached data to expected format
            profile = cached_ticker_data.get('profile', {})
            quote = cached_ticker_data.get('quote', {})
            company_data = {
                'Name': profile.get('companyName', symbol),
                'Symbol': symbol,
                'Price': quote.get('price') or profile.get('price', 0),
                'Price change': 0,  # Cache doesn't have change
                'Market Cap': profile.get('mktCap') or profile.get('marketCap', 0),
                'Sector': profile.get('sector', ''),
                'Industry': profile.get('industry', ''),
                'Website': profile.get('website', ''),
                'Currency': profile.get('currency') or quote.get('currency') or 'USD',
                'Exchange': profile.get('exchangeShortName') or profile.get('exchange') or quote.get('exchange') or 'NYSE',
            }
            
            # Convert list data back to DataFrames for compatibility
            # IMPORTANT: Preserve date indexes for plotting (like Stock Dashboard reference)
            def list_to_df(data):
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                    # Try to set date index if date column exists (for plotting)
                    # Priority: date > fiscalDateEnding > fiscalYear > calendarYear
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df = df.set_index('date').sort_index(ascending=False)
                    elif 'fiscalDateEnding' in df.columns:
                        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'], errors='coerce')
                        df = df.set_index('fiscalDateEnding').sort_index(ascending=False)
                    elif 'fiscalYear' in df.columns:
                        # Convert fiscalYear to datetime for index (preserves for plotting)
                        try:
                            # Try to convert fiscalYear to datetime
                            fiscal_years = pd.to_datetime(df['fiscalYear'].astype(str), format='%Y', errors='coerce')
                            df = df.set_index(fiscal_years).sort_index(ascending=False)
                            df.index.name = 'date'  # Rename for consistency
                        except:
                            # Fallback: use as string index
                            df = df.set_index('fiscalYear').sort_index(ascending=False)
                    elif 'calendarYear' in df.columns:
                        # Use calendarYear as index
                        try:
                            calendar_years = pd.to_datetime(df['calendarYear'].astype(str), format='%Y', errors='coerce')
                            df = df.set_index(calendar_years).sort_index(ascending=False)
                            df.index.name = 'date'
                        except:
                            df = df.set_index('calendarYear').sort_index(ascending=False)
                    return df
                elif isinstance(data, dict):
                    return pd.DataFrame([data])
                return pd.DataFrame()
            
            metrics_data = list_to_df(cached_ticker_data.get('key_metrics', []))
            ratios_data = list_to_df(cached_ticker_data.get('ratios', []))
            income_data = list_to_df(cached_ticker_data.get('income_statement', []))
            balance_sheet_data = list_to_df(cached_ticker_data.get('balance_sheet', []))
            cashflow_data = list_to_df(cached_ticker_data.get('cashflow', []))
            
            # For data not in cache, still fetch from API
            try:
                performance_data = get_stock_price(symbol, years=5, client=client)
            except:
                performance_data = pd.DataFrame()
            
            # Optional data - try API but don't fail if missing
            try:
                dcf_data = get_dcf(symbol, client=client)
                valuation_metrics = calculate_intrinsic_value_metrics(dcf_data)
            except:
                dcf_data = pd.DataFrame()
                valuation_metrics = {}
            
            try:
                analyst_estimates = get_analyst_estimates(symbol, limit=5, client=client)
            except Exception as e:
                st.warning(f"Could not fetch analyst estimates: {str(e)}")
                analyst_estimates = pd.DataFrame()
            
            # Rating is optional - don't fail if it's missing
            try:
                rating = get_rating(symbol, client=client)
            except Exception as e:
                rating = None
                # Don't show error for rating - it's optional
            
            try:
                earnings_surprises = get_earnings_surprises(symbol, limit=20, client=client)
            except Exception as e:
                st.warning(f"Could not fetch earnings surprises: {str(e)}")
                earnings_surprises = pd.DataFrame()
            
            try:
                news_sentiment = get_company_news_sentiment(symbol, client=client)
            except Exception as e:
                st.warning(f"Could not fetch news sentiment: {str(e)}")
                news_sentiment = pd.DataFrame()
            
            # Fetch press releases
            try:
                press_releases = client.get_press_releases(symbol, limit=100)
            except Exception as e:
                press_releases = pd.DataFrame()
            
            try:
                aggregate_sentiment = get_aggregate_sentiment(symbol, client=client)
            except Exception as e:
                aggregate_sentiment = {'avg_compound': 0.0, 'avg_pos': 0.0, 'avg_neu': 0.0, 'avg_neg': 0.0, 'news_count': 0}
            
            try:
                institutional_holders = get_institutional_holders(symbol, client=client)
            except Exception as e:
                st.warning(f"Could not fetch institutional holders: {str(e)}")
                institutional_holders = pd.DataFrame()
            
            try:
                institutional_historical = get_institutional_holders_historical(symbol, years=5, client=client)
            except Exception as e:
                institutional_historical = pd.DataFrame()
            
            try:
                institutional_summary = get_institutional_summary(symbol, client=client)
            except Exception as e:
                institutional_summary = {}
        else:
            # No cache - fetch from API
            try:
                company_data = get_company_info(symbol, client)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [403, 401]:
                    st.error(f"""
                    **API Access Error (HTTP {e.response.status_code})**
                    
                    Your FMP API key appears to be invalid, expired, or doesn't have access to this endpoint.
                    
                    **To fix this:**
                    1. Check your API key in the `.env` file or Streamlit secrets
                    2. Verify your API key is active at [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/)
                    3. Ensure your API plan includes access to the required endpoints
                    4. Check if you've exceeded your API quota
                    
                    **Current API Key:** `{client.api_key[:10]}...` (first 10 chars)
                    """)
                    st.stop()
                else:
                    raise
            metrics_data = get_key_metrics(symbol, limit=5, client=client)
            income_data = get_income_statement(symbol, limit=5, client=client)
            performance_data = get_stock_price(symbol, years=5, client=client)
            ratios_data = get_financial_ratios(symbol, limit=5, client=client)
            balance_sheet_data = get_balance_sheet(symbol, limit=5, client=client)
            cashflow_data = get_cash_flow(symbol, limit=5, client=client)
            
            # Valuation data
            dcf_data = get_dcf(symbol, client=client)
            valuation_metrics = calculate_intrinsic_value_metrics(dcf_data)
            
            # Analyst data
            try:
                analyst_estimates = get_analyst_estimates(symbol, limit=5, client=client)
            except Exception as e:
                st.warning(f"Could not fetch analyst estimates: {str(e)}")
                analyst_estimates = pd.DataFrame()
            
            try:
                rating = get_rating(symbol, client=client)
            except Exception as e:
                rating = None
            
            try:
                earnings_surprises = get_earnings_surprises(symbol, limit=20, client=client)
            except Exception as e:
                st.warning(f"Could not fetch earnings surprises: {str(e)}")
                earnings_surprises = pd.DataFrame()
            
            # Sentiment data
            try:
                news_sentiment = get_company_news_sentiment(symbol, client=client)
            except Exception as e:
                st.warning(f"Could not fetch news sentiment: {str(e)}")
                news_sentiment = pd.DataFrame()
            
            # Fetch press releases
            try:
                press_releases = client.get_press_releases(symbol, limit=100)
            except Exception as e:
                press_releases = pd.DataFrame()
            
            try:
                aggregate_sentiment = get_aggregate_sentiment(symbol, client=client)
            except Exception as e:
                aggregate_sentiment = {'avg_compound': 0.0, 'avg_pos': 0.0, 'avg_neu': 0.0, 'avg_neg': 0.0, 'news_count': 0}
            
            # Institutional data
            try:
                institutional_holders = get_institutional_holders(symbol, client=client)
            except Exception as e:
                st.warning(f"Could not fetch institutional holders: {str(e)}")
                institutional_holders = pd.DataFrame()
            
            try:
                institutional_historical = get_institutional_holders_historical(symbol, years=5, client=client)
            except Exception as e:
                institutional_historical = pd.DataFrame()
            
            try:
                institutional_summary = get_institutional_summary(symbol, client=client)
            except Exception as e:
                institutional_summary = {}
    
    if not company_data:
        st.error(f'Unable to retrieve data for {symbol}. Please verify the symbol and try again.')
        st.stop()
    
    # Initialize additional data variables if not already set
    if 'company_description' not in locals():
        company_description = None
    if 'historical_performance' not in locals():
        historical_performance = pd.DataFrame()
    if 'social_sentiment' not in locals():
        social_sentiment = pd.DataFrame()
    if 'analyst_grades' not in locals():
        analyst_grades = pd.DataFrame()
    
    # Fetch additional data (always fetch, even if using cache)
    # Company description
    if not cached_ticker_data:
        try:
            outlook = client.get_company_outlook(symbol)
            if outlook and 'profile' in outlook:
                company_description = outlook['profile'].get('description', '')
        except:
            pass
    
    # Social sentiment - always fetch
    try:
        social_data = client._get("v4/historical/social-sentiment", params={"symbol": symbol, "page": 0})
        if social_data:
            social_sentiment = pd.DataFrame(social_data)
            if 'date' in social_sentiment.columns:
                social_sentiment['date'] = pd.to_datetime(social_sentiment['date'])
                social_sentiment = social_sentiment.set_index('date').sort_index(ascending=False)
    except Exception as e:
        # Silently fail for social sentiment
        pass
    
    # Analyst grades - always fetch (critical for Analyst Estimate Analysis section)
    try:
        # Try v3/grade/{symbol} first (as per Company Sentiment.ipynb)
        grades_data = client._get(f"v3/grade/{symbol}", params={"limit": 500})
        if not grades_data:
            # Fallback: try alternative endpoint
            try:
                grades_data = client._get(f"grade/{symbol}", params={"limit": 500})
            except:
                pass
        
        if grades_data and isinstance(grades_data, list) and len(grades_data) > 0:
            analyst_grades = pd.DataFrame(grades_data)
            if 'date' in analyst_grades.columns:
                analyst_grades['date'] = pd.to_datetime(analyst_grades['date'], errors='coerce')
                # Remove rows with invalid dates
                analyst_grades = analyst_grades.dropna(subset=['date'])
                if not analyst_grades.empty:
                    analyst_grades = analyst_grades.set_index('date').sort_index(ascending=False)
            elif not analyst_grades.empty:
                # If no date column, keep as is
                pass
        elif grades_data and isinstance(grades_data, dict):
            # Single record returned as dict
            analyst_grades = pd.DataFrame([grades_data])
    except Exception as e:
        # Log error but don't show to user - analyst grades are optional
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Could not fetch analyst grades for {symbol}: {str(e)}")
        analyst_grades = pd.DataFrame()
    
    # Company header
    empty_lines(1)
    col1, col2 = st.columns((8.5, 1.5))
    with col1:
        generate_card(company_data['Name'])
    with col2:
        if company_data.get('Website'):
            image_html = f"<a href='{company_data['Website']}' target='_blank'>{company_data['Name']}</a>"
            st.markdown(image_html, unsafe_allow_html=True)
    
    # Company Description
    if company_description:
        with st.expander("📝 Company Description", expanded=False):
            st.write(company_description)
    
    # Key metrics row
    col3, col4, col5, col6, col7 = st.columns((0.2, 1.4, 1.4, 2, 2.6))
    
    with col4:
        empty_lines(1)
        st.metric(label="Price", value=f"${company_data['Price']:.2f}", 
                 delta=f"{company_data['Price change']:.2f}")
        empty_lines(2)
    
    with col5:
        empty_lines(1)
        generate_card(company_data.get('Currency', 'USD'))
        empty_lines(2)
    
    with col6:
        empty_lines(1)
        generate_card(company_data.get('Exchange', 'NYSE'))
        empty_lines(2)
    
    with col7:
        empty_lines(1)
        generate_card(company_data['Sector'])
        empty_lines(2)
    
    # Financial metrics
    col8, col9, col10 = st.columns((2, 2, 3))
    
    # Helper function to safely get metric value with multiple possible column names
    def get_metric_value_safe(df, possible_names, default=0):
        if df.empty:
            return default
        for name in possible_names:
            if name in df.columns:
                val = df[name].iloc[0]
                if pd.notna(val):
                    return val
        return default
    
    with col8:
        empty_lines(3)
        # Market Cap - try multiple column names
        market_cap = get_metric_value_safe(metrics_data, ['marketCap', 'Market Cap', 'mktCap', 'market_cap'], 
                                     company_data.get('Market Cap', 0))
        if market_cap:
            delta_val = None
            if 'marketCap' in metrics_data.columns:
                delta_val = get_delta(metrics_data, 'marketCap')
            elif 'Market Cap' in metrics_data.columns:
                delta_val = get_delta(metrics_data, 'Market Cap')
            st.metric(label="Market Cap", 
                     value=millify(market_cap, precision=2),
                     delta=delta_val)
            st.write("")
        
        # DCF Value (right below Market Cap)
        if not dcf_data.empty:
            latest_dcf = dcf_data.iloc[0]
            dcf_value = latest_dcf.get('dcf', 0)
            if dcf_value and dcf_value > 0:
                st.metric(label="DCF Value", value=f"${dcf_value:,.2f}")
            st.write("")
        
        # D/E Ratio - check both metrics and ratios
        de_ratio = get_metric_value_safe(metrics_data, ['debtToEquity', 'D/E ratio', 'debtEquityRatio', 'Debt to Equity'])
        if not de_ratio and not ratios_data.empty:
            de_ratio = get_metric_value_safe(ratios_data, ['debtEquityRatio', 'Debt to Equity', 'D/E ratio', 'debtToEquity'])
        if de_ratio:
            st.metric(label="D/E Ratio", 
                     value=round(de_ratio, 2),
                     delta=get_delta(metrics_data, 'debtToEquity') if 'debtToEquity' in metrics_data.columns else None)
            st.write("")
        
        # ROE - check ratios data
        roe = get_metric_value_safe(ratios_data, ['returnOnEquity', 'ROE', 'Return on Equity', 'roe']) if not ratios_data.empty else None
        if roe:
            st.metric(label="Return on Equity", 
                     value=f"{round(roe * 100, 2)}%",
                     delta=get_delta(ratios_data, 'returnOnEquity') if 'returnOnEquity' in ratios_data.columns else None)
    
    with col9:
        empty_lines(3)
        # Working Capital - try multiple column names
        working_capital = get_metric_value_safe(metrics_data, ['workingCapital', 'Working Capital', 'working_capital'])
        if working_capital:
            delta_val = None
            if 'workingCapital' in metrics_data.columns:
                delta_val = get_delta(metrics_data, 'workingCapital')
            elif 'Working Capital' in metrics_data.columns:
                delta_val = get_delta(metrics_data, 'Working Capital')
            st.metric(label="Working Capital", 
                     value=millify(working_capital, precision=2),
                     delta=delta_val)
            st.write("")
        
        # P/E Ratio - check both metrics and ratios
        pe_ratio = get_metric_value_safe(metrics_data, ['peRatio', 'P/E Ratio', 'priceEarningsRatio', 'PE Ratio'])
        if not pe_ratio and not ratios_data.empty:
            pe_ratio = get_metric_value_safe(ratios_data, ['priceEarningsRatio', 'Price Earnings Ratio', 'P/E Ratio', 'peRatio'])
        if pe_ratio:
            st.metric(label="P/E Ratio", 
                     value=round(pe_ratio, 2),
                     delta=get_delta(metrics_data, 'peRatio') if 'peRatio' in metrics_data.columns else None)
            st.write("")
        
        # Dividend Yield - check ratios data
        div_yield = get_metric_value_safe(ratios_data, ['dividendYield', 'Dividend Yield', 'dividend_yield']) if not ratios_data.empty else 0
        if div_yield == 0:
            st.metric(label="Dividends (yield)", value='0%')
        else:
            delta_val = None
            if 'dividendYield' in ratios_data.columns:
                delta_val = get_delta(ratios_data, 'dividendYield')
            elif 'Dividend Yield' in ratios_data.columns:
                delta_val = get_delta(ratios_data, 'Dividend Yield')
            st.metric(label="Dividends (yield)", 
                     value=f"{round(div_yield * 100, 2)}%",
                     delta=delta_val)
    
    with col10:
        if not income_data.empty:
            st.markdown('**Income Statement**')
            # Transpose for display (like Stock Dashboard reference)
            income_statement_data = income_data.T
            
            # Filter out non-numeric columns and get valid year columns
            # When transposed, dates become column names - filter valid ones
            year_columns = []
            for col in income_statement_data.columns:
                # Accept datetime objects (Timestamps), years (int/float), or date strings
                if isinstance(col, pd.Timestamp):
                    year_columns.append(col)
                elif isinstance(col, (int, float)) and str(col).strip() != '0':
                    year_columns.append(col)
                elif isinstance(col, str) and (col.strip() != '0' and col not in ['symbol', 'reportedCurrency', 'cik', 'filingDate', 'acceptedDate', 'period']):
                    year_columns.append(col)
            
            if not year_columns:
                # If no valid columns, use all columns except metadata
                year_columns = [col for col in income_statement_data.columns 
                              if col not in ['symbol', 'reportedCurrency', 'cik', 'filingDate', 'acceptedDate', 'period']]
            
            if year_columns:
                year = st.selectbox('All numbers in thousands', year_columns, 
                                  index=0, label_visibility='collapsed', key='income_year')
                income_statement_data = income_statement_data.loc[:, [year]]
            else:
                # Fallback: use first column if available
                if len(income_statement_data.columns) > 0:
                    year = income_statement_data.columns[0]
                    income_statement_data = income_statement_data.loc[:, [year]]
            
            # Only format numeric columns, skip non-numeric columns like 'symbol', 'date', etc.
            # Create a copy to avoid modifying the original
            income_display = income_statement_data.copy()
            
            # Convert column name to string if it's a Timestamp to avoid PyArrow issues
            if len(income_display.columns) > 0:
                col_name = income_display.columns[0]
                if isinstance(col_name, pd.Timestamp):
                    income_display.columns = [str(col_name)]
                elif not isinstance(col_name, str):
                    income_display.columns = [str(col_name)]
            
            # Format numeric values as strings to avoid type mixing
            numeric_cols = income_display.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                income_display[col] = income_display[col].apply(
                    lambda x: millify(x, precision=2) if pd.notna(x) and isinstance(x, (int, float)) else str(x) if pd.notna(x) else ''
                )
            
            # Convert all columns to string type to ensure Arrow compatibility
            for col in income_display.columns:
                income_display[col] = income_display[col].astype(str)
            
            # Ensure index is also string type to avoid Arrow issues
            income_display.index = income_display.index.astype(str)
            
            income_styled = income_display.style.map(color_highlighter)
            headers = {
                'selector': 'th:not(.index_name)',
                'props': [('color', 'black')]
            }
            income_styled.set_table_styles([headers])
            st.table(income_styled)
    
    # Economic Indicators Section
    st.header("📈 Economic Indicators")
    
    # List of economic indicators to fetch
    economic_indicators = [
        'GDP', 'realGDP', 'nominalPotentialGDP', 'realGDPPerCapita', 'federalFunds',
        'CPI', 'inflationRate', 'inflation', 'retailSales', 'consumerSentiment',
        'durableGoods', 'unemploymentRate', 'totalNonfarmPayroll', 'initialClaims',
        'industrialProductionTotalIndex', 'newPrivatelyOwnedHousingUnitsStartedTotalUnits',
        'totalVehicleSales', 'retailMoneyFunds', 'smoothedUSRecessionProbabilities',
        '3MonthOr90DayRatesAndYieldsCertificatesOfDeposit',
        'commercialBankInterestRateOnCreditCardPlansAllAccounts',
        '30YearFixedRateMortgageAverage', '15YearFixedRateMortgageAverage',
        'tradeBalanceGoodsAndServices'
    ]
    
    # Calculate date range (1 year)
    to_date = pd.Timestamp.now()
    from_date = to_date - pd.Timedelta(days=365)
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')
    
    # Fetch economic indicators
    economic_data = []
    try:
        # Get market risk premium
        market_risk_df = client.get_market_risk_premium()
        market_risk_us = None
        if not market_risk_df.empty:
            # Find US data
            us_data = market_risk_df[market_risk_df['country'] == 'United States']
            if not us_data.empty:
                market_risk_us = us_data.iloc[0]
    except Exception as e:
        market_risk_us = None
    
    # Define formatting rules for each indicator based on the example provided
    def format_indicator_value(indicator_name: str, value: float) -> str:
        """Format indicator value with proper units based on indicator type."""
        if pd.isna(value):
            return 'N/A'
        
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return str(value)
        
        # GDP indicators - check if already in billions (value < 1e6) or needs conversion
        if indicator_name in ['GDP', 'realGDP', 'nominalPotentialGDP']:
            if abs(num_value) < 1e6:
                # Already in billions
                return f"${num_value:,.2f} billion"
            else:
                # Convert from raw dollars to billions
                return f"${num_value/1e9:,.2f} billion"
        
        # Real GDP per Capita - just dollar amount
        elif indicator_name == 'realGDPPerCapita':
            return f"${num_value:,.2f}"
        
        # Percentage indicators
        elif indicator_name in ['federalFunds', 'inflationRate', 'inflation', 'unemploymentRate',
                                'smoothedUSRecessionProbabilities', '3MonthOr90DayRatesAndYieldsCertificatesOfDeposit',
                                'commercialBankInterestRateOnCreditCardPlansAllAccounts',
                                '30YearFixedRateMortgageAverage', '15YearFixedRateMortgageAverage']:
            return f"{num_value:.2f}%"
        
        # CPI - just number
        elif indicator_name == 'CPI':
            return f"{num_value:.2f}"
        
        # Retail Sales and Durable Goods - check if already in millions or needs conversion
        elif indicator_name in ['retailSales', 'durableGoods', 'tradeBalanceGoodsAndServices']:
            if abs(num_value) < 1e9:
                # Already in millions
                return f"${num_value:,.0f} million"
            else:
                # Convert from raw dollars to millions
                return f"${num_value/1e6:,.0f} million"
        
        # Consumer Sentiment - just number
        elif indicator_name == 'consumerSentiment':
            return f"{num_value:.2f}"
        
        # Total Nonfarm Payroll and Initial Claims - comma separated numbers
        elif indicator_name in ['totalNonfarmPayroll', 'initialClaims']:
            return f"{num_value:,.0f}"
        
        # Industrial Production Index - just number
        elif indicator_name == 'industrialProductionTotalIndex':
            return f"{num_value:.2f}"
        
        # Housing Starts - comma separated
        elif indicator_name == 'newPrivatelyOwnedHousingUnitsStartedTotalUnits':
            return f"{num_value:,.0f}"
        
        # Vehicle Sales - check if already in millions or needs conversion
        elif indicator_name == 'totalVehicleSales':
            if abs(num_value) < 100:
                # Already in millions
                return f"{num_value:.2f} million units"
            else:
                # Convert from raw units to millions
                return f"{num_value/1e6:.2f} million units"
        
        # Retail Money Funds - check if already in billions or needs conversion
        elif indicator_name == 'retailMoneyFunds':
            if abs(num_value) < 1e6:
                # Already in billions
                return f"${num_value:,.2f} billion"
            else:
                # Convert from raw dollars to billions
                return f"${num_value/1e9:,.2f} billion"
        
        # Default formatting
        else:
            return f"{num_value:,.2f}"
    
    # Fetch each economic indicator
    for indicator_name in economic_indicators:
        try:
            indicator_df = client.get_economic_indicators(
                name=indicator_name,
                from_date=from_date_str,
                to_date=to_date_str
            )
            if not indicator_df.empty:
                # Get the latest value (most recent date)
                latest_value = indicator_df.iloc[0]
                # Get value column - check multiple possible column names
                value_col = None
                possible_cols = ['value', indicator_name, 'Value', 'Value_' + indicator_name]
                # Also check if any column contains 'value' (case insensitive)
                for col in indicator_df.columns:
                    if col.lower() in ['value', indicator_name.lower()]:
                        possible_cols.append(col)
                
                for col in possible_cols:
                    if col in latest_value.index or col in indicator_df.columns:
                        value_col = col
                        break
                
                # If still no column found, try to get the first numeric column
                if not value_col:
                    for col in indicator_df.columns:
                        if col not in ['date', 'Date', 'symbol', 'Symbol']:
                            try:
                                test_val = latest_value[col] if col in latest_value.index else indicator_df[col].iloc[0]
                                if pd.notna(test_val) and isinstance(test_val, (int, float)):
                                    value_col = col
                                    break
                            except:
                                continue
                
                if value_col:
                    # Get value from the row
                    try:
                        if value_col in latest_value.index:
                            value = latest_value[value_col]
                        elif value_col in indicator_df.columns:
                            value = indicator_df[value_col].iloc[0]
                        else:
                            value = None
                        
                        # Ensure we have a valid numeric value
                        if value is not None and pd.notna(value):
                            # Try to convert to float if it's a string
                            if isinstance(value, str):
                                try:
                                    value = float(value.replace(',', ''))
                                except:
                                    pass
                            
                            formatted_value = format_indicator_value(indicator_name, value)
                            
                            economic_data.append({
                                'Indicator': indicator_name,
                                'Latest Value': formatted_value
                            })
                    except Exception as e:
                        # If extraction fails, try to get any numeric value from the row
                        try:
                            for col in indicator_df.columns:
                                if col not in ['date', 'Date', 'symbol', 'Symbol', 'name', 'Name']:
                                    test_val = latest_value[col] if col in latest_value.index else indicator_df[col].iloc[0]
                                    if pd.notna(test_val):
                                        try:
                                            num_val = float(test_val) if not isinstance(test_val, (int, float)) else test_val
                                            formatted_value = format_indicator_value(indicator_name, num_val)
                                            economic_data.append({
                                                'Indicator': indicator_name,
                                                'Latest Value': formatted_value
                                            })
                                            break
                                        except:
                                            continue
                        except:
                            continue
        except Exception as e:
            # Skip if indicator fails
            continue
    
    # Add market risk premium if available
    if market_risk_us is not None:
        country_risk = market_risk_us.get('countryRiskPremium', None)
        equity_risk = market_risk_us.get('totalEquityRiskPremium', None)
        country_risk_str = f"{country_risk:.2f}%" if pd.notna(country_risk) else 'N/A'
        equity_risk_str = f"{equity_risk:.2f}%" if pd.notna(equity_risk) else 'N/A'
        economic_data.append({
            'Indicator': 'Market Risk Premium (US)',
            'Latest Value': f"Country Risk: {country_risk_str}, Total Equity Risk: {equity_risk_str}"
        })
    
    if economic_data:
        economic_df = pd.DataFrame(economic_data)
        st.dataframe(economic_df, width='stretch', hide_index=True)
    else:
        st.info("Economic indicators data not available.")
    
    # Charts section
    st.header("📊 Financial Charts")
    
    # Time range selector
    col_range1, col_range2 = st.columns(2)
    with col_range1:
        years = st.selectbox("Time Range", [5, 3, 1], index=0, key='time_range')
    
    # Market performance
    if not performance_data.empty:
        st.subheader("📈 Stock Price Performance")
        plot_stock_price(performance_data, symbol, years=years)
    else:
        st.info("Price data not available for charting")
    
    # Net income
    if not income_data.empty:
        st.subheader("💰 Net Income Trend")
        # Map FMP API column names to expected names for plotting
        income_for_plot = income_data.copy()
        # FMP API uses 'netIncome' as the column name
        if 'netIncome' in income_for_plot.columns and '= Net Income' not in income_for_plot.columns:
            income_for_plot['= Net Income'] = income_for_plot['netIncome']
        elif 'net_income' in income_for_plot.columns and '= Net Income' not in income_for_plot.columns:
            income_for_plot['= Net Income'] = income_for_plot['net_income']
        elif 'Net Income' in income_for_plot.columns and '= Net Income' not in income_for_plot.columns:
            income_for_plot['= Net Income'] = income_for_plot['Net Income']
        plot_net_income(income_for_plot, symbol)
    
    # Profitability margins
    if not ratios_data.empty:
        st.subheader("📊 Profitability Margins")
        # Map FMP API column names to expected names for plotting
        ratios_for_plot = ratios_data.copy()
        column_mapping = {
            'grossProfitMargin': 'Gross Profit Margin',
            'operatingProfitMargin': 'Operating Profit Margin',
            'netProfitMargin': 'Net Profit Margin',
            'returnOnEquity': 'Return on Equity',
            'returnOnAssets': 'Return on Assets',
        }
        for old_col, new_col in column_mapping.items():
            if old_col in ratios_for_plot.columns and new_col not in ratios_for_plot.columns:
                ratios_for_plot[new_col] = ratios_for_plot[old_col]
        plot_profitability_margins(ratios_for_plot, symbol)
    
    # Balance sheet
    if not balance_sheet_data.empty:
        st.subheader("📋 Balance Sheet Overview")
        # Map FMP API column names to expected names for plotting
        # Based on Stock Dashboard reference: totalAssets -> Assets, totalLiabilities -> Liabilities, totalEquity -> Equity
        balance_for_plot = balance_sheet_data.copy()
        
        # Direct mapping like Stock Dashboard does
        if 'totalAssets' in balance_for_plot.columns:
            balance_for_plot['Assets'] = balance_for_plot['totalAssets']
        if 'totalLiabilities' in balance_for_plot.columns:
            balance_for_plot['Liabilities'] = balance_for_plot['totalLiabilities']
        if 'totalEquity' in balance_for_plot.columns:
            balance_for_plot['Equity'] = balance_for_plot['totalEquity']
        elif 'totalStockholdersEquity' in balance_for_plot.columns:
            balance_for_plot['Equity'] = balance_for_plot['totalStockholdersEquity']
        
        # Check if required columns exist after mapping
        if 'Assets' in balance_for_plot.columns and 'Liabilities' in balance_for_plot.columns and 'Equity' in balance_for_plot.columns:
            plot_balance_sheet(balance_for_plot, symbol)
        else:
            # Show what we have for debugging
            available_cols = list(balance_for_plot.columns)
            st.warning(f"Balance sheet data missing required columns. Looking for: Assets, Liabilities, Equity. Available columns: {available_cols[:15]}")
    
    # ROE and ROA
    # ROE and ROA are in key_metrics, not ratios - use metrics_data
    if not metrics_data.empty:
        st.subheader("📈 Return on Equity & Return on Assets Analysis")
        # Create a dataframe with ROE and ROA from metrics_data
        roe_roa_data = metrics_data.copy()
        
        # Map column names
        column_mapping = {
            'returnOnEquity': 'Return on Equity',
            'returnOnAssets': 'Return on Assets',
        }
        for old_col, new_col in column_mapping.items():
            if old_col in roe_roa_data.columns and new_col not in roe_roa_data.columns:
                roe_roa_data[new_col] = roe_roa_data[old_col]
        
        # Check if required columns exist
        if 'Return on Equity' in roe_roa_data.columns and 'Return on Assets' in roe_roa_data.columns:
            plot_roe_roa(roe_roa_data, symbol)
        else:
            st.info("Return on Equity and Return on Assets data not available. These metrics require key metrics data.")
    
    # Cash flows
    if not cashflow_data.empty:
        st.subheader("💵 Cash Flow Analysis")
        # Map FMP API column names to expected names for plotting
        # Based on actual FMP API response structure
        cashflow_for_plot = cashflow_data.copy()
        # Based on Stock Dashboard reference - exact column names from data.py
        # netCashProvidedByOperatingActivities -> Cash flows from operating activities
        # netCashUsedForInvestingActivites -> Cash flows from investing activities (note: API has typo "Activites")
        # netCashUsedProvidedByFinancingActivities -> Cash flows from financing activities
        # freeCashFlow -> Free cash flow
        if 'netCashProvidedByOperatingActivities' in cashflow_for_plot.columns:
            cashflow_for_plot['Cash flows from operating activities'] = cashflow_for_plot['netCashProvidedByOperatingActivities']
        elif 'operatingCashFlow' in cashflow_for_plot.columns:
            cashflow_for_plot['Cash flows from operating activities'] = cashflow_for_plot['operatingCashFlow']
        
        # Try the exact key from Stock Dashboard (with typo)
        if 'netCashUsedForInvestingActivites' in cashflow_for_plot.columns:
            cashflow_for_plot['Cash flows from investing activities'] = cashflow_for_plot['netCashUsedForInvestingActivites']
        elif 'netCashProvidedByInvestingActivities' in cashflow_for_plot.columns:
            cashflow_for_plot['Cash flows from investing activities'] = cashflow_for_plot['netCashProvidedByInvestingActivities']
        elif 'netCashUsedForInvestingActivities' in cashflow_for_plot.columns:
            cashflow_for_plot['Cash flows from investing activities'] = cashflow_for_plot['netCashUsedForInvestingActivities']
        
        # Try the exact key from Stock Dashboard
        if 'netCashUsedProvidedByFinancingActivities' in cashflow_for_plot.columns:
            cashflow_for_plot['Cash flows from financing activities'] = cashflow_for_plot['netCashUsedProvidedByFinancingActivities']
        elif 'netCashProvidedByFinancingActivities' in cashflow_for_plot.columns:
            cashflow_for_plot['Cash flows from financing activities'] = cashflow_for_plot['netCashProvidedByFinancingActivities']
        elif 'netCashUsedForFinancingActivities' in cashflow_for_plot.columns:
            cashflow_for_plot['Cash flows from financing activities'] = cashflow_for_plot['netCashUsedForFinancingActivities']
        
        if 'freeCashFlow' in cashflow_for_plot.columns:
            cashflow_for_plot['Free cash flow'] = cashflow_for_plot['freeCashFlow']
        
        # Check if required columns exist after mapping
        required_cols = ['Cash flows from operating activities', 'Cash flows from investing activities', 
                        'Cash flows from financing activities', 'Free cash flow']
        if all(col in cashflow_for_plot.columns for col in required_cols):
            plot_cash_flows(cashflow_for_plot, symbol)
        else:
            missing = [col for col in required_cols if col not in cashflow_for_plot.columns]
            available_cols = list(cashflow_for_plot.columns)
            st.warning(f"Cash flow data missing required columns: {missing}. Available columns: {available_cols[:20]}")
    
    # Historical Performance vs Index
    if not performance_data.empty:
        st.header("📈 Historical Performance vs S&P 500")
        try:
            # Get S&P 500 performance for comparison
            index_symbol = '^GSPC'
            index_performance = get_stock_price(index_symbol, years=5, client=client)
            if not index_performance.empty:
                # Calculate cumulative returns - try different column names
                stock_price_col = 'Price' if 'Price' in performance_data.columns else ('adjClose' if 'adjClose' in performance_data.columns else 'close')
                index_price_col = 'Price' if 'Price' in index_performance.columns else ('adjClose' if 'adjClose' in index_performance.columns else 'close')
                
                if stock_price_col in performance_data.columns and index_price_col in index_performance.columns:
                    stock_returns = performance_data[stock_price_col].pct_change().dropna()
                    index_returns = index_performance[index_price_col].pct_change().dropna()
                else:
                    raise ValueError("Price column not found")
                
                # Align dates
                common_dates = stock_returns.index.intersection(index_returns.index)
                if len(common_dates) > 0:
                    stock_cumulative = (1 + stock_returns.loc[common_dates]).cumprod()
                    index_cumulative = (1 + index_returns.loc[common_dates]).cumprod()
                    
                    # Create comparison chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                        x=common_dates,
                        y=stock_cumulative * 100,
                        mode='lines',
                        name=f'{symbol}',
                        line=dict(color='#00D4AA', width=2)
                ))
                fig.add_trace(go.Scatter(
                        x=common_dates,
                        y=index_cumulative * 100,
                        mode='lines',
                        name='S&P 500',
                        line=dict(color='#FF6B6B', width=2)
                ))
                fig.update_layout(
                    title=f'{symbol} vs S&P 500 Cumulative Returns',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return (%)',
                    hovermode='x unified',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.info("Index comparison data not available")
    
    # News Sentiment Analysis Section
    st.header("📰 News Sentiment Analysis")
    
    # Top Articles
    if not news_sentiment.empty:
        st.subheader("📄 Top Articles")
        
        # Display top articles in a beautiful table
        articles_display = news_sentiment[['title', 'publishedDate', 'site', 'sentiment_compound', 'sentiment_pos', 'sentiment_neg']].copy()
        articles_display['publishedDate'] = pd.to_datetime(articles_display['publishedDate'], errors='coerce').dt.strftime('%Y-%m-%d')
        articles_display = articles_display.sort_values('sentiment_compound', ascending=False).head(20)
        
        # Rename columns for better display
        articles_display.columns = ['Title', 'Date', 'Source', 'Compound Score', 'Positive', 'Negative']
        
        # Format sentiment scores
        articles_display['Compound Score'] = articles_display['Compound Score'].apply(lambda x: f"{x:.3f}")
        articles_display['Positive'] = articles_display['Positive'].apply(lambda x: f"{x:.3f}")
        articles_display['Negative'] = articles_display['Negative'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(articles_display, width='stretch', hide_index=True)
    else:
        st.info("News sentiment data not available for this stock.")
    
    # Calculate and Display Sentiment Scores
    if not news_sentiment.empty:
        st.subheader("📊 Sentiment Score Calculations")
        
        # Calculate mean sentiment scores (like Company Sentiment.ipynb)
        sentiment_cols = ['sentiment_compound', 'sentiment_pos', 'sentiment_neu', 'sentiment_neg']
        available_cols = [col for col in sentiment_cols if col in news_sentiment.columns]
        if available_cols:
            sentiment_scores = news_sentiment[available_cols].mean()
        else:
            sentiment_scores = pd.Series({
                'sentiment_compound': 0.0,
                'sentiment_pos': 0.0,
                'sentiment_neu': 0.0,
                'sentiment_neg': 0.0
            })
        
        # Display in columns (as percentages)
        col_sent1, col_sent2, col_sent3, col_sent4 = st.columns(4)
        with col_sent1:
            st.metric("Average Compound Score", f"{sentiment_scores.get('sentiment_compound', 0.0) * 100:.2f}%")
        with col_sent2:
            st.metric("Average Positive", f"{sentiment_scores.get('sentiment_pos', 0.0) * 100:.2f}%")
        with col_sent3:
            st.metric("Average Neutral", f"{sentiment_scores.get('sentiment_neu', 0.0) * 100:.2f}%")
        with col_sent4:
            st.metric("Average Negative", f"{sentiment_scores.get('sentiment_neg', 0.0) * 100:.2f}%")
        
        # Plot Mean Sentiment (like Company Sentiment.ipynb)
        st.subheader("📈 Mean Sentiment Analysis")
        
        # Prepare data for bar chart
        scores_mean = pd.DataFrame({
            'Negative': [sentiment_scores.get('sentiment_neg', 0.0)],
            'Neutral': [sentiment_scores.get('sentiment_neu', 0.0)],
            'Positive': [sentiment_scores.get('sentiment_pos', 0.0)],
            'Compound': [sentiment_scores.get('sentiment_compound', 0.0)]
        }, index=[symbol])
        
        # Create bar chart
        fig = go.Figure()
        colors = {'Negative': '#F44336', 'Neutral': '#FFA500', 'Positive': '#4CAF50', 'Compound': '#00D4AA'}
        
        for col in ['Negative', 'Neutral', 'Positive', 'Compound']:
            fig.add_trace(go.Bar(
                name=col,
                x=[symbol],
                y=[scores_mean[col].iloc[0]],
                marker_color=colors[col],
                text=[f"{scores_mean[col].iloc[0]:.3f}"],
                textposition='auto',
            ))
        
        fig.update_layout(
            title=f'{symbol} Mean Sentiment Analysis',
            xaxis_title='Stock',
            yaxis_title='Sentiment Score',
            barmode='group',
            template='plotly_dark',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, config=create_plotly_config(), width='stretch')
    
    # Company Press Releases
    if 'press_releases' in locals() and not press_releases.empty:
        st.subheader("📢 Company Press Releases")
        
        # Display press releases
        press_display = press_releases.copy()
        if 'date' in press_display.columns:
            press_display['date'] = pd.to_datetime(press_display['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Select relevant columns
        display_cols = ['title', 'date', 'text']
        available_cols = [col for col in display_cols if col in press_display.columns]
        press_display = press_display[available_cols].head(20)
        
        # Rename for display
        if 'title' in press_display.columns:
            press_display = press_display.rename(columns={'title': 'Title', 'date': 'Date', 'text': 'Content'})
        
        # Truncate text for display
        if 'Content' in press_display.columns:
            press_display['Content'] = press_display['Content'].apply(lambda x: x[:200] + '...' if isinstance(x, str) and len(x) > 200 else x)
        
        st.dataframe(press_display, width='stretch', hide_index=True)
    else:
        st.info("Press releases data not available for this stock.")
    
    # Social Sentiment Analysis Section
    st.header("💬 Social Sentiment Analysis")
    
    if 'social_sentiment' in locals() and not social_sentiment.empty:
        # Filter for current week
        social_copy = social_sentiment.copy()
        # Reset index if date is in index to make it a column
        if isinstance(social_copy.index, pd.DatetimeIndex):
            social_copy = social_copy.reset_index()
            if 'index' in social_copy.columns:
                social_copy = social_copy.rename(columns={'index': 'date'})
        
        if 'date' in social_copy.columns:
            social_copy['date'] = pd.to_datetime(social_copy['date'], errors='coerce')
            current_week_start = pd.Timestamp.now() - pd.Timedelta(days=7)
            current_week_social = social_copy[social_copy['date'] >= current_week_start].copy()
        else:
            # If no date column, show all
            current_week_social = social_copy.copy()
        
        if not current_week_social.empty:
            # Calculate mean sentiment using groupby (like Company Sentiment.ipynb)
            if 'symbol' in current_week_social.columns:
                social_mean = current_week_social.groupby('symbol').mean()
            else:
                # If no symbol column, just calculate mean of numeric columns
                numeric_data = current_week_social.select_dtypes(include=[np.number])
                if not numeric_data.empty:
                    social_mean = numeric_data.mean()
                    social_mean = pd.DataFrame([social_mean], index=[symbol])
                else:
                    social_mean = pd.DataFrame()
            
            # Plot Mean Social Sentiment (like News Sentiment Analysis)
            
            # Prepare data for bar chart - use different colors from news sentiment
            if isinstance(social_mean, pd.Series):
                # Convert Series to DataFrame for consistent handling
                social_mean = pd.DataFrame([social_mean], index=[symbol])
            
            if isinstance(social_mean, pd.DataFrame) and len(social_mean) > 0:
                # Include specific columns for social sentiment
                social_cols_to_include = [
                    'stocktwitsPosts', 'twitterPosts', 'stocktwitsComments', 'twitterComments',
                    'stocktwitsLikes', 'twitterLikes', 'stocktwitsImpressions', 'twitterImpressions',
                    'stocktwitsSentiment', 'twitterSentiment'
                ]
                # Get all numeric columns, prioritizing the social columns
                all_numeric_cols = social_mean.select_dtypes(include=[np.number]).columns.tolist()
                # Filter to include social columns if they exist, plus other numeric columns
                numeric_cols = [col for col in social_cols_to_include if col in all_numeric_cols]
                # Add other numeric columns that aren't in the social list
                other_cols = [col for col in all_numeric_cols if col not in social_cols_to_include]
                numeric_cols.extend(other_cols)
                
                if len(numeric_cols) > 0:
                    # Create bar chart - each bar with different color
                    fig = go.Figure()
                    distinct_colors = ['#FF6B9D', '#C44569', '#F8B500', '#FFA07A', '#9B59B6', '#E74C3C', '#3498DB', '#1ABC9C', '#16A085', '#27AE60']
                    
                    for idx, col in enumerate(numeric_cols[:10]):  # Limit to 10 columns to include all social columns
                        if col in social_mean.columns:
                            value = social_mean[col].iloc[0] if len(social_mean) > 0 else 0
                            color = distinct_colors[idx % len(distinct_colors)]  # Cycle through colors
                            # Format column name for display
                            display_name = col.replace('Index', ' Index').replace('sentiment', 'Sentiment').title()
                            fig.add_trace(go.Bar(
                                name=display_name,
                                x=[symbol],
                                y=[value],
                                marker_color=color,
                                text=[f"{value:.3f}"],
                                textposition='auto',
                            ))
                    
                    fig.update_layout(
                        title=f'{symbol} Mean Social Sentiment Analysis',
                        xaxis_title='Stock',
                        yaxis_title='Sentiment Score',
                        barmode='group',
                        template='plotly_dark',
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, config=create_plotly_config(), width='stretch')
                else:
                    st.info("Insufficient numeric data for plotting social sentiment metrics.")
            else:
                # Fallback: plot directly from current_week_social
                # Include specific columns for social sentiment
                social_cols_to_include = [
                    'stocktwitsPosts', 'twitterPosts', 'stocktwitsComments', 'twitterComments',
                    'stocktwitsLikes', 'twitterLikes', 'stocktwitsImpressions', 'twitterImpressions',
                    'stocktwitsSentiment', 'twitterSentiment'
                ]
                # Get all numeric columns, prioritizing the social columns
                all_numeric_cols = current_week_social.select_dtypes(include=[np.number]).columns.tolist()
                # Filter to include social columns if they exist, plus other numeric columns
                numeric_cols = [col for col in social_cols_to_include if col in all_numeric_cols]
                # Add other numeric columns that aren't in the social list
                other_cols = [col for col in all_numeric_cols if col not in social_cols_to_include]
                numeric_cols.extend(other_cols)
                
                if len(numeric_cols) > 0:
                    fig = go.Figure()
                    # Different distinct colors for each bar
                    distinct_colors = ['#FF6B9D', '#C44569', '#F8B500', '#FFA07A', '#9B59B6', '#E74C3C', '#3498DB', '#1ABC9C', '#16A085', '#27AE60']
                    
                    for idx, col in enumerate(numeric_cols[:10]):  # Limit to 10 columns to include all social columns
                        if col in current_week_social.columns:
                            value = current_week_social[col].mean()
                            color = distinct_colors[idx % len(distinct_colors)]  # Cycle through colors
                            display_name = col.replace('Index', ' Index').replace('sentiment', 'Sentiment').title()
                            fig.add_trace(go.Bar(
                                name=display_name,
                                x=[symbol],
                                y=[value],
                                marker_color=color,
                                text=[f"{value:.3f}"],
                                textposition='auto',
                            ))
                    
                    fig.update_layout(
                        title=f'{symbol} Mean Social Sentiment Analysis',
                        xaxis_title='Stock',
                        yaxis_title='Sentiment Score',
                        barmode='group',
                        template='plotly_dark',
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig, config=create_plotly_config(), width='stretch')
                else:
                    st.info("Insufficient data for plotting social sentiment metrics.")
        else:
            st.info("No social sentiment data found for the current week. Showing all available data.")
            # Show all data if no current week data
            if 'symbol' in social_copy.columns:
                social_mean = social_copy.groupby('symbol').mean()
            else:
                numeric_data = social_copy.select_dtypes(include=[np.number])
                if not numeric_data.empty:
                    social_mean = numeric_data.mean()
                    social_mean = pd.DataFrame([social_mean], index=[symbol])
                else:
                    social_mean = pd.DataFrame()
            
            # Plot with different colors
            if isinstance(social_mean, pd.Series):
                social_mean = pd.DataFrame([social_mean], index=[symbol])
            
            if isinstance(social_mean, pd.DataFrame) and len(social_mean) > 0:
                # Include specific columns for social sentiment
                social_cols_to_include = [
                    'stocktwitsPosts', 'twitterPosts', 'stocktwitsComments', 'twitterComments',
                    'stocktwitsLikes', 'twitterLikes', 'stocktwitsImpressions', 'twitterImpressions',
                    'stocktwitsSentiment', 'twitterSentiment'
                ]
                # Get all numeric columns, prioritizing the social columns
                all_numeric_cols = social_mean.select_dtypes(include=[np.number]).columns.tolist()
                # Filter to include social columns if they exist, plus other numeric columns
                numeric_cols = [col for col in social_cols_to_include if col in all_numeric_cols]
                # Add other numeric columns that aren't in the social list
                other_cols = [col for col in all_numeric_cols if col not in social_cols_to_include]
                numeric_cols.extend(other_cols)
                social_mean_filtered = social_mean
            else:
                # Include specific columns for social sentiment
                social_cols_to_include = [
                    'stocktwitsPosts', 'twitterPosts', 'stocktwitsComments', 'twitterComments',
                    'stocktwitsLikes', 'twitterLikes', 'stocktwitsImpressions', 'twitterImpressions',
                    'stocktwitsSentiment', 'twitterSentiment'
                ]
                # Get all numeric columns, prioritizing the social columns
                all_numeric_cols = social_copy.select_dtypes(include=[np.number]).columns.tolist()
                # Filter to include social columns if they exist, plus other numeric columns
                numeric_cols = [col for col in social_cols_to_include if col in all_numeric_cols]
                # Add other numeric columns that aren't in the social list
                other_cols = [col for col in all_numeric_cols if col not in social_cols_to_include]
                numeric_cols.extend(other_cols)
                social_copy_filtered = social_copy
            
            if len(numeric_cols) > 0:
                fig = go.Figure()
                # Different distinct colors for each bar
                distinct_colors = ['#FF6B9D', '#C44569', '#F8B500', '#FFA07A', '#9B59B6', '#E74C3C', '#3498DB', '#1ABC9C', '#16A085', '#27AE60']
                
                for idx, col in enumerate(numeric_cols[:8]):  # Limit to 8 columns, each with different color
                    if isinstance(social_mean, pd.DataFrame) and len(social_mean) > 0 and col in social_mean_filtered.columns:
                        value = social_mean_filtered[col].iloc[0]
                    elif col in social_copy_filtered.columns:
                        value = social_copy_filtered[col].mean()
                    else:
                        continue
                    
                    color = distinct_colors[idx % len(distinct_colors)]  # Cycle through colors
                    display_name = col.replace('Index', ' Index').replace('sentiment', 'Sentiment').title()
                    fig.add_trace(go.Bar(
                        name=display_name,
                        x=[symbol],
                        y=[value],
                        marker_color=color,
                        text=[f"{value:.3f}"],
                        textposition='auto',
                    ))
                
                fig.update_layout(
                    title=f'{symbol} Mean Social Sentiment Analysis',
                    xaxis_title='Stock',
                    yaxis_title='Sentiment Score',
                    barmode='group',
                    template='plotly_dark',
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, config=create_plotly_config(), width='stretch')
    else:
        st.info("Social sentiment data not available for this stock.")
    
    # Analyst Estimate Analysis Section
    st.header("📊 Analyst Estimate Analysis")
    
    # Ensure analyst_grades is defined
    if 'analyst_grades' not in locals():
        analyst_grades = pd.DataFrame()
    
    if not analyst_grades.empty:
        # Filter for current year first, then previous year if current year not available
        analyst_copy = analyst_grades.copy()
        current_year = pd.Timestamp.now().year
        previous_year = current_year - 1
        
        if isinstance(analyst_copy.index, pd.DatetimeIndex):
            # Try current year first
            current_year_start = pd.Timestamp(current_year, 1, 1)
            current_year_end = pd.Timestamp(current_year, 12, 31, 23, 59, 59)
            recent_analyst = analyst_copy[
                (analyst_copy.index >= current_year_start) & 
                (analyst_copy.index <= current_year_end)
            ].copy()
            
            # If current year is empty, try previous year
            if recent_analyst.empty:
                previous_year_start = pd.Timestamp(previous_year, 1, 1)
                previous_year_end = pd.Timestamp(previous_year, 12, 31, 23, 59, 59)
                recent_analyst = analyst_copy[
                    (analyst_copy.index >= previous_year_start) & 
                    (analyst_copy.index <= previous_year_end)
                ].copy()
        elif 'date' in analyst_copy.columns:
            analyst_copy['date'] = pd.to_datetime(analyst_copy['date'], errors='coerce')
            # Remove rows with invalid dates
            analyst_copy = analyst_copy.dropna(subset=['date'])
            if not analyst_copy.empty:
                # Try current year first
                current_year_start = pd.Timestamp(current_year, 1, 1)
                current_year_end = pd.Timestamp(current_year, 12, 31, 23, 59, 59)
                recent_analyst = analyst_copy[
                    (analyst_copy['date'] >= current_year_start) & 
                    (analyst_copy['date'] <= current_year_end)
                ].copy()
                
                # If current year is empty, try previous year
                if recent_analyst.empty:
                    previous_year_start = pd.Timestamp(previous_year, 1, 1)
                    previous_year_end = pd.Timestamp(previous_year, 12, 31, 23, 59, 59)
                    recent_analyst = analyst_copy[
                        (analyst_copy['date'] >= previous_year_start) & 
                        (analyst_copy['date'] <= previous_year_end)
                    ].copy()
            else:
                recent_analyst = pd.DataFrame()
        else:
            # No date column - use all data
            recent_analyst = analyst_copy.copy()
        
        # If still empty after trying current and previous year, show all available data
        if recent_analyst.empty and not analyst_copy.empty:
            recent_analyst = analyst_copy.copy()
        
        if not recent_analyst.empty:
            # Display as table (like Company Sentiment.ipynb)
            st.subheader("📈 Analyst Sentiment Data")
            analyst_display = recent_analyst.copy()
            if isinstance(analyst_display.index, pd.DatetimeIndex):
                analyst_display = analyst_display.reset_index()
                if 'index' in analyst_display.columns:
                    analyst_display = analyst_display.rename(columns={'index': 'date'})
            
            # Remove columns with impressions if they exist
            columns_to_remove = [col for col in analyst_display.columns if 'impression' in col.lower() or 'twitter' in col.lower()]
            if columns_to_remove:
                analyst_display = analyst_display.drop(columns=columns_to_remove)
            
            st.dataframe(analyst_display, width='stretch', hide_index=True)
            
            # Count by newGrade and display as pie chart (like Company Sentiment.ipynb)
            if 'newGrade' in recent_analyst.columns:
                st.subheader("🥧 Analyst Sentiment Count by Grade")
                
                # Count by newGrade
                if isinstance(recent_analyst.index, pd.DatetimeIndex):
                    analyst_count = recent_analyst.groupby('newGrade').size()
                else:
                    analyst_count = recent_analyst.groupby('newGrade').size()
                
                if len(analyst_count) > 0:
                    # Create pie chart with labels for majority portions only
                    total = analyst_count.sum()
                    # Sort by value descending to get top portions
                    sorted_counts = analyst_count.sort_values(ascending=False)
                    # Get top 3 labels or labels with > 10%
                    top_labels = set()
                    for idx, (label, value) in enumerate(sorted_counts.items()):
                        percentage = (value / total) * 100
                        if percentage > 10 or idx < 3:
                            top_labels.add(label)
                    
                    # Create textinfo list matching original order
                    textinfo_list = []
                    for label in analyst_count.index:
                        if label in top_labels:
                            percentage = (analyst_count[label] / total) * 100
                            textinfo_list.append(f"{label}<br>{percentage:.1f}%")
                        else:
                            textinfo_list.append("")
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=analyst_count.index.tolist(),
                        values=analyst_count.values.tolist(),
                        hole=0.3,  # Creates a donut chart
                        text=textinfo_list,
                        textinfo='text',
                        textposition='outside',
                        marker=dict(
                            colors=['#FF6B9D', '#C44569', '#F8B500', '#FFA07A', '#9B59B6', '#E74C3C', '#3498DB', '#1ABC9C'],
                            line=dict(color='#1E2229', width=2)
                        )
                    )])
                    
                    fig.update_layout(
                        title=f'{symbol} Analyst Sentiment Distribution',
                        template='plotly_dark',
                        height=500,
                        showlegend=True
                    )
                    st.plotly_chart(fig, config=create_plotly_config(), width='stretch')
                else:
                    st.info("No grade data available for pie chart.")
            else:
                st.info("'newGrade' column not found in analyst data.")
        else:
            # This shouldn't happen now due to fallback, but just in case
            st.info("No analyst data available for the selected period.")
    else:
        st.info("Analyst grades data not available for this stock. The data may not be available from the API for this symbol.")
    
    # Analyst Ratings subsection (FMP Rating) - subheader removed
    if 'rating' in locals() and rating is not None:
        # Rating is typically a dict from get_rating
        # Convert to DataFrame for display (like Company Sentiment.ipynb)
        if isinstance(rating, dict):
            # If rating has 'date', use it as index; otherwise create single-row DataFrame
            rating_df = pd.DataFrame([rating])
            if 'date' in rating_df.columns:
                rating_df['date'] = pd.to_datetime(rating_df['date'], errors='coerce')
                rating_df = rating_df.set_index('date')
            elif 'date' in rating:
                # If date is in the dict but not in DataFrame columns yet
                rating_df = pd.DataFrame([rating])
                rating_df['date'] = pd.to_datetime(rating_df.get('date', pd.Timestamp.now()), errors='coerce')
                rating_df = rating_df.set_index('date')
        elif isinstance(rating, pd.DataFrame):
            rating_df = rating.copy()
            if 'date' in rating_df.columns and not isinstance(rating_df.index, pd.DatetimeIndex):
                rating_df['date'] = pd.to_datetime(rating_df['date'], errors='coerce')
                rating_df = rating_df.set_index('date')
        else:
            rating_df = pd.DataFrame()
        
        if not rating_df.empty:
            # Display table (horizontal, without date, no transpose)
            st.markdown("#### 📋 Rating Details")
            rating_display = rating_df.copy()
            # Remove date column/index if present
            if 'date' in rating_display.columns:
                rating_display = rating_display.drop(columns=['date'])
            if isinstance(rating_display.index, pd.DatetimeIndex) or (hasattr(rating_display.index, 'name') and rating_display.index.name == 'date'):
                rating_display = rating_display.reset_index(drop=True)
            # Display as horizontal table (no transpose)
            st.dataframe(rating_display, width='stretch', hide_index=True)
            
            # Bar graph for scores
            st.markdown("#### 📊 Rating Scores")
            # Extract score columns (ratingScore, ratingDetailsDCFScore, etc.)
            score_cols = [col for col in rating_df.columns if 'Score' in col]
            # Prioritize ratingScore first
            if 'ratingScore' in score_cols:
                score_cols.remove('ratingScore')
                score_cols.insert(0, 'ratingScore')
            
            if score_cols:
                # Get the latest rating (most recent date if indexed, otherwise first row)
                if isinstance(rating_df.index, pd.DatetimeIndex):
                    latest_rating = rating_df.iloc[-1]  # Most recent
                else:
                    latest_rating = rating_df.iloc[0]  # First row
                
                # Prepare data for bar chart
                scores_data = []
                for col in score_cols:
                    if col in latest_rating.index and pd.notna(latest_rating[col]):
                        try:
                            score_value = float(latest_rating[col])
                            # Format column name for display
                            display_name = col.replace('ratingDetails', '').replace('Score', '').replace('rating', 'Overall Rating')
                            # Convert camelCase to Title Case
                            display_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', display_name).title()
                            scores_data.append({
                                'Metric': display_name,
                                'Score': score_value
                            })
                        except (ValueError, TypeError):
                            continue
                
                if scores_data:
                    scores_df = pd.DataFrame(scores_data)
                    
                    # Create bar chart (without text labels)
                    fig_bar = go.Figure(data=[go.Bar(
                        x=scores_df['Metric'],
                        y=scores_df['Score'],
                        marker=dict(
                            color=scores_df['Score'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Score")
                        ),
                        hovertemplate='<b>%{x}</b><br>Score: %{y:.2f}/5<extra></extra>'
                    )])
                    
                    fig_bar.update_layout(
                        title=f'{symbol} FMP Rating Scores',
                        xaxis_title='Rating Metric',
                        yaxis_title='Score (out of 5)',
                        template='plotly_dark',
                        height=400,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig_bar, config=create_plotly_config(), width='stretch')
                else:
                    st.info("No score data available for bar chart.")
            else:
                st.info("No score columns found in rating data.")
        else:
            st.info("Rating data is empty.")
    else:
        st.info("FMP Rating data not available for this stock.")
    
    # Analyst Estimates section (for next year onwards, don't mention in text)
    if 'analyst_estimates' in locals() and not analyst_estimates.empty:
        st.subheader("📊 Analyst Estimates")
        
        # Filter for next year onwards (exclude current year)
        estimates_copy = analyst_estimates.copy()
        if isinstance(estimates_copy.index, pd.DatetimeIndex):
            today = pd.Timestamp.now()
            next_year_start = pd.Timestamp(today.year + 1, 1, 1)  # Start of next year
            five_years_later = next_year_start + pd.Timedelta(days=365*5)
            future_estimates = estimates_copy[(estimates_copy.index >= next_year_start) & (estimates_copy.index <= five_years_later)].copy()
        elif 'date' in estimates_copy.columns:
            estimates_copy['date'] = pd.to_datetime(estimates_copy['date'], errors='coerce')
            today = pd.Timestamp.now()
            next_year_start = pd.Timestamp(today.year + 1, 1, 1)  # Start of next year
            five_years_later = next_year_start + pd.Timedelta(days=365*5)
            future_estimates = estimates_copy[(estimates_copy['date'] >= next_year_start) & (estimates_copy['date'] <= five_years_later)].copy()
        else:
            # If no date index/column, try to filter by year if there's a year column
            if 'fiscalYear' in estimates_copy.columns or 'calendarYear' in estimates_copy.columns:
                year_col = 'fiscalYear' if 'fiscalYear' in estimates_copy.columns else 'calendarYear'
                current_year = pd.Timestamp.now().year
                future_estimates = estimates_copy[estimates_copy[year_col] > current_year].copy()
            else:
                future_estimates = estimates_copy.copy()
        
        if not future_estimates.empty:
            # Display table
            estimates_display = future_estimates.copy()
            if isinstance(estimates_display.index, pd.DatetimeIndex):
                estimates_display = estimates_display.reset_index()
                if 'index' in estimates_display.columns:
                    estimates_display = estimates_display.rename(columns={'index': 'date'})
            st.dataframe(estimates_display, width='stretch', hide_index=True)
            
            # Plot for Analyst Estimates
            st.markdown("### 📈 Analyst Estimates Trends")
            
            # Use the original future_estimates (before display modifications) for plotting
            plot_source = future_estimates.copy()
            
            # Debug: Check what columns we actually have
            all_plot_cols = list(plot_source.columns)
            
            # Get all estimated columns - check both naming conventions
            estimated_cols = [
                # With "estimated" prefix
                'estimatedRevenueLow', 'estimatedRevenueHigh', 'estimatedRevenueAvg',
                'estimatedEbitdaLow', 'estimatedEbitdaHigh', 'estimatedEbitdaAvg',
                'estimatedEbitLow', 'estimatedEbitHigh', 'estimatedEbitAvg',
                'estimatedNetIncomeLow', 'estimatedNetIncomeHigh', 'estimatedNetIncomeAvg',
                'estimatedSgaExpenseLow', 'estimatedSgaExpenseHigh', 'estimatedSgaExpenseAvg',
                'estimatedEpsAvg', 'estimatedEpsHigh', 'estimatedEpsLow',
                'numberAnalystEstimatedRevenue', 'numberAnalystsEstimatedEps',
                # Without "estimated" prefix (actual API format)
                'revenueLow', 'revenueHigh', 'revenueAvg',
                'ebitdaLow', 'ebitdaHigh', 'ebitdaAvg',
                'ebitLow', 'ebitHigh', 'ebitAvg',
                'netIncomeLow', 'netIncomeHigh', 'netIncomeAvg',
                'sgaExpenseLow', 'sgaExpenseHigh', 'sgaExpenseAvg',
                'epsAvg', 'epsHigh', 'epsLow',
                'numAnalystsRevenue', 'numAnalystsEps'
            ]
            
            # Filter columns that exist in the data - check both exact match and case-insensitive
            available_cols = []
            for c in estimated_cols:
                if c in plot_source.columns:
                    # Also check if it's numeric
                    try:
                        if pd.api.types.is_numeric_dtype(plot_source[c]):
                            available_cols.append(c)
                    except:
                        # Try to convert to numeric
                        try:
                            pd.to_numeric(plot_source[c], errors='coerce')
                            available_cols.append(c)
                        except:
                            pass
            
            # If no exact matches, try to find any columns that contain relevant keywords
            if not available_cols:
                for col in all_plot_cols:
                    if col == 'symbol' or col == 'date' or col == 'index':
                        continue
                    col_lower = str(col).lower()
                    # Check for revenue, ebitda, ebit, netincome, sga, eps, or analyst-related columns
                    if any(keyword in col_lower for keyword in ['revenue', 'ebitda', 'ebit', 'netincome', 'sga', 'eps', 'analyst', 'estimated']):
                        try:
                            # Check if numeric
                            test_vals = pd.to_numeric(plot_source[col], errors='coerce')
                            if not test_vals.isna().all():
                                available_cols.append(col)
                        except:
                            pass
            
            # If still no columns found, try using estimates_display (which has the data we showed in the table)
            if not available_cols and not estimates_display.empty:
                # Try again with estimates_display
                for col in estimates_display.columns:
                    if col == 'symbol' or col == 'date' or col == 'index':
                        continue
                    col_lower = str(col).lower()
                    # Check for revenue, ebitda, ebit, netincome, sga, eps, or analyst-related columns
                    if any(keyword in col_lower for keyword in ['revenue', 'ebitda', 'ebit', 'netincome', 'sga', 'eps', 'analyst', 'estimated']):
                        try:
                            test_vals = pd.to_numeric(estimates_display[col], errors='coerce')
                            if not test_vals.isna().all():
                                available_cols.append(col)
                        except:
                            pass
                # If we found columns in estimates_display, use that dataframe for plotting
                if available_cols:
                    plot_source = estimates_display.copy()
                    if 'date' in plot_source.columns:
                        plot_source['date'] = pd.to_datetime(plot_source['date'], errors='coerce')
                        plot_source = plot_source.set_index('date').sort_index()
                    elif not isinstance(plot_source.index, pd.DatetimeIndex):
                        # Try to find date column
                        for col in plot_source.columns:
                            if 'date' in col.lower():
                                plot_source[col] = pd.to_datetime(plot_source[col], errors='coerce')
                                plot_source = plot_source.set_index(col).sort_index()
                                break
            
            if available_cols:
                # Prepare data for plotting - use plot_source (original future_estimates)
                plot_data = plot_source.copy()
                
                # Ensure we have a datetime index
                if not isinstance(plot_data.index, pd.DatetimeIndex):
                    if 'date' in plot_data.columns:
                        plot_data['date'] = pd.to_datetime(plot_data['date'], errors='coerce')
                        plot_data = plot_data.set_index('date').sort_index()
                    elif 'index' in plot_data.columns:
                        plot_data['index'] = pd.to_datetime(plot_data['index'], errors='coerce')
                        plot_data = plot_data.set_index('index').sort_index()
                    else:
                        # Try to use the first column that looks like a date
                        for col in plot_data.columns:
                            if 'date' in col.lower() or 'time' in col.lower():
                                plot_data[col] = pd.to_datetime(plot_data[col], errors='coerce')
                                plot_data = plot_data.set_index(col).sort_index()
                                break
                
                # Sort by index (date) ascending for proper time series
                if isinstance(plot_data.index, pd.DatetimeIndex):
                    plot_data = plot_data.sort_index(ascending=True)
                
                # Create interactive multiselect
                col_display_names = {}
                for col_key in available_cols:
                    display_name = re.sub(r'([A-Z])', r' \1', col_key).strip()
                    display_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', display_name)
                    display_name = display_name.replace('_', ' ').title()
                    col_display_names[col_key] = display_name
                
                col_options = {col_display_names[c]: c for c in available_cols}
                selected_col_names = st.multiselect(
                    "Select Analyst Estimate Metrics to Display",
                    options=list(col_options.keys()),
                    default=list(col_options.keys())[:8] if len(col_options) > 8 else list(col_options.keys()),
                    key='analyst_estimates_plot_selector'
                )
                
                if selected_col_names:
                    # Create a single combined plot with selected metrics
                    fig = go.Figure()
                    
                    # Color palette for multiple lines
                    colors = ['#00D4AA', '#4A90E2', '#7B68EE', '#FF6B9D', '#C44569', '#F8B500', 
                             '#FFA07A', '#9B59B6', '#E74C3C', '#3498DB', '#1ABC9C', '#16A085',
                             '#27AE60', '#F39C12', '#E67E22', '#95A5A6', '#34495E', '#9B59B6',
                             '#E91E63', '#00BCD4', '#4CAF50', '#FF9800', '#795548', '#607D8B']
                    
                    for idx, col_display_name in enumerate(selected_col_names):
                        col_key = col_options[col_display_name]
                        y_values = plot_data[col_key].copy()
                        
                        color = colors[idx % len(colors)]
                        
                        fig.add_trace(go.Scatter(
                            x=plot_data.index,
                            y=y_values,
                            mode='lines+markers',
                            name=col_display_name,
                            line=dict(color=color, width=2),
                            marker=dict(size=4),
                            hovertemplate=f'<b>{col_display_name}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title=f'{symbol} Analyst Estimates Overview',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        template='plotly_dark',
                        height=600,
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    st.plotly_chart(fig, config=create_plotly_config(), width='stretch')
                else:
                    st.info("Please select at least one analyst estimate metric to display.")
            else:
                st.info("No analyst estimate metrics available for plotting.")
        else:
            st.info("No future analyst estimates available.")
    else:
        st.info("Analyst estimates data not available for this stock.")
    
    # Institutional Investors section
    st.header("🏛️ Institutional Investors")
    
    if 'institutional_holders' in locals() and not institutional_holders.empty:
        # Overview metrics (from Company Overview.ipynb cell 1-10)
        # Calculate the sum of the shares held by institutions and net change
        sum_institutional_shares = institutional_holders['shares'].sum()
        net_change = institutional_holders['change'].sum() if 'change' in institutional_holders.columns else None
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Total Shares Held by Institutions",
                value=millify(sum_institutional_shares, precision=2)
            )
        with col2:
            if net_change is not None:
                st.metric(
                    label="Net Change in Shares",
                    value=millify(net_change, precision=2),
                    delta=f"{net_change:,.0f}" if net_change != 0 else "0"
                )
            else:
                st.metric(
                    label="Net Change in Shares",
                    value="N/A"
                )
        
        # Top 20 Institutional Holders Table (from Company Overview.ipynb cell 1-17)
        # Sort the DataFrame by number of shares in descending order
        institutional_holders_sorted = institutional_holders.sort_values(by='shares', ascending=False)
        st.markdown("### 🏢 Top 20 Institutional Holders")
        top_holders = institutional_holders_sorted.head(20).copy()
        
        # Format the dataframe for display
        if 'shares' in top_holders.columns:
            top_holders['shares'] = top_holders['shares'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        if 'change' in top_holders.columns:
            top_holders['change'] = top_holders['change'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
        if 'value' in top_holders.columns:
            top_holders['value'] = top_holders['value'].apply(lambda x: millify(x, precision=2) if pd.notna(x) else "N/A")
        
        st.dataframe(top_holders, width='stretch', hide_index=True)
        
        # Charts and graphs
        if 'shares' in institutional_holders_sorted.columns:
            # Bar chart of top 20 holders by shares
            st.markdown("### 📊 Top 20 Institutional Holders by Shares")
            top_20_for_chart = institutional_holders_sorted.head(20).copy()
            
            # Find the name column (could be 'name', 'holderName', 'institutionName', etc.)
            name_col = None
            for col in ['name', 'holderName', 'institutionName', 'holder']:
                if col in top_20_for_chart.columns:
                    name_col = col
                    break
            
            if name_col and len(top_20_for_chart) > 0:
                # Ensure shares are numeric
                top_20_for_chart['shares'] = pd.to_numeric(top_20_for_chart['shares'], errors='coerce')
                top_20_for_chart = top_20_for_chart.dropna(subset=['shares'])
                
                if len(top_20_for_chart) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=top_20_for_chart['shares'],
                        y=top_20_for_chart[name_col],
                        orientation='h',
                        marker=dict(color='#4A90E2'),
                        hovertemplate='<b>%{y}</b><br>Shares: %{x:,.0f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f'{symbol} Top 20 Institutional Holders',
                        xaxis_title='Shares Held',
                        yaxis_title='Institution Name',
                        template='plotly_dark',
                        height=600,
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig, config=create_plotly_config(), width='stretch')
                    
                    # Pie chart showing distribution of top 20 holders
                    st.markdown("### 📊 Share Distribution")
                    
                    # Calculate percentages and show labels only for major portions (>10% or top 3)
                    total_shares = top_20_for_chart['shares'].sum()
                    top_20_for_chart['percentage'] = (top_20_for_chart['shares'] / total_shares * 100)
                    
                    # Create custom text labels - show only for portions >10% or top 3
                    custom_text = []
                    for idx, row in top_20_for_chart.iterrows():
                        if row['percentage'] > 10 or idx < 3:
                            # Show name and percentage for major portions
                            custom_text.append(f"{row[name_col]}<br>{row['percentage']:.1f}%")
                        else:
                            # Show only percentage for minor portions
                            custom_text.append(f"{row['percentage']:.1f}%")
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=top_20_for_chart[name_col],
                        values=top_20_for_chart['shares'],
                        hole=0.3,
                        text=custom_text,
                        textinfo='text',
                        textposition='outside',
                        hovertemplate='<b>%{label}</b><br>Shares: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
                    )])
                    
                    fig_pie.update_layout(
                        title=f'{symbol} Institutional Holdings Distribution',
                        template='plotly_dark',
                        height=500
                    )
                    st.plotly_chart(fig_pie, config=create_plotly_config(), width='stretch')
            else:
                st.warning("Institution name column not found in the data.")
        
    else:
        st.info("Institutional holders data not available for this stock.")
    
    # Earnings Surprises section (for past 3 years, don't mention in text)
    if 'earnings_surprises' in locals() and not earnings_surprises.empty:
        st.subheader("🎯 Earnings Surprises")
        
        # Filter for past 3 years
        surprises_copy = earnings_surprises.copy()
        if isinstance(surprises_copy.index, pd.DatetimeIndex):
            three_years_ago = pd.Timestamp.now() - pd.Timedelta(days=365*3)
            past_three_years_surprises = surprises_copy[surprises_copy.index >= three_years_ago].copy()
        elif 'date' in surprises_copy.columns:
            surprises_copy['date'] = pd.to_datetime(surprises_copy['date'], errors='coerce')
            three_years_ago = pd.Timestamp.now() - pd.Timedelta(days=365*3)
            past_three_years_surprises = surprises_copy[surprises_copy['date'] >= three_years_ago].copy()
        else:
            past_three_years_surprises = surprises_copy.copy()
        
        if not past_three_years_surprises.empty:
            surprises_display = past_three_years_surprises.copy()
            if isinstance(surprises_display.index, pd.DatetimeIndex):
                surprises_display = surprises_display.reset_index()
                if 'index' in surprises_display.columns:
                    surprises_display = surprises_display.rename(columns={'index': 'date'})
            
            # Add boolean column "Actual Greater Than Estimated"
            if 'actualEarningResult' in surprises_display.columns and 'estimatedEarning' in surprises_display.columns:
                surprises_display['Actual Greater Than Estimated'] = (
                    surprises_display['actualEarningResult'] > surprises_display['estimatedEarning']
                )
            
            st.dataframe(surprises_display, width='stretch', hide_index=True)
        else:
            st.info("No earnings surprises data available for the past 3 years.")
    else:
        st.info("Earnings surprises data not available for this stock.")
    
    # Financial ratios table
    st.header("Financial Ratios")
    if not ratios_data.empty:
        # Filter for last 5 years
        ratios_for_plot = ratios_data.copy()
        if isinstance(ratios_for_plot.index, pd.DatetimeIndex):
            five_years_ago = pd.Timestamp.now() - pd.Timedelta(days=365*5)
            ratios_for_plot = ratios_for_plot[ratios_for_plot.index >= five_years_ago].copy()
        elif 'date' in ratios_for_plot.columns:
            ratios_for_plot['date'] = pd.to_datetime(ratios_for_plot['date'], errors='coerce')
            five_years_ago = pd.Timestamp.now() - pd.Timedelta(days=365*5)
            ratios_for_plot = ratios_for_plot[ratios_for_plot['date'] >= five_years_ago].copy()
            if 'date' in ratios_for_plot.columns:
                ratios_for_plot = ratios_for_plot.set_index('date').sort_index()
        
        # Display table
        ratios_table = ratios_data.rename(columns={
            'Days of Sales Outstanding': 'Days of Sales Outstanding (days)',
            'Days of Inventory Outstanding': 'Days of Inventory Outstanding (days)',
            'Operating Cycle': 'Operating Cycle (days)',
            'Days of Payables Outstanding': 'Days of Payables Outstanding (days)',
            'Cash Conversion Cycle': 'Cash Conversion Cycle (days)',
            'Gross Profit Margin': 'Gross Profit Margin (%)',
            'Operating Profit Margin': 'Operating Profit Margin (%)',
            'Pretax Profit Margin': 'Pretax Profit Margin (%)',
            'Net Profit Margin': 'Net Profit Margin (%)',
            'Effective Tax Rate': 'Effective Tax Rate (%)',
            'Return on Assets': 'Return on Assets (%)',
            'Return on Equity': 'Return on Equity (%)',
            'Return on Capital Employed': 'Return on Capital Employed (%)',
            'EBIT per Revenue': 'EBIT per Revenue (%)',
            'Debt Ratio': 'Debt Ratio (%)',
            'Long-term Debt to Capitalization': 'Long-term Debt to Capitalization (%)',
            'Total Debt to Capitalization': 'Total Debt to Capitalization (%)',
            'Payout Ratio': 'Payout Ratio (%)',
            'Operating Cash Flow Sales Ratio': 'Operating Cash Flow Sales Ratio (%)',
            'Dividend Yield': 'Dividend Yield (%)',
        })
        
        for col in ratios_table.columns:
            if "%" in col:
                ratios_table[col] = ratios_table[col] * 100
        
        ratios_table = round(ratios_table.T, 2)
        ratios_table = ratios_table.sort_index(axis=1, ascending=True)
        # Show the index (ratio names) - don't hide it
        st.dataframe(ratios_table, width='stretch', height=400, hide_index=False)
        
        # Create interactive plot for financial ratios
        if not ratios_for_plot.empty:
            st.subheader("📊 Financial Ratios Trends")
            
            # All ratios to plot (excluding reportedCurrency)
            all_ratios = [
                'grossProfitMargin', 'ebitMargin', 'ebitdaMargin', 
                'operatingProfitMargin', 'pretaxProfitMargin', 'continuousOperationsProfitMargin',
                'netProfitMargin', 'bottomLineProfitMargin', 'receivablesTurnover', 
                'payablesTurnover', 'inventoryTurnover', 'fixedAssetTurnover', 'assetTurnover',
                'currentRatio', 'quickRatio', 'solvencyRatio', 'cashRatio', 
                'priceToEarningsRatio', 'priceToEarningsGrowthRatio', 'forwardPriceToEarningsGrowthRatio',
                'priceToBookRatio', 'priceToSalesRatio', 'priceToFreeCashFlowRatio',
                'priceToOperatingCashFlowRatio', 'debtToAssetsRatio', 'debtToEquityRatio',
                'debtToCapitalRatio', 'longTermDebtToCapitalRatio', 'financialLeverageRatio',
                'workingCapitalTurnoverRatio', 'operatingCashFlowRatio', 'operatingCashFlowSalesRatio',
                'freeCashFlowOperatingCashFlowRatio', 'debtServiceCoverageRatio', 'interestCoverageRatio',
                'shortTermOperatingCashFlowCoverageRatio', 'operatingCashFlowCoverageRatio',
                'capitalExpenditureCoverageRatio', 'dividendPaidAndCapexCoverageRatio',
                'dividendPayoutRatio', 'dividendYield', 'dividendYieldPercentage',
                'revenuePerShare', 'netIncomePerShare', 'interestDebtPerShare', 'cashPerShare',
                'bookValuePerShare', 'tangibleBookValuePerShare', 'shareholdersEquityPerShare',
                'operatingCashFlowPerShare', 'capexPerShare', 'freeCashFlowPerShare',
                'netIncomePerEBT', 'ebtPerEbit', 'priceToFairValue', 'debtToMarketCap',
                'effectiveTaxRate', 'enterpriseValueMultiple', 'dividendPerShare'
            ]
            
            # Filter ratios that exist in the data
            available_ratios = [r for r in all_ratios if r in ratios_for_plot.columns]
            
            if available_ratios:
                # Format ratio names for display
                ratio_display_names = {}
                for ratio_key in available_ratios:
                    display_name = re.sub(r'([A-Z])', r' \1', ratio_key).strip()
                    display_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', display_name)
                    display_name = display_name.replace('_', ' ').title()
                    ratio_display_names[ratio_key] = display_name
                
                # Create interactive multiselect
                ratio_options = {ratio_display_names[r]: r for r in available_ratios}
                selected_ratio_names = st.multiselect(
                    "Select Financial Ratios to Display",
                    options=list(ratio_options.keys()),
                    default=list(ratio_options.keys())[:10] if len(ratio_options) > 10 else list(ratio_options.keys()),
                    key='financial_ratios_selector'
                )
                
                if selected_ratio_names:
                    # Create a single combined plot with selected ratios (like Balance Sheet Overview)
                    fig = go.Figure()
                    
                    # Color palette for multiple lines
                    colors = ['#00D4AA', '#4A90E2', '#7B68EE', '#FF6B9D', '#C44569', '#F8B500', 
                             '#FFA07A', '#9B59B6', '#E74C3C', '#3498DB', '#1ABC9C', '#16A085',
                             '#27AE60', '#F39C12', '#E67E22', '#95A5A6', '#34495E', '#9B59B6',
                             '#E91E63', '#00BCD4', '#4CAF50', '#FF9800', '#795548', '#607D8B']
                    
                    # Determine which ratios should be percentages
                    non_percentage_ratios = {
                        'priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio', 
                        'priceToFreeCashFlowRatio', 'priceToOperatingCashFlowRatio',
                        'debtToEquityRatio', 'debtToAssetsRatio', 'debtToCapitalRatio',
                        'longTermDebtToCapitalRatio', 'financialLeverageRatio',
                        'workingCapitalTurnoverRatio', 'operatingCashFlowRatio',
                        'operatingCashFlowSalesRatio', 'freeCashFlowOperatingCashFlowRatio',
                        'debtServiceCoverageRatio', 'interestCoverageRatio',
                        'shortTermOperatingCashFlowCoverageRatio', 'operatingCashFlowCoverageRatio',
                        'capitalExpenditureCoverageRatio', 'dividendPaidAndCapexCoverageRatio',
                        'dividendPayoutRatio', 'priceToEarningsGrowthRatio',
                        'forwardPriceToEarningsGrowthRatio', 'currentRatio', 'quickRatio',
                        'cashRatio', 'solvencyRatio', 'receivablesTurnover', 'payablesTurnover',
                        'inventoryTurnover', 'fixedAssetTurnover', 'assetTurnover',
                        'revenuePerShare', 'netIncomePerShare', 'interestDebtPerShare',
                        'cashPerShare', 'bookValuePerShare', 'tangibleBookValuePerShare',
                        'shareholdersEquityPerShare', 'operatingCashFlowPerShare',
                        'capexPerShare', 'freeCashFlowPerShare', 'dividendPerShare',
                        'priceToFairValue', 'enterpriseValueMultiple', 'debtToMarketCap',
                        'netIncomePerEBT', 'ebtPerEbit'
                    }
                    
                    for idx, ratio_display_name in enumerate(selected_ratio_names):
                        ratio_key = ratio_options[ratio_display_name]
                        y_values = ratios_for_plot[ratio_key].copy()
                        
                        # Convert to percentage for margins, yields, rates (but not multipliers)
                        is_percentage = (ratio_key.endswith('Margin') or 
                                       ratio_key.endswith('Yield') or 
                                       ratio_key.endswith('Rate') or
                                       ratio_key == 'dividendYieldPercentage' or
                                       ratio_key == 'effectiveTaxRate') and ratio_key not in non_percentage_ratios
                        
                        if is_percentage:
                            y_values = y_values * 100
                        
                        color = colors[idx % len(colors)]
                        
                        fig.add_trace(go.Scatter(
                            x=ratios_for_plot.index,
                            y=y_values,
                            mode='lines+markers',
                            name=ratio_display_name,
                            line=dict(color=color, width=2),
                            marker=dict(size=4),
                            hovertemplate=f'<b>{ratio_display_name}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title=f'{symbol} Financial Ratios Overview',
                        xaxis_title='Date',
                        yaxis_title='Value',
                        template='plotly_dark',
                        height=600,
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    st.plotly_chart(fig, config=create_plotly_config(), width='stretch')
                else:
                    st.info("Please select at least one financial ratio to display.")
            else:
                st.info("No ratio data available for plotting.")
    else:
        st.info("Financial ratios data not available for this stock.")

finally:
    client.close()

