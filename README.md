# Equities Analytics Dashboard

A comprehensive Streamlit-based dashboard for equity analysis with automated index data ingestion, AI-powered agentic team, and sector-first navigation.

## Features

- **AI-Powered Agentic Team**: Specialized AI agents with OpenAI GPT-4o integration:
  - Discover indexes automatically
  - Fetch and validate constituents (with cross-validation using FMP and YFinance)
  - Segregate by industry and sector (with AI insights)
  - Comprehensive data validation and quality scoring
  - Gather market intelligence using Exa search (optional)
  - Enrich with comprehensive FMP data
  - Build optimized cache structures with AI-generated insights

- **MCP Tools Integration**: Uses Model Context Protocol tools:
  - **FMP Tools**: Financial data fetching and validation
    - `validate_fmp_profile(symbol)` - Validate single ticker
    - `validate_fmp_batch(tickers)` - Batch validation
  - **YFinance Tools**: Cross-validation with Yahoo Finance
    - `validate_yfinance_ticker(symbol)` - YFinance validation
    - `cross_validate_fmp_yfinance(symbol)` - Cross-source validation
    - `validate_ticker_data_quality(symbol)` - Quality scoring (0-100)
  - **Exa Search**: Market intelligence and news gathering

- **Data Validation**: 
  - Cross-validates FMP responses with YFinance
  - Quality scoring (EXCELLENT/GOOD/FAIR/POOR)
  - Discrepancy detection (>1% price diff, >5% market cap diff, sector mismatches)
  - Data completeness checks

- **Index Dashboard**: Sector-first navigation with stock ranking
- **Stock Detail**: Comprehensive analysis including:
  - Financial statements (Income, Balance Sheet, Cash Flow)
  - Key metrics and ratios
  - DCF valuation
  - Analyst estimates and ratings
  - Earnings surprises
  - Sentiment analysis
  - Institutional holdings (5-year historical)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **agno>=2.2.7** (Agent framework)
- **yfinance>=0.2.0** (YFinance for validation)
- Streamlit, OpenAI, Exa-py, and all dependencies

### 2. Configure API Keys

Create a `.env` file in the project root:

```
FMP_API_KEY=your_fmp_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
EXA_API_KEY=your_exa_api_key_here  # Optional, for market intelligence
```

Or set them as environment variables:

```bash
export FMP_API_KEY=your_fmp_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here
export EXA_API_KEY=your_exa_api_key_here  # Optional
```

For Streamlit, also create `.streamlit/secrets.toml`:

```toml
FMP_API_KEY = "your_fmp_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"  # Optional
```

**Note**: 
- `FMP_API_KEY` is required
- `OPENAI_API_KEY` is required for AI-powered agent insights
- `EXA_API_KEY` is optional - enables market intelligence gathering
- YFinance is free (no API key needed)

### 3. Build Index Cache

```bash
python scripts/build_index_cache_agno.py --out data/index_cache --min-coverage 0.9
```

**What happens:**
1. Agno Team coordinates specialized agents
2. Agents use @tool decorators for structured operations
3. FMP and YFinance cross-validation ensures data quality
4. Market intelligence gathered via Exa search (if available)
5. AI insights generated using OpenAI GPT-4o
6. All data cached to disk

**Time:** 10-30 minutes (first time only)

**Expected output:**
```
================================================================
INDEX CACHE BUILDER - Agno Framework
================================================================

✅ Agno Team initialized
✓ Discovered 4 indexes

================================================================
Building cache for S&P 500 (^GSPC)
================================================================

🤖 [Agno Team] Fetching constituents...
✓ Fetched 503 constituents

🤖 [Agno Team] Validating tickers...
✓ Validated 485/503 tickers (96.42% coverage)

🤖 [Data Validation] Cross-validating with YFinance...
✓ Quality scores calculated

🤖 [Agno Team] Segregating by sector/industry...
✓ Segregated into 11 sectors and 68 industries

🤖 [Agno Team] Gathering market intelligence...
✓ Market intelligence gathered

🤖 [Data Enrichment] Enriching tickers with FMP data...
✓ Enriched 485 tickers

✅ Cache built successfully for S&P 500!
```

### 4. Run Streamlit App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

## Agentic Team Architecture

The system uses an **Agno-based agentic team** where specialized agents work together:

1. **ConstituentFetcherAgent**: Fetches index constituents
2. **TickerValidatorAgent**: Validates tickers using FMP and YFinance (cross-validation)
3. **SectorSegregatorAgent**: Groups tickers by sector/industry (with validation)
4. **DataValidationAgent**: Comprehensive quality checks and cross-validation
   - Cross-validates FMP vs YFinance
   - Checks price consistency (>1% threshold)
   - Validates sector/industry matches
   - Calculates quality scores
   - Flags discrepancies
5. **MarketIntelligenceAgent**: Gathers market insights via Exa search (optional)
6. **DataEnricherAgent**: Enriches with validated FMP data
7. **CacheBuilderAgent**: Saves all data + AI insights + validation reports

### Agno Framework Benefits

- ✅ Industry-standard agent framework
- ✅ Built-in agent orchestration
- ✅ Structured @tool decorators
- ✅ Follows market research team pattern
- ✅ Better for complex multi-agent workflows

## Data Validation Features

### Quality Score Calculation
- **EXCELLENT** (90-100): Both sources valid, no discrepancies
- **GOOD** (70-89): Both sources valid, minor discrepancies
- **FAIR** (50-69): One source valid or significant discrepancies
- **POOR** (<50): Missing data or major issues

### Discrepancy Thresholds
- **Price**: >1% difference flagged
- **Market Cap**: >5% difference flagged
- **Sector/Industry**: Any mismatch flagged

### Validation Checks
- Profile completeness (required fields)
- Price data accuracy
- Sector/industry classification consistency
- Market cap consistency
- Data freshness

## Project Structure

```
.
├── app.py                          # Main Streamlit app
├── pages/
│   ├── 1_Index_Dashboard.py       # Index and sector ranking
│   └── 2_Stock_Detail.py          # Detailed stock analysis
├── src/
│   ├── fmp_client.py              # FMP API client
│   ├── indexes.py                 # Index data loading
│   ├── scoring.py                 # Stock ranking/scoring
│   ├── data.py                    # Financial data fetching
│   ├── valuation.py               # DCF and valuation
│   ├── sentiment.py               # Sentiment analysis
│   ├── analyst.py                 # Analyst estimates
│   ├── earnings.py                # Earnings data
│   ├── institutional.py           # Institutional holdings
│   ├── components.py             # Reusable UI components
│   └── utils.py                   # Utility functions
├── scripts/
│   ├── agents/
│   │   ├── agno_agents.py        # Agno-based agents and tools
│   │   └── config.py             # API keys and configuration
│   └── build_index_cache_agno.py # Index cache builder (Agno framework)
├── data/
│   └── index_cache/               # Cached index data (generated)
├── docs/                          # Additional documentation
├── requirements.txt
└── README.md
```

## Usage

### Index Dashboard

1. Select an index from the sidebar
2. Choose a sector (or "All Sectors")
3. View ranked stocks with composite scores
4. Click on any stock symbol to view detailed analysis

### Stock Detail

- Enter a ticker symbol or click from Index Dashboard
- View comprehensive financial analysis
- All charts default to 5-year window (configurable)
- Download financial data as Excel

## Troubleshooting

### "ModuleNotFoundError: No module named 'agno'"
```bash
pip install agno>=2.2.7
```

### "ModuleNotFoundError: No module named 'yfinance'"
```bash
pip install yfinance>=0.2.0
```

### "No cached indexes found"
Run the cache builder first:
```bash
python scripts/build_index_cache_agno.py --out data/index_cache --min-coverage 0.9
```

### API key errors
- Check `.env` file exists and has all required keys
- Verify keys are valid
- For Streamlit, also check `.streamlit/secrets.toml`

### Agno team initialization fails
- Check OPENAI_API_KEY is valid
- Verify agno is installed: `pip install agno>=2.2.7`

## Quick Test

```bash
# Test Agno installation
python -c "from agno.agent import Agent; print('✅ Agno installed')"

# Test YFinance installation
python -c "import yfinance as yf; print('✅ YFinance installed')"

# Test API keys
python -c "from scripts.agents.config import FMP_API_KEY, OPENAI_API_KEY, EXA_API_KEY; print(f'✅ Keys loaded: FMP={bool(FMP_API_KEY)}, OpenAI={bool(OPENAI_API_KEY)}, Exa={bool(EXA_API_KEY)}')"
```

## Data Sources

- **Primary**: Financial Modeling Prep (FMP) API
- **Validation**: Yahoo Finance (YFinance) - free, no API key needed
- **Intelligence**: Exa Search (optional)
- **Caching**: Aggressive on-disk caching for performance
- **Fallback**: Live API fetch if cache unavailable

## Requirements

- Python 3.8+
- FMP API key (required)
- OpenAI API key (required for AI agents)
- Exa API key (optional, for market intelligence)
- See `requirements.txt` for all dependencies

## Notes

- Index cache must be built before using Index Dashboard
- All plots default to 5-year window
- Institutional holdings show 5-year historical data
- Sector-first ranking with normalization toggle
- All FMP responses are cross-validated with YFinance
- Quality scores and validation reports are stored in cache

## Next Steps

1. **First run**: `python scripts/build_index_cache_agno.py --out data/index_cache --min-coverage 0.9`
2. **Daily use**: `streamlit run app.py`
3. **Refresh cache**: Re-run the build command when needed

Enjoy your AI-powered Equities Analytics Dashboard! 🚀
