# Equities Analytics Dashboard

A comprehensive Streamlit-based dashboard for equity analysis with automated index data ingestion and AI-powered agentic team.

## Features

- **Index Dashboard**: Sector-first navigation with interactive visualizations, stock rankings, and composite scoring
- **Stock Analysis Dashboard**: Comprehensive financial analysis including:
  - Financial statements (Income, Balance Sheet, Cash Flow)
  - Key metrics and ratios
  - DCF valuation
  - Analyst estimates, ratings, and grades
  - Earnings surprises
  - Sentiment analysis (News & Social)
  - Institutional holdings (5-year historical)
  - Economic indicators

- **AI-Powered Agentic Team**: Specialized AI agents using Agno framework:
  - Automatic index discovery
  - Constituent fetching and validation
  - Sector/industry segregation
  - Data enrichment and validation
  - Market intelligence gathering

- **Composite Scoring System**: Multi-factor scoring (0-5 scale) with:
  - 50% weight to rating scores (ratingScore + ratingDetails scores)
  - 50% weight to other parameters (market cap, volume, beta, financial ratios, analyst sentiment, earnings, institutional changes)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```
FMP_API_KEY=your_fmp_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional, for AI agents
EXA_API_KEY=your_exa_api_key_here  # Optional, for market intelligence
```

For Streamlit, also create `.streamlit/secrets.toml`:

```toml
FMP_API_KEY = "your_fmp_api_key_here"
OPENAI_API_KEY = "your_openai_api_key_here"  # Optional
```

**Note**: `FMP_API_KEY` is required. Other keys are optional.

### 3. Build Index Cache

```bash
python scripts/build_index_cache_agno.py --out data/index_cache --min-coverage 0.9
```

This builds cached data for all discovered indexes (typically 10-30 minutes first time).

### 4. Run Streamlit App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

## Project Structure

```
.
├── app.py                          # Main Streamlit app
├── pages/
│   ├── 1_Index_Dashboard.py       # Index and sector ranking
│   └── 2_Stock_Analysis_Dashboard.py  # Detailed stock analysis
├── src/
│   ├── fmp_client.py              # FMP API client
│   ├── indexes.py                 # Index data loading
│   ├── scoring.py                 # Stock ranking/scoring
│   ├── data.py                    # Financial data fetching
│   ├── valuation.py               # DCF and valuation
│   ├── sentiment.py                # Sentiment analysis
│   ├── analyst.py                 # Analyst estimates
│   ├── earnings.py                # Earnings data
│   ├── institutional.py           # Institutional holdings
│   ├── components.py              # Reusable UI components
│   ├── utils.py                   # Utility functions
│   └── fmp_fallback.py            # FMP API with web scraping fallback
├── scripts/
│   ├── agents/
│   │   ├── agno_agents.py        # Agno-based agents and tools
│   │   └── config.py             # API keys and configuration
│   └── build_index_cache_agno.py # Index cache builder
├── data/
│   └── index_cache/               # Cached index data (generated)
├── docs/                          # Additional documentation
├── requirements.txt
└── README.md
```

## Usage

### Index Dashboard

1. Select an index from the sidebar (e.g., S&P 500)
2. Choose a sector or "All Sectors"
3. View ranked stocks with composite scores
4. Click on any stock symbol to view detailed analysis

### Stock Analysis Dashboard

- Enter a ticker symbol or click from Index Dashboard
- View comprehensive financial analysis
- All charts default to 5-year window
- Download financial data as Excel

## Scoring Algorithm

The composite score (0-5) is calculated as:

1. **Rating Scores Group** (50% weight):
   - Average of: `ratingScore`, `ratingDetailsDCFScore`, `ratingDetailsROEScore`, `ratingDetailsROAScore`, `ratingDetailsDEScore`, `ratingDetailsPEScore`, `ratingDetailsPBScore`

2. **Other Parameters Group** (50% weight):
   - Market Cap, Volume, Beta, Dividend Yield, ROE, Debt to Equity, Current Ratio
   - FCF Growth, Net Income Growth, Operating Margin, Avg Shares Dil Growth
   - Analyst Sentiment
   - Earnings Surprises, Institutional Net Change

Final score = `(avg_rating_score × 0.5) + (avg_other_score × 0.5)`

## Data Sources

- **Primary**: Financial Modeling Prep (FMP) API
- **Fallback**: Web scraping with AI agents (when FMP unavailable)
- **Caching**: Aggressive on-disk caching for performance

## Requirements

- Python 3.8+
- FMP API key (required)
- OpenAI/Gemini API key (optional, for AI agents)
- Exa API key (optional, for market intelligence)

## Troubleshooting

### "No cached indexes found"
Run the cache builder first:
```bash
python scripts/build_index_cache_agno.py --out data/index_cache --min-coverage 0.9
```

### API key errors
- Check `.env` file exists and has `FMP_API_KEY`
- For Streamlit, also check `.streamlit/secrets.toml`

### Module not found errors
```bash
pip install -r requirements.txt
```

## Notes

- Index cache must be built before using Index Dashboard
- All plots default to 5-year window
- Institutional holdings show 5-year historical data
- Analyst sentiment uses actual `newGrade` values from analyst grades
- Composite score is calculated on 0-5 scale
