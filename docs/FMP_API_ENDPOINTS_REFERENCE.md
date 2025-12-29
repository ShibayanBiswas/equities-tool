# FMP API Endpoints Reference

This document serves as a reference for Financial Modeling Prep (FMP) API endpoints based on the official documentation.

**Base URL:** `https://financialmodelingprep.com/stable/`

**API Key Format:** Append `?apikey=YOUR_API_KEY` or `&apikey=YOUR_API_KEY` (if other params exist)

**Documentation:** https://site.financialmodelingprep.com/developer/docs

---

## Analyst Endpoints

### Analyst Estimates
- **Endpoint:** `analyst-estimates`
- **Parameters:** `symbol`, `period` (annual/quarter), `limit`
- **Example:** `analyst-estimates?symbol=AAPL&period=annual&limit=5`
- **Fallback:** `v3/analyst-estimates/{symbol}`

### Analyst Rating
- **Endpoint:** `rating`
- **Parameters:** `symbol`
- **Example:** `rating?symbol=AAPL`
- **Alternatives:**
  - `v3/rating/{symbol}`
  - `rating-bulk?symbol={symbol}`

### Price Target Summary
- **Endpoint:** `price-target-summary`
- **Parameters:** `symbol`
- **Example:** `price-target-summary?symbol=AAPL`

### Upgrades/Downgrades
- **Endpoint:** `upgrades-downgrades`
- **Parameters:** `symbol`, `limit`
- **Example:** `upgrades-downgrades?symbol=AAPL&limit=100`

### Analyst Recommendations
- **Endpoint:** `analyst-recommendations`
- **Parameters:** `symbol`
- **Example:** `analyst-recommendations?symbol=AAPL`

### Grade (Analyst Actions)
- **Endpoint:** `grade`
- **Parameters:** `symbol`, `limit`
- **Example:** `grade?symbol=AAPL&limit=500`
- **Note:** Includes previous and new grades from analysts

---

## News Endpoints

### Stock News
- **Endpoint:** `v3/stock_news` (Note: v3 endpoint, not stable)
- **Parameters:** `tickers` (comma-separated), `limit`
- **Example:** `v3/stock_news?tickers=AAPL&limit=100`
- **Returns:** News articles with text, title, publishedDate, url, site

### Company News (via Company Outlook)
- **Endpoint:** `v4/company-outlook`
- **Parameters:** `symbol`
- **Example:** `v4/company-outlook?symbol=AAPL`
- **Returns:** Includes `news` array in response

### Press Releases
- **Endpoint:** `v3/press-releases/{symbol}`
- **Parameters:** `limit`
- **Example:** `v3/press-releases/AAPL?limit=100`

---

## Institutional Holdings (Form 13F)

### Institutional Holders
- **Endpoint:** `v3/institutional-holder/{symbol}`
- **Alternative:** `institutional-holder?symbol={symbol}`
- **Example:** `v3/institutional-holder/AAPL`
- **Returns:** Current institutional holdings with name, shares, change, dateReported

### Form 13F Filings
- **Endpoint:** `v3/form-thirteen/{symbol}`
- **Parameters:** `limit`
- **Example:** `v3/form-thirteen/AAPL?limit=20`

### Form 13F Holdings
- **Endpoint:** `v3/form-thirteen-holdings/{symbol}`
- **Parameters:** `limit`
- **Example:** `v3/form-thirteen-holdings/AAPL?limit=20`

---

## Market Performance Endpoints

### Historical Price Full
- **Endpoint:** `v3/historical-price-full/{symbol}`
- **Parameters:** `from` (date), `to` (date)
- **Example:** `v3/historical-price-full/AAPL?from=2020-01-01&to=2024-12-31`

### Historical Price (Intraday)
- **Endpoint:** `v3/historical-chart/{interval}/{symbol}`
- **Intervals:** `1min`, `5min`, `15min`, `30min`, `1hour`, `4hour`, `1day`
- **Example:** `v3/historical-chart/1hour/AAPL`

### Stock Price Change
- **Endpoint:** `v3/stock-price-change/{symbol}`
- **Example:** `v3/stock-price-change/AAPL`

---

## Earnings & Transcripts

### Earnings Surprises
- **Endpoint:** `earnings-surprises`
- **Parameters:** `symbol`, `limit`
- **Example:** `earnings-surprises?symbol=AAPL&limit=20`
- **Fallback:** `v3/earnings-surprises/{symbol}`

### Earnings Calendar
- **Endpoint:** `v3/earnings-calendar`
- **Parameters:** `from` (date), `to` (date)
- **Example:** `v3/earnings-calendar?from=2024-01-01&to=2024-12-31`

### Earnings Transcripts
- **Endpoint:** `v3/earnings_transcript/{symbol}`
- **Parameters:** `quarter`, `year`
- **Example:** `v3/earnings_transcript/AAPL?quarter=1&year=2024`

---

## Discounted Cash Flow (DCF)

### DCF Valuation
- **Endpoint:** `discounted-cash-flow`
- **Parameters:** `symbol`, `limit`
- **Example:** `discounted-cash-flow?symbol=AAPL&limit=10`
- **Fallback:** `v3/discounted-cash-flow/{symbol}`

### DCF Bulk
- **Endpoint:** `dcf-bulk`
- **Example:** `dcf-bulk`

---

## Indexes

### S&P 500 Constituents
- **Endpoint:** `sp500-constituent`
- **Example:** `sp500-constituent`
- **Fallback:** `v3/sp500_constituent`

### NASDAQ Constituents
- **Endpoint:** `nasdaq-constituent`
- **Example:** `nasdaq-constituent`
- **Fallback:** `v3/nasdaq_constituent`

### Dow Jones Constituents
- **Endpoint:** `dowjones-constituent`
- **Example:** `dowjones-constituent`

### Russell 2000 Constituents
- **Endpoint:** `russell2000-constituent`
- **Example:** `russell2000-constituent`

---

## Insider Trades

### Insider Trades
- **Endpoint:** `v4/insider-trading`
- **Parameters:** `symbol`, `limit`, `transactionType`
- **Example:** `v4/insider-trading?symbol=AAPL&limit=100`

### Insider Roster
- **Endpoint:** `v4/insider-roster`
- **Parameters:** `symbol`
- **Example:** `v4/insider-roster?symbol=AAPL`

### Insider Transactions Summary
- **Endpoint:** `v4/insider-summary`
- **Parameters:** `symbol`
- **Example:** `v4/insider-summary?symbol=AAPL`

---

## SEC Filings

### SEC Filings
- **Endpoint:** `v3/sec_filings/{symbol}`
- **Parameters:** `type`, `limit`
- **Example:** `v3/sec_filings/AAPL?type=10-K&limit=10`

### SEC Filing Types
- **Types:** `10-K`, `10-Q`, `8-K`, `DEF 14A`, etc.

---

## ESG Data

### ESG Score
- **Endpoint:** `v4/esg-environmental-social-governance-data`
- **Parameters:** `symbol`
- **Example:** `v4/esg-environmental-social-governance-data?symbol=AAPL`

### ESG Risk Rating
- **Endpoint:** `v4/esg-risk-rating`
- **Parameters:** `symbol`
- **Example:** `v4/esg-risk-rating?symbol=AAPL`

---

## Economics Data

### Economic Indicators
- **Endpoint:** `v4/economic`
- **Parameters:** `name`, `from`, `to`
- **Example:** `v4/economic?name=GDP&from=2020-01-01&to=2024-12-31`

### Treasury Rates
- **Endpoint:** `v4/treasury`
- **Parameters:** `from`, `to`
- **Example:** `v4/treasury?from=2020-01-01&to=2024-12-31`

---

## Market Calendar

### Market Hours
- **Endpoint:** `v3/market-hours`
- **Example:** `v3/market-hours`

### Market Status
- **Endpoint:** `v3/is-the-market-open`
- **Example:** `v3/is-the-market-open`

### Stock Market Holidays
- **Endpoint:** `v3/stock_market_holidays`
- **Example:** `v3/stock_market_holidays`

---

## Social Sentiment

### Historical Social Sentiment
- **Endpoint:** `v4/historical/social-sentiment`
- **Parameters:** `symbol`, `page`
- **Example:** `v4/historical/social-sentiment?symbol=AAPL&page=0`
- **Returns:** sentiment, absoluteIndex, relativeIndex, generalPerception

---

## Notes

1. **Stable API:** Most endpoints should use `/stable/` prefix, but some (like `v3/stock_news`, `v4/company-outlook`) still use versioned paths.

2. **API Key:** Always append API key as query parameter: `?apikey=KEY` or `&apikey=KEY`

3. **Rate Limits:** Check your API plan for rate limits and quota restrictions.

4. **Premium Features:** Some endpoints may require premium API plans (e.g., institutional data, ESG data).

5. **Error Handling:** Always implement fallback logic for endpoints that may not be available in free tier.

---

## Current Implementation Status

### ✅ Implemented
- Analyst Estimates (`analyst-estimates`)
- Analyst Rating (`rating`)
- Earnings Surprises (`earnings-surprises`)
- DCF (`discounted-cash-flow`)
- Stock News (`v3/stock_news`)
- Social Sentiment (`v4/historical/social-sentiment`)
- Institutional Holders (`v3/institutional-holder/{symbol}`)
- Grade/Analyst Actions (`grade`)

### ⚠️ Needs Update
- Some endpoints may need to switch from v3 to stable format
- Institutional holders endpoint may need Form 13F endpoint as alternative

### 📋 To Implement
- Price Target Summary
- Upgrades/Downgrades
- Press Releases
- Form 13F Filings
- Insider Trades
- SEC Filings
- ESG Data
- Earnings Transcripts
- Market Calendar

---

## References

- [FMP API Documentation](https://site.financialmodelingprep.com/developer/docs)
- [Analyst Endpoints](https://site.financialmodelingprep.com/developer/docs#analyst)
- [News Endpoints](https://site.financialmodelingprep.com/developer/docs#news)
- [Form 13F Endpoints](https://site.financialmodelingprep.com/developer/docs#form-13f)
- [Market Performance](https://site.financialmodelingprep.com/developer/docs#market-performance)
- [Insider Trades](https://site.financialmodelingprep.com/developer/docs#insider-trades)
- [SEC Filings](https://site.financialmodelingprep.com/developer/docs#sec-filings)
- [Indexes](https://site.financialmodelingprep.com/developer/docs#indexes)
- [DCF](https://site.financialmodelingprep.com/developer/docs#discounted-cash-flow)
- [ESG](https://site.financialmodelingprep.com/developer/docs#esg)
- [Economics](https://site.financialmodelingprep.com/developer/docs#economics)
- [Calendar](https://site.financialmodelingprep.com/developer/docs#calendar)
- [Earnings Transcript](https://site.financialmodelingprep.com/developer/docs#earnings-transcript)

