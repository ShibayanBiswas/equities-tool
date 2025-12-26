"""
Module for retrieving financial data using Financial Modeling Prep and yfinance APIs.
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

# Access API keys securely from Streamlit's secrets
FMP_API_KEY = st.secrets["FMP_API_KEY"]

def get_company_info(symbol: str) -> dict:
    """
    Fetches company information based on the stock symbol.

    Args:
        symbol (str): The stock symbol of the company.

    Returns:
        dict: Company details or None if an error occurs.
    """
    api_endpoint = f'https://financialmodelingprep.com/api/v3/profile/{symbol}/'
    params = {'apikey': FMP_API_KEY}

    try:
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        data = response.json()[0]
        company_info = {
            'Name': data['companyName'],
            'Exchange': data['exchangeShortName'],
            'Currency': data['currency'],
            'Country': data['country'],            
            'Sector': data['sector'],
            'Market Cap': data['mktCap'],
            'Price': data['price'],
            'Beta': data['beta'],
            'Price change': data['changes'],
            'Website': data['website'],
            'Image': data['image']
        }
        return company_info

    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return None

    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return None

def get_stock_price(symbol: str) -> pd.DataFrame:
    """
    Retrieves daily closing prices for the past 5 years using yfinance.

    Args:
        symbol (str): The stock symbol.

    Returns:
        pd.DataFrame: DataFrame with date index and closing prices.
    """
    try:
        # Define the time period
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=5)
        
        # Download data using yfinance
        df = yf.download(symbol, start=start_date, end=end_date)
        df = df[['Close']]
        df = df.rename(columns={'Close': 'Price'})
        return df

    except Exception as e:
        print(f"Error retrieving stock price data: {e}")
        return None

def get_income_statement(symbol: str) -> pd.DataFrame:
    """
    Retrieves the income statement for the specified company.

    Args:
        symbol (str): The stock symbol.

    Returns:
        pd.DataFrame: DataFrame containing income statement data.
    """
    api_endpoint = f'https://financialmodelingprep.com/api/v3/income-statement/{symbol}/'
    params = {'limit': 5, 'apikey': FMP_API_KEY}

    try:
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        income_statement_data = [{
            'Year': report['calendarYear'],
            'Revenue': report['revenue'],
            '(-) Cost of Revenue': report['costOfRevenue'],
            '= Gross Profit': report['grossProfit'],
            '(-) Operating Expense': report['operatingExpenses'],
            '= Operating Income': report['operatingIncome'],
            '(+-) Other Income/Expenses': report['totalOtherIncomeExpensesNet'],
            '= Income Before Tax': report['incomeBeforeTax'],                
            '(+-) Tax Income/Expense': report['incomeTaxExpense'],
            '= Net Income': report['netIncome'],
        } for report in response_data]
        income_statement = pd.DataFrame(income_statement_data).set_index('Year')
        return income_statement

    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return None

    except ValueError as e:
        print(f"JSON parsing error: {e}")
        return None

def get_balance_sheet(symbol: str) -> pd.DataFrame:
    """
    Retrieves the balance sheet for the specified company.

    Args:
        symbol (str): The stock symbol.

    Returns:
        pd.DataFrame: DataFrame containing balance sheet data.
    """
    api_endpoint = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}'
    params = {'limit': 5, 'apikey': FMP_API_KEY}

    try:
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        balance_sheet_data = [{
            'Year': report['calendarYear'],
            'Assets': report['totalAssets'],
            'Current Assets': report['totalCurrentAssets'],
            'Non-Current Assets': report['totalNonCurrentAssets'],
            'Current Liabilities': report['totalCurrentLiabilities'],
            'Non-Current Liabilities': report['totalNonCurrentLiabilities'],
            'Liabilities': report['totalLiabilities'],
            'Equity': report['totalEquity']
        } for report in response_data]
        balance_sheet_df = pd.DataFrame(balance_sheet_data).set_index('Year')
        return balance_sheet_df

    except requests.exceptions.RequestException as e:
        print(f"Balance sheet API error: {e}")
        return None

def get_cash_flow(symbol: str) -> pd.DataFrame:
    """
    Retrieves the cash flow statement for the specified company.

    Args:
        symbol (str): The stock symbol.

    Returns:
        pd.DataFrame: DataFrame containing cash flow data.
    """
    api_endpoint = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}'
    params = {'limit': 5, 'apikey': FMP_API_KEY}
        
    try:
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        cashflow_data = [{
            'Year': report['date'].split('-')[0],
            "Cash flows from operating activities": report['netCashProvidedByOperatingActivities'],
            'Cash flows from investing activities': report['netCashUsedForInvestingActivites'],
            'Cash flows from financing activities': report['netCashUsedProvidedByFinancingActivities'],
            'Free cash flow': report['freeCashFlow']
        } for report in response_data]
        cashflow_df = pd.DataFrame(cashflow_data).set_index('Year')
        return cashflow_df

    except requests.exceptions.RequestException as e:
        print(f"Cash flow API error: {e}")
        return None

def get_key_metrics(symbol: str) -> pd.DataFrame:
    """
    Retrieves key financial metrics for the specified company.

    Args:
        symbol (str): The stock symbol.

    Returns:
        pd.DataFrame: DataFrame containing key metrics.
    """
    api_endpoint = f'https://financialmodelingprep.com/api/v3/key-metrics/{symbol}'
    params = {'limit': 5, 'apikey': FMP_API_KEY}

    try:
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        metrics_data = [{
            'Year': report['date'].split('-')[0],
            "Market Cap": report['marketCap'],
            'Working Capital': report['workingCapital'],
            'D/E ratio': report['debtToEquity'],
            'P/E Ratio': report['peRatio'],
            'ROE': report['roe'], 
            'Dividend Yield': report['dividendYield']
        } for report in response_data]
        metrics_df = pd.DataFrame(metrics_data).set_index('Year')
        return metrics_df

    except requests.exceptions.RequestException as e:
        print(f"Key metrics API error: {e}")
        return None

def get_financial_ratios(symbol: str) -> pd.DataFrame:
    """
    Retrieves financial ratios for the specified company.

    Args:
        symbol (str): The stock symbol.

    Returns:
        pd.DataFrame: DataFrame containing financial ratios.
    """
    api_endpoint = f'https://financialmodelingprep.com/api/v3/ratios/{symbol}'
    params = {'limit': 5, 'apikey': FMP_API_KEY}

    try:
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()
        response_data = response.json()
        ratios_data = [{
            'Year': report['date'].split('-')[0],
            'Current Ratio': report['currentRatio'],
            'Quick Ratio': report['quickRatio'],
            'Cash Ratio': report['cashRatio'],
            'Days of Sales Outstanding': report['daysOfSalesOutstanding'],
            'Days of Inventory Outstanding': report['daysOfInventoryOutstanding'],
            'Operating Cycle': report['operatingCycle'],
            'Days of Payables Outstanding': report['daysOfPayablesOutstanding'],
            'Cash Conversion Cycle': report['cashConversionCycle'],
            'Gross Profit Margin': report['grossProfitMargin'], 
            'Operating Profit Margin': report['operatingProfitMargin'],
            'Pretax Profit Margin': report['pretaxProfitMargin'],
            'Net Profit Margin': report['netProfitMargin'],
            'Effective Tax Rate': report['effectiveTaxRate'],
            'Return on Assets': report['returnOnAssets'],
            'Return on Equity': report['returnOnEquity'],
            'Return on Capital Employed': report['returnOnCapitalEmployed'],
            'Net Income per EBT': report['netIncomePerEBT'],
            'EBT per EBIT': report['ebtPerEbit'],
            'EBIT per Revenue': report['ebitPerRevenue'],
            'Debt Ratio': report['debtRatio'],
            'Debt Equity Ratio': report['debtEquityRatio'],
            'Long-term Debt to Capitalization': report['longTermDebtToCapitalization'],
            'Total Debt to Capitalization': report['totalDebtToCapitalization'],
            'Interest Coverage': report['interestCoverage'],
            'Cash Flow to Debt Ratio': report['cashFlowToDebtRatio'],
            'Company Equity Multiplier': report['companyEquityMultiplier'],
            'Receivables Turnover': report['receivablesTurnover'],
            'Payables Turnover': report['payablesTurnover'],
            'Inventory Turnover': report['inventoryTurnover'],
            'Fixed Asset Turnover': report['fixedAssetTurnover'],
            'Asset Turnover': report['assetTurnover'],
            'Operating Cash Flow per Share': report['operatingCashFlowPerShare'],
            'Free Cash Flow per Share': report['freeCashFlowPerShare'],
            'Cash per Share': report['cashPerShare'],
            'Payout Ratio': report['payoutRatio'],
            'Operating Cash Flow Sales Ratio': report['operatingCashFlowSalesRatio'],
            'Free Cash Flow Operating Cash Flow Ratio': report['freeCashFlowOperatingCashFlowRatio'],
            'Cash Flow Coverage Ratios': report['cashFlowCoverageRatios'],
            'Price to Book Value Ratio': report['priceToBookRatio'],
            'Price to Earnings Ratio': report['priceEarningsRatio'],
            'Price to Sales Ratio': report['priceToSalesRatio'],
            'Dividend Yield': report['dividendYield'],
            'Enterprise Value to EBITDA': report['enterpriseValueMultiple'],
            'Price to Fair Value': report['priceFairValue']
        } for report in response_data]
        ratios_df = pd.DataFrame(ratios_data).set_index('Year')
        return ratios_df

    except requests.exceptions.RequestException as e:
        print(f"Financial ratios API error: {e}")
        return None
