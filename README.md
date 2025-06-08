# Stock Portfolio Dashboard

A Streamlit application for analyzing and visualizing your stock portfolio performance.

## Features

- **Portfolio Overview**: View total value, investment, and performance of each stock
- **Performance vs S&P500**: Compare your portfolio performance against the S&P 500 index
- **Sector Diversification**: Analyze portfolio holdings and revenue diversification by sector
- **Metrics Comparison**: Compare key financial metrics across your portfolio stocks
- **Stock Comparison**: Compare performance and metrics between any two stocks in your portfolio

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`

## Installation

1. Install the required packages:
```
pip install -r requirements.txt
```

2. Make sure your portfolio data is in an Excel file named `port.xlsx` with the following columns:
   - Stock Name
   - Stock Ticker
   - Stock Number
   - Stock Purchase Price

## Usage

Run the application with:
```
streamlit run app.py
```

Use the sidebar to navigate between different dashboard views and adjust the time period for historical data analysis.

## Data Source

This application uses Yahoo Finance (yfinance) API to fetch real-time and historical stock data.
