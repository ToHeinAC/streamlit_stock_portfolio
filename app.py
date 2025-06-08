import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import requests
from functools import lru_cache

# Set page configuration
st.set_page_config(
    page_title="Stock Portfolio Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Stock Portfolio Dashboard")
st.markdown("---")

# Function to load portfolio data
@st.cache_data(ttl=3600)
def load_portfolio_data():
    try:
        df = pd.read_excel('port.xlsx')
        required_columns = ['Stock Name', 'Stock Ticker', 'Stock Number', 'Stock Purchase Price']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return None
        return df
    except Exception as e:
        st.error(f"Error loading portfolio data: {e}")
        return None

# Function to get stock history data
@st.cache_data(ttl=3600)
def get_stock_history(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching history data for {ticker}: {e}")
        return None

# Function to get stock info
def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock
    except Exception as e:
        st.error(f"Error fetching info for {ticker}: {e}")
        return None

# Function to get exchange rate from any currency to EUR
@st.cache_data(ttl=3600)
def get_exchange_rate_to_eur(currency_code):
    if currency_code == 'EUR':
        return 1.0
    
    try:
        # For most currencies, we need to get the rate from XXX to EUR
        # Yahoo Finance format for EUR/XXX is EURXXX=X
        # For example, EUR/USD is EURUSD=X
        
        # For common currencies, we can get direct rates
        if currency_code in ['USD', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD']:
            # Get EUR to XXX rate (e.g., EURUSD=X)
            ticker_symbol = f"EUR{currency_code}=X"
            forex_data = yf.Ticker(ticker_symbol)
            hist = forex_data.history(period="1d")
            
            if not hist.empty:
                # This gives us how many XXX per 1 EUR
                # For conversion from XXX to EUR, we need to divide by this rate
                rate = hist['Close'].iloc[-1]
                return 1.0 / rate
        
        # For JPY, which has a special case due to its value
        elif currency_code == 'JPY':
            ticker_symbol = f"EURJPY=X"
            forex_data = yf.Ticker(ticker_symbol)
            hist = forex_data.history(period="1d")
            
            if not hist.empty:
                # This gives us how many JPY per 1 EUR
                # For conversion from JPY to EUR, we need to divide by this rate
                rate = hist['Close'].iloc[-1]
                return 1.0 / rate
        
        # For other currencies, try to get XXX to EUR directly
        else:
            # Try direct XXX to EUR rate (e.g., USDEUR=X)
            ticker_symbol = f"{currency_code}EUR=X"
            forex_data = yf.Ticker(ticker_symbol)
            hist = forex_data.history(period="1d")
            
            if not hist.empty and hist['Close'].iloc[-1] > 0:
                return hist['Close'].iloc[-1]
            else:
                # If direct rate not available, try via USD
                # First get XXX to USD
                ticker_symbol_to_usd = f"{currency_code}USD=X"
                forex_data_to_usd = yf.Ticker(ticker_symbol_to_usd)
                hist_to_usd = forex_data_to_usd.history(period="1d")
                
                # Then get USD to EUR
                ticker_symbol_usd_to_eur = "USDEUR=X"
                forex_data_usd_to_eur = yf.Ticker(ticker_symbol_usd_to_eur)
                hist_usd_to_eur = forex_data_usd_to_eur.history(period="1d")
                
                if not hist_to_usd.empty and not hist_usd_to_eur.empty:
                    rate_to_usd = hist_to_usd['Close'].iloc[-1]
                    rate_usd_to_eur = hist_usd_to_eur['Close'].iloc[-1]
                    return rate_to_usd * rate_usd_to_eur
        
        # If all else fails, use a fallback method
        st.warning(f"Could not fetch reliable exchange rate for {currency_code}. Using fallback method.")
        # Try EUR to XXX and invert it
        ticker_symbol = f"EUR{currency_code}=X"
        forex_data = yf.Ticker(ticker_symbol)
        hist = forex_data.history(period="1d")
        
        if not hist.empty and hist['Close'].iloc[-1] > 0:
            return 1.0 / hist['Close'].iloc[-1]
        
        st.warning(f"All exchange rate methods failed for {currency_code}. Using 1.0.")
        return 1.0
    except Exception as e:
        st.warning(f"Error fetching exchange rate for {currency_code}: {e}. Using 1.0.")
        return 1.0

# Function to convert currency to EUR
def convert_to_eur(amount, currency_code):
    if currency_code == 'EUR':
        return amount
    
    # Get the exchange rate (how many EUR per 1 unit of currency_code)
    exchange_rate = get_exchange_rate_to_eur(currency_code)
    
    # Multiply by the rate to get EUR
    return amount * exchange_rate

# Function to get S&P 500 data
@st.cache_data(ttl=3600)
def get_sp500_data(period="1y"):
    try:
        sp500 = yf.Ticker("^GSPC")
        hist = sp500.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching S&P 500 data: {e}")
        return None

# Load portfolio data
portfolio_df = load_portfolio_data()

# Dictionary to store exchange rates for reuse
exchange_rates_cache = {}

# Initialize metrics DataFrame
metrics_df = pd.DataFrame()
if portfolio_df is not None:
    metrics_df['Stock Name'] = portfolio_df['Stock Name']
    metrics_df['Stock Ticker'] = portfolio_df['Stock Ticker']
    
    # Initialize metrics columns
    metrics_to_extract = [
        'debtToEquity', 'revenuePerShare', 'trailingPE', 'pegRatio', 'trailingPegRatio',
        'forwardPE', 'priceToBook', 'earningsPerShare', 'dividendYield', 'dividendRate',
        'dividendGrowth', 'targetLowPrice', 'targetMedianPrice', 'targetHighPrice', 'currentPrice'
    ]
    
    for metric in metrics_to_extract:
        metrics_df[metric] = np.nan

if portfolio_df is not None:
    # Calculate current values and add to dataframe
    portfolio_df['Current Price'] = 0.0
    portfolio_df['Current Value'] = 0.0
    portfolio_df['Profit/Loss'] = 0.0
    portfolio_df['Profit/Loss %'] = 0.0
    portfolio_df['Sector'] = ""
    portfolio_df['Currency'] = ""  # Add currency column
    portfolio_df['Original Current Price'] = 0.0  # Add column for original currency price
    
    # Get current prices and calculate values
    for i, row in portfolio_df.iterrows():
        ticker = row['Stock Ticker']
        hist = get_stock_history(ticker)
        
        if hist is not None and not hist.empty:
            current_price = hist['Close'].iloc[-1]
            
            # Get currency and sector information
            currency = 'USD'  # Default to USD if not found
            try:
                stock_info = get_stock_info(ticker)
                if stock_info is not None:
                    info = stock_info.info
                    portfolio_df.at[i, 'Sector'] = info.get('sector', 'Unknown')
                    
                    # Get currency information
                    currency = info.get('currency', 'USD')
                    portfolio_df.at[i, 'Currency'] = currency
                else:
                    portfolio_df.at[i, 'Sector'] = 'Unknown'
                    portfolio_df.at[i, 'Currency'] = currency
            except Exception as e:
                portfolio_df.at[i, 'Sector'] = 'Unknown'
                portfolio_df.at[i, 'Currency'] = currency
                st.error(f"Error getting stock info for {ticker}: {e}")
            
            # Store original currency values for reference
            portfolio_df.at[i, 'Original Current Price'] = current_price
            
            # Convert prices to EUR
            current_price_eur = convert_to_eur(current_price, currency)
            purchase_price_eur = convert_to_eur(row['Stock Purchase Price'], currency)
            
            portfolio_df.at[i, 'Current Price'] = current_price_eur
            portfolio_df.at[i, 'Current Value'] = current_price_eur * row['Stock Number']
            portfolio_df.at[i, 'Investment Value'] = purchase_price_eur * row['Stock Number']  # Add Investment Value
            portfolio_df.at[i, 'Profit/Loss'] = (current_price_eur - purchase_price_eur) * row['Stock Number']
            portfolio_df.at[i, 'Profit/Loss %'] = (current_price_eur / purchase_price_eur - 1) * 100
            # Also update the Stock Purchase Price to the converted EUR value
            portfolio_df.at[i, 'Stock Purchase Price'] = purchase_price_eur
    
    # Sidebar with navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Dashboard",
        ["Portfolio Overview", "Performance vs S&P500", "Sector Diversification", "Metrics Comparison", "Stock Comparison"]
    )
    
    # Time period selection for historical data
    st.sidebar.title("Time Period")
    time_period = st.sidebar.slider(
        "Select Time Period",
        min_value=1,
        max_value=60,
        value=12,
        step=1,
        help="Number of months to look back"
    )
    
    period_str = f"{time_period}mo"
    start_date = (datetime.now() - timedelta(days=30*time_period)).strftime('%Y-%m-%d')
    
    # Portfolio Overview
    if page == "Portfolio Overview":
        st.header("Portfolio Overview")
        
        # Calculate total metrics
        total_investment = portfolio_df['Investment Value'].sum()  # Use the new Investment Value column
        total_current_value = portfolio_df['Current Value'].sum()
        total_profit_loss = portfolio_df['Profit/Loss'].sum()
        total_profit_loss_percent = (total_current_value / total_investment - 1) * 100
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Investment", f"€{total_investment:,.2f}")
        col2.metric("Current Value", f"€{total_current_value:,.2f}")
        col3.metric("Profit/Loss", f"€{total_profit_loss:,.2f}", f"{total_profit_loss_percent:.2f}%")
        col4.metric("Number of Stocks", f"{len(portfolio_df)}")
        
        # Display portfolio table
        st.subheader("Portfolio Holdings")
        display_df = portfolio_df[['Stock Name', 'Stock Ticker', 'Stock Number', 
                                'Stock Purchase Price', 'Current Price', 'Investment Value', 'Current Value', 
                                'Profit/Loss', 'Profit/Loss %', 'Sector', 'Currency']]
        display_df = display_df.sort_values(by='Current Value', ascending=False)
        st.dataframe(display_df.style.format({
            'Stock Purchase Price': '€{:.2f}',
            'Current Price': '€{:.2f}',
            'Investment Value': '€{:.2f}',
            'Current Value': '€{:.2f}',
            'Profit/Loss': '€{:.2f}',
            'Profit/Loss %': '{:.2f}%'
        }))
        
        # Portfolio composition visualization
        st.subheader("Portfolio Composition")
        fig = px.pie(portfolio_df, values='Current Value', names='Stock Name', 
                    title='Portfolio Allocation by Current Value')
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance visualization
        st.subheader("Individual Stock Performance")
        
        # Create a copy of the dataframe for visualization
        perf_df = portfolio_df.copy()
        
        # Create separate dataframes for positive and negative performance
        positive_df = perf_df[perf_df['Profit/Loss %'] >= 0]
        negative_df = perf_df[perf_df['Profit/Loss %'] < 0]
        
        # Create a figure with two traces
        fig = go.Figure()
        
        # Add positive performance bars with continuous green color scale
        if not positive_df.empty:
            # Sort by performance for better visualization
            positive_df = positive_df.sort_values('Profit/Loss %')
            
            # Get the maximum positive value for scaling
            max_positive = positive_df['Profit/Loss %'].max()
            
            # Create a list of colors from light green to dark green based on values
            green_colors = []
            for val in positive_df['Profit/Loss %']:
                # Calculate intensity (0 to 1) based on the value's position in the range
                if max_positive > 0:
                    intensity = val / max_positive
                else:
                    intensity = 0.5  # Default mid-intensity if all values are 0
                
                # Generate RGB color: interpolate from light green (144,238,144) to dark green (0,100,0)
                r = int(144 - intensity * 144)
                g = int(238 - intensity * 138)
                b = int(144 - intensity * 144)
                green_colors.append(f'rgb({r},{g},{b})')
            
            fig.add_trace(go.Bar(
                x=positive_df['Stock Name'],
                y=positive_df['Profit/Loss %'],
                marker=dict(
                    color=green_colors,
                    showscale=False
                ),
                name='Positive Performance'
            ))
        
        # Add negative performance bars with continuous red color scale
        if not negative_df.empty:
            # Sort by performance for better visualization
            negative_df = negative_df.sort_values('Profit/Loss %')
            
            # Get the minimum negative value for scaling
            min_negative = negative_df['Profit/Loss %'].min()
            
            # Create a list of colors from dark red to light red based on values
            red_colors = []
            for val in negative_df['Profit/Loss %']:
                # Calculate intensity (0 to 1) based on the value's position in the range
                if min_negative < 0:
                    intensity = val / min_negative  # Will be between 0 and 1
                else:
                    intensity = 0.5  # Default mid-intensity
                
                # Generate RGB color: interpolate from dark red (139,0,0) to light red (255,99,71)
                r = int(139 + (255-139) * (1-intensity))
                g = int(0 + 99 * (1-intensity))
                b = int(0 + 71 * (1-intensity))
                red_colors.append(f'rgb({r},{g},{b})')
            
            fig.add_trace(go.Bar(
                x=negative_df['Stock Name'],
                y=negative_df['Profit/Loss %'],
                marker=dict(
                    color=red_colors,
                    showscale=False
                ),
                name='Negative Performance'
            ))
        
        # Update layout
        fig.update_layout(
            title='Stock Performance (% Gain/Loss)',
            xaxis_title='Stock Name',
            yaxis_title='Profit/Loss %',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Performance vs S&P500
    elif page == "Performance vs S&P500":
        st.header("Portfolio Performance vs S&P500")
        
        # Get S&P500 data
        sp500_hist = get_sp500_data(period=period_str)
        
        if sp500_hist is not None:
            # Create a dataframe to track portfolio performance over time
            performance_df = pd.DataFrame()
            performance_df['Date'] = sp500_hist.index
            
            # Calculate S&P500 performance
            sp500_start = sp500_hist['Close'].iloc[0]
            performance_df['S&P500'] = (sp500_hist['Close'] / sp500_start - 1) * 100
            
            # Calculate portfolio performance
            performance_df['Portfolio'] = 0
            
            # Weight each stock by its initial investment proportion
            total_investment = portfolio_df['Investment Value'].sum()  # Use Investment Value which is already in EUR
            
            for i, row in portfolio_df.iterrows():
                ticker = row['Stock Ticker']
                weight = row['Investment Value'] / total_investment if total_investment > 0 else 0
                hist = get_stock_history(ticker, period=period_str)
                
                if hist is not None and not hist.empty:
                    # Get the currency for this stock
                    currency = row['Currency']
                    
                    # Reindex to match the S&P500 dates
                    hist = hist.reindex(sp500_hist.index, method='ffill')
                    
                    # Convert all historical prices to EUR
                    converted_prices = []
                    for price in hist['Close']:
                        converted_prices.append(convert_to_eur(price, currency))
                    
                    # Create a series with converted prices
                    converted_series = pd.Series(converted_prices, index=hist.index)
                    
                    # Calculate performance based on EUR prices
                    stock_start = converted_series.iloc[0] if not pd.isna(converted_series.iloc[0]) else 1.0
                    if stock_start > 0:  # Avoid division by zero
                        stock_perf = (converted_series / stock_start - 1) * 100
                        performance_df['Portfolio'] += stock_perf * weight
            
            # Plot the comparison
            st.subheader(f"Performance Comparison (Last {time_period} Months)")
            fig = px.line(performance_df, x='Date', y=['Portfolio', 'S&P500'],
                        title=f'Portfolio vs S&P500 Performance (%)',
                        labels={'value': 'Return (%)', 'variable': 'Index'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display metrics
            portfolio_return = performance_df['Portfolio'].iloc[-1] if not pd.isna(performance_df['Portfolio'].iloc[-1]) else 0.0
            sp500_return = performance_df['S&P500'].iloc[-1] if not pd.isna(performance_df['S&P500'].iloc[-1]) else 0.0
            outperformance = portfolio_return - sp500_return
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Portfolio Return", f"{portfolio_return:.2f}%")
            col2.metric("S&P500 Return", f"{sp500_return:.2f}%")
            col3.metric("Outperformance", f"{outperformance:.2f}%", 
                        f"{'+' if outperformance > 0 else ''}{outperformance:.2f}%")
    
    # Sector Diversification
    elif page == "Sector Diversification":
        st.header("Sector Diversification")
        
        # Group by sector
        sector_investment = portfolio_df.groupby('Sector', observed=True).agg({
            'Stock Purchase Price': lambda x: (portfolio_df.loc[x.index, 'Stock Number'] * x).sum(),
            'Current Value': 'sum'
        }).reset_index()
        
        sector_investment.columns = ['Sector', 'Total Investment', 'Current Value']
        sector_investment['Profit/Loss'] = sector_investment['Current Value'] - sector_investment['Total Investment']
        sector_investment['Return %'] = (sector_investment['Current Value'] / sector_investment['Total Investment'] - 1) * 100
        
        # Calculate value-weighted performance for each sector
        # First, get individual stock returns and weights within each sector
        sector_weighted_returns = {}
        for sector in portfolio_df['Sector'].unique():
            sector_stocks = portfolio_df[portfolio_df['Sector'] == sector]
            total_sector_value = sector_stocks['Current Value'].sum()
            
            # Calculate weighted return for each stock in the sector
            weighted_return = 0
            for _, stock in sector_stocks.iterrows():
                weight = stock['Current Value'] / total_sector_value if total_sector_value > 0 else 0
                weighted_return += stock['Profit/Loss %'] * weight
            
            sector_weighted_returns[sector] = weighted_return
        
        # Add weighted returns to sector_investment dataframe
        sector_investment['Weighted Return %'] = sector_investment['Sector'].map(sector_weighted_returns)
        
        # Display sector table
        st.subheader("Sector Allocation")
        st.dataframe(sector_investment.style.format({
            'Total Investment': '€{:.2f}',
            'Current Value': '€{:.2f}',
            'Profit/Loss': '€{:.2f}',
            'Return %': '{:.2f}%'
        }))
        
        # Sector allocation visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Investment by Sector")
            fig = px.pie(sector_investment, values='Total Investment', names='Sector',
                        title='Portfolio Investment Allocation by Sector')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Current Value by Sector")
            fig = px.pie(sector_investment, values='Current Value', names='Sector',
                        title='Portfolio Current Value Allocation by Sector')
            st.plotly_chart(fig, use_container_width=True)
        
        # Sector performance
        st.subheader("Sector Performance (Value-Weighted)")
        
        # Create separate dataframes for positive and negative performance
        positive_sectors = sector_investment[sector_investment['Weighted Return %'] >= 0]
        negative_sectors = sector_investment[sector_investment['Weighted Return %'] < 0]
        
        # Create a figure with two traces
        fig = go.Figure()
        
        # Add positive performance bars with continuous green color scale
        if not positive_sectors.empty:
            # Sort by performance for better visualization
            positive_sectors = positive_sectors.sort_values('Weighted Return %')
            
            # Get the maximum positive value for scaling
            max_positive = positive_sectors['Weighted Return %'].max()
            
            # Create a list of colors from light green to dark green based on values
            green_colors = []
            for val in positive_sectors['Weighted Return %']:
                # Calculate intensity (0 to 1) based on the value's position in the range
                if max_positive > 0:
                    intensity = val / max_positive
                else:
                    intensity = 0.5  # Default mid-intensity if all values are 0
                
                # Generate RGB color: interpolate from light green (144,238,144) to dark green (0,100,0)
                r = int(144 - intensity * 144)
                g = int(238 - intensity * 138)
                b = int(144 - intensity * 144)
                green_colors.append(f'rgb({r},{g},{b})')
            
            fig.add_trace(go.Bar(
                x=positive_sectors['Sector'],
                y=positive_sectors['Weighted Return %'],
                marker=dict(
                    color=green_colors,
                    showscale=False
                ),
                name='Positive Performance'
            ))
        
        # Add negative performance bars with continuous red color scale
        if not negative_sectors.empty:
            # Sort by performance for better visualization
            negative_sectors = negative_sectors.sort_values('Weighted Return %')
            
            # Get the minimum negative value for scaling
            min_negative = negative_sectors['Weighted Return %'].min()
            
            # Create a list of colors from dark red to light red based on values
            red_colors = []
            for val in negative_sectors['Weighted Return %']:
                # Calculate intensity (0 to 1) based on the value's position in the range
                if min_negative < 0:
                    intensity = val / min_negative  # Will be between 0 and 1
                else:
                    intensity = 0.5  # Default mid-intensity
                
                # Generate RGB color: interpolate from dark red (139,0,0) to light red (255,99,71)
                r = int(139 + (255-139) * (1-intensity))
                g = int(0 + 99 * (1-intensity))
                b = int(0 + 71 * (1-intensity))
                red_colors.append(f'rgb({r},{g},{b})')
            
            fig.add_trace(go.Bar(
                x=negative_sectors['Sector'],
                y=negative_sectors['Weighted Return %'],
                marker=dict(
                    color=red_colors,
                    showscale=False
                ),
                name='Negative Performance'
            ))
        
        # Update layout
        fig.update_layout(
            title='Sector Performance (Value-Weighted % Return)',
            xaxis_title='Sector',
            yaxis_title='Weighted Return %',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Metrics Comparison
    elif page == "Metrics Comparison":
        st.header("Stock Metrics Comparison")
        
        # Extract metrics for each stock
        for i, row in portfolio_df.iterrows():
            ticker = row['Stock Ticker']
            stock_info = get_stock_info(ticker)
            
            if stock_info is not None:
                try:
                    info = stock_info.info
                    for metric in metrics_to_extract:
                        if metric == 'currentPrice':
                            metrics_df.at[i, metric] = portfolio_df.at[i, 'Current Price']
                        elif metric == 'dividendGrowth':
                            # Calculate dividend growth if possible
                            if 'dividendRate' in info and info.get('dividendRate', 0) > 0:
                                try:
                                    div_history = stock_info.dividends
                                    if len(div_history) > 1:
                                        annual_div = div_history.groupby(div_history.index.year).sum()
                                        if len(annual_div) >= 2:
                                            growth = (annual_div.iloc[-1] / annual_div.iloc[-2] - 1) * 100
                                            metrics_df.at[i, metric] = growth
                                except:
                                    pass
                        else:
                            metrics_df.at[i, metric] = info.get(metric, np.nan)
                except:
                    pass
        
        # Convert dividend yield to percentage
        if 'dividendYield' in metrics_df.columns:
            metrics_df['dividendYield'] = metrics_df['dividendYield'] * 100
        
        # Display metrics table
        st.subheader("Stock Metrics")
        display_metrics = metrics_df.copy()
        
        # Rename columns for better readability
        column_rename = {
            'debtToEquity': 'Debt to Equity',
            'revenuePerShare': 'Revenue per Share',
            'trailingPE': 'P/E Ratio',
            'pegRatio': 'PEG Ratio',
            'trailingPegRatio': 'Trailing PEG',
            'forwardPE': 'Forward P/E',
            'priceToBook': 'Price to Book',
            'earningsPerShare': 'EPS',
            'dividendYield': 'Dividend Yield (%)',
            'dividendRate': 'Dividend Rate',
            'dividendGrowth': 'Dividend Growth (%)',
            'targetLowPrice': 'Target Low',
            'targetMedianPrice': 'Target Median',
            'targetHighPrice': 'Target High',
            'currentPrice': 'Current Price'
        }
        
        display_metrics = display_metrics.rename(columns=column_rename)
        st.dataframe(display_metrics)
        
        # Visualization of selected metrics
        st.subheader("Metrics Visualization")
        
        # Select metrics to visualize
        metrics_to_viz = st.multiselect(
            "Select metrics to visualize",
            options=list(column_rename.values()),
            default=['P/E Ratio', 'Dividend Yield (%)', 'EPS']
        )
        
        if metrics_to_viz:
            # Create a long-format dataframe for visualization
            viz_df = pd.melt(
                display_metrics,
                id_vars=['Stock Name'],
                value_vars=metrics_to_viz,
                var_name='Metric',
                value_name='Value'
            )
            
            # Create a grouped bar chart
            fig = px.bar(
                viz_df,
                x='Stock Name',
                y='Value',
                color='Metric',
                barmode='group',
                title='Selected Metrics Comparison'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Price targets vs current price
        st.subheader("Price Targets vs Current Price")
        price_targets = display_metrics[['Stock Name', 'Target Low', 'Target Median', 'Target High', 'Current Price']]
        
        # Create a dataframe for visualization
        price_viz_df = pd.melt(
            price_targets,
            id_vars=['Stock Name'],
            value_vars=['Target Low', 'Target Median', 'Target High', 'Current Price'],
            var_name='Price Type',
            value_name='Price'
        )
        
        # Create a grouped bar chart
        fig = px.bar(
            price_viz_df,
            x='Stock Name',
            y='Price',
            color='Price Type',
            barmode='group',
            title='Price Targets vs Current Price'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Stock Comparison
    elif page == "Stock Comparison":
        st.header("Stock Comparison")
        
        # Select stocks to compare
        stock_options = portfolio_df['Stock Name'].tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            stock1 = st.selectbox("Select first stock", stock_options, index=0)
        with col2:
            stock2 = st.selectbox("Select second stock", stock_options, index=min(1, len(stock_options)-1))
        
        # Get ticker symbols
        ticker1 = portfolio_df[portfolio_df['Stock Name'] == stock1]['Stock Ticker'].iloc[0]
        ticker2 = portfolio_df[portfolio_df['Stock Name'] == stock2]['Stock Ticker'].iloc[0]
        
        # Get historical data
        hist1 = get_stock_history(ticker1, period=period_str)
        hist2 = get_stock_history(ticker2, period=period_str)
        
        if hist1 is not None and hist2 is not None and not hist1.empty and not hist2.empty:
            # Normalize prices to compare percentage change
            start_price1 = hist1['Close'].iloc[0]
            start_price2 = hist2['Close'].iloc[0]
            
            hist1['Normalized'] = (hist1['Close'] / start_price1 - 1) * 100
            hist2['Normalized'] = (hist2['Close'] / start_price2 - 1) * 100
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame()
            comparison_df['Date'] = hist1.index
            comparison_df[stock1] = hist1['Normalized']
            comparison_df[stock2] = hist2['Normalized']
            
            # Plot price comparison
            st.subheader("Price Performance Comparison")
            fig = px.line(
                comparison_df,
                x='Date',
                y=[stock1, stock2],
                title=f'Price Performance Comparison (% Change)',
                labels={'value': 'Return (%)', 'variable': 'Stock'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display key metrics comparison
            st.subheader("Key Metrics Comparison")
            
            # Create comparison table
            metrics_comparison = pd.DataFrame({
                'Metric': ['Current Price', 'Purchase Price', 'Profit/Loss %', 'P/E Ratio', 'EPS', 'Dividend Yield'],
                stock1: [
                    f"€{portfolio_df[portfolio_df['Stock Name'] == stock1][['Current Price']].iloc[0, 0]:.2f}",
                    f"€{portfolio_df[portfolio_df['Stock Name'] == stock1][['Stock Purchase Price']].iloc[0, 0]:.2f}",
                    f"{portfolio_df[portfolio_df['Stock Name'] == stock1][['Profit/Loss %']].iloc[0, 0]:.2f}%",
                    f"{metrics_df[metrics_df['Stock Name'] == stock1][['trailingPE']].iloc[0, 0]:.2f}" if not pd.isna(metrics_df[metrics_df['Stock Name'] == stock1][['trailingPE']].iloc[0, 0]) else 'N/A',
                    f"€{metrics_df[metrics_df['Stock Name'] == stock1][['earningsPerShare']].iloc[0, 0]:.2f}" if not pd.isna(metrics_df[metrics_df['Stock Name'] == stock1][['earningsPerShare']].iloc[0, 0]) else 'N/A',
                    f"{metrics_df[metrics_df['Stock Name'] == stock1][['dividendYield']].iloc[0, 0]:.2f}%" if not pd.isna(metrics_df[metrics_df['Stock Name'] == stock1][['dividendYield']].iloc[0, 0]) else 'N/A'
                ],
                stock2: [
                    f"€{portfolio_df[portfolio_df['Stock Name'] == stock2][['Current Price']].iloc[0, 0]:.2f}",
                    f"€{portfolio_df[portfolio_df['Stock Name'] == stock2][['Stock Purchase Price']].iloc[0, 0]:.2f}",
                    f"{portfolio_df[portfolio_df['Stock Name'] == stock2][['Profit/Loss %']].iloc[0, 0]:.2f}%",
                    f"{metrics_df[metrics_df['Stock Name'] == stock2][['trailingPE']].iloc[0, 0]:.2f}" if not pd.isna(metrics_df[metrics_df['Stock Name'] == stock2][['trailingPE']].iloc[0, 0]) else 'N/A',
                    f"€{metrics_df[metrics_df['Stock Name'] == stock2][['earningsPerShare']].iloc[0, 0]:.2f}" if not pd.isna(metrics_df[metrics_df['Stock Name'] == stock2][['earningsPerShare']].iloc[0, 0]) else 'N/A',
                    f"{metrics_df[metrics_df['Stock Name'] == stock2][['dividendYield']].iloc[0, 0]:.2f}%" if not pd.isna(metrics_df[metrics_df['Stock Name'] == stock2][['dividendYield']].iloc[0, 0]) else 'N/A'
                ]
            })
            
            st.table(metrics_comparison)
            
            # Volume comparison
            st.subheader("Trading Volume Comparison")
            volume_df = pd.DataFrame()
            volume_df['Date'] = hist1.index
            volume_df[f"{stock1} Volume"] = hist1['Volume']
            volume_df[f"{stock2} Volume"] = hist2['Volume']
            
            fig = px.line(
                volume_df,
                x='Date',
                y=[f"{stock1} Volume", f"{stock2} Volume"],
                title='Trading Volume Comparison',
                labels={'value': 'Volume', 'variable': 'Stock'}
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Failed to load portfolio data. Please check that the port.xlsx file exists and has the required columns.")
