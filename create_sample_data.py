# create_sample_data_fixed.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data():
    """Generate complete sample data for NSE testing"""
    
    print("Generating comprehensive NSE sample data...")
    
    # Base date for sample data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # All NSE symbols with realistic data
    nse_stocks = {
        # Major blue chips with high liquidity
        'SCOM': {
            'name': 'Safaricom Plc',
            'sector': 'TELECOMMUNICATION',
            'issued_shares': 40065428000,
            'base_price': 15.0,
            'volatility': 0.02,
            'avg_volume': 5000000
        },
        'KCB': {
            'name': 'KCB Group Plc',
            'sector': 'BANKING',
            'issued_shares': 3213462815,
            'base_price': 25.0,
            'volatility': 0.03,
            'avg_volume': 3000000
        },
        'EQTY': {
            'name': 'Equity Group Holdings Plc',
            'sector': 'BANKING',
            'issued_shares': 3773674802,
            'base_price': 40.0,
            'volatility': 0.025,
            'avg_volume': 2500000
        },
        'EABL': {
            'name': 'East African Breweries Ltd',
            'sector': 'MANUFACTURING',
            'issued_shares': 790774356,
            'base_price': 130.0,
            'volatility': 0.015,
            'avg_volume': 1000000
        },
        'COOP': {
            'name': 'Co-operative Bank of Kenya Ltd',
            'sector': 'BANKING',
            'issued_shares': 5867174695,
            'base_price': 12.5,
            'volatility': 0.02,
            'avg_volume': 2000000
        }
    }
    
    # 1. GENERATE PRICE DATA
    print("Generating price data...")
    price_data = []
    trading_days = pd.bdate_range(start=start_date, end=end_date)
    
    for symbol, info in nse_stocks.items():
        current_price = info['base_price']
        
        for date in trading_days:
            # Generate realistic price movement
            daily_change = np.random.normal(0, info['volatility'])
            current_price *= (1 + daily_change)
            
            # Ensure price doesn't go below 0.01
            current_price = max(current_price, 0.01)
            
            # Generate OHLC prices
            open_price = current_price
            high = open_price * (1 + abs(np.random.normal(0, info['volatility']/2)))
            low = open_price * (1 - abs(np.random.normal(0, info['volatility']/2)))
            close = open_price * (1 + np.random.normal(0, info['volatility']/3))
            
            # Ensure high >= low
            high = max(open_price, close, high)
            low = min(open_price, close, low)
            
            # Generate volume with some randomness
            volume = int(info['avg_volume'] * np.random.uniform(0.7, 1.3))
            
            price_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Symbol': symbol,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume
            })
            
            # Update current price for next day
            current_price = close
    
    price_df = pd.DataFrame(price_data)
    print(f"Price data: {len(price_df)} records generated")
    
    # 2. GENERATE FINANCIAL DATA
    print("Generating financial data...")
    financial_data = []
    years = [2020, 2021, 2022, 2023]
    
    # Sector growth rates (simulated)
    sector_growth = {
        'BANKING': 0.08,
        'TELECOMMUNICATION': 0.12,
        'MANUFACTURING': 0.06
    }
    
    for symbol, info in nse_stocks.items():
        sector = info['sector']
        base_revenue = info['base_price'] * info['issued_shares'] * np.random.uniform(0.1, 0.3)
        
        for year_idx, year in enumerate(years):
            # Progressive growth each year
            growth_factor = (1 + sector_growth.get(sector, 0.06)) ** year_idx
            
            # Generate realistic financials
            revenue = base_revenue * growth_factor * np.random.uniform(0.9, 1.1)
            
            # Different profit margins by sector
            if sector == 'BANKING':
                net_margin = np.random.uniform(0.18, 0.25)
            elif sector == 'TELECOMMUNICATION':
                net_margin = np.random.uniform(0.20, 0.28)
            elif sector == 'MANUFACTURING':
                net_margin = np.random.uniform(0.12, 0.18)
            else:
                net_margin = np.random.uniform(0.08, 0.15)
            
            net_income = revenue * net_margin
            total_assets = revenue * np.random.uniform(2.0, 3.0)
            
            # Different debt levels by sector
            if sector == 'BANKING':
                debt_ratio = np.random.uniform(0.75, 0.85)
            elif sector == 'INSURANCE':
                debt_ratio = np.random.uniform(0.65, 0.75)
            else:
                debt_ratio = np.random.uniform(0.50, 0.70)
            
            total_liabilities = total_assets * debt_ratio
            shareholders_equity = total_assets - total_liabilities
            
            # EPS and Dividends
            eps = net_income / info['issued_shares']
            
            # Different payout ratios
            if sector == 'BANKING':
                payout_ratio = np.random.uniform(0.40, 0.60)
            elif sector == 'TELECOMMUNICATION':
                payout_ratio = np.random.uniform(0.50, 0.70)
            else:
                payout_ratio = np.random.uniform(0.30, 0.50)
            
            dividends = eps * payout_ratio
            
            financial_data.append({
                'Symbol': symbol,
                'Year': year,
                'Revenue': round(revenue, 2),
                'Net_Income': round(net_income, 2),
                'Total_Assets': round(total_assets, 2),
                'Total_Liabilities': round(total_liabilities, 2),
                'Shareholders_Equity': round(shareholders_equity, 2),
                'EPS': round(eps, 4),
                'Dividends': round(dividends, 4)
            })
    
    financial_df = pd.DataFrame(financial_data)
    print(f"Financial data: {len(financial_df)} records generated")
    
    # 3. GENERATE FUNDAMENTALS DATA
    print("Generating fundamentals data...")
    fundamentals_data = []
    
    # Get latest prices
    latest_prices = {}
    for symbol in nse_stocks.keys():
        symbol_data = price_df[price_df['Symbol'] == symbol]
        if not symbol_data.empty:
            latest_date = symbol_data['Date'].max()
            latest_price = symbol_data[symbol_data['Date'] == latest_date]['Close'].iloc[0]
            latest_prices[symbol] = latest_price
    
    for symbol, info in nse_stocks.items():
        # Get latest financials
        latest_fin = financial_df[(financial_df['Symbol'] == symbol) & 
                                 (financial_df['Year'] == 2023)]
        
        if latest_fin.empty:
            continue
        
        latest_fin = latest_fin.iloc[0]
        current_price = latest_prices.get(symbol, info['base_price'])
        
        # Calculate market cap
        market_cap = current_price * info['issued_shares']
        
        # Calculate ratios
        eps = latest_fin['EPS']
        book_value_per_share = latest_fin['Shareholders_Equity'] / info['issued_shares']
        dividend = latest_fin['Dividends']
        
        pe_ratio = current_price / eps if eps > 0 else 0
        pb_ratio = current_price / book_value_per_share if book_value_per_share > 0 else 0
        dividend_yield = (dividend / current_price * 100) if current_price > 0 else 0
        
        # Determine if stock is over/undervalued based on sector
        sector = info['sector']
        sector_avg_pe = {
            'BANKING': 8.0,
            'TELECOMMUNICATION': 14.0,
            'MANUFACTURING': 12.0
        }
        
        avg_pe = sector_avg_pe.get(sector, 10.0)
        if pe_ratio < avg_pe * 0.8:
            valuation_status = "UNDERVALUED"
        elif pe_ratio > avg_pe * 1.2:
            valuation_status = "OVERVALUED"
        else:
            valuation_status = "FAIRLY VALUED"
        
        fundamentals_data.append({
            'Symbol': symbol,
            'Name': info['name'],
            'Sector': sector,
            'Market_Cap': round(market_cap, 2),
            'Issued_Shares': info['issued_shares'],
            'Current_Price': round(current_price, 2),
            'PE_Ratio': round(pe_ratio, 2),
            'PB_Ratio': round(pb_ratio, 2),
            'Dividend_Yield': round(dividend_yield, 2),
            'Valuation_Status': valuation_status,
            'Recommendation': 'HOLD'  # Will be calculated by app
        })
    
    fundamentals_df = pd.DataFrame(fundamentals_data)
    print(f"Fundamentals data: {len(fundamentals_df)} records generated")
    
    # 4. SAVE ALL DATA FILES
    print("\nSaving data files...")
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Save CSV files
    price_df.to_csv('data/nse_price_data.csv', index=False)
    financial_df.to_csv('data/nse_financial_data.csv', index=False)
    fundamentals_df.to_csv('data/nse_fundamentals.csv', index=False)
    
    # Save Excel files with formatting
    with pd.ExcelWriter('templates/NSE_Sample_Data_Complete.xlsx', engine='openpyxl') as writer:
        # Price data
        price_df.to_excel(writer, sheet_name='PRICE_DATA', index=False)
        
        # Financial data
        financial_df.to_excel(writer, sheet_name='FINANCIAL_DATA', index=False)
        
        # Fundamentals data
        fundamentals_df.to_excel(writer, sheet_name='FUNDAMENTALS', index=False)
        
        # Create summary sheet
        summary_data = []
        for _, row in fundamentals_df.iterrows():
            summary_data.append({
                'Symbol': row['Symbol'],
                'Company': row['Name'],
                'Sector': row['Sector'],
                'Current Price (KES)': row['Current_Price'],
                'Market Cap (KES)': f"{row['Market_Cap']:,.0f}",
                'P/E Ratio': row['PE_Ratio'],
                'Dividend Yield %': row['Dividend_Yield'],
                'Valuation': row['Valuation_Status']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='SUMMARY', index=False)
        
        # Add instructions sheet
        instructions = pd.DataFrame({
            'Section': ['PRICE_DATA', 'FINANCIAL_DATA', 'FUNDAMENTALS', 'USAGE'],
            'Description': [
                'Daily historical price data for NSE stocks',
                'Annual financial statements for companies',
                'Current company fundamentals and ratios',
                'This is sample data for testing the NSE Stock Analyzer app'
            ],
            'Records': [
                f'{len(price_df):,} price records',
                f'{len(financial_df):,} financial records',
                f'{len(fundamentals_df):,} company records',
                'Generated for testing purposes'
            ],
            'Coverage': [
                f'{len(nse_stocks)} stocks, 1 year of daily data',
                '4 years (2020-2023) of financial data',
                'Current fundamentals with valuations',
                'All major NSE sectors represented'
            ]
        })
        instructions.to_excel(writer, sheet_name='INSTRUCTIONS', index=False)
    
    # 5. CREATE README FILE - Fixed version without nested f-string issues
    print("Creating README file...")
    
    readme_content = f"""# NSE SAMPLE DATA FOR TESTING

This directory contains comprehensive sample data for testing the NSE Stock Analysis application.

## 📁 FILES GENERATED

### 1. CSV Files (in 'data/' directory):
   - `nse_price_data.csv` - Daily price data for {len(nse_stocks)} NSE stocks
   - `nse_financial_data.csv` - Financial statements (2020-2023)
   - `nse_fundamentals.csv` - Current company fundamentals

### 2. Excel File (in 'templates/' directory):
   - `NSE_Sample_Data_Complete.xlsx` - All data in one Excel file with multiple sheets

## 📊 DATA SPECIFICATIONS

### Price Data:
- **Period**: 1 year of daily data (approx. {len(trading_days)} trading days)
- **Stocks**: {len(nse_stocks)} major NSE companies
- **Fields**: Date, Symbol, Open, High, Low, Close, Volume
- **Total Records**: {len(price_df):,}

### Financial Data:
- **Years**: 2020, 2021, 2022, 2023
- **Metrics**: Revenue, Net Income, Assets, Liabilities, Equity, EPS, Dividends
- **Total Records**: {len(financial_df):,}

### Fundamentals Data:
- **Companies**: {len(fundamentals_df)} with complete data
- **Ratios**: P/E, P/B, Dividend Yield, Market Cap
- **Sectors**: All major NSE sectors included

## 🏢 COMPANIES INCLUDED

### Major NSE Companies:
- Safaricom (SCOM) - Telecommunications leader
- KCB Group (KCB) - Largest bank by assets
- Equity Bank (EQTY) - Leading regional bank
- Co-operative Bank (COOP) - Major retail bank
- East African Breweries (EABL) - Brewing giant

## 💡 HOW TO USE

### Option 1: Use CSV files directly
1. Place CSV files in the `data/` directory
2. Run the Streamlit app: `streamlit run app.py`
3. Click "Use Sample Data from data/ directory" in the sidebar

### Option 2: Use Excel template
1. Open `NSE_Sample_Data_Complete.xlsx`
2. Review the data in different sheets
3. Use as reference for your own data collection

### Option 3: Upload via app interface
1. Run the Streamlit app
2. Use the file uploaders in the sidebar
3. Select the CSV files from the `data/` directory

## 📈 SAMPLE ANALYSIS RESULTS EXPECTED

Based on the generated data, you should see:
- **Different valuations** based on P/E ratios
- **Various recommendations** (Buy/Hold/Sell)
- **Technical indicators** calculated from price data
- **Sector analysis** showing performance by industry

## ⚠️ IMPORTANT NOTES

1. **This is sample data** - Not real market data
2. **Prices are simulated** - For testing purposes only
3. **Financials are realistic** but not actual company data
4. **Use for app testing** - Not for investment decisions

## 🔄 REGENERATING DATA

To regenerate with different random values:
```bash
python create_sample_data_fixed.py"""
