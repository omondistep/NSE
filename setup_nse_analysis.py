# setup_nse_analysis.py
import os
import pandas as pd
import numpy as np

def create_minimal_templates():
    """Create minimal blank templates for user to fill"""
    
    # 1. PRICE DATA TEMPLATE
    price_columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Add example rows for clarity
    price_examples = [
        ['2024-01-02', 'SCOM', 15.20, 15.50, 15.10, 15.45, 5000000],
        ['2024-01-02', 'KCB', 25.10, 25.80, 25.00, 25.50, 3000000],
        ['2024-01-03', 'SCOM', 15.50, 15.70, 15.40, 15.60, 4500000],
        ['2024-01-03', 'KCB', 25.60, 25.90, 25.30, 25.70, 2800000],
    ]
    
    price_df = pd.DataFrame(price_examples, columns=price_columns)
    
    # 2. FINANCIAL DATA TEMPLATE
    financial_columns = [
        'Symbol', 'Year', 'Revenue', 'Net_Income', 'Total_Assets',
        'Total_Liabilities', 'Shareholders_Equity', 'EPS', 'Dividends'
    ]
    
    financial_examples = [
        ['SCOM', 2023, 300500000000, 62300000000, 450200000000, 
         180400000000, 269800000000, 1.55, 0.92],
        ['SCOM', 2022, 280300000000, 58000000000, 420000000000,
         165000000000, 255000000000, 1.45, 0.85],
        ['KCB', 2023, 120300000000, 18500000000, 1200500000000,
         950200000000, 250300000000, 5.76, 3.20],
    ]
    
    financial_df = pd.DataFrame(financial_examples, columns=financial_columns)
    
    # 3. FUNDAMENTALS TEMPLATE
    fundamentals_columns = [
        'Symbol', 'Name', 'Sector', 'Market_Cap', 'Issued_Shares',
        'PE_Ratio', 'PB_Ratio', 'Dividend_Yield'
    ]
    
    fundamentals_examples = [
        ['SCOM', 'Safaricom Plc', 'Telecommunication', 600000000000,
         40065428000, 14.5, 3.2, 6.0],
        ['KCB', 'KCB Group Plc', 'Banking', 85000000000,
         3213462815, 6.8, 1.1, 9.5],
        ['EQTY', 'Equity Group Holdings Plc', 'Banking', 120000000000,
         3773674802, 8.2, 1.5, 7.8],
    ]
    
    fundamentals_df = pd.DataFrame(fundamentals_examples, columns=fundamentals_columns)
    
    # 4. CREATE ALL NSE SYMBOLS REFERENCE FILE
    nse_symbols = [
        ['SCOM', 'Safaricom Plc', 'Telecommunication', 40065428000],
        ['KCB', 'KCB Group Plc', 'Banking', 3213462815],
        ['EQTY', 'Equity Group Holdings Plc', 'Banking', 3773674802],
        ['EABL', 'East African Breweries Ltd', 'Manufacturing', 790774356],
        ['COOP', 'Co-operative Bank of Kenya Ltd', 'Banking', 5867174695],
        ['ABSA', 'Absa Bank Kenya Plc', 'Banking', 5431536000],
        ['NCBA', 'NCBA Group Plc', 'Banking', 1647519532],
        ['DTK', 'Diamond Trust Bank Kenya Ltd', 'Banking', 279602220],
        ['BAT', 'British American Tobacco Kenya', 'Manufacturing', 100000000],
        ['SCBK', 'Standard Chartered Bank Kenya Ltd', 'Banking', 377861629],
        ['JUB', 'Jubilee Holdings Ltd', 'Insurance', 72472950],
        ['BRIT', 'Britam Holdings Plc', 'Insurance', 2523486816],
        ['CTUM', 'Centum Investment Co Plc', 'Investment', 665441714],
        ['BAMB', 'Bamburi Cement Ltd', 'Construction', 362959275],
        ['CARB', 'Carbacid Investments Ltd', 'Manufacturing', 254851985],
        ['UNGA', 'Unga Group Ltd', 'Manufacturing', 75708873],
        ['SASN', 'Sasini Plc', 'Agricultural', 228055500],
        ['EGAD', 'Eaagads Ltd', 'Agricultural', 32157000],
        ['KUKZ', 'Kakuzi Plc', 'Agricultural', 19599999],
    ]
    
    nse_ref_df = pd.DataFrame(nse_symbols, columns=['Symbol', 'Name', 'Sector', 'Issued_Shares'])
    
    # 5. SAVE ALL FILES
    # Save to templates directory
    template_path = 'templates/NSE_Analysis_Templates.xlsx'
    with pd.ExcelWriter(template_path) as writer:
        price_df.to_excel(writer, sheet_name='PRICE_DATA', index=False)
        financial_df.to_excel(writer, sheet_name='FINANCIAL_DATA', index=False)
        fundamentals_df.to_excel(writer, sheet_name='FUNDAMENTALS', index=False)
        nse_ref_df.to_excel(writer, sheet_name='ALL_NSE_SYMBOLS', index=False)
        
        # Add instruction sheet
        instructions = pd.DataFrame({
            'Step': [1, 2, 3, 4, 5],
            'Action': [
                'Fill PRICE_DATA sheet with historical price data',
                'Fill FINANCIAL_DATA sheet with company financials',
                'Update FUNDAMENTALS sheet with current ratios',
                'Use ALL_NSE_SYMBOLS as reference for correct symbols',
                'Save this file and use it in the main app'
            ],
            'Format': [
                'Date: YYYY-MM-DD, Prices: KES, Volume: integer',
                'All amounts in KES, EPS: per share, Dividends: per share',
                'Market Cap in KES, Yield in percentage',
                'Copy symbols exactly as shown',
                'Keep column names unchanged'
            ]
        })
        instructions.to_excel(writer, sheet_name='INSTRUCTIONS', index=False)
    
    print("✅ Template file created: templates/NSE_Analysis_Templates.xlsx")
    print("\n📋 This file contains:")
    print("   1. PRICE_DATA - For daily stock prices")
    print("   2. FINANCIAL_DATA - For annual financial statements")
    print("   3. FUNDAMENTALS - For current company metrics")
    print("   4. ALL_NSE_SYMBOLS - Complete NSE stock list")
    print("   5. INSTRUCTIONS - How to fill the templates")
    
    # Also create CSV versions in data directory
    price_df.to_csv('data/nse_price_data.csv', index=False)
    financial_df.to_csv('data/nse_financial_data.csv', index=False)
    fundamentals_df.to_csv('data/nse_fundamentals.csv', index=False)
    
    print("\n📁 CSV files also created in data/ directory:")
    print("   - data/nse_price_data.csv")
    print("   - data/nse_financial_data.csv")
    print("   - data/nse_fundamentals.csv")
    
    return True

def setup_project():
    """Setup complete project structure for NSE analysis"""
    
    print("Setting up NSE Stock Analysis Project...")
    print("="*60)
    
    # Create project directory structure
    directories = ['data', 'reports', 'templates', 'backups']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}/")
    
    # Create the main templates
    create_minimal_templates()
    
    # Create sample data structure
    print("\n" + "="*60)
    print("PROJECT STRUCTURE CREATED:")
    print("="*60)
    
    project_structure = """
    nse_stock_analyzer/
    │
    ├── app.py                    # Main Streamlit application
    ├── requirements.txt          # Python dependencies
    ├── setup_nse_analysis.py     # This setup script
    │
    ├── data/                     # Your data goes here
    │   ├── nse_price_data.csv
    │   ├── nse_financial_data.csv
    │   └── nse_fundamentals.csv
    │
    ├── templates/                # Template files
    │   └── NSE_Analysis_Templates.xlsx
    │
    ├── reports/                  # Generated reports
    │   └── (reports will be saved here)
    │
    └── backups/                  # Data backups
    """
    
    print(project_structure)
    
    # Create requirements.txt if it doesn't exist
    if not os.path.exists('requirements.txt'):
        with open('requirements.txt', 'w') as f:
            f.write("""pandas>=1.5.0
numpy>=1.24.0
openpyxl>=3.0.0
plotly>=5.13.0
streamlit>=1.22.0
python-dateutil>=2.8.2
xlrd>=2.0.0
""")
        print("\n📦 requirements.txt created")
    
    # Create a basic app.py if it doesn't exist
    if not os.path.exists('app.py'):
        with open('app.py', 'w') as f:
            f.write('''import streamlit as st

st.set_page_config(page_title="NSE Stock Analyzer", layout="wide")

st.title("📈 Nairobi Securities Exchange Stock Analysis")
st.markdown("---")

st.info("🚀 Welcome to NSE Stock Analysis App!")
st.write("""
### How to use this app:

1. **Prepare your data** using the templates in the `templates/` directory
2. **Place your data files** in the `data/` directory:
   - `nse_price_data.csv` - Daily price data
   - `nse_financial_data.csv` - Financial statements
   - `nse_fundamentals.csv` - Company fundamentals
3. **Run the main analysis app** (coming soon!)

### Features:
- Stock valuation analysis
- Buy/Sell/Hold recommendations
- Technical indicator analysis
- Excel report generation
""')

st.warning("⚠️ Please run the main valuation script after setting up your data.")
''')
        print("📄 app.py created with basic interface")
    
    print("\n✅ Setup complete!")
    print("\n📝 NEXT STEPS:")
    print("1. Fill the template file: templates/NSE_Analysis_Templates.xlsx")
    print("2. Copy your data to: data/nse_price_data.csv, etc.")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run the app: streamlit run app.py")
    print("5. Upload your filled Excel/CSV files when prompted")

def create_data_collection_guide():
    """Create a guide for collecting NSE data"""
    
    guide = """
    =========================================================================
                    HOW TO COLLECT NSE STOCK DATA
    =========================================================================
    
    SOURCES FOR NSE DATA:
    
    1. NSE WEBSITE (https://www.nse.co.ke/)
       - Price data: Market > Equity Market > Price List
       - Financials: Listed Companies > Company Filings
    
    2. CMA PORTAL (https://portal.cma.or.ke/)
       - Company annual reports
       - Financial statements
    
    3. INVESTMENT BANKS & BROKERAGES:
       - Dyer & Blair
       - Genghis Capital
       - Sterling Capital
       - ABC Bank (Investment arm)
    
    4. FINANCIAL DATA PROVIDERS:
       - Reuters Eikon
       - Bloomberg Terminal (if available)
       - Refinitiv
    
    5. FREE/PAID SERVICES:
       - Investing.com Kenya
       - TradingView (NSE data)
       - Yahoo Finance (some NSE stocks)
    
    =========================================================================
                    DATA COLLECTION TEMPLATE
    =========================================================================
    
    A. DAILY PRICE DATA (Minimum 1 year recommended):
       -------------------------------------------------
       Date, Symbol, Open, High, Low, Close, Volume
       2024-01-02, SCOM, 15.20, 15.50, 15.10, 15.45, 5000000
       2024-01-02, KCB, 25.10, 25.80, 25.00, 25.50, 3000000
       ...
    
    B. FINANCIAL DATA (Minimum 3 years):
       -------------------------------------------------
       Symbol, Year, Revenue, Net_Income, Total_Assets, 
       Total_Liabilities, Shareholders_Equity, EPS, Dividends
       SCOM, 2023, 300500000000, 62300000000, 450200000000,
       180400000000, 269800000000, 1.55, 0.92
       ...
    
    C. FUNDAMENTALS (Current data):
       -------------------------------------------------
       Symbol, Name, Sector, Market_Cap, Issued_Shares,
       PE_Ratio, PB_Ratio, Dividend_Yield
       SCOM, Safaricom Plc, Telecommunication, 600000000000,
       40065428000, 14.5, 3.2, 6.0
       ...
    
    =========================================================================
                        TIPS FOR DATA ENTRY
    =========================================================================
    
    1. Use consistent date format: YYYY-MM-DD
    2. All monetary values in KES (Kenyan Shillings)
    3. Market Cap = Current Price × Issued Shares
    4. PE Ratio = Current Price ÷ EPS
    5. PB Ratio = Current Price ÷ Book Value per Share
    6. Dividend Yield = (Annual Dividend ÷ Current Price) × 100%
    
    7. For missing data:
       - Use "0" for zero values
       - Leave empty for truly unavailable data
       - The app will handle missing data gracefully
    
    =========================================================================
    """
    
    with open('DATA_COLLECTION_GUIDE.txt', 'w') as f:
        f.write(guide)
    
    print("\n📘 Data collection guide created: DATA_COLLECTION_GUIDE.txt")

if __name__ == "__main__":
    setup_project()
    create_data_collection_guide()
    
    # Show final instructions
    print("\n" + "="*60)
    print("SETUP COMPLETE! SUMMARY OF CREATED FILES:")
    print("="*60)
    
    # List created files
    created_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(('.py', '.txt', '.xlsx', '.csv')):
                rel_path = os.path.join(root, file)
                created_files.append(rel_path)
    
    for file in sorted(created_files):
        if not file.startswith('./venv'):  # Skip venv files
            print(f"✓ {file}")
    
    print("\n🎉 You're ready to start analyzing NSE stocks!")
