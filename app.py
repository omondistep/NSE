# app.py - Complete Fixed Version
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import os
import sys

warnings.filterwarnings('ignore')

# Complete NSE symbols list from your data
COMPLETE_NSE_SYMBOLS = {
    # AGRICULTURAL
    'EGAD': 'Eaagads Ltd',
    'KUKZ': 'Kakuzi Plc',
    'KAPC': 'Kapchorua Tea Co. Ltd',
    'LIMT': 'The Limuru Tea Co. Plc',
    'SASN': 'Sasini Plc',
    'WTK': 'Williamson Tea Kenya Ltd',
    
    # AUTOMOBILES & ACCESSORIES
    'CGEN': 'Car & General (K) Ltd',
    
    # BANKING
    'ABSA': 'ABSA Bank Kenya Plc',
    'BKG': 'BK Group Plc',
    'DTK': 'Diamond Trust Bank Kenya Ltd',
    'EQTY': 'Equity Group Holdings Plc',
    'HFCK': 'HF Group Plc',
    'IMH': 'I&M Holdings Plc',
    'KCB': 'KCB Group Plc',
    'NCBA': 'NCBA Group Plc',
    'SBIC': 'Stanbic Holdings Plc',
    'SCBK': 'Standard Chartered Bank Kenya Ltd',
    'COOP': 'The Co-operative Bank of Kenya Ltd',
    
    # COMMERCIAL AND SERVICES
    'DCON': 'Deacons (East Africa) Plc',
    'EVRD': 'Eveready East Africa Ltd',
    'XPRS': 'Express Kenya Plc',
    'KQ': 'Kenya Airways Ltd',
    'LKL': 'Longhorn Publishers Plc',
    'NBV': 'Nairobi Business Ventures Ltd',
    'NMG': 'Nation Media Group Plc',
    'SMER': 'Sameer Africa Plc',
    'SGL': 'Standard Group Plc',
    'TPSE': 'TPS Eastern Africa Ltd',
    'UCHM': 'Uchumi Supermarket Plc',
    'SCAN': 'WPP Scangroup Plc',
    
    # CONSTRUCTION & ALLIED
    'ARM': 'ARM Cement Plc',
    'BAMB': 'Bamburi Cement Ltd',
    'CRWN': 'Crown Paints Kenya Plc',
    'CABL': 'E.A.Cables Ltd',
    'PORT': 'E.A.Portland Cement Co. Ltd',
    
    # ENERGY & PETROLEUM
    'KEGN': 'KenGen Co. Plc',
    'KPLC': 'Kenya Power & Lighting Co Plc',
    'KPLC.P0004': 'Kenya Power 4% Pref',
    'KPLC.P0007': 'Kenya Power 7% Pref',
    'TOTL': 'Total Kenya Ltd',
    'UMME': 'Umeme Ltd',
    
    # INSURANCE
    'BRIT': 'Britam Holdings Plc',
    'CIC': 'CIC Insurance Group Ltd',
    'JUB': 'Jubilee Holdings Ltd',
    'KNRE': 'Kenya Re Insurance Corporation Ltd',
    'LBTY': 'Liberty Kenya Holdings Ltd',
    'SLAM': 'Sanlam Kenya Plc',
    
    # INVESTMENT
    'CTUM': 'Centum Investment Co Plc',
    'HAFR': 'Home Afrika Ltd',
    'KURV': 'Kurwitu Ventures Ltd',
    'OCH': 'Olympia Capital Holdings Ltd',
    'TCL': 'Trans-Century Plc',
    
    # INVESTMENT SERVICES
    'NSE': 'Nairobi Securities Exchange Plc',
    
    # MANUFACTURING & ALLIED
    'BOC': 'B.O.C Kenya Plc',
    'BAT': 'British American Tobacco Kenya Plc',
    'CARB': 'Carbacid Investments Ltd',
    'EABL': 'East African Breweries Ltd',
    'FTGH': 'Flame Tree Group Holdings Ltd',
    'ORCH': 'Kenya Orchards Ltd',
    'MSC': 'Mumias Sugar Co. Ltd',
    'UNGA': 'Unga Group Ltd',
    
    # TELECOMMUNICATION
    'SCOM': 'Safaricom Plc',
    
    # REAL ESTATE INVESTMENT TRUST
    'FAHR': 'ILAM FAHARI I-REIT',
    
    # EXCHANGE TRADED FUNDS
    'GLD': 'NEW GOLD ETF'
}

# Sector classification
SECTOR_CLASSIFICATION = {
    'AGRICULTURAL': ['EGAD', 'KUKZ', 'KAPC', 'LIMT', 'SASN', 'WTK'],
    'AUTOMOBILES': ['CGEN'],
    'BANKING': ['ABSA', 'BKG', 'DTK', 'EQTY', 'HFCK', 'IMH', 'KCB', 'NCBA', 'SBIC', 'SCBK', 'COOP'],
    'COMMERCIAL': ['DCON', 'EVRD', 'XPRS', 'KQ', 'LKL', 'NBV', 'NMG', 'SMER', 'SGL', 'TPSE', 'UCHM', 'SCAN'],
    'CONSTRUCTION': ['ARM', 'BAMB', 'CRWN', 'CABL', 'PORT'],
    'ENERGY': ['KEGN', 'KPLC', 'KPLC.P0004', 'KPLC.P0007', 'TOTL', 'UMME'],
    'INSURANCE': ['BRIT', 'CIC', 'JUB', 'KNRE', 'LBTY', 'SLAM'],
    'INVESTMENT': ['CTUM', 'HAFR', 'KURV', 'OCH', 'TCL'],
    'INVESTMENT_SERVICES': ['NSE'],
    'MANUFACTURING': ['BOC', 'BAT', 'CARB', 'EABL', 'FTGH', 'ORCH', 'MSC', 'UNGA'],
    'TELECOMMUNICATION': ['SCOM'],
    'REIT': ['FAHR'],
    'ETF': ['GLD']
}

class NSEStockValuator:
    def __init__(self):
        self.nse_symbols = COMPLETE_NSE_SYMBOLS
        self.sector_classification = SECTOR_CLASSIFICATION
        self.price_data = None
        self.financial_data = None
        self.fundamentals = None
        
    def load_data(self, price_file, financial_file, fundamentals_file):
        """Load data from uploaded files"""
        try:
            # Read data based on file type
            if price_file.name.endswith('.csv'):
                self.price_data = pd.read_csv(price_file)
            else:
                self.price_data = pd.read_excel(price_file)
                
            if financial_file.name.endswith('.csv'):
                self.financial_data = pd.read_csv(financial_file)
            else:
                self.financial_data = pd.read_excel(financial_file)
                
            if fundamentals_file.name.endswith('.csv'):
                self.fundamentals = pd.read_csv(fundamentals_file)
            else:
                self.fundamentals = pd.read_excel(fundamentals_file)
            
            # Convert date columns
            if 'Date' in self.price_data.columns:
                self.price_data['Date'] = pd.to_datetime(self.price_data['Date'])
            
            # Clean data
            self._clean_data()
            
            st.success(f"✅ Data loaded successfully!")
            st.info(f"📊 Price records: {len(self.price_data):,}")
            st.info(f"🏢 Companies loaded: {self.fundamentals['Symbol'].nunique()}")
            
            return True
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def _clean_data(self):
        """Clean and prepare data"""
        # Ensure numeric columns are numeric
        numeric_cols_price = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols_price:
            if col in self.price_data.columns:
                self.price_data[col] = pd.to_numeric(self.price_data[col], errors='coerce')
        
        numeric_cols_financial = ['Revenue', 'Net_Income', 'Total_Assets', 'Total_Liabilities', 
                                 'Shareholders_Equity', 'EPS', 'Dividends']
        for col in numeric_cols_financial:
            if col in self.financial_data.columns:
                self.financial_data[col] = pd.to_numeric(self.financial_data[col], errors='coerce')
        
        numeric_cols_fundamentals = ['Market_Cap', 'Issued_Shares', 'PE_Ratio', 'PB_Ratio', 'Dividend_Yield']
        for col in numeric_cols_fundamentals:
            if col in self.fundamentals.columns:
                self.fundamentals[col] = pd.to_numeric(self.fundamentals[col], errors='coerce')
    
    def calculate_technical_indicators(self, symbol):
        """Calculate technical indicators for a stock"""
        if self.price_data is None or 'Symbol' not in self.price_data.columns:
            return None
            
        stock_data = self.price_data[self.price_data['Symbol'] == symbol].copy()
        
        if len(stock_data) < 20:
            return None
        
        # Sort by date
        stock_data = stock_data.sort_values('Date')
        
        # Ensure we have required columns
        if 'Close' not in stock_data.columns:
            return None
        
        # Moving averages
        stock_data['MA_20'] = stock_data['Close'].rolling(window=20).mean()
        stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['MA_200'] = stock_data['Close'].rolling(window=200).mean()
        
        # RSI
        delta = stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        stock_data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = stock_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = stock_data['Close'].ewm(span=26, adjust=False).mean()
        stock_data['MACD'] = exp1 - exp2
        stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        stock_data['BB_Middle'] = stock_data['Close'].rolling(window=20).mean()
        bb_std = stock_data['Close'].rolling(window=20).std()
        stock_data['BB_Upper'] = stock_data['BB_Middle'] + (bb_std * 2)
        stock_data['BB_Lower'] = stock_data['BB_Middle'] - (bb_std * 2)
        
        return stock_data
    
    def calculate_valuation_ratios(self, symbol):
        """Calculate valuation ratios"""
        try:
            if self.fundamentals is None or self.financial_data is None:
                return None
            
            fund = self.fundamentals[self.fundamentals['Symbol'] == symbol]
            if fund.empty:
                return None
            fund = fund.iloc[0]
            
            fin = self.financial_data[self.financial_data['Symbol'] == symbol]
            if fin.empty:
                return None
            
            latest_fin = fin.sort_values('Year', ascending=False).iloc[0]
            
            # Get current price (most recent)
            if self.price_data is None:
                return None
            price_data = self.price_data[self.price_data['Symbol'] == symbol]
            if price_data.empty:
                return None
            
            current_price = price_data.sort_values('Date')['Close'].iloc[-1]
            
            # Calculate ratios
            pe_ratio = current_price / latest_fin['EPS'] if latest_fin['EPS'] > 0 else np.nan
            pb_ratio = current_price / (latest_fin['Shareholders_Equity'] / fund['Issued_Shares']) if fund['Issued_Shares'] > 0 else np.nan
            dividend_yield = (latest_fin['Dividends'] / current_price * 100) if latest_fin['Dividends'] > 0 else 0
            
            return {
                'PE_Ratio': pe_ratio,
                'PB_Ratio': pb_ratio,
                'Dividend_Yield': dividend_yield,
                'Current_Price': current_price,
                'EPS': latest_fin['EPS'],
                'Book_Value': latest_fin['Shareholders_Equity'] / fund['Issued_Shares'] if fund['Issued_Shares'] > 0 else np.nan
            }
        except Exception as e:
            st.warning(f"Could not calculate valuation for {symbol}: {str(e)}")
            return None
    
    def dcf_valuation(self, symbol, growth_rate=0.05, discount_rate=0.12, years=5):
        """Discounted Cash Flow Valuation"""
        try:
            if self.financial_data is None:
                return None
            
            fin_data = self.financial_data[self.financial_data['Symbol'] == symbol]
            if len(fin_data) < 3:
                return None
            
            # Get latest free cash flow (simplified as Net Income)
            latest_ni = fin_data.sort_values('Year', ascending=False).iloc[0]['Net_Income']
            
            # Project future cash flows
            cash_flows = []
            for year in range(1, years + 1):
                future_cf = latest_ni * ((1 + growth_rate) ** year)
                discounted_cf = future_cf / ((1 + discount_rate) ** year)
                cash_flows.append(discounted_cf)
            
            # Terminal value
            terminal_cf = latest_ni * ((1 + growth_rate) ** (years + 1)) / (discount_rate - growth_rate)
            terminal_value = terminal_cf / ((1 + discount_rate) ** years)
            
            total_value = sum(cash_flows) + terminal_value
            
            # Get shares outstanding
            if self.fundamentals is None:
                return None
            fund = self.fundamentals[self.fundamentals['Symbol'] == symbol]
            if fund.empty:
                return None
            shares = fund['Issued_Shares'].values[0]
            
            intrinsic_value = total_value / shares if shares > 0 else np.nan
            
            return intrinsic_value
        except:
            return None
    
    def generate_recommendation(self, symbol):
        """Generate buy/sell/hold recommendation"""
        valuation = self.calculate_valuation_ratios(symbol)
        technical_data = self.calculate_technical_indicators(symbol)
        dcf_value = self.dcf_valuation(symbol)
        
        if not valuation:
            return {
                'Symbol': symbol,
                'Recommendation': 'NO_DATA',
                'Score': 0,
                'Reasons': ['Insufficient data for analysis'],
                'Current_Price': 0,
                'PE_Ratio': 0,
                'Dividend_Yield': '0%',
                'DCF_Value': 0,
                'Upside_Potential': 'N/A',
                'Sector': self.get_sector(symbol)
            }
        
        score = 0
        reasons = []
        
        # PE Ratio Analysis (NSE average ~10-12)
        if valuation['PE_Ratio'] < 8:
            score += 2
            reasons.append("Low P/E ratio suggests undervaluation")
        elif valuation['PE_Ratio'] > 15:
            score -= 2
            reasons.append("High P/E ratio suggests overvaluation")
        elif pd.isna(valuation['PE_Ratio']):
            reasons.append("P/E ratio not available")
        
        # Dividend Yield Analysis (NSE average ~5-7%)
        if valuation['Dividend_Yield'] > 8:
            score += 1.5
            reasons.append("High dividend yield attractive")
        elif valuation['Dividend_Yield'] < 3:
            score -= 1
            reasons.append("Low dividend yield")
        
        # Technical Analysis
        if technical_data is not None and not technical_data.empty:
            latest_price = technical_data['Close'].iloc[-1]
            
            if 'MA_20' in technical_data.columns and 'MA_50' in technical_data.columns:
                ma_20 = technical_data['MA_20'].iloc[-1]
                ma_50 = technical_data['MA_50'].iloc[-1]
                
                if latest_price > ma_20 and latest_price > ma_50:
                    score += 1
                    reasons.append("Trading above key moving averages")
                elif latest_price < ma_20 and latest_price < ma_50:
                    score -= 1
                    reasons.append("Trading below key moving averages")
            
            if 'RSI' in technical_data.columns:
                rsi = technical_data['RSI'].iloc[-1]
                if rsi < 30:
                    score += 1
                    reasons.append("Oversold based on RSI")
                elif rsi > 70:
                    score -= 1
                    reasons.append("Overbought based on RSI")
        else:
            reasons.append("Limited price history for technical analysis")
        
        # DCF Analysis
        if dcf_value and not np.isnan(dcf_value):
            latest_price = valuation['Current_Price']
            if latest_price < dcf_value * 0.8:
                score += 2
                reasons.append(f"Significantly undervalued (DCF: KES {dcf_value:.2f})")
            elif latest_price > dcf_value * 1.2:
                score -= 2
                reasons.append(f"Significantly overvalued (DCF: KES {dcf_value:.2f})")
            upside = ((dcf_value/latest_price - 1)*100) if latest_price > 0 else 0
        else:
            upside = 0
            reasons.append("DCF valuation not available")
        
        # Generate recommendation
        if score >= 3:
            recommendation = "STRONG_BUY"
        elif score >= 1:
            recommendation = "BUY"
        elif score == 0:
            recommendation = "HOLD"
        elif score >= -2:
            recommendation = "SELL"
        else:
            recommendation = "STRONG_SELL"
        
        return {
            'Symbol': symbol,
            'Name': self.nse_symbols.get(symbol, symbol),
            'Recommendation': recommendation,
            'Score': score,
            'Reasons': reasons,
            'Current_Price': valuation['Current_Price'],
            'PE_Ratio': valuation['PE_Ratio'],
            'Dividend_Yield': f"{valuation['Dividend_Yield']:.2f}%",
            'DCF_Value': dcf_value if dcf_value else 0,
            'Upside_Potential': f"{upside:.1f}%" if upside != 0 else "N/A",
            'Sector': self.get_sector(symbol)
        }
    
    def get_sector(self, symbol):
        """Get sector for a symbol"""
        for sector, symbols in self.sector_classification.items():
            if symbol in symbols:
                return sector
        return "UNCLASSIFIED"
    
    def analyze_all_stocks(self):
        """Analyze all NSE stocks"""
        results = []
        
        available_symbols = self.get_available_symbols()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(available_symbols):
            status_text.text(f"Analyzing {symbol}... ({i+1}/{len(available_symbols)})")
            try:
                result = self.generate_recommendation(symbol)
                results.append(result)
            except Exception as e:
                st.warning(f"Error analyzing {symbol}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(available_symbols))
        
        status_text.text("Analysis complete!")
        progress_bar.empty()
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # Clean the Sector column to ensure it's all strings
            if 'Sector' in results_df.columns:
                results_df['Sector'] = results_df['Sector'].astype(str).str.strip()
                # Replace any remaining float-like strings
                results_df['Sector'] = results_df['Sector'].replace({
                    'nan': 'UNCLASSIFIED',
                    'NaN': 'UNCLASSIFIED',
                    'None': 'UNCLASSIFIED'
                })
            
            # Sort by recommendation score
            results_df = results_df.sort_values('Score', ascending=False)
        
        return results_df
    
    def get_available_symbols(self):
        """Get symbols that have data available"""
        available = set()
        
        if self.fundamentals is not None and 'Symbol' in self.fundamentals.columns:
            available.update(self.fundamentals['Symbol'].unique())
        
        # Filter to only include known NSE symbols
        return [sym for sym in available if sym in self.nse_symbols]
    
    def generate_report(self, results_df):
        """Generate Excel report"""
        if results_df.empty:
            st.warning("No data to generate report")
            return None
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_cols = ['Symbol', 'Name', 'Sector', 'Recommendation', 'Current_Price', 
                          'PE_Ratio', 'Dividend_Yield', 'Upside_Potential', 'Score']
            summary_cols = [col for col in summary_cols if col in results_df.columns]
            results_df[summary_cols].to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed analysis sheet
            results_df.to_excel(writer, sheet_name='Detailed_Analysis', index=False)
            
            # Sector analysis
            sector_analysis = self.analyze_by_sector(results_df)
            sector_analysis.to_excel(writer, sheet_name='Sector_Analysis', index=False)
            
            # Top picks
            top_buys = results_df[results_df['Recommendation'].isin(['STRONG_BUY', 'BUY'])].head(10)
            top_buys.to_excel(writer, sheet_name='Top_Buy_Recommendations', index=False)
        
        output.seek(0)
        return output
    
    def analyze_by_sector(self, results_df):
        """Analyze performance by sector"""
        if results_df.empty or 'Sector' not in results_df.columns:
            return pd.DataFrame()
        
        sector_data = []
        
        # First, ensure all sectors are strings
        results_df['Sector'] = results_df['Sector'].astype(str).str.strip()
        
        for sector in results_df['Sector'].unique():
            if pd.isna(sector) or sector == 'nan' or sector == 'NaN':
                continue
                
            sector_results = results_df[results_df['Sector'] == sector]
            
            if not sector_results.empty:
                avg_score = sector_results['Score'].mean()
                buy_percentage = len(sector_results[sector_results['Recommendation'].isin(['BUY', 'STRONG_BUY'])]) / len(sector_results) * 100
                
                # Get top stock in this sector
                top_stock = sector_results.iloc[0]['Symbol'] if len(sector_results) > 0 else 'N/A'
                
                sector_data.append({
                    'Sector': sector,
                    'Number_of_Stocks': len(sector_results),
                    'Average_Score': avg_score,
                    'Buy_Recommendation_Percentage': f"{buy_percentage:.1f}%",
                    'Top_Stock': top_stock
                })
        
        return pd.DataFrame(sector_data).sort_values('Average_Score', ascending=False)

def create_sample_data_structure():
    """Create the sample data directory structure"""
    sample_dir = 'data/sample_data'
    
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir, exist_ok=True)
        
        # Create empty placeholder files
        placeholder_df = pd.DataFrame({'Note': ['Placeholder for sample data - run create_sample_data.py to generate full data']})
        
        for filename in ['nse_price_data.csv', 'nse_financial_data.csv', 'nse_fundamentals.csv']:
            filepath = os.path.join(sample_dir, filename)
            placeholder_df.to_csv(filepath, index=False)
        
        return True, sample_dir
    else:
        return False, sample_dir

def generate_simple_sample_data():
    """Generate simple sample data directly in the app (fallback option)"""
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        st.info("Generating simple sample data...")
        
        # Create directories
        os.makedirs('data/sample_data', exist_ok=True)
        
        # Create basic price data for 5 major stocks
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        symbols = ['SCOM', 'KCB', 'EQTY', 'EABL', 'COOP']
        
        price_data = []
        for symbol in symbols:
            if symbol == 'SCOM':
                base_price = 15.0
                volatility = 0.02
            elif symbol == 'KCB':
                base_price = 25.0
                volatility = 0.03
            elif symbol == 'EQTY':
                base_price = 40.0
                volatility = 0.025
            elif symbol == 'EABL':
                base_price = 130.0
                volatility = 0.015
            else:  # COOP
                base_price = 12.5
                volatility = 0.02
            
            for date in dates:
                daily_change = np.random.normal(0, volatility)
                base_price *= (1 + daily_change)
                base_price = max(base_price, 0.01)
                
                open_price = base_price
                high = open_price * (1 + abs(np.random.normal(0, volatility/2)))
                low = open_price * (1 - abs(np.random.normal(0, volatility/2)))
                close = open_price * (1 + np.random.normal(0, volatility/3))
                
                # Ensure high >= low
                high = max(open_price, close, high)
                low = min(open_price, close, low)
                
                volume = np.random.randint(10000, 1000000)
                
                price_data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Symbol': symbol,
                    'Open': round(open_price, 2),
                    'High': round(high, 2),
                    'Low': round(low, 2),
                    'Close': round(close, 2),
                    'Volume': volume
                })
        
        price_df = pd.DataFrame(price_data)
        
        # Create fundamentals data
        fundamentals_data = [
            {'Symbol': 'SCOM', 'Name': 'Safaricom Plc', 'Sector': 'TELECOMMUNICATION', 
             'Market_Cap': 600000000000, 'Issued_Shares': 40065428000,
             'PE_Ratio': 12.5, 'PB_Ratio': 4.2, 'Dividend_Yield': 5.8},
            {'Symbol': 'KCB', 'Name': 'KCB Group Plc', 'Sector': 'BANKING', 
             'Market_Cap': 150000000000, 'Issued_Shares': 3213462815,
             'PE_Ratio': 8.2, 'PB_Ratio': 1.5, 'Dividend_Yield': 6.2},
            {'Symbol': 'EQTY', 'Name': 'Equity Group Holdings Plc', 'Sector': 'BANKING', 
             'Market_Cap': 180000000000, 'Issued_Shares': 3773674802,
             'PE_Ratio': 9.1, 'PB_Ratio': 1.8, 'Dividend_Yield': 5.5},
            {'Symbol': 'EABL', 'Name': 'East African Breweries Ltd', 'Sector': 'MANUFACTURING', 
             'Market_Cap': 100000000000, 'Issued_Shares': 790774356,
             'PE_Ratio': 14.2, 'PB_Ratio': 3.5, 'Dividend_Yield': 4.8},
            {'Symbol': 'COOP', 'Name': 'Co-operative Bank of Kenya Ltd', 'Sector': 'BANKING', 
             'Market_Cap': 80000000000, 'Issued_Shares': 5867174695,
             'PE_Ratio': 7.5, 'PB_Ratio': 1.2, 'Dividend_Yield': 7.1}
        ]
        
        fundamentals_df = pd.DataFrame(fundamentals_data)
        
        # Create financial data
        financial_data = []
        for symbol in symbols:
            for year in [2020, 2021, 2022, 2023]:
                if symbol == 'SCOM':
                    revenue = 300000000000 + year * 20000000000
                    net_income = revenue * 0.25
                elif symbol == 'KCB':
                    revenue = 120000000000 + year * 8000000000
                    net_income = revenue * 0.22
                elif symbol == 'EQTY':
                    revenue = 100000000000 + year * 10000000000
                    net_income = revenue * 0.24
                elif symbol == 'EABL':
                    revenue = 80000000000 + year * 5000000000
                    net_income = revenue * 0.18
                else:  # COOP
                    revenue = 60000000000 + year * 4000000000
                    net_income = revenue * 0.20
                
                financial_data.append({
                    'Symbol': symbol,
                    'Year': year,
                    'Revenue': round(revenue, 2),
                    'Net_Income': round(net_income, 2),
                    'Total_Assets': round(revenue * 2.5, 2),
                    'Total_Liabilities': round(revenue * 1.5, 2),
                    'Shareholders_Equity': round(revenue * 1.0, 2),
                    'EPS': round(net_income / fundamentals_df[fundamentals_df['Symbol'] == symbol]['Issued_Shares'].values[0], 4),
                    'Dividends': round(net_income * 0.4 / fundamentals_df[fundamentals_df['Symbol'] == symbol]['Issued_Shares'].values[0], 4)
                })
        
        financial_df = pd.DataFrame(financial_data)
        
        # Save files
        price_df.to_csv('data/sample_data/nse_price_data.csv', index=False)
        fundamentals_df.to_csv('data/sample_data/nse_fundamentals.csv', index=False)
        financial_df.to_csv('data/sample_data/nse_financial_data.csv', index=False)
        
        return True, f"Generated {len(price_df):,} price records for {len(symbols)} companies"
        
    except ImportError as e:
        return False, f"Missing package: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

# Streamlit App
def main():
    st.set_page_config(
        page_title="NSE Stock Analyzer", 
        layout="wide",
        page_icon="📈"
    )
    
    st.title("📈 Nairobi Securities Exchange Stock Analysis")
    st.markdown("---")
    
    # Initialize session state
    if 'valuator' not in st.session_state:
        st.session_state.valuator = NSEStockValuator()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'data_source' not in st.session_state:  # Track data source
        st.session_state.data_source = None  # 'sample', 'uploaded', or None
    
    # Sidebar
    with st.sidebar:
        st.header("📂 Load Data")
        
        # Create two columns for data loading options
        col_sample, col_upload = st.columns(2)
        
        with col_sample:
            st.subheader("🎯 Try Sample Data")
            st.markdown("""
            Load pre-configured sample data to test the app.
            Files should be in `data/sample_data/` directory.
            """)
            
            if st.button("📁 Load Sample Data", key="load_sample", use_container_width=True):
                try:
                    # Define sample data paths
                    sample_data_dir = 'data/sample_data'
                    price_path = os.path.join(sample_data_dir, 'nse_price_data.csv')
                    financial_path = os.path.join(sample_data_dir, 'nse_financial_data.csv')
                    fundamentals_path = os.path.join(sample_data_dir, 'nse_fundamentals.csv')
                    
                    if os.path.exists(price_path) and os.path.exists(financial_path) and os.path.exists(fundamentals_path):
                        
                        with st.spinner("Loading sample data..."):
                            # Read the sample data
                            price_data = pd.read_csv(price_path)
                            financial_data = pd.read_csv(financial_path)
                            fundamentals = pd.read_csv(fundamentals_path)
                            
                            # Clean the fundamentals data - ensure Sector column is string
                            if 'Sector' in fundamentals.columns:
                                fundamentals['Sector'] = fundamentals['Sector'].astype(str).str.strip()
                            
                            # Store the data source
                            st.session_state.data_source = 'sample'
                            
                            # Store the data directly in the valuator
                            st.session_state.valuator.price_data = price_data
                            st.session_state.valuator.financial_data = financial_data
                            st.session_state.valuator.fundamentals = fundamentals
                            
                            # Clean the data
                            st.session_state.valuator._clean_data()
                            
                            st.session_state.data_loaded = True
                            st.success("✅ Sample data loaded successfully!")
                            
                            # Show data info with source indicator
                            st.info(f"📊 Price records: {len(price_data):,}")
                            st.info(f"🏢 Companies loaded: {fundamentals['Symbol'].nunique()}")
                            st.info(f"📂 Data source: Sample data (demo dataset)")
                            
                            # Automatically run analysis
                            with st.spinner("Analyzing stocks..."):
                                results = st.session_state.valuator.analyze_all_stocks()
                                st.session_state.analysis_results = results
                                st.success("✅ Analysis complete!")
                    else:
                        # If sample_data directory doesn't exist, check the root data directory as fallback
                        fallback_price = 'data/nse_price_data.csv'
                        fallback_financial = 'data/nse_financial_data.csv'
                        fallback_fundamentals = 'data/nse_fundamentals.csv'
                        
                        if os.path.exists(fallback_price) and os.path.exists(fallback_financial) and os.path.exists(fallback_fundamentals):
                            st.warning("⚠️ Sample data not found in 'data/sample_data/' directory, but found in 'data/' directory. Using those files.")
                            
                            with st.spinner("Loading data from data/ directory..."):
                                price_data = pd.read_csv(fallback_price)
                                financial_data = pd.read_csv(fallback_financial)
                                fundamentals = pd.read_csv(fallback_fundamentals)
                                
                                if 'Sector' in fundamentals.columns:
                                    fundamentals['Sector'] = fundamentals['Sector'].astype(str).str.strip()
                                
                                st.session_state.data_source = 'sample'
                                st.session_state.valuator.price_data = price_data
                                st.session_state.valuator.financial_data = financial_data
                                st.session_state.valuator.fundamentals = fundamentals
                                st.session_state.valuator._clean_data()
                                st.session_state.data_loaded = True
                                
                                st.success("✅ Data loaded from data/ directory!")
                                st.info(f"📊 Price records: {len(price_data):,}")
                                st.info(f"🏢 Companies loaded: {fundamentals['Symbol'].nunique()}")
                                st.info(f"📂 Data source: Sample data")
                                
                                with st.spinner("Analyzing stocks..."):
                                    results = st.session_state.valuator.analyze_all_stocks()
                                    st.session_state.analysis_results = results
                                    st.success("✅ Analysis complete!")
                        else:
                            st.error("""
                            ❌ Sample data files not found!
                            
                            Please ensure you have one of these:
                            1. **Sample data** in `data/sample_data/` directory with:
                               - `nse_price_data.csv`
                               - `nse_financial_data.csv`
                               - `nse_fundamentals.csv`
                            
                            2. **OR** data files in `data/` directory with same names
                            
                            3. **OR** upload your own data files
                            
                            4. **OR** click "Generate Sample Data" below
                            """)
                except Exception as e:
                    st.error(f"❌ Error loading sample data: {str(e)}")
        
        with col_upload:
            st.subheader("📤 Upload Your Data")
            st.markdown("""
            Upload your own CSV/Excel files.
            Ensure they follow the required format.
            """)
        
        st.markdown("---")
        
        # File uploaders for custom data
        price_file = st.file_uploader("Upload Price Data", type=['csv', 'xlsx'], 
                                     help="CSV or Excel with columns: Date, Symbol, Open, High, Low, Close, Volume")
        financial_file = st.file_uploader("Upload Financial Data", type=['csv', 'xlsx'],
                                         help="CSV or Excel with columns: Symbol, Year, Revenue, Net_Income, etc.")
        fundamentals_file = st.file_uploader("Upload Fundamentals Data", type=['csv', 'xlsx'],
                                           help="CSV or Excel with columns: Symbol, Name, Sector, Market_Cap, etc.")
        
        if price_file and financial_file and fundamentals_file:
            if st.button("🚀 Load & Analyze Uploaded Data", key="load_uploaded", use_container_width=True):
                with st.spinner("Loading and analyzing uploaded data..."):
                    success = st.session_state.valuator.load_data(price_file, financial_file, fundamentals_file)
                    if success:
                        st.session_state.data_loaded = True
                        st.session_state.data_source = 'uploaded'
                        results = st.session_state.valuator.analyze_all_stocks()
                        st.session_state.analysis_results = results
                        st.success("✅ Analysis complete!")
        
        st.markdown("---")
        st.header("⚙️ Analysis Settings")
        
        # DCF parameters
        st.subheader("DCF Parameters")
        growth_rate = st.slider("Growth Rate (%)", 0.0, 15.0, 5.0, 0.5) / 100
        discount_rate = st.slider("Discount Rate (%)", 5.0, 20.0, 12.0, 0.5) / 100
        
        # Update DCF parameters in valuator
        original_dcf = st.session_state.valuator.dcf_valuation
        st.session_state.valuator.dcf_valuation = lambda symbol, gr=growth_rate, dr=discount_rate, years=5: \
            original_dcf(symbol, gr, dr, years)
        
        st.markdown("---")
        st.header("🛠️ Setup")
        
        # Create sample data directory button
        if st.button("📁 Create Sample Data Directory Structure", key="create_sample_dir"):
            created, sample_dir = create_sample_data_structure()
            if created:
                st.success(f"✅ Created sample data directory: {sample_dir}")
                st.info("📝 Now click 'Generate Sample Data' to populate with data")
            else:
                st.info(f"📁 Sample data directory already exists: {sample_dir}")
                
                # Check if files exist
                files = ['nse_price_data.csv', 'nse_financial_data.csv', 'nse_fundamentals.csv']
                existing_files = []
                
                for filename in files:
                    filepath = os.path.join(sample_dir, filename)
                    if os.path.exists(filepath):
                        existing_files.append(filename)
                
                if existing_files:
                    st.success(f"✅ Found {len(existing_files)} sample data file(s)")
                    for f in existing_files:
                        st.write(f"  - {f}")
                else:
                    st.warning("⚠️ No sample data files found in the directory")
                    st.info("📝 Click 'Generate Sample Data' to create data files")
        
        # Run sample data generation - FIXED VERSION
        if st.button("🔄 Generate Sample Data", key="generate_sample_data"):
            try:
                # First, check if required packages are available
                try:
                    import pandas as pd
                    import numpy as np
                    packages_available = True
                except ImportError as e:
                    packages_available = False
                    missing_package = str(e).split("'")[1] if "'" in str(e) else "unknown"
                
                if not packages_available:
                    st.error(f"""
                    ❌ Missing required package: `{missing_package}`
                    
                    Please install required packages first:
                    ```bash
                    pip install pandas numpy
                    ```
                    
                    Then restart the application.
                    """)
                    return
                
                # Try to use the external script first
                if os.path.exists('create_sample_data.py'):
                    try:
                        with st.spinner("Generating comprehensive sample data..."):
                            # Import the module
                            import importlib.util
                            import sys
                            
                            # Check if module is already loaded
                            if 'create_sample_data' in sys.modules:
                                del sys.modules['create_sample_data']
                            
                            # Load and run the module
                            spec = importlib.util.spec_from_file_location("create_sample_data", "create_sample_data.py")
                            module = importlib.util.module_from_spec(spec)
                            
                            # Capture output
                            import io
                            from contextlib import redirect_stdout, redirect_stderr
                            
                            output = io.StringIO()
                            with redirect_stdout(output), redirect_stderr(output):
                                try:
                                    spec.loader.exec_module(module)
                                    # Check if module has a main function
                                    if hasattr(module, 'main'):
                                        module.main()
                                    elif hasattr(module, 'generate_sample_data'):
                                        module.generate_sample_data()
                                    else:
                                        # Assume the script runs on import
                                        pass
                                except Exception as module_error:
                                    st.error(f"❌ Error in sample data script: {str(module_error)}")
                                    # Fall back to simple data generation
                                    success, message = generate_simple_sample_data()
                                    if success:
                                        st.success(f"✅ {message}")
                                    else:
                                        st.error(f"❌ {message}")
                                    return
                            
                            # Show output
                            output_text = output.getvalue()
                            if output_text:
                                st.success("✅ Sample data generated successfully!")
                                st.code(output_text[:1000] + "..." if len(output_text) > 1000 else output_text)
                            else:
                                st.success("✅ Sample data generated successfully!")
                                
                            # Show generated files
                            if os.path.exists('data/sample_data'):
                                files = os.listdir('data/sample_data')
                                if files:
                                    st.info(f"📁 Generated {len(files)} files in data/sample_data/")
                                    for file in files:
                                        filepath = os.path.join('data/sample_data', file)
                                        size = os.path.getsize(filepath)
                                        st.write(f"  - {file} ({size:,} bytes)")
                            
                    except Exception as e:
                        st.error(f"❌ Error running sample data script: {str(e)}")
                        st.info("Trying simple data generation...")
                        
                        # Fall back to simple data generation
                        success, message = generate_simple_sample_data()
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                
                else:
                    # No external script, use the built-in function
                    st.info("No external create_sample_data.py found. Using built-in data generator...")
                    success, message = generate_simple_sample_data()
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                        
            except Exception as e:
                st.error(f"❌ Error generating sample data: {str(e)}")
        
        st.markdown("---")
        st.header("🔄 Data Management")
        
        if st.button("🗑️ Clear Current Data", key="clear_data"):
            st.session_state.data_loaded = False
            st.session_state.analysis_results = None
            st.session_state.data_source = None
            st.session_state.valuator = NSEStockValuator()  # Reset the valuator
            st.rerun()
        
        # Display current status
        if st.session_state.data_source:
            st.info(f"**Current Data:** {st.session_state.data_source.upper()}")
        else:
            st.info("**Current Data:** No data loaded")
        
        st.markdown("---")
        st.info("""
        **💡 Tips:**
        1. Ensure your CSV/Excel files follow the template format
        2. Dates should be in YYYY-MM-DD format
        3. All prices in KES (Kenyan Shillings)
        4. For best results, include at least 1 year of price data
        """)
    
    # Main content area
    if not st.session_state.data_loaded:
        # Welcome screen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("""
            ### 🚀 Welcome to NSE Stock Analysis App!
            
            This app helps you analyze stocks listed on the **Nairobi Securities Exchange (NSE)** 
            using fundamental and technical analysis.
            
            **How to use this app:**
            
            1. **📁 Prepare sample data** by clicking "Generate Sample Data" in the sidebar
            2. **📤 OR upload your own data** using the file uploaders in the sidebar
            3. **⚙️ Configure analysis settings** (DCF parameters, etc.)
            4. **🚀 Click 'Load & Analyze'** to start analysis
            
            **Features included:**
            - 📊 Fundamental analysis (P/E, P/B, Dividend Yield)
            - 📈 Technical analysis (RSI, MACD, Moving Averages)
            - 💰 Discounted Cash Flow (DCF) valuation
            - 🏢 Sector-based analysis
            - 📥 Excel report generation
            - 🎯 Buy/Sell/Hold recommendations
            """)
        
        with col2:
            st.markdown("### 📊 NSE Market Overview")
            st.metric("Total Listed Companies", len(COMPLETE_NSE_SYMBOLS))
            st.metric("Sectors Covered", len(SECTOR_CLASSIFICATION))
            
            # Quick sector breakdown
            sector_counts = {sector: len(symbols) for sector, symbols in SECTOR_CLASSIFICATION.items()}
            top_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            st.markdown("**Top Sectors:**")
            for sector, count in top_sectors:
                st.write(f"- {sector}: {count} companies")
            
            # Show data status
            st.markdown("### 📂 Data Status")
            if os.path.exists('data/sample_data'):
                st.success("✅ Sample data directory exists")
            else:
                st.warning("⚠️ Sample data directory not found")
            
            if os.path.exists('create_sample_data.py'):
                st.success("✅ Sample data script available")
            else:
                st.info("ℹ️ No external sample data script found (built-in generator available)")
        
        st.markdown("---")
        st.warning("""
        ⚠️ **Please load data first** using one of the options in the sidebar:
        
        1. **🎯 Try Sample Data**: Click "Generate Sample Data" then "Load Sample Data"
        2. **📤 Upload Your Own Data**: Use the file uploaders below the sample data option
        3. **🛠️ Setup**: Use the setup tools if you need to create directories
        """)
        
    else:
        # Display data source banner at the top
        if st.session_state.data_source == 'sample':
            st.success("🎯 **Currently analyzing: SAMPLE DATA** - This is demonstration data from the sample_data directory.")
        elif st.session_state.data_source == 'uploaded':
            st.success("📤 **Currently analyzing: YOUR UPLOADED DATA** - This is your custom data.")
        
        # Analysis results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Summary", "🔍 Stock Analysis", "📊 Sector View", "📈 Charts", "📥 Export"
        ])
        
        results_df = st.session_state.analysis_results
        
        with tab1:
            st.header("Stock Recommendations Summary")
            
            if results_df is not None and not results_df.empty:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Analyzed", len(results_df))
                with col2:
                    buy_count = len(results_df[results_df['Recommendation'].isin(['BUY', 'STRONG_BUY'])])
                    st.metric("Buy Recommendations", buy_count)
                with col3:
                    hold_count = len(results_df[results_df['Recommendation'] == 'HOLD'])
                    st.metric("Hold Recommendations", hold_count)
                with col4:
                    sell_count = len(results_df[results_df['Recommendation'].isin(['SELL', 'STRONG_SELL'])])
                    st.metric("Sell Recommendations", sell_count)
                
                # Color coding function
                def color_recommendation(val):
                    if val == 'STRONG_BUY':
                        return 'background-color: #2E7D32; color: white; font-weight: bold'
                    elif val == 'BUY':
                        return 'background-color: #4CAF50; color: white'
                    elif val == 'HOLD':
                        return 'background-color: #FFC107; color: black'
                    elif val == 'SELL':
                        return 'background-color: #FF9800; color: white'
                    elif val == 'STRONG_SELL':
                        return 'background-color: #F44336; color: white; font-weight: bold'
                    return ''
                
                # Filter options
                col_filter1, col_filter2, col_filter3 = st.columns(3)
                with col_filter1:
                    # Ensure Recommendation column exists
                    if 'Recommendation' in results_df.columns:
                        rec_options = results_df['Recommendation'].unique()
                        
                        # Create default values based on available options
                        default_values = []
                        if 'STRONG_BUY' in rec_options:
                            default_values.append('STRONG_BUY')
                        if 'BUY' in rec_options:
                            default_values.append('BUY')
                        # If no BUY recommendations, default to first available option
                        if not default_values and len(rec_options) > 0:
                            default_values = [rec_options[0]]
                        
                        filter_recommendation = st.multiselect(
                            "Filter by Recommendation",
                            options=list(rec_options),
                            default=default_values
                        )
                    else:
                        filter_recommendation = []
                
                with col_filter2:
                    # Clean Sector column before displaying options
                    if 'Sector' in results_df.columns:
                        # Ensure Sector column is clean string
                        results_df['Sector'] = results_df['Sector'].astype(str).str.strip()
                        sector_options = sorted(results_df['Sector'].unique())
                        filter_sector = st.multiselect(
                            "Filter by Sector",
                            options=sector_options,
                            default=[]
                        )
                    else:
                        filter_sector = []
                
                with col_filter3:
                    if 'Current_Price' in results_df.columns:
                        min_price_val = results_df['Current_Price'].min()
                        max_price_val = results_df['Current_Price'].max()
                        min_price = st.number_input("Min Price (KES)", 
                                                   value=float(min_price_val), 
                                                   min_value=0.0, 
                                                   step=0.1)
                        max_price = st.number_input("Max Price (KES)", 
                                                   value=float(max_price_val), 
                                                   min_value=0.0, 
                                                   step=0.1)
                    else:
                        min_price = 0.0
                        max_price = 1000.0
                        st.warning("Current Price data not available")
                
                # Apply filters
                filtered_df = results_df.copy()
                if filter_recommendation:
                    filtered_df = filtered_df[filtered_df['Recommendation'].isin(filter_recommendation)]
                if filter_sector:
                    filtered_df = filtered_df[filtered_df['Sector'].isin(filter_sector)]
                if 'Current_Price' in filtered_df.columns:
                    filtered_df = filtered_df[
                        (filtered_df['Current_Price'] >= min_price) & 
                        (filtered_df['Current_Price'] <= max_price)
                    ]
                
                # Display table
                display_cols = ['Symbol', 'Name', 'Sector', 'Recommendation', 'Current_Price', 
                              'PE_Ratio', 'Dividend_Yield', 'Upside_Potential', 'Score']
                
                # Ensure all columns exist
                display_cols = [col for col in display_cols if col in filtered_df.columns]
                
                if not filtered_df.empty:
                    st.dataframe(
                        filtered_df[display_cols].style.applymap(
                            color_recommendation, subset=['Recommendation']
                        ),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Top picks
                    st.subheader("🎯 Top Investment Picks")
                    top_picks = filtered_df.head(5)
                    
                    for idx, row in top_picks.iterrows():
                        with st.expander(f"{row['Symbol']} - {row['Name']} ({row['Recommendation']})"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Current Price", f"KES {row['Current_Price']:.2f}")
                                if 'PE_Ratio' in row and not pd.isna(row['PE_Ratio']):
                                    st.metric("P/E Ratio", f"{row['PE_Ratio']:.2f}")
                                else:
                                    st.metric("P/E Ratio", "N/A")
                            with col_b:
                                if 'Dividend_Yield' in row:
                                    st.metric("Dividend Yield", row['Dividend_Yield'])
                                if 'Upside_Potential' in row:
                                    st.metric("Upside Potential", row['Upside_Potential'])
                            
                            st.write("**Analysis Reasons:**")
                            if 'Reasons' in row and isinstance(row['Reasons'], list):
                                for reason in row['Reasons']:
                                    st.write(f"• {reason}")
                            elif 'Reasons' in row:
                                st.write(f"• {row['Reasons']}")
                else:
                    st.warning("No data matches the selected filters")
            else:
                st.warning("No analysis results available. Please load and analyze data first.")
        
        with tab2:
            st.header("Individual Stock Analysis")
            
            if results_df is not None and not results_df.empty:
                # Stock selector
                stock_options = []
                for symbol in results_df['Symbol'].unique():
                    name = results_df[results_df['Symbol'] == symbol]['Name'].iloc[0] if 'Name' in results_df.columns else symbol
                    stock_options.append(f"{symbol} - {name}")
                
                selected_stock = st.selectbox(
                    "Select a Stock",
                    options=stock_options
                )
                
                if selected_stock:
                    selected_symbol = selected_stock.split(" - ")[0]
                    stock_data = st.session_state.valuator.calculate_technical_indicators(selected_symbol)
                    stock_rec = results_df[results_df['Symbol'] == selected_symbol].iloc[0]
                    
                    # Stock overview
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if 'Current_Price' in stock_rec:
                            st.metric("Current Price", f"KES {stock_rec['Current_Price']:.2f}")
                    with col2:
                        if 'Recommendation' in stock_rec:
                            st.metric("Recommendation", stock_rec['Recommendation'])
                    with col3:
                        if 'PE_Ratio' in stock_rec and not pd.isna(stock_rec['PE_Ratio']):
                            st.metric("P/E Ratio", f"{stock_rec['PE_Ratio']:.2f}")
                        else:
                            st.metric("P/E Ratio", "N/A")
                    with col4:
                        if 'Dividend_Yield' in stock_rec:
                            st.metric("Dividend Yield", stock_rec['Dividend_Yield"])
                    
                    # Charts
                    if stock_data is not None and not stock_data.empty:
                        st.subheader("Price Chart with Indicators")
                        
                        # Price chart with moving averages
                        fig_price = go.Figure()
                        fig_price.add_trace(go.Scatter(
                            x=stock_data['Date'], y=stock_data['Close'],
                            mode='lines', name='Close Price',
                            line=dict(color='#1f77b4', width=2)
                        ))
                        
                        if 'MA_20' in stock_data.columns:
                            fig_price.add_trace(go.Scatter(
                                x=stock_data['Date'], y=stock_data['MA_20'],
                                mode='lines', name='20-Day MA',
                                line=dict(color='orange', width=1, dash='dash')
                            ))
                        
                        if 'MA_50' in stock_data.columns:
                            fig_price.add_trace(go.Scatter(
                                x=stock_data['Date'], y=stock_data['MA_50'],
                                mode='lines', name='50-Day MA',
                                line=dict(color='red', width=1, dash='dash')
                            ))
                        
                        fig_price.update_layout(
                            title=f"{selected_symbol} Price History",
                            xaxis_title="Date",
                            yaxis_title="Price (KES)",
                            hovermode='x unified',
                            height=400
                        )
                        st.plotly_chart(fig_price, use_container_width=True)
                        
                        # RSI Chart
                        if 'RSI' in stock_data.columns:
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(
                                x=stock_data['Date'], y=stock_data['RSI'],
                                mode='lines', name='RSI',
                                line=dict(color='purple', width=2)
                            ))
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                                            annotation_text="Overbought")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                                            annotation_text="Oversold")
                            fig_rsi.update_layout(
                                title="RSI (14-day)",
                                xaxis_title="Date",
                                yaxis_title="RSI",
                                height=300
                            )
                            st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # Detailed Analysis
                    st.subheader("Detailed Analysis")
                    
                    col_analysis1, col_analysis2 = st.columns(2)
                    
                    with col_analysis1:
                        st.markdown("**Valuation Metrics:**")
                        if 'PE_Ratio' in stock_rec and not pd.isna(stock_rec['PE_Ratio']):
                            st.write(f"- **P/E Ratio:** {stock_rec['PE_Ratio']:.2f}")
                        if 'Dividend_Yield' in stock_rec:
                            st.write(f"- **Dividend Yield:** {stock_rec['Dividend_Yield']}")
                        if 'DCF_Value' in stock_rec and stock_rec['DCF_Value'] and stock_rec['DCF_Value'] > 0:
                            st.write(f"- **DCF Intrinsic Value:** KES {stock_rec['DCF_Value']:.2f}")
                        if 'Upside_Potential' in stock_rec:
                            st.write(f"- **Margin of Safety:** {stock_rec['Upside_Potential']}")
                    
                    with col_analysis2:
                        st.markdown("**Recommendation Reasons:**")
                        if 'Reasons' in stock_rec and isinstance(stock_rec['Reasons'], list):
                            for reason in stock_rec['Reasons']:
                                st.write(f"• {reason}")
                        elif 'Reasons' in stock_rec:
                            st.write(f"• {stock_rec['Reasons']}")
                    
                    # Risk Assessment
                    st.subheader("Risk Assessment")
                    if 'Score' in stock_rec:
                        risk_score = -stock_rec['Score'] if stock_rec['Score'] < 0 else 10 - stock_rec['Score']
                        risk_level = "Low" if risk_score <= 3 else "Medium" if risk_score <= 6 else "High"
                        st.progress(min(risk_score / 10, 1.0), text=f"Risk Level: {risk_level} ({risk_score}/10)")
            else:
                st.warning("No stock data available. Please analyze stocks first.")
        
        with tab3:
            st.header("Sector Analysis")
            
            if results_df is not None and not results_df.empty:
                # Ensure Sector column is clean
                results_df['Sector'] = results_df['Sector'].astype(str).str.strip()
                
                sector_analysis = st.session_state.valuator.analyze_by_sector(results_df)
                
                if not sector_analysis.empty:
                    # Sector metrics
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        best_sector = sector_analysis.iloc[0]['Sector']
                        best_score = sector_analysis.iloc[0]['Average_Score']
                        st.metric("Best Performing Sector", best_sector, f"Score: {best_score:.1f}")
                    with col_s2:
                        worst_sector = sector_analysis.iloc[-1]['Sector']
                        worst_score = sector_analysis.iloc[-1]['Average_Score']
                        st.metric("Worst Performing Sector", worst_sector, f"Score: {worst_score:.1f}")
                    with col_s3:
                        avg_sector_score = sector_analysis['Average_Score'].mean()
                        st.metric("Average Sector Score", f"{avg_sector_score:.1f}")
                    
                    # Sector comparison chart
                    fig_sector = px.bar(
                        sector_analysis,
                        x='Sector',
                        y='Average_Score',
                        color='Average_Score',
                        title='Sector Performance Comparison',
                        labels={'Average_Score': 'Average Score', 'Sector': 'Sector'},
                        color_continuous_scale='RdYlGn'
                    )
                    fig_sector.update_layout(height=500)
                    st.plotly_chart(fig_sector, use_container_width=True)
                    
                    # Sector details table
                    st.dataframe(
                        sector_analysis,
                        use_container_width=True
                    )
                    
                    # Sector recommendations
                    st.subheader("Sector Investment Recommendations")
                    for idx, row in sector_analysis.iterrows():
                        with st.expander(f"{row['Sector']} Sector - Score: {row['Average_Score']:.1f}"):
                            st.write(f"**Number of Stocks:** {row['Number_of_Stocks']}")
                            st.write(f"**Buy Recommendation Percentage:** {row['Buy_Recommendation_Percentage']}")
                            st.write(f"**Top Stock:** {row['Top_Stock']}")
                            
                            # Show stocks in this sector
                            sector_stocks = results_df[results_df['Sector'] == row['Sector']]
                            if not sector_stocks.empty:
                                st.write("**Stocks in this sector:**")
                                for _, stock in sector_stocks.iterrows():
                                    st.write(f"- {stock['Symbol']}: {stock['Recommendation']} (Score: {stock['Score']:.1f})")
                else:
                    st.warning("No sector data available.")
            else:
                st.warning("No analysis results available.")
        
        with tab4:
            st.header("Market Charts & Visualizations")
            
            if results_df is not None and not results_df.empty:
                # Chart type selection
                chart_type = st.selectbox(
                    "Select Chart Type",
                    ["Recommendation Distribution", "Price vs P/E Ratio", "Dividend Yield Heatmap", "Sector Performance"]
                )
                
                if chart_type == "Recommendation Distribution":
                    if 'Recommendation' in results_df.columns:
                        rec_counts = results_df['Recommendation'].value_counts()
                        fig_dist = px.pie(
                            values=rec_counts.values,
                            names=rec_counts.index,
                            title="Recommendation Distribution",
                            color=rec_counts.index,
                            color_discrete_map={
                                'STRONG_BUY': '#2E7D32',
                                'BUY': '#4CAF50',
                                'HOLD': '#FFC107',
                                'SELL': '#FF9800',
                                'STRONG_SELL': '#F44336'
                            }
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    else:
                        st.warning("Recommendation data not available")
                
                elif chart_type == "Price vs P/E Ratio":
                    if 'PE_Ratio' in results_df.columns and 'Current_Price' in results_df.columns and 'Recommendation' in results_df.columns:
                        # Filter out NaN values
                        chart_data = results_df.dropna(subset=['PE_Ratio', 'Current_Price'])
                        fig_scatter = px.scatter(
                            chart_data,
                            x='PE_Ratio',
                            y='Current_Price',
                            color='Recommendation',
                            size='Score' if 'Score' in chart_data.columns else None,
                            hover_name='Symbol',
                            title="Price vs P/E Ratio Analysis",
                            labels={'PE_Ratio': 'P/E Ratio', 'Current_Price': 'Current Price (KES)'},
                            color_discrete_map={
                                'STRONG_BUY': '#2E7D32',
                                'BUY': '#4CAF50',
                                'HOLD': '#FFC107',
                                'SELL': '#FF9800',
                                'STRONG_SELL': '#F44336'
                            }
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.warning("Required data columns not available")
                
                elif chart_type == "Dividend Yield Heatmap":
                    # Prepare data for heatmap
                    if 'Dividend_Yield' in results_df.columns and 'Sector' in results_df.columns and 'Recommendation' in results_df.columns:
                        # Clean the Dividend_Yield column
                        results_df['Dividend_Yield_Clean'] = results_df['Dividend_Yield'].astype(str).str.replace('%', '').astype(float)
                        
                        heatmap_data = results_df.pivot_table(
                            values='Dividend_Yield_Clean',
                            index='Sector',
                            columns='Recommendation',
                            aggfunc='mean',
                            fill_value=0
                        )
                        
                        if not heatmap_data.empty:
                            fig_heatmap = px.imshow(
                                heatmap_data,
                                title="Average Dividend Yield by Sector and Recommendation",
                                labels=dict(x="Recommendation", y="Sector", color="Dividend Yield (%)"),
                                aspect="auto",
                                color_continuous_scale="YlOrRd"
                            )
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                        else:
                            st.warning("No data available for heatmap")
                    else:
                        st.warning("Required data columns not available")
                
                elif chart_type == "Sector Performance":
                    if 'Sector' in results_df.columns and 'Score' in results_df.columns and 'Current_Price' in results_df.columns:
                        sector_perf = results_df.groupby('Sector').agg({
                            'Score': 'mean',
                            'Current_Price': 'mean',
                            'Symbol': 'count'
                        }).reset_index()
                        
                        fig_sector_perf = px.scatter(
                            sector_perf,
                            x='Current_Price',
                            y='Score',
                            size='Symbol',
                            color='Sector',
                            hover_name='Sector',
                            title="Sector Performance: Average Score vs Average Price",
                            labels={'Current_Price': 'Average Price (KES)', 'Score': 'Average Score'}
                        )
                        st.plotly_chart(fig_sector_perf, use_container_width=True)
                    else:
                        st.warning("Required data columns not available")
            else:
                st.warning("No data available for charts.")
        
        with tab5:
            st.header("Export Results")
            
            if results_df is not None and not results_df.empty:
                # Export options
                st.subheader("Export Options")
                
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    export_format = st.selectbox(
                        "Export Format",
                        ["Excel Report", "CSV Summary", "JSON Data"]
                    )
                
                with col_export2:
                    include_details = st.checkbox("Include Detailed Analysis", value=True)
                
                # Generate export
                if st.button("📥 Generate Export"):
                    with st.spinner("Generating export file..."):
                        if export_format == "Excel Report":
                            excel_file = st.session_state.valuator.generate_report(results_df)
                            if excel_file:
                                st.download_button(
                                    label="⬇️ Download Excel Report",
                                    data=excel_file,
                                    file_name=f"nse_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        
                        elif export_format == "CSV Summary":
                            # Clean the data before exporting
                            export_df = results_df.copy()
                            if 'Reasons' in export_df.columns:
                                export_df['Reasons'] = export_df['Reasons'].apply(
                                    lambda x: '; '.join(x) if isinstance(x, list) else str(x)
                                )
                            csv_data = export_df.to_csv(index=False)
                            st.download_button(
                                label="⬇️ Download CSV Summary",
                                data=csv_data,
                                file_name=f"nse_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        elif export_format == "JSON Data":
                            # Clean the data before exporting
                            export_df = results_df.copy()
                            if 'Reasons' in export_df.columns:
                                export_df['Reasons'] = export_df['Reasons'].apply(
                                    lambda x: x if isinstance(x, list) else [str(x)]
                                )
                            json_data = export_df.to_json(orient='records', indent=2)
                            st.download_button(
                                label="⬇️ Download JSON Data",
                                data=json_data,
                                file_name=f"nse_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                
                # Preview data
                st.subheader("Data Preview")
                preview_df = results_df.head(10).copy()
                # Clean Reasons column for display
                if 'Reasons' in preview_df.columns:
                    preview_df['Reasons'] = preview_df['Reasons'].apply(
                        lambda x: '; '.join(x) if isinstance(x, list) else str(x)
                    )
                st.dataframe(preview_df, use_container_width=True)
                
                # Data statistics
                st.subheader("Data Statistics")
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Total Records", len(results_df))
                with col_stat2:
                    if 'PE_Ratio' in results_df.columns:
                        avg_pe = results_df['PE_Ratio'].mean()
                        st.metric("Average P/E Ratio", f"{avg_pe:.2f}" if not pd.isna(avg_pe) else "N/A")
                    else:
                        st.metric("Average P/E Ratio", "N/A")
                with col_stat3:
                    if 'Score' in results_df.columns:
                        avg_score = results_df['Score'].mean()
                        st.metric("Average Score", f"{avg_score:.2f}")
                    else:
                        st.metric("Average Score", "N/A")
                
            else:
                st.warning("No data available for export.")

if __name__ == "__main__":
    main()
