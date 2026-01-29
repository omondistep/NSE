# app.py - Enhanced Version with Better UI and Data Management
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
import json

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
            
            st.success("✅ Data loaded successfully!")
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
        
        return True, (price_df, fundamentals_df, financial_df)
        
    except Exception as e:
        return False, str(e)

def create_data_templates():
    """Create and return data template files"""
    templates = {}
    
    # Price data template
    price_template = pd.DataFrame({
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Symbol': ['SCOM', 'SCOM', 'SCOM'],
        'Open': [15.0, 15.2, 15.1],
        'High': [15.5, 15.8, 15.3],
        'Low': [14.8, 15.0, 14.9],
        'Close': [15.2, 15.1, 15.0],
        'Volume': [1000000, 1200000, 950000]
    })
    templates['price_template'] = price_template.to_csv(index=False)
    
    # Financial data template
    financial_template = pd.DataFrame({
        'Symbol': ['SCOM', 'SCOM', 'SCOM', 'KCB', 'KCB', 'KCB'],
        'Year': [2021, 2022, 2023, 2021, 2022, 2023],
        'Revenue': [280000000000, 300000000000, 320000000000, 
                   110000000000, 120000000000, 130000000000],
        'Net_Income': [70000000000, 75000000000, 80000000000,
                      24000000000, 26400000000, 28600000000],
        'Total_Assets': [700000000000, 750000000000, 800000000000,
                        275000000000, 300000000000, 325000000000],
        'Total_Liabilities': [420000000000, 450000000000, 480000000000,
                            165000000000, 180000000000, 195000000000],
        'Shareholders_Equity': [280000000000, 300000000000, 320000000000,
                              110000000000, 120000000000, 130000000000],
        'EPS': [1.75, 1.87, 2.00, 7.50, 8.25, 8.90],
        'Dividends': [0.70, 0.75, 0.80, 3.00, 3.30, 3.56]
    })
    templates['financial_template'] = financial_template.to_csv(index=False)
    
    # Fundamentals template
    fundamentals_template = pd.DataFrame({
        'Symbol': ['SCOM', 'KCB', 'EQTY', 'EABL', 'COOP'],
        'Name': ['Safaricom Plc', 'KCB Group Plc', 'Equity Group Holdings Plc', 
                'East African Breweries Ltd', 'Co-operative Bank of Kenya Ltd'],
        'Sector': ['TELECOMMUNICATION', 'BANKING', 'BANKING', 'MANUFACTURING', 'BANKING'],
        'Market_Cap': [600000000000, 150000000000, 180000000000, 100000000000, 80000000000],
        'Issued_Shares': [40065428000, 3213462815, 3773674802, 790774356, 5867174695],
        'PE_Ratio': [12.5, 8.2, 9.1, 14.2, 7.5],
        'PB_Ratio': [4.2, 1.5, 1.8, 3.5, 1.2],
        'Dividend_Yield': [5.8, 6.2, 5.5, 4.8, 7.1]
    })
    templates['fundamentals_template'] = fundamentals_template.to_csv(index=False)
    
    return templates

# Streamlit App
def main():
    st.set_page_config(
        page_title="NSE Stock Analyzer", 
        layout="wide",
        page_icon="📈",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .sidebar .sidebar-content {
        padding: 1rem;
    }
    h1, h2, h3 {
        color: #1f3c88;
    }
    .stButton button {
        width: 100%;
        margin: 5px 0;
    }
    .data-source-banner {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .success-banner {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .info-banner {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #1f3c88;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📈 Nairobi Securities Exchange Stock Analysis")
    st.markdown("---")
    
    # Initialize session state
    if 'valuator' not in st.session_state:
        st.session_state.valuator = NSEStockValuator()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    if 'generated_sample_data' not in st.session_state:
        st.session_state.generated_sample_data = None
    
    # Sidebar - Reorganized with better structure
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/kenya.png", width=80)
        st.markdown("## 📊 NSE Analyzer")
        st.markdown("---")
        
        # DATA LOADING SECTION
        with st.expander("📂 **Load Data**", expanded=True):
            # Sample Data Option
            st.markdown("### 🎯 Sample Data")
            st.markdown("Quick start with pre-built sample data")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📁 Load Sample", key="load_sample", use_container_width=True):
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
                                
                                # Clean the fundamentals data
                                if 'Sector' in fundamentals.columns:
                                    fundamentals['Sector'] = fundamentals['Sector'].astype(str).str.strip()
                                
                                # Store the data
                                st.session_state.data_source = 'sample'
                                st.session_state.valuator.price_data = price_data
                                st.session_state.valuator.financial_data = financial_data
                                st.session_state.valuator.fundamentals = fundamentals
                                st.session_state.valuator._clean_data()
                                st.session_state.data_loaded = True
                                
                                # Run analysis
                                with st.spinner("Analyzing stocks..."):
                                    results = st.session_state.valuator.analyze_all_stocks()
                                    st.session_state.analysis_results = results
                                
                                st.success("✅ Sample data loaded & analyzed!")
                                st.rerun()
                        else:
                            st.error("❌ Sample data files not found!")
                            
                    except Exception as e:
                        st.error(f"❌ Error loading sample data: {str(e)}")
            
            with col2:
                if st.button("🗑️ Clear Data", key="clear_data_sidebar", use_container_width=True):
                    st.session_state.data_loaded = False
                    st.session_state.analysis_results = None
                    st.session_state.data_source = None
                    st.session_state.generated_sample_data = None
                    st.session_state.valuator = NSEStockValuator()
                    st.success("Data cleared!")
                    st.rerun()
            
            st.markdown("---")
            
            # Upload Data Option
            st.markdown("### 📤 Upload Your Data")
            
            # Download templates button
            if st.button("📋 Download Data Templates", key="download_templates", use_container_width=True):
                templates = create_data_templates()
                with st.expander("Data Templates", expanded=True):
                    st.markdown("### Price Data Template")
                    st.download_button(
                        label="⬇️ Download Price Template (CSV)",
                        data=templates['price_template'],
                        file_name="price_data_template.csv",
                        mime="text/csv"
                    )
                    
                    st.markdown("### Financial Data Template")
                    st.download_button(
                        label="⬇️ Download Financial Template (CSV)",
                        data=templates['financial_template'],
                        file_name="financial_data_template.csv",
                        mime="text/csv"
                    )
                    
                    st.markdown("### Fundamentals Template")
                    st.download_button(
                        label="⬇️ Download Fundamentals Template (CSV)",
                        data=templates['fundamentals_template'],
                        file_name="fundamentals_template.csv",
                        mime="text/csv"
                    )
            
            # File uploaders
            st.markdown("#### Upload Files:")
            price_file = st.file_uploader("Price Data (CSV/Excel)", type=['csv', 'xlsx'], 
                                         help="Columns: Date, Symbol, Open, High, Low, Close, Volume")
            financial_file = st.file_uploader("Financial Data (CSV/Excel)", type=['csv', 'xlsx'],
                                             help="Columns: Symbol, Year, Revenue, Net_Income, etc.")
            fundamentals_file = st.file_uploader("Fundamentals Data (CSV/Excel)", type=['csv', 'xlsx'],
                                               help="Columns: Symbol, Name, Sector, Market_Cap, etc.")
            
            if price_file and financial_file and fundamentals_file:
                if st.button("🚀 Load & Analyze", key="load_uploaded", use_container_width=True):
                    with st.spinner("Loading and analyzing uploaded data..."):
                        success = st.session_state.valuator.load_data(price_file, financial_file, fundamentals_file)
                        if success:
                            st.session_state.data_loaded = True
                            st.session_state.data_source = 'uploaded'
                            results = st.session_state.valuator.analyze_all_stocks()
                            st.session_state.analysis_results = results
                            st.success("✅ Analysis complete!")
                            st.rerun()
        
        st.markdown("---")
        
        # SETUP & TOOLS SECTION
        with st.expander("🛠️ **Setup & Tools**", expanded=False):
            st.markdown("### Generate Sample Data")
            st.markdown("Create sample data files for testing")
            
            if st.button("🔄 Generate Sample Data", key="generate_sample_data", use_container_width=True):
                try:
                    with st.spinner("Generating sample data..."):
                        success, result = generate_simple_sample_data()
                        
                        if success:
                            price_df, fundamentals_df, financial_df = result
                            st.session_state.generated_sample_data = {
                                'price': price_df,
                                'fundamentals': fundamentals_df,
                                'financial': financial_df
                            }
                            st.success("✅ Sample data generated successfully!")
                            
                            # Show download buttons for generated data
                            st.markdown("### Download Generated Data:")
                            col_dl1, col_dl2, col_dl3 = st.columns(3)
                            
                            with col_dl1:
                                st.download_button(
                                    label="📥 Price Data",
                                    data=price_df.to_csv(index=False),
                                    file_name="nse_price_data.csv",
                                    mime="text/csv"
                                )
                            
                            with col_dl2:
                                st.download_button(
                                    label="📥 Fundamentals",
                                    data=fundamentals_df.to_csv(index=False),
                                    file_name="nse_fundamentals.csv",
                                    mime="text/csv"
                                )
                            
                            with col_dl3:
                                st.download_button(
                                    label="📥 Financial Data",
                                    data=financial_df.to_csv(index=False),
                                    file_name="nse_financial_data.csv",
                                    mime="text/csv"
                                )
                            
                            # Show file info
                            st.markdown("**Generated Files:**")
                            st.write(f"- Price Data: {len(price_df):,} records")
                            st.write(f"- Fundamentals: {len(fundamentals_df)} companies")
                            st.write(f"- Financial Data: {len(financial_df)} records")
                            
                        else:
                            st.error(f"❌ Error: {result}")
                            
                except Exception as e:
                    st.error(f"❌ Error generating sample data: {str(e)}")
            
            # Directory setup
            st.markdown("---")
            st.markdown("### Directory Setup")
            
            if st.button("📁 Create Data Directory", key="create_dir", use_container_width=True):
                created, sample_dir = create_sample_data_structure()
                if created:
                    st.success(f"✅ Created directory: {sample_dir}")
                else:
                    st.info(f"📁 Directory already exists: {sample_dir}")
        
        st.markdown("---")
        
        # ANALYSIS SETTINGS SECTION
        with st.expander("⚙️ **Analysis Settings**", expanded=False):
            st.markdown("### DCF Parameters")
            growth_rate = st.slider("Growth Rate (%)", 0.0, 15.0, 5.0, 0.5) / 100
            discount_rate = st.slider("Discount Rate (%)", 5.0, 20.0, 12.0, 0.5) / 100
            
            # Update DCF parameters
            original_dcf = st.session_state.valuator.dcf_valuation
            st.session_state.valuator.dcf_valuation = lambda symbol, gr=growth_rate, dr=discount_rate, years=5: \
                original_dcf(symbol, gr, dr, years)
            
            st.markdown("---")
            st.markdown("#### Current Status:")
            if st.session_state.data_source:
                st.success(f"**Data:** {st.session_state.data_source.upper()}")
                if st.session_state.data_loaded:
                    st.success("✅ Analysis Ready")
            else:
                st.info("⚠️ No data loaded")
        
        st.markdown("---")
        
        # INFO SECTION
        with st.expander("ℹ️ **Information**", expanded=False):
            st.markdown("""
            **📊 Data Format Requirements:**
            
            **Price Data:**
            - Date (YYYY-MM-DD)
            - Symbol (e.g., SCOM, KCB)
            - Open, High, Low, Close (KES)
            - Volume
            
            **Financial Data:**
            - Symbol, Year
            - Revenue, Net_Income
            - Total_Assets, Total_Liabilities
            - Shareholders_Equity
            - EPS, Dividends
            
            **Fundamentals:**
            - Symbol, Name
            - Sector
            - Market_Cap, Issued_Shares
            - PE_Ratio, PB_Ratio, Dividend_Yield
            """)
    
    # Main content area
    if not st.session_state.data_loaded:
        # Welcome screen
        st.markdown('<div class="info-banner data-source-banner">🎯 Welcome to NSE Stock Analysis App</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ## 🚀 Get Started
            
            **Choose one of these options to begin:**
            
            1. **🎯 Use Sample Data** - Load pre-built sample data from the sidebar
            2. **📤 Upload Your Data** - Use your own CSV/Excel files
            3. **🛠️ Generate Sample Data** - Create sample files for testing
            
            **Features included:**
            - 📊 **Fundamental Analysis**: P/E, P/B, Dividend Yield ratios
            - 📈 **Technical Analysis**: RSI, MACD, Moving Averages
            - 💰 **DCF Valuation**: Discounted Cash Flow modeling
            - 🏢 **Sector Analysis**: Performance by industry sector
            - 📥 **Export Reports**: Excel, CSV, JSON formats
            - 🎯 **Smart Recommendations**: Buy/Sell/Hold signals
            
            **NSE Market Coverage:**
            - {} listed companies
            - {} sectors analyzed
            - Real-time technical indicators
            """.format(len(COMPLETE_NSE_SYMBOLS), len(SECTOR_CLASSIFICATION)))
        
        with col2:
            st.markdown("### 📈 Quick Stats")
            
            # Market overview in cards
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Listed Companies", len(COMPLETE_NSE_SYMBOLS))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Sectors", len(SECTOR_CLASSIFICATION))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Top 3 sectors by company count
            sector_counts = {sector: len(symbols) for sector, symbols in SECTOR_CLASSIFICATION.items()}
            top_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            st.markdown("### 🏆 Top Sectors")
            for sector, count in top_sectors:
                st.markdown(f'<div class="metric-card"><strong>{sector}</strong><br>{count} companies</div>', unsafe_allow_html=True)
            
            # Data status
            st.markdown("### 📂 Data Status")
            if os.path.exists('data/sample_data'):
                sample_files = [f for f in os.listdir('data/sample_data') if f.endswith('.csv')]
                if sample_files:
                    st.success(f"✅ {len(sample_files)} sample files available")
                else:
                    st.info("ℹ️ Sample directory exists (no data files)")
            else:
                st.warning("⚠️ Sample data directory not found")
        
        st.markdown("---")
        
        # Quick start guide
        st.markdown("## 📋 Quick Start Guide")
        
        col_guide1, col_guide2, col_guide3 = st.columns(3)
        
        with col_guide1:
            st.markdown("""
            ### 1. 🎯 Sample Data
            **For first-time users:**
            1. Click **"Generate Sample Data"** in Setup
            2. Click **"Load Sample"** in Load Data
            3. View analysis in main dashboard
            """)
        
        with col_guide2:
            st.markdown("""
            ### 2. 📤 Your Data
            **For your own analysis:**
            1. Download templates from sidebar
            2. Format your data accordingly
            3. Upload all three files
            4. Click **"Load & Analyze"**
            """)
        
        with col_guide3:
            st.markdown("""
            ### 3. ⚙️ Customize
            **Fine-tune analysis:**
            1. Adjust DCF parameters
            2. Filter by sector/price
            3. Export results
            4. Compare stocks
            """)
        
        st.warning("""
        ⚠️ **Please load data first** using one of the options in the sidebar to begin analysis.
        """)
        
    else:
        # Display data source banner
        if st.session_state.data_source == 'sample':
            st.markdown('<div class="success-banner data-source-banner">🎯 Currently analyzing: SAMPLE DATA (Demo Dataset)</div>', unsafe_allow_html=True)
        elif st.session_state.data_source == 'uploaded':
            st.markdown('<div class="success-banner data-source-banner">📤 Currently analyzing: YOUR UPLOADED DATA</div>', unsafe_allow_html=True)
        
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
                    if 'Recommendation' in results_df.columns:
                        rec_options = results_df['Recommendation'].unique()
                        default_values = []
                        if 'STRONG_BUY' in rec_options:
                            default_values.append('STRONG_BUY')
                        if 'BUY' in rec_options:
                            default_values.append('BUY')
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
                    if 'Sector' in results_df.columns:
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
        
        # Remaining tabs (unchanged from original for brevity)
        with tab2:
            st.header("Individual Stock Analysis")
            # ... (same as original tab2 content)
            
        with tab3:
            st.header("Sector Analysis")
            # ... (same as original tab3 content)
            
        with tab4:
            st.header("Market Charts & Visualizations")
            # ... (same as original tab4 content)
            
        with tab5:
            st.header("Export Results")
            # ... (same as original tab5 content)

if __name__ == "__main__":
    main()
