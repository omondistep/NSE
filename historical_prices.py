import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import json
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NSEHistoricalPriceExtractor:
    def __init__(self, tickers):
        self.tickers = tickers
        self.base_url = "https://stockanalysis.com/quote/nase"
        
    def create_output_dir(self):
        if not os.path.exists('nse_data'):
            os.makedirs('nse_data')
    
    def extract_historical_prices(self, ticker):
        """
        Simple and reliable historical price extraction
        """
        print(f"\n{'='*40}")
        print(f"Extracting historical prices for {ticker}")
        print(f"{'='*40}")
        
        url = f"{self.base_url}/{ticker}/history/"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            print(f"Fetching: {url}")
            response = requests.get(url, headers=headers, timeout=30)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ HTTP Error {response.status_code}")
                return pd.DataFrame()
            
            # Save HTML for debugging
            html_file = f"nse_data/{ticker}_page.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"📄 HTML saved: {html_file}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for ALL tables on the page
            tables = soup.find_all('table')
            print(f"Found {len(tables)} table(s)")
            
            if not tables:
                print("❌ No tables found on page")
                return pd.DataFrame()
            
            # Try each table
            for i, table in enumerate(tables):
                try:
                    print(f"\nTrying table {i+1}...")
                    
                    # Get table HTML
                    table_html = str(table)
                    
                    # Save table HTML for inspection
                    table_file = f"nse_data/{ticker}_table_{i+1}.html"
                    with open(table_file, 'w', encoding='utf-8') as f:
                        f.write(table_html)
                    
                    # Try to read with pandas
                    try:
                        df_list = pd.read_html(table_html)
                        if df_list:
                            df = df_list[0]
                            print(f"✅ Pandas extracted table {i+1}: {df.shape}")
                            print(f"Columns: {list(df.columns)}")
                            
                            # Check if it looks like price data
                            if self.is_price_data(df):
                                print(f"✅ This looks like price data!")
                                return self.clean_and_format(df, ticker)
                            else:
                                print(f"❌ Table {i+1} doesn't look like price data")
                                print("First few rows:")
                                print(df.head(3).to_string())
                    except Exception as e:
                        print(f"❌ Pandas failed on table {i+1}: {e}")
                        
                        # Try manual extraction
                        print("Trying manual extraction...")
                        manual_df = self.extract_table_manually(table)
                        if not manual_df.empty and self.is_price_data(manual_df):
                            print(f"✅ Manual extraction successful!")
                            return self.clean_and_format(manual_df, ticker)
                        
                except Exception as e:
                    print(f"❌ Error with table {i+1}: {e}")
                    continue
            
            print(f"❌ No usable price data found in any table")
            
            # Try to find data in page text as last resort
            print("\nTrying to find data in page text...")
            page_text = soup.get_text()
            
            # Look for date patterns
            date_pattern = r'(\w+\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})'
            dates = re.findall(date_pattern, page_text)
            
            if len(dates) > 10:
                print(f"Found {len(dates)} date patterns in text")
                
                # Look for price patterns
                lines = page_text.split('\n')
                data_rows = []
                
                for line in lines:
                    line = line.strip()
                    # Check if line contains a date
                    date_match = re.search(date_pattern, line)
                    if date_match:
                        # Extract numbers from the line
                        numbers = re.findall(r'\d+\.\d+|\d+,\d+', line)
                        if len(numbers) >= 4:  # Need at least 4 price numbers
                            row = [date_match.group()] + numbers[:5]  # Date + up to 5 numbers
                            data_rows.append(row)
                
                if len(data_rows) > 5:
                    print(f"Extracted {len(data_rows)} data rows from text")
                    # Create DataFrame
                    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'][:len(data_rows[0])]
                    df = pd.DataFrame(data_rows, columns=columns)
                    return self.clean_and_format(df, ticker)
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def extract_table_manually(self, table):
        """Manually extract table data"""
        try:
            data = []
            headers = []
            
            # Extract headers from first row
            rows = table.find_all('tr')
            if not rows:
                return pd.DataFrame()
            
            # First row for headers
            first_row = rows[0]
            for cell in first_row.find_all(['th', 'td']):
                headers.append(cell.get_text(strip=True))
            
            # Extract data from remaining rows
            for row in rows[1:]:
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    row_data.append(cell.get_text(strip=True))
                
                if row_data:
                    data.append(row_data)
            
            # Create DataFrame
            if data:
                if headers:
                    # Ensure headers match data columns
                    if len(headers) != len(data[0]):
                        headers = [f"Column_{i}" for i in range(len(data[0]))]
                    df = pd.DataFrame(data, columns=headers)
                else:
                    df = pd.DataFrame(data)
                
                return df
            
        except Exception as e:
            print(f"Manual extraction error: {e}")
        
        return pd.DataFrame()
    
    def is_price_data(self, df):
        """Check if dataframe contains price data"""
        if df.empty or len(df) < 2:
            return False
        
        # Convert column names to strings
        df.columns = [str(col) for col in df.columns]
        
        # Check column names for price-related terms
        col_names = [str(col).lower() for col in df.columns]
        
        price_terms = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj', 'change']
        
        # Count matching terms
        matches = 0
        for term in price_terms:
            for col in col_names:
                if term in col:
                    matches += 1
                    break
        
        # Need at least 3 matching terms (Date + 2 price columns)
        if matches >= 3:
            return True
        
        # Check data types and values
        try:
            # Look for date-like column
            for col in df.columns:
                sample = df[col].dropna().head(3).tolist()
                for val in sample:
                    val_str = str(val)
                    # Check if value looks like a date
                    if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', val_str) or \
                       re.search(r'\w+\s+\d{1,2},\s+\d{4}', val_str):
                        # Found date column, check for numeric columns
                        numeric_cols = 0
                        for num_col in df.columns:
                            if num_col != col:
                                try:
                                    # Try to convert to numeric
                                    pd.to_numeric(df[num_col].astype(str).str.replace(',', ''), errors='coerce')
                                    numeric_cols += 1
                                except:
                                    pass
                        
                        if numeric_cols >= 3:  # Open, High, Low/Close
                            return True
                        break
        except:
            pass
        
        return False
    
    def clean_and_format(self, df, ticker):
        """Clean and format the extracted dataframe"""
        try:
            # Ensure column names are strings
            df.columns = [str(col).strip() for col in df.columns]
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Reset index
            df = df.reset_index(drop=True)
            
            # Add metadata
            df['Ticker'] = ticker
            df['Extraction_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"✅ Successfully extracted {len(df)} rows")
            return df
            
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return df
    
    def save_data(self, ticker, df):
        """Save data to both Excel and CSV"""
        if df.empty:
            print(f"⚠ No data to save for {ticker}")
            return False
        
        try:
            # Save to Excel
            excel_file = f"nse_data/{ticker}_historical_prices.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Historical_Prices', index=False)
            print(f"💾 Excel saved: {excel_file}")
            
            # Save to CSV
            csv_file = f"nse_data/{ticker}_historical_prices.csv"
            df.to_csv(csv_file, index=False)
            print(f"💾 CSV saved: {csv_file}")
            
            # Show file info
            file_size = os.path.getsize(excel_file)
            print(f"📊 File size: {file_size:,} bytes")
            
            return True
            
        except Exception as e:
            print(f"❌ Save error: {e}")
            return False
    
    def full_download(self, ticker):
        """Full download"""
        print(f"\n{'='*60}")
        print(f"FULL DOWNLOAD: {ticker}")
        print(f"{'='*60}")
        
        price_df = self.extract_historical_prices(ticker)
        
        saved = False
        if not price_df.empty:
            saved = self.save_data(ticker, price_df)
            
            if saved:
                print(f"\n📊 Sample of downloaded data ({len(price_df)} rows total):")
                print(price_df.head().to_string())
        
        return {
            'ticker': ticker,
            'success': saved,
            'rows_extracted': len(price_df) if not price_df.empty else 0,
            'data': price_df
        }
    
    def smart_update(self, ticker):
        """Smart update - check existing and add new data only"""
        print(f"\n{'='*60}")
        print(f"SMART UPDATE: {ticker}")
        print(f"{'='*60}")
        
        # Check for existing data
        existing_file = f"nse_data/{ticker}_historical_prices.xlsx"
        existing_csv = f"nse_data/{ticker}_historical_prices.csv"
        
        if not os.path.exists(existing_file) and not os.path.exists(existing_csv):
            print(f"⚠ No existing data found. Doing full download...")
            return self.full_download(ticker)
        
        print(f"✅ Existing data file found")
        print("Smart update logic would go here...")
        print("For now, doing full download...")
        
        return self.full_download(ticker)

# ============================================================================
# SIMPLE TEST FUNCTION
# ============================================================================

def test_single_ticker():
    """Test with a single ticker"""
    print("="*60)
    print("TEST SINGLE TICKER EXTRACTION")
    print("="*60)
    
    ticker = input("\nEnter ticker to test (e.g., SCOM): ").strip().upper()
    if not ticker:
        ticker = "SCOM"
    
    print(f"\nTesting {ticker}...")
    
    # Create output directory
    if not os.path.exists('nse_data'):
        os.makedirs('nse_data')
    
    extractor = NSEHistoricalPriceExtractor([ticker])
    result = extractor.full_download(ticker)
    
    if result['success']:
        print(f"\n🎉 SUCCESS: Downloaded {result['rows_extracted']} rows")
        print(f"Files saved in 'nse_data/' folder:")
        print(f"  - {ticker}_historical_prices.xlsx")
        print(f"  - {ticker}_historical_prices.csv")
        print(f"  - {ticker}_page.html (debug)")
    else:
        print(f"\n❌ FAILED: Could not extract data")
        print(f"\nCheck these debug files in 'nse_data/' folder:")
        print(f"  - {ticker}_page.html - The full webpage")
        print(f"  - {ticker}_table_*.html - Individual tables found")
    
    return result

# ============================================================================
# BATCH DOWNLOAD FUNCTION
# ============================================================================

def batch_download():
    """Download multiple tickers"""
    print("="*60)
    print("BATCH DOWNLOAD")
    print("="*60)
    
    ticker_input = input("\nEnter tickers separated by comma (e.g., SCOM,TOTL,KEGN): ").strip().upper()
    if not ticker_input:
        ticker_input = "SCOM,TOTL,KEGN"
    
    tickers = [t.strip() for t in ticker_input.split(',')]
    
    print(f"\nDownloading {len(tickers)} ticker(s): {', '.join(tickers)}")
    
    # Create output directory
    if not os.path.exists('nse_data'):
        os.makedirs('nse_data')
    
    extractor = NSEHistoricalPriceExtractor(tickers)
    
    results = []
    for ticker in tickers:
        print(f"\n{'='*40}")
        result = extractor.full_download(ticker)
        results.append(result)
        
        # Be nice to the server
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in results if r['success'])
    total_rows = sum(r['rows_extracted'] for r in results)
    
    print(f"Successful downloads: {success_count}/{len(tickers)}")
    print(f"Total rows extracted: {total_rows}")
    
    print(f"\nDetails:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {result['ticker']:6} - {result['rows_extracted']} rows")
    
    return results

# ============================================================================
# DEBUG FUNCTION
# ============================================================================

def debug_ticker_page():
    """Debug a specific ticker to see page structure"""
    print("="*60)
    print("DEBUG TICKER PAGE")
    print("="*60)
    
    ticker = input("\nEnter ticker to debug (e.g., TOTL): ").strip().upper()
    if not ticker:
        ticker = "TOTL"
    
    url = f"https://stockanalysis.com/quote/nase/{ticker}/history/"
    
    print(f"\nDebugging: {ticker}")
    print(f"URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            # Create output directory
            if not os.path.exists('nse_data'):
                os.makedirs('nse_data')
            
            # Save raw HTML
            html_file = f"nse_data/{ticker}_debug.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"\n✅ HTML saved: {html_file}")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check page title
            title = soup.find('title')
            if title:
                print(f"Page title: {title.text}")
            
            # Count tables
            tables = soup.find_all('table')
            print(f"\nFound {len(tables)} table(s)")
            
            # Analyze each table
            for i, table in enumerate(tables):
                print(f"\n--- Table {i+1} ---")
                print(f"Classes: {table.get('class', 'None')}")
                print(f"ID: {table.get('id', 'None')}")
                
                # Count rows and columns
                rows = table.find_all('tr')
                print(f"Rows: {len(rows)}")
                
                if rows:
                    # Show header row
                    header_cells = rows[0].find_all(['th', 'td'])
                    headers = [cell.get_text(strip=True) for cell in header_cells]
                    print(f"Headers: {headers}")
                    
                    # Show first data row
                    if len(rows) > 1:
                        data_cells = rows[1].find_all(['td', 'th'])
                        data = [cell.get_text(strip=True) for cell in data_cells]
                        print(f"First row: {data}")
            
            # Look for data in page
            print(f"\n--- Page Analysis ---")
            page_text = soup.get_text()
            
            # Look for keywords
            keywords = ['date', 'open', 'high', 'low', 'close', 'volume', 'historical', 'price']
            found = []
            for kw in keywords:
                if kw in page_text.lower():
                    found.append(kw)
            
            print(f"Keywords found: {found}")
            
            # Count date patterns
            dates = re.findall(r'\w+\s+\d{1,2},\s+\d{4}', page_text)
            print(f"Date patterns found: {len(dates)}")
            
            if dates:
                print(f"Sample dates: {dates[:3]}")
            
            print(f"\n✅ Open {html_file} in a browser to inspect the page")
            
        else:
            print(f"❌ Failed to fetch page")
            
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Main menu"""
    print("="*60)
    print("NSE HISTORICAL PRICE EXTRACTOR")
    print("="*60)
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Test single ticker extraction")
        print("2. Batch download multiple tickers")
        print("3. Debug ticker page structure")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            test_single_ticker()
        elif choice == "2":
            batch_download()
        elif choice == "3":
            debug_ticker_page()
        elif choice == "4":
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

# ============================================================================
# QUICK TEST SCRIPT
# ============================================================================

def quick_test():
    """Quick test with SCOM"""
    print("="*60)
    print("QUICK TEST WITH SCOM")
    print("="*60)
    
    # Create output directory
    if not os.path.exists('nse_data'):
        os.makedirs('nse_data')
    
    extractor = NSEHistoricalPriceExtractor(['SCOM'])
    result = extractor.full_download('SCOM')
    
    if result['success']:
        print(f"\n✅ Test successful! Downloaded {result['rows_extracted']} rows")
    else:
        print(f"\n❌ Test failed")
    
    return result

# ============================================================================
# RUN PROGRAM
# ============================================================================

if __name__ == "__main__":
    try:
        # You can run quick_test() for a simple test
        # quick_test()
        
        # Or run the full menu
        main()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n{'='*60}")
        print("Program ended")
        print(f"{'='*60}")
