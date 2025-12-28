import os
import time
import threading
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template_string
from io import StringIO
from datetime import timedelta

# ==========================================
# CONFIGURATION
# ==========================================
# API KEYS (Set these in your environment)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

BASE_URL_FRED = "https://api.stlouisfed.org/fred/series/observations"
BASE_URL_FINNHUB = "https://finnhub.io/api/v1"

# FETCH LIMITS
# Set to 500 for full production run. Set to 50 for quick testing.
STOCK_FETCH_LIMIT = 500
TOP_N_SELECTION = 15  # Top 15 Longs / Bottom 15 Shorts

# GLOBAL STATE
APP_DATA = None
IS_READY = False

# ==========================================
# 1. FRED ENGINE (Regime A/B/C/D)
# ==========================================
class FredEngine:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_series(self, series_id):
        """Fetches historical data for a series to calculate long-term averages."""
        if not self.api_key: return pd.DataFrame()
        
        # Fetch 10 years to ensure we have enough for the 5-year (260 week) moving average
        observation_start = (pd.Timestamp.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        
        params = {
            "series_id": series_id, 
            "api_key": self.api_key,
            "file_type": "json", 
            "observation_start": observation_start,
            "sort_order": "asc"
        }
        try:
            response = requests.get(BASE_URL_FRED, params=params)
            response.raise_for_status()
            data = response.json().get("observations", [])
            
            df = pd.DataFrame(data)
            if df.empty: return pd.DataFrame()

            df = df[df['value'] != '.'] 
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'])
            return df[['date', 'value']].sort_values('date')
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            return pd.DataFrame()

    def determine_regime(self):
        print("📊 Analyzing Economic Regimes (A/B/C/D)...")
        # 1. Fetch Data
        df_rate = self.fetch_series("FEDFUNDS") # Monthly
        df_bs = self.fetch_series("WALCL")      # Weekly

        if df_rate.empty or df_bs.empty: return None

        df_rate.rename(columns={'value': 'ir'}, inplace=True)
        df_bs.rename(columns={'value': 'bs'}, inplace=True)

        # 2. Merge (Align Monthly Rate to Weekly Balance Sheet)
        df = pd.merge_asof(df_bs, df_rate, on='date', direction='backward').dropna()

        # 3. Calculate Rolling Averages (Logic from uploaded file)
        # Interest Rate: 5 Year Average (approx 260 weeks)
        df['ir_avg'] = df['ir'].rolling(window=260, min_periods=50).mean()
        # Balance Sheet: 1 Year Average (52 weeks)
        df['bs_avg'] = df['bs'].rolling(window=52, min_periods=20).mean()

        # 4. Get Current State (Latest row)
        current = df.iloc[-1]
        
        ir_curr, ir_avg = current['ir'], current['ir_avg']
        bs_curr, bs_avg = current['bs'], current['bs_avg']

        # 5. Determine Regime
        # Logic:
        # A: Rates LOW (vs Avg) AND Liquidity HIGH (vs Avg)
        # B: Rates LOW (vs Avg) AND Liquidity LOW (vs Avg)
        # C: Rates HIGH (vs Avg) AND Liquidity HIGH (vs Avg)
        # D: Rates HIGH (vs Avg) AND Liquidity LOW (vs Avg)
        
        high_rate = ir_curr > ir_avg
        high_bs = bs_curr > bs_avg

        if not high_rate and high_bs:
            regime, desc = "A", "Expansion (Low Rates, Rising Liquidity)"
        elif not high_rate and not high_bs:
            regime, desc = "B", "Deflation/Slow (Low Rates, Falling Liquidity)"
        elif high_rate and high_bs:
            regime, desc = "C", "Inflationary Boom (High Rates, Rising Liquidity)"
        else: # high_rate and not high_bs
            regime, desc = "D", "Tightening/Risk-Off (High Rates, Falling Liquidity)"

        return {
            "regime": regime,
            "description": desc,
            "date": current['date'].strftime('%Y-%m-%d'),
            "ir_curr": ir_curr, "ir_avg": ir_avg,
            "bs_curr": bs_curr, "bs_avg": bs_avg
        }

# ==========================================
# 2. STOCK ENGINE (Finnhub)
# ==========================================
class StockEngine:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_sp500_tickers(self):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            # Try multiple parsers to handle missing lxml dependency
            try:
                tables = pd.read_html(StringIO(response.text))
            except ImportError:
                # Fallback to html5lib if lxml is not installed
                tables = pd.read_html(StringIO(response.text), flavor='html5lib')
            tickers = [t.replace('.', '-') for t in tables[0]['Symbol'].tolist()]
            return tickers
        except Exception as e:
            print(f"Error getting tickers: {e}")
            return []

    def get_metrics(self, ticker):
        if not self.api_key: return None
        params = {'symbol': ticker, 'metric': 'all', 'token': self.api_key}
        
        try:
            r = requests.get(f"{BASE_URL_FINNHUB}/stock/metric", params=params)
            
            if r.status_code == 429:
                print(f"⚠️ Rate limit {ticker}. Sleeping 30s...")
                time.sleep(30) 
                return self.get_metrics(ticker)

            if r.status_code != 200: return None
            d = r.json()
            m = d.get('metric', {})
            
            # MAPPING LOGIC:
            # We need Growth and Profitability to match the backtest logic.
            # Backtest 'g': Revenue Growth
            # Backtest 'pr': Operating Margin * (1 / PE) (which is Margin * EarningsYield)
            
            pe = m.get('peBasicExclExtraTTM')
            margin = m.get('operatingMarginTTM')
            growth = m.get('revenueGrowthQuarterlyYoy') # Using Quarterly YoY to catch recent trends
            
            return {
                'Ticker': ticker,
                'Price': m.get('52WeekHigh'), # Approx, Finnhub free tier limit
                'PE': pe,
                'Margin': margin,
                'Growth': growth
            }
        except: return None

# ==========================================
# 3. ANALYSIS LOGIC (Z-Score + Regimes)
# ==========================================
def run_strategy():
    global APP_DATA, IS_READY
    print("🚀 Starting Z-Score Strategy Analysis...")
    
    if not FRED_API_KEY or not FINNHUB_API_KEY:
        print("❌ API Keys missing.")
        return

    # --- STEP 1: GET REGIME ---
    fred = FredEngine(FRED_API_KEY)
    econ = fred.determine_regime()
    if not econ: 
        print("❌ Failed to determine economic regime.")
        return
    print(f"🌍 Current Regime: {econ['regime']} ({econ['description']})")

    # --- STEP 2: GET STOCK DATA ---
    stock_engine = StockEngine(FINNHUB_API_KEY)
    tickers = stock_engine.get_sp500_tickers()
    
    if STOCK_FETCH_LIMIT:
        tickers = tickers[:STOCK_FETCH_LIMIT]

    raw_data = []
    print(f"📥 Fetching data for {len(tickers)} stocks...")
    
    for i, t in enumerate(tickers):
        if i % 25 == 0: print(f"Processing {i}/{len(tickers)}...")
        m = stock_engine.get_metrics(t)
        if m: raw_data.append(m)
        time.sleep(1.1) # Respect API limits

    # [FIX] Check for empty data before creating DataFrame
    if not raw_data:
        print("❌ No stock data fetched. Check API Limit or Key.")
        return

    df = pd.DataFrame(raw_data)

    # --- STEP 3: CALCULATE METRICS (Mapping Logic) ---
    # Ensure required columns exist
    required_cols = ['PE', 'Margin', 'Growth']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Missing column: {col}. Data frame is malformed.")
            return

    # Clean data: drop rows where essential metrics are missing
    df = df.dropna(subset=required_cols)
    df = df[df['PE'] > 0] # Filter out negative earnings for PE calc validity

    if df.empty:
        print("❌ No valid stocks remaining after filtering (PE > 0 and non-null metrics).")
        return

    # Calculate 'pr' (Profitability Score)
    # Logic: Operating Margin * (1 / PE)
    # This rewards high margins bought at a cheap price.
    df['ProfitScore'] = df['Margin'] * (1 / df['PE'])
    
    # Growth is already 'Growth'
    
    # --- STEP 4: Z-SCORE STANDARDIZATION ---
    # Z = (Value - Mean) / StdDev
    # This normalizes the data so we can compare Growth vs Profitability apples-to-apples
    
    mean_g, std_g = df['Growth'].mean(), df['Growth'].std()
    mean_p, std_p = df['ProfitScore'].mean(), df['ProfitScore'].std()
    
    # Avoid division by zero
    if std_g == 0: std_g = 1
    if std_p == 0: std_p = 1

    df['z_growth'] = (df['Growth'] - mean_g) / std_g
    df['z_profit'] = (df['ProfitScore'] - mean_p) / std_p

    # --- STEP 5: REGIME-BASED SCORING ---
    regime = econ['regime']
    
    if regime == 'A':
        # Expansion: Growth is King
        df['FinalScore'] = df['z_growth']
        strategy_note = "Regime A: Prioritizing pure Growth (z_growth)."
        
    elif regime == 'D':
        # Tightening: Profitability/Value is King
        df['FinalScore'] = df['z_profit']
        strategy_note = "Regime D: Prioritizing Profitability & Value (z_profit)."
        
    else: # B or C
        # Mixed/Transitional: Balanced Approach
        df['FinalScore'] = df['z_growth'] + df['z_profit']
        strategy_note = f"Regime {regime}: Balanced Growth + Profitability."

    df = df.sort_values(by='FinalScore', ascending=False)
    
    # Format for display
    df['Growth'] = (df['Growth']).apply(lambda x: f"{x:.2f}%")
    df['Margin'] = (df['Margin']).apply(lambda x: f"{x:.2f}%")
    df['FinalScore'] = df['FinalScore'].round(2)
    df['z_growth'] = df['z_growth'].round(2)
    df['z_profit'] = df['z_profit'].round(2)

    top_n = df.head(TOP_N_SELECTION).to_dict('records')
    bottom_n = df.tail(TOP_N_SELECTION).to_dict('records')

    APP_DATA = {
        "economy": econ,
        "strategy_note": strategy_note,
        "top_stocks": top_n,
        "short_candidates": bottom_n,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    IS_READY = True
    print("✅ Analysis Complete.")

# ==========================================
# 4. FLASK DASHBOARD
# ==========================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Regime Z-Score Dashboard</title>
    <meta http-equiv="refresh" content="60">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 20px; color: #333; }
        .container { max-width: 1100px; margin: 0 auto; }
        
        .header { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
        .regime-badge { background: #333; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; font-size: 1.2em; }
        .regime-desc { color: #666; margin-top: 5px; font-size: 0.9em; }

        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        
        h2 { border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 0; font-size: 1.1em; text-transform: uppercase; letter-spacing: 1px; }
        .long { border-top: 4px solid #48bb78; }
        .short { border-top: 4px solid #f56565; }

        table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
        th { text-align: left; color: #888; padding: 8px; border-bottom: 1px solid #eee; }
        td { padding: 8px; border-bottom: 1px solid #f9f9f9; }
        tr:last-child td { border-bottom: none; }
        
        .score { font-weight: bold; }
        .pos { color: #2f855a; }
        .neg { color: #c53030; }
        
        .loading { text-align: center; padding: 50px; }
    </style>
</head>
<body>
    <div class="container">
        {% if data %}
            <div class="header">
                <div>
                    <h1>Z-Score Strategy Dashboard</h1>
                    <div class="regime-desc">{{ data.strategy_note }}</div>
                    <small>Last Updated: {{ data.timestamp }}</small>
                </div>
                <div style="text-align: right;">
                    <div class="regime-badge">REGIME {{ data.economy.regime }}</div>
                    <div style="font-size: 0.8em; margin-top:5px;">
                        Rate: {{ "%.2f"|format(data.economy.ir_curr) }}% (Avg {{ "%.2f"|format(data.economy.ir_avg) }})<br>
                        Liquidity: ${{ "{:,.0f}".format(data.economy.bs_curr/1000) }}B (Avg ${{ "{:,.0f}".format(data.economy.bs_avg/1000) }}B)
                    </div>
                </div>
            </div>

            <div class="grid">
                <!-- LONG CANDIDATES -->
                <div class="card long">
                    <h2>🚀 Top {{ data.top_stocks|length }} Longs (Highest Score)</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Ticker</th>
                                <th>Growth</th>
                                <th>Margin</th>
                                <th>PE</th>
                                <th>Z-Grow</th>
                                <th>Z-Prof</th>
                                <th>Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for stock in data.top_stocks %}
                            <tr>
                                <td><strong>{{ stock.Ticker }}</strong></td>
                                <td>{{ stock.Growth }}</td>
                                <td>{{ stock.Margin }}</td>
                                <td>{{ "%.1f"|format(stock.PE) }}</td>
                                <td>{{ stock.z_growth }}</td>
                                <td>{{ stock.z_profit }}</td>
                                <td class="score pos">{{ stock.FinalScore }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>

                <!-- SHORT CANDIDATES -->
                <div class="card short">
                    <h2>🔻 Top {{ data.short_candidates|length }} Shorts (Lowest Score)</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Ticker</th>
                                <th>Growth</th>
                                <th>Margin</th>
                                <th>PE</th>
                                <th>Z-Grow</th>
                                <th>Z-Prof</th>
                                <th>Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for stock in data.short_candidates|reverse %}
                            <tr>
                                <td><strong>{{ stock.Ticker }}</strong></td>
                                <td>{{ stock.Growth }}</td>
                                <td>{{ stock.Margin }}</td>
                                <td>{{ "%.1f"|format(stock.PE) }}</td>
                                <td>{{ stock.z_growth }}</td>
                                <td>{{ stock.z_profit }}</td>
                                <td class="score neg">{{ stock.FinalScore }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        {% else %}
            <div class="card loading">
                <h1>Running Analysis...</h1>
                <p>Fetching S&P 500 Metrics & Calculating Z-Scores.</p>
                <p>Please wait ~5-10 minutes for full API fetch.</p>
                <p><strong>Page will auto-refresh.</strong></p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE, data=APP_DATA)

if __name__ == "__main__":
    t = threading.Thread(target=run_strategy)
    t.daemon = True
    t.start()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)