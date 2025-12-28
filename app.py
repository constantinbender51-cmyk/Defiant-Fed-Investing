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
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

BASE_URL_FRED = "https://api.stlouisfed.org/fred/series/observations"
BASE_URL_FINNHUB = "https://finnhub.io/api/v1"

# FETCH LIMITS
STOCK_FETCH_LIMIT = 500  # Set to 50 for testing
TOP_N_SELECTION = 15     # Top/Bottom N stocks to display

# GLOBAL STATE
APP_DATA = None
IS_READY = False

# ==========================================
# 1. DATA COLLECTION ENGINES
# ==========================================
class FredEngine:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_series(self, series_id):
        if not self.api_key: return pd.DataFrame()
        observation_start = (pd.Timestamp.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        params = {
            "series_id": series_id, "api_key": self.api_key,
            "file_type": "json", "observation_start": observation_start,
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
        df_rate = self.fetch_series("FEDFUNDS") # Monthly Interest Rate
        df_bs = self.fetch_series("WALCL")      # Weekly Balance Sheet

        if df_rate.empty or df_bs.empty: return None

        df_rate.rename(columns={'value': 'ir'}, inplace=True)
        df_bs.rename(columns={'value': 'bs'}, inplace=True)

        # Merge Monthly Rate to Weekly Balance Sheet
        df = pd.merge_asof(df_bs, df_rate, on='date', direction='backward').dropna()

        # Rolling Averages: IR (5yr/260wk), BS (1yr/52wk)
        df['ir_avg'] = df['ir'].rolling(window=260, min_periods=50).mean()
        df['bs_avg'] = df['bs'].rolling(window=52, min_periods=20).mean()

        current = df.iloc[-1]
        ir_curr, ir_avg = current['ir'], current['ir_avg']
        bs_curr, bs_avg = current['bs'], current['bs_avg']

        high_rate = ir_curr > ir_avg
        high_bs = bs_curr > bs_avg

        # Regime Logic
        if not high_rate and high_bs:
            regime, desc = "A", "Expansion"
        elif not high_rate and not high_bs:
            regime, desc = "B", "Deflation"
        elif high_rate and high_bs:
            regime, desc = "C", "Inflationary Boom"
        else:
            regime, desc = "D", "Tightening"

        return {
            "regime": regime, "description": desc,
            "date": current['date'].strftime('%Y-%m-%d'),
            "ir_curr": ir_curr, "ir_avg": ir_avg,
            "bs_curr": bs_curr, "bs_avg": bs_avg
        }

class StockEngine:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_sp500_tickers(self):
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            try:
                tables = pd.read_html(StringIO(response.text))
            except ImportError:
                tables = pd.read_html(StringIO(response.text), flavor='html5lib')
            return [t.replace('.', '-') for t in tables[0]['Symbol'].tolist()]
        except Exception as e:
            print(f"Ticker fetch error: {e}")
            return []

    def get_metrics(self, ticker):
        if not self.api_key: return None
        params = {'symbol': ticker, 'metric': 'all', 'token': self.api_key}
        try:
            r = requests.get(f"{BASE_URL_FINNHUB}/stock/metric", params=params)
            if r.status_code == 429:
                time.sleep(30)
                return self.get_metrics(ticker)
            if r.status_code != 200: return None
            
            m = r.json().get('metric', {})
            pe = m.get('peBasicExclExtraTTM')
            margin = m.get('operatingMarginTTM')
            growth = m.get('revenueGrowthQuarterlyYoy')

            return {'Ticker': ticker, 'PE': pe, 'Margin': margin, 'Growth': growth}
        except: return None

# ==========================================
# 2. ANALYSIS LOGIC
# ==========================================
def run_strategy():
    global APP_DATA, IS_READY
    
    if not FRED_API_KEY or not FINNHUB_API_KEY:
        print("System Error: API Keys not found.")
        return

    # 1. Macro Analysis
    fred = FredEngine(FRED_API_KEY)
    econ = fred.determine_regime()
    if not econ: return

    # 2. Micro Analysis
    stock_engine = StockEngine(FINNHUB_API_KEY)
    tickers = stock_engine.get_sp500_tickers()[:STOCK_FETCH_LIMIT]
    
    raw_data = []
    print(f"Sampling {len(tickers)} instruments...")
    for t in tickers:
        m = stock_engine.get_metrics(t)
        if m: raw_data.append(m)
        time.sleep(1.0) # Rate limit compliance

    if not raw_data: return

    df = pd.DataFrame(raw_data)
    required_cols = ['PE', 'Margin', 'Growth']
    df = df.dropna(subset=required_cols)
    df = df[df['PE'] > 0]

    # 3. Z-Score Calculation
    df['ProfitScore'] = df['Margin'] * (1 / df['PE'])
    
    mean_g, std_g = df['Growth'].mean(), df['Growth'].std()
    mean_p, std_p = df['ProfitScore'].mean(), df['ProfitScore'].std()
    
    if std_g == 0: std_g = 1
    if std_p == 0: std_p = 1

    df['z_growth'] = (df['Growth'] - mean_g) / std_g
    df['z_profit'] = (df['ProfitScore'] - mean_p) / std_p

    # 4. Regime Scoring
    regime = econ['regime']
    if regime == 'A':
        df['FinalScore'] = df['z_growth']
        logic_str = "Growth Factor (z_growth)"
    elif regime == 'D':
        df['FinalScore'] = df['z_profit']
        logic_str = "Value Factor (z_profit)"
    else:
        df['FinalScore'] = df['z_growth'] + df['z_profit']
        logic_str = "Balanced (z_growth + z_profit)"

    df = df.sort_values(by='FinalScore', ascending=False)
    
    # Formatting
    df['Growth'] = (df['Growth']).apply(lambda x: f"{x:.4f}")
    df['Margin'] = (df['Margin']).apply(lambda x: f"{x:.4f}")
    df['FinalScore'] = df['FinalScore'].round(4)
    df['z_growth'] = df['z_growth'].round(4)
    df['z_profit'] = df['z_profit'].round(4)

    APP_DATA = {
        "economy": econ,
        "logic_str": logic_str,
        "top_stocks": df.head(TOP_N_SELECTION).to_dict('records'),
        "short_candidates": df.tail(TOP_N_SELECTION).to_dict('records'),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    }
    IS_READY = True

# ==========================================
# 3. SERVER & TEMPLATE
# ==========================================
app = Flask(__name__)

# Minimized, "Academic Paper" style template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Quantitative Assessment Report</title>
    <meta http-equiv="refresh" content="60">
    <style>
        body {
            font-family: "Times New Roman", Times, serif;
            background-color: #FFFFFF;
            color: #000000;
            margin: 40px auto;
            max-width: 900px;
            line-height: 1.4;
        }
        h1 { text-align: center; font-size: 24px; text-transform: uppercase; margin-bottom: 10px; }
        .meta { text-align: center; font-size: 12px; margin-bottom: 40px; }
        h2 { font-size: 16px; text-transform: uppercase; border-bottom: 1px solid #000; padding-bottom: 5px; margin-top: 30px; }
        p, li { font-size: 14px; text-align: justify; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 15px; }
        th, td { border: 1px solid #000; padding: 5px; text-align: right; }
        th { text-align: center; background-color: #f0f0f0; }
        td:first-child { text-align: left; }
        .caption { font-size: 12px; font-style: italic; margin-top: 5px; }
    </style>
</head>
<body>

{% if data %}
    <h1>Quantitative Regime Analysis</h1>
    <div class="meta">
        Generated: {{ data.timestamp }} | Server: PID {{ pid }}
    </div>

    <h2>I. Macroeconomic Abstract</h2>
    <p>
        The system has identified the current economic regime as <strong>Type {{ data.economy.regime }} ({{ data.economy.description }})</strong>.
        This classification is derived from the divergence of current Fed Funds Rate ({{ "%.2f"|format(data.economy.ir_curr) }}%) against its 5-year moving average ({{ "%.2f"|format(data.economy.ir_avg) }}%), 
        and Federal Reserve Total Assets (${{ "{:,.0f}".format(data.economy.bs_curr/1000) }}B) against the 1-year moving average (${{ "{:,.0f}".format(data.economy.bs_avg/1000) }}B).
    </p>

    <h2>II. Methodology</h2>
    <p>
        <strong>Selection Criterion:</strong> {{ data.logic_str }}.<br>
        Standardized Z-Scores are calculated for Revenue Growth (Quarterly YoY) and Profitability (Operating Margin adjusted for Earnings Yield). 
        The ranking methodology adapts dynamically to the regime identified in Section I.
    </p>

    <h2>III. Data: Long Candidates (Top Decile)</h2>
    <table>
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Revenue Growth</th>
                <th>Op Margin</th>
                <th>P/E Ratio</th>
                <th>Z(Growth)</th>
                <th>Z(Profit)</th>
                <th>Composite Score</th>
            </tr>
        </thead>
        <tbody>
            {% for stock in data.top_stocks %}
            <tr>
                <td>{{ stock.Ticker }}</td>
                <td>{{ stock.Growth }}</td>
                <td>{{ stock.Margin }}</td>
                <td>{{ "%.2f"|format(stock.PE) }}</td>
                <td>{{ stock.z_growth }}</td>
                <td>{{ stock.z_profit }}</td>
                <td><strong>{{ stock.FinalScore }}</strong></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="caption">Table 1: Instruments exhibiting highest regime-congruent factor scores.</div>

    <h2>IV. Data: Short Candidates (Bottom Decile)</h2>
    <table>
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Revenue Growth</th>
                <th>Op Margin</th>
                <th>P/E Ratio</th>
                <th>Z(Growth)</th>
                <th>Z(Profit)</th>
                <th>Composite Score</th>
            </tr>
        </thead>
        <tbody>
            {% for stock in data.short_candidates|reverse %}
            <tr>
                <td>{{ stock.Ticker }}</td>
                <td>{{ stock.Growth }}</td>
                <td>{{ stock.Margin }}</td>
                <td>{{ "%.2f"|format(stock.PE) }}</td>
                <td>{{ stock.z_growth }}</td>
                <td>{{ stock.z_profit }}</td>
                <td><strong>{{ stock.FinalScore }}</strong></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    <div class="caption">Table 2: Instruments exhibiting lowest regime-congruent factor scores.</div>

{% else %}
    <h1>System Status: Initializing</h1>
    <hr>
    <p>The analysis engine is currently aggregating data from external endpoints. Please wait. The page will reload automatically upon completion.</p>
    <p><i>Processes active: FRED API, Finnhub API, Pandas Vectorization.</i></p>
{% endif %}

</body>
</html>
"""

@app.route('/')
def report():
    return render_template_string(HTML_TEMPLATE, data=APP_DATA, pid=os.getpid())

if __name__ == "__main__":
    t = threading.Thread(target=run_strategy)
    t.daemon = True
    t.start()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)