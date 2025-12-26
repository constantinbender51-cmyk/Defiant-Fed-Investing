import os
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import logging
import time
import threading
import json
from datetime import datetime, timedelta
from io import StringIO
from flask import Flask, render_template_string, jsonify

# ==========================================
# CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# SIMULATION PARAMETERS
DEV_STOCK_LIMIT = 60  # Set to None for full S&P 500
TOP_N = 15            # Number of Longs and Shorts (Total portfolio = 2 * TOP_N)
BACKTEST_START_DATE = datetime(2025, 10, 8) # Hardcoded start: October 8th, 2025
INITIAL_CAPITAL = 100000.0

# GLOBAL STORAGE FOR SERVER
BACKTEST_RESULTS = None
ECON_REGIME_DATA = None
FETCHED_STOCK_DATA = {} # Ticker -> List of dicts

# ==========================================
# 1. RISK & POSITION MANAGEMENT
# ==========================================
class Position:
    def __init__(self, ticker, entry_price, side='long', weight=0.0):
        self.ticker = ticker
        self.entry_price = entry_price
        self.side = side 
        self.weight = weight 
        
        self.is_active = True
        self.is_trailing = False
        self.frozen_price = None 
        self.extreme_price = entry_price 

    def update(self, current_price):
        if not self.is_active:
            return self.frozen_price

        if self.side == 'long':
            if current_price > self.extreme_price: self.extreme_price = current_price
            if current_price < (self.entry_price * 0.80):
                self.is_active, self.frozen_price = False, self.entry_price * 0.80
                return self.frozen_price
            if not self.is_trailing and current_price > (self.entry_price * 1.20):
                self.is_trailing = True
            if self.is_trailing and current_price < (self.extreme_price * 0.90):
                self.is_active, self.frozen_price = False, self.extreme_price * 0.90
                return self.frozen_price
        elif self.side == 'short':
            if current_price < self.extreme_price: self.extreme_price = current_price
            if current_price > (self.entry_price * 1.20):
                self.is_active, self.frozen_price = False, self.entry_price * 1.20 
                return self.frozen_price
            if not self.is_trailing and current_price < (self.entry_price * 0.80):
                self.is_trailing = True
            if self.is_trailing and current_price > (self.extreme_price * 1.10):
                self.is_active, self.frozen_price = False, self.extreme_price * 1.10
                return self.frozen_price
        return current_price

    def get_pct_return(self, current_price):
        eff_price = self.frozen_price if not self.is_active else current_price
        if self.side == 'long':
            return (eff_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - eff_price) / self.entry_price

# ==========================================
# 2. DATA ENGINES
# ==========================================
class FredHistoricalEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch(self, sid):
        if not self.api_key: return pd.DataFrame()
        try:
            r = requests.get(self.base_url, params={"series_id": sid, "api_key": self.api_key, "file_type": "json"})
            df = pd.DataFrame(r.json()['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df[['date', 'value']].dropna()
        except: return pd.DataFrame()

    def get_regimes(self):
        logger.info("Building Economic Regimes...")
        # Still fetch 9 years to calculate rolling averages correctly
        start = (datetime.now() - timedelta(days=365*9)).strftime('%Y-%m-%d')
        rate = self.fetch("FEDFUNDS").rename(columns={'value': 'ir'})
        bs = self.fetch("WALCL").rename(columns={'value': 'bs'})
        if rate.empty or bs.empty: return pd.DataFrame()
        df = pd.merge_asof(bs.sort_values('date'), rate.sort_values('date'), on='date')
        df = df.set_index('date')
        df['ir_avg'] = df['ir'].rolling(window=260, min_periods=50).mean()
        df['bs_avg'] = df['bs'].rolling(window=52, min_periods=20).mean()
        v = df['ir_avg'].notna() & df['bs_avg'].notna()
        hr = df['ir'] > df['ir_avg']
        hb = df['bs'] > df['bs_avg']
        df['regime'] = "None"
        df.loc[v & (~hr) & hb, 'regime'] = "A"
        df.loc[v & (~hr) & (~hb), 'regime'] = "B"
        df.loc[v & hr & hb, 'regime'] = "C"
        df.loc[v & hr & (~hb), 'regime'] = "D"
        
        # SLICE FOR BACKTEST START: October 8th, 2025
        return df[df.index >= BACKTEST_START_DATE]

class StockEngine:
    def process(self, ticker, dates):
        try:
            s = yf.Ticker(ticker)
            # Fetch enough history for quarterly calculations
            h = s.history(start="2024-01-01", interval="1wk")
            if h.empty: return None
            h = h[['Close']].rename(columns={'Close': 'price'})
            h.index = h.index.tz_localize(None)
            f = s.quarterly_financials.T
            if f.empty: return None
            f.columns = f.columns.str.strip()
            f = f.apply(pd.to_numeric, errors='coerce').sort_index()
            fd = pd.DataFrame(index=f.index)
            if 'Total Revenue' in f.columns: fd['growth'] = f['Total Revenue'].pct_change(periods=4)
            if 'Operating Income' in f.columns and 'Total Revenue' in f.columns:
                fd['margin'] = f['Operating Income'] / f['Total Revenue']
            if 'Basic EPS' in f.columns: fd['eps'] = f['Basic EPS'].rolling(window=4).sum()
            fd.index = pd.to_datetime(fd.index).tz_localize(None)
            df = h.join(fd.reindex(h.index, method='ffill'))
            if 'eps' in df.columns and 'margin' in df.columns:
                df['profit'] = df['margin'] * (df['eps'] / df['price'])
            df['ticker'] = ticker
            return df.reindex(dates, method='ffill')
        except: return None

# ==========================================
# 3. BACKTEST EXECUTION
# ==========================================
def run_backtest():
    global BACKTEST_RESULTS, ECON_REGIME_DATA, FETCHED_STOCK_DATA
    fred = FredHistoricalEngine(FRED_API_KEY)
    econ = fred.get_regimes()
    if econ.empty: 
        logger.error("No Economic Data. Check FRED Key.")
        return
    
    ECON_REGIME_DATA = econ.reset_index()
    ECON_REGIME_DATA['date'] = ECON_REGIME_DATA['date'].dt.strftime('%Y-%m-%d')

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ts = pd.read_html(StringIO(requests.get(url, headers={"User-Agent":"Mozilla"}).text))[0]['Symbol'].tolist()
    ts = [t.replace('.','-') for t in ts][:DEV_STOCK_LIMIT]

    cache = {}
    logger.info(f"Loading {len(ts)} stocks for timeline starting {BACKTEST_START_DATE.date()}...")
    for t in ts:
        d = StockEngine().process(t, econ.index)
        if d is not None: 
            cache[t] = d
            display_df = d.reset_index().copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            FETCHED_STOCK_DATA[t] = display_df.to_dict(orient='records')

    val = INITIAL_CAPITAL
    ps = []
    cur_reg = None
    hist = []
    
    for date in econ.index.sort_values():
        reg = econ.loc[date, 'regime']
        if reg != cur_reg:
            if ps:
                pnl = sum(p.get_pct_return(cache[p.ticker].loc[date, 'price']) * p.weight for p in ps)
                val *= (1 + pnl)
            cur_reg, ps = reg, []
            snaps = []
            for t, d in cache.items():
                try:
                    row = d.loc[date]
                    if pd.notna(row['price']):
                        snaps.append({'t': t, 'p': row['price'], 'g': row.get('growth', 0), 'pr': row.get('profit', 0)})
                except: pass
            sdf = pd.DataFrame(snaps).dropna()
            if sdf.empty: continue
            sdf['zg'] = (sdf['g'] - sdf['g'].mean()) / (sdf['g'].std() + 1e-6)
            sdf['zp'] = (sdf['pr'] - sdf['pr'].mean()) / (sdf['pr'].std() + 1e-6)
            if reg == "A": sdf['score'] = sdf['zg']
            elif reg == "D": sdf['score'] = sdf['zp']
            else: sdf['score'] = sdf['zg'] + sdf['zp']
            sdf = sdf.sort_values('score', ascending=False)
            w = 0.5 / TOP_N
            for _, r in sdf.head(TOP_N).iterrows(): ps.append(Position(r['t'], r['p'], 'long', w))
            for _, r in sdf.tail(TOP_N).iterrows(): ps.append(Position(r['t'], r['p'], 'short', w))

        pnl_now = sum(p.get_pct_return(cache[p.ticker].loc[date, 'price']) * p.weight for p in ps) if ps else 0
        hist.append({'date': date.strftime('%Y-%m-%d'), 'val': round(val * (1 + pnl_now), 2), 'reg': reg})

    BACKTEST_RESULTS = pd.DataFrame(hist)
    logger.info("Backtest Complete.")

# ==========================================
# 4. WEB SERVER
# ==========================================
app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Regime Backtest Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; background: #f5f7f9; color: #333; line-height: 1.5; }
        .container { max-width: 1200px; margin: auto; }
        .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px; }
        h1, h2 { margin-top: 0; color: #1a202c; }
        .stats { display: flex; gap: 20px; margin-bottom: 25px; }
        .stat-card { flex: 1; padding: 20px; background: #edf2f7; border-radius: 8px; }
        .stat-label { font-size: 11px; text-transform: uppercase; color: #718096; letter-spacing: 0.05em; font-weight: bold; }
        .stat-value { font-size: 24px; font-weight: bold; color: #2d3748; }
        
        .tabs { display: flex; gap: 10px; margin-bottom: 15px; border-bottom: 2px solid #edf2f7; padding-bottom: 10px; }
        .tab-btn { padding: 10px 20px; border: none; background: transparent; cursor: pointer; font-weight: bold; color: #718096; border-radius: 6px; }
        .tab-btn.active { background: #3182ce; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { text-align: left; background: #f8fafc; padding: 12px; border-bottom: 2px solid #edf2f7; color: #4a5568; }
        td { padding: 10px 12px; border-bottom: 1px solid #edf2f7; }
        tr:hover { background: #fdfdfd; }
        .scroll-box { max-height: 500px; overflow-y: auto; border: 1px solid #edf2f7; border-radius: 6px; }
        
        select { padding: 8px 12px; border-radius: 6px; border: 1px solid #cbd5e0; background: white; font-size: 14px; margin-bottom: 15px; width: 200px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Regime Strategy Analysis (From Oct 2025)</h1>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-label">Initial Capital</div>
                    <div class="stat-value">${{ "{:,.2f}".format(initial_cap) }}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Final Value</div>
                    <div class="stat-value">${{ "{:,.2f}".format(final_val) }}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Return</div>
                    <div class="stat-value">{{ "{:.2f}%".format(total_ret) }}</div>
                </div>
            </div>
            <canvas id="mainChart" width="800" height="300"></canvas>
        </div>

        <div class="card">
            <div class="tabs">
                <button class="tab-btn active" onclick="showTab('stock-data')">Individual Stock Data</button>
                <button class="tab-btn" onclick="showTab('regime-data')">Economic Regimes</button>
                <button class="tab-btn" onclick="showTab('performance-data')">Performance History</button>
            </div>

            <div id="stock-data" class="tab-content active">
                <h2>Stock Data Explorer</h2>
                <select id="tickerSelect" onchange="updateStockTable()">
                    {% for ticker in tickers %}
                    <option value="{{ ticker }}">{{ ticker }}</option>
                    {% endfor %}
                </select>
                <div class="scroll-box">
                    <table id="stockTable">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Price</th>
                                <th>Rev Growth</th>
                                <th>Op Margin</th>
                                <th>Profit Score</th>
                            </tr>
                        </thead>
                        <tbody id="stockTableBody"></tbody>
                    </table>
                </div>
            </div>

            <div id="regime-data" class="tab-content">
                <h2>Economic Regime History</h2>
                <div class="scroll-box">
                    <table>
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Int Rate (IR)</th>
                                <th>IR 5Y Avg</th>
                                <th>Bal Sheet (BS)</th>
                                <th>BS 1Y Avg</th>
                                <th>Regime</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in econ_data %}
                            <tr>
                                <td>{{ row.date }}</td>
                                <td>{{ "%.4f"|format(row.ir) }}</td>
                                <td>{{ "%.4f"|format(row.ir_avg) }}</td>
                                <td>{{ "{:,.0f}".format(row.bs) }}</td>
                                <td>{{ "{:,.0f}".format(row.bs_avg) }}</td>
                                <td><strong>{{ row.regime }}</strong></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>

            <div id="performance-data" class="tab-content">
                <h2>Strategy PnL History</h2>
                <div class="scroll-box">
                    <table>
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Regime</th>
                                <th>Portfolio Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in chart_data_raw %}
                            <tr>
                                <td>{{ row.date }}</td>
                                <td>{{ row.reg }}</td>
                                <td>${{ "{:,.2f}".format(row.val) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        const chartDataRaw = {{ chart_data_raw | tojson }};
        const stockCache = {{ stock_data | tojson }};

        const ctx = document.getElementById('mainChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartDataRaw.map(d => d.date),
                datasets: [{
                    label: 'Portfolio Value ($)',
                    data: chartDataRaw.map(d => d.val),
                    borderColor: '#3182ce',
                    backgroundColor: 'rgba(49, 130, 206, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: { x: { grid: { display: false } }, y: { beginAtZero: false } }
            }
        });

        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }

        function updateStockTable() {
            const ticker = document.getElementById('tickerSelect').value;
            const data = stockCache[ticker] || [];
            const body = document.getElementById('stockTableBody');
            body.innerHTML = data.map(row => `
                <tr>
                    <td>${row.date}</td>
                    <td>$${(row.price || 0).toFixed(2)}</td>
                    <td>${row.growth ? (row.growth * 100).toFixed(2) + '%' : 'N/A'}</td>
                    <td>${row.margin ? (row.margin * 100).toFixed(2) + '%' : 'N/A'}</td>
                    <td>${row.profit ? row.profit.toFixed(6) : 'N/A'}</td>
                </tr>
            `).join('');
        }
        updateStockTable();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    if BACKTEST_RESULTS is None:
        return """<body style="font-family:sans-serif; text-align:center; padding:50px;">
                    <h2>Backtest in progress...</h2>
                    <p>Starting Oct 2025. Data is limited, results will appear shortly.</p>
                    <script>setTimeout(() => location.reload(), 5000);</script>
                  </body>"""
    
    final_val = BACKTEST_RESULTS['val'].iloc[-1]
    total_ret = ((final_val / INITIAL_CAPITAL) - 1) * 100
    
    return render_template_string(
        DASHBOARD_HTML,
        initial_cap=INITIAL_CAPITAL,
        final_val=final_val,
        total_ret=total_ret,
        chart_data_raw=BACKTEST_RESULTS.to_dict(orient='records'),
        econ_data=ECON_REGIME_DATA.to_dict(orient='records'),
        stock_data=FETCHED_STOCK_DATA,
        tickers=sorted(FETCHED_STOCK_DATA.keys())
    )

def start_server():
    app.run(host='0.0.0.0', port=8080)

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    logger.info("Backtest starting... dashboard will be available at http://localhost:8080")
    run_backtest()
    while True:
        time.sleep(1)

