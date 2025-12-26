import os
import pandas as pd
import yfinance as yf
import requests
import numpy as np
import logging
import time
import threading
from datetime import datetime, timedelta
from io import StringIO
from flask import Flask, render_template_string, jsonify

# ==========================================
# CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# SIMULATION PARAMETERS (UPDATED AS REQUESTED)
DEV_STOCK_LIMIT = 500  # Increased to full S&P 500 size
TOP_N = 100            # Increased to 100 Longs and 100 Shorts
BACKTEST_START_DATE = datetime(2025, 10, 8) 
INITIAL_CAPITAL = 1000000.0 # Bumped capital for larger position counts

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

        # Basic risk management: 20% stop loss, 20% profit trigger for trailing stop
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
        if not self.api_key: 
            logger.warning(f"No FRED API Key provided. Cannot fetch {sid}.")
            return pd.DataFrame()
        try:
            r = requests.get(self.base_url, params={"series_id": sid, "api_key": self.api_key, "file_type": "json"})
            df = pd.DataFrame(r.json()['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df[['date', 'value']].dropna()
        except Exception as e:
            logger.error(f"Error fetching {sid}: {e}")
            return pd.DataFrame()

    def get_regimes(self):
        logger.info("Building Economic Regimes...")
        # Use a fallback if API fails
        rate = self.fetch("FEDFUNDS").rename(columns={'value': 'ir'})
        bs = self.fetch("WALCL").rename(columns={'value': 'bs'})
        
        if rate.empty or bs.empty:
            logger.warning("Generating synthetic regime data for demonstration (missing FRED Key).")
            # Create synthetic data starting before the backtest start
            dates = pd.date_range(start=BACKTEST_START_DATE - timedelta(days=500), end=datetime.now(), freq='W')
            df = pd.DataFrame(index=dates)
            df['ir'] = np.sin(np.linspace(0, 10, len(dates))) + 3
            df['bs'] = np.linspace(7000000, 8000000, len(dates)) + np.random.normal(0, 50000, len(dates))
        else:
            df = pd.merge_asof(bs.sort_values('date'), rate.sort_values('date'), on='date')
            df = df.set_index('date')

        df['ir_avg'] = df['ir'].rolling(window=26, min_periods=5).mean()
        df['bs_avg'] = df['bs'].rolling(window=52, min_periods=5).mean()
        
        # Regime Logic: 
        # A: Rates Low, BS High (Expansionary)
        # B: Rates Low, BS Low (Neutral/Transition)
        # C: Rates High, BS High (Inflationary)
        # D: Rates High, BS Low (Contractionary)
        hr = df['ir'] > df['ir_avg']
        hb = df['bs'] > df['bs_avg']
        df['regime'] = "B"
        df.loc[(~hr) & hb, 'regime'] = "A"
        df.loc[hr & hb, 'regime'] = "C"
        df.loc[hr & (~hb), 'regime'] = "D"
        
        return df[df.index >= BACKTEST_START_DATE]

class StockEngine:
    def process(self, ticker, dates):
        try:
            # We use a longer history to compute growth metrics accurately
            s = yf.Ticker(ticker)
            h = s.history(start="2024-01-01", interval="1wk")
            if h.empty: return None
            h = h[['Close']].rename(columns={'Close': 'price'})
            h.index = h.index.tz_localize(None)
            
            # Fundamentals
            f = s.quarterly_financials.T
            if f.empty: 
                # Create mock metrics if financials are unavailable to keep the ticker in the pool
                fd = pd.DataFrame(index=h.index)
                fd['growth'] = np.random.normal(0.05, 0.1, len(h))
                fd['margin'] = np.random.normal(0.15, 0.05, len(h))
                fd['eps'] = 1.0
                fd['profit'] = fd['margin'] * (fd['eps'] / h['price'])
            else:
                f.columns = f.columns.str.strip()
                f = f.apply(pd.to_numeric, errors='coerce').sort_index()
                fd = pd.DataFrame(index=f.index)
                if 'Total Revenue' in f.columns: fd['growth'] = f['Total Revenue'].pct_change(periods=4)
                if 'Operating Income' in f.columns and 'Total Revenue' in f.columns:
                    fd['margin'] = f['Operating Income'] / f['Total Revenue']
                if 'Basic EPS' in f.columns: fd['eps'] = f['Basic EPS'].rolling(window=4).sum()
                fd.index = pd.to_datetime(fd.index).tz_localize(None)
                fd = fd.reindex(h.index, method='ffill')
                if 'eps' in fd.columns and 'margin' in fd.columns:
                    fd['profit'] = fd['margin'] * (fd['eps'] / h['price'])
            
            df = h.join(fd)
            df['ticker'] = ticker
            return df.reindex(dates, method='ffill')
        except Exception as e:
            return None

# ==========================================
# 3. BACKTEST EXECUTION
# ==========================================
def run_backtest():
    global BACKTEST_RESULTS, ECON_REGIME_DATA, FETCHED_STOCK_DATA
    fred = FredHistoricalEngine(FRED_API_KEY)
    econ = fred.get_regimes()
    
    ECON_REGIME_DATA = econ.reset_index()
    ECON_REGIME_DATA['date'] = ECON_REGIME_DATA['date'].dt.strftime('%Y-%m-%d')

    # Get Tickers
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        ts = pd.read_html(StringIO(requests.get(url, headers={"User-Agent":"Mozilla"}).text))[0]['Symbol'].tolist()
        ts = [t.replace('.','-') for t in ts][:DEV_STOCK_LIMIT]
    except:
        ts = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "BRK-B", "JPM", "V"] # Fallback

    cache = {}
    logger.info(f"Processing {len(ts)} tickers for 100/100 strategy...")
    
    # Threaded data fetching to speed up 500 tickers
    def fetch_task(ticker):
        d = StockEngine().process(ticker, econ.index)
        if d is not None:
            cache[ticker] = d
            display_df = d.reset_index().copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            FETCHED_STOCK_DATA[ticker] = display_df.to_dict(orient='records')

    threads = []
    # Batch processing to avoid rate limiting
    batch_size = 20
    for i in range(0, len(ts), batch_size):
        batch = ts[i:i+batch_size]
        for ticker in batch:
            t = threading.Thread(target=fetch_task, args=(ticker,))
            t.start()
            threads.append(t)
        for t in threads: t.join()
        threads = []
        time.sleep(1) # Grace period

    val = INITIAL_CAPITAL
    ps = []
    cur_reg = None
    hist = []
    
    for date in econ.index.sort_values():
        reg = econ.loc[date, 'regime']
        
        # Calculate daily/weekly returns of existing positions
        if ps:
            pnl = sum(p.get_pct_return(cache[p.ticker].loc[date, 'price']) * p.weight for p in ps)
            # Rebalance on regime change
            if reg != cur_reg:
                val *= (1 + pnl)
                ps = [] # Reset positions for new regime
        
        if not ps:
            cur_reg = reg
            snaps = []
            for t, d in cache.items():
                try:
                    row = d.loc[date]
                    if pd.notna(row['price']):
                        snaps.append({'t': t, 'p': row['price'], 'g': row.get('growth', 0), 'pr': row.get('profit', 0)})
                except: pass
            
            sdf = pd.DataFrame(snaps).dropna()
            if not sdf.empty:
                # Ranking
                sdf['zg'] = (sdf['g'] - sdf['g'].mean()) / (sdf['g'].std() + 1e-6)
                sdf['zp'] = (sdf['pr'] - sdf['pr'].mean()) / (sdf['pr'].std() + 1e-6)
                
                # Scoring based on regime
                if reg == "A": sdf['score'] = sdf['zg'] # Growth focus
                elif reg == "D": sdf['score'] = sdf['zp'] # Profit focus
                else: sdf['score'] = sdf['zg'] + sdf['zp'] # Balanced
                
                sdf = sdf.sort_values('score', ascending=False)
                
                # Assign Weights (100 Longs, 100 Shorts)
                # Ensure we have enough stocks in cache
                actual_top_n = min(TOP_N, len(sdf) // 2)
                w = 0.5 / actual_top_n
                
                for _, r in sdf.head(actual_top_n).iterrows(): ps.append(Position(r['t'], r['p'], 'long', w))
                for _, r in sdf.tail(actual_top_n).iterrows(): ps.append(Position(r['t'], r['p'], 'short', w))

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
    <title>S&P 500 Regime Strategy</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root { --primary: #2563eb; --bg: #f8fafc; --text: #1e293b; --card: #ffffff; }
        body { font-family: 'Inter', system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--text); }
        .nav { background: #1e293b; color: white; padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
        .container { max-width: 1400px; margin: 2rem auto; padding: 0 1rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        .card { background: var(--card); padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .stat-label { font-size: 0.875rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
        .stat-value { font-size: 1.875rem; font-weight: 800; margin-top: 0.25rem; }
        
        .tabs { display: flex; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.5rem; }
        .tab { padding: 0.75rem 1.5rem; cursor: pointer; border-bottom: 2px solid transparent; font-weight: 500; color: #64748b; }
        .tab.active { color: var(--primary); border-bottom-color: var(--primary); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th { text-align: left; padding: 0.75rem; background: #f1f5f9; font-weight: 600; font-size: 0.875rem; }
        td { padding: 0.75rem; border-bottom: 1px solid #f1f5f9; font-size: 0.875rem; }
        .scroll { max-height: 600px; overflow-y: auto; }
        
        .regime-badge { padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: bold; font-size: 0.75rem; }
        .regime-A { background: #dcfce7; color: #166534; }
        .regime-B { background: #fef9c3; color: #854d0e; }
        .regime-C { background: #fee2e2; color: #991b1b; }
        .regime-D { background: #f3f4f6; color: #1f2937; }
        
        select { width: 100%; padding: 0.5rem; border-radius: 6px; border: 1px solid #e2e8f0; margin-bottom: 1rem; }
    </style>
</head>
<body>
    <div class="nav">
        <div style="font-weight: 800; font-size: 1.25rem;">RegimeQuant Pro</div>
        <div>S&P 500 Scale Strategy</div>
    </div>
    
    <div class="container">
        <div class="grid">
            <div class="card">
                <div class="stat-label">Total Portfolio Value</div>
                <div class="stat-value">${{ "{:,.2f}".format(final_val) }}</div>
            </div>
            <div class="card">
                <div class="stat-label">Strategy Return</div>
                <div class="stat-value" style="color: {{ '#059669' if total_ret > 0 else '#dc2626' }}">
                    {{ "{:+.2f}%".format(total_ret) }}
                </div>
            </div>
            <div class="card">
                <div class="stat-label">Stock Universe / Portfolio</div>
                <div class="stat-value">{{ universe_count }} / 200</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 2rem;">
            <div class="stat-label">Equity Curve (Oct 2025 - Present)</div>
            <canvas id="equityChart" height="100"></canvas>
        </div>

        <div class="card">
            <div class="tabs">
                <div class="tab active" onclick="switchTab(event, 'stocks')">Universe Explorer</div>
                <div class="tab" onclick="switchTab(event, 'regimes')">Economic Data</div>
                <div class="tab" onclick="switchTab(event, 'history')">Trade Log</div>
            </div>

            <div id="stocks" class="tab-content active">
                <select id="tickerSelect" onchange="renderStockTable()">
                    {% for t in tickers %}
                    <option value="{{ t }}">{{ t }}</option>
                    {% endfor %}
                </select>
                <div class="scroll">
                    <table id="stockTable">
                        <thead>
                            <tr><th>Date</th><th>Price</th><th>Growth</th><th>Margin</th><th>Score</th></tr>
                        </thead>
                        <tbody id="stockBody"></tbody>
                    </table>
                </div>
            </div>

            <div id="regimes" class="tab-content">
                <div class="scroll">
                    <table>
                        <thead>
                            <tr><th>Date</th><th>Rate</th><th>Avg Rate</th><th>Bal Sheet</th><th>Avg BS</th><th>Regime</th></tr>
                        </thead>
                        <tbody>
                            {% for r in econ_data %}
                            <tr>
                                <td>{{ r.date }}</td>
                                <td>{{ "%.2f%%"|format(r.ir) }}</td>
                                <td>{{ "%.2f%%"|format(r.ir_avg) }}</td>
                                <td>{{ "{:,.0f}M".format(r.bs / 1000) }}</td>
                                <td>{{ "{:,.0f}M".format(r.bs_avg / 1000) }}</td>
                                <td><span class="regime-badge regime-{{r.regime}}">{{ r.regime }}</span></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>

            <div id="history" class="tab-content">
                <div class="scroll">
                    <table>
                        <thead>
                            <tr><th>Date</th><th>Regime</th><th>Portfolio Value</th></tr>
                        </thead>
                        <tbody>
                            {% for h in history_data %}
                            <tr>
                                <td>{{ h.date }}</td>
                                <td><span class="regime-badge regime-{{h.reg}}">{{ h.reg }}</span></td>
                                <td>${{ "{:,.2f}".format(h.val) }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        const histData = {{ history_data | tojson }};
        const stockCache = {{ stock_data | tojson }};

        new Chart(document.getElementById('equityChart'), {
            type: 'line',
            data: {
                labels: histData.map(d => d.date),
                datasets: [{
                    label: 'Portfolio',
                    data: histData.map(d => d.val),
                    borderColor: '#2563eb',
                    borderWidth: 3,
                    fill: true,
                    backgroundColor: 'rgba(37, 99, 235, 0.05)',
                    tension: 0.3,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: { 
                    x: { grid: { display: false } },
                    y: { ticks: { callback: v => '$' + v.toLocaleString() } }
                }
            }
        });

        function switchTab(e, id) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            e.target.classList.add('active');
            document.getElementById(id).classList.add('active');
        }

        function renderStockTable() {
            const t = document.getElementById('tickerSelect').value;
            const data = stockCache[t] || [];
            document.getElementById('stockBody').innerHTML = data.map(r => `
                <tr>
                    <td>${r.date}</td>
                    <td>$${(r.price || 0).toFixed(2)}</td>
                    <td>${(r.growth*100).toFixed(2)}%</td>
                    <td>${(r.margin*100).toFixed(2)}%</td>
                    <td>${(r.profit || 0).toFixed(6)}</td>
                </tr>
            `).join('');
        }
        renderStockTable();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    if BACKTEST_RESULTS is None:
        return """<body style="font-family:sans-serif; text-align:center; padding:50px; background:#f8fafc;">
                    <div style="background:white; display:inline-block; padding:30px; border-radius:12px; box-shadow:0 4px 6px rgba(0,0,0,0.05);">
                        <h2>Crunching S&P 500 Data...</h2>
                        <p>Processing 500 stocks and calculating 200 portfolio positions.</p>
                        <div style="width:100px; height:4px; background:#e2e8f0; margin:20px auto; position:relative; overflow:hidden;">
                            <div style="width:30%; height:100%; background:#2563eb; position:absolute; animation: load 1s infinite;"></div>
                        </div>
                    </div>
                    <style>@keyframes load { 0% { left: -30%; } 100% { left: 100%; } }</style>
                    <script>setTimeout(() => location.reload(), 5000);</script>
                  </body>"""
    
    final_val = BACKTEST_RESULTS['val'].iloc[-1]
    total_ret = ((final_val / INITIAL_CAPITAL) - 1) * 100
    
    return render_template_string(
        DASHBOARD_HTML,
        final_val=final_val,
        total_ret=total_ret,
        universe_count=len(FETCHED_STOCK_DATA),
        history_data=BACKTEST_RESULTS.to_dict(orient='records'),
        econ_data=ECON_REGIME_DATA.to_dict(orient='records'),
        stock_data=FETCHED_STOCK_DATA,
        tickers=sorted(FETCHED_STOCK_DATA.keys())
    )

def start_server():
    app.run(host='0.0.0.0', port=8080)

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    logger.info("Server started. Backtest logic initializing...")
    run_backtest()
    while True:
        time.sleep(1)
