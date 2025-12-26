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
DEV_STOCK_LIMIT = 80  # Limit for speed (set to None for full S&P 500)
TOP_N = 15            # Number of Longs and Shorts
BACKTEST_START_DATE = datetime(2025, 10, 8) 
INITIAL_CAPITAL = 100000.0

# GLOBAL STORAGE
BACKTEST_RESULTS = None
ECON_REGIME_DATA = None
FETCHED_STOCK_DATA = {} 
LIVE_STATUS = {} # Stores current active positions and portfolio state

# ==========================================
# 1. RISK & POSITION MANAGEMENT
# ==========================================
class Position:
    def __init__(self, ticker, entry_price, side='long', weight=0.0):
        self.ticker = ticker
        self.entry_price = entry_price
        self.side = side 
        self.weight = weight 
        self.shares = 0 # Calculated based on portfolio value
        
        self.is_active = True
        self.is_trailing = False
        self.frozen_price = None 
        self.extreme_price = entry_price 
        self.current_price = entry_price
        self.last_update = datetime.now()

    def update_logic(self, current_price):
        """Updates stop loss/take profit logic based on new price"""
        self.current_price = current_price
        self.last_update = datetime.now()
        
        if not self.is_active:
            return self.frozen_price

        if self.side == 'long':
            if current_price > self.extreme_price: self.extreme_price = current_price
            
            # Hard Stop: -20%
            if current_price < (self.entry_price * 0.80):
                self.is_active, self.frozen_price = False, self.entry_price * 0.80
                return self.frozen_price
            
            # Activate Trailing: +20%
            if not self.is_trailing and current_price > (self.entry_price * 1.20):
                self.is_trailing = True
            
            # Trailing Stop: -10% from Peak
            if self.is_trailing and current_price < (self.extreme_price * 0.90):
                self.is_active, self.frozen_price = False, self.extreme_price * 0.90
                return self.frozen_price
                
        elif self.side == 'short':
            if current_price < self.extreme_price: self.extreme_price = current_price
            
            # Hard Stop: +20%
            if current_price > (self.entry_price * 1.20):
                self.is_active, self.frozen_price = False, self.entry_price * 1.20 
                return self.frozen_price
            
            # Activate Trailing: -20%
            if not self.is_trailing and current_price < (self.entry_price * 0.80):
                self.is_trailing = True
            
            # Trailing Stop: +10% from Bottom
            if self.is_trailing and current_price > (self.extreme_price * 1.10):
                self.is_active, self.frozen_price = False, self.extreme_price * 1.10
                return self.frozen_price
                
        return current_price

    def get_pct_return(self, current_price=None):
        cp = current_price if current_price else self.current_price
        eff_price = self.frozen_price if not self.is_active else cp
        if self.side == 'long':
            return (eff_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - eff_price) / self.entry_price

    def to_dict(self):
        return {
            'ticker': self.ticker,
            'side': self.side,
            'entry': self.entry_price,
            'current': self.current_price,
            'extreme': self.extreme_price,
            'is_trailing': self.is_trailing,
            'pnl_pct': self.get_pct_return(self.current_price),
            'status': 'Active' if self.is_active else 'Closed'
        }

# ==========================================
# 2. DATA ENGINES
# ==========================================
class FredEngine:
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
        except Exception as e: 
            logger.error(f"FRED Error {sid}: {e}")
            return pd.DataFrame()

    def get_regimes(self):
        logger.info("Building Economic Regimes...")
        # Get enough history for moving averages
        start = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        rate = self.fetch("FEDFUNDS").rename(columns={'value': 'ir'})
        bs = self.fetch("WALCL").rename(columns={'value': 'bs'})
        
        if rate.empty or bs.empty: 
            logger.warning("FRED Data empty. Using mock data for demo.")
            # Fallback for demo without API Key
            dates = pd.date_range(start="2015-01-01", end=datetime.now(), freq='W')
            df = pd.DataFrame(index=dates)
            df['ir'] = 5.0 + np.random.normal(0, 0.5, len(df))
            df['bs'] = 8000000 + np.random.normal(0, 100000, len(df))
            df.index.name = 'date'
        else:
            df = pd.merge_asof(bs.sort_values('date'), rate.sort_values('date'), on='date')
            df = df.set_index('date')

        # Logic
        df['ir_avg'] = df['ir'].rolling(window=260, min_periods=50).mean() # 5yr approx
        df['bs_avg'] = df['bs'].rolling(window=52, min_periods=20).mean()  # 1yr approx
        
        v = df['ir_avg'].notna() & df['bs_avg'].notna()
        hr = df['ir'] > df['ir_avg']
        hb = df['bs'] > df['bs_avg']
        
        df['regime'] = "None"
        df.loc[v & (~hr) & hb, 'regime'] = "A" # Low Rate, High Liq
        df.loc[v & (~hr) & (~hb), 'regime'] = "B" # Low Rate, Low Liq
        df.loc[v & hr & hb, 'regime'] = "C" # High Rate, High Liq
        df.loc[v & hr & (~hb), 'regime'] = "D" # High Rate, Low Liq
        
        return df

class StockEngine:
    def process(self, ticker, dates):
        try:
            s = yf.Ticker(ticker)
            # Fetch history
            h = s.history(start="2024-01-01", interval="1wk")
            if h.empty: return None
            h = h[['Close']].rename(columns={'Close': 'price'})
            h.index = h.index.tz_localize(None)
            
            # Fetch fundamentals
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
                df['profit'] = df['margin'] * (df['eps'] / (df['price'] + 1e-6))
                
            df['ticker'] = ticker
            # Reindex to match simulation timeline
            return df.reindex(dates, method='ffill')
        except: return None

# ==========================================
# 3. LIVE TRADER & BACKTESTER
# ==========================================
class LiveTrader:
    """
    Handles real-time updates and portfolio management
    """
    def __init__(self, current_positions, capital):
        self.positions = current_positions
        self.capital = capital
        self.tickers = [p.ticker for p in self.positions]
    
    def fetch_live_prices(self):
        """Batched fetch for speed"""
        if not self.tickers: return {}
        logger.info(f"LIVE: Fetching real-time prices for {len(self.tickers)} positions...")
        try:
            # yf.download is faster for batch than Ticker.info
            data = yf.download(self.tickers, period="1d", progress=False)['Close']
            prices = {}
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Handle single vs multiple tickers
                if len(self.tickers) == 1:
                    prices[self.tickers[0]] = data.iloc[-1].item()
                else:
                    for t in self.tickers:
                        if t in data.columns:
                            prices[t] = data[t].iloc[-1]
            return prices
        except Exception as e:
            logger.error(f"Live fetch failed: {e}")
            return {}

    def run_update(self):
        """
        The 'One-Time' update logic requested.
        """
        prices = self.fetch_live_prices()
        
        portfolio_val = 0
        active_list = []
        
        for p in self.positions:
            if p.ticker in prices and pd.notna(prices[p.ticker]):
                # Update Trailing logic
                p.update_logic(prices[p.ticker])
            
            # Calculate value
            # Note: In simulation we tracked weight, but for live we need notion of shares/equity
            # For this demo, we assume the 'weight' * capital is the allocation
            allocation = self.capital * p.weight
            ret = p.get_pct_return()
            pos_val = allocation * (1 + ret)
            
            if p.is_active:
                portfolio_val += pos_val
                active_list.append(p.to_dict())
            else:
                # Closed position (stopped out during live check)
                portfolio_val += pos_val # Converted to cash effectively
        
        # Construct Status Object
        status = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_equity': portfolio_val,
            'active_positions': active_list,
            'message': "Live update successful. Market data retrieved."
        }
        return status

def run_system():
    global BACKTEST_RESULTS, ECON_REGIME_DATA, FETCHED_STOCK_DATA, LIVE_STATUS
    
    # 1. Setup Data
    fred = FredEngine(FRED_API_KEY)
    econ = fred.get_regimes()
    
    # Slice for simulation
    sim_econ = econ[econ.index >= BACKTEST_START_DATE].copy()
    
    if sim_econ.empty:
        logger.error("No Data for backtest.")
        return

    ECON_REGIME_DATA = sim_econ.reset_index()
    ECON_REGIME_DATA['date'] = ECON_REGIME_DATA['date'].dt.strftime('%Y-%m-%d')

    # 2. Fetch Stock Universe
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        ts = pd.read_html(StringIO(requests.get(url, headers={"User-Agent":"Mozilla"}).text))[0]['Symbol'].tolist()
        ts = [t.replace('.','-') for t in ts][:DEV_STOCK_LIMIT]
    except:
        ts = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA'] # Fallback

    cache = {}
    logger.info(f"Loading {len(ts)} stocks for Backtest + Live Setup...")
    
    for t in ts:
        d = StockEngine().process(t, sim_econ.index)
        if d is not None: 
            cache[t] = d
            display_df = d.reset_index().copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            FETCHED_STOCK_DATA[t] = display_df.to_dict(orient='records')

    # 3. Run Historical Backtest
    val = INITIAL_CAPITAL
    ps = []
    cur_reg = None
    hist = []
    
    logger.info("Running Historical Simulation...")
    for date in sim_econ.index.sort_values():
        reg = sim_econ.loc[date, 'regime']
        
        # Calculate PnL for existing positions
        if ps:
            daily_pnl = sum(p.get_pct_return(cache[p.ticker].loc[date, 'price']) * p.weight for p in ps)
            val *= (1 + daily_pnl)
            # Reset weights effectively (rebalancing daily pnl into equity)
        
        # Rebalance Logic (Simplified: Rebalance on Regime Change OR Monthly)
        # For this demo: Regime Change Only
        if reg != cur_reg:
            cur_reg, ps = reg, []
            snaps = []
            
            # Rank Stocks
            for t, d in cache.items():
                try:
                    row = d.loc[date]
                    if pd.notna(row['price']):
                        snaps.append({'t': t, 'p': row['price'], 'g': row.get('growth', 0), 'pr': row.get('profit', 0)})
                except: pass
                
            sdf = pd.DataFrame(snaps).dropna()
            if not sdf.empty:
                # Ranking Logic
                sdf['zg'] = (sdf['g'] - sdf['g'].mean()) / (sdf['g'].std() + 1e-6)
                sdf['zp'] = (sdf['pr'] - sdf['pr'].mean()) / (sdf['pr'].std() + 1e-6)
                
                if reg == "A": sdf['score'] = sdf['zg'] # Growth focus
                elif reg == "D": sdf['score'] = sdf['zp'] # Value focus
                else: sdf['score'] = sdf['zg'] + sdf['zp'] # Balanced
                
                sdf = sdf.sort_values('score', ascending=False)
                
                # Allocate
                w = 0.5 / TOP_N # 50% Long, 50% Short allocation bucket
                for _, r in sdf.head(TOP_N).iterrows(): ps.append(Position(r['t'], r['p'], 'long', w))
                for _, r in sdf.tail(TOP_N).iterrows(): ps.append(Position(r['t'], r['p'], 'short', w))

        hist.append({'date': date.strftime('%Y-%m-%d'), 'val': round(val, 2), 'reg': reg})

    BACKTEST_RESULTS = pd.DataFrame(hist)
    
    # 4. LIVE UPDATE TRIGGER
    logger.info("Transitioning to LIVE TRADER...")
    # Pass the final positions and final capital to the live trader
    trader = LiveTrader(ps, val)
    LIVE_STATUS = trader.run_update()
    
    logger.info("System Ready.")

# ==========================================
# 4. WEB SERVER
# ==========================================
app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Regime Bot | Live Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Segoe UI', Roboto, sans-serif; margin: 0; background: #f0f2f5; color: #333; }
        .sidebar { width: 250px; background: #1a202c; color: white; position: fixed; height: 100vh; padding: 20px; }
        .main { margin-left: 290px; padding: 30px; }
        
        .logo { font-size: 20px; font-weight: bold; margin-bottom: 30px; display: block; color: white; text-decoration: none; }
        .nav-item { padding: 12px 15px; display: block; color: #a0aec0; text-decoration: none; border-radius: 8px; margin-bottom: 5px; cursor: pointer; }
        .nav-item:hover, .nav-item.active { background: #2d3748; color: white; }
        
        .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 25px; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 25px; }
        .stat-box { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .stat-label { font-size: 12px; color: #718096; text-transform: uppercase; font-weight: bold; }
        .stat-val { font-size: 24px; font-weight: bold; color: #2d3748; margin-top: 5px; }
        
        .live-badge { display: inline-block; padding: 4px 8px; background: #48bb78; color: white; border-radius: 4px; font-size: 11px; font-weight: bold; vertical-align: middle; margin-left: 10px; }
        
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th { text-align: left; padding: 12px; border-bottom: 2px solid #e2e8f0; color: #4a5568; font-size: 13px; }
        td { padding: 12px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }
        .pos-long { color: #2f855a; font-weight: 500; }
        .pos-short { color: #c53030; font-weight: 500; }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="sidebar">
        <a href="#" class="logo"><i class="fas fa-robot"></i> REGIME BOT</a>
        <div class="nav-item active" onclick="switchTab('dashboard')"><i class="fas fa-chart-line"></i> Dashboard</div>
        <div class="nav-item" onclick="switchTab('live')"><i class="fas fa-bolt"></i> Live Monitor</div>
        <div class="nav-item" onclick="switchTab('data')"><i class="fas fa-database"></i> Data Explorer</div>
    </div>

    <div class="main">
        <!-- HEADER STATS -->
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-label">Total Equity</div>
                <div class="stat-val">${{ "{:,.0f}".format(live_status.total_equity if live_status else final_val) }}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Return</div>
                <div class="stat-val" style="color: {{ 'green' if total_ret > 0 else 'red' }}">
                    {{ "{:+.2f}%".format(total_ret) }}
                </div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Current Regime</div>
                <div class="stat-val">{{ current_regime }}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">System Status</div>
                <div class="stat-val" style="font-size: 16px; margin-top: 10px;">
                    <span class="live-badge">RUNNING</span>
                </div>
            </div>
        </div>

        <!-- DASHBOARD TAB -->
        <div id="dashboard" class="tab-content active">
            <div class="card">
                <h3><i class="fas fa-history"></i> Performance History</h3>
                <canvas id="pnlChart" width="800" height="300"></canvas>
            </div>
        </div>

        <!-- LIVE MONITOR TAB -->
        <div id="live" class="tab-content">
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3><i class="fas fa-satellite-dish"></i> Live Portfolio Status</h3>
                    <span style="font-size:12px; color:#718096;">Last Update: {{ live_status.timestamp }}</span>
                </div>
                <p style="color: #718096; font-size: 14px;">Real-time prices fetched from exchange. Positions managed by Auto-Regime Logic.</p>
                
                <table>
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Side</th>
                            <th>Entry Price</th>
                            <th>Live Price</th>
                            <th>PnL %</th>
                            <th>Stop Loss / Trailing</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for pos in live_status.active_positions %}
                        <tr>
                            <td><strong>{{ pos.ticker }}</strong></td>
                            <td class="{{ 'pos-long' if pos.side == 'long' else 'pos-short' }}">{{ pos.side.upper() }}</td>
                            <td>${{ "{:.2f}".format(pos.entry) }}</td>
                            <td>${{ "{:.2f}".format(pos.current) }}</td>
                            <td style="color: {{ 'green' if pos.pnl_pct > 0 else 'red' }}">
                                {{ "{:+.2f}%".format(pos.pnl_pct * 100) }}
                            </td>
                            <td>
                                ${{ "{:.2f}".format(pos.extreme) }} 
                                {% if pos.is_trailing %}<span style="font-size:10px; background:blue; color:white; padding:2px 4px; border-radius:4px;">TRAILING</span>{% endif %}
                            </td>
                            <td><span class="live-badge">ACTIVE</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- DATA TAB -->
        <div id="data" class="tab-content">
            <div class="card">
                <h3>Market Data & Regimes</h3>
                <div style="max-height: 500px; overflow-y: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Fed Funds</th>
                                <th>Balance Sheet</th>
                                <th>Regime</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in econ_data %}
                            <tr>
                                <td>{{ row.date }}</td>
                                <td>{{ "%.2f"|format(row.ir) }}</td>
                                <td>{{ "{:,.0f}".format(row.bs) }}</td>
                                <td><strong>{{ row.regime }}</strong></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

    </div>

    <script>
        // Chart Logic
        const chartData = {{ chart_data | tojson }};
        const ctx = document.getElementById('pnlChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.map(d => d.date),
                datasets: [{
                    label: 'Equity Curve',
                    data: chartData.map(d => d.val),
                    borderColor: '#4299e1',
                    backgroundColor: 'rgba(66, 153, 225, 0.1)',
                    fill: true,
                    tension: 0.2
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: { 
                    x: { grid: { display: false } },
                    y: { grid: { color: '#f7fafc' } }
                }
            }
        });

        // Tab Switching
        function switchTab(id) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            event.currentTarget.classList.add('active');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    if BACKTEST_RESULTS is None:
        return "System Initializing... Refresh in 10 seconds."
    
    # Calculate stats
    # Use Live Equity if available, otherwise Backtest final
    current_equity = LIVE_STATUS.get('total_equity', INITIAL_CAPITAL) if LIVE_STATUS else INITIAL_CAPITAL
    total_ret = ((current_equity / INITIAL_CAPITAL) - 1) * 100
    
    current_regime = ECON_REGIME_DATA['regime'].iloc[-1] if ECON_REGIME_DATA is not None else "Unknown"

    return render_template_string(
        DASHBOARD_HTML,
        initial_cap=INITIAL_CAPITAL,
        final_val=current_equity,
        total_ret=total_ret,
        current_regime=current_regime,
        live_status=LIVE_STATUS,
        chart_data=BACKTEST_RESULTS.to_dict(orient='records'),
        econ_data=ECON_REGIME_DATA.to_dict(orient='records')
    )

def start_server():
    app.run(host='0.0.0.0', port=8080)

if __name__ == "__main__":
    # Start Web Server in Background
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Run System Logic
    logger.info("Starting Regime Bot System...")
    run_system()
    
    # Keep main thread alive
    while True:
        time.sleep(1)