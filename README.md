Quantitative Regime Analysis Engine
1. System Overview
This application functions as an automated quantitative research assistant. It dynamically assesses the current US macroeconomic environment ("Regime") using Federal Reserve data and applies a regime-congruent factor scoring model to the S&P 500 equity universe.
The final output is a web-based, academic-style report that identifies long and short candidates based on standardized factor exposure.
System Architecture
The application consists of four distinct stages:
 * Macro-Engine (FredEngine): Queries the St. Louis Fed (FRED) to determine the current economic state.
 * Micro-Engine (StockEngine): Scrapes the S&P 500 universe and fetches fundamental metrics via Finnhub.
 * Analysis Logic: Normalizes data using Z-Scores and ranks assets based on the active regime.
 * Presentation: Renders a minimal, auto-refreshing HTML report via Flask.
2. Theoretical Framework & Methodology
The system relies on specific heuristic assumptions to operationalize "investment strategy" into code. These hardcoded assumptions define the model's worldview.
A. Macroeconomic Regime Definition
The model segments the economy into four distinct regimes based on the rate of change of Liquidity (Fed Balance Sheet) and Cost of Capital (Fed Funds Rate).
Trend Definition:
 * Interest Rates (Ir): A 5-year (260-week) rolling average is used as the baseline.
 * Balance Sheet (Bs): A 1-year (52-week) rolling average is used as the baseline.
Regime Logic:
 * Regime A: Expansion
   * Logic: Interest Rates are BELOW average AND Balance Sheet is ABOVE average.
   * Description: Liquidity is high, money is cheap.
   * Strategy: Favor Growth.
 * Regime B: Deflation
   * Logic: Interest Rates are BELOW average AND Balance Sheet is BELOW average.
   * Description: Money is cheap but liquidity is contracting.
   * Strategy: Balanced.
 * Regime C: Inflationary Boom
   * Logic: Interest Rates are ABOVE average AND Balance Sheet is ABOVE average.
   * Description: Costs rising but liquidity remains high.
   * Strategy: Balanced.
 * Regime D: Tightening
   * Logic: Interest Rates are ABOVE average AND Balance Sheet is BELOW average.
   * Description: Costly money and draining liquidity.
   * Strategy: Favor Profit/Value.
B. Factor Construction & Scoring
Raw fundamental data is standardized relative to the peer group (S&P 500) to create comparable rankings.
1. Profit Score Formulation
The model creates a custom metric combining operational efficiency and valuation.
 * Assumption: High margins are only valuable if bought at a reasonable earnings yield.
 * Formula: Profit Score = Operating Margin + (1 / PE Ratio)
2. Z-Score Normalization
To make "Growth" and "Profit" comparable, metrics are converted to Z-scores.
 * Assumption: A stock is "good" if it is a statistically significant positive outlier relative to the current market distribution.
 * Formula: Z = (Value - Average) / Standard Deviation
C. Universe Selection
 * Source: Wikipedia's "List of S&P 500 companies" is used as the source of truth for the investable universe.
 * Filters: The system strictly removes instruments with a PE of 0 or less, or missing growth data, assuming unprofitable companies are too volatile for this specific logic.
3. Technical Constraints & Setup
Dependencies
The application requires the following Python libraries:
 * flask
 * pandas
 * numpy
 * requests
 * lxml (or html5lib)
Environment Variables
The system requires two API keys. Without these, engines will initialize but return empty datasets.
 * FRED_API_KEY: Access to Federal Reserve Economic Data (Source: stlouisfed.org).
 * FINNHUB_API_KEY: Access to stock fundamentals (Source: finnhub.io).
Execution
Run the application directly via Python. The analysis runs in a background thread to prevent blocking the web server startup.
Command:
python app.py
 * Port: Defaults to 5000 (or uses PORT env variable).
 * Startup Time: The initial analysis involves fetching data for up to 500 stocks with a 1-second delay per request. Expect a ~10-minute warmup time before data appears on the web page.
4. Known Limitations
 * Data Latency: The system fetches data once upon startup (triggered by threading). It does not re-run the analysis automatically on a schedule. To refresh data, the application must be restarted.
 * Rate Limiting: The StockEngine enforces a strict 1-second sleep between requests. While this ensures compliance with free-tier API limits, it significantly slows down report generation.
 * Memory Persistence: Global state is held in memory. If the application crashes or restarts, historical analysis is lost.

Disclaimer: This software is for educational and research purposes only. It does not constitute financial advice, and the "Regime" logic is a simplified heuristic model.
