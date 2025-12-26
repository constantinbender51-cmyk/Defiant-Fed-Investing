# Defiant-Fed-Policy-Investment-Strategy

Quantitative Methodology: Regime-Based Factor Rotation
This document outlines the formal logic used to categorize economic environments, evaluate stock performance, and manage risk within the S&P 500 Regime Strategy.
1. Economic Regime Classification
The strategy identifies market cycles by comparing current macroeconomic indicators against their historical averages (Simple Moving Averages).
Indicator Definitions
 * IR: Effective Federal Funds Rate
 * BS: Federal Reserve Total Assets (Balance Sheet)
 * SMA_26: 26-week (6-month) Simple Moving Average
 * SMA_52: 52-week (1-year) Simple Moving Average
1. Rate Trend (T_IR)
If Current_IR > SMA_26(IR): 
    Rate Trend = HIGH
Else: 
    Rate Trend = LOW

2. Liquidity Trend (T_BS)
If Current_BS > SMA_52(BS): 
    Liquidity Trend = EXPANSION
Else: 
    Liquidity Trend = CONTRACTION

Regime Matrix
| Regime | Rate Trend | Liquidity Trend | Economic Context | Factor Focus |
|---|---|---|---|---|
| A | LOW | EXPANSION | Easy Money / High Liquidity | Growth (Z_g) |
| B | LOW | CONTRACTION | Transition / Neutral | Balanced |
| C | HIGH | EXPANSION | Inflationary / Overheating | Balanced |
| D | HIGH | CONTRACTION | Tightening / Risk-Off | Profitability (Z_p) |
2. Factor Normalization (Z-Scores)
To compare different metrics (like % growth vs profit margins), we use cross-sectional Z-scores. This measures how many standard deviations a stock is from the S&P 500 average.
Z_score = (Stock_Value - Universe_Average) / Universe_Standard_Deviation

 * Z_g: Normalized Revenue Growth Score
 * Z_p: Normalized Profitability/Quality Score
3. Factor Scoring Engine
The final score for each stock determines its rank in the portfolio.
Final_Score = (Weight_g * Z_g) + (Weight_p * Z_p)

Regime Weights
The strategy rotates what it values based on the regime:
 * Regime A: Weight_g = 1.0, Weight_p = 0.0 (Growth focus)
 * Regime D: Weight_g = 0.0, Weight_p = 1.0 (Profit focus)
 * Regimes B & C: Weight_g = 0.5, Weight_p = 0.5 (Balanced focus)
4. Portfolio Construction
The strategy is "Market Neutral," meaning it bets on the relative performance of stocks rather than the direction of the overall market.
 * Selection:
   * Longs: The top 100 stocks with the highest scores.
   * Shorts: The bottom 100 stocks with the lowest scores.
 * Weighting:
   * Each Long position = +0.5% of portfolio (when 100 positions)
   * Each Short position = -0.5% of portfolio (when 100 positions)
   * Actual weight = 0.5 / min(TOP_N, available_stocks/2)
 * Exposure:
   * Gross Exposure = 100% (50% Long + 50% Short)
   * Net Exposure = 0% (Market Neutral)
5. Risk Management
Trailing Stop-Loss
Each position tracks its "Extreme Price" (the highest price since buying for longs, or the lowest for shorts).
 * Initial Stop: 20% loss from the entry price.
 * Profit Trigger: Once a position gains 20%, a trailing stop is activated.
 * Exit: If the trailing stop is active, the position is closed if it drops 10% from its Extreme Price for longs, or rises 10% from its Extreme Price for shorts.
Portfolio Value Tracking
New_Portfolio_Value = Old_Value * (1 + sum(Position_Return * Position_Weight))

6. Simulation Parameters
 * Stock Universe: S&P 500 (limited to DEV_STOCK_LIMIT = 500 for development)
 * Portfolio Size: TOP_N = 100 Longs and 100 Shorts
 * Initial Capital: $1,000,000
 * Backtest Start Date: October 8, 2025
 * Data Frequency: Weekly
 * Economic Indicators: Federal Funds Rate (FEDFUNDS) and Fed Balance Sheet (WALCL)
 * Fallback: Synthetic data generated when FRED API key is unavailable

