# Defiant-Fed-Policy-Investment-Strategy

Quantitative Methodology: Regime-Based Factor Rotation
This document outlines the formal mathematical framework used to categorize economic environments, evaluate stock performance, and manage risk within the S&P 500 Regime Strategy.
1. Economic Regime Classification
The strategy identifies market cycles by comparing current macroeconomic indicators against their historical averages (Simple Moving Averages).
Indicator Definitions
Let IR_t be the Effective Federal Funds Rate and BS_t be the Federal Reserve Total Assets (Balance Sheet) at time t.
 * Rate Trend (T_{IR}): Determined by a 26-week (6-month) SMA.
   
 * Liquidity Trend (T_{BS}): Determined by a 52-week (1-year) SMA.
   
Regime Matrix
The interaction of these two binary states defines the four regimes:
| Regime | T_{IR} | T_{BS} | Economic Context | Factor Focus |
|---|---|---|---|---|
| A | Low | Expansion | Liquidity Injection / "Easy Money" | Growth (Z_g) |
| B | Low | Contraction | Transition / Neutral | Balanced (Z_g + Z_p) |
| C | High | Expansion | Inflationary / Overheating | Balanced (Z_g + Z_p) |
| D | High | Contraction | Tightening / "Risk-Off" | Profitability (Z_p) |
2. Factor Normalization (Cross-Sectional Z-Scores)
To ensure that different units (percentages for growth vs. ratios for profit) are comparable, we apply a cross-sectional Z-score normalization across the universe U.
For a specific attribute x (e.g., Revenue Growth) of stock i:

Where:
 * \bar{x} = \frac{1}{|U|} \sum_{j \in U} x_j (Universe Mean)
 * \sigma_x = \sqrt{\frac{1}{|U|} \sum_{j \in U} (x_j - \bar{x})^2} (Universe Standard Deviation)
3. Factor Scoring Engine
The final score S_i for each security is a weighted combination of its Growth Score (Z_{i,g}) and Profitability Score (Z_{i,p}).
Regime-Dependent Weighting (w)
The weights shift to prioritize different characteristics based on the identified regime:
 * Regime A: w_g = 1.0, w_p = 0.0
 * Regime D: w_g = 0.0, w_p = 1.0
 * Regimes B & C: w_g = 0.5, w_p = 0.5
4. Portfolio Construction
The strategy maintains a Market Neutral posture through a Long/Short structure.
 * Selection: - L = \{ \text{Top } N \text{ stocks by } S \}
   *  * Weighting: - For each j \in L, weight w_j = \frac{0.5}{N}
   * For each j \in S, weight w_j = \frac{0.5}{N}
 * Gross Exposure: \sum |w_j| = 1.0 (100%)
 * Net Exposure: \sum w_j = 0.0 (0%)
5. Risk and Return Attribution
Individual Position Return (R_j)
For a long position, R_j = \frac{P_{current} - P_{entry}}{P_{entry}}. For a short position, R_j = \frac{P_{entry} - P_{current}}{P_{entry}}.
Trailing Stop-Loss (SL)
The strategy tracks the "Extreme Price" (P_{ext}), which is the peak for longs or trough for shorts.
 * Profit Threshold: Trailing stop activates when R_j > 20\%.
 * Exit Condition: If active, exit if P_{current} retraces 10\% from P_{ext}.
Portfolio Value (V_t)
