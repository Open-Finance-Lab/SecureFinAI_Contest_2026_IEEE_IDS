# FinAI Task III: Prediction Market Arbitrage

## Task Overview

This task focuses on developing trading agents that identify and execute arbitrage opportunities across two prediction markets — **Kalshi** and **Polymarket** — for a series of sports events with binary options.

Models may incorporate sentiment signals in addition to market data to anticipate market moves when new information changes expectations during a game.

Evaluation will be conducted via **paper trading**, where agents perform simulated trading on real market data without using real capital.

---

## Objective

Your agent should:

- Identify cross-market arbitrage opportunities between Kalshi and Polymarket binary sports contracts
- Ingest and interpret real-time market data:
  - Prices
  - Bid–ask spreads
  - Liquidity
  - Order book signals
- Maintain basic risk management:
  - Position sizing
  - Exposure limits
  - Avoiding overtrading
- Execute paper trades by generating explicit trade actions:
  - Event selection
  - Contract direction (YES / NO)
  - Position size
  - Simulated execution under realistic market conditions
- Optionally incorporate sentiment signals from external data sources

---

## Why It Matters

Prediction markets are becoming increasingly mainstream as real-time signals of collective expectations.

Cross-venue arbitrage is meaningful because it tests whether an agent can:

- Capture temporary mispricing between Kalshi and Polymarket
- React under latency constraints
- Account for bid–ask spreads
- Handle liquidity constraints
- Operate within risk limits

This task evaluates both market efficiency detection and execution robustness.

---

# Starter Kit Overview

The starter kit allows contestants to:

- Pull real-time market data from Kalshi and Polymarket
- Collect external sports sentiment data via RSS feeds
- Build dashboards
- Generate features for trading agents
- Detect market-moving updates (injuries, trades, lineup changes, major headlines)

---

# Polymarket Data

Documentation:  
https://docs.polymarket.com/quickstart/overview#apis-at-a-glance

Polymarket uses a two-layer data access system:

- **Gamma API** → Market discovery
- **CLOB API** → Live pricing and order book data

Includes:
- Public WebSocket stream for real-time updates
- Configurable runtime duration for streaming

### What You Get from Polymarket API

- List of sports series from Gamma (series_id)
- Pull events and markets
- Extract token IDs for CLOB usage
- Fetch:
  - Current prices via `/price`
  - Current order book via `/book`
- WebSocket live stream for low-latency bid/ask updates

---

# Kalshi Data

Documentation:  
https://docs.kalshi.com/welcome

This demo retrieves sports market data from Kalshi API and includes a liquidity quality filter.

### What You Get from Kalshi API

- Market list including:
  - Bid
  - Ask
  - Last price
  - Volume
  - Open interest
- Computed:
  - Spread
  - Midpoint
- Filtered view of higher-liquidity, tighter-spread markets
- Optional live polling output (auto-stops after configurable runtime)

---

# Sports Sentiment Data via RSS

This demo creates a **Sports Sentiment Signal Feed** using RSS feeds.

Features:

- Polls multiple sports news RSS sources
- Normalizes timestamps (UTC)
- Tags items by league (NFL, NBA, etc.)
- Applies signal labels:
  - injury
  - lineup
  - trade
  - suspension
  - game_event
- Deduplicates entries
- Clean polling loop with configurable runtime

### What You Get from the RSS Notebook

- Multi-source RSS ingestion
- Deduplication
- Signal tagging
- Auto-stopping clean polling loop

---

# Evaluation

Each submission will be evaluated in a **paper-trading environment** using real-time market data from Kalshi and Polymarket.

- No real capital is used.
- Agents are evaluated on their ability to capture cross-market arbitrage opportunities.

### Initial Capital

- Polymarket Account: $10,000
- Kalshi Account: $10,000

---

# Arbitrage Performance Metrics

### 1. Total Profit / Loss (P&L)
Net paper-trading return over the evaluation period.

### 2. Sharpe Ratio
Risk-adjusted return consistency.

### 3. Maximum Drawdown
Largest peak-to-trough decline in cumulative P&L.

### 4. Number of Successful Arbitrage Trades
Count of profitable arbitrage executions.

---

## Summary

This task evaluates:

- Cross-venue mispricing detection
- Real-time execution capability
- Risk-aware trading strategies
- Optional sentiment-enhanced market anticipation
- Performance consistency under realistic market conditions