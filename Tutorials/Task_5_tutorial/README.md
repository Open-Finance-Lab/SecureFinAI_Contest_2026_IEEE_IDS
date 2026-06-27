# Task 5 Starter Kit: Agentic Trading

This starter kit provides a template for the **Agentic Trading** task.

> **IMPORTANT**: This is only an example. You should design and implement your own trading agent for submission.

## 📂 Directory Structure

```
Task_5_tutorial/
├── README.md               # This file
├── Task_5_Description.md   # Task description and dataset info
├── requirements.txt        # Python dependencies (add your own)
├── data_loader.py          # Script to load market data
├── example_agent.py        # Template agent (implement your own)
├── export_alpaca_logs.py   # Export Alpaca paper trading logs for submission
└── main.py                 # Main script to run trading simulation
```

## 🚀 Setup

```bash
cd Tutorials/Task_5_tutorial
pip install -r requirements.txt
```

## ▶️ Running

```bash
python main.py
```

## 📊 Datasets
Participants may fetch historical market data from Alpaca Market Data API:
* **Stock**: [https://alpaca.markets/sdks/python/api_reference/data/stock.html](https://alpaca.markets/sdks/python/api_reference/data/stock.html)
* **Crypto**: [https://alpaca.markets/sdks/python/api_reference/data/crypto.html](https://alpaca.markets/sdks/python/api_reference/data/crypto.html)

## 🏁 Evaluation
Participants will create an Alpaca paper trading account, run their agent to trade during the evaluation period, and submit required files at the end of the evaluation period.
* **Time Period:** **April 20 – May 1**.
* **Initial Capital:** Each account starts with a fixed capital of **$100,000**.

## 📝 Export Alpaca Trading Logs

After the evaluation period ends, run the log export script to generate the required submission files from your Alpaca paper trading account.

```bash
python export_alpaca_logs.py --start 2026-04-20 --end 2026-05-01
```

## 📦 What to Submit
* an **orders JSONL** file,
* a **daily equity CSV** file,
* a **snapshot of the Alpaca portfolio value** at the end of the evaluation period.

## 📈 Metrics
* **Primary:** Cumulative Return (**CR**)
* **Secondary:** Sharpe Ratio (**SR**), Maximum Drawdown (**MD**), Daily Volatility (**DV**), Annualized Volatility (**AV**)

> *A single paper trading account trade asset from **only one market type: stock or crypto**. Create two accounts if you are participating in both the stock and crypto tracks. Stock and crypto tracks are evaluated separately.*