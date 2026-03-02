"""
Task 5 Main Script - Agentic Trading (Crypto Only)

This script runs the trading simulation with Alpaca data for 20 crypto coins.
Multi-run consensus: 6am, 12pm, 5pm → trade only if 2+ agree
"""

from data_loader import load_market_data
from securefin_agent import SecureFinAIAgent
from consensus import load_signals, record_run, should_trade, save_signals
import json
from datetime import datetime


def main():
    now = datetime.now()
    hour = now.hour
    
    # Simplified: 3 runs per day
    # 4am Moscow = 9am China (start of Chinese trading)
    # 12pm Moscow = 5pm China (end of trading)
    # 8pm Moscow = 11pm China (after hours)
    
    is_chinese_morning = hour == 4   # 9am China - collect signal
    is_chinese_noon = hour == 12      # 5pm China - decision + trade
    is_chinese_evening = hour == 20   # 11pm China - manage/close
    # 1. Load market data
    print("="*50)
    print("Task 5: Agentic Trading (Crypto)")
    print("="*50)
    
    print("\n📥 Loading market data...")
    
    # 5 Alpaca crypto assets (reduced for faster processing)
    crypto_assets = [
        "BTC", "ETH", "SOL", "XRP", "DOGE"
    ]
    
    import time
    DELAY = 0.3  # 300ms delay between API calls
    crypto_data = {}
    
    for asset in crypto_assets:
        data = load_market_data(asset, "2025-08-01")
        if len(data) > 0:
            crypto_data[asset] = data
            print(f"   ✅ {asset}: {len(data)} days")
        time.sleep(DELAY)  # Rate limiting
    
    # 2. Initialize agent with NOFX and Alpaca trading
    print("\n🤖 Initializing agent (with NOFX + News + Trading)...")
    EXECUTE_TRADES = True  # Set to False for simulation only
    agent = SecureFinAIAgent(use_llm=False, use_nofx=True, execute_trades=EXECUTE_TRADES)
    
    # 3. Generate decisions for last 5 days
    results = []
    
    # Crypto decisions
    best_action = None
    best_asset = None
    best_score = 0
    
    for asset, data in crypto_data.items():
        if data is None or len(data) < 5:
            print(f"⚠️ Skipping {asset} - insufficient data")
            continue
        
        # Get last 5 days
        recent_data = data.tail(30)  # Need 30 days for proper indicators
        
        # Make decision (news sentiment will be fetched automatically)
        decision = agent.decide(recent_data, symbol=asset)
        
        # Track best opportunity (non-Hold with strongest signal)
        action = decision["action"]
        nofx = abs(decision.get("nofx_score", 0))
        
        if action in ["Buy", "Sell"] and nofx > best_score:
            best_score = nofx
            best_action = action
            best_asset = asset
        
        result = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "asset": asset,
            "action": decision["action"],
            "rationale": decision["rationale"],
            "nofx_score": decision.get("nofx_score", 0),
            "sentiment_score": decision.get("sentiment_score", 0)
        }
        
        results.append(result)
        
        print(f"\n📊 {asset}:")
        print(f"   Action: {decision['action']}")
        print(f"   NOFX Score: {decision.get('nofx_score', 'N/A')}")
        print(f"   Rationale: {decision['rationale'][:80]}...")
    
    # 4. Save results
    with open("submission.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # 5. Trading System (simplified: 3 runs)
    print(f"\n🇨🇳 Trading System (hour {hour}:00):")
    
    # 4am Moscow = 9am China: Collect first signal
    if is_chinese_morning:
        if best_action and best_asset:
            record_run(best_asset, best_action, 
                       results[0].get("nofx_score", 0),
                       results[0].get("sentiment_score", 0))
            print(f"   🌅 Chinese morning: Signal recorded: {best_action} {best_asset}")
    
    # 12pm Moscow = 5pm China: Decision time
    elif is_chinese_noon:
        # Check if we need to open a position
        positions = agent.alpaca.get_positions() if agent.alpaca else []
        
        # If no positions and have Buy signal with strong NOFX → open!
        if not positions and best_action == "Buy" and best_score > 0.3:
            print(f"   🫰 NO POSITIONS + STRONG BUY → OPENING: {best_asset}!")
            
            if best_asset and agent.alpaca and agent.alpaca.connected:
                data = crypto_data.get(best_asset)
                if data and len(data) >= 30:
                    recent = data.tail(30)
                    decision = agent.decide(recent, symbol=best_asset)
                    print(f"   ✅ Executed: {decision.get('action')} {best_asset}")
        
        # Also check consensus
        action, confidence = should_trade()
        print(f"   �🇨🇳 Chinese close: Decision - {action} ({confidence} signals)")
    
    # 8pm Moscow = 11pm China: Manage + close
    elif is_chinese_evening:
        print(f"   🌙 Chinese after-hours: Managing positions...")
        if agent.alpaca and agent.alpaca.connected:
            actions = agent.alpaca.manage_positions()
            if actions:
                print(f"   ✅ Closed {len(actions)} positions")
            # Execute the trade - agent will handle position sizing
    
    # Show consensus status
    data = load_signals()
    print(f"   Total runs today: {len(data.get('runs', []))}")
    
    # 6. EOD Risk Management: Close all positions at 23:00
    now = datetime.now()
    if now.hour >= 23:
        print("\n🌙 EOD: Closing all positions (no overnight holds)...")
        if agent.alpaca and agent.alpaca.connected:
            agent.alpaca.close_all()
    
    # 6. Show account balance
    if agent.alpaca and agent.alpaca.connected:
        try:
            account = agent.alpaca.trading_client.get_account()
            print(f"\n💰 Account Balance: ${float(account.portfolio_value):.2f}")
            print(f"   Cash: ${float(account.cash):.2f}")
            print(f"   Equity: ${float(account.equity):.2f}")
        except Exception as e:
            print(f"\n⚠️ Could not fetch balance: {e}")
    
    print(f"\n✅ Saved {len(results)} decisions to submission.jsonl")
    print("="*50)


if __name__ == "__main__":
    main()
