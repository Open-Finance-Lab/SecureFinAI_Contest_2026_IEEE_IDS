"""
Multi-run consensus trading system

Run at: 6:00, 12:00, 17:00
- 6am: Store signal
- 12pm: Compare with morning, store
- 5pm: Final decision - only trade if 2/3 runs agree
- 11pm: Close all (from main.py)
"""

import json
import os
from datetime import datetime

SIGNAL_FILE = "/Users/alxy/.openclaw/workspace/SecureFinAI_Contest_2026/Tutorials/Task_5_tutorial/daily_signals.json"

def load_signals():
    """Load stored signals from today"""
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE, "r") as f:
            data = json.load(f)
            # Reset if new day
            if data.get("date") != datetime.now().strftime("%Y-%m-%d"):
                return {"date": datetime.now().strftime("%Y-%m-%d"), "runs": []}
            return data
    return {"date": datetime.now().strftime("%Y-%m-%d"), "runs": []}

def save_signals(data):
    """Save signals to file"""
    data["date"] = datetime.now().strftime("%Y-%m-%d")
    with open(SIGNAL_FILE, "w") as f:
        json.dump(data, f)

def should_trade():
    """
    Determine if we should trade based on consensus
    Returns: (action, confidence)
    - action: "Buy", "Sell", or None
    - confidence: how many runs agree (1-3)
    """
    data = load_signals()
    runs = data.get("runs", [])
    
    if len(runs) < 2:
        return None, len(runs)
    
    # Count signals
    buy_count = sum(1 for r in runs if r.get("action") == "Buy")
    sell_count = sum(1 for r in runs if r.get("action") == "Sell")
    hold_count = sum(1 for r in runs if r.get("action") == "Hold")
    
    # Consensus: 2+ runs agree
    if buy_count >= 2:
        return "Buy", buy_count
    elif sell_count >= 2:
        return "Sell", sell_count
    else:
        return None, max(buy_count, sell_count, hold_count)

def record_run(symbol, action, nofx_score, sentiment):
    """Record a signal run"""
    data = load_signals()
    
    runs = data.get("runs", [])
    
    # Add new run
    runs.append({
        "time": datetime.now().strftime("%H:%M"),
        "symbol": symbol,
        "action": action,
        "nofx_score": nofx_score,
        "sentiment": sentiment
    })
    
    # Keep only today
    data["runs"] = runs
    save_signals(data)
    
    return runs

# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "check":
            action, conf = should_trade()
            print(f"Should trade: {action} (confidence: {conf}/3)")
            
        elif cmd == "reset":
            save_signals({"date": datetime.now().strftime("%Y-%m-%d"), "runs": []})
            print("Signals reset")
            
        elif cmd == "status":
            data = load_signals()
            print(f"Date: {data.get('date')}")
            print(f"Runs: {len(data.get('runs', []))}")
            for r in data.get("runs", []):
                print(f"  {r['time']}: {r['action']} {r.get('symbol', '')}")
    
    else:
        # Show current status
        data = load_signals()
        print(f"Today: {data.get('date')}")
        print(f"Runs: {len(data.get('runs', []))}")
        for r in data.get("runs", []):
            print(f"  {r['time']}: {r['action']}")
