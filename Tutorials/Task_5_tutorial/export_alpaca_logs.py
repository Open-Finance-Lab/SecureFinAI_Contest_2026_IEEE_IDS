#!/usr/bin/env python3
"""
Export from Alpaca Paper Trading Account.

Outputs:
A) Orders JSONL (filled orders):
   alpaca_export_orders_<timestamp>.jsonl

B) Daily portfolio value CSV (equity from portfolio history):
   alpaca_export_daily_equity_<timestamp>.csv
   columns: date,equity

Environment variables:
  APCA_API_KEY_ID
  APCA_API_SECRET_KEY
  APCA_API_BASE_URL (optional; default: https://paper-api.alpaca.markets)

Run this:
  python export_alpaca_log.py --start 2026-04-20 --end 2026-05-01
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests
from dateutil import tz


@dataclass
class AlpacaConfig:
    key_id: str
    secret_key: str
    api_base: str


def headers(cfg: AlpacaConfig) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": cfg.key_id,
        "APCA-API-SECRET-KEY": cfg.secret_key,
        "Accept": "application/json",
    }


def alpaca_get(cfg: AlpacaConfig, path: str, params: Dict) -> dict:
    url = cfg.api_base.rstrip("/") + path
    r = requests.get(url, headers=headers(cfg), params=params, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} failed ({r.status_code}): {r.text}")
    return r.json()


def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--period", help="Portfolio history period (e.g., 1M, 3M, 1A).")
    g.add_argument("--start", help="Start date (YYYY-MM-DD). If set, --end is recommended.")
    p.add_argument("--end", help="End date (YYYY-MM-DD). Inclusive end for output.")
    p.add_argument("--tz", default="America/New_York", help="Timezone for daily bucketing.")
    p.add_argument("--outdir", default=".", help="Output directory.")
    p.add_argument("--orders_status", default="all", help="Orders status filter (all/open/closed).")
    p.add_argument(
        "--include_partially_filled",
        action="store_true",
        help="Include partially_filled orders too (still requires filled_qty > 0).",
    )
    args = p.parse_args()

    if not args.period and not args.start:
        args.period = "1M"

    return args


def get_portfolio_history_daily_equity(
    cfg: AlpacaConfig,
    start_utc: Optional[datetime],
    end_exclusive_utc: Optional[datetime],
    period: str,
    local_tz,
) -> List[Dict]:
    params = {"timeframe": "1D", "extended_hours": "false"}

    if start_utc is None:
        params["period"] = period
    else:
        params["start"] = iso_utc(start_utc)
        if end_exclusive_utc is not None:
            params["end"] = iso_utc(end_exclusive_utc)

    data = alpaca_get(cfg, "/v2/account/portfolio/history", params)
    equity = data.get("equity", [])
    timestamps = data.get("timestamp", [])

    if not equity or not timestamps or len(equity) != len(timestamps):
        raise RuntimeError(f"Unexpected portfolio history response keys={list(data.keys())}")

    by_date: Dict[str, float] = {}
    for ts_value, eq_value in zip(timestamps, equity):
        dt_utc = datetime.fromtimestamp(int(ts_value), tz=timezone.utc)
        local_date = dt_utc.astimezone(local_tz).date().isoformat()
        by_date[local_date] = float(eq_value)

    dates = sorted(by_date)
    min_date = datetime.fromisoformat(dates[0]).date()
    max_date = datetime.fromisoformat(dates[-1]).date()

    rows = []
    last_equity = None
    d = min_date
    while d <= max_date:
        ds = d.isoformat()
        if ds in by_date:
            last_equity = by_date[ds]
        if last_equity is None:
            last_equity = by_date[dates[0]]
        rows.append({"date": ds, "equity": float(last_equity)})
        d += timedelta(days=1)

    return rows


def list_orders_all(cfg: AlpacaConfig, after_utc: datetime, until_utc: datetime, status: str) -> List[dict]:
    orders: List[dict] = []
    after = after_utc
    limit = 500

    while True:
        params = {
            "status": status,
            "after": iso_utc(after),
            "until": iso_utc(until_utc),
            "direction": "asc",
            "limit": limit,
            "nested": "true",
        }
        batch = alpaca_get(cfg, "/v2/orders", params)

        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected /v2/orders response type: {type(batch)}")
        if not batch:
            break

        orders.extend(batch)

        if len(batch) < limit:
            break

        last_submitted = batch[-1].get("submitted_at")
        if not last_submitted:
            break

        after = datetime.fromisoformat(last_submitted.replace("Z", "+00:00")) + timedelta(seconds=1)

    return orders


def write_orders_jsonl(path: str, orders: List[dict], include_partially_filled: bool) -> int:
    allowed = {"filled"}
    if include_partially_filled:
        allowed.add("partially_filled")

    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for order in orders:
            status = order.get("status")
            if status not in allowed:
                continue

            filled_at = order.get("filled_at")
            filled_qty = order.get("filled_qty")
            filled_avg_price = order.get("filled_avg_price")

            if not filled_at or filled_qty is None or filled_avg_price is None:
                continue
            if float(filled_qty) <= 0:
                continue

            symbol = order.get("symbol")
            asset_class = order.get("asset_class") or ("crypto" if symbol and "/" in symbol else "us_equity")

            record = {
                "order_id": order.get("id"),
                "client_order_id": order.get("client_order_id"),
                "symbol": symbol,
                "asset_class": asset_class,
                "side": (order.get("side") or "").upper(),
                "qty": float(order["qty"]) if order.get("qty") is not None else None,
                "status": status,
                "submitted_at": order.get("submitted_at"),
                "type": order.get("type"),
                "time_in_force": order.get("time_in_force"),
                "filled_at": filled_at,
                "filled_qty": float(filled_qty),
                "filled_avg_price": float(filled_avg_price),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def write_equity_csv(path: str, rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "equity"])
        for row in rows:
            writer.writerow([row["date"], row["equity"]])


def main() -> None:
    args = parse_args()

    key_id = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    api_base = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets").strip()

    if not key_id or not secret_key:
        raise SystemExit("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in environment variables.")

    cfg = AlpacaConfig(key_id=key_id, secret_key=secret_key, api_base=api_base)

    local_tz = tz.gettz(args.tz)
    if local_tz is None:
        raise SystemExit(f"Invalid timezone: {args.tz}")

    start_utc = None
    end_exclusive_utc = None
    if args.start:
        start_utc = datetime.fromisoformat(args.start).replace(tzinfo=local_tz).astimezone(timezone.utc)
        end_local = (
            datetime.fromisoformat(args.end).replace(tzinfo=local_tz)
            if args.end
            else datetime.now(tz=local_tz)
        )
        end_exclusive_utc = (end_local + timedelta(days=1)).astimezone(timezone.utc)

    equity_rows = get_portfolio_history_daily_equity(
        cfg, start_utc, end_exclusive_utc, args.period, local_tz
    )

    min_date = datetime.fromisoformat(equity_rows[0]["date"]).date()
    max_date = datetime.fromisoformat(equity_rows[-1]["date"]).date()

    orders_after = (
        datetime.combine(min_date, datetime.min.time())
        .replace(tzinfo=local_tz)
        .astimezone(timezone.utc)
        - timedelta(days=2)
    )
    orders_until = (
        datetime.combine(max_date + timedelta(days=1), datetime.min.time())
        .replace(tzinfo=local_tz)
        .astimezone(timezone.utc)
        + timedelta(days=2)
    )

    orders = list_orders_all(cfg, orders_after, orders_until, status=args.orders_status)

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    orders_path = os.path.join(args.outdir, f"alpaca_export_orders_{stamp}.jsonl")
    equity_path = os.path.join(args.outdir, f"alpaca_export_daily_equity_{stamp}.csv")

    n_orders = write_orders_jsonl(
        orders_path, orders, include_partially_filled=args.include_partially_filled
    )
    write_equity_csv(equity_path, equity_rows)

    print("Wrote:")
    print(" ", orders_path, f"({n_orders} orders)")
    print(" ", equity_path, f"({len(equity_rows)} days)")


if __name__ == "__main__":
    main()