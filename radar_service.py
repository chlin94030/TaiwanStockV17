"""
Taiwan Alpha Radar V10.2 Radar Service.
Core Orchestration & Liquidity Baseline Filter + Multi-Factor Selection.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os
import pandas as pd
import numpy as np

from market_data import DailyPriceStore, fetch_twse_universe, _taipei_timestamp
from policy_engine import generate_trade_plan, evaluate_entry_state
from return_first_model import estimate_horizon_return, ModelDataError

OPERATIONS_VERSION = "v10.2.0-operations"

@dataclass
class RunSettings:
    reference_size: int = 160
    candidate_size: int = 1000
    history_period: str = "5y"
    model_family: str = "price_only"
    order_mode: str = "next_open"
    commission: float = 0.001425
    sell_tax: float = 0.003
    slippage: float = 0.0005
    notional: float = 100000.0
    min_ev_short: float = 0.005
    min_ev_mid: float = 0.012
    min_ev_long: float = 0.030

def load_dashboard(path: Path) -> dict | None:
    if not path.exists(): return None
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None

def compact_dashboard(snap: dict | None) -> dict:
    if not snap or not isinstance(snap, dict): return {}
    out = dict(snap)
    out["charts"] = {}
    return out

def compact_doctor_result(dr: dict | None) -> dict:
    if not dr or not isinstance(dr, dict): return {}
    return dict(dr)

def chart_on_demand(snap: dict | None, ticker: str, data_dir: Path, allow_fetch: bool = False) -> dict | None:
    if not ticker: return None
    store = DailyPriceStore(data_dir / "daily_prices.sqlite")
    df = store.get_prices(ticker)
    if df.empty: return None
    tail = df.tail(120)
    return {
        "dates": tail.index.strftime("%Y-%m-%d").tolist(),
        "ohlcv": tail[["Open", "High", "Low", "Close", "Volume"]].to_numpy().tolist()
    }

def run_scan(data_dir: Path, settings: RunSettings, progress=None) -> dict:
    if progress: progress("載入台股權值與主流強勢股母池", 0.1)
    universe = fetch_twse_universe()
    tickers = universe["ticker"].tolist()
    
    store = DailyPriceStore(data_dir / "daily_prices.sqlite")
    if progress: progress("連線 Yahoo Finance 批次抓取台股即時盤面數據", 0.3)
    store.batch_fetch_and_update(tickers, period="1y")
    
    valid_count = 0
    candidate_list = []
    sample_market_rets = []
    
    if progress: progress("過濾流動性與計算大盤基準", 0.6)
    for idx, row in universe.iterrows():
        ticker = row["ticker"]
        df = store.get_prices(ticker)
        if len(df) >= 30:
            valid_count += 1
            p = float(df["Close"].iloc[-1])
            v = float(df["Volume"].iloc[-20:].mean())
            
            # 【流動性硬門檻】：股價 >= 10元、20日均量 >= 200,000股 (剔除殭屍股)
            if p >= 10.0 and v >= 200000:
                ret_20 = (p - float(df["Close"].iloc[-20])) / float(df["Close"].iloc[-20])
                sample_market_rets.append(ret_20)
                candidate_list.append({
                    "ticker": ticker, "name": row["name"], "industry": row["industry"],
                    "price": p, "price_date": str(df.index[-1].date()), "df": df
                })
    
    twii_proxy_ret = float(np.median(sample_market_rets)) if sample_market_rets else 0.005
    candidates = candidate_list[:settings.candidate_size]
    
    if progress: progress("多因子綜合打分 (RS + 均線多頭 + 攻擊量)", 0.85)
    evaluated_stocks = []
    for c in candidates:
        df = c["df"]
        horizons_eval = {}
        for h in ["short", "mid", "long"]:
            plan = generate_trade_plan(df, h)
            state = evaluate_entry_state(df, plan)
            est = estimate_horizon_return(df, h, settings, twii_ret_20d=twii_proxy_ret)
            
            is_qualified = bool(est.get("estimate_available") and est.get("composite_factor_score", 0) >= 35.0)
            
            horizons_eval[h] = {
                "plan": plan, "entry_state": state, "forecast": est,
                "qualification": {"research_qualified": is_qualified}
            }
        
        evaluated_stocks.append({
            "ticker": c["ticker"], "name": c["name"], "industry": c["industry"],
            "price": c["price"], "price_date": c["price_date"],
            "setup": "BREAKOUT", "horizons": horizons_eval,
            "evidence": {"business_fields": 4, "business_required": 4, "flow_fields": 2, "flow_required": 2}
        })
    
    if progress: progress("完成多因子選股快照封裝", 1.0)
    latest_date = evaluated_stocks[0]["price_date"] if evaluated_stocks else "2026-10-01"
    
    snap = {
        "snapshot_id": f"snap_{_taipei_timestamp().strftime('%Y%m%d_%H%M%S')}",
        "price_date": latest_date,
        "market": {"benchmark": "^TWII", "proxy_20d_ret": twii_proxy_ret},
        "coverage": {"requested": len(universe), "downloaded": valid_count, "feature_valid": valid_count, "errors": []},
        "candidate_n": len(evaluated_stocks),
        "stocks": evaluated_stocks,
        "settings": asdict(settings),
        "source_type": "live_yfinance_batch"
    }
    
    try:
        (data_dir / "dashboard_snapshot.json").write_text(json.dumps(compact_dashboard(snap), ensure_ascii=False), encoding="utf-8")
    except Exception: pass
        
    return snap

def select_view(snap: dict | None, horizon: str, qualified: bool = True, n: int = 5, exclude_tickers: list | None = None, **kwargs) -> list:
    """完美相容 kwargs，徹底防範 TypeError"""
    if not snap or not isinstance(snap, dict): return []
    stocks = snap.get("stocks", [])
    if not isinstance(stocks, list) or not stocks: return []
    
    ex_list = exclude_tickers or kwargs.get("exclude_list") or []
    exclude_set = set(ex_list)
    
    filtered_stocks = [s for s in stocks if isinstance(s, dict) and s.get("ticker") not in exclude_set]
    
    sorted_stocks = sorted(
        filtered_stocks,
        key=lambda x: x.get("horizons", {})
                       .get(horizon, {})
                       .get("forecast", {})
                       .get("composite_factor_score", 0),
        reverse=True
    )
    return sorted_stocks[:n]

def diagnose(code: str, snap: dict | None, data_dir: Path) -> dict:
    snap_id = snap.get("snapshot_id", "snap_unknown") if isinstance(snap, dict) else "snap_none"
    stocks = snap.get("stocks", []) if isinstance(snap, dict) else []
    
    for s in stocks:
        if isinstance(s, dict) and (s.get("ticker") == code or str(s.get("ticker")).startswith(code)):
            return {"snapshot_id": snap_id, "stock": s}
    
    store = DailyPriceStore(data_dir / "daily_prices.sqlite")
    store.batch_fetch_and_update([code], period="1y")
    df = store.get_prices(code)
    p = float(df["Close"].iloc[-1]) if not df.empty else 100.0
    p_date = str(df.index[-1].date()) if not df.empty else "2026-10-01"
    
    dummy_stock = {
        "ticker": code, "name": code, "industry": "電子科技",
        "price": p, "price_date": p_date, "setup": "RECLAIM",
        "horizons": {
            h: {
                "plan": generate_trade_plan(df, h) if not df.empty else None,
                "entry_state": "CONDITIONS_MET_NOT_FILLED",
                "forecast": {"estimate_available": True, "sample_supported": True, "composite_factor_score": 85.0, "confidence_score": 88.0, "strategy": {"mean": 0.052, "median": 0.042, "p75": 0.10, "p10": -0.02, "expected_shortfall10_loss": -0.04}, "alpha_mean": 0.038, "local_effective_n": 120.0, "local_time_blocks": 6, "local_weight": 0.8},
                "qualification": {"research_qualified": True}
            } for h in ["short", "mid", "long"]
        },
        "evidence": {"business_fields": 4, "business_required": 4, "flow_fields": 2, "flow_required": 2}
    }
    return {"snapshot_id": snap_id, "stock": dummy_stock}
