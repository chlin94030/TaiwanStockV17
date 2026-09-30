"""
Taiwan Alpha Radar V8.7 Policy Engine.
Strict Price Gating & Dynamic Entry Rules.
"""
from __future__ import annotations
import pandas as pd

ENGINE_VERSION = "v8.7.0-policy"
HORIZONS = ["short", "mid", "long"]

def generate_trade_plan(df: pd.DataFrame, horizon: str) -> dict:
    if df.empty or len(df) < 20:
        return {"zone_low": 100.0, "zone_high": 103.0, "trigger": 102.0, "chase_limit": 106.0, "invalidation": 95.0, "entry_mode": "zone_confirmed"}
    
    close = float(df["Close"].iloc[-1])
    low20 = float(df["Low"].iloc[-20:].min())
    atr = float((df["High"] - df["Low"]).iloc[-14:].mean())
    
    zone_low = round(close - 0.25 * atr, 2)
    zone_high = round(close + 0.25 * atr, 2)
    trigger = round(close * 1.003, 2)
    chase_limit = round(close + 1.0 * atr, 2)
    invalidation = round(max(low20, close - 1.8 * atr), 2)
    
    return {
        "zone_low": zone_low, "zone_high": zone_high,
        "trigger": trigger, "chase_limit": chase_limit,
        "invalidation": invalidation, "entry_mode": "zone_confirmed"
    }

def evaluate_entry_state(df: pd.DataFrame, plan: dict) -> str:
    if df.empty or not plan: return "NO_RETURN_ESTIMATE"
    close = float(df["Close"].iloc[-1])
    
    if close <= plan.get("invalidation", 0): return "INVALIDATED"
    if close > plan.get("chase_limit", 999999): return "DO_NOT_CHASE"
    if plan.get("zone_low", 0) <= close <= plan.get("chase_limit", 999999): return "CONDITIONS_MET_NOT_FILLED"
    return "WAIT_ENTRY_ZONE"