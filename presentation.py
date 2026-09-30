"""
Taiwan Alpha Radar V8.7 Presentation Helpers.
Clean Plain-Text Summaries.
"""
from __future__ import annotations

def plain_summary(f: dict) -> str:
    if not f or not isinstance(f, dict) or not f.get("estimate_available"):
        return "數據累積中，暫提供基本價量參考。"
    
    strat = f.get("strategy", {})
    ev = strat.get("mean", 0.0) * 100
    es10 = strat.get("expected_shortfall10_loss", 0.0) * 100
    alpha = f.get("alpha_mean", 0.0) * 100
    
    return f"預期淨收益 (EV) 為 {ev:+.1f}%（超額 Alpha {alpha:+.1f}%）；極端狀況下之平均損失約 {es10:.1f}%。"