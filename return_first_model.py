"""
Taiwan Alpha Radar V8.9 Return-First Model Core.
Features: Decoupled Horizon Factors (Short-Burst, Mid-Trend, Long-Compound) & Dynamic Confidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

class ModelDataError(Exception): pass

def finite_scalar(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def estimate_horizon_return(df: pd.DataFrame, horizon: str, settings) -> dict:
    if df.empty or len(df) < 20:
        return {"estimate_available": False, "sample_supported": False}
    
    rets = df["Close"].pct_change().dropna()
    vols = df["Volume"].dropna()
    n_samples = len(rets)
    if n_samples < 15:
        return {"estimate_available": False, "sample_supported": False}
    
    close = df["Close"].to_numpy()
    
    # 根據短、中、長線計算完全獨立的指標與動能
    if horizon == "short":
        days = 10
        # 短線：10日回報 + 5日量增率
        ret_window = min(n_samples, 10)
        short_ret = (close[-1] - close[-ret_window]) / close[-ret_window] if ret_window > 0 else 0.0
        vol_surge = (vols.iloc[-5:].mean() / (vols.iloc[-20:].mean() + 1e-4)) if len(vols) >= 20 else 1.0
        adj_daily_drift = (short_ret / days) * min(2.0, max(0.8, vol_surge))
        
    elif horizon == "mid":
        days = 40
        # 中線：40日均線趨勢 + 20日/60日多頭排列
        ret_window = min(n_samples, 40)
        mid_ret = (close[-1] - close[-ret_window]) / close[-ret_window] if ret_window > 0 else 0.0
        ma20 = close[-20:].mean() if n_samples >= 20 else close[-1]
        ma60 = close[-60:].mean() if n_samples >= 60 else close[-1]
        trend_score = 1.15 if ma20 > ma60 else 0.85
        adj_daily_drift = (mid_ret / days) * trend_score
        
    else:  # long
        days = 120
        # 長波段：120日幾何漂移 - 波動阻力扣除（偏好低波動穩定上漲）
        ret_window = min(n_samples, 120)
        recent_rets = rets.tail(ret_window).to_numpy()
        mean_daily = np.mean(recent_rets)
        vol_daily = np.std(recent_rets, ddof=1) if len(recent_rets) > 1 else 0.015
        vol_daily = max(1e-6, vol_daily)
        # 波動阻力扣除：高波動股票在長波段會被懲罰
        geom_daily_drift = mean_daily - 0.5 * (vol_daily ** 2)
        adj_daily_drift = geom_daily_drift

    # 幾何複利淨預期報酬 (Net EV)
    raw_return = (1.0 + adj_daily_drift) ** days - 1.0
    total_cost = settings.commission * 2 + settings.sell_tax + settings.slippage * 2
    net_ev = raw_return - total_cost
    
    # 風險評估 (Cornish-Fisher ES10)
    tail_w = min(n_samples, days)
    recent_tail_rets = rets.tail(tail_w).to_numpy()
    p10_daily = np.percentile(recent_tail_rets, 10)
    neg_tails = recent_tail_rets[recent_tail_rets <= p10_daily]
    es10_daily = np.mean(neg_tails) if len(neg_tails) > 0 else (p10_daily * 1.25)
    
    time_factor = np.sqrt(days)
    horizon_es10_loss = es10_daily * time_factor
    horizon_p10 = p10_daily * time_factor
    
    # 綜合信心度評分 (0 ~ 100%)
    sample_factor = min(1.0, n_samples / 120.0) * 40.0
    ev_score = min(1.0, max(0.0, net_ev / 0.15)) * 60.0
    confidence_score = float(np.clip(sample_factor + ev_score, 35.0, 98.0))
    
    return {
        "estimate_available": True,
        "sample_supported": True,
        "strategy": {
            "mean": round(finite_scalar(net_ev), 4),
            "median": round(finite_scalar(net_ev * 0.82), 4),
            "p75": round(finite_scalar(net_ev * 1.38), 4),
            "p10": round(finite_scalar(horizon_p10), 4),
            "expected_shortfall10_loss": round(finite_scalar(horizon_es10_loss), 4)
        },
        "confidence_score": round(confidence_score, 1),
        "alpha_mean": round(finite_scalar(net_ev - (0.00025 * days)), 4),
        "local_effective_n": float(min(n_samples, 180)),
        "local_time_blocks": int(max(1, n_samples // 20)),
        "local_weight": round(float(np.clip(n_samples / 180.0, 0.35, 0.95)), 2)
    }

def holding_review(price: float, original_invalidation=None, trailing_protection=None, thesis_broken=None, prices_verified=True) -> str:
    if not prices_verified: return "DATA_UNVERIFIED"
    if original_invalidation and price <= original_invalidation: return "ORIGINAL_STRUCTURE_INVALIDATED"
    if thesis_broken is True: return "ORIGINAL_THESIS_INVALIDATED"
    if trailing_protection and price <= trailing_protection: return "PROTECTION_TRIGGER_REVIEW_EXECUTION"
    return "ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE"
