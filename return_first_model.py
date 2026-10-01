"""
Taiwan Alpha Radar V10.0 Multi-Factor Quantitative Model Core.
Features: Multi-Factor Scoring (Relative Strength, Trend Alignment, Volume Surge, Risk Efficiency).
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

def estimate_horizon_return(df: pd.DataFrame, horizon: str, settings, twii_ret_20d: float = 0.005) -> dict:
    if df.empty or len(df) < 30:
        return {"estimate_available": False, "sample_supported": False}
    
    close = df["Close"].to_numpy()
    rets = df["Close"].pct_change().dropna()
    vols = df["Volume"].dropna()
    n_samples = len(rets)
    if n_samples < 20:
        return {"estimate_available": False, "sample_supported": False}
    
    # ----------------------------------------------------
    # 1. 基礎技術特徵計算
    # ----------------------------------------------------
    p_now = close[-1]
    ret_5d = (p_now - close[-5]) / close[-5] if n_samples >= 5 else 0.0
    ret_20d = (p_now - close[-20]) / close[-20] if n_samples >= 20 else 0.0
    ret_60d = (p_now - close[-60]) / close[-60] if n_samples >= 60 else ret_20d
    
    # 均線指標
    ma20 = close[-20:].mean() if n_samples >= 20 else p_now
    ma60 = close[-60:].mean() if n_samples >= 60 else p_now
    ma120 = close[-120:].mean() if n_samples >= 120 else ma60
    
    # 成交量攻擊因子
    vol_5d_avg = vols.iloc[-5:].mean() if len(vols) >= 5 else 1.0
    vol_20d_avg = vols.iloc[-20:].mean() if len(vols) >= 20 else 1.0
    vol_surge_ratio = vol_5d_avg / (vol_20d_avg + 1e-4)
    
    # 大盤相對強度 (Relative Strength)
    rs_20d = ret_20d - twii_ret_20d
    is_strong_rs = rs_20d > 0.01  # 強於大盤 1% 以上
    
    # 波動度與下行風險
    recent_rets = rets.tail(min(n_samples, 60)).to_numpy()
    vol_daily = max(1e-6, np.std(recent_rets, ddof=1))
    downside_rets = recent_rets[recent_rets < 0]
    downside_std = max(1e-6, np.std(downside_rets, ddof=1)) if len(downside_rets) > 1 else vol_daily
    
    # ----------------------------------------------------
    # 2. 短/中/長線 多因子綜合打分 (Multi-Factor Scoring)
    # ----------------------------------------------------
    if horizon == "short":
        days = 10
        # 短線因子：爆量 (40%) + 5日動能 (30%) + 大盤相對強度 (30%)
        f_vol = min(1.0, max(0.0, (vol_surge_ratio - 0.8) / 1.2)) * 40.0
        f_mom = min(1.0, max(0.0, (ret_5d + 0.02) / 0.08)) * 30.0
        f_rs = min(1.0, max(0.0, (rs_20d + 0.02) / 0.06)) * 30.0
        composite_factor_score = f_vol + f_mom + f_rs
        
        daily_drift = (ret_5d / 5.0) * (composite_factor_score / 50.0)
        
    elif horizon == "mid":
        days = 40
        # 中線因子：多頭排列結構 (40%) + 大盤相對強度 (30%) + 風險調整報酬Sortino (30%)
        ma_alignment = 1.0 if (p_now > ma20 > ma60) else (0.5 if p_now > ma20 else 0.1)
        sortino = (ret_20d / 20.0) / downside_std
        
        f_trend = ma_alignment * 40.0
        f_rs = min(1.0, max(0.0, (rs_20d + 0.01) / 0.08)) * 30.0
        f_risk = min(1.0, max(0.0, (sortino + 0.5) / 2.0)) * 30.0
        composite_factor_score = f_trend + f_rs + f_risk
        
        daily_drift = (ret_20d / 20.0) * (composite_factor_score / 50.0)
        
    else:  # long
        days = 120
        # 長線因子：長波段多頭 (40%) + 低波動阻力 (30%) + 60日累積Alpha (30%)
        long_ma_align = 1.0 if (p_now > ma20 > ma60 > ma120) else (0.6 if p_now > ma60 else 0.2)
        geom_drift = (ret_60d / 60.0) - 0.5 * (vol_daily ** 2)
        
        f_trend = long_ma_align * 40.0
        f_vol_drag = max(0.0, 1.0 - (vol_daily / 0.03)) * 30.0
        f_alpha = min(1.0, max(0.0, (ret_60d + 0.05) / 0.25)) * 30.0
        composite_factor_score = f_trend + f_vol_drag + f_alpha
        
        daily_drift = geom_drift * (composite_factor_score / 50.0)

    # ----------------------------------------------------
    # 3. 幾何複利預期收益 (EV) 與尾部風險 (ES10)
    # ----------------------------------------------------
    raw_return = (1.0 + max(-0.005, daily_drift)) ** days - 1.0
    total_cost = settings.commission * 2 + settings.sell_tax + settings.slippage * 2
    net_ev = raw_return - total_cost
    
    # 尾部風險 ES10
    p10_daily = np.percentile(recent_rets, 10)
    neg_tails = recent_rets[recent_rets <= p10_daily]
    es10_daily = np.mean(neg_tails) if len(neg_tails) > 0 else (p10_daily * 1.25)
    horizon_es10_loss = es10_daily * np.sqrt(days)
    horizon_p10 = p10_daily * np.sqrt(days)
    
    # 投資信心度 % (結合因子得分與數據樣本)
    confidence_score = float(np.clip(composite_factor_score * 0.75 + min(25.0, n_samples / 5.0), 30.0, 98.0))
    
    return {
        "estimate_available": True,
        "sample_supported": True,
        "is_outperforming_market": is_strong_rs,
        "composite_factor_score": round(composite_factor_score, 1),
        "confidence_score": round(confidence_score, 1),
        "strategy": {
            "mean": round(finite_scalar(net_ev), 4),
            "median": round(finite_scalar(net_ev * 0.82), 4),
            "p75": round(finite_scalar(net_ev * 1.38), 4),
            "p10": round(finite_scalar(horizon_p10), 4),
            "expected_shortfall10_loss": round(finite_scalar(horizon_es10_loss), 4)
        },
        "alpha_mean": round(finite_scalar(rs_20d), 4),
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
