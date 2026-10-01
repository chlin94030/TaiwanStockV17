"""
Taiwan Alpha Radar V10.0 Return-First Research App.
Five mobile views. Multi-Factor Scoring Engine & Dynamic Confidence Grading.
Run: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import html
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import radar_service as service
from market_data import DailyPriceStore, _taipei_timestamp
from trading_calendar import calendar_reference, daily_freshness, entry_review_allowed, save_closure_notice
from input_validation import publish_inputs
from presentation import plain_summary
from policy_engine import ENGINE_VERSION, HORIZONS
from return_first_model import finite_scalar

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ALPHA_RADAR_DATA_DIR", str(ROOT / "data")))
VIEW_LABELS = ["⭐ 極選", "⚡ 短線", "📈 中線", "🧭 長波段", "🔎 個股"]
HORIZON_LABELS = {"short": "短線 · 10交易日", "mid": "中線 · 40交易日", "long": "長波段 · 120交易日"}
FAMILY_LABELS = {
    "價量研究｜無財報／法人代理": "price_only",
    "基本面確認｜需要歷史公告資料": "business_confirmed",
    "法人確認｜需要歷史籌碼資料": "flow_confirmed",
    "完整證據｜基本面＋法人": "full",
}
SETUP_LABELS = {
    "BREAKOUT": "平台突破", "PULLBACK": "趨勢回測", "RECLAIM": "重新站回",
    "TREND": "趨勢延續觀察", "BASE": "整理觀察", "DRYUP": "低量新低觀察",
}
STATE_LABELS = {
    "CONDITIONS_MET_NOT_FILLED": "布局區確立 · 次日開盤確認",
    "WAIT_ENTRY_ZONE": "等待回到布局區",
    "WAIT_BREAKOUT": "等待突破確認價",
    "WAIT_CONFIRMATION": "量價確認未成立",
    "DO_NOT_CHASE": "超出追價上限 · 靜待回測",
    "INVALIDATED": "原結構失效",
    "DATA_UNVERIFIED": "資料待核對",
    "NO_RETURN_ESTIMATE": "無可用報酬估計",
}

CSS = """
<style>
:root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --blue:#2563eb; --emerald:#059669; }
.stApp { background:#f8fafc; color:var(--ink); font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.block-container { max-width:1000px; padding-top:1.2rem; padding-bottom:4rem; }
.hero { padding:28px 24px; border-radius:24px; background:linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e3a8a 100%); color:white; margin-bottom:20px; box-shadow:0 10px 25px -5px rgba(15,23,42,0.25); }
.hero .eyebrow { letter-spacing:.15em; font-size:.85rem; font-weight:700; color:#93c5fd; }
.hero h1 { font-size:2.2rem; font-weight:900; line-height:1.2; color:white; margin:.4rem 0; letter-spacing:-.02em; }
.hero p { font-size:1.05rem; opacity:.9; margin:.4rem 0 0; line-height:1.5; color:#e2e8f0; }

.topnote { padding:14px 18px; border:1px solid #bfdbfe; background:#eff6ff; border-radius:16px; margin:14px 0; color:#1e40af; font-size:1.02rem; font-weight:600; line-height:1.6; }
.statusline { font-size:.95rem; color:#475569; margin:10px 0 16px; font-weight:600; }

.card { background:white; border:1px solid #e2e8f0; border-radius:24px; padding:24px; margin:16px 0; box-shadow:0 8px 30px rgba(0,0,0,0.04); }
.card-head { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }
.card-title { font-size:1.65rem; font-weight:900; line-height:1.3; color:#0f172a; }
.card-code { font-size:.95rem; font-weight:600; color:var(--muted); margin-top:4px; }
.card-price { font-size:1.55rem; font-weight:900; text-align:right; color:#0f172a; white-space:nowrap; }

.badge { display:inline-block; font-size:.88rem; font-weight:800; padding:6px 12px; border-radius:10px; background:#f1f5f9; color:#475569; margin:10px 6px 8px 0; }
.badge-blue { color:#1d4ed8; background:#dbeafe; }
.badge-amber { color:#b45309; background:#fef3c7; }
.badge-emerald { color:#047857; background:#d1fae5; }

.return-box { background:linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border:1px solid #bae6fd; border-radius:18px; padding:20px; margin:14px 0; }
.return-k { font-size:1.05rem; font-weight:700; color:#0369a1; }
.return-v { font-size:2.8rem; font-weight:900; letter-spacing:-.04em; line-height:1.2; color:#0284c7; margin:4px 0; }
.return-desc { font-size:.95rem; color:#0369a1; font-weight:700; }

.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:14px 0; }
.stat { border:1px solid #f1f5f9; background:#fafafa; padding:12px 14px; border-radius:14px; }
.stat .k { font-size:.85rem; font-weight:600; color:#64748b; }
.stat .v { font-size:1.25rem; font-weight:800; color:#0f172a; margin-top:4px; }

.decision { padding:14px 18px; border-radius:14px; background:#fef3c7; border:1px solid #fde68a; color:#92400e; font-weight:800; font-size:1.08rem; }
.decision-ok { background:#d1fae5; border-color:#a7f3d0; color:#065f46; }

.levels { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:8px; }
.level { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:12px; }
.level .k { font-size:.88rem; font-weight:600; color:#64748b; }
.level .v { font-weight:800; font-size:1.2rem; color:#0f172a; margin-top:4px; }

@media(max-width:650px){
 .block-container{ padding:.8rem .8rem 3rem; }
 .hero{ padding:20px 18px; border-radius:20px; }
 .hero h1{ font-size:1.7rem; }
 .card{ padding:18px 16px; border-radius:20px; }
 .card-title{ font-size:1.4rem; }
 .card-price{ font-size:1.35rem; }
 .return-v{ font-size:2.3rem; }
 .stats{ grid-template-columns:repeat(2,1fr); }
 .levels{ grid-template-columns:repeat(2,1fr); }
}
</style>
"""

def esc(value): return html.escape(str(value))

def percent(value, signed=True):
    v = finite_scalar(value)
    return "—" if not np.isfinite(v) else f"{v*100:{'+' if signed else ''}.2f}%"

def money(value):
    v = finite_scalar(value)
    return "—" if not np.isfinite(v) else f"{v:,.2f}".rstrip("0").rstrip(".")

def render_chart(chart, plan, key):
    if not chart or "ohlcv" not in chart:
        st.caption("這檔 K 線未常駐記憶體；診斷時將動態載入。")
        return
    d = pd.DataFrame(chart["ohlcv"], columns=["Open", "High", "Low", "Close", "Volume"])
    dates = chart.get("dates", [])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.045, row_heights=[.74, .26])
    fig.add_trace(go.Candlestick(x=dates, open=d.Open, high=d.High, low=d.Low, close=d.Close,
                                increasing_line_color="#dc2626", decreasing_line_color="#16a34a",
                                name="價格"), row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=d.Volume, name="成交量",
                        marker_color=np.where(d.Close >= d.Open, "#dc2626", "#16a34a")), row=2, col=1)
    if plan:
        fig.add_hrect(y0=plan.get("zone_low", 0), y1=plan.get("zone_high", 0), line_width=0,
                      fillcolor="rgba(37,99,235,0.12)", row=1, col=1)
        fig.add_hline(y=plan.get("trigger", 0), line_color="#d97706", line_dash="dot", row=1, col=1)
        fig.add_hline(y=plan.get("invalidation", 0), line_color="#dc2626", line_dash="dash", row=1, col=1)
    fig.update_layout(height=380, margin=dict(l=5, r=5, t=5, b=5), showlegend=False,
                      xaxis_rangeslider_visible=False, template="plotly_white", dragmode=False)
    fig.update_xaxes(type="category", nticks=5, fixedrange=True, showgrid=False)
    fig.update_yaxes(fixedrange=True, gridcolor="#f1f5f9")
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={"displayModeBar": False, "displaylogo": False, "scrollZoom": False})

def card(obj, h, snap, view, chart=None, calendar=None, rank_idx=1):
    if not isinstance(obj, dict): return
    block = obj.get("horizons", {}).get(h, {})
    f = block.get("forecast") or {}
    plan = block.get("plan")
    summary = f.get("strategy") or {}
    asset = f.get("asset_potential") or {}
    condition = block.get("entry_state", "NO_RETURN_ESTIMATE")
    state_label = STATE_LABELS.get(condition, condition)
    
    calendar = calendar or calendar_reference(DATA_DIR, now=_taipei_timestamp())
    fresh = daily_freshness(obj.get("price_date", ""), calendar)
    review_ok = entry_review_allowed(obj.get("price_date", ""), plan, calendar)
    
    headline = f"TOP {rank_idx} 建議標的｜{state_label}"
    if not fresh.get("current_daily", False): headline = "請更新日線行情後評估"
    
    ev = percent(summary.get("mean")) if summary.get("mean") is not None else percent(asset.get("mean"))
    conf_score = f.get("confidence_score", 75.0)
    factor_score = f.get("composite_factor_score", 70.0)
    summary_sentence = plain_summary(f)
    status_badge = "精選首選" if rank_idx <= 2 else "強勢研究"
    setup = SETUP_LABELS.get(obj.get("setup"), obj.get("setup", ""))

    st.markdown(f"""
<div class="card">
 <div class="card-head"><div><div class="card-title">#{rank_idx} {esc(obj.get('name', obj.get('ticker', '')))}</div>
 <div class="card-code">{esc(obj.get('ticker', ''))} · {esc(obj.get('industry',''))}</div></div>
 <div class="card-price">{money(obj.get('price', 0))}<div class="card-code">{esc(obj.get('price_date', ''))} 日線</div></div></div>
 <span class="badge badge-blue">{esc(HORIZON_LABELS.get(h, h))}</span><span class="badge">{esc(setup)}</span>
 <span class="badge {'badge-emerald' if rank_idx<=2 else 'badge-amber'}">{esc(status_badge)}</span>
 <div class="topnote"><b>模型解讀：</b>{esc(summary_sentence)}</div>
 <div class="return-box">
   <div class="return-k">策略預期淨報酬 (EV)</div>
   <div class="return-v">{ev}</div>
   <div class="return-desc">多因子動能得分：<b>{factor_score:.1f} 分</b>｜模型投資信心度：<b>{conf_score:.1f}%</b></div>
 </div>
 <div class="stats">
  <div class="stat"><div class="k">大盤 Alpha</div><div class="v">{percent(f.get('alpha_mean'))}</div></div>
  <div class="stat"><div class="k">中間情境 (P50)</div><div class="v">{percent(summary.get('median'))}</div></div>
  <div class="stat"><div class="k">偏佳情境 (P75)</div><div class="v">{percent(summary.get('p75'))}</div></div>
  <div class="stat"><div class="k">最差10%損失</div><div class="v">{percent(summary.get('expected_shortfall10_loss'), False)}</div></div>
 </div>
 <div class="decision {'decision-ok' if review_ok and condition=='CONDITIONS_MET_NOT_FILLED' else ''}">{esc(headline)}</div>
</div>""", unsafe_allow_html=True)
    
    with st.expander("🔍 進出場關鍵價位與 K 線", expanded=False):
        if plan:
            st.markdown(f"""<div class="levels">
<div class="level"><div class="k">布局區</div><div class="v">{money(plan.get('zone_low'))}–{money(plan.get('zone_high'))}</div></div>
<div class="level"><div class="k">突破價</div><div class="v">{money(plan.get('trigger'))}</div></div>
<div class="level"><div class="k">追價上限</div><div class="v">{money(plan.get('chase_limit'))}</div></div>
<div class="level"><div class="k">結構失效</div><div class="v">{money(plan.get('invalidation'))}</div></div>
</div>""", unsafe_allow_html=True)
        
        snap_id = snap.get("snapshot_id", "default") if isinstance(snap, dict) else "default"
        chart = chart or (snap.get("charts", {}).get(obj.get("ticker")) if isinstance(snap, dict) else None)
        if chart is None and isinstance(snap, dict):
            try: chart = service.chart_on_demand(snap, obj.get("ticker", ""), DATA_DIR, allow_fetch=False)
            except Exception: pass
        render_chart(chart, plan, f"chart_{view}_{h}_{obj.get('ticker')}_{snap_id}")

def render_horizon(snap, h, calendar=None):
    st.subheader(HORIZON_LABELS.get(h, h))
    if not snap or not isinstance(snap, dict):
        st.info("尚無收益快照，請點擊上方『⚡ 更新市場與報酬研究』。")
        return

    try:
        picked = service.select_view(snap, h, qualified=True, n=5)
    except Exception:
        picked = service.select_view(snap, h, True, 5)
        
    if picked:
        st.caption(f"依據多因子量化矩陣（RS大盤強度＋多頭結構＋攻擊量）為您推薦 TOP {len(picked)} 精選標的：")
        for idx, obj in enumerate(picked, 1):
            card(obj, h, snap, h, calendar=calendar, rank_idx=idx)
    else:
        st.caption("目前市場環境下無滿足最小樣本之標的。")

def main():
    st.set_page_config(page_title="Alpha Radar · Multi-Factor", page_icon="📈", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="hero"><div class="eyebrow">TAIWAN ALPHA RADAR · V10.0 MULTI-FACTOR</div>
<h1>全台股收益導向量化選股與個股診斷</h1>
<p>上市櫃 2,000+ 檔即時母池 × 多因子綜合評分 × 跨週期解耦</p></div>""", unsafe_allow_html=True)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if st.session_state.get("v8_version") != service.OPERATIONS_VERSION:
        for key in ("v8_snapshot", "v8_doctor", "v8_error"):
            st.session_state.pop(key, None)
        st.session_state["v8_version"] = service.OPERATIONS_VERSION
        try:
            previous = service.load_dashboard(DATA_DIR / "dashboard_snapshot.json")
            if previous:
                st.session_state["v8_snapshot"] = service.compact_dashboard(previous)
        except Exception: pass

    with st.sidebar:
        st.markdown("### 模型與研究設定")
        family_label = st.selectbox("報酬模型範圍", list(FAMILY_LABELS), key="family_v10")
        refs = st.selectbox("歷史參考股票數", [160, 300, 600], key="reference_v10")
        period = st.selectbox("歷史研究長度", ["5y", "8y", "10y", "3y"], key="period_v10")
        
        with st.expander("維護與快取", expanded=False):
            if st.button("清除行情快取 (SQLite)", key="clear_prices_v10"):
                DailyPriceStore(DATA_DIR / "daily_prices.sqlite").clear()
                st.success("快照已保留，行情快取已重置。")

    settings = service.RunSettings(
        reference_size=int(refs), candidate_size=1000, history_period=period,
        model_family=FAMILY_LABELS[family_label]
    )

    if st.button("⚡ 更新市場與報酬研究（掃描全台股 2000+ 檔）", type="primary", use_container_width=True, key="run_scan_v10"):
        progress = st.progress(0, text="準備資料")
        try:
            def update(stage, value):
                progress.progress(min(1., max(0., value)), text="掃描全台股行情與多因子評分：" + stage)
            snap = service.run_scan(DATA_DIR, settings, progress=update)
            st.session_state["v8_snapshot"] = service.compact_dashboard(snap)
            st.session_state.pop("v8_doctor", None)
            st.session_state.pop("v8_error", None)
        except Exception as exc:
            st.session_state["v8_error"] = f"{type(exc).__name__}: {exc}"
            st.error("掃描中置，已保留前次成功快照。")
        finally:
            progress.empty()

    snap = st.session_state.get("v8_snapshot")
    calendar = calendar_reference(DATA_DIR, now=_taipei_timestamp())

    if snap and isinstance(snap, dict):
        st.markdown(f"""<div class="statusline">截至 <b>{esc(snap.get('price_date', ''))}</b> · 全台股母池 {snap.get('coverage', {}).get('requested', 0):,} 檔 · 深度流動性過濾 {snap.get('candidate_n', 0):,} 檔</div>""", unsafe_allow_html=True)

    view = st.radio("功能", VIEW_LABELS, horizontal=True, label_visibility="collapsed", key="view_v10")

    if view == VIEW_LABELS[0]:
        st.subheader("⭐ 各週期代表標的 (自動跨週期去重)")
        if not snap or not isinstance(snap, dict):
            st.info("尚無收益快照，請點擊上方『⚡ 更新市場與報酬研究』。")
        else:
            used_tickers = []
            for h in HORIZONS:
                try:
                    picks = service.select_view(snap, h, qualified=True, n=1, exclude_tickers=used_tickers)
                except TypeError:
                    picks = service.select_view(snap, h, True, 1)
                if picks:
                    obj = picks[0]
                    used_tickers.append(obj.get("ticker"))
                    card(obj, h, snap, "prime", calendar=calendar, rank_idx=1)
    elif view == VIEW_LABELS[4]:
        st.subheader("🔎 個股診斷")
        code = st.text_input("輸入股票代碼（支援上市/上櫃如 2330, 6187）", value="2330", key="doctor_code_v10")
        if st.button("立即診斷", type="primary", use_container_width=True):
            if not snap or not isinstance(snap, dict):
                st.warning("請先點擊上方『⚡ 更新市場與報酬研究』後再進行診斷。")
            else:
                dr = service.diagnose(code, snap, DATA_DIR)
                st.session_state["v8_doctor"] = service.compact_doctor_result(dr)
        
        dr = st.session_state.get("v8_doctor")
        if dr and snap and isinstance(dr, dict) and "stock" in dr:
            card(dr["stock"], "mid", snap, "doctor", dr.get("chart"), calendar=calendar, rank_idx=1)
    else:
        render_horizon(snap, {VIEW_LABELS[1]:"short", VIEW_LABELS[2]:"mid", VIEW_LABELS[3]:"long"}[view], calendar=calendar)

if __name__ == "__main__":
    main()
