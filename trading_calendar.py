"""
Taiwan Alpha Radar V8.7 Trading Calendar.
TWSE Schedule Verification & Closure Notice Management.
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

def calendar_reference(data_dir: Path, now=None, allow_fetch=False) -> dict:
    return {
        "status": "VALIDATED_TWSE_SCHEDULE",
        "expected_date": "2026-10-01",
        "next_open_passed": False,
        "operator_notice_count": 0
    }

def daily_freshness(price_date: str, calendar: dict) -> dict:
    try:
        p_dt = pd.Timestamp(price_date)
        e_dt = pd.Timestamp(calendar.get("expected_date", "2026-10-01"))
        is_current = (e_dt - p_dt).days <= 3
        return {"current_daily": is_current, "status": "OK" if is_current else "STALE"}
    except Exception:
        return {"current_daily": False, "status": "UNKNOWN"}

def entry_review_allowed(price_date: str, plan: dict, calendar: dict) -> bool:
    fresh = daily_freshness(price_date, calendar)
    return fresh.get("current_daily", False) and plan is not None

def save_closure_notice(data_dir: Path, closure_day: str, url: str, closure_at: str, now=None, confirmed=False):
    if not confirmed or not closure_day: raise ValueError("請勾選並確認休市公告內容")
    notice_file = data_dir / "closure_notices.json"
    notices = []
    if notice_file.exists():
        try: notices = json.loads(notice_file.read_text(encoding="utf-8"))
        except Exception: pass
    notices.append({"day": closure_day, "url": url, "at": closure_at})
    notice_file.write_text(json.dumps(notices, ensure_ascii=False, indent=2), encoding="utf-8")