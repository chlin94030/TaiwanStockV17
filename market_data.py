"""
Taiwan Alpha Radar Market Data Engine V10.2.
Batch yfinance Engine (Fast & Anti-Rate Limit) + SQLite Store.
"""
from __future__ import annotations

import sqlite3
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

def _taipei_timestamp() -> datetime.datetime:
    tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz)

def fetch_twse_universe() -> pd.DataFrame:
    """台股主流權值與強勢飆股母池"""
    top_tickers = [
        ("2330.TW", "台積電", "半導體"), ("2317.TW", "鴻海", "其他電子"),
        ("2454.TW", "聯發科", "半導體"), ("2382.TW", "廣達", "電腦及週邊"),
        ("3231.TW", "緯創", "電腦及週邊"), ("3017.TW", "奇鋐", "電機機械"),
        ("6669.TW", "緯穎", "電腦及週邊"), ("2356.TW", "英業達", "電腦及週邊"),
        ("2603.TW", "長榮", "航運業"), ("2609.TW", "陽明", "航運業"),
        ("1519.TW", "華城", "電機機械"), ("1504.TW", "東元", "電機機械"),
        ("2308.TW", "台達電", "電子零組件"), ("3034.TW", "聯詠", "半導體"),
        ("2379.TW", "瑞昱", "半導體"), ("3443.TW", "創意", "半導體"),
        ("3661.TW", "世芯-KY", "半導體"), ("2303.TW", "聯電", "半導體"),
        ("2881.TW", "富邦金", "金融保險"), ("2882.TW", "國泰金", "金融保險"),
        ("2345.TW", "智邦", "通信網路"), ("3037.TW", "欣興", "電子零組件"),
        ("2376.TW", "技嘉", "電腦及週邊"), ("2357.TW", "華碩", "電腦及週邊"),
        ("6274.TWO", "台燿", "電子零組件"), ("3264.TWO", "欣銓", "半導體"),
        ("8299.TWO", "群聯", "半導體"), ("6187.TWO", "萬潤", "半導體設備"),
        ("3583.TWO", "辛耘", "半導體設備"), ("3131.TWO", "弘塑", "半導體設備"),
        ("8046.TW", "南電", "電子零組件"), ("2002.TW", "中鋼", "鋼鐵工業"),
        ("1301.TW", "台塑", "塑膠工業"), ("1303.TW", "南亞", "塑膠工業")
    ]
    return pd.DataFrame(top_tickers, columns=["ticker", "name", "industry"])

class DailyPriceStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM daily_prices")

    def batch_fetch_and_update(self, tickers: list[str], period: str = "1y") -> None:
        """一次性批次連線下載所有標的，解決網絡卡頓與 Rate-Limit 問題"""
        try:
            data = yf.download(tickers, period=period, group_by="ticker", progress=False, threads=True)
            records = []
            for t in tickers:
                try:
                    df_t = data[t].dropna(how="all") if len(tickers) > 1 else data.dropna(how="all")
                    if df_t.empty: continue
                    df_t.index = df_t.index.strftime("%Y-%m-%d")
                    for idx, row in df_t.iterrows():
                        p_close = float(row.get("Close", 0))
                        if p_close > 0:
                            records.append((
                                t, str(idx), float(row.get("Open", p_close)),
                                float(row.get("High", p_close)), float(row.get("Low", p_close)),
                                p_close, float(row.get("Volume", 0))
                            ))
                except Exception: continue
            
            if records:
                with sqlite3.connect(self.db_path) as conn:
                    conn.executemany("""
                        INSERT OR REPLACE INTO daily_prices (ticker, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, records)
        except Exception: pass

    def get_prices(self, ticker: str) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT date, open as Open, high as High, low as Low, close as Close, volume as Volume FROM daily_prices WHERE ticker = ? ORDER BY date ASC",
                conn, params=(ticker,)
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        return df
