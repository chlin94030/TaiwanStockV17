"""
Taiwan Alpha Radar V8.7 Input Validation.
"""
from __future__ import annotations
from pathlib import Path
import io
import pandas as pd

def publish_inputs(data_dir: Path, uploads: dict, units_mode: str = "decimal", acknowledge_extremes: bool = False):
    for filename, content in uploads.items():
        df = pd.read_csv(io.BytesIO(content))
        if units_mode == "percent_points":
            for col in ["revenue_yoy", "eps_yoy", "roe"]:
                if col in df.columns: df[col] = df[col] / 100.0
        
        num_cols = df.select_dtypes(include=["number"]).columns
        has_extremes = (df[num_cols].abs() > 5.0).any().any()
        if has_extremes and not acknowledge_extremes:
            raise ValueError("數據包含超過 500% (>5.0) 極端值，請確認單位並勾選確認。")
        df.to_csv(data_dir / filename, index=False)