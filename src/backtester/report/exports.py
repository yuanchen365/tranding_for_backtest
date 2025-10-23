from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_csvs(
    output_dir: Path,
    symbol: str,
    n: int,
    l: int,
    trade_log: pd.DataFrame,
    performance_df: pd.DataFrame,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    trade_path = output_dir / f"{symbol}_全部交易紀錄_{n}_{l}.csv"
    perf_path = output_dir / f"{symbol}_策略績效總表_{n}_{l}.csv"
    trade_log.to_csv(trade_path, index=False, encoding="utf_8_sig")
    performance_df.to_csv(perf_path, encoding="utf_8_sig")
    return {"trades": trade_path, "performance": perf_path}
