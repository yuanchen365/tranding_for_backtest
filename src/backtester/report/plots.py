from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


FONT_NAME = "Noto Sans CJK TC"
FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
FONT_PATH = Path(__file__).resolve().parent / "_fonts" / "NotoSansCJKtc-Regular.otf"


def _ensure_font():
    if FONT_PATH.exists():
        return FONT_PATH

    FONT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        import ssl

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    except Exception:
        return None
    return FONT_PATH if FONT_PATH.exists() else None


def _setup_matplotlib():
    plt.rcParams["axes.unicode_minus"] = False
    font_path = _ensure_font()
    if font_path:
        try:
            import matplotlib.font_manager as fm

            fm.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = FONT_NAME
        except Exception:
            plt.rcParams.setdefault("font.family", FONT_NAME)


def monthly_win_table(exits: pd.DataFrame) -> pd.DataFrame:
    if exits.empty:
        return pd.DataFrame(columns=["Year", "Month", "勝率百分比", "出場筆數", "年月"])
    g = exits.copy()
    g["Date"] = pd.to_datetime(g["Date"])
    g["Year"] = g["Date"].dt.year
    g["Month"] = g["Date"].dt.month
    g["win"] = (g["Net Profit"] > 0).astype(float)
    tbl = (
        g.groupby(["Year", "Month"])
        .agg(勝率百分比=("win", "mean"), 出場筆數=("win", "size"))
        .reset_index()
        .sort_values(["Year", "Month"])
    )
    tbl["勝率百分比"] = (tbl["勝率百分比"] * 100).round(2)
    tbl["年月"] = tbl["Year"].astype(str) + "-" + tbl["Month"].astype(str).str.zfill(2)
    return tbl


def yearly_profit_table(exits: pd.DataFrame) -> pd.DataFrame:
    if exits.empty:
        return pd.DataFrame(columns=["Year", "年淨利"])
    g = exits.copy()
    g["Date"] = pd.to_datetime(g["Date"])
    tbl = (
        g.groupby(g["Date"].dt.year)["Net Profit"]
        .sum()
        .rename("年淨利")
        .reset_index()
        .rename(columns={"Date": "Year"})
        .sort_values("Year")
    )
    return tbl


def plot_strategy_charts(
    symbol: str,
    strategy_label: str,
    exits: pd.DataFrame,
    date_span: str,
    n: int,
    l: int,
    output_dir: Path,
    embed: bool = True,
) -> Tuple[Dict[str, Path], pd.DataFrame, pd.DataFrame]:
    """
    Generate monthly win-rate and yearly net-profit charts for a single strategy.
    Returns (paths, monthly_table, yearly_table).
    """
    _setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    month_tbl = monthly_win_table(exits)
    year_tbl = yearly_profit_table(exits)
    paths: Dict[str, Path] = {}

    if not exits.empty and embed:
        monthly_path = output_dir / f"{symbol}_{strategy_label.replace(' ', '_')}_monthly_{n}_{l}.png"
        plt.figure(figsize=(12, 4))
        plt.bar(month_tbl["年月"], month_tbl["勝率百分比"])
        for xi, yi, cnt in zip(range(len(month_tbl)), month_tbl["勝率百分比"], month_tbl["出場筆數"]):
            plt.text(xi, min(yi + 2, 100), str(int(cnt)), ha="center", va="bottom", fontsize=8)
        plt.ylim(0, 100)
        plt.title(f"{symbol} {strategy_label} 每月勝率（{date_span}；N={n}, L={l}）")
        plt.xlabel("年月")
        plt.ylabel("勝率(%)")
        plt.xticks(rotation=75)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(monthly_path, dpi=150, bbox_inches="tight")
        plt.close()
        paths["monthly"] = monthly_path

        yearly_path = output_dir / f"{symbol}_{strategy_label.replace(' ', '_')}_yearly_{n}_{l}.png"
        plt.figure(figsize=(10, 4))
        plt.bar(year_tbl["Year"].astype(str), year_tbl["年淨利"])
        plt.title(f"{symbol} {strategy_label} 每年淨利（{date_span}；N={n}, L={l}）")
        plt.xlabel("年度")
        plt.ylabel("淨利（元）")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(yearly_path, dpi=150, bbox_inches="tight")
        plt.close()
        paths["yearly"] = yearly_path

    return paths, month_tbl, year_tbl


def plot_equity_all_strategies(
    exits: pd.DataFrame,
    symbol: str,
    date_span: str,
    n: int,
    l: int,
    output_dir: Path,
    embed: bool = True,
) -> Optional[Path]:
    _setup_matplotlib()
    if exits.empty or not embed:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    for strategy_label, g in exits.sort_values("Date").groupby("Strategy"):
        gg = g.copy()
        gg["Cumulative Net Profit"] = gg["Net Profit"].cumsum()
        plt.plot(pd.to_datetime(gg["Date"]), gg["Cumulative Net Profit"], label=strategy_label)
    plt.title(f"{symbol} 各策略權益曲線（{date_span}；N={n}, L={l}）")
    plt.xlabel("日期")
    plt.ylabel("累積淨利（元）")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    eq_path = output_dir / f"{symbol}_equity_all_{n}_{l}.png"
    plt.savefig(eq_path, dpi=150, bbox_inches="tight")
    plt.close()
    return eq_path
