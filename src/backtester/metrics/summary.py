from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import RiskConfig
from ..engine.costs import CostModel


@dataclass(frozen=True)
class StrategyPerformance:
    strategy: str
    metrics: Dict[str, float]
    daily_equity: pd.Series
    daily_returns: pd.Series


def compute_strategy_performance(
    trade_log: pd.DataFrame,
    price: pd.DataFrame,
    cost_model: CostModel,
    risk_config: RiskConfig,
    strategy_labels: List[str] | None = None,
) -> Tuple[List[StrategyPerformance], pd.DataFrame]:
    perfs: List[StrategyPerformance] = []
    summary_rows: List[Dict[str, object]] = []

    exits = trade_log[trade_log["Action"].isin(["SELL", "BUYTOCOVER"])].copy()
    if not exits.empty:
        exits = exits.sort_values(["Strategy", "Date"], kind="mergesort")
        exits = exits.reset_index(drop=True)

    if strategy_labels is None:
        labels = list(trade_log["Strategy"].unique())
    else:
        labels = list(strategy_labels)

    for strategy in labels:
        strat_log = trade_log[trade_log["Strategy"] == strategy].copy()
        strat_log = strat_log.sort_values(["Date", "Action"]).reset_index(drop=True)
        strat_exits = exits[exits["Strategy"] == strategy].copy()
        event_metrics = _event_metrics(strat_exits)
        daily_equity, daily_returns = _daily_equity(strat_log, price, cost_model.multiplier)
        risk_metrics = _risk_metrics(daily_equity, daily_returns, risk_config)
        combined = {**event_metrics, **risk_metrics, "策略": strategy}
        summary_rows.append(combined)
        perfs.append(StrategyPerformance(strategy, combined, daily_equity, daily_returns))

    summary_df = pd.DataFrame(summary_rows).set_index("策略") if summary_rows else pd.DataFrame()
    return perfs, summary_df


def _event_metrics(exits: pd.DataFrame) -> Dict[str, float]:
    if exits.empty:
        return {
            "淨利潤": 0.0,
            "毛利": 0.0,
            "毛損": 0.0,
            "最大回撤(金額)": 0.0,
            "最大回撤(%)": np.nan,
            "總交易次數": 0,
            "贏家交易次數": 0,
            "輸家交易次數": 0,
            "勝率(%)": 0.0,
            "平均每筆淨利潤": 0.0,
            "單筆最大獲利": 0.0,
            "單筆最大損失": 0.0,
            "Profit Factor": np.inf,
            "Expectancy": 0.0,
            "Payoff Ratio": np.inf,
            "最長連勝": 0,
            "最長連敗": 0,
            "CAGR(%)": np.nan,
            "MAR": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
        }

    exits = exits.copy()
    exits["Cumulative Net Profit"] = exits["Net Profit"].cumsum()
    net_profit = exits["Net Profit"].sum()
    gross_profit = exits.loc[exits["Net Profit"] > 0, "Net Profit"].sum()
    gross_loss = exits.loc[exits["Net Profit"] < 0, "Net Profit"].sum()
    dd = exits["Cumulative Net Profit"] - exits["Cumulative Net Profit"].cummax()
    max_dd_amt = -dd.min() if len(dd) else 0.0

    total = len(exits)
    wins_mask = exits["Net Profit"] > 0
    losses_mask = exits["Net Profit"] < 0
    wins = int(wins_mask.sum())
    losses = int(losses_mask.sum())
    winrate = (wins / total * 100.0) if total else 0.0
    avg_per_trade = (net_profit / total) if total else 0.0
    max_win = exits["Net Profit"].max() if total else 0.0
    max_loss = exits["Net Profit"].min() if total else 0.0

    gl_abs = abs(gross_loss)
    profit_factor = gross_profit / gl_abs if gl_abs > 0 else np.inf
    avg_win = exits.loc[wins_mask, "Net Profit"].mean() if wins else 0.0
    avg_loss = -exits.loc[losses_mask, "Net Profit"].mean() if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else np.inf
    p_win = wins / total if total else 0.0
    expectancy = p_win * avg_win - (1 - p_win) * avg_loss

    streak_series = wins_mask.astype(int).reset_index(drop=True)
    if len(streak_series) == 0:
        longest_win = longest_loss = 0
    else:
        group_ids = (streak_series != streak_series.shift()).cumsum()
        streak_sizes = streak_series.groupby(group_ids).transform("size")
        longest_win = int(streak_sizes[streak_series == 1].max() if (streak_series == 1).any() else 0)
        longest_loss = int(streak_sizes[streak_series == 0].max() if (streak_series == 0).any() else 0)

    return {
        "淨利潤": round(float(net_profit), 4),
        "毛利": round(float(gross_profit), 4),
        "毛損": round(float(gross_loss), 4),
        "最大回撤(金額)": round(float(max_dd_amt), 4),
        "最大回撤(%)": np.nan,
        "總交易次數": int(total),
        "贏家交易次數": wins,
        "輸家交易次數": losses,
        "勝率(%)": round(float(winrate), 2),
        "平均每筆淨利潤": round(float(avg_per_trade), 4),
        "單筆最大獲利": round(float(max_win), 4),
        "單筆最大損失": round(float(max_loss), 4),
        "Profit Factor": round(float(profit_factor), 4) if np.isfinite(profit_factor) else np.inf,
        "Expectancy": round(float(expectancy), 4),
        "Payoff Ratio": round(float(payoff), 4) if np.isfinite(payoff) else np.inf,
        "最長連勝": longest_win,
        "最長連敗": longest_loss,
        "CAGR(%)": np.nan,
        "MAR": np.nan,
        "Sharpe": np.nan,
        "Sortino": np.nan,
    }


def _daily_equity(trade_log: pd.DataFrame, price: pd.DataFrame, multiplier: float) -> Tuple[pd.Series, pd.Series]:
    if trade_log.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    closes = price["Close"].copy()
    closes.index = pd.to_datetime(closes.index)
    idx = pd.date_range(start=closes.index.min(), end=closes.index.max(), freq="D")
    equity = pd.Series(0.0, index=idx)
    realized = 0.0

    rows = trade_log.sort_values(["Date", "Action"]).to_dict("records")
    i = 0
    while i < len(rows):
        row = rows[i]
        action = row["Action"]
        if action not in ("BUY", "SELLSHORT"):
            i += 1
            continue

        entry = row
        if i + 1 >= len(rows):
            break
        exit_row = rows[i + 1]
        if exit_row["Action"] not in ("SELL", "BUYTOCOVER"):
            i += 1
            continue

        entry_date = pd.to_datetime(entry["Date"])
        exit_date = pd.to_datetime(exit_row["Date"])
        side = entry["Side"]
        entry_price = float(entry["Price"])
        exit_net = float(exit_row["Net Profit"])

        hold_index = pd.date_range(start=entry_date, end=exit_date, freq="D")
        hold_prices = closes.reindex(hold_index).ffill()
        if hold_prices.isna().any():
            hold_prices = hold_prices.bfill().ffill()

        if side == "long":
            pnl = (hold_prices - entry_price) * multiplier
        else:
            pnl = (entry_price - hold_prices) * multiplier

        base = realized
        if not pd.Index(hold_index).isin(equity.index).all():
            equity = equity.reindex(equity.index.union(hold_index)).sort_index()
            equity = equity.ffill().fillna(realized)

        equity.loc[hold_index] = base + pnl.values
        realized += exit_net
        equity.loc[exit_date] = realized

        i += 2

    equity = equity.ffill().fillna(realized)
    daily_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return equity, daily_returns


def _risk_metrics(
    equity: pd.Series,
    daily_returns: pd.Series,
    risk_config: RiskConfig,
) -> Dict[str, float]:
    if equity.empty or len(equity) < 2:
        return {
            "最大回撤(%)": np.nan,
            "CAGR(%)": np.nan,
            "MAR": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
        }

    peak = equity.cummax().replace(0, np.nan)
    dd_pct_series = (peak - equity) / peak
    max_dd_pct = float(dd_pct_series.max() * 100.0) if dd_pct_series.notna().any() else np.nan

    days = len(equity)
    start_value = equity.iloc[0]
    end_value = equity.iloc[-1]
    cagr_val = np.nan
    if start_value != 0:
        years = days / risk_config.trading_days_per_year
        cagr_val = (end_value / start_value) ** (1 / years) - 1 if years > 0 else np.nan
    cagr_pct = float(cagr_val * 100.0) if pd.notna(cagr_val) else np.nan

    if daily_returns.empty:
        sharpe = sortino = np.nan
    else:
        excess = daily_returns - risk_config.risk_free_rate / risk_config.trading_days_per_year
        mu = float(excess.mean())
        sigma = float(excess.std(ddof=1))
        downside = float(excess[excess < 0].std(ddof=1)) if (excess < 0).any() else np.nan
        sharpe = (mu / sigma * np.sqrt(risk_config.trading_days_per_year)) if sigma > 0 else np.nan
        sortino = (
            (mu / downside * np.sqrt(risk_config.trading_days_per_year))
            if (not np.isnan(downside) and downside > 0)
            else np.nan
        )

    if pd.notna(cagr_val) and pd.notna(max_dd_pct) and max_dd_pct != 0:
        mar = float(cagr_val / (max_dd_pct / 100.0))
    else:
        mar = np.nan

    return {
        "最大回撤(%)": round(float(max_dd_pct), 2) if pd.notna(max_dd_pct) else np.nan,
        "CAGR(%)": round(float(cagr_pct), 2) if pd.notna(cagr_pct) else np.nan,
        "MAR": round(float(mar), 4) if pd.notna(mar) else np.nan,
        "Sharpe": round(float(sharpe), 4) if pd.notna(sharpe) else np.nan,
        "Sortino": round(float(sortino), 4) if pd.notna(sortino) else np.nan,
    }
