from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import RiskConfig
from ..strategies.base import Strategy, get_strategy
from .costs import CostModel


@dataclass
class ParameterRunResult:
    symbol: str
    n: int
    l: int
    price: pd.DataFrame
    trade_log: pd.DataFrame
    warnings: List[str] = field(default_factory=list)


class BacktestEngine:
    def __init__(self, cost_model: CostModel, risk_config: RiskConfig, slippage: float):
        self.cost_model = cost_model
        self.risk_config = risk_config
        self.slippage = slippage

    def simulate(
        self,
        symbol: str,
        price: pd.DataFrame,
        n: int,
        l: int,
        strategy_codes: Iterable[str],
    ) -> ParameterRunResult:
        records: List[Dict[str, object]] = []
        warnings: List[str] = []
        price = price.sort_index()
        for code in strategy_codes:
            strategy = get_strategy(code)
            label = f"{strategy.code} {strategy.name}"
            strat_records, strat_warnings = self._simulate_strategy(symbol, price, n, l, strategy, label)
            records.extend(strat_records)
            if strat_warnings:
                warnings.extend([f"{label}: {msg}" for msg in strat_warnings])

        if records:
            trade_log = pd.DataFrame(records).sort_values(["Strategy", "Date", "Action", "Signal Date"]).reset_index(drop=True)
            trade_log["Cumulative Net Profit"] = 0.0
            for strat_name in trade_log["Strategy"].unique():
                mask = trade_log["Strategy"] == strat_name
                trade_log.loc[mask, "Cumulative Net Profit"] = trade_log.loc[mask, "Net Profit"].cumsum()
        else:
            trade_log = pd.DataFrame(columns=[
                "Symbol",
                "Strategy",
                "Date",
                "Action",
                "Price",
                "Reason",
                "Side",
                "Net Profit",
                "Cumulative Net Profit",
                "N",
                "L",
                "Signal Date",
                "Signal_Rolling_High",
                "Signal_Rolling_Low",
                "entry_close_t",
                "exit_close_t",
                "EntryPrice",
            ])

        return ParameterRunResult(symbol=symbol, n=n, l=l, price=price, trade_log=trade_log, warnings=warnings)

    def _simulate_strategy(
        self,
        symbol: str,
        price: pd.DataFrame,
        n: int,
        l: int,
        strategy: Strategy,
        label: str,
    ) -> Tuple[List[Dict[str, object]], List[str]]:
        signals = strategy.generate_signals(price, n, l)
        records: List[Dict[str, object]] = []
        warnings: List[str] = []
        if signals.empty:
            return records, warnings

        position = 0
        entry_info: Optional[Dict[str, object]] = None

        price_index = price.index
        price_lookup = price

        for ts in signals.index[:-1]:  # need next day to trade
            try:
                idx_pos = price_index.get_loc(ts)
            except KeyError:
                warnings.append(f"Signal date missing in price data: {ts}")
                continue

            if idx_pos + 1 >= len(price_index):
                break  # no t+1

            trade_day = price_index[idx_pos + 1]
            today_row = price_lookup.iloc[idx_pos]
            next_row = price_lookup.iloc[idx_pos + 1]
            next_open = float(next_row["Open"])

            if not np.isfinite(next_open):
                warnings.append(f"{trade_day.date()} 開盤價缺失，跳過撮合")
                continue

            rh = float(signals.loc[ts, "Rolling_High"])
            rl = float(signals.loc[ts, "Rolling_Low"])
            entry_flag = bool(signals.loc[ts, "entry_flag"])
            exit_flag = bool(signals.loc[ts, "exit_flag"])
            side_mode = signals.loc[ts, "side_mode"]

            if position == 0 and entry_flag:
                entry_price = next_open + self.slippage if side_mode == "long" else next_open - self.slippage
                entry_info = {
                    "entry_trade_date": trade_day,
                    "entry_price": entry_price,
                    "entry_reason": signals.loc[ts, "entry_reason"],
                    "entry_signal_date": ts,
                    "entry_rh": rh,
                    "entry_rl": rl,
                    "entry_close_t": float(today_row["Close"]),
                    "side": side_mode,
                }
                position = 1 if side_mode == "long" else -1
                records.append(
                    {
                        "Symbol": symbol,
                        "Strategy": label,
                        "Date": trade_day.date(),
                        "Action": "BUY" if side_mode == "long" else "SELLSHORT",
                        "Price": round(entry_price, 6),
                        "Reason": signals.loc[ts, "entry_reason"],
                        "Side": side_mode,
                        "Net Profit": 0.0,
                        "Cumulative Net Profit": 0.0,
                        "N": n,
                        "L": l,
                        "Signal Date": ts.date(),
                        "Signal_Rolling_High": round(rh, 6),
                        "Signal_Rolling_Low": round(rl, 6),
                        "entry_close_t": round(float(today_row["Close"]), 6),
                        "exit_close_t": np.nan,
                        "EntryPrice": np.nan,
                    }
                )
                continue

            if position != 0 and exit_flag:
                if entry_info is None:
                    warnings.append(f"{ts.date()} exit triggered but entry info missing; skipping")
                    continue
                exit_price = next_open - self.slippage if side_mode == "long" else next_open + self.slippage
                entry_price = float(entry_info["entry_price"])
                raw_net = exit_price - entry_price if entry_info["side"] == "long" else entry_price - exit_price
                net_profit = self.cost_model.apply(entry_price, exit_price, raw_net)
                records.append(
                    {
                        "Symbol": symbol,
                        "Strategy": label,
                        "Date": trade_day.date(),
                        "Action": "SELL" if entry_info["side"] == "long" else "BUYTOCOVER",
                        "Price": round(exit_price, 6),
                        "Reason": signals.loc[ts, "exit_reason"],
                        "Side": entry_info["side"],
                        "Net Profit": round(float(net_profit), 6),
                        "Cumulative Net Profit": 0.0,
                        "N": n,
                        "L": l,
                        "Signal Date": ts.date(),
                        "Signal_Rolling_High": round(rh, 6),
                        "Signal_Rolling_Low": round(rl, 6),
                        "entry_close_t": round(float(entry_info["entry_close_t"]), 6),
                        "exit_close_t": round(float(today_row["Close"]), 6),
                        "EntryPrice": round(entry_price, 6),
                    }
                )
                position = 0
                entry_info = None

        return records, warnings
