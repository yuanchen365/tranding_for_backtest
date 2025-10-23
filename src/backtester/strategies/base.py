from __future__ import annotations

"""
Strategy definitions compliant with StrategySpec.
"""

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class Strategy:
    code: str
    name: str
    side: str  # "long" or "short"
    entry_reason: str
    exit_reason: str

    def generate_signals(self, df: pd.DataFrame, n: int, l: int) -> pd.DataFrame:
        """Return a DataFrame with rolling bands and entry/exit flags."""
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "Rolling_High",
                    "Rolling_Low",
                    "entry_flag",
                    "exit_flag",
                    "side_mode",
                    "entry_reason",
                    "exit_reason",
                ]
            )
        rolling_high = df["High"].rolling(window=n, min_periods=n).max().shift(1)
        rolling_low = df["Low"].rolling(window=l, min_periods=l).min().shift(1)
        close_t = df["Close"]

        if self.code == "S1":  # Trend-Long
            entry_flag = close_t > rolling_high
            exit_flag = close_t < rolling_low
        elif self.code == "S2":  # Trend-Short
            entry_flag = close_t < rolling_low
            exit_flag = close_t > rolling_high
        elif self.code == "S3":  # MeanRevert-Long
            entry_flag = close_t < rolling_low
            exit_flag = close_t > rolling_high
        elif self.code == "S4":  # MeanRevert-Short
            entry_flag = close_t > rolling_high
            exit_flag = close_t < rolling_low
        else:  # pragma: no cover - guarded by registry
            raise ValueError(f"Unknown strategy code: {self.code}")

        signals = pd.DataFrame(
            {
                "Rolling_High": rolling_high,
                "Rolling_Low": rolling_low,
                "entry_flag": entry_flag.fillna(False),
                "exit_flag": exit_flag.fillna(False),
                "side_mode": self.side,
                "entry_reason": self.entry_reason,
                "exit_reason": self.exit_reason,
            },
            index=df.index,
        )
        signals = signals.dropna(subset=["Rolling_High", "Rolling_Low"])
        return signals


STRATEGY_REGISTRY: Dict[str, Strategy] = {
    "S1": Strategy("S1", "Trend-Long", "long", "收盤 > N日高", "收盤 < L日低"),
    "S2": Strategy("S2", "Trend-Short", "short", "收盤 < L日低", "收盤 > N日高"),
    "S3": Strategy("S3", "MeanRevert-Long", "long", "收盤 < L日低", "收盤 > N日高"),
    "S4": Strategy("S4", "MeanRevert-Short", "short", "收盤 > N日高", "收盤 < L日低"),
}


def get_strategy(code: str) -> Strategy:
    try:
        return STRATEGY_REGISTRY[code]
    except KeyError as exc:
        raise KeyError(f"Unsupported strategy code: {code}") from exc
