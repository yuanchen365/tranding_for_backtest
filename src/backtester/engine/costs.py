from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    multiplier: float = 1.0
    fee_fixed_per_side: float = 0.0
    fee_rate_per_side: float = 0.0

    def apply(self, entry_price: float, exit_price: float, raw_net_profit: float) -> float:
        """Apply multiplier and transaction costs to a raw round-trip PnL."""
        notional = (entry_price + exit_price) / 2.0
        per_unit_fee = (self.fee_fixed_per_side * 2.0) + (notional * self.fee_rate_per_side * 2.0)
        adjusted = raw_net_profit * self.multiplier - per_unit_fee * self.multiplier
        return adjusted
