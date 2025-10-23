from __future__ import annotations

"""
Configuration parsing utilities adhering to the SSOT (v1.0) specification.

The module provides typed dataclasses that mirror the YAML schema described in
ConfigSpec.  No optional third party validation framework is used so that the
project remains lightweight and easy to run in notebook or CLI environments.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "PyYAML is required to load configuration files. "
        "Install with `pip install pyyaml`."
    ) from exc


def _coerce_date(value: Optional[str]) -> Optional[date]:
    if value in ("", None):
        return None
    ts = pd.to_datetime(value, utc=False, errors="raise")
    return ts.date()


def _require_keys(source: Dict[str, Any], allowed: Iterable[str], section: str) -> None:
    unknown = set(source) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {sorted(unknown)}")


@dataclass(frozen=True)
class DataConfig:
    sources: List[str]
    csv_pattern: Optional[str] = None
    symbol_map: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DataConfig":
        _require_keys(payload, ("source", "sources", "source_priority", "csv_pattern", "symbol_map"), "data")
        raw_source = payload.get("source")
        raw_sources = payload.get("sources") or payload.get("source_priority")

        if raw_sources is None:
            if raw_source is None:
                raise ValueError("Provide data.source or data.sources/data.source_priority")
            sources = [raw_source]
        elif raw_source is not None:
            raise ValueError("Use either data.source or data.sources/source_priority, not both")
        else:
            sources = list(raw_sources if isinstance(raw_sources, list) else [raw_sources])

        normalized = [str(s).lower() for s in sources]
        allowed = {"csv", "shioaji"}
        for src in normalized:
            if src not in allowed:
                raise ValueError("data.sources entries must be 'csv' or 'shioaji'")

        csv_pattern = payload.get("csv_pattern")
        symbol_map = payload.get("symbol_map") or {}
        return cls(sources=normalized, csv_pattern=csv_pattern, symbol_map=symbol_map)


@dataclass(frozen=True)
class RunConfig:
    symbols: List[str]
    start: date
    end: Optional[date]
    nl_grid: List[Tuple[int, int]]
    strategies: List[str]
    slippage: float
    param_grid: List["ParamGridEntry"] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunConfig":
        _require_keys(
            payload,
            ("symbols", "start", "end", "nl_grid", "strategies", "slippage", "param_grid"),
            "run",
        )
        symbols = payload.get("symbols") or payload.get("symbol") or []
        if isinstance(symbols, str):
            symbols = [symbols]
        if not symbols:
            raise ValueError("run.symbols must contain at least one symbol")
        start = _coerce_date(payload.get("start"))
        end = _coerce_date(payload.get("end"))
        if start is None:
            raise ValueError("run.start is required")
        nl_grid_raw = payload.get("nl_grid") or []
        nl_grid: List[Tuple[int, int]] = []
        for pair in nl_grid_raw:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each entry in run.nl_grid must be a pair [N, L]")
            n, l = int(pair[0]), int(pair[1])
            if n <= 0 or l <= 0:
                raise ValueError("N and L must be positive integers")
            nl_grid.append((n, l))
        strategies = payload.get("strategies") or []
        if not strategies:
            raise ValueError("run.strategies must list at least one strategy code")
        slippage = float(payload.get("slippage", 0.0))
        if slippage < 0:
            raise ValueError("run.slippage must be non-negative")
        param_grid_payload = payload.get("param_grid") or []
        param_grid = [ParamGridEntry.from_dict(entry) for entry in param_grid_payload]
        if not nl_grid and not param_grid:
            raise ValueError("Provide run.nl_grid or run.param_grid with at least one entry")
        return cls(
            symbols=list(map(str, symbols)),
            start=start,
            end=end,
            nl_grid=nl_grid,
            strategies=list(map(str, strategies)),
            slippage=slippage,
            param_grid=param_grid,
        )


@dataclass(frozen=True)
class CostConfig:
    multiplier: float = 1.0
    fee_fixed_per_side: float = 0.0
    fee_rate_per_side: float = 0.0

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "CostConfig":
        payload = payload or {}
        _require_keys(payload, ("multiplier", "fee_fixed_per_side", "fee_rate_per_side"), "costs")
        multiplier = float(payload.get("multiplier", 1.0))
        if multiplier <= 0:
            raise ValueError("costs.multiplier must be positive")
        fee_fixed = float(payload.get("fee_fixed_per_side", 0.0))
        fee_rate = float(payload.get("fee_rate_per_side", 0.0))
        if fee_rate < 0 or fee_fixed < 0:
            raise ValueError("cost parameters must be non-negative")
        return cls(multiplier=multiplier, fee_fixed_per_side=fee_fixed, fee_rate_per_side=fee_rate)


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    embed_images: bool = True

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "OutputConfig":
        payload = payload or {}
        _require_keys(payload, ("dir", "embed_images"), "output")
        directory = Path(payload.get("dir") or ".")
        embed_images = bool(payload.get("embed_images", True))
        return cls(directory=directory, embed_images=embed_images)


@dataclass(frozen=True)
class RiskConfig:
    trading_days_per_year: int = 252
    risk_free_rate: float = 0.0

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "RiskConfig":
        payload = payload or {}
        _require_keys(payload, ("trading_days_per_year", "risk_free_rate"), "risk")
        tdy = int(payload.get("trading_days_per_year", 252))
        if tdy <= 0:
            raise ValueError("risk.trading_days_per_year must be positive")
        rfr = float(payload.get("risk_free_rate", 0.0))
        return cls(trading_days_per_year=tdy, risk_free_rate=rfr)


@dataclass(frozen=True)
class BacktestConfig:
    data: DataConfig
    run: RunConfig
    costs: CostConfig
    output: OutputConfig
    risk: RiskConfig

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BacktestConfig":
        return cls(
            data=DataConfig.from_dict(payload.get("data", {})),
            run=RunConfig.from_dict(payload.get("run", {})),
            costs=CostConfig.from_dict(payload.get("costs")),
            output=OutputConfig.from_dict(payload.get("output")),
            risk=RiskConfig.from_dict(payload.get("risk")),
        )


def load_config(path: Path) -> BacktestConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping/object")
    return BacktestConfig.from_dict(payload)


@dataclass(frozen=True)
class ParamGridEntry:
    n: int
    l: int
    slippage: Optional[float] = None
    multiplier: Optional[float] = None
    fee_fixed_per_side: Optional[float] = None
    fee_rate_per_side: Optional[float] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ParamGridEntry":
        if not isinstance(payload, dict):
            raise ValueError("Each param_grid entry must be a mapping")
        required = {"n", "l"}
        if not required.issubset(payload):
            raise ValueError("param_grid entry must include 'n' and 'l'")
        n = int(payload["n"])
        l = int(payload["l"])
        if n <= 0 or l <= 0:
            raise ValueError("param_grid n and l must be positive integers")
        slippage = payload.get("slippage")
        slippage_val = float(slippage) if slippage is not None else None
        multiplier = payload.get("multiplier")
        fee_fixed = payload.get("fee_fixed_per_side")
        fee_rate = payload.get("fee_rate_per_side")
        multiplier_val = float(multiplier) if multiplier is not None else None
        fee_fixed_val = float(fee_fixed) if fee_fixed is not None else None
        fee_rate_val = float(fee_rate) if fee_rate is not None else None
        return cls(
            n=n,
            l=l,
            slippage=slippage_val,
            multiplier=multiplier_val,
            fee_fixed_per_side=fee_fixed_val,
            fee_rate_per_side=fee_rate_val,
        )
