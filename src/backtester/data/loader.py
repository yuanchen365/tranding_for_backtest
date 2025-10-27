from __future__ import annotations

"""
Data acquisition utilities.

Two sources are supported:
1) CSV files containing daily bars.
2) Shioaji API (optional dependency) with automatic batching to daily bars.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import DataConfig

TAIWAN_TZ = "Asia/Taipei"


class DataLoaderError(Exception):
    """Raised when data cannot be retrieved."""


@dataclass(frozen=True)
class PriceData:
    data: pd.DataFrame
    diagnostics: Dict[str, object]


@dataclass
class CsvDataSource:
    pattern: str

    def fetch(self, symbol: str, start: datetime, end: Optional[datetime]) -> pd.DataFrame:
        path = Path(self.pattern.format(symbol=symbol))
        if not path.exists():
            raise DataLoaderError(f"CSV file not found for {symbol}: {path}")
        df = pd.read_csv(path)
        date_col = None
        for candidate in ("Date", "date", "datetime", "time", "timestamp"):
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            raise DataLoaderError(f"{path} must contain a date column (Date/date/...)")
        df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col).sort_index()
        cols_required = ["Open", "High", "Low", "Close"]
        missing = [c for c in cols_required if c not in df.columns]
        if missing:
            raise DataLoaderError(f"{path} missing columns: {missing}")
        df = df[["Open", "High", "Low", "Close", *(["Volume"] if "Volume" in df.columns else [])]]
        localized_index = df.index.tz_localize(
            TAIWAN_TZ, nonexistent="shift_forward", ambiguous="NaT"
        )
        df.index = localized_index.tz_localize(None)
        mask = (df.index.date >= start.date()) & (True if end is None else df.index.date <= end.date())
        df = df.loc[mask]
        return df


class ShioajiDataSource:
    def __init__(self, symbol_map: Dict[str, str]):
        try:
            import shioaji as sj
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "Shioaji data source requested but shioaji package is not installed."
            ) from exc

        import os

        api_key = os.getenv("SHIOAJI_API_KEY")
        secret_key = os.getenv("SHIOAJI_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError("Shioaji credentials not provided via environment variables.")

        self._sj = sj
        self.api = sj.Shioaji()
        self.api.login(api_key=api_key, secret_key=secret_key)
        self.symbol_map = symbol_map or {}

    def _contract_from_path(self, path: str):
        parts = [segment for segment in re.split(r"[/.]", path) if segment]
        if not parts:
            raise DataLoaderError(f"Invalid Shioaji contract path: {path}")
        node = self.api.Contracts
        for segment in parts:
            attr_value = None
            if hasattr(node, segment):
                attr_value = getattr(node, segment)
            if attr_value not in (None, node):
                node = attr_value
                continue
            try:
                node = node[segment]
            except Exception as exc:
                raise DataLoaderError(
                    f"Unable to traverse Shioaji contract path '{path}' at segment '{segment}'"
                ) from exc
        return node

    def _resolve_contract(self, symbol: str):
        key = self.symbol_map.get(symbol, symbol)
        if not key:
            raise DataLoaderError(f"Symbol mapping for {symbol} is empty.")

        # 1) allow direct contract path syntax (e.g. Futures/TXF/TXFR1)
        if isinstance(key, str) and ("/" in key or "." in key):
            return self._contract_from_path(key)

        # 2) default to stock dictionary lookup
        stocks = getattr(self.api.Contracts, "Stocks", None)
        if stocks is not None and isinstance(key, str):
            try:
                return stocks[key]
            except KeyError:
                pass

        # 3) heuristic for futures/options symbols (e.g. TXFR1 -> Futures/TXF/TXFR1)
        if isinstance(key, str) and key.isupper() and len(key) >= 4:
            guessed_product = key[:3]
            try:
                return self._contract_from_path(f"Futures/{guessed_product}/{key}")
            except DataLoaderError:
                pass

        raise DataLoaderError(
            "Unknown Shioaji contract key for "
            f"symbol {symbol}: {key}. 請參考 Shioaji 合約文件，使用如 Futures/TXF/TXFR1 的完整路徑。"
        )

    def fetch(self, symbol: str, start: datetime, end: Optional[datetime]) -> pd.DataFrame:
        contract = self._resolve_contract(symbol)
        frames = []
        cur = start
        end_dt = end or datetime.now()
        while cur <= end_dt:
            batch_end = min(cur + timedelta(days=90), end_dt)
            try:
                kb = self.api.kbars(contract, start=cur.strftime("%Y-%m-%d"), end=batch_end.strftime("%Y-%m-%d"))
                df = pd.DataFrame(
                    {
                        "ts": kb.ts,
                        "Open": kb.Open,
                        "High": kb.High,
                        "Low": kb.Low,
                        "Close": kb.Close,
                        "Volume": kb.Volume,
                    }
                )
            except Exception as exc:  # pragma: no cover - remote API path
                raise DataLoaderError(f"Failed to fetch {symbol} kbars: {exc}") from exc

            if len(df) == 0:
                cur = batch_end + timedelta(days=1)
                continue

            df["Time"] = pd.to_datetime(df["ts"], unit="ns", utc=True).dt.tz_convert(TAIWAN_TZ)
            df = df.set_index("Time").drop(columns=["ts"]).sort_index()
            frames.append(df)
            cur = batch_end + timedelta(days=1)

        if not frames:
            raise DataLoaderError(f"No data returned from Shioaji for {symbol}")
        raw = pd.concat(frames).sort_index()
        daily = raw.resample("1D").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        ).dropna()
        daily.index = daily.index.tz_localize(None)
        mask = (daily.index.date >= start.date()) & (True if end is None else daily.index.date <= end.date())
        return daily.loc[mask]


class DataLoader:
    """Facade that tries configured data sources in order until one returns data."""

    def __init__(self, config: DataConfig):
        self.config = config
        self._csv_source: Optional[CsvDataSource] = None
        self._shioaji_source: Optional[ShioajiDataSource] = None

    def _get_csv_source(self) -> CsvDataSource:
        if self._csv_source is None:
            if not self.config.csv_pattern:
                raise DataLoaderError("csv source requested but data.csv_pattern is not configured")
            self._csv_source = CsvDataSource(self.config.csv_pattern)
        return self._csv_source

    def _get_shioaji_source(self) -> ShioajiDataSource:
        if self._shioaji_source is None:
            try:
                self._shioaji_source = ShioajiDataSource(self.config.symbol_map)
            except RuntimeError as exc:
                raise DataLoaderError(str(exc)) from exc
        return self._shioaji_source

    def load_daily(self, symbol: str, start: datetime, end: Optional[datetime]) -> PriceData:
        errors = []
        for source_name in self.config.sources:
            try:
                if source_name == "csv":
                    df = self._get_csv_source().fetch(symbol, start, end)
                elif source_name == "shioaji":
                    df = self._get_shioaji_source().fetch(symbol, start, end)
                else:  # pragma: no cover - config validation ensures this
                    raise DataLoaderError(f"Unsupported source: {source_name}")
                raw = df.copy()
                if df.empty:
                    raise DataLoaderError(f"{source_name} source returned empty dataset for {symbol}")
                df.index.name = "Date"
                df = df[["Open", "High", "Low", "Close", *(["Volume"] if "Volume" in df.columns else [])]]
                df = df.astype(
                    {
                        "Open": float,
                        "High": float,
                        "Low": float,
                        "Close": float,
                        **({"Volume": float} if "Volume" in df.columns else {}),
                    }
                )
                missing_mask = df[["Open", "High", "Low", "Close"]].isna().any(axis=1)
                df = df.dropna(subset=["Open", "High", "Low", "Close"])
                df.index = pd.to_datetime(df.index).tz_localize(None)
                diagnostics = self._diagnostics(raw, df, missing_mask, start, end)
                return PriceData(data=df, diagnostics=diagnostics)
            except DataLoaderError as exc:
                errors.append(f"{source_name}: {exc}")
        raise DataLoaderError("; ".join(errors))

    @staticmethod
    def _diagnostics(
        raw: pd.DataFrame,
        cleaned: pd.DataFrame,
        missing_mask: pd.Series,
        start: datetime,
        end: Optional[datetime],
    ) -> Dict[str, object]:
        diag: Dict[str, object] = {}
        diag["requested_start"] = start.date().isoformat()
        diag["requested_end"] = end.date().isoformat() if end else None
        if not raw.empty:
            diag["raw_start"] = pd.to_datetime(raw.index).min().date().isoformat()
            diag["raw_end"] = pd.to_datetime(raw.index).max().date().isoformat()
        else:
            diag["raw_start"] = None
            diag["raw_end"] = None
        diag["raw_rows"] = int(len(raw))
        diag["clean_rows"] = int(len(cleaned))
        diag["rows_dropped"] = int(len(raw) - len(cleaned))
        for col in ["Open", "High", "Low", "Close"]:
            diag[f"missing_{col.lower()}"] = int(raw[col].isna().sum()) if col in raw.columns else 0
        diag["missing_price_rows"] = int(missing_mask.sum())
        if not raw.empty:
            observed_index = pd.to_datetime(raw.index).drop_duplicates().sort_values()
            expected_index = pd.date_range(observed_index.min(), observed_index.max(), freq="B")
            missing_dates = expected_index.difference(observed_index)
            diag["expected_business_days"] = int(len(expected_index))
            diag["missing_business_days"] = int(len(missing_dates))
            diag["missing_business_dates_sample"] = ",".join(
                d.date().isoformat() for d in missing_dates[:10]
            )
            diag["missing_business_dates_more"] = bool(len(missing_dates) > 10)
        else:
            diag["expected_business_days"] = 0
            diag["missing_business_days"] = 0
            diag["missing_business_dates_sample"] = ""
            diag["missing_business_dates_more"] = False
        return diag
