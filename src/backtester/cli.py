from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import BacktestConfig, ParamGridEntry, load_config
from .data.loader import DataLoader
from .engine.backtester import BacktestEngine
from .engine.costs import CostModel
from .metrics.summary import compute_strategy_performance
from .report.excel import export_excel
from .report.exports import export_csvs
from .report.plots import plot_equity_all_strategies, plot_strategy_charts
from .strategies.base import get_strategy
from .utils import load_env_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular backtest runner (SSOT compliant).")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def run_backtest_from_config(config_path: str | Path) -> Dict[str, List[Path]]:
    load_env_file(".env")
    config = load_config(Path(config_path))
    data_loader = DataLoader(config.data)
    base_cost_model = CostModel(
        multiplier=config.costs.multiplier,
        fee_fixed_per_side=config.costs.fee_fixed_per_side,
        fee_rate_per_side=config.costs.fee_rate_per_side,
    )

    output_records: Dict[str, List[Path]] = {}
    start_dt = datetime.combine(config.run.start, datetime.min.time())
    end_dt = datetime.combine(config.run.end, datetime.min.time()) if config.run.end else None
    output_root = config.output.directory
    output_root.mkdir(parents=True, exist_ok=True)

    strategy_labels = [f"{code} {get_strategy(code).name}" for code in config.run.strategies]

    for symbol in config.run.symbols:
        print(f"[Backtest] Loading data for {symbol}...")
        price_bundle = data_loader.load_daily(symbol, start_dt, end_dt)
        price = price_bundle.data
        symbol_outputs: List[Path] = []
        quality_row = {"Symbol": symbol}
        for key, value in price_bundle.diagnostics.items():
            if isinstance(value, list):
                quality_row[key] = ",".join(map(str, value))
            else:
                quality_row[key] = value
        quality_df = pd.DataFrame([quality_row])
        symbol_dir = output_root / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        quality_path = symbol_dir / f"{symbol}_data_quality.csv"
        quality_df.to_csv(quality_path, index=False, encoding="utf_8_sig")
        symbol_outputs.append(quality_path)

        param_entries = config.run.param_grid or [
            ParamGridEntry(n=n, l=l) for n, l in (config.run.nl_grid or [])
        ]
        grid_rows = []

        for entry in param_entries:
            slippage = entry.slippage if entry.slippage is not None else config.run.slippage
            cost_model = CostModel(
                multiplier=entry.multiplier if entry.multiplier is not None else base_cost_model.multiplier,
                fee_fixed_per_side=(
                    entry.fee_fixed_per_side
                    if entry.fee_fixed_per_side is not None
                    else base_cost_model.fee_fixed_per_side
                ),
                fee_rate_per_side=(
                    entry.fee_rate_per_side
                    if entry.fee_rate_per_side is not None
                    else base_cost_model.fee_rate_per_side
                ),
            )
            engine = BacktestEngine(cost_model=cost_model, risk_config=config.risk, slippage=slippage)
            print(
                f"[Backtest] Running {symbol} with N={entry.n}, L={entry.l}, slip={slippage}, "
                f"mult={cost_model.multiplier}"
            )
            result = engine.simulate(
                symbol=symbol,
                price=price,
                n=entry.n,
                l=entry.l,
                strategy_codes=config.run.strategies,
            )

            trade_log = result.trade_log.copy()
            exits = trade_log[trade_log["Action"].isin(["SELL", "BUYTOCOVER"])].copy()

            _, performance_df = compute_strategy_performance(
                trade_log=trade_log,
                price=result.price,
                cost_model=cost_model,
                risk_config=config.risk,
                strategy_labels=strategy_labels,
            )

            date_span = f"{config.run.start.strftime('%Y-%m-%d')} ~ {(config.run.end.strftime('%Y-%m-%d') if config.run.end else price.index.max().strftime('%Y-%m-%d'))}"
            img_dir = symbol_dir / f"{symbol}_imgs_{entry.n}_{entry.l}"

            monthly_tables: Dict[str, pd.DataFrame] = {}
            yearly_tables: Dict[str, pd.DataFrame] = {}
            plot_paths: Dict[str, Dict[str, Path]] = {}

            for label in strategy_labels:
                strat_exits = exits[exits["Strategy"] == label].copy()
                paths, mtbl, ytbl = plot_strategy_charts(
                    symbol=symbol,
                    strategy_label=label,
                    exits=strat_exits,
                    date_span=date_span,
                    n=entry.n,
                    l=entry.l,
                    output_dir=img_dir,
                    embed=config.output.embed_images,
                )
                monthly_tables[label] = mtbl
                yearly_tables[label] = ytbl
                if paths:
                    plot_paths[label] = paths

            equity_path = plot_equity_all_strategies(
                exits=exits,
                symbol=symbol,
                date_span=date_span,
                n=entry.n,
                l=entry.l,
                output_dir=img_dir,
                embed=config.output.embed_images,
            )

            excel_path = symbol_dir / f"{symbol}_交易紀錄_{entry.n}_{entry.l}.xlsx"
            export_excel(
                output_path=excel_path,
                strategies=strategy_labels,
                trade_log=trade_log,
                monthly_tables=monthly_tables,
                yearly_tables=yearly_tables,
                plot_paths=plot_paths,
                equity_path=equity_path,
                performance_df=performance_df,
                n=entry.n,
                l=entry.l,
                embed_images=config.output.embed_images,
            )

            csv_paths = export_csvs(
                output_dir=symbol_dir,
                symbol=symbol,
                n=entry.n,
                l=entry.l,
                trade_log=trade_log,
                performance_df=performance_df,
            )

            plot_files = [path for charts in plot_paths.values() for path in charts.values()]
            artifact_paths = [excel_path, *plot_files, *( [equity_path] if equity_path else [] ), *csv_paths.values()]
            for p in artifact_paths:
                if isinstance(p, Path) and p.exists():
                    symbol_outputs.append(p)

            performance_snapshot = performance_df.copy()
            if not performance_snapshot.empty:
                performance_snapshot = performance_snapshot.reset_index().rename(columns={"策略": "Strategy"})
            else:
                performance_snapshot = pd.DataFrame(columns=["Strategy"])
            performance_snapshot["Symbol"] = symbol
            performance_snapshot["N"] = entry.n
            performance_snapshot["L"] = entry.l
            performance_snapshot["Slippage"] = slippage
            performance_snapshot["Cost_Multiplier"] = cost_model.multiplier
            performance_snapshot["Cost_Fixed_Per_Side"] = cost_model.fee_fixed_per_side
            performance_snapshot["Cost_Rate_Per_Side"] = cost_model.fee_rate_per_side
            performance_snapshot["Price_Bars"] = price_bundle.diagnostics.get("clean_rows", len(price))
            performance_snapshot["Missing_Business_Days"] = price_bundle.diagnostics.get(
                "missing_business_days", 0
            )
            performance_snapshot["Missing_Price_Rows"] = price_bundle.diagnostics.get("missing_price_rows", 0)
            performance_snapshot["Rows_Dropped"] = price_bundle.diagnostics.get("rows_dropped", 0)
            grid_rows.append(performance_snapshot)

            if result.warnings:
                print(f"[Backtest][{symbol}] warnings:")
                for msg in result.warnings:
                    print(f"  - {msg}")

        if grid_rows:
            grid_summary_df = pd.concat(grid_rows, ignore_index=True)
            grid_summary_path = symbol_dir / f"{symbol}_param_grid_summary.csv"
            grid_summary_df.to_csv(grid_summary_path, index=False, encoding="utf_8_sig")
            symbol_outputs.append(grid_summary_path)

        output_records[symbol] = symbol_outputs

    return output_records


def main() -> None:
    args = _parse_args()
    run_backtest_from_config(args.config)


if __name__ == "__main__":
    main()
