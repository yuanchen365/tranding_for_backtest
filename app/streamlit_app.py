from __future__ import annotations

import io
import sys
import tempfile
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
FUTURES_DATA_DIR = PROJECT_ROOT / "data" / "futures"

from backtester.config import CostConfig, DataConfig, ParamGridEntry, RiskConfig
from backtester.data.loader import DataLoader, DataLoaderError
from backtester.engine.backtester import BacktestEngine
from backtester.engine.costs import CostModel
from backtester.metrics.summary import StrategyPerformance, compute_strategy_performance
from backtester.report.exports import export_csvs
from backtester.report.excel import export_excel
from backtester.report.plots import plot_equity_all_strategies, plot_strategy_charts
from backtester.strategies.base import STRATEGY_REGISTRY, get_strategy
from backtester.utils import load_env_file


st.set_page_config(page_title="Channel Backtest Dashboard", layout="wide")

# Load environment (such as SHIOAJI credentials) on app startup.
load_env_file(".env")
st.title("📈 四策略通道回測儀表板")
st.markdown(
    """
    上傳或串接資料，設定 N/L 參數網格與成本模型，並立即取得策略績效與互動圖表。<br/>
    本工具遵循專案 SSOT 規格，可匯出 Excel、CSV 與圖檔。
    """,
    unsafe_allow_html=True,
)

# ?�設?�數網格 Session State
if "param_grid_df" not in st.session_state:
    st.session_state["param_grid_df"] = pd.DataFrame(
        [
            {"n": 10, "l": 20, "slippage": 0.2},
            {"n": 20, "l": 40, "slippage": 0.2},
        ]
    )

if "futures_downloads" not in st.session_state:
    st.session_state["futures_downloads"] = []


def _render_kline_chart(price_df: pd.DataFrame | None, trade_log: pd.DataFrame) -> None:
    if price_df is None or price_df.empty:
        st.info("K 線資料尚未載入。")
        return

    price = price_df.copy()
    if not isinstance(price.index, pd.DatetimeIndex):
        price.index = pd.to_datetime(price.index)

    required_cols = {"Open", "High", "Low", "Close"}
    if not required_cols.issubset(price.columns):
        missing = ", ".join(sorted(required_cols - set(price.columns)))
        st.warning(f"價格資料缺少 K 線必要欄位：{missing}")
        return

    fig = go.Figure(
        data=[
            go.Scatter(
                x=price.index,
                y=price["Close"],
                mode="lines",
                name="收盤價",
                line=dict(color="#1f77b4", width=2),
            )
        ]
    )

    channel_points = trade_log.dropna(subset=["Signal_Rolling_High", "Signal_Rolling_Low"]).copy()
    if not channel_points.empty:
        channel_points["Signal Date"] = pd.to_datetime(channel_points["Signal Date"])
        palette = px.colors.qualitative.Plotly

        for idx, (strategy, grp) in enumerate(channel_points.groupby("Strategy")):
            color = palette[idx % len(palette)]
            grp = grp.sort_values("Signal Date")
            fig.add_trace(
                go.Scatter(
                    x=grp["Signal Date"],
                    y=grp["Signal_Rolling_High"],
                    mode="lines",
                    line=dict(color=color, dash="dash"),
                    name=f"{strategy} 高點通道",
                    legendgroup=strategy,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=grp["Signal Date"],
                    y=grp["Signal_Rolling_Low"],
                    mode="lines",
                    line=dict(color=color, dash="dot"),
                    name=f"{strategy} 低點通道",
                    legendgroup=strategy,
                )
            )

    fig.update_layout(
        height=520,
        hovermode="x unified",
        xaxis_title="日期",
        yaxis_title="價格",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(label="All", step="all"),
                ]
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
        yaxis=dict(fixedrange=False),
)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


def _build_data_config(source_mode: str, symbol: str, uploaded_file: io.BytesIO | None) -> DataConfig:
    temp_dir = Path(tempfile.gettempdir()) / "backtester_ui_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    csv_pattern = None
    if source_mode == "CSV 上傳":
        if uploaded_file is None:
            raise ValueError("請先上傳 CSV 檔案。")
        target_path = temp_dir / f"{symbol}.csv"
        with target_path.open("wb") as fh:
            fh.write(uploaded_file.getvalue())
        sources.append("csv")
        csv_pattern = str(target_path.parent / "{symbol}.csv")
    elif source_mode == "Shioaji API":
        sources.append("shioaji")
    elif source_mode == "CSV + Shioaji Fallback":
        if uploaded_file is None:
            raise ValueError("請提供 CSV 檔案作為第一來源。")
        target_path = temp_dir / f"{symbol}.csv"
        with target_path.open("wb") as fh:
            fh.write(uploaded_file.getvalue())
        sources.extend(["csv", "shioaji"])
        csv_pattern = str(target_path.parent / "{symbol}.csv")
    else:
        raise ValueError(f"未知資料來源：{source_mode}")

    return DataConfig(sources=sources, csv_pattern=csv_pattern)


def _run_backtest(
    symbol: str,
    price_bundle,
    param_entries: List[ParamGridEntry],
    strategies: List[str],
    base_cost: CostConfig,
    base_slippage: float,
    risk_cfg: RiskConfig,
    output_dir: Path,
):
    results = []
    strategy_labels = [f"{code} {get_strategy(code).name}" for code in strategies]

    for entry in param_entries:
        slippage = entry.slippage if entry.slippage is not None else base_slippage
        cost_model = CostModel(
            multiplier=entry.multiplier if entry.multiplier is not None else base_cost.multiplier,
            fee_fixed_per_side=(
                entry.fee_fixed_per_side
                if entry.fee_fixed_per_side is not None
                else base_cost.fee_fixed_per_side
            ),
            fee_rate_per_side=(
                entry.fee_rate_per_side
                if entry.fee_rate_per_side is not None
                else base_cost.fee_rate_per_side
            ),
        )
        engine = BacktestEngine(
            cost_model=cost_model,
            risk_config=risk_cfg,
            slippage=slippage,
        )
        engine_result = engine.simulate(
            symbol=symbol,
            price=price_bundle.data,
            n=entry.n,
            l=entry.l,
            strategy_codes=strategies,
        )

        perfs, perf_df = compute_strategy_performance(
            trade_log=engine_result.trade_log,
            price=engine_result.price,
            cost_model=cost_model,
            risk_config=risk_cfg,
            strategy_labels=strategy_labels,
        )

        results.append(
            {
                "entry": entry,
                "cost_model": cost_model,
                "slippage": slippage,
                "trade_log": engine_result.trade_log,
                "price": engine_result.price,
                "perfs": perfs,
                "performance_df": perf_df,
                "warnings": engine_result.warnings,
                "img_dir": output_dir / f"{symbol}_imgs_{entry.n}_{entry.l}",
            }
        )
    return results


def _download_futures_data(
    symbol_alias: str,
    contract_key: str,
    start_dt: datetime,
    end_dt: datetime,
):
    data_cfg = DataConfig(sources=["shioaji"], csv_pattern=None, symbol_map={symbol_alias: contract_key})
    loader = DataLoader(data_cfg)
    return loader.load_daily(symbol_alias, start_dt, end_dt)


def _save_futures_csv(symbol: str, price_df: pd.DataFrame, start_date: date, end_date: date) -> Path:
    futures_dir = FUTURES_DATA_DIR / symbol
    futures_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{start_date}_{end_date}_{timestamp}.csv"
    export_df = price_df.copy()
    export_df.index.name = "Date"
    path = futures_dir / filename
    export_df.to_csv(path, encoding="utf-8-sig")
    return path


def _persist_outputs(symbol: str, quality_df: pd.DataFrame, summary_df: pd.DataFrame, results: List[dict]) -> None:
    """Mirror CLI 輸出的資料夾結構，方便使用者在 ./outputs 下找到結果。"""
    output_root = PROJECT_ROOT / "outputs"
    symbol_dir = output_root / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    quality_path = symbol_dir / f"{symbol}_data_quality.csv"
    quality_df.to_csv(quality_path, index=False, encoding="utf-8-sig")

    if not summary_df.empty:
        summary_path = symbol_dir / f"{symbol}_param_grid_summary_ui.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for res in results:
        trade_log = res["trade_log"]
        perf_df = res["performance_df"]
        export_csvs(
            output_dir=symbol_dir,
            symbol=symbol,
            n=res["entry"].n,
            l=res["entry"].l,
            trade_log=trade_log,
            performance_df=perf_df,
        )


with st.sidebar:
    st.header("⚙️ 回測設定")
    source_option = st.selectbox(
        "資料來源",
        ["CSV 上傳", "Shioaji API", "CSV + Shioaji Fallback"],
    )
    uploaded_csv = None
    if source_option != "Shioaji API":
        uploaded_csv = st.file_uploader("上傳日 K CSV (含 Date/Open/High/Low/Close)", type="csv")

    symbol = st.text_input("商品代碼", value="0050")
    col_dates = st.columns(2)
    start_date = col_dates[0].date_input("起始日", value=date(2018, 1, 1))
    end_date = col_dates[1].date_input("結束日", value=date(2025, 1, 1))

    strategies_selected = st.multiselect(
        "策略選擇",
        options=list(STRATEGY_REGISTRY.keys()),
        default=["S1", "S2", "S3", "S4"],
        format_func=lambda code: f"{code} {STRATEGY_REGISTRY[code].name}",
    )

    st.markdown("#### 預設成本模型")
    default_slippage = st.number_input("預設滑價", value=0.2, step=0.05)
    cost_multiplier = st.number_input("成本 Multiplier", value=1.0, step=0.1, min_value=0.1)
    cost_fixed = st.number_input("固定費/邊", value=0.0, step=0.1, min_value=0.0)
    cost_rate = st.number_input("比例費/邊", value=0.0, step=0.0001, min_value=0.0, format="%.4f")

    st.markdown("#### 參數網格 (可新增列)")
    current_grid = st.session_state["param_grid_df"].copy()
    if "slippage" in current_grid.columns:
        current_grid["slippage"] = current_grid["slippage"].fillna(default_slippage)
    else:
        current_grid["slippage"] = default_slippage
    grid_df = st.data_editor(
        current_grid,
        num_rows="dynamic",
        key="param_grid_editor",
        hide_index=True,
        column_config={
            "n": st.column_config.NumberColumn("N", min_value=1, step=1),
            "l": st.column_config.NumberColumn("L", min_value=1, step=1),
            "slippage": st.column_config.NumberColumn("滑價", step=0.05),
        },
    )
    st.session_state["param_grid_df"] = grid_df

    st.markdown("#### 風險設定")
    trading_days = st.number_input("年化交易日數", value=252, min_value=1)
    risk_free_rate = st.number_input("年化無風險利率", value=0.0, step=0.01)

    run_button = st.button("🚀 執行回測", use_container_width=True)

if run_button:
    try:
        data_config = _build_data_config(source_option, symbol, uploaded_csv)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    loader = DataLoader(data_config)
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())

    try:
        price_bundle = loader.load_daily(symbol, start_dt, end_dt)
    except DataLoaderError as exc:
        st.error(f"載入資料失敗：{exc}")
        st.stop()

    quality_df = pd.DataFrame([price_bundle.diagnostics])
    st.session_state["backtest_quality"] = quality_df
    st.session_state["backtest_symbol"] = symbol

    base_cost_config = CostConfig(
        multiplier=cost_multiplier,
        fee_fixed_per_side=cost_fixed,
        fee_rate_per_side=cost_rate,
    )
    risk_cfg = RiskConfig(trading_days_per_year=int(trading_days), risk_free_rate=risk_free_rate)

    param_entries: List[ParamGridEntry] = []
    for _, row in grid_df.fillna("").iterrows():
        if not str(row.get("n", "")).strip() or not str(row.get("l", "")).strip():
            continue
        try:
            entry = ParamGridEntry(
                n=int(row["n"]),
                l=int(row["l"]),
                slippage=float(row["slippage"]) if row.get("slippage", "") != "" else None,
                multiplier=float(row["multiplier"]) if row.get("multiplier", "") != "" else None,
                fee_fixed_per_side=float(row["fee_fixed_per_side"])
                if row.get("fee_fixed_per_side", "") != ""
                else None,
                fee_rate_per_side=float(row["fee_rate_per_side"])
                if row.get("fee_rate_per_side", "") != ""
                else None,
            )
            param_entries.append(entry)
        except (ValueError, TypeError):
            st.warning(f"忽略無效列：{row.to_dict()}")

    if not param_entries:
        st.error("請至少設定一組有效的 N、L 參數。")
        st.stop()

    output_dir = Path(tempfile.gettempdir()) / "backtester_ui_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _run_backtest(
        symbol=symbol,
        price_bundle=price_bundle,
        param_entries=param_entries,
        strategies=strategies_selected,
        base_cost=base_cost_config,
        base_slippage=default_slippage,
        risk_cfg=risk_cfg,
        output_dir=output_dir,
    )

    for res in results:
        perf_df = res["performance_df"].reset_index()
        perf_df["N"] = res["entry"].n
        perf_df["L"] = res["entry"].l
        perf_df["Slippage"] = res["slippage"]
        perf_df["Cost_Multiplier"] = res["cost_model"].multiplier
        perf_df["Cost_Fixed_Per_Side"] = res["cost_model"].fee_fixed_per_side
        perf_df["Cost_Rate_Per_Side"] = res["cost_model"].fee_rate_per_side
        perf_df["Missing_Business_Days"] = price_bundle.diagnostics.get("missing_business_days", 0)
        perf_df["Missing_Price_Rows"] = price_bundle.diagnostics.get("missing_price_rows", 0)
        res["performance_table"] = perf_df

    summary_df = (
        pd.concat([res["performance_table"] for res in results], ignore_index=True)
        if results
        else pd.DataFrame()
    )

    st.session_state["backtest_results"] = {
        "results": results,
        "summary": summary_df,
        "symbol": symbol,
        "quality": quality_df,
    }
    _persist_outputs(symbol=symbol, quality_df=quality_df, summary_df=summary_df, results=results)
    st.success("回測完成，可於下方切換不同參數組合檢視結果。")


dashboard_tab, futures_tab = st.tabs(["📈 回測儀表板", "📥 期貨資料下載"])

with dashboard_tab:
    payload = st.session_state.get("backtest_results")
    if payload:
        results = payload["results"]
        summary_df = payload["summary"]
        symbol = payload["symbol"]
        quality_df = payload["quality"]

        st.subheader("📊 資料品質摘要")
        st.dataframe(quality_df.T.rename(columns={0: "值"}))

        st.subheader("🧮 參數網格績效總表")
        if results:
            tabs = st.tabs([f"N={r['entry'].n}, L={r['entry'].l}" for r in results])

            for tab, res in zip(tabs, results):
                perf_df = res["performance_table"]
                with tab:
                    st.dataframe(perf_df, use_container_width=True)

                    trade_log = res["trade_log"]
                    exit_actions = trade_log[trade_log["Action"].isin(["SELL", "BUYTOCOVER"])].copy()
                    if not exit_actions.empty:
                        exit_actions["Date"] = pd.to_datetime(exit_actions["Date"])
                    price_df = res.get("price")

                    overview_tab, kline_tab = st.tabs(["績�??�表", "K 線�?"])

                    with overview_tab:
                        equity_series = []
                        if not exit_actions.empty:
                            for strategy, group in exit_actions.groupby("Strategy"):
                                strat_series = (
                                    group.sort_values("Date")
                                    .groupby("Date")["Cumulative Net Profit"]
                                    .last()
                                    .rename(strategy)
                                    .astype(float)
                                )
                                if not strat_series.empty:
                                    equity_series.append((strategy, strat_series))

                        if equity_series:
                            st.markdown("###### 全部策略累積權益（含合計）")
                            chart_df = (
                                pd.concat([series.rename(label) for label, series in equity_series], axis=1)
                                .sort_index()
                                .ffill()
                                .fillna(0.0)
                            )
                            chart_df.index.name = "Date"

                            combined_df = chart_df.assign(**{"全部策略合計": chart_df.sum(axis=1, min_count=1)})
                            total_long = (
                                combined_df.reset_index()
                                .melt(id_vars="Date", var_name="策略", value_name="累積權益")
                                .dropna()
                            )
                            if total_long.empty:
                                st.info("全部策略尚無累積曲線資料。")
                            else:
                                total_fig = px.line(total_long, x="Date", y="累積權益", color="策略")
                                total_fig.update_layout(height=400, legend_title_text="策略", hovermode="x unified")
                                st.plotly_chart(total_fig, use_container_width=True)

                            available_labels = list(chart_df.columns)
                            selected_labels = st.multiselect(
                                "選擇策略顯示（可多選）",
                                available_labels,
                                default=available_labels,
                                key=f"equity_select_{res['entry'].n}_{res['entry'].l}",
                            )
                            if selected_labels:
                                selected_exits = exit_actions[exit_actions["Strategy"].isin(selected_labels)].copy()
                                if selected_exits.empty:
                                    st.info("所選策略尚無月度損益資料。")
                                else:
                                    selected_exits["YearMonth"] = (
                                        selected_exits["Date"].dt.to_period("M").dt.to_timestamp()
                                    )
                                    monthly_bars = (
                                        selected_exits.groupby(["YearMonth", "Strategy"])["Net Profit"]
                                        .sum()
                                        .reset_index()
                                    )
                                    if monthly_bars.empty:
                                        st.info("所選策略尚無月度損益資料。")
                                    else:
                                        bar_fig = px.bar(
                                            monthly_bars,
                                            x="YearMonth",
                                            y="Net Profit",
                                            color="Strategy",
                                            barmode="group",
                                            labels={"YearMonth": "月份", "Net Profit": "月度淨利", "Strategy": "策略"},
                                        )
                                        bar_fig.update_layout(height=320, hovermode="x unified")
                                        st.plotly_chart(bar_fig, use_container_width=True)
                            else:
                                st.info("請至少選擇一個策略以顯示月度損益。")
                        else:
                            st.info("累積曲線尚未計算。")

                    with kline_tab:
                        _render_kline_chart(price_df, trade_log)

                st.markdown("###### 交易檔案下載")
                trade_csv = res["trade_log"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="下載交易紀錄 CSV",
                    data=trade_csv,
                    file_name=f"{symbol}_trade_log_N{res['entry'].n}_L{res['entry'].l}.csv",
                    mime="text/csv",
                    key=f"download_trades_{res['entry'].n}_{res['entry'].l}",
                )

                if res["warnings"]:
                    st.warning("⚠️ " + "；".join(res["warnings"]))
        else:
            st.info("尚未有回測結果，請於左側設定後執行。")

        if not summary_df.empty:
            st.markdown("###### 參數網格摘要 (合併)")
            st.dataframe(summary_df, use_container_width=True)
            csv_bytes = summary_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="下載參數網格 CSV",
                data=csv_bytes,
                file_name=f"{symbol}_param_grid_summary_ui.csv",
                mime="text/csv",
                key="download_param_grid_summary",
            )
    else:
        st.info("尚未有回測結果，請於左側設定後執行。")

with futures_tab:
    st.subheader("📥 期貨資料下載（Shioaji）")
    with st.expander("操作說明", expanded=False):
        st.markdown(
            "1. 輸入回測代碼（如 `TFX`）。\n"
            "2. 指定 Shioaji 合約鍵（建議完整路徑，例如 `Futures/TXF/TXFR1`）。\n"
            "3. 選擇日期區間後按『下載日 K』。\n"
            "4. 檔案將儲存於 `data/futures/<代碼>/`。"
        )

    default_symbol = st.session_state.get("last_futures_symbol", "TFX")
    default_contract = st.session_state.get("last_futures_contract", "TXFR1")
    col_alias, col_contract = st.columns(2)
    futures_symbol = col_alias.text_input("回測商品代碼", value=default_symbol).strip().upper()
    futures_contract = col_contract.text_input("Shioaji 合約鍵", value=default_contract).strip()

    col_dates = st.columns(2)
    futures_start = col_dates[0].date_input("開始日期", value=st.session_state.get("futures_start", date(2018, 1, 1)))
    futures_end = col_dates[1].date_input("結束日期", value=st.session_state.get("futures_end", date.today()))

    download_futures = st.button("下載日 K", key="download_futures_button", use_container_width=True)

    if download_futures:
        if not futures_symbol:
            st.error("請輸入回測商品代碼。")
        elif not futures_contract:
            st.error("請輸入 Shioaji 合約鍵。")
        elif futures_end < futures_start:
            st.error("結束日期不可早於開始日期。")
        else:
            import importlib.util, os
            if importlib.util.find_spec("shioaji") is None:
                st.error("尚未安裝 shioaji 套件，請先安裝：pip install shioaji")
            elif not os.getenv("SHIOAJI_API_KEY") or not os.getenv("SHIOAJI_SECRET_KEY"):
                st.error("未偵測到 Shioaji 憑證（SHIOAJI_API_KEY/SHIOAJI_SECRET_KEY）。請確認 .env 並重新啟動應用程式。")
            else:
                start_dt = datetime.combine(futures_start, datetime.min.time())
                end_dt = datetime.combine(futures_end, datetime.min.time())
                try:
                    with st.spinner("下載 Shioaji 日線資料中…"):
                        price_bundle = _download_futures_data(
                            symbol_alias=futures_symbol,
                            contract_key=futures_contract,
                            start_dt=start_dt,
                            end_dt=end_dt,
                        )
                except DataLoaderError as exc:
                    st.error(f"下載失敗：{exc}")
                except Exception as exc:  # pragma: no cover - defensive
                    st.error(f"Shioaji 下載時發生未預期錯誤：{exc}")
                else:
                    save_path = _save_futures_csv(futures_symbol, price_bundle.data, futures_start, futures_end)
                    st.session_state["last_futures_symbol"] = futures_symbol
                    st.session_state["last_futures_contract"] = futures_contract
                    st.session_state["futures_start"] = futures_start
                    st.session_state["futures_end"] = futures_end

                    download_record = {
                        "商品": futures_symbol,
                        "合約鍵": futures_contract,
                        "開始": futures_start.isoformat(),
                        "結束": futures_end.isoformat(),
                        "筆數": int(price_bundle.data.shape[0]),
                        "檔案": str((FUTURES_DATA_DIR / futures_symbol / save_path.name).relative_to(PROJECT_ROOT)),
                    }
                    st.session_state["futures_downloads"].insert(0, download_record)
                    st.success(f"下載完成，已儲存至 {save_path}")

                    st.markdown("###### 最近 50 筆資料預覽")
                    st.dataframe(price_bundle.data.tail(min(len(price_bundle.data), 50)))

                    csv_bytes = price_bundle.data.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button(
                        label="立即下載 CSV",
                        data=csv_bytes,
                        file_name=save_path.name,
                        mime="text/csv",
                        key=f"download_shioaji_csv_{save_path.name}",
                    )

                    diag_df = pd.DataFrame([price_bundle.diagnostics]).T.rename(columns={0: "值"})
                    st.markdown("###### 資料品質")
                    st.dataframe(diag_df)

    if st.session_state["futures_downloads"]:
        st.markdown("###### 最近下載紀錄")
        history_df = pd.DataFrame(st.session_state["futures_downloads"])
        st.dataframe(history_df)
