from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
FUTURES_DATA_DIR = PROJECT_ROOT / "data" / "futures"
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtester.strategies.base import STRATEGY_REGISTRY, get_strategy
from backtester.config import DataConfig
from backtester.data.loader import DataLoader, DataLoaderError


st.set_page_config(page_title="交易模組 Trading", layout="wide")
st.title("🧭 交易模組（預設全日盤 all）")
st.caption("以日線口徑：t 收盤判斷、t+1 開盤撮合。此頁用下載的期貨資料產生委託草案。")


@st.cache_data(show_spinner=False)
def _read_daily_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    date_col = None
    for c in ("Date", "date", "datetime", "time", "timestamp"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("CSV 缺少日期欄位（Date/date/...）")
    df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要欄位：{col}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", *(["Volume"] if "Volume" in df.columns else [])]]


def _download_futures_data(symbol_alias: str, contract_key: str, start_dt: datetime, end_dt: datetime):
    cfg = DataConfig(sources=["shioaji"], csv_pattern=None, symbol_map={symbol_alias: contract_key})
    loader = DataLoader(cfg)
    return loader.load_daily(symbol_alias, start_dt, end_dt)


def _save_futures_csv(symbol: str, price_df: pd.DataFrame, start_date: date, end_date: date) -> Path:
    target_dir = FUTURES_DATA_DIR / symbol
    target_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{symbol}_{start_date}_{end_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = target_dir / fname
    df = price_df.copy()
    df.index.name = "Date"
    df.to_csv(path, encoding="utf-8-sig")
    return path


with st.sidebar:
    st.header("連線與模式")
    mode = st.radio("交易模式", options=["Paper", "Live"], index=0, horizontal=True)
    session_choice = st.selectbox("交易時段（資料口徑）", options=["all", "regular", "after_hours"], index=0)
    st.caption("預設全日盤 all；此為資料與下單時段概念，非合約路徑。")

    st.header("商品與資料")
    symbol = st.text_input("回測/交易代碼", value=st.session_state.get("last_futures_symbol", "TFX")).strip().upper()
    contract = st.text_input(
        "Shioaji 合約鍵（建議完整路徑）",
        value=st.session_state.get("last_futures_contract", "Futures/TXF/TXFR1"),
    ).strip()

    csv_options = []
    futures_dir = FUTURES_DATA_DIR / (symbol or "")
    if futures_dir.exists():
        files = sorted(futures_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        csv_options = [str(p) for p in files]
    latest = st.session_state.get("latest_download_csv")
    default_idx = csv_options.index(latest) if latest in csv_options else (0 if csv_options else -1)
    selected_csv = st.selectbox("選擇期貨日線 CSV", options=csv_options, index=default_idx)

    st.header("策略與參數")
    strat_code = st.selectbox(
        "策略",
        options=list(STRATEGY_REGISTRY.keys()),
        index=0,
        format_func=lambda c: f"{c} {STRATEGY_REGISTRY[c].name}",
    )
    col1, col2, col3 = st.columns(3)
    n_val = col1.number_input("N", min_value=1, step=1, value=10)
    l_val = col2.number_input("L", min_value=1, step=1, value=20)
    slip_val = col3.number_input("滑價", min_value=0.0, step=0.05, value=0.2)
    qty = st.number_input("下單張數（草案）", min_value=1, step=1, value=1)

st.markdown("### 期貨資料下載（Shioaji）")

col_dl1, col_dl2 = st.columns(2)
start_input = col_dl1.date_input("開始日期", value=st.session_state.get("futures_start", date(2018, 1, 1)))
end_input = col_dl2.date_input("結束日期", value=st.session_state.get("futures_end", date.today()))

do_download = st.button("下載日 K", use_container_width=True)

if do_download:
    import importlib.util, os
    if not symbol:
        st.error("請先在左側輸入回測/交易代碼。")
    elif not contract:
        st.error("請先在左側輸入 Shioaji 合約鍵（建議完整路徑）。")
    elif end_input < start_input:
        st.error("結束日期不可早於開始日期。")
    elif importlib.util.find_spec("shioaji") is None:
        st.error("尚未安裝 shioaji 套件，請先安裝：pip install shioaji")
    elif not os.getenv("SHIOAJI_API_KEY") or not os.getenv("SHIOAJI_SECRET_KEY"):
        st.error("未偵測到 Shioaji 憑證（SHIOAJI_API_KEY/SHIOAJI_SECRET_KEY）。請確認 .env 並重新啟動應用程式。")
    else:
        sdt = datetime.combine(start_input, datetime.min.time())
        edt = datetime.combine(end_input, datetime.min.time())
        try:
            with st.spinner("下載 Shioaji 日線資料中…"):
                bundle = _download_futures_data(symbol, contract, sdt, edt)
        except DataLoaderError as exc:
            st.error(f"下載失敗：{exc}")
        except Exception as exc:  # pragma: no cover
            st.error(f"Shioaji 下載時發生未預期錯誤：{exc}")
        else:
            save_path = _save_futures_csv(symbol, bundle.data, start_input, end_input)
            st.session_state["last_futures_symbol"] = symbol
            st.session_state["last_futures_contract"] = contract
            st.session_state["futures_start"] = start_input
            st.session_state["futures_end"] = end_input

            # 更新最近下載檔路徑，並觸發 rerun 以套用預設選擇
            st.session_state["latest_download_csv"] = str(save_path)
            st.success(f"下載完成，已儲存至 {save_path}")
            st.rerun()

            # 使用目前策略與 N/L 計算最新訊號，顯示 RH / RL（高點/低點）
            try:
                strat = get_strategy(strat_code)
                sig = strat.generate_signals(bundle.data, int(n_val), int(l_val))
                if not sig.empty:
                    ts = sig.index.max()
                    rh = float(sig.loc[ts, "Rolling_High"]) if pd.notna(sig.loc[ts, "Rolling_High"]) else None
                    rl = float(sig.loc[ts, "Rolling_Low"]) if pd.notna(sig.loc[ts, "Rolling_Low"]) else None
                    st.markdown("#### 最新交易訊號參考（依目前策略與 N/L）")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("訊號日", pd.Timestamp(ts).date())
                    c2.metric("高點 RH", f"{rh:.2f}" if rh is not None else "-")
                    c3.metric("低點 RL", f"{rl:.2f}" if rl is not None else "-")
                else:
                    st.info("資料長度不足以計算訊號（請調整 N/L 或日期區間）。")
            except Exception as exc:  # pragma: no cover
                st.warning(f"顯示訊號 RH/RL 時發生例外：{exc}")

st.markdown("### 產生委託草案")
gen = st.button("生成委託草案")

if gen:
    if not selected_csv:
        st.error("找不到可用的期貨 CSV，請先至『期貨資料下載』頁面下載。")
    else:
        try:
            df = _read_daily_csv(Path(selected_csv))
            strat = get_strategy(strat_code)
            sig = strat.generate_signals(df, int(n_val), int(l_val))
            if sig.empty:
                st.info("資料長度不足以計算訊號（請確認 N/L 與資料期間）。")
            else:
                ts = sig.index.max()
                row = sig.loc[ts]
                action = None
                side = row["side_mode"]
                if bool(row.get("entry_flag", False)):
                    action = "BUY" if side == "long" else "SELLSHORT"
                elif bool(row.get("exit_flag", False)):
                    action = "SELL" if side == "long" else "BUYTOCOVER"

                # t 與 t+1（使用商業日近似）
                from pandas.tseries.offsets import BDay
                t_signal = pd.Timestamp(ts).date()
                t1_trade = (pd.Timestamp(ts) + BDay(1)).date()

                est = {
                    "Symbol": symbol,
                    "Contract": contract,
                    "Session": session_choice,
                    "Mode": mode,
                    "Strategy": f"{strat_code} {STRATEGY_REGISTRY[strat_code].name}",
                    "N": int(n_val),
                    "L": int(l_val),
                    "Slippage": float(slip_val),
                    "Qty": int(qty),
                    "Signal Date": t_signal.isoformat(),
                    "Trade Date (t+1)": t1_trade.isoformat(),
                    "Signal_RH": float(row["Rolling_High"]),
                    "Signal_RL": float(row["Rolling_Low"]),
                    "Close(t)": float(df.loc[ts, "Close"]),
                    "Proposed Action": action or "（目前無訊號）",
                    "Price Mode": "Open@t+1 ± slip",
                }
                st.dataframe(pd.DataFrame([est]))
                csv_bytes = pd.DataFrame([est]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="下載委託草案 CSV",
                    data=csv_bytes,
                    file_name=f"intent_{symbol}_{t1_trade}.csv",
                    mime="text/csv",
                )
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"產生草案失敗：{exc}")
