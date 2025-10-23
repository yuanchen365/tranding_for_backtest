# SSOT（Single Source of Truth）— 模組化回測權威規格
> 本檔為專案**唯一權威**：策略介面、撮合與成本口徑、績效/風險指標、報表與輸出命名、設定 Schema、驗收與 PR 檢核，皆以此為準。  
> 規格版本：v1.0 ｜ 最後更新：2025-10-19

---

## 0) 範圍與依據
- 涵蓋：資料、策略、撮合、成本、績效/風險、報表輸出、執行設定、驗收與貢獻規範。
- 依據：四策略一鍵回測（S1/S2/S3/S4）、t→t+1 開盤撮合、滑價、成本一次扣、月勝率/年淨利/權益曲線、Excel/CSV 輸出。
- 語系與時間：繁體中文、Asia/Taipei、日期 `YYYY-MM-DD`。

---

## A. 策略介面規格（StrategySpec）
### A1. 策略識別
- 代碼/名稱/方向：
  - S1「Trend-Long」(long)
  - S2「Trend-Short」(short)
  - S3「MeanRevert-Long」(long)
  - S4「MeanRevert-Short」(short)
- 市場限制：邏輯允許做空；股票做空的實務限制（券源/稅費/漲跌停）需於說明文件註記。

### A2. 輸入資料（最小欄）
- 頻率：日（D）
- 欄位：Open / High / Low / Close / Volume(可缺)
- 非交易日不產生 bar；t+1 **Open 缺值** → 跳過該訊號並記錄，不中止流程。

### A3. 指標與避免前視
- 通道高 RH = `High` 的 N 日滾動最大，**使用前一日值**（shift(1)，min_periods=N）
- 通道低 RL = `Low` 的 L 日滾動最小，**使用前一日值**（shift(1)，min_periods=L）
- 其他指標預設一律使用前一日值，嚴禁前視。

### A4. 訊號定義（在 t）
- S1：`Close(t) > RH(t)` 進多；`Close(t) < RL(t)` 出多  
- S2：`Close(t) < RL(t)` 進空；`Close(t) > RH(t)` 出空  
- S3：`Close(t) < RL(t)` 進多；`Close(t) > RH(t)` 出多  
- S4：`Close(t) > RH(t)` 進空；`Close(t) < RL(t)` 出空  
- 策略輸出欄位：`RH, RL, entry_flag, exit_flag, side_mode`

### A5. 撮合（在 t+1）
- 成交：t+1 開盤價
- 滑價：多進 +slip、 多出 −slip；空進 −slip、 空出 +slip
- 單一持倉：同一策略同時最多 1 筆。

### A6. 交易紀錄（雙列制）
- 進場列：`BUY`(多) / `SELLSHORT`(空)；`Net Profit=0`
- 出場列：`SELL`(多) / `BUYTOCOVER`(空)；`Net Profit=單筆損益（含滑價）`
- 必備欄位：  
  `Symbol, Strategy, Date(成交日), Action, Price, Reason, Side,  
   Net Profit, Cumulative Net Profit,  
   N, L, Signal Date(訊號日), Signal_Rolling_High, Signal_Rolling_Low,  
   entry_close_t, exit_close_t, EntryPrice`

### A7. 邊界
- 同日多重訊號：依持倉狀態處理一次；不允許同日反手。
- 漲跌停/異常開盤：視同缺價處理（跳過並記錄）。

---

## B. 績效與風險口徑（MetricsSpec）
### B1. 成本模型（與績效關聯）
- 套用點：**出場列一次性**（避免重複扣）
- 構成：`MULTIPLIER`、`FEE_FIXED_PER_SIDE`、`FEE_RATE_PER_SIDE`（費用皆×2 邊）
- 名目本金近似：`((EntryPrice + ExitPrice)/2) × MULTIPLIER`
- 所有績效統計以**扣完成本後**的 `Net Profit` 為準。

### B2. 事件版指標（基於出場列）
- 淨利潤 = Σ `Net Profit`  
- 毛利 = Σ `Net Profit > 0`  
- 毛損 = Σ `Net Profit < 0`（表中呈負；計 Profit Factor 時取絕對值）  
- 最大回撤（**金額**）= `min(CumPNL - cummax(CumPNL))` 的**絕對值呈現**  
- 總交易次數 = 出場列筆數（round-trip 數）
- 勝率(%) = `贏家數/總數 × 100`
- 平均每筆 = `淨利/總數`
- 單筆最大獲利/損失 = `max/min(Net Profit)`
- Profit Factor = `毛利 / |毛損|`
- Expectancy = `勝率 × 平均贏家 − 敗率 × |平均輸家|`
- Payoff Ratio = `平均贏家 / |平均輸家|`
- 最長連勝 / 最長連敗：依出場順序計連段長度。

### B3. 逐日權益與風險版
- 權益估值：持倉期間以**每日收盤**計浮盈虧；**出場日**併入**含成本的已實現**累積。
- 年化基準：252 交易日；日報酬 `r_t = equity_t / equity_{t-1} − 1`，`r_f=0`。
- 最大回撤（**百分比**）= `max((峰值−當前)/峰值) × 100`
- Sharpe = `(mean(r)/std(r)) × sqrt(252)`
- Sortino = `(mean(r)/std(r<0)) × sqrt(252)`
- CAGR(%) = `((equity_end / equity_start)^(252/樣本天數) − 1) × 100`
- MAR = `CAGR / 最大回撤(%)`
- 零/空集合：無法計算時以 `NaN` 或 `∞` 呈現，報表需原樣保留。

---

## C. 報表與檔案輸出（ExportSpec）
### C1. 圖表
- **每月勝率**：Y 軸固定 0–100%，柱上標示「當月出場筆數」
- **每年淨利**：年度 Σ `Net Profit`
- **各策略同圖權益曲線**：以**出場列累積淨利**繪製（風險指標仍以逐日權益）
- **全部策略合計權益曲線**：在 UI 與 PNG 輸出中，同圖顯示「各策略」與「全部策略合計」之累積權益；提供 PNG 下載（需 `kaleido`）
- 標題模板：`{SYMBOL} {STRATEGY} {每月勝率|每年淨利}（{DATE_SPAN}；N={N}, L={L}）`

### C2. Excel
- 每策略一分頁：
  1) 上方：**雙列交易表**（凍結首列）  
  2) 中左：**月勝率表**；中右：**年淨利表**  
  3) 下方：**兩張 PNG**（對應上方兩表）
- `SUMMARY` 分頁：四策略**績效總表**（事件+風險指標）＋「**同圖權益曲線**」PNG
- Streamlit UI：每個參數組合需提供 Plotly 累積權益曲線（含全部策略合計）與 PNG 下載鈕
- 嵌圖：**必須在 ExcelWriter 存活期間**完成；圖檔不存在則**跳過**插入。
- 無出場資料：仍建分頁與表格，插圖段落跳過並加註「無出場資料」。

### C3. CSV 與命名
- `全部交易紀錄`：完整欄位（見 A6）
- `策略績效總表`：每策略一列（含事件版 + 風險版指標）
- `參數網格總表`：每個 N/L/成本/滑價組合 × 策略的績效彙整（含資料品質欄位）
- `data_quality`：原始/清理筆數、缺值統計、缺少交易日資訊
- 命名規則：
  - 圖：`{SYMBOL}_{STRATEGY}_{monthly|yearly}_{N}_{L}.png`、`{SYMBOL}_equity_all_{N}_{L}.png`
  - Excel：`{SYMBOL}_交易紀錄_{N}_{L}.xlsx`
  - CSV：`{SYMBOL}_全部交易紀錄_{N}_{L}.csv`、`{SYMBOL}_策略績效總表_{N}_{L}.csv`、`{SYMBOL}_param_grid_summary.csv`、`{SYMBOL}_data_quality.csv`
- 輸出根目錄：由設定 `output.dir` 指定。

---

## D. 執行設定規格（ConfigSpec）
### D1. 檔案與驗證
- 建議檔名：`configs/example.yaml`
- 建議於 CI 做 Schema 驗證（欄位齊備、型別正確）。

### D2. 資料來源
- `data.sources`: 清單（例：`["csv","shioaji"]`），依順序嘗試；亦可使用舊欄位 `data.source` 指定單一來源。
- `data.csv_pattern`: 路徑模板（例：`./data/{symbol}.csv`）
- （可選）`data.symbol_map`: 代碼→Shioaji 合約鍵映射。

### D3. 回測與參數
- `run.symbols`: 樣本清單（如：`["0050"]`）
- `run.start` / `run.end`: 區間（`end: null` 表示至今日）
- `run.nl_grid`: 參數網格（如：`[[10,20]]`）
- （可選）`run.param_grid`: 清單，元素可指定 `n`、`l`、`slippage` 及成本覆寫（`multiplier`、`fee_fixed_per_side`、`fee_rate_per_side`）；此欄位存在時優先於 `run.nl_grid`
- `run.strategies`: `["S1","S2","S3","S4"]`
- `run.slippage`: 例：`0.2`

### D4. 成本與輸出
- `costs.multiplier`、`costs.fee_fixed_per_side`、`costs.fee_rate_per_side`
- `output.dir`: 輸出資料夾
- `output.embed_images`: `true|false`
- `risk.trading_days_per_year`: 252
- `risk.risk_free_rate`: 0

### D5. 例外處理
- t+1 Open 缺值：跳過並記錄，不中止流程。
- 無出場資料：仍產生分頁/CSV；圖表段落跳過。
- 資料品質：每次載入資料時需紀錄原始/清理筆數、缺值統計、缺口交易日並輸出 `data_quality` 報表。

---

## E. 驗收標準（Acceptance Criteria）
1) **四策略回測完整**：雙列交易紀錄；損益僅出現在出場列；無出場資料策略不失敗。  
2) **績效總表**：含事件版（淨利/毛利/毛損/最大回撤額/勝率/平均每筆/單筆極值）＋擴充（PF/Expectancy/Payoff/連勝連敗）＋風險版（最大回撤%、Sharpe、Sortino、CAGR%、MAR）。  
3) **輸出**：每策略月勝率 PNG、年淨利 PNG、同圖權益曲線 PNG；Excel（各策略分頁＋兩圖＋分表；SUMMARY＋權益曲線）；CSV（交易紀錄/績效表）。  
4) **一致性**：所有口徑與欄位名稱與本 SSOT 完全一致；嵌圖在 Writer 存活期間完成；檔案不存在時跳過插圖。

---

## F. PR/審核檢核表（Submit 前逐項勾）
- [ ] 僅以本 SSOT 為準；程式與文件由此對齊。  
- [ ] 年化=252、最大回撤(%)/額定義、成本扣法一致。  
- [ ] 欄位名與中文標籤與 SSOT 完全一致。  
- [ ] 無出場資料策略：仍產生分頁/CSV，圖表跳過。  
- [ ] 圖/Excel/CSV 檔名與路徑符合 C3。  
- [ ] 若有修改 SSOT，PR 需列**影響清單**與**回滾方案**。

---

## G. 變更流程（誰能改與如何改）
1) **先改 SSOT.md**（本檔）；  
2) 開 Issue/PR，標註受影響模組（strategies/engine/costs/metrics/report/cli）；  
3) 通過審核後，同步更新程式與測試；  
4) 重新產出報表，附輸出截圖於 PR。

---
