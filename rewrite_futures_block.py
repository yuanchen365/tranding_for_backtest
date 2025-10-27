from pathlib import Path
from datetime import date
import sys
import io

p = Path('app/streamlit_app.py')
text = p.read_text(encoding='utf-8', errors='replace')
lines = text.splitlines()
start_idx = None
end_idx = None
for i,l in enumerate(lines):
    if l.strip().startswith('with futures_tab:'):
        start_idx = i
        break
if start_idx is None:
    print('no futures_tab block found')
    sys.exit(1)
for j in range(start_idx, len(lines)):
    if 'history_df' in lines[j] and 'futures_downloads' in lines[j]:
        end_idx = min(j+2, len(lines)-1)
        break
if end_idx is None:
    for j in range(start_idx+1, len(lines)):
        if lines[j].startswith('with '):
            end_idx = j-1
            break
    if end_idx is None:
        end_idx = len(lines)-1

new_block = []
new_block.append('with futures_tab:')
new_block.append('    st.subheader("📥 期貨資料下載（Shioaji）")')
new_block.append('    with st.expander("操作說明", expanded=False):')
new_block.append('        st.markdown("""')
new_block.append('        1. 輸入回測代碼（如 `TFX`）。')
new_block.append('        2. 指定 Shioaji 合約鍵（建議完整路徑，例如 `Futures/TXF/TXFR1`）。')
new_block.append('        3. 選擇日期區間後按「下載日 K」。')
new_block.append('        4. 檔案將儲存於 `data/futures/<代碼>/`。')
new_block.append('        """)')
new_block.append('')
new_block.append('    default_symbol = st.session_state.get("last_futures_symbol", "TFX")')
new_block.append('    default_contract = st.session_state.get("last_futures_contract", "TXFR1")')
new_block.append('    col_alias, col_contract = st.columns(2)')
new_block.append('    futures_symbol = col_alias.text_input("回測商品代碼", value=default_symbol).strip().upper()')
new_block.append('    futures_contract = col_contract.text_input("Shioaji 合約鍵", value=default_contract).strip()')
new_block.append('')
new_block.append('    col_dates = st.columns(2)')
new_block.append('    futures_start = col_dates[0].date_input("開始日期", value=st.session_state.get("futures_start", date(2018, 1, 1)))')
new_block.append('    futures_end = col_dates[1].date_input("結束日期", value=st.session_state.get("futures_end", date.today()))')
new_block.append('')
new_block.append('    download_futures = st.button("下載日 K", key="download_futures_button", use_container_width=True)')
new_block.append('')

new_text = '\n'.join(lines[:start_idx] + new_block + lines[end_idx+1:])
p.write_text(new_text, encoding='utf-8')
print('rewritten block', start_idx, end_idx)
