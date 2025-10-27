from pathlib import Path
p=Path('fix_tabs.py')
p.write_text('from pathlib import Path\np=Path(\'app/streamlit_app.py\')\ntext=p.read_text(encoding=\'utf-8\')\nlines=text.splitlines()\nfor i,l in enumerate(lines):\n    if l.strip().startswith(\'dashboard_tab, futures_tab = st.tabs(\'):\n        lines[i] = \"dashboard_tab, futures_tab, trading_tab = st.tabs(['Dashboard', 'Futures', 'Trading'])\"\n        break\np.write_text(\\'\\n\\'.join(lines), encoding=\\'utf-8\\')\nprint(\'ok\')')
