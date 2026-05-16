"""
tabs/tab_buget.py
Tab: Budget Analysis
自动读取 budget_analysis.ipynb，依次展示 Markdown 文字分析与图表输出。
运行前请先在 Jupyter 里执行该 notebook 并保存输出。
"""

import base64
import json
from pathlib import Path

import streamlit as st


def render(analysis_dir: Path) -> None:
    st.header("Budget Analysis")
    st.caption("以下内容自动从 budget_analysis.ipynb 读取，含文字说明与图表。")

    nb_path = analysis_dir / "budget_analysis.ipynb"
    if not nb_path.exists():
        st.error("未找到 budget_analysis.ipynb，请确认文件存在于 analysis/ 目录。")
        return

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    has_output = False

    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type")

        # ── Markdown 文字分析 ──────────────────────────────────────────────
        if cell_type == "markdown":
            source = cell.get("source", [])
            text = "".join(source) if isinstance(source, list) else str(source)
            if text.strip():
                st.markdown(text)

        # ── Code 单元：展示图片 & stdout ───────────────────────────────────
        elif cell_type == "code":
            outputs = cell.get("outputs", [])
            for out in outputs:
                # 图片输出
                png_data = out.get("data", {}).get("image/png")
                if png_data:
                    if isinstance(png_data, list):
                        png_data = "".join(png_data)
                    try:
                        img_bytes = base64.b64decode(png_data)
                        st.image(img_bytes, use_container_width=True)
                        has_output = True
                    except Exception:
                        continue

                # 文本输出（print 语句等）
                if out.get("output_type") == "stream" and out.get("name") == "stdout":
                    text_out = out.get("text", "")
                    if isinstance(text_out, list):
                        text_out = "".join(text_out)
                    if text_out.strip():
                        st.code(text_out.strip(), language="text")
                        has_output = True

    if not has_output:
        st.info(
            "budget_analysis.ipynb 目前没有保存任何图表输出。\n\n"
            "请先在 Jupyter 中运行该 notebook 并保存（Ctrl+S），"
            "再重新刷新此页面即可显示全部图表。"
        )
