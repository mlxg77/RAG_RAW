"""RAG Law 的 Streamlit 演示页面。
uv run streamlit run src/rag_law/ui/app.py
"""

import logging
import tempfile
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from rag_law.knowledge_base import (
    add_document,
    rebuild_knowledge_base,
)
from rag_law.schemas import AnswerPipelineResult, ValidatedCitation
from rag_law.ui.presentation import (
    EXAMPLE_QUESTIONS,
    citation_location,
    from_session_payload,
    load_law_options,
    result_metrics,
    status_presentation,
    to_session_payload,
)


ASSET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.yaml"


def _configure_console_logging() -> None:
    """确保 RAG 流水线的 INFO 日志显示在 Streamlit 控制台。"""

    rag_logger = logging.getLogger("rag_law")
    rag_logger.setLevel(logging.INFO)

    if any(
        getattr(handler, "_rag_law_console", False)
        for handler in rag_logger.handlers
    ):
        return

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s"
        )
    )
    handler._rag_law_console = True  # type: ignore[attr-defined]
    rag_logger.addHandler(handler)
    rag_logger.propagate = False


_configure_console_logging()


st.set_page_config(
    page_title="法条问答 · RAG Law",
    page_icon="§",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _apply_styles() -> None:
    css = (ASSET_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    from rag_law.generation.answer_pipeline import AnswerPipeline

    return AnswerPipeline.load()


@st.cache_data(show_spinner=False)
def _load_laws():
    return load_law_options(MANIFEST_PATH)


def _render_add_law_form() -> None:
    """渲染「上传新法律并重建」面板。"""

    uploaded = st.file_uploader(
        "上传法律文件",
        type=["pdf", "docx", "md"],
        help="支持 PDF、DOCX 和 Markdown；文件名建议使用法律正式名称。",
    )
    law_id = st.text_input(
        "law_id",
        placeholder="cn_company_law",
        help="稳定英文标识：cn_ 前缀加小写字母数字下划线。",
    )
    law_name = st.text_input(
        "法律名称（可选）",
        placeholder="中华人民共和国公司法",
        help="留空时使用文件名（不含扩展名）。",
    )

    if st.button(
        "上传并重建索引",
        type="primary",
        use_container_width=True,
    ):
        _add_uploaded_law(uploaded, law_id, law_name)


def _render_rebuild_form() -> None:
    """渲染已有语料的全量重建入口。"""

    st.caption(
        "根据 manifest.yaml 和 data/raw 重新生成结构化语料、"
        "BM25 与 Chroma。该操作会调用 Embedding 服务。"
    )
    if st.button(
        "重新构建现有知识库",
        use_container_width=True,
    ):
        _rebuild_existing_knowledge_base()


def _rebuild_existing_knowledge_base() -> None:
    """从服务器已有原始文件重建全部知识库产物。"""

    _load_pipeline.clear()
    _load_laws.clear()

    with st.status(
        "正在重新构建现有知识库…",
        expanded=True,
    ) as progress:

        def show_progress(stage: str, message: str) -> None:
            progress.update(
                label=message,
                state="complete" if stage == "complete" else "running",
            )
            progress.write(message)

        try:
            result = rebuild_knowledge_base(
                progress_callback=show_progress,
            )
        except Exception as error:
            progress.update(
                label="重建失败",
                state="error",
                expanded=True,
            )
            st.error(
                "重建失败：请检查 manifest、原始法律文件、模型配置和目录权限。"
                f"\n\n技术信息：{type(error).__name__}: {error}"
            )
            return

    st.session_state.rebuild_notice = (
        "知识库重建完成："
        f"{result.law_count} 部法律、{result.chunk_total} 个 chunk。"
    )
    st.rerun()


def _add_uploaded_law(
    uploaded: Any,
    law_id: str,
    law_name: str,
) -> None:
    """把上传的文件交给 add_document 完成登记与全链重建。"""

    if uploaded is None:
        st.warning("请先选择要上传的法律文件。")
        return

    cleaned_law_id = law_id.strip()

    if not cleaned_law_id:
        st.warning("请填写 law_id，例如 cn_company_law。")
        return

    # 先释放旧流水线，避免重建索引时仍持有旧向量库连接。
    _load_pipeline.clear()
    _load_laws.clear()

    with st.status(
        "正在添加法律并重建索引…",
        expanded=True,
    ) as progress:

        def show_progress(
            stage: str,
            message: str,
        ) -> None:
            progress.update(
                label=message,
                state=(
                    "complete"
                    if stage == "complete"
                    else "running"
                ),
            )
            progress.write(message)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                uploaded_path = (
                    Path(temp_dir)
                    / Path(uploaded.name).name
                )
                uploaded_path.write_bytes(
                    uploaded.getvalue()
                )

                result = add_document(
                    uploaded_path,
                    law_id=cleaned_law_id,
                    law_name=law_name.strip() or None,
                    progress_callback=show_progress,
                )
        except Exception as error:
            progress.update(
                label="添加失败",
                state="error",
                expanded=True,
            )
            st.error(
                "添加失败：请检查文件内容与索引环境。"
                f"\n\n技术信息：{type(error).__name__}: {error}"
            )
            return

    st.session_state.add_law_notice = (
        f"已成功添加《{result.law_name}》："
        f"{result.article_count} 条法条，"
        f"知识库共 {result.law_count} 部法律、"
        f"{result.chunk_total} 个 chunk。"
    )
    st.rerun()


def _render_sidebar(law_names: list[str]) -> str | None:
    with st.sidebar:
        st.markdown('<div class="brand-mark">RAG Law</div>', unsafe_allow_html=True)

        notice = st.session_state.pop(
            "add_law_notice",
            None,
        )
        if notice:
            st.success(notice)

        st.markdown("### 检索范围")
        selected = st.selectbox(
            "指定法律",
            ["全部法律", *law_names],
            help="指定后只在这一部法律中检索。",
            label_visibility="collapsed",
        )

        st.markdown("### 当前知识库")
        st.caption(f"已收录 {len(law_names)} 部法律。原始文本以本地知识库为准。")

        with st.expander("查看法律清单"):
            for index, law_name in enumerate(law_names, start=1):
                st.caption(f"{index:02d}  {law_name}")

        with st.expander("添加新法律"):
            _render_add_law_form()

        with st.expander("维护现有知识库"):
            _render_rebuild_form()

        rebuild_notice = st.session_state.pop(
            "rebuild_notice",
            None,
        )
        if rebuild_notice:
            st.success(rebuild_notice)
            st.info(
                "若 API 运行在独立进程或容器中，请重启 API 以加载新索引。"
            )

        st.markdown("---")
        st.caption("回答仅供法律信息检索与参考，不构成法律意见。")
        if st.button("清空当前对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return None if selected == "全部法律" else selected


def _render_hero(law_count: int) -> None:
    st.markdown(
        f"""
        <section class="hero">
          <div class="brand-mark">Evidence-first legal search</div>
          <h1>把生活问题，落到具体法条。</h1>
          <p>描述你的情况，系统会从本地法律知识库检索依据、生成说明，并只展示通过校验的引用。</p>
          <div class="scope-strip">
            <span class="scope-chip">{law_count} 部法律</span>
            <span class="scope-chip">引用可追溯</span>
            <span class="scope-chip">证据不足时拒答</span>
            <span class="scope-chip">不保存完整检索正文</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_citation(citation: ValidatedCitation, index: int) -> None:
    title = f"{index:02d} · {citation.law_name} · {citation.article_no}"
    with st.expander(title, expanded=index == 1):
        st.markdown(
            f'<div class="citation-meta">证据 {citation.evidence_id} · '
            f'{escape(citation_location(citation))}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="citation-quote">{escape(citation.quote)}</div>',
            unsafe_allow_html=True,
        )


def _render_result(result: AnswerPipelineResult) -> None:
    answer = result.answer
    status = status_presentation(answer.status)

    st.markdown(
        f"""
        <div class="status-row">
          <span class="status-pill {status.tone}">{status.label}</span>
          <span class="status-description">{status.description}</span>
        </div>
        <div class="answer-summary">{escape(answer.summary)}</div>
        """,
        unsafe_allow_html=True,
    )

    metrics = result_metrics(result)
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items(), strict=True):
        column.metric(label, value)

    if answer.follow_up_question:
        st.markdown(
            f'<div class="follow-up"><strong>还需要确认一件事</strong>'
            f'{escape(answer.follow_up_question)}</div>',
            unsafe_allow_html=True,
        )

    if answer.analysis:
        st.markdown('<div class="section-kicker">分析说明</div>', unsafe_allow_html=True)
        st.write(answer.analysis)

    if answer.conditions:
        st.markdown('<div class="section-kicker">适用条件</div>', unsafe_allow_html=True)
        for condition in answer.conditions:
            st.markdown(f"- {condition}")

    if answer.citations:
        st.markdown('<div class="section-kicker">法条依据</div>', unsafe_allow_html=True)
        for index, citation in enumerate(answer.citations, start=1):
            _render_citation(citation, index)

    if answer.limitations:
        with st.expander("边界与限制"):
            st.write(answer.limitations)

    st.markdown(
        '<div class="disclaimer">引用经过程序校验，但法律适用仍取决于完整事实、有效版本和司法实践。'
        '如涉及重大权益，请咨询执业律师。</div>',
        unsafe_allow_html=True,
    )


def _render_history(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
                if message.get("law_name"):
                    st.caption(f"检索范围：{message['law_name']}")
            elif message.get("result"):
                _render_result(from_session_payload(message["result"]))
            else:
                st.error(message["content"])


def _answer(query: str, law_name: str | None) -> None:
    st.session_state.messages.append(
        {"role": "user", "content": query, "law_name": law_name}
    )

    with st.chat_message("user"):
        st.write(query)
        if law_name:
            st.caption(f"检索范围：{law_name}")

    with st.chat_message("assistant"):
        try:
            with st.status(
                "正在准备法律检索…",
                expanded=True,
            ) as progress:
                shown_stages: set[str] = set()

                def show_progress(
                    stage: str,
                    message: str,
                ) -> None:
                    state = (
                        "complete"
                        if stage == "complete"
                        else "error"
                        if stage == "error"
                        else "running"
                    )
                    progress.update(
                        label=message,
                        state=state,
                        expanded=stage not in {"complete", "error"},
                    )

                    if stage not in shown_stages:
                        progress.write(message)
                        shown_stages.add(stage)

                result = _load_pipeline().run(
                    query,
                    law_name=law_name,
                    progress_callback=show_progress,
                )
            _render_result(result)
            st.session_state.messages.append(
                {"role": "assistant", "result": to_session_payload(result)}
            )
        except Exception as error:
            message = (
                "暂时无法完成查询。请确认 `.env` 模型配置、结构化法条数据和索引均已准备好。"
                f"\n\n技术信息：{type(error).__name__}: {error}"
            )
            st.error(message)
            st.session_state.messages.append(
                {"role": "assistant", "content": message}
            )


def main() -> None:
    _apply_styles()
    laws = _load_laws()
    law_filter = _render_sidebar([law.law_name for law in laws])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        _render_hero(len(laws))
        st.markdown('<div class="section-kicker">试着这样问</div>', unsafe_allow_html=True)
        example_columns = st.columns(len(EXAMPLE_QUESTIONS))
        selected_example: str | None = None
        for column, question in zip(example_columns, EXAMPLE_QUESTIONS, strict=True):
            if column.button(question, use_container_width=True):
                selected_example = question
        st.markdown(
            '<div class="empty-state">也可以直接在下方输入真实情况。请避免填写身份证号、银行卡号等敏感信息。</div>',
            unsafe_allow_html=True,
        )
        with st.form("first_query", border=False):
            prompt = st.text_input(
                "法律问题",
                placeholder="描述你的法律问题…",
                max_chars=1000,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(
                "查找相关法条",
                type="primary",
                use_container_width=True,
            )
        if not submitted:
            prompt = None
    else:
        selected_example = None
        _render_history(st.session_state.messages)
        prompt = st.chat_input("描述你的法律问题…", max_chars=1000)

    query = selected_example or prompt
    if query:
        _answer(query.strip(), law_filter)


if __name__ == "__main__":
    main()
