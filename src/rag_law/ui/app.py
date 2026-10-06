"""RAG Law 的 Streamlit 演示页面。"""

from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

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


def _render_sidebar(law_names: list[str]) -> str | None:
    with st.sidebar:
        st.markdown('<div class="brand-mark">RAG Law</div>', unsafe_allow_html=True)
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
            with st.spinner("正在检索法条、核对证据…"):
                result = _load_pipeline().run(query, law_name=law_name)
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
