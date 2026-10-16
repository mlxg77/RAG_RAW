from pathlib import Path

from streamlit.testing.v1 import AppTest

from rag_law.knowledge_base import (
    AddDocumentResult,
    RebuildKnowledgeBaseResult,
)


APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "rag_law"
    / "ui"
    / "app.py"
)

COMPANY_LAW_TEXT = (
    "# 中华人民共和国公司法\n"
    "\n"
    "第一条　为了规范公司的组织和行为，制定本法。\n"
)


def test_streamlit_landing_page_renders() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)

    assert not app.exception
    assert app.selectbox[0].value == "全部法律"

    query_input = next(
        item
        for item in app.text_input
        if item.placeholder == "描述你的法律问题…"
    )
    assert query_input.label == "法律问题"

    assert app.file_uploader[0].label == "上传法律文件"

    assert {button.label for button in app.button} == {
        "清空当前对话",
        "上传并重建索引",
        "重新构建现有知识库",
        "公司一直没和我签书面劳动合同，我可以要求什么？",
        "酒后驾驶机动车会受到什么处罚？",
        "消费者买到不符合食品安全标准的食品，可以要求赔偿吗？",
        "查找相关法条",
    }


def test_upload_new_law_rebuilds_and_shows_notice(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_add_document(
        source_path,
        **kwargs,
    ) -> AddDocumentResult:
        calls["file_name"] = Path(source_path).name
        calls["content"] = Path(source_path).read_bytes()
        calls.update(kwargs)

        return AddDocumentResult(
            law_id=kwargs["law_id"],
            law_name="中华人民共和国公司法",
            file="data/raw/中华人民共和国公司法.md",
            article_count=2,
            chunk_count=2,
            law_count=16,
            chunk_total=3000,
            manifest_updated=True,
        )

    monkeypatch.setattr(
        "rag_law.knowledge_base.add_document",
        fake_add_document,
    )

    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)

    app.file_uploader[0].set_value(
        (
            "中华人民共和国公司法.md",
            COMPANY_LAW_TEXT.encode("utf-8"),
            "text/markdown",
        )
    )

    law_id_input = next(
        item
        for item in app.text_input
        if item.label == "law_id"
    )
    law_id_input.set_value("cn_company_law")

    upload_button = next(
        button
        for button in app.button
        if button.label == "上传并重建索引"
    )
    upload_button.click()

    app.run(timeout=30)

    assert not app.exception
    assert calls["law_id"] == "cn_company_law"
    assert calls["law_name"] is None
    assert calls["file_name"] == (
        "中华人民共和国公司法.md"
    )
    assert calls["content"] == (
        COMPANY_LAW_TEXT.encode("utf-8")
    )
    assert callable(calls["progress_callback"])
    assert any(
        "已成功添加" in item.value
        for item in app.success
    )


def test_rebuild_existing_knowledge_base_from_sidebar(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_rebuild_knowledge_base(
        **kwargs,
    ) -> RebuildKnowledgeBaseResult:
        calls.update(kwargs)
        kwargs["progress_callback"](
            "complete",
            "现有知识库重建全部完成。",
        )
        return RebuildKnowledgeBaseResult(
            law_count=15,
            chunk_total=2998,
        )

    monkeypatch.setattr(
        "rag_law.knowledge_base.rebuild_knowledge_base",
        fake_rebuild_knowledge_base,
    )

    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)
    rebuild_button = next(
        button
        for button in app.button
        if button.label == "重新构建现有知识库"
    )
    rebuild_button.click()
    app.run(timeout=30)

    assert not app.exception
    assert callable(calls["progress_callback"])
    assert any(
        "知识库重建完成" in item.value
        for item in app.success
    )
    assert any(
        "重启 API" in item.value
        for item in app.info
    )
