"""添加法律文件并自动重建全链的测试。"""

from pathlib import Path

import pytest
import yaml
from langchain_core.embeddings import Embeddings

from rag_law.cli import (
    build_parser,
    main as cli_main,
    print_add_document_result,
)
from rag_law.knowledge_base import (
    AddDocumentResult,
    add_document,
)
from rag_law.retrieval.bm25 import BM25Retriever


LABOR_LAW_TEXT = (
    "# 中华人民共和国劳动法\n"
    "\n"
    "第一条　为了保护劳动者的合法权益，制定本法。\n"
    "\n"
    "第二条　在中华人民共和国境内的企业"
    "与劳动者，适用本法。\n"
)

COMPANY_LAW_TEXT = (
    "# 中华人民共和国公司法\n"
    "\n"
    "第一条　为了规范公司的组织和行为，制定本法。\n"
    "\n"
    "第二条　本法所称公司，是指依照本法设立的"
    "有限责任公司和股份有限公司。\n"
)

PLAIN_TEXT = (
    "这里没有任何法条结构，只是一段普通文字。\n"
)


class FakeEmbeddings(Embeddings):
    """只用于测试的确定性假向量模型。"""

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [float(len(text) % 7 + 1), 1.0]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            self._embed(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self._embed(text)


def make_environment(
    tmp_path: Path,
) -> dict[str, Path]:
    """搭建一份隔离的迷你知识库目录。"""

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    manifest_path = tmp_path / "data" / "manifest.yaml"
    manifest_path.write_text(
        "# 法律清单（测试）\n"
        "\n"
        "- law_id: cn_labor_law\n"
        "  law_name: 中华人民共和国劳动法\n"
        "  file: data/raw/中华人民共和国劳动法.md\n",
        encoding="utf-8",
    )

    (raw_dir / "中华人民共和国劳动法.md").write_text(
        LABOR_LAW_TEXT,
        encoding="utf-8",
    )

    return {
        "project_root": tmp_path,
        "manifest_path": manifest_path,
        "articles_path": (
            tmp_path
            / "data"
            / "processed"
            / "articles.jsonl"
        ),
        "report_path": (
            tmp_path
            / "data"
            / "processed"
            / "parse_report.json"
        ),
        "bm25_index_path": (
            tmp_path
            / "indexes"
            / "bm25"
            / "bm25_index.json"
        ),
        "chroma_directory": (
            tmp_path / "indexes" / "chroma"
        ),
    }


def write_source_file(
    tmp_path: Path,
    name: str,
    text: str,
) -> Path:
    """在项目目录之外模拟一份待添加的原始文件。"""

    source_dir = tmp_path / "uploads"
    source_dir.mkdir(exist_ok=True)

    source_path = source_dir / name
    source_path.write_text(text, encoding="utf-8")

    return source_path


def call_add_document(
    environment: dict[str, Path],
    source_path: Path,
    *,
    law_id: str = "cn_company_law",
    law_name: str | None = None,
    progress_callback=None,
) -> AddDocumentResult:
    return add_document(
        source_path,
        law_id=law_id,
        law_name=law_name,
        manifest_path=environment["manifest_path"],
        project_root=environment["project_root"],
        articles_path=environment["articles_path"],
        report_path=environment["report_path"],
        bm25_index_path=environment["bm25_index_path"],
        chroma_directory=environment["chroma_directory"],
        embeddings=FakeEmbeddings(),
        progress_callback=progress_callback,
    )


def test_add_document_builds_full_chain(
    tmp_path: Path,
) -> None:
    environment = make_environment(tmp_path)

    source_path = write_source_file(
        tmp_path,
        "中华人民共和国公司法.md",
        COMPANY_LAW_TEXT,
    )

    result = call_add_document(
        environment,
        source_path,
    )

    assert result.law_id == "cn_company_law"
    assert result.law_name == "中华人民共和国公司法"
    assert result.file == (
        "data/raw/中华人民共和国公司法.md"
    )
    assert result.article_count == 2
    assert result.chunk_count == 2
    assert result.law_count == 2
    assert result.chunk_total == 4
    assert result.manifest_updated is True

    # 文件已复制进 data/raw。
    assert (
        environment["project_root"]
        / "data"
        / "raw"
        / "中华人民共和国公司法.md"
    ).is_file()

    # manifest 保留原有注释并新增一条登记。
    manifest_text = environment[
        "manifest_path"
    ].read_text(encoding="utf-8")

    assert "# 法律清单（测试）" in manifest_text

    entries = yaml.safe_load(manifest_text)

    assert len(entries) == 2
    assert entries[1] == {
        "law_id": "cn_company_law",
        "law_name": "中华人民共和国公司法",
        "file": "data/raw/中华人民共和国公司法.md",
    }

    # 新语料包含两部法律，共 4 条 chunk。
    articles_text = environment[
        "articles_path"
    ].read_text(encoding="utf-8")

    assert "cn_labor_law" in articles_text
    assert "cn_company_law" in articles_text
    assert (
        len(articles_text.strip().splitlines())
        == 4
    )

    # BM25 索引与 Chroma 向量库均已重建。
    bm25_retriever = BM25Retriever.load(
        environment["bm25_index_path"]
    )

    assert len(bm25_retriever.chunks) == 4

    assert (
        environment["chroma_directory"]
        / "chroma.sqlite3"
    ).is_file()


def test_add_document_emits_progress_events(
    tmp_path: Path,
) -> None:
    environment = make_environment(tmp_path)

    source_path = write_source_file(
        tmp_path,
        "中华人民共和国公司法.md",
        COMPANY_LAW_TEXT,
    )

    events: list[tuple[str, str]] = []

    call_add_document(
        environment,
        source_path,
        progress_callback=(
            lambda stage, message: events.append(
                (stage, message)
            )
        ),
    )

    stages = [stage for stage, _ in events]

    assert stages[0] == "validate"
    assert "bm25" in stages
    assert "dense" in stages
    assert stages[-1] == "complete"
    assert any(
        "校验通过" in message
        for _, message in events
    )


def test_add_document_rejects_unsupported_suffix(
    tmp_path: Path,
) -> None:
    environment = make_environment(tmp_path)

    manifest_before = environment[
        "manifest_path"
    ].read_text(encoding="utf-8")

    source_path = write_source_file(
        tmp_path,
        "中华人民共和国公司法.txt",
        COMPANY_LAW_TEXT,
    )

    with pytest.raises(ValueError) as error_info:
        call_add_document(environment, source_path)

    assert "不支持的文件格式" in str(
        error_info.value
    )

    assert (
        environment["manifest_path"].read_text(
            encoding="utf-8"
        )
        == manifest_before
    )
    assert not environment["articles_path"].exists()


def test_add_document_rejects_bad_law_id(
    tmp_path: Path,
) -> None:
    environment = make_environment(tmp_path)

    source_path = write_source_file(
        tmp_path,
        "中华人民共和国公司法.md",
        COMPANY_LAW_TEXT,
    )

    with pytest.raises(ValueError) as error_info:
        call_add_document(
            environment,
            source_path,
            law_id="company_law",
        )

    assert "law_id 命名不规范" in str(
        error_info.value
    )


def test_add_document_rejects_duplicate_law_id(
    tmp_path: Path,
) -> None:
    environment = make_environment(tmp_path)

    source_path = write_source_file(
        tmp_path,
        "中华人民共和国公司法.md",
        COMPANY_LAW_TEXT,
    )

    with pytest.raises(ValueError) as error_info:
        call_add_document(
            environment,
            source_path,
            law_id="cn_labor_law",
        )

    assert "已登记且信息不同" in str(
        error_info.value
    )


def test_add_document_rejects_file_without_articles(
    tmp_path: Path,
) -> None:
    environment = make_environment(tmp_path)

    manifest_before = environment[
        "manifest_path"
    ].read_text(encoding="utf-8")

    source_path = write_source_file(
        tmp_path,
        "中华人民共和国公司法.md",
        PLAIN_TEXT,
    )

    with pytest.raises(ValueError) as error_info:
        call_add_document(environment, source_path)

    assert "未识别到任何法条" in str(
        error_info.value
    )

    # 校验失败发生在任何写入之前：
    # manifest 不变、未复制文件、未生成语料。
    assert (
        environment["manifest_path"].read_text(
            encoding="utf-8"
        )
        == manifest_before
    )
    assert not (
        environment["project_root"]
        / "data"
        / "raw"
        / "中华人民共和国公司法.md"
    ).exists()
    assert not environment["articles_path"].exists()


def test_add_document_rerun_skips_registration(
    tmp_path: Path,
) -> None:
    environment = make_environment(tmp_path)

    source_path = write_source_file(
        tmp_path,
        "中华人民共和国公司法.md",
        COMPANY_LAW_TEXT,
    )

    first = call_add_document(
        environment,
        source_path,
    )

    second = call_add_document(
        environment,
        source_path,
    )

    assert first.manifest_updated is True
    assert second.manifest_updated is False
    assert second.chunk_total == 4

    entries = yaml.safe_load(
        environment["manifest_path"].read_text(
            encoding="utf-8"
        )
    )

    assert len(entries) == 2

    bm25_retriever = BM25Retriever.load(
        environment["bm25_index_path"]
    )

    assert len(bm25_retriever.chunks) == 4


def test_add_document_registers_file_already_in_raw(
    tmp_path: Path,
) -> None:
    environment = make_environment(tmp_path)

    raw_path = (
        environment["project_root"]
        / "data"
        / "raw"
        / "中华人民共和国公司法.md"
    )
    raw_path.write_text(
        COMPANY_LAW_TEXT,
        encoding="utf-8",
    )

    result = call_add_document(
        environment,
        raw_path,
    )

    assert result.manifest_updated is True
    assert result.chunk_total == 4


def test_parser_accepts_add_document_command() -> None:
    parser = build_parser()

    arguments = parser.parse_args(
        [
            "add-document",
            "--file",
            "data/raw/中华人民共和国公司法.md",
            "--law-id",
            "cn_company_law",
        ]
    )

    assert arguments.command == "add-document"
    assert arguments.law_id == "cn_company_law"
    assert arguments.law_name is None
    assert arguments.max_chars == 1200


def test_main_dispatches_add_document(
    monkeypatch,
    capsys,
) -> None:
    captured: dict[str, object] = {}

    def fake_add_document(
        source_path,
        **kwargs,
    ) -> AddDocumentResult:
        captured["source_path"] = source_path
        captured.update(kwargs)

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
        "rag_law.cli.add_document",
        fake_add_document,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "rag-law",
            "add-document",
            "--file",
            "data/raw/中华人民共和国公司法.md",
            "--law-id",
            "cn_company_law",
        ],
    )

    cli_main()

    printed = capsys.readouterr().out

    assert "法律文件添加完成" in printed
    assert "中华人民共和国公司法" in printed
    assert captured["law_id"] == "cn_company_law"
    assert captured["max_chars"] == 1200
    assert Path(
        str(captured["source_path"])
    ) == Path("data/raw/中华人民共和国公司法.md")


def test_result_printer_shows_rerun_note(
    capsys,
) -> None:
    print_add_document_result(
        AddDocumentResult(
            law_id="cn_company_law",
            law_name="中华人民共和国公司法",
            file="data/raw/中华人民共和国公司法.md",
            article_count=2,
            chunk_count=2,
            law_count=16,
            chunk_total=3000,
            manifest_updated=False,
        )
    )

    printed = capsys.readouterr().out

    assert "法律文件添加完成" in printed
    assert "此前已登记" in printed
