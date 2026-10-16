"""知识库维护：添加法律文件并自动重建全链产物。

一次完整的添加流程：

1. 校验新文件格式，并预演"加载 → 清洗 → 切分"确认能产出法条；
2. 复制文件到 data/raw 并登记 manifest.yaml；
3. 重跑阶段 1 流水线，重新生成 articles.jsonl 与解析报告；
4. 重建 BM25 索引；
5. 重建 Chroma 向量库。

整个流程可安全重试：同一条目已登记且信息一致时，
跳过复制与登记，直接重新执行重建链。
"""

import filecmp
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml
from langchain_core.embeddings import Embeddings

from rag_law.ingestion.article_splitter import (
    split_articles,
)
from rag_law.ingestion.loaders import (
    SUPPORTED_SUFFIXES,
    load_document,
)
from rag_law.ingestion.normalizer import (
    normalize_document,
)
from rag_law.ingestion.pipeline import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MAX_CHARS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REPORT_PATH,
    load_manifest,
    run_pipeline,
)
from rag_law.retrieval.bm25 import (
    DEFAULT_BM25_INDEX_PATH,
    BM25Retriever,
)
from rag_law.retrieval.dense import (
    DEFAULT_CHROMA_DIRECTORY,
    DenseRetriever,
)
from rag_law.schemas import SplitResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 与 scripts/check_manifest.py 保持一致的命名规范：
# cn_ 前缀 + 小写字母数字下划线。
LAW_ID_PATTERN = re.compile(r"cn_[a-z0-9_]+")

# 进度回调：(stage, message)，与 AnswerPipeline 的约定一致。
# ProgressCallback 指的是"接收两个字符串、不返回东西"的函数。
# Callable是Python标准库typing模块中的一个类型提示工具。
ProgressCallback = Callable[[str, str], None]

# 函数返回一组只读的结构化结果时，用 @dataclass(frozen=True)；
# @dataclass：自动生成样板代码，如 __init__、__repr__ 等。
# frozen=True 表示不可变数据类，对象一旦创建，字段就不能再改。
@dataclass(frozen=True)
class AddDocumentResult:
    """一次"添加法律文件并重建"的结果摘要。"""

    law_id: str
    law_name: str
    file: str
    article_count: int
    chunk_count: int
    law_count: int
    chunk_total: int
    manifest_updated: bool


@dataclass(frozen=True)
class RebuildKnowledgeBaseResult:
    """一次全量知识库重建的结果摘要。"""

    law_count: int
    chunk_total: int


def _emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    message: str,
) -> None:
    """上报进度；未提供回调时退回控制台打印。"""

    if progress_callback is not None:
        progress_callback(stage, message)
    else:
        print(message)


def _write_text_atomic(
    path: Path,
    text: str,
) -> None:
    """先写临时文件，成功后再替换正式文本文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(text)

    temporary_path.replace(path)


def _append_manifest_entry(
    manifest_path: Path,
    entry: dict[str, str],
) -> None:
    """把新条目追加到 manifest 末尾，保留原有注释和排版。"""

    text = manifest_path.read_text(encoding="utf-8")

    if text and not text.endswith("\n"):
        text += "\n"

    block = yaml.safe_dump(
        [entry],
        allow_unicode=True,
        sort_keys=False,
        width=4096,
    )

    updated = f"{text}\n{block}" if text else block

    # 追加后必须仍然是合法的条目列表，否则不改动原文件。
    parsed = yaml.safe_load(updated)

    if not isinstance(parsed, list):
        raise RuntimeError(
            "manifest 追加条目后不再是列表结构"
        )

    _write_text_atomic(manifest_path, updated)


def _validate_source_file(
    source_path: Path,
    *,
    law_id: str,
    law_name: str,
    source_file: str,
    max_chars: int,
) -> SplitResult:
    """预演"加载 → 清洗 → 切分"，确认文件能产出法条。"""

    loaded = load_document(
        source_path,
        law_id=law_id,
        law_name=law_name,
        source_file=source_file,
    )

    normalized = normalize_document(loaded)

    split_result = split_articles(
        normalized,
        max_chars=max_chars,
    )

    if split_result.article_count == 0:
        raise ValueError(
            "未识别到任何法条，文件内容可能不是法律文本："
            f"{source_path}"
        )

    return split_result


def _rebuild_all(
    *,
    manifest_path: Path,
    project_root: Path,
    articles_path: Path,
    report_path: Path,
    bm25_index_path: Path,
    chroma_directory: Path,
    max_chars: int,
    embeddings: Embeddings | None,
    progress_callback: ProgressCallback | None,
    completion_message: str,
) -> dict:
    """从 manifest 重建结构化语料、BM25 与 Chroma。"""

    _emit_progress(
        progress_callback,
        "pipeline",
        "正在重跑阶段 1 流水线…",
    )

    report = run_pipeline(
        manifest_path=manifest_path,
        output_path=articles_path,
        report_path=report_path,
        project_root=project_root,
        max_chars=max_chars,
    )

    summary = report["summary"]

    _emit_progress(
        progress_callback,
        "pipeline",
        "阶段 1 完成："
        f"全库 {summary['law_count']} 部法律、"
        f"{summary['chunk_count']} 个 chunk",
    )

    _emit_progress(
        progress_callback,
        "bm25",
        "正在重建 BM25 索引…",
    )

    bm25_retriever = BM25Retriever.build(articles_path)
    bm25_retriever.save(bm25_index_path)

    _emit_progress(
        progress_callback,
        "bm25",
        "BM25 索引重建完成："
        f"{len(bm25_retriever.chunks)} 条法条",
    )

    _emit_progress(
        progress_callback,
        "dense",
        "正在重建 Chroma 向量库"
        "（将调用 Embedding 接口，请耐心等待）…",
    )

    DenseRetriever.build(
        articles_path=articles_path,
        persist_directory=chroma_directory,
        embeddings=embeddings,
    )

    _emit_progress(
        progress_callback,
        "dense",
        "Chroma 向量库重建完成",
    )

    _emit_progress(
        progress_callback,
        "complete",
        completion_message,
    )

    return report


def rebuild_knowledge_base(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    project_root: Path = PROJECT_ROOT,
    articles_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    bm25_index_path: Path = DEFAULT_BM25_INDEX_PATH,
    chroma_directory: Path = DEFAULT_CHROMA_DIRECTORY,
    max_chars: int = DEFAULT_MAX_CHARS,
    embeddings: Embeddings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> RebuildKnowledgeBaseResult:
    """使用已有 manifest 和原始文件全量重建知识库。"""

    report = _rebuild_all(
        manifest_path=manifest_path,
        project_root=project_root,
        articles_path=articles_path,
        report_path=report_path,
        bm25_index_path=bm25_index_path,
        chroma_directory=chroma_directory,
        max_chars=max_chars,
        embeddings=embeddings,
        progress_callback=progress_callback,
        completion_message="现有知识库重建全部完成。",
    )
    summary = report["summary"]

    return RebuildKnowledgeBaseResult(
        law_count=summary["law_count"],
        chunk_total=summary["chunk_count"],
    )


def add_document(
    source_path: Path,
    *,
    law_id: str,
    law_name: str | None = None,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    project_root: Path = PROJECT_ROOT,
    articles_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    bm25_index_path: Path = DEFAULT_BM25_INDEX_PATH,
    chroma_directory: Path = DEFAULT_CHROMA_DIRECTORY,
    max_chars: int = DEFAULT_MAX_CHARS,
    embeddings: Embeddings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AddDocumentResult:
    """添加法律文件并自动重建语料与全部索引。"""

    source_path = Path(source_path).resolve()

    if not source_path.is_file():
        raise FileNotFoundError(
            f"法律文件不存在：{source_path}"
        )

    suffix = source_path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"不支持的文件格式 {suffix!r}：{source_path}；"
            f"当前支持 {sorted(SUPPORTED_SUFFIXES)}"
        )

    if not LAW_ID_PATTERN.fullmatch(law_id):
        raise ValueError(
            "law_id 命名不规范（应为 cn_ 前缀加"
            f"小写字母数字下划线）：{law_id!r}"
        )

    final_name = (law_name or source_path.stem).strip()

    if not final_name:
        raise ValueError("法律名称不能为空")

    relative_file = f"data/raw/{source_path.name}"
    destination = (
        project_root
        / "data"
        / "raw"
        / source_path.name
    )

    entries = load_manifest(manifest_path)

    # next(生成器, 默认值) 是 Python 里取第一个满足条件的元素的惯用写法，找不到就返回默认值。
    existing = next(
        (
            entry
            for entry in entries
            if entry["law_id"] == law_id
        ),
        None,
    )

    if existing is not None:
        if (
            existing["file"] != relative_file
            or existing["law_name"] != final_name
        ):
            raise ValueError(
                f"law_id 已登记且信息不同：{law_id}；"
                "更新内容请放置同路径文件后重跑，"
                "改名请手动编辑 manifest"
            )
    elif any(
        entry["file"] == relative_file
        for entry in entries
    ):
        raise ValueError(
            "该文件已登记在其他 law_id 下："
            f"{relative_file}"
        )

    # 任何写入之前先预演完整切分流程，确保文件有效。
    _emit_progress(
        progress_callback,
        "validate",
        f"正在校验文件：{source_path.name}",
    )

    split_result = _validate_source_file(
        source_path,
        law_id=law_id,
        law_name=final_name,
        source_file=relative_file,
        max_chars=max_chars,
    )

    _emit_progress(
        progress_callback,
        "validate",
        f"校验通过：《{final_name}》"
        f"识别到 {split_result.article_count} 条法条",
    )

    same_location = (
        destination.resolve() == source_path
    )

    if not same_location:
        if destination.exists():
            # 重试场景：上次运行已经把同名文件复制到位。
            if not filecmp.cmp(
                source_path,
                destination,
                shallow=False,
            ):
                raise ValueError(
                    "data/raw 中已存在同名但内容不同的"
                    f"文件：{source_path.name}；"
                    "如需更新请替换该文件后直接重跑，"
                    "或先移除该文件"
                )
        else:
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            shutil.copyfile(source_path, destination)
            _emit_progress(
                progress_callback,
                "copy",
                "已复制文件到："
                f"data/raw/{source_path.name}",
            )

    if existing is None:
        _append_manifest_entry(
            manifest_path,
            {
                "law_id": law_id,
                "law_name": final_name,
                "file": relative_file,
            },
        )
        _emit_progress(
            progress_callback,
            "manifest",
            "已登记 manifest.yaml："
            f"共 {len(entries) + 1} 部法律",
        )
    else:
        _emit_progress(
            progress_callback,
            "manifest",
            "该法律此前已登记，直接执行重建",
        )

    report = _rebuild_all(
        manifest_path=manifest_path,
        project_root=project_root,
        articles_path=articles_path,
        report_path=report_path,
        bm25_index_path=bm25_index_path,
        chroma_directory=chroma_directory,
        max_chars=max_chars,
        embeddings=embeddings,
        progress_callback=progress_callback,
        completion_message="添加与重建全部完成。",
    )

    law_report = next(
        (
            item
            for item in report["laws"]
            if item["law_id"] == law_id
        ),
        None,
    )

    if law_report is None:
        raise RuntimeError(
            f"流水线结果中缺少法律记录：{law_id}"
        )

    summary = report["summary"]

    return AddDocumentResult(
        law_id=law_id,
        law_name=final_name,
        file=relative_file,
        article_count=law_report["article_count"],
        chunk_count=law_report["chunk_count"],
        law_count=summary["law_count"],
        chunk_total=summary["chunk_count"],
        manifest_updated=existing is None,
    )
