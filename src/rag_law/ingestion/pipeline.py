"""阶段 1 全量文档处理流水线。

运行方式：

    uv run python -m rag_law.ingestion.pipeline
"""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from rag_law.ingestion.article_splitter import split_articles
from rag_law.ingestion.loaders import load_document
from rag_law.ingestion.normalizer import normalize_document
from rag_law.schemas import ArticleChunk


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.yaml"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "articles.jsonl"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "parse_report.json"
)
DEFAULT_MAX_CHARS = 1200


def load_manifest(
    manifest_path: Path,
) -> list[dict[str, str]]:
    """读取并校验 manifest 的基础结构。"""

    with manifest_path.open(encoding="utf-8") as file:
        raw_entries = yaml.safe_load(file)

    if not isinstance(raw_entries, list):
        raise ValueError("manifest 顶层必须是列表")

    entries: list[dict[str, str]] = []
    seen_law_ids: set[str] = set()

    for index, raw_entry in enumerate(
        raw_entries,
        start=1,
    ):
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"manifest 第 {index} 项不是字典"
            )

        entry: dict[str, str] = {}

        for field_name in ("law_id", "law_name", "file"):
            value = raw_entry.get(field_name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"manifest 第 {index} 项缺少有效的 "
                    f"{field_name}"
                )

            entry[field_name] = value.strip()

        law_id = entry["law_id"]

        if law_id in seen_law_ids:
            raise ValueError(
                f"manifest 中 law_id 重复：{law_id}"
            )

        seen_law_ids.add(law_id)
        entries.append(entry)

    return entries


def _write_jsonl_atomic(
    path: Path,
    chunks: list[ArticleChunk],
) -> None:
    """先写临时文件，成功后再替换正式 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for chunk in chunks:
            record = chunk.model_dump(mode="json")

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    temporary_path.replace(path)


def _write_json_atomic(
    path: Path,
    value: dict[str, Any],
) -> None:
    """先写临时文件，成功后再替换正式 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temporary_path.replace(path)


def run_pipeline(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    project_root: Path = PROJECT_ROOT,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """处理 manifest 中的全部法律并写出结果。"""

    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")

    entries = load_manifest(manifest_path)

    all_chunks: list[ArticleChunk] = []
    law_reports: list[dict[str, Any]] = []

    for entry in entries:
        source_file = entry["file"].replace("\\", "/")
        source_path = project_root / source_file

        loaded = load_document(
            source_path,
            law_id=entry["law_id"],
            law_name=entry["law_name"],
            source_file=source_file,
        )

        normalized = normalize_document(loaded)

        split_result = split_articles(
            normalized,
            max_chars=max_chars,
        )

        oversized_chunks = [
            chunk
            for chunk in split_result.chunks
            if len(chunk.text) > max_chars
        ]

        max_chunk_chars = max(
            (
                len(chunk.text)
                for chunk in split_result.chunks
            ),
            default=0,
        )

        status = (
            "warning"
            if split_result.warnings
            else "ok"
        )

        law_report = {
            "law_id": entry["law_id"],
            "law_name": entry["law_name"],
            "source_file": source_file,
            "source_format": loaded.source_format,
            "page_count": loaded.page_count,
            "article_count": split_result.article_count,
            "chunk_count": len(split_result.chunks),
            "empty_article_count": len(
                split_result.empty_articles
            ),
            "empty_articles": split_result.empty_articles,
            "duplicate_article_count": len(
                split_result.duplicate_articles
            ),
            "duplicate_articles": (
                split_result.duplicate_articles
            ),
            "oversized_chunk_count": len(
                oversized_chunks
            ),
            "max_chunk_chars": max_chunk_chars,
            "warnings": split_result.warnings,
            "status": status,
        }

        law_reports.append(law_report)
        all_chunks.extend(split_result.chunks)

    chunk_ids = [
        chunk.chunk_id
        for chunk in all_chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError(
            "全量结果中出现重复 chunk_id，停止写出"
        )

    summary = {
        "law_count": len(law_reports),
        "article_count": sum(
            report["article_count"]
            for report in law_reports
        ),
        "chunk_count": len(all_chunks),
        "empty_article_count": sum(
            report["empty_article_count"]
            for report in law_reports
        ),
        "duplicate_article_count": sum(
            report["duplicate_article_count"]
            for report in law_reports
        ),
        "oversized_chunk_count": sum(
            report["oversized_chunk_count"]
            for report in law_reports
        ),
        "warning_law_count": sum(
            report["status"] == "warning"
            for report in law_reports
        ),
        "max_chars": max_chars,
    }

    report: dict[str, Any] = {
        "summary": summary,
        "laws": law_reports,
    }

    # 全部法律都成功处理以后才替换正式文件。
    _write_jsonl_atomic(output_path, all_chunks)
    _write_json_atomic(report_path, report)

    return report


def _resolve_cli_path(value: str) -> Path:
    """将命令行相对路径解释为相对项目根目录。"""

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="解析法律原始文件并生成法条 JSONL"
    )

    parser.add_argument(
        "--manifest",
        default="data/manifest.yaml",
        help="manifest 路径",
    )
    parser.add_argument(
        "--output",
        default="data/processed/articles.jsonl",
        help="JSONL 输出路径",
    )
    parser.add_argument(
        "--report",
        default="data/processed/parse_report.json",
        help="质量报告输出路径",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="单个 chunk 的建议最大字符数",
    )

    arguments = parser.parse_args()

    report = run_pipeline(
        manifest_path=_resolve_cli_path(
            arguments.manifest
        ),
        output_path=_resolve_cli_path(
            arguments.output
        ),
        report_path=_resolve_cli_path(
            arguments.report
        ),
        max_chars=arguments.max_chars,
    )

    summary = report["summary"]

    print("阶段 1 解析完成")
    print(f"  法律数量：{summary['law_count']}")
    print(f"  法条数量：{summary['article_count']}")
    print(f"  chunk 数量：{summary['chunk_count']}")
    print(f"  空条数量：{summary['empty_article_count']}")
    print(
        f"  重复条号数量："
        f"{summary['duplicate_article_count']}"
    )
    print(
        f"  超长 chunk 数量："
        f"{summary['oversized_chunk_count']}"
    )
    print(
        f"  有警告的法律："
        f"{summary['warning_law_count']}"
    )


if __name__ == "__main__":
    main()