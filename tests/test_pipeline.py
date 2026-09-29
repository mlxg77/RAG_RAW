"""阶段 1 全量处理流水线测试。"""

import json
from pathlib import Path

import yaml

from rag_law.ingestion.pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_with_three_real_formats(
    tmp_path: Path,
) -> None:
    entries = [
        {
            "law_id": "cn_labor_contract_law",
            "law_name": "中华人民共和国劳动合同法",
            "file": (
                "data/raw/"
                "中华人民共和国劳动合同法.pdf"
            ),
        },
        {
            "law_id": "cn_criminal_law",
            "law_name": "中华人民共和国刑法",
            "file": (
                "data/raw/"
                "中华人民共和国刑法.docx"
            ),
        },
        {
            "law_id": "cn_road_traffic_safety_law",
            "law_name": "中华人民共和国道路交通安全法",
            "file": (
                "data/raw/"
                "中华人民共和国道路交通安全法.md"
            ),
        },
    ]

    manifest_path = tmp_path / "manifest.yaml"
    output_path = tmp_path / "processed" / "articles.jsonl"
    report_path = tmp_path / "processed" / "parse_report.json"

    manifest_path.write_text(
        yaml.safe_dump(
            entries,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report = run_pipeline(
        manifest_path=manifest_path,
        output_path=output_path,
        report_path=report_path,
        project_root=PROJECT_ROOT,
        max_chars=1200,
    )

    assert output_path.is_file()
    assert report_path.is_file()

    assert report["summary"]["law_count"] == 3
    assert report["summary"]["article_count"] >= 600
    assert report["summary"]["empty_article_count"] == 0
    assert report["summary"]["duplicate_article_count"] == 0

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    records = [
        json.loads(line)
        for line in lines
    ]

    assert len(records) == report["summary"]["chunk_count"]

    chunk_ids = [
        record["chunk_id"]
        for record in records
    ]

    assert len(chunk_ids) == len(set(chunk_ids))

    assert {
        record["law_id"]
        for record in records
    } == {
        "cn_labor_contract_law",
        "cn_criminal_law",
        "cn_road_traffic_safety_law",
    }

    assert all(
        record["text"].strip()
        for record in records
    )

    # 相同输入重复运行必须产生完全相同的 JSONL。
    first_output = output_path.read_bytes()
    first_report = report_path.read_bytes()

    run_pipeline(
        manifest_path=manifest_path,
        output_path=output_path,
        report_path=report_path,
        project_root=PROJECT_ROOT,
        max_chars=1200,
    )

    assert output_path.read_bytes() == first_output
    assert report_path.read_bytes() == first_report