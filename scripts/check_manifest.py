"""校验 data/manifest.yaml：条目唯一性、字段完整性与文件路径有效性。

用法（项目根目录执行）：
    uv run python scripts/check_manifest.py
"""

import re
import sys

import yaml

from rag_law.config import PROJECT_ROOT

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# law_id 规范：cn_ 前缀 + 小写字母数字下划线
LAW_ID_PATTERN = re.compile(r"cn_[a-z0-9_]+")


def main() -> None:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        laws = yaml.safe_load(f)

    if not isinstance(laws, list):
        print(f"manifest 顶层必须是列表，实际是 {type(laws).__name__}")
        sys.exit(1)

    print(f"manifest 共 {len(laws)} 条：")
    errors: list[str] = []
    seen_ids: set[str] = set()
    listed_files: set[str] = set()

    for entry in laws:
        if not isinstance(entry, dict):
            errors.append(f"条目不是字典结构：{entry!r}")
            continue

        law_id = str(entry.get("law_id", ""))
        law_name = str(entry.get("law_name", ""))
        file_path = str(entry.get("file", "")).replace("\\", "/")

        if not LAW_ID_PATTERN.fullmatch(law_id):
            errors.append(f"law_id 命名不规范：{law_id!r}")
        if law_id in seen_ids:
            errors.append(f"law_id 重复：{law_id}")
        seen_ids.add(law_id)

        if not law_name:
            errors.append(f"{law_id} 缺少 law_name")
        if not file_path:
            errors.append(f"{law_id} 缺少 file")
        elif not (PROJECT_ROOT / file_path).is_file():
            errors.append(f"{law_id} 文件不存在：{file_path}")
        else:
            listed_files.add(file_path)

        print(f"  {law_id:<38} {law_name}")

    raw_files = {f"data/raw/{p.name}" for p in RAW_DIR.iterdir() if p.is_file()}
    for path in sorted(raw_files - listed_files):
        errors.append(f"raw 目录中的文件未被 manifest 覆盖：{path}")

    print()
    if errors:
        print(f"发现 {len(errors)} 处问题：")
        for item in errors:
            print("  -", item)
        sys.exit(1)

    print(
        f"校验通过：{len(laws)} 条，law_id 唯一且规范、字段完整、"
        f"路径有效、覆盖 raw 目录全部 {len(raw_files)} 个文件"
    )


if __name__ == "__main__":
    main()