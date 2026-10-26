"""法律清单读取：API 与前端共用的筛选项。"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LawOption:
    """法律筛选器中的一项。"""

    law_id: str
    law_name: str


def load_law_options(manifest_path: Path) -> list[LawOption]:
    """从 manifest 读取稳定、有序的法律筛选项。"""

    with manifest_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, list):
        raise ValueError("法律清单必须是列表")

    options: list[LawOption] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("法律清单中的每一项必须是对象")

        law_id = str(item.get("law_id", "")).strip()
        law_name = str(item.get("law_name", "")).strip()
        if not law_id or not law_name:
            raise ValueError("法律清单缺少 law_id 或 law_name")

        options.append(LawOption(law_id=law_id, law_name=law_name))

    return options
