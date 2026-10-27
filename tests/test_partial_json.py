"""增量 JSON 解析：只用于打字机展示，最终仍由 json.loads 兜底。"""

from rag_law.generation.partial_json import (
    PartialFieldTracker,
    parse_partial_string_fields,
)


def test_reads_unfinished_string_value() -> None:
    text = '{"summary": "建立劳动关系应当'

    assert parse_partial_string_fields(text) == {
        "summary": "建立劳动关系应当"
    }


def test_ignores_nested_fields_and_arrays() -> None:
    text = (
        '{"citations": [{"evidence_id": "E001"}],'
        ' "summary": "结论"}'
    )

    assert parse_partial_string_fields(text) == {"summary": "结论"}


def test_decodes_escapes_and_partial_escape() -> None:
    complete = '{"analysis": "他说：\\"可以\\"，但要看情况"}'

    assert parse_partial_string_fields(complete) == {
        "analysis": '他说："可以"，但要看情况'
    }

    # 末尾刚出现反斜杠时，转义还没到齐，先保留已经解出的部分。
    trailing = '{"analysis": "最后一行\\'
    assert parse_partial_string_fields(trailing) == {"analysis": "最后一行"}


def test_skips_non_string_values_between_fields() -> None:
    text = (
        '{"summary": "结论", "order": 3,'
        ' "follow_up_question": null,'
        ' "limitations": "仅供参考"}'
    )

    assert parse_partial_string_fields(text) == {
        "summary": "结论",
        "limitations": "仅供参考",
    }


def test_tracker_accumulates_across_chunks() -> None:
    tracker = PartialFieldTracker(("summary", "analysis"))
    snapshots: list[dict[str, str]] = []

    for chunk in ('{"summary"', ': "先', "结论", '", "analysis"', ': "再分析'):
        snapshots.append(tracker.feed(chunk))

    assert snapshots[-1] == {
        "summary": "先结论",
        "analysis": "再分析",
    }
    # 没出现的字段不应被塞进结果。
    assert snapshots[0] == {}
