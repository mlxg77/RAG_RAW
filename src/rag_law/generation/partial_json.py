"""从尚未闭合的 JSON 文本里增量读取顶层字符串字段。

推理模型在 JSON 模式下会把对象一个字符一个字符地吐出来，
这里用一个小型扫描器把「已经写出来但还没收尾」的字段值取出来，
供前端做打字机效果。整段文本收齐后仍然交给 ``json.loads`` 做最终校验，
本模块只负责展示用的部分值，不承担正确性担保。
"""

from collections.abc import Sequence


WHITESPACE = " \t\r\n"
_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def parse_partial_string_fields(text: str) -> dict[str, str]:
    """返回顶层对象中「字符串类型」字段的当前值。

    - 只统计深度为 1 的字段，数组或嵌套对象里的同名字段不会干扰。
    - 值还没写完时返回已写出的部分；值不是字符串（数字、布尔、null）
      或尚未出现时，字段不会出现在结果里。
    """

    values: dict[str, str] = {}
    depth = 0
    index = 0
    length = len(text)
    key: str | None = None

    while index < length:
        char = text[index]

        if char == "{":
            depth += 1
            index += 1
            continue
        if char == "}":
            depth -= 1
            index += 1
            continue
        if char == "[":
            depth += 1
            index += 1
            continue
        if char == "]":
            depth -= 1
            index += 1
            continue
        if char in WHITESPACE or char in ":,":
            index += 1
            continue

        if char == '"':
            value, index, _closed = _read_string(text, index + 1)
            if depth != 1:
                continue
            if _key_follows(text, index):
                key = value
            elif key is not None:
                values[key] = value
            continue

        # 数字、true / false / null：跳过这个字面量，
        # 否则后续的字符串会被当成上一个键的值。
        cursor = index
        while cursor < length and text[cursor] not in ",}] \t\r\n":
            cursor += 1
        if depth == 1 and cursor < length:
            key = None
        index = max(cursor, index + 1)

    return values


def _read_string(text: str, index: int) -> tuple[str, int, bool]:
    """从 ``index`` 开始读一段字符串内容，返回 (值, 新下标, 是否已闭合)。"""

    length = len(text)
    buffer: list[str] = []

    while index < length:
        char = text[index]

        if char == "\\":
            if index + 1 >= length:
                # 转义序列还没到齐，等下一段文本再读。
                break
            escaped = text[index + 1]
            if escaped == "u":
                if index + 5 >= length:
                    break
                digits = text[index + 2 : index + 6]
                try:
                    buffer.append(chr(int(digits, 16)))
                except ValueError:
                    break
                index += 6
                continue
            buffer.append(_ESCAPES.get(escaped, escaped))
            index += 2
            continue

        if char == '"':
            return "".join(buffer), index + 1, True

        buffer.append(char)
        index += 1

    return "".join(buffer), length, False


def _key_follows(text: str, index: int) -> bool:
    """判断刚读完的字符串是不是键：后面是否跟着冒号。"""

    length = len(text)
    while index < length and text[index] in WHITESPACE:
        index += 1
    return index < length and text[index] == ":"


class PartialFieldTracker:
    """只关心指定字段的增量跟踪器，用于过滤掉不展示的字段。"""

    def __init__(self, fields: Sequence[str]) -> None:
        self._fields = tuple(fields)
        self._buffer = ""

    def feed(self, chunk: str) -> dict[str, str]:
        """追加一段文本，返回关注字段的当前值（已开始的字段才会出现）。"""

        self._buffer += chunk
        parsed = parse_partial_string_fields(self._buffer)
        return {
            field: value
            for field, value in parsed.items()
            if field in self._fields
        }

    @property
    def buffer(self) -> str:
        return self._buffer
