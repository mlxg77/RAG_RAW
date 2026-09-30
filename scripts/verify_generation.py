"""验证真实 LLM 的结构化回答能力。"""

from rag_law.generation.chain import (
    load_generation_chains,
)
from rag_law.schemas import GeneratedAnswer


QUESTION = "公司没有和我签劳动合同怎么办？"

CONTEXT = """
[E001]
法律：《中华人民共和国劳动合同法》
条号：第十条
证据类型：直接命中
正文：
建立劳动关系，应当订立书面劳动合同。
""".strip()


def main() -> None:
    chains = load_generation_chains()

    answer = chains.answer_chain.invoke(
        {
            "question": QUESTION,
            "context": CONTEXT,
        }
    )

    if not isinstance(
        answer,
        GeneratedAnswer,
    ):
        raise TypeError(
            "结构化生成结果不是 GeneratedAnswer："
            f"{type(answer)!r}"
        )

    print(
        answer.model_dump_json(
            indent=2,
        )
    )


if __name__ == "__main__":
    main()