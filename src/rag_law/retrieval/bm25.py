"""
    基于 jieba 和 rank-bm25 的中文法条检索。
    整体流程：
            articles.jsonl ──build──▶ 内存里的 BM25Retriever ──save──▶ bm25_index.json
                                                                         │
            CLI / UI / fusion / 评测 ◀──────────────load─────────────────┘
"""

import argparse
import json
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from rag_law.retrieval.common import (
    DEFAULT_ARTICLES_PATH,
    PROJECT_ROOT,
    build_search_text,
    load_articles,
    make_retrieval_result,
)
from rag_law.schemas import (
    ArticleChunk,
    RetrievalResult,
)


DEFAULT_BM25_INDEX_PATH = (
    PROJECT_ROOT
    / "indexes"
    / "bm25"
    / "bm25_index.json"
)

# 连续中文、英文单词、阿拉伯数字。
TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]+"
    r"|[A-Za-z]+"
    r"|\d+"
)


def tokenize(text: str) -> list[str]:
    """对中文法律文本进行确定性的基础分词。
    输入：劳动合同法第38条，用人单位（ABC）
    输出：['劳动', '合同法', '第', '38', '条', '用人单位', 'abc']
    """

    normalized = (
        text.replace("\u3000", " ")
        .replace("\xa0", " ")
        .strip()
    )

    tokens: list[str] = []

    # findall 只做粗切：它从左往右扫描文本，把连续的汉字、连续的英文字母、连续的阿拉伯数字各自拿出来作为一段，其他字符（标点、空格、符号）一律跳过。
    #     '劳动合同法第38条，用人单位（ABC）'
    #   -> ['劳动合同法第', '38', '条', '用人单位', 'ABC']

    for segment in TOKEN_PATTERN.findall(normalized):
        if re.fullmatch(
            r"[\u3400-\u4dbf\u4e00-\u9fff]+",
            segment,
        ):
            segment_tokens = jieba.lcut(
                segment,
                cut_all=False,
            )
        else:
            segment_tokens = [segment.lower()]

        tokens.extend(
            token.strip().lower()
            for token in segment_tokens
            if token.strip()
        )

    return tokens


class BM25Retriever:
    """内存中的 BM25 检索器。"""

    def __init__(
        self,
        chunks: list[ArticleChunk],
        tokenized_corpus: list[list[str]],
    ) -> None:
        if not chunks:
            raise ValueError("BM25 语料不能为空")

        if len(chunks) != len(tokenized_corpus):
            raise ValueError(
                "法条数量与分词结果数量不一致"
            )

        self.chunks = chunks
        # 份语料分好词之后的结果
        self.tokenized_corpus = tokenized_corpus
        # 用来给分好词的法条打关键词相关性分数。
        self._bm25 = BM25Okapi(tokenized_corpus)

    @classmethod
    def build(
        cls,
        articles_path: Path = DEFAULT_ARTICLES_PATH,
    ) -> "BM25Retriever":
        """从阶段 1 JSONL 构建索引。"""

        chunks = load_articles(articles_path)

        tokenized_corpus = [
            tokenize(build_search_text(chunk))
            for chunk in chunks
        ]

        return cls(
            chunks=chunks,
            tokenized_corpus=tokenized_corpus,
        )

    def save(
        self,
        path: Path = DEFAULT_BM25_INDEX_PATH,
    ) -> None:
        """保存法条和分词结果。

        BM25Okapi 对象不直接 pickle，避免 Python 或依赖升级后
        无法加载。加载时使用保存的分词结果重新构造 BM25。
        """

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "schema_version": 1,
            "document_count": len(self.chunks),
            "tokenizer": "jieba-accurate-v1",
            "chunks": [
                # 保存时：ArticleChunk → model_dump → dict → 写入 bm25_index.json
                chunk.model_dump(mode="json")
                for chunk in self.chunks
            ],
            "tokenized_corpus": self.tokenized_corpus,
        }

        temporary_path = path.with_name(
            f".{path.name}.tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            file.write("\n")

        temporary_path.replace(path)

    @classmethod
    def load(
        cls,
        path: Path = DEFAULT_BM25_INDEX_PATH,
    ) -> "BM25Retriever":
        """从磁盘加载 BM25 索引。"""

        if not path.exists():
            raise FileNotFoundError(
                f"BM25 索引不存在：{path}\n"
                "请先执行 --build。"
            )

        with path.open(encoding="utf-8") as file:
            payload = json.load(file)

        if payload.get("schema_version") != 1:
            raise ValueError(
                "不支持的 BM25 索引版本"
            )

        chunks = [
            # 加载时：json.load 读出 dict → model_validate → ArticleChunk
            ArticleChunk.model_validate(item)
            for item in payload["chunks"]
        ]

        tokenized_corpus = payload[
            "tokenized_corpus"
        ]

        if payload.get("document_count") != len(chunks):
            raise ValueError(
                "BM25 索引中的 document_count 不一致"
            )

        return cls(
            chunks=chunks,
            tokenized_corpus=tokenized_corpus,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
        """检索相关法条，可按正式法律名称过滤。"""

        if not query.strip():
            raise ValueError("检索问题不能为空")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(
            query_tokens
        )

        candidate_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if (
                law_name is None
                or chunk.law_name == law_name
            )
        ]

        # 先按分数降序，再按 chunk_id 排序。
        # 第二排序条件用于保证相同分数时结果可重复。
        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: (
                -float(scores[index]),
                self.chunks[index].chunk_id,
            ),
        )

        results: list[RetrievalResult] = []

        for index in ranked_indices[:top_k]:
            chunk = self.chunks[index]
            score = float(scores[index])

            results.append(
                make_retrieval_result(
                    chunk,
                    score=score,
                    source="bm25",
                    component_scores={
                        "bm25": score,
                    },
                )
            )

        return results


def _print_results(
    results: list[RetrievalResult],
) -> None:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        preview = result.text.replace(
            "\n",
            " ",
        )[:160]

        print(
            f"{rank:>2}. "
            f"{result.law_name}"
            f"{result.article_no} "
            f"score={result.score:.6f}"
        )
        print(f"    chunk_id={result.chunk_id}")
        print(f"    {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建或测试 BM25 法条索引"
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="从 articles.jsonl 重建 BM25 索引",
    )
    parser.add_argument(
        "--articles",
        default=str(DEFAULT_ARTICLES_PATH),
        help="articles.jsonl 路径",
    )
    parser.add_argument(
        "--index",
        default=str(DEFAULT_BM25_INDEX_PATH),
        help="BM25 索引路径",
    )
    parser.add_argument(
        "--query",
        help="构建或加载索引后执行一次查询",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--law-name",
        help="使用正式法律名称过滤结果",
    )

    arguments = parser.parse_args()

    articles_path = Path(arguments.articles)
    index_path = Path(arguments.index)

    if arguments.build:
        retriever = BM25Retriever.build(
            articles_path
        )
        retriever.save(index_path)

        print(
            f"BM25 索引构建完成："
            f"{len(retriever.chunks)} 条法条"
        )
        print(f"索引位置：{index_path}")
    else:
        retriever = BM25Retriever.load(
            index_path
        )

        print(
            f"BM25 索引加载完成："
            f"{len(retriever.chunks)} 条法条"
        )

    if arguments.query:
        results = retriever.search(
            arguments.query,
            top_k=arguments.top_k,
            law_name=arguments.law_name,
        )
        _print_results(results)


if __name__ == "__main__":
    main()