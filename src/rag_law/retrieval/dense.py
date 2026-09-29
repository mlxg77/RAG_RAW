"""基于 LangChain Embedding 和 Chroma 的向量检索。"""

import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

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


DEFAULT_CHROMA_DIRECTORY = (
    PROJECT_ROOT
    / "indexes"
    / "chroma"
)

DEFAULT_COLLECTION_NAME = "law_articles"
DEFAULT_BATCH_SIZE = 32


def build_embeddings(
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> OpenAIEmbeddings:
    """根据项目配置创建 Embedding 客户端。

    把 config 导入放在函数内部，使单元测试注入假 Embedding
    时不需要真实 API Key。
    """

    from rag_law.config import settings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        check_embedding_ctx_length=False,
        chunk_size=batch_size,
    )


def chunk_to_document(
    chunk: ArticleChunk,
) -> Document:
    """把法条转换成写入 Chroma 的 LangChain Document。

    page_content 使用增强后的检索文本；
    原始法条仍以 articles.jsonl 中的数据为准。
    """

    return Document(
        page_content=build_search_text(chunk),
        metadata={
            "chunk_id": chunk.chunk_id,
            "law_id": chunk.law_id,
            "law_name": chunk.law_name,
            "article_no": chunk.article_no,
            "source_file": chunk.source_file,
        },
    )


def distance_to_score(distance: float) -> float:
    """把越小越好的距离转换成越大越好的展示分数。

    这不是概率，只是保持排序方向一致的单调转换。
    RRF 阶段只使用排名，不依赖这个分数的绝对大小。
    """

    safe_distance = max(float(distance), 0.0)
    return 1.0 / (1.0 + safe_distance)


class DenseRetriever:
    """Chroma 向量检索器。"""

    def __init__(
        self,
        *,
        chunks: list[ArticleChunk],
        vector_store: Chroma,
    ) -> None:
        if not chunks:
            raise ValueError("Dense 语料不能为空")

        self.chunks = chunks
        self.chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }
        self.vector_store = vector_store

        if len(self.chunks_by_id) != len(chunks):
            raise ValueError(
                "Dense 语料中包含重复 chunk_id"
            )

    @staticmethod
    def _create_vector_store(
        *,
        embeddings: Embeddings,
        persist_directory: Path,
        collection_name: str,
    ) -> Chroma:
        persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(
                persist_directory
            ),
            collection_metadata={
                "hnsw:space": "cosine",
            },
        )

    @classmethod
    def build(
        cls,
        *,
        articles_path: Path = DEFAULT_ARTICLES_PATH,
        persist_directory: Path = (
            DEFAULT_CHROMA_DIRECTORY
        ),
        collection_name: str = (
            DEFAULT_COLLECTION_NAME
        ),
        embeddings: Embeddings | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> "DenseRetriever":
        """从 articles.jsonl 重建 Chroma collection。

        build 表示完整重建。如果 collection 中已有数据，
        会先删除旧 collection，防止残留已经失效的法条。
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size 必须大于 0"
            )

        chunks = load_articles(articles_path)

        actual_embeddings = (
            embeddings
            if embeddings is not None
            else build_embeddings(
                batch_size=batch_size
            )
        )

        vector_store = cls._create_vector_store(
            embeddings=actual_embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        existing = vector_store.get(limit=1)

        if existing.get("ids"):
            vector_store.delete_collection()

            vector_store = cls._create_vector_store(
                embeddings=actual_embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )

        documents = [
            chunk_to_document(chunk)
            for chunk in chunks
        ]

        for start in range(
            0,
            len(documents),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(documents),
            )

            document_batch = documents[start:end]
            id_batch = [
                chunk.chunk_id
                for chunk in chunks[start:end]
            ]

            vector_store.add_documents(
                documents=document_batch,
                ids=id_batch,
            )

            print(
                f"Embedding 写入进度："
                f"{end}/{len(documents)}"
            )

        indexed_ids = vector_store.get().get(
            "ids",
            [],
        )

        if len(indexed_ids) != len(chunks):
            raise RuntimeError(
                "Chroma 写入数量不正确："
                f"期望 {len(chunks)}，"
                f"实际 {len(indexed_ids)}"
            )

        return cls(
            chunks=chunks,
            vector_store=vector_store,
        )

    @classmethod
    def load(
        cls,
        *,
        articles_path: Path = DEFAULT_ARTICLES_PATH,
        persist_directory: Path = (
            DEFAULT_CHROMA_DIRECTORY
        ),
        collection_name: str = (
            DEFAULT_COLLECTION_NAME
        ),
        embeddings: Embeddings | None = None,
        validate_ids: bool = True,
    ) -> "DenseRetriever":
        """加载已经构建的 Chroma collection。"""

        chunks = load_articles(articles_path)

        actual_embeddings = (
            embeddings
            if embeddings is not None
            else build_embeddings()
        )

        vector_store = cls._create_vector_store(
            embeddings=actual_embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        indexed_ids = vector_store.get().get(
            "ids",
            [],
        )

        if not indexed_ids:
            raise FileNotFoundError(
                "Chroma collection 为空，"
                "请先执行 --build。"
            )

        if validate_ids:
            expected_ids = {
                chunk.chunk_id
                for chunk in chunks
            }
            actual_ids = set(indexed_ids)

            if expected_ids != actual_ids:
                missing_count = len(
                    expected_ids - actual_ids
                )
                stale_count = len(
                    actual_ids - expected_ids
                )

                raise ValueError(
                    "Chroma 索引与 articles.jsonl "
                    "不一致，请重建索引。"
                    f"缺少 {missing_count} 条，"
                    f"多出 {stale_count} 条。"
                )

        return cls(
            chunks=chunks,
            vector_store=vector_store,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
        """执行向量相似度检索。"""

        if not query.strip():
            raise ValueError("检索问题不能为空")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        chroma_filter = (
            {"law_name": law_name}
            if law_name is not None
            else None
        )

        matches = (
            self.vector_store
            .similarity_search_with_score(
                query=query,
                k=top_k,
                filter=chroma_filter,
            )
        )

        intermediate: list[
            tuple[ArticleChunk, float, float]
        ] = []

        for document, raw_distance in matches:
            chunk_id = document.metadata.get(
                "chunk_id"
            )

            if not isinstance(chunk_id, str):
                raise ValueError(
                    "Chroma 文档缺少有效 chunk_id"
                )

            chunk = self.chunks_by_id.get(
                chunk_id
            )

            if chunk is None:
                raise ValueError(
                    "Chroma 返回了未知 chunk_id："
                    f"{chunk_id}"
                )

            distance = float(raw_distance)
            score = distance_to_score(distance)

            intermediate.append(
                (chunk, distance, score)
            )

        # Chroma 已经按距离排序；这里再增加 chunk_id
        # 作为相同距离时的稳定排序条件。
        intermediate.sort(
            key=lambda item: (
                item[1],
                item[0].chunk_id,
            )
        )

        results: list[RetrievalResult] = []

        for chunk, distance, score in intermediate:
            results.append(
                make_retrieval_result(
                    chunk,
                    score=score,
                    source="dense",
                    component_scores={
                        "dense": score,
                        "dense_distance": distance,
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

        distance = result.component_scores.get(
            "dense_distance",
            0.0,
        )

        print(
            f"{rank:>2}. "
            f"{result.law_name}"
            f"{result.article_no} "
            f"score={result.score:.6f} "
            f"distance={distance:.6f}"
        )
        print(f"    chunk_id={result.chunk_id}")
        print(f"    {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建或测试 Dense 法条索引"
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="重建 Chroma 向量索引",
    )
    parser.add_argument(
        "--articles",
        default=str(DEFAULT_ARTICLES_PATH),
        help="articles.jsonl 路径",
    )
    parser.add_argument(
        "--persist-directory",
        default=str(DEFAULT_CHROMA_DIRECTORY),
        help="Chroma 持久化目录",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help="Chroma collection 名称",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
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
    persist_directory = Path(
        arguments.persist_directory
    )

    if arguments.build:
        retriever = DenseRetriever.build(
            articles_path=articles_path,
            persist_directory=persist_directory,
            collection_name=arguments.collection,
            batch_size=arguments.batch_size,
        )

        print(
            "Dense 索引构建完成："
            f"{len(retriever.chunks)} 条法条"
        )
        print(
            "Collection："
            f"{arguments.collection}"
        )
        print(
            "持久化目录："
            f"{persist_directory}"
        )
    else:
        retriever = DenseRetriever.load(
            articles_path=articles_path,
            persist_directory=persist_directory,
            collection_name=arguments.collection,
        )

        print(
            "Dense 索引加载完成："
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