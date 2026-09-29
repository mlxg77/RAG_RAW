from pathlib import Path

from rag_law.retrieval.bm25 import (
    BM25Retriever,
    tokenize,
)
from rag_law.retrieval.common import (
    build_search_text,
)
from rag_law.schemas import ArticleChunk

from langchain_core.embeddings import Embeddings

from rag_law.retrieval.dense import (
    DenseRetriever,
    chunk_to_document,
    distance_to_score,
)

from rag_law.retrieval.common import (
    build_search_text,
    make_retrieval_result,
)
from rag_law.retrieval.fusion import (
    HybridRetriever,
    reciprocal_rank_fusion,
)
from rag_law.schemas import (
    ArticleChunk,
    RetrievalResult,
)

def make_chunk(
    *,
    chunk_id: str,
    law_name: str,
    article_no: str,
    text: str,
) -> ArticleChunk:
    return ArticleChunk(
        chunk_id=chunk_id,
        law_id=f"law-{chunk_id}",
        law_name=law_name,
        article_no=article_no,
        text=text,
        source_file=f"data/raw/{law_name}.md",
        text_hash=f"hash-{chunk_id}",
    )


def build_test_retriever() -> BM25Retriever:
    chunks = [
        make_chunk(
            chunk_id="chunk-labor",
            law_name="中华人民共和国劳动合同法",
            article_no="第十条",
            text=(
                "建立劳动关系，应当订立"
                "书面劳动合同。"
            ),
        ),
        make_chunk(
            chunk_id="chunk-traffic",
            law_name="中华人民共和国道路交通安全法",
            article_no="第四十七条",
            text=(
                "机动车行经人行横道时，"
                "应当减速行驶。"
            ),
        ),
        make_chunk(
            chunk_id="chunk-consumer",
            law_name="中华人民共和国消费者权益保护法",
            article_no="第二十五条",
            text=(
                "经营者采用网络方式销售商品，"
                "消费者有权自收到商品之日起"
                "七日内退货。"
            ),
        ),
    ]

    tokenized_corpus = [
        tokenize(build_search_text(chunk))
        for chunk in chunks
    ]

    return BM25Retriever(
        chunks=chunks,
        tokenized_corpus=tokenized_corpus,
    )


def test_tokenize_returns_chinese_terms() -> None:
    tokens = tokenize(
        "公司应当签订书面劳动合同"
    )

    assert tokens
    assert all(token.strip() for token in tokens)
    assert " " not in tokens


def test_search_text_contains_metadata() -> None:
    chunk = make_chunk(
        chunk_id="chunk-1",
        law_name="中华人民共和国劳动合同法",
        article_no="第八十七条",
        text="违法解除劳动合同的，应当支付赔偿金。",
    )

    search_text = build_search_text(chunk)

    assert "中华人民共和国劳动合同法" in search_text
    assert "第八十七条" in search_text
    assert "赔偿金" in search_text


def test_bm25_returns_relevant_article_first() -> None:
    retriever = build_test_retriever()

    results = retriever.search(
        "公司没有和员工签书面劳动合同",
        top_k=3,
    )

    assert results[0].chunk_id == "chunk-labor"
    assert results[0].source == "bm25"
    assert results[0].score >= results[1].score


def test_bm25_supports_law_name_filter() -> None:
    retriever = build_test_retriever()

    results = retriever.search(
        "法律规定",
        top_k=10,
        law_name="中华人民共和国道路交通安全法",
    )

    assert len(results) == 1
    assert (
        results[0].law_name
        == "中华人民共和国道路交通安全法"
    )


def test_bm25_index_can_be_saved_and_loaded(
    tmp_path: Path,
) -> None:
    retriever = build_test_retriever()
    index_path = tmp_path / "bm25_index.json"

    retriever.save(index_path)
    loaded = BM25Retriever.load(index_path)

    original_results = retriever.search(
        "七日无理由退货",
        top_k=3,
    )
    loaded_results = loaded.search(
        "七日无理由退货",
        top_k=3,
    )

    assert [
        result.chunk_id
        for result in loaded_results
    ] == [
        result.chunk_id
        for result in original_results
    ]


def test_bm25_rejects_empty_query() -> None:
    retriever = build_test_retriever()

    try:
        retriever.search("   ")
    except ValueError as error:
        assert "不能为空" in str(error)
    else:
        raise AssertionError(
            "空问题应当抛出 ValueError"
        )
    
class FakeLegalEmbeddings(Embeddings):
    """只用于测试的确定性假向量模型。"""

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()

        labor = float(
            lowered.count("劳动")
            + lowered.count("合同")
            + lowered.count("公司")
        )
        traffic = float(
            lowered.count("机动车")
            + lowered.count("驾驶")
            + lowered.count("道路")
        )
        consumer = float(
            lowered.count("退货")
            + lowered.count("消费者")
            + lowered.count("商品")
        )

        # 最后一维用于避免生成全零向量。
        return [
            labor,
            traffic,
            consumer,
            1.0,
        ]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            self._embed(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self._embed(text)


def write_test_articles(
    path: Path,
) -> list[ArticleChunk]:
    chunks = [
        make_chunk(
            chunk_id="dense-labor",
            law_name="中华人民共和国劳动合同法",
            article_no="第十条",
            text=(
                "建立劳动关系，应当订立"
                "书面劳动合同。"
            ),
        ),
        make_chunk(
            chunk_id="dense-traffic",
            law_name="中华人民共和国道路交通安全法",
            article_no="第四十七条",
            text=(
                "机动车行经人行横道时，"
                "应当减速行驶。"
            ),
        ),
        make_chunk(
            chunk_id="dense-consumer",
            law_name="中华人民共和国消费者权益保护法",
            article_no="第二十五条",
            text=(
                "消费者有权自收到商品之日起"
                "七日内退货。"
            ),
        ),
    ]

    content = "\n".join(
        chunk.model_dump_json()
        for chunk in chunks
    )

    path.write_text(
        content + "\n",
        encoding="utf-8",
    )

    return chunks


def test_chunk_to_document_contains_chunk_id() -> None:
    chunk = make_chunk(
        chunk_id="document-test",
        law_name="中华人民共和国劳动合同法",
        article_no="第十条",
        text="应当订立书面劳动合同。",
    )

    document = chunk_to_document(chunk)

    assert (
        document.metadata["chunk_id"]
        == "document-test"
    )
    assert "中华人民共和国劳动合同法" in (
        document.page_content
    )
    assert "第十条" in document.page_content
    assert "书面劳动合同" in document.page_content


def test_distance_to_score_is_monotonic() -> None:
    close_score = distance_to_score(0.1)
    far_score = distance_to_score(0.8)

    assert close_score > far_score
    assert distance_to_score(0.0) == 1.0


def test_dense_search_and_persistence(
    tmp_path: Path,
) -> None:
    articles_path = tmp_path / "articles.jsonl"
    chroma_path = tmp_path / "chroma"

    write_test_articles(articles_path)

    embeddings = FakeLegalEmbeddings()

    retriever = DenseRetriever.build(
        articles_path=articles_path,
        persist_directory=chroma_path,
        collection_name="test_law_articles",
        embeddings=embeddings,
        batch_size=2,
    )

    results = retriever.search(
        "公司没有跟我签劳动合同",
        top_k=3,
    )

    assert results[0].chunk_id == "dense-labor"
    assert results[0].source == "dense"
    assert (
        "dense_distance"
        in results[0].component_scores
    )

    loaded = DenseRetriever.load(
        articles_path=articles_path,
        persist_directory=chroma_path,
        collection_name="test_law_articles",
        embeddings=embeddings,
    )

    loaded_results = loaded.search(
        "公司没有跟我签劳动合同",
        top_k=3,
    )

    assert [
        result.chunk_id
        for result in loaded_results
    ] == [
        result.chunk_id
        for result in results
    ]


def test_dense_supports_law_name_filter(
    tmp_path: Path,
) -> None:
    articles_path = tmp_path / "articles.jsonl"
    chroma_path = tmp_path / "chroma"

    write_test_articles(articles_path)

    retriever = DenseRetriever.build(
        articles_path=articles_path,
        persist_directory=chroma_path,
        collection_name="test_filter",
        embeddings=FakeLegalEmbeddings(),
        batch_size=2,
    )

    results = retriever.search(
        "法律有什么规定",
        top_k=10,
        law_name="中华人民共和国道路交通安全法",
    )

    assert len(results) == 1
    assert (
        results[0].law_name
        == "中华人民共和国道路交通安全法"
    )
    
def make_search_result(
    *,
    chunk_id: str,
    source: str,
    score: float,
) -> RetrievalResult:
    chunk = make_chunk(
        chunk_id=chunk_id,
        law_name="中华人民共和国测试法",
        article_no=f"第{chunk_id}条",
        text=f"这是 {chunk_id} 的法条正文。",
    )

    return make_retrieval_result(
        chunk,
        score=score,
        source=source,
        component_scores={
            source: score,
        },
    )


def test_rrf_rewards_results_from_both_routes() -> None:
    bm25_results = [
        make_search_result(
            chunk_id="a",
            source="bm25",
            score=10.0,
        ),
        make_search_result(
            chunk_id="shared",
            source="bm25",
            score=8.0,
        ),
    ]

    dense_results = [
        make_search_result(
            chunk_id="c",
            source="dense",
            score=0.9,
        ),
        make_search_result(
            chunk_id="shared",
            source="dense",
            score=0.8,
        ),
    ]

    results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        top_k=10,
        rrf_k=60,
    )

    assert results[0].chunk_id == "shared"
    assert results[0].source == "hybrid"

    components = results[0].component_scores

    assert components["bm25_rank"] == 2.0
    assert components["dense_rank"] == 2.0
    assert "bm25_rrf" in components
    assert "dense_rrf" in components
    assert components["rrf"] == results[0].score


def test_rrf_deduplicates_chunk_ids() -> None:
    duplicated = make_search_result(
        chunk_id="duplicate",
        source="bm25",
        score=5.0,
    )

    results = reciprocal_rank_fusion(
        [duplicated, duplicated],
        [],
        top_k=10,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "duplicate"

    expected_score = 1.0 / 61.0

    assert abs(
        results[0].score - expected_score
    ) < 1e-12


def test_rrf_uses_chunk_id_for_stable_ties() -> None:
    bm25_results = [
        make_search_result(
            chunk_id="b",
            source="bm25",
            score=10.0,
        )
    ]
    dense_results = [
        make_search_result(
            chunk_id="a",
            source="dense",
            score=0.9,
        )
    ]

    results = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        top_k=10,
    )

    assert [
        result.chunk_id
        for result in results
    ] == ["a", "b"]


def test_rrf_rejects_conflicting_chunk_content() -> None:
    first = make_search_result(
        chunk_id="same-id",
        source="bm25",
        score=10.0,
    )

    conflicting_chunk = make_chunk(
        chunk_id="same-id",
        law_name="中华人民共和国另一部法律",
        article_no="第一条",
        text="冲突的正文。",
    )

    second = make_retrieval_result(
        conflicting_chunk,
        score=0.9,
        source="dense",
        component_scores={
            "dense": 0.9,
        },
    )

    try:
        reciprocal_rank_fusion(
            [first],
            [second],
        )
    except ValueError as error:
        assert "不同内容" in str(error)
    else:
        raise AssertionError(
            "chunk_id 冲突时应抛出 ValueError"
        )


class StubRetriever:
    """用于测试 HybridRetriever 调用参数。"""

    def __init__(
        self,
        results: list[RetrievalResult],
    ) -> None:
        self.results = results
        self.calls: list[
            tuple[str, int, str | None]
        ] = []

    def search(
        self,
        query: str,
        *,
        top_k: int,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append(
            (query, top_k, law_name)
        )
        return self.results[:top_k]


def test_hybrid_passes_filter_to_both_routes() -> None:
    bm25_stub = StubRetriever(
        [
            make_search_result(
                chunk_id="shared",
                source="bm25",
                score=7.0,
            )
        ]
    )

    dense_stub = StubRetriever(
        [
            make_search_result(
                chunk_id="shared",
                source="dense",
                score=0.8,
            )
        ]
    )

    hybrid = HybridRetriever(
        bm25_retriever=bm25_stub,
        dense_retriever=dense_stub,
        candidate_top_k=20,
    )

    results = hybrid.search(
        "测试问题",
        top_k=10,
        law_name="中华人民共和国测试法",
    )

    assert bm25_stub.calls == [
        (
            "测试问题",
            20,
            "中华人民共和国测试法",
        )
    ]
    assert dense_stub.calls == [
        (
            "测试问题",
            20,
            "中华人民共和国测试法",
        )
    ]

    assert len(results) == 1
    assert results[0].source == "hybrid"
    assert (
        results[0].component_scores[
            "bm25_rank"
        ]
        == 1.0
    )
    assert (
        results[0].component_scores[
            "dense_rank"
        ]
        == 1.0
    )