"""环境验证脚本：真实调用 embedding、Chroma、LLM 三条链路。

用法（项目根目录执行）：
    uv run python scripts/verify_env.py
"""

import math
import tempfile

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_law.config import PROJECT_ROOT, settings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度：点积 / 两个模长的乘积。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def build_embeddings() -> OpenAIEmbeddings:
    """按 .env 配置创建 embedding 客户端。

    check_embedding_ctx_length=False：第三方 OpenAI 兼容服务的必须设置。
    默认 True 时，langchain 会用 tiktoken 把文本转成 token id 数组再发送，
    这只适用于 OpenAI 官方 API；第三方服务按自己的词表解释这些 id，
    会得到语义错乱的向量（接口正常但相似度方向错误）。
    """
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        check_embedding_ctx_length=False,
    )


def check_embeddings(embeddings: OpenAIEmbeddings) -> None:
    """第 1 步：embedding 能调用，且语义相似度方向正确。"""
    sentences = [
        "建立劳动关系，应当订立书面劳动合同。",  # 0：相关句
        "公司必须与员工签订书面劳动合同。",      # 1：相关句
        "机动车行经人行横道时，应当减速行驶。",  # 2：无关句
    ]
    vectors = embeddings.embed_documents(sentences)
    dims = {len(v) for v in vectors}
    assert len(dims) == 1, f"向量维度不一致：{dims}"
    dim = dims.pop()

    sim_related = cosine_similarity(vectors[0], vectors[1])
    sim_unrelated = cosine_similarity(vectors[0], vectors[2])
    print(f"[1/3] Embedding 调用成功：3 条文本 → {dim} 维向量")
    print(f"      相关句相似度 {sim_related:.4f} / 无关句相似度 {sim_unrelated:.4f}")
    assert sim_related > sim_unrelated, "语义相似度方向异常：相关句应高于无关句"


def check_chroma(embeddings: OpenAIEmbeddings) -> None:
    """第 2 步：Chroma 可写入、可按语义召回。"""
    docs = [
        Document(
            page_content="用人单位自用工之日起即与劳动者建立劳动关系。",
            metadata={"law_name": "中华人民共和国劳动合同法", "article_no": "第七条"},
        ),
        Document(
            page_content="建立劳动关系，应当订立书面劳动合同。",
            metadata={"law_name": "中华人民共和国劳动合同法", "article_no": "第十条"},
        ),
        Document(
            page_content="机动车行经人行横道时，应当减速行驶。",
            metadata={"law_name": "中华人民共和国道路交通安全法", "article_no": "第四十七条"},
        ),
    ]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        store = Chroma(
            collection_name="verify_test",
            embedding_function=embeddings,
            persist_directory=tmp_dir,
        )
        store.add_documents(docs)
        results = store.similarity_search("公司不签书面劳动合同怎么办", k=2)
        assert results, "检索结果为空"

        top = results[0]
        hit = any("书面劳动合同" in doc.page_content for doc in results)
        print(f"[2/3] Chroma 写入 {len(docs)} 条、检索成功，Top1：")
        print(f"      {top.metadata['law_name']} {top.metadata['article_no']}")
        print(f"      {top.page_content}")
        assert hit, f"未召回预期法条，实际结果：{[d.page_content for d in results]}"


def check_llm() -> None:
    """第 3 步：大模型能调用并返回内容。"""
    llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0,
    )
    reply = llm.invoke("用一句话回答：建立劳动关系应当订立什么形式的合同？")
    print(f"[3/3] LLM 调用成功，模型回答：{reply.content}")
    assert reply.content, "模型返回内容为空"


def main() -> None:
    print("配置来源：", PROJECT_ROOT / ".env")
    print("LLM：", settings.llm_model, "@", settings.llm_base_url)
    print("Embedding：", settings.embedding_model, "@", settings.embedding_base_url)
    print("-" * 60)

    embeddings = build_embeddings()
    check_embeddings(embeddings)
    check_chroma(embeddings)
    check_llm()

    print("-" * 60)
    print("全部通过：embedding / Chroma / LLM 三条链路可用")


if __name__ == "__main__":
    main()