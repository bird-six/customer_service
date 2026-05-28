from __future__ import annotations

"""RAG 重排（rerank）Demo

这个示例演示：
1. 准备一组参考文档
2. 给定一个用户问题
3. 先做简单召回（这里直接把候选文档全部交给重排器）
4. 使用 LangChain 封装的交叉编码器（Cross Encoder）进行重排

说明：
- 这里使用 `langchain_community.cross_encoders.HuggingFaceCrossEncoder`
- 再通过 `langchain_classic.retrievers.document_compressors.CrossEncoderReranker`
  完成文档重排
- 默认模型为 `BAAI/bge-reranker-base`
- 首次运行会下载模型，若本地环境没有 `sentence-transformers`，请先安装
  `pip install sentence-transformers`
"""

from typing import List

from langchain_core.documents import Document
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)


def get_local_device() -> str:
    """优先使用本地 GPU，没有则回退到 CPU。"""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_documents() -> List[Document]:
    """构造演示用参考文档。"""
    return [
        Document(
            page_content=(
                "退款规则：订单签收后 7 天内、商品不影响二次销售的情况下，"
                "支持申请退货退款。若商品存在质量问题，可优先发起售后工单。"
            ),
            metadata={"doc_id": "doc_1", "title": "退款政策"},
        ),
        Document(
            page_content=(
                "物流查询：客户可在订单详情页查看物流单号，也可以通过客服系统"
                "输入手机号+订单号快速查询包裹当前位置。"
            ),
            metadata={"doc_id": "doc_2", "title": "物流查询"},
        ),
        Document(
            page_content=(
                "会员权益：银卡会员享受生日券，金卡会员享受 95 折，铂金会员"
                "可以优先排队并获得专属客服通道。"
            ),
            metadata={"doc_id": "doc_3", "title": "会员权益"},
        ),
        Document(
            page_content=(
                "支付失败排查：当用户支付失败时，先检查银行卡余额、支付限额、"
                "网络状态和是否触发风控拦截，再尝试重新支付。"
            ),
            metadata={"doc_id": "doc_4", "title": "支付失败排查"},
        ),
        Document(
            page_content=(
                "售后时效：工单提交后，普通问题通常在 24 小时内响应，复杂问题"
                "会在 48 小时内给出处理方案。"
            ),
            metadata={"doc_id": "doc_5", "title": "售后时效"},
        ),
    ]


def print_docs(title: str, docs: List[Document]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        print(f"{idx}. {meta.get('title', 'Untitled')} | {meta.get('doc_id', '-')}")
        print(f"   {doc.page_content}")


def main() -> None:
    # 用户问题可以根据业务场景自定义
    query = "如果客户说签收后想退款，而且商品没有明显使用痕迹，应该参考哪条规则？"

    documents = build_documents()

    # 先展示候选文档，模拟召回阶段拿到的结果
    print(f"用户问题：{query}")
    print_docs("召回阶段候选文档", documents)

    # LangChain 封装的交叉编码器
    # 使用本地算力：优先 GPU，若无 GPU 则使用 CPU
    device = get_local_device()
    print(f"交叉编码器运行设备：{device}")

    # 可按需修改 model_name，例如使用其他 reranker 模型
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base",
        model_kwargs={"device": device},
    )

    reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
    reranked_docs = reranker.compress_documents(documents=documents, query=query)

    print_docs("交叉编码器重排后的 Top 3 文档", list(reranked_docs))

    print("\n结论")
    print("----")
    if reranked_docs:
        best_doc = reranked_docs[0]
        print(
            "最相关的文档是："
            f"{best_doc.metadata.get('title', 'Untitled')}（{best_doc.metadata.get('doc_id', '-')})"
        )
        print(f"原因：{best_doc.page_content}")


if __name__ == "__main__":
    main()
