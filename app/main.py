from __future__ import annotations

"""FastAPI service for the customer service assistant."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from rag.generation import DEFAULT_FALLBACK_ANSWER, generate_answer_from_retrieval
from rag.pipeline import retrieve_rag_context

BASE_DIR = Path(__file__).resolve().parent.parent
FRONT_DIR = BASE_DIR / "font"

app = FastAPI(title="Customer Service Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONT_DIR.exists():
    app.mount("/font", StaticFiles(directory=str(FRONT_DIR), html=True), name="font")


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户输入的问题")
    model_name: str | None = Field(default=None, description="可选的模型名称")
    extra_instructions: str | None = Field(default=None, description="额外回复约束")


class ChatSource(BaseModel):
    index: int
    content: str
    metadata: dict[str, Any]
    rerank_score: float | None = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    is_fallback: bool
    sources: list[ChatSource] = Field(default_factory=list)
    used_context: str = ""
    error: str | None = None


@lru_cache(maxsize=1)
def _warmup_env() -> None:
    load_dotenv()


@app.get("/")
def root() -> FileResponse:
    index_file = FRONT_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="前端页面不存在，请先在 font 文件夹中创建 index.html")
    return FileResponse(index_file)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _build_retrieval_result(query: str):
    rag_context = retrieve_rag_context(query)
    if rag_context.error:
        raise RuntimeError(rag_context.error)
    if rag_context.retrieval_result is not None:
        return rag_context.retrieval_result

    # query 为空或未检索到内容时，构造一个空的 RetrievalResult
    from rag.retrieval import RetrievalResult

    return RetrievalResult(query=query, documents=[])


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    _warmup_env()
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    try:
        retrieval_result = _build_retrieval_result(query)
        generation_result = generate_answer_from_retrieval(
            retrieval_result=retrieval_result,
            model_name=request.model_name or os.getenv("DEEPSEEK_MODEL", None) or "deepseek-v4-flash",
            extra_instructions=request.extra_instructions,
        )
        return ChatResponse(
            query=generation_result.query,
            answer=generation_result.answer or DEFAULT_FALLBACK_ANSWER,
            is_fallback=generation_result.is_fallback,
            used_context=generation_result.used_context,
            sources=[ChatSource.model_validate(source) for source in generation_result.sources],
        )
    except Exception as exc:
        return ChatResponse(
            query=query,
            answer=DEFAULT_FALLBACK_ANSWER,
            is_fallback=True,
            sources=[],
            used_context="",
            error=str(exc),
        )
