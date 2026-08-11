from __future__ import annotations

from rag_agent.config import settings
from rag_agent.embeddings.factory import get_embedding_provider
from rag_agent.generation.llm import LLMGenerator, RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
from rag_agent.retrieval.retriever import Retriever
from rag_agent.storage.doc_store import DocStore
from rag_agent.storage.vector_store import VectorStore


def query_document(
    question: str,
    top_k: int = 5,
    stream: bool = False,
    conversation_history: list[dict] | None = None,
):
    """Run the RAG query flow and return either a plain answer or a stream of events."""

    embedding_provider = get_embedding_provider(settings)
    vector_store = VectorStore(str(settings.chroma_persist_dir))
    retriever = Retriever(embedding_provider=embedding_provider, vector_store=vector_store)
    doc_store = DocStore(str(settings.doc_store_path))

    _model_map = {
        "claude": settings.claude_model,
        "openai": settings.openai_model,
        "ollama": settings.ollama_model,
    }

    llm = LLMGenerator(
        provider=settings.llm_provider,
        model=_model_map[settings.llm_provider],
        api_key=getattr(settings, f"{settings.llm_provider}_api_key", None),
        base_url=(settings.openai_base_url if settings.llm_provider == "openai" else (settings.ollama_base_url if settings.llm_provider == "ollama" else None)),
    )

    results = retriever.retrieve_with_threshold(
        question, top_k=top_k, threshold=settings.similarity_threshold
    )

    if not results:
        if stream:
            return _empty_stream()
        return ("文档中没有找到相关信息。", [])

    sources = []
    for item in results:
        doc = doc_store.get_document(item["doc_id"])
        sources.append(
            {
                "filename": doc.filename if doc else "unknown",
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "chunk_index": item.get("chunk_index", item["chunk_id"].rsplit("_", 1)[-1]),
            }
        )

    context = "\n\n".join(item["text"] for item in results)
    current_prompt = RAG_USER_PROMPT.format(context=context, question=question)
    messages = list(conversation_history or [])
    messages.append({"role": "user", "content": current_prompt})

    if stream:
        return _stream_answer(sources, llm, messages)

    answer = llm.generate_with_messages(messages=messages, system_prompt=RAG_SYSTEM_PROMPT)
    return (answer, sources)


def _empty_stream():
    yield {"type": "token", "data": "文档中没有找到相关信息。"}


def _stream_answer(sources, llm, messages):
    for src in sources:
        yield {"type": "citation", "data": src}

    for token in llm.generate_stream_with_messages(messages=messages, system_prompt=RAG_SYSTEM_PROMPT):
        yield {"type": "token", "data": token}