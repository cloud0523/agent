import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_agent.pipeline.query import query_document


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    stream: bool = False
    conversation_history: Optional[List[Dict[str, Any]]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


router = APIRouter()


@router.post("/query")
def query_endpoint(body: QueryRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    if body.stream:
        def event_generator():
            for item in query_document(
                body.question,
                top_k=body.top_k,
                stream=True,
                conversation_history=body.conversation_history,
            ):
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    answer, sources = query_document(
        body.question,
        top_k=body.top_k,
        stream=False,
        conversation_history=body.conversation_history,
    )
    return {"answer": answer, "sources": sources}
