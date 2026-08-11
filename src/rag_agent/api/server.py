from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_agent.api.routes.documents import router as documents_router
from rag_agent.api.routes.ingest import router as ingest_router
from rag_agent.api.routes.query import router as query_router


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Agent API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(query_router, prefix="/api")
    app.include_router(ingest_router, prefix="/api")
    app.include_router(documents_router, prefix="/api")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    from rag_agent.config import settings
    uvicorn.run("rag_agent.api.server:app", host=settings.host, port=settings.port, reload=True)
