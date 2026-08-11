"""RAG Agent CLI — Document ingestion, query, and server management."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from rag_agent.config import settings
from rag_agent.utils.logging import setup_logging

app = typer.Typer(
    name="rag-cli",
    help="RAG Agent — Document Q&A with Retrieval-Augmented Generation",
    add_completion=False,
)

console = Console()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    file: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to the document file to ingest"
    ),
    chunk_size: Optional[int] = typer.Option(
        None, "--chunk-size", "-c", help="Override chunk size"
    ),
    chunk_overlap: Optional[int] = typer.Option(
        None, "--chunk-overlap", "-o", help="Override chunk overlap"
    ),
) -> None:
    """Ingest a document into the RAG knowledge base.

    Supports PDF, DOCX, TXT, and Markdown files.
    """
    setup_logging()
    from loguru import logger

    try:
        from rag_agent.pipeline.ingestion import ingest_document

        c_size = chunk_size or settings.chunk_size
        c_overlap = chunk_overlap or settings.chunk_overlap

        console.print(f"[bold]Ingesting:[/] {file.name}")
        doc = ingest_document(file, chunk_size=c_size, chunk_overlap=c_overlap)

        table = Table(title="Document Ingested ✓")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("ID", doc.id)
        table.add_row("Filename", doc.filename)
        table.add_row("Type", doc.file_type)
        table.add_row("Chunks", str(doc.num_chunks))
        table.add_row("Size", f"{doc.size_bytes:,} bytes")
        console.print(table)

    except Exception as e:
        logger.error("Ingestion failed: {}", e)
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Ingest Directory
# ---------------------------------------------------------------------------

@app.command()
def ingest_dir(
    directory: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Directory containing documents"
    ),
    chunk_size: Optional[int] = typer.Option(
        None, "--chunk-size", "-c", help="Override chunk size"
    ),
    chunk_overlap: Optional[int] = typer.Option(
        None, "--chunk-overlap", "-o", help="Override chunk overlap"
    ),
) -> None:
    """Batch ingest all supported documents from a directory."""
    setup_logging()
    from loguru import logger

    try:
        from rag_agent.pipeline.ingestion import ingest_directory

        c_size = chunk_size or settings.chunk_size
        c_overlap = chunk_overlap or settings.chunk_overlap

        console.print(f"[bold]Scanning:[/] {directory}")
        results = ingest_directory(directory, chunk_size=c_size, chunk_overlap=c_overlap)

        table = Table(title=f"Ingestion Complete — {len(results)} documents")
        table.add_column("Filename", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Chunks")
        for doc in results:
            table.add_row(doc.filename, doc.status, str(doc.num_chunks))
        console.print(table)

    except Exception as e:
        logger.error("Batch ingestion failed: {}", e)
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask about your documents"),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", "-k", help="Number of chunks to retrieve"
    ),
    no_stream: bool = typer.Option(
        False, "--no-stream", help="Disable streaming output"
    ),
    no_citations: bool = typer.Option(
        False, "--no-citations", help="Hide source citations"
    ),
) -> None:
    """Ask a question about your ingested documents."""
    setup_logging()
    from loguru import logger

    try:
        from rag_agent.pipeline.query import query_document

        k = top_k or settings.top_k
        stream = not no_stream

        console.print(f"[bold]Question:[/] {question}")
        console.print("[dim]Searching documents...[/]")
        console.print()

        response = query_document(
            question,
            top_k=k,
            stream=stream,
        )

        if stream:
            console.print("[bold]Answer:[/] ", end="")
            full_answer = ""
            sources = []
            for chunk in response:
                if isinstance(chunk, dict) and chunk.get("type") == "token":
                    console.print(chunk["data"], end="")
                    full_answer += chunk["data"]
                elif isinstance(chunk, dict) and chunk.get("type") == "citation":
                    sources.append(chunk["data"])
            console.print()
        else:
            answer_text, sources = response
            console.print(f"[bold]Answer:[/] {answer_text}")

        if sources and not no_citations:
            console.print()
            console.print("[bold]Sources:[/]")
            for i, src in enumerate(sources, 1):
                console.print(
                    f"  [{i}] [cyan]{src.get('filename', 'unknown')}[/cyan] "
                    f"— Chunk {src.get('chunk_index', '?')}"
                )

    except Exception as e:
        logger.error("Query failed: {}", e)
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Chat (interactive)
# ---------------------------------------------------------------------------

@app.command()
def chat() -> None:
    """Start an interactive chat session with your documents."""
    setup_logging()
    from loguru import logger

    try:
        from rag_agent.pipeline.query import query_document
        from rag_agent.conversation.memory import ConversationMemory

        memory = ConversationMemory(
            max_turns=settings.max_history_turns,
            max_context_tokens=settings.max_context_tokens,
        )

        console.print("[bold]Interactive Chat[/] (type 'exit' to quit, 'clear' to reset)")
        console.print()

        while True:
            question = console.input("[bold green]You:[/] ").strip()

            if not question:
                continue
            if question.lower() == "exit":
                console.print("[dim]Goodbye![/]")
                break
            if question.lower() == "clear":
                memory.clear()
                console.print("[dim]History cleared.[/]")
                continue

            console.print("[bold blue]Assistant:[/] ", end="")
            full_answer = ""
            sources = []

            response = query_document(
                question,
                top_k=settings.top_k,
                stream=True,
                conversation_history=memory.get_history(),
            )

            for chunk in response:
                if isinstance(chunk, dict) and chunk.get("type") == "token":
                    console.print(chunk["data"], end="")
                    full_answer += chunk["data"]
                elif isinstance(chunk, dict) and chunk.get("type") == "citation":
                    sources.append(chunk["data"])

            console.print()
            memory.add_user_message(question)
            memory.add_assistant_message(full_answer, sources)

    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/]")
    except Exception as e:
        logger.error("Chat error: {}", e)
        console.print(f"\n[red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", "-h", help="Bind address"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Start the FastAPI web server and serve the frontend."""
    setup_logging()
    from loguru import logger

    try:
        import uvicorn

        bind_host = host or settings.host
        bind_port = port or settings.port

        logger.info("Starting server at http://{}:{}", bind_host, bind_port)
        console.print(f"[bold]Server:[/] http://{bind_host}:{bind_port}")
        console.print(f"[bold]API Docs:[/] http://{bind_host}:{bind_port}/docs")
        console.print()

        uvicorn.run(
            "rag_agent.api.server:app",
            host=bind_host,
            port=bind_port,
            reload=reload,
            log_level=settings.log_level.lower(),
        )

    except Exception as e:
        logger.error("Server failed to start: {}", e)
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@app.command()
def list() -> None:
    """List all indexed documents."""
    setup_logging()
    from loguru import logger

    try:
        from rag_agent.storage.doc_store import DocStore

        store = DocStore(str(settings.doc_store_path))
        docs = store.list_documents()

        if not docs:
            console.print("[dim]No documents indexed yet.[/]")
            return

        table = Table(title=f"Indexed Documents ({len(docs)})")
        table.add_column("ID", style="dim")
        table.add_column("Filename", style="cyan")
        table.add_column("Type")
        table.add_column("Chunks")
        table.add_column("Size")
        table.add_column("Status", style="green")
        table.add_column("Uploaded")

        for doc in docs:
            table.add_row(
                doc.id[:8],
                doc.filename,
                doc.file_type,
                str(doc.num_chunks),
                f"{doc.size_bytes:,}",
                doc.status,
                doc.uploaded_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

    except Exception as e:
        logger.error("List failed: {}", e)
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@app.command()
def delete(
    doc_id: str = typer.Argument(..., help="Document ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a document and its vectors from the knowledge base."""
    setup_logging()
    from loguru import logger

    try:
        from rag_agent.storage.doc_store import DocStore
        from rag_agent.storage.vector_store import VectorStore

        store = DocStore(str(settings.doc_store_path))
        doc = store.get_document(doc_id)

        if doc is None:
            console.print(f"[red]Document not found:[/] {doc_id}")
            raise typer.Exit(code=1)

        if not force:
            confirmed = typer.confirm(
                f"Delete '{doc.filename}' ({doc.num_chunks} chunks)?"
            )
            if not confirmed:
                console.print("[dim]Cancelled.[/]")
                return

        # Delete from vector store
        vs = VectorStore(str(settings.chroma_persist_dir))
        vs.delete_document(doc_id)

        # Delete from metadata store
        store.delete_document(doc_id)

        console.print(f"[green]Deleted:[/] {doc.filename}")

    except typer.Exit:
        raise
    except Exception as e:
        logger.error("Delete failed: {}", e)
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.command()
def status() -> None:
    """Show current configuration and system status."""
    setup_logging()

    table = Table(title="RAG Agent Status")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Embedding Provider", settings.embedding_provider)
    table.add_row("Embedding Model", settings.embedding_model)
    table.add_row("LLM Provider", settings.llm_provider)
    table.add_row("CLI Model", settings.claude_model)

    table.add_row("Chunk Size", str(settings.chunk_size))
    table.add_row("Chunk Overlap", str(settings.chunk_overlap))
    table.add_row("Chunk Strategy", settings.chunking_strategy)

    table.add_row("Top-K", str(settings.top_k))
    table.add_row("Similarity Threshold", str(settings.similarity_threshold))
    table.add_row("Reranker", "enabled" if settings.use_reranker else "disabled")

    table.add_row("Data Directory", str(settings.data_dir.absolute()))
    table.add_row("ChromaDB Directory", str(settings.chroma_persist_dir.absolute()))

    table.add_row("Server", f"{settings.host}:{settings.port}")
    table.add_row("Max History Turns", str(settings.max_history_turns))
    table.add_row("Log Level", settings.log_level)

    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
