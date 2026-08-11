from fastapi.testclient import TestClient

from rag_agent.api.server import app, create_app


client = TestClient(app)


def _collect_paths(routes, prefix=""):
    """Recursively collect all route paths from FastAPI/Starlette routes."""
    paths = set()
    for r in routes:
        t = type(r).__name__
        if t in ("Route", "APIRoute") and hasattr(r, "path"):
            paths.add(prefix + r.path)
        elif t == "_IncludedRouter":
            # FastAPI includes routers as _IncludedRouter wrappers
            original = getattr(r, "original_router", None)
            if original is not None:
                included_prefix = r.include_context.prefix if hasattr(r, "include_context") else ""
                paths |= _collect_paths(original.routes, prefix + included_prefix)
    return paths


# ============================================================================
# Route registration
# ============================================================================

def test_all_routes_are_registered():
    route_paths = _collect_paths(app.router.routes)
    assert "/api/query" in route_paths
    assert "/api/ingest/document" in route_paths
    assert "/api/ingest/directory" in route_paths
    assert "/api/ingest/upload" in route_paths
    assert "/api/documents" in route_paths
    assert "/api/documents/{document_id}" in route_paths


# ============================================================================
# create_app factory
# ============================================================================

class TestCreateApp:
    def test_sets_title_and_version(self):
        app_instance = create_app()
        assert app_instance.title == "DocQuery API"
        assert app_instance.version == "0.1.0"

    def test_creates_unique_instances(self):
        app1 = create_app()
        app2 = create_app()
        assert app1 is not app2


# ============================================================================
# CORS
# ============================================================================

class TestCORS:
    def test_cors_headers_on_preflight(self):
        resp = client.options(
            "/api/documents",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS middleware should respond with allow-* headers
        assert resp.status_code in (200, 204, 405)
        assert "access-control-allow-origin" in resp.headers

    def test_cors_on_actual_request(self):
        resp = client.get("/api/documents", headers={"Origin": "http://localhost:5173"})
        assert "access-control-allow-origin" in resp.headers
        assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


# ============================================================================
# OpenAPI / docs
# ============================================================================

def test_openapi_schema_accessible():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert schema["info"]["title"] == "DocQuery API"
    assert "/api/query" in schema["paths"]
    assert "/api/documents" in schema["paths"]
