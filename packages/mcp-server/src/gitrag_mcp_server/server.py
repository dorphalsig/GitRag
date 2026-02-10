"""FastMCP server exposing secure code search via Scalekit auth."""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.auth import TokenVerifier
from fastmcp.server.auth.providers.scalekit import ScalekitProvider


class _RetrieverProtocol:
    def retrieve(self, query: str, *, top_k: int = 10) -> list[Any]: ...


def build_scalekit_provider(
    *,
    token_verifier: TokenVerifier | None = None,
    base_url: str | None = None,
) -> ScalekitProvider:
    """Construct Scalekit auth provider from environment variables."""
    environment_url = os.environ.get("SCALEKIT_ENVIRONMENT_URL")
    client_id = os.environ.get("SCALEKIT_CLIENT_ID")
    resource_id = os.environ.get("SCALEKIT_RESOURCE_ID")

    missing = [
        name
        for name, value in {
            "SCALEKIT_ENVIRONMENT_URL": environment_url,
            "SCALEKIT_CLIENT_ID": client_id,
            "SCALEKIT_RESOURCE_ID": resource_id,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required Scalekit environment variables: {', '.join(missing)}")

    resolved_base_url = base_url or os.environ.get("MCP_BASE_URL") or "http://127.0.0.1:8000/mcp"

    return ScalekitProvider(
        environment_url=environment_url,
        client_id=client_id,
        resource_id=resource_id,
        base_url=resolved_base_url,
        token_verifier=token_verifier,
    )


def create_mcp_server(
    *,
    retriever: _RetrieverProtocol,
    token_verifier: TokenVerifier | None = None,
    base_url: str | None = None,
) -> FastMCP:
    """Create an authenticated MCP server with `search_code` tool."""
    auth_provider = build_scalekit_provider(token_verifier=token_verifier, base_url=base_url)
    mcp = FastMCP(name="GitRag MCP Server", auth=auth_provider)

    @mcp.tool(name="search_code")
    def search_code(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search indexed code and return top snippets."""
        chunks = retriever.retrieve(query, top_k=top_k)
        output: list[dict[str, Any]] = []
        for chunk in chunks:
            payload = asdict(chunk)
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, (bytes, bytearray, memoryview)):
                payload["embeddings"] = None
            output.append(payload)
        return output

    return mcp
