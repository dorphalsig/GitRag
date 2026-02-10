from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass

import pytest
import uvicorn
from fastmcp import Client
from fastmcp.server.auth import AccessToken, TokenVerifier

from gitrag_mcp_server.server import build_scalekit_provider, create_mcp_server


@dataclass
class FakeChunk:
    chunk: str
    repo: str
    path: str
    language: str
    start_rc: tuple[int, int]
    end_rc: tuple[int, int]
    start_bytes: int
    end_bytes: int
    signature: str = ""
    embeddings: bytes | None = None
    metadata: dict | None = None


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None, str | None]] = []

    def retrieve(self, query: str, *, top_k: int = 10, repo: str | None = None, branch: str | None = None):
        self.calls.append((query, top_k, repo, branch))
        return [
            FakeChunk(
                chunk=f"match for {query}",
                repo="repo",
                path="src/main.py",
                language="python",
                start_rc=(1, 0),
                end_rc=(1, 12),
                start_bytes=0,
                end_bytes=12,
                embeddings=b"abc",
                metadata={"score": 1.0},
            )
        ]


class StubTokenVerifier(TokenVerifier):
    async def verify_token(self, token: str):
        if token == "valid-token":
            return AccessToken(token=token, client_id="test-client", scopes=["search:code"])
        return None


@pytest.fixture
def scalekit_env(monkeypatch):
    monkeypatch.setenv("SCALEKIT_ENVIRONMENT_URL", "https://env.scalekit.test")
    monkeypatch.setenv("SCALEKIT_CLIENT_ID", "client-123")
    monkeypatch.setenv("SCALEKIT_RESOURCE_ID", "resource-456")


def test_build_scalekit_provider_from_env(scalekit_env):
    provider = build_scalekit_provider(token_verifier=StubTokenVerifier(), base_url="http://127.0.0.1:8766/mcp")
    assert provider.environment_url == "https://env.scalekit.test"
    assert provider.resource_id == "resource-456"


def test_build_scalekit_provider_requires_env(monkeypatch):
    monkeypatch.delenv("SCALEKIT_ENVIRONMENT_URL", raising=False)
    monkeypatch.delenv("SCALEKIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("SCALEKIT_RESOURCE_ID", raising=False)
    with pytest.raises(RuntimeError, match="Missing required Scalekit environment variables"):
        build_scalekit_provider(base_url="http://127.0.0.1:8766/mcp")


def _run_server(server: uvicorn.Server) -> None:
    asyncio.run(server.serve())


def test_search_code_tool_with_authenticated_request(scalekit_env):
    retriever = FakeRetriever()
    mcp = create_mcp_server(
        retriever=retriever,
        token_verifier=StubTokenVerifier(),
        base_url="http://127.0.0.1:8766/mcp",
    )
    app = mcp.http_app(path="/mcp", transport="streamable-http")

    config = uvicorn.Config(app, host="127.0.0.1", port=8766, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=_run_server, args=(server,), daemon=True)
    thread.start()
    time.sleep(0.8)

    async def _call_tool():
        client = Client("http://127.0.0.1:8766/mcp", auth="valid-token")
        async with client:
            return await client.call_tool("search_code", {"query": "needle", "top_k": 3})

    result = asyncio.run(_call_tool())

    server.should_exit = True
    thread.join(timeout=2)

    assert result.is_error is False
    assert isinstance(result.structured_content, dict)
    chunks = result.structured_content["result"]
    assert chunks[0]["path"] == "src/main.py"
    assert chunks[0]["embeddings"] is None
    assert retriever.calls == [("needle", 3, None, None)]


def test_search_code_tool_rejects_invalid_token(scalekit_env):
    retriever = FakeRetriever()
    mcp = create_mcp_server(
        retriever=retriever,
        token_verifier=StubTokenVerifier(),
        base_url="http://127.0.0.1:8767/mcp",
    )
    app = mcp.http_app(path="/mcp", transport="streamable-http")

    config = uvicorn.Config(app, host="127.0.0.1", port=8767, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=_run_server, args=(server,), daemon=True)
    thread.start()
    time.sleep(0.8)

    async def _call_tool_unauthorized():
        client = Client("http://127.0.0.1:8767/mcp", auth="bad-token")
        async with client:
            await client.call_tool("search_code", {"query": "needle"})

    with pytest.raises(Exception):
        asyncio.run(_call_tool_unauthorized())

    server.should_exit = True
    thread.join(timeout=2)
