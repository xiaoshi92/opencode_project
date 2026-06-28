import json
from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP
from qdrant_controller import QdrantController

# Create an MCP server instance named "qdrant-hybrid-search"
mcp = FastMCP("qdrant-hybrid-search")

# Initialize our controller (FastEmbed and Qdrant client connection)
controller = QdrantController()

@mcp.tool()
def init_index() -> str:
    """
    Deletes the existing database collection and creates a fresh, empty one with dual-vector indexing.
    """
    return controller.init_index()

@mcp.tool()
def upsert_document(text: str, metadata: Optional[Dict[str, Any]] = None, chunk_strategy: str = "fixed") -> str:
    """
    Preprocesses, chunks, embeds, and indexes a text document into Qdrant using either fixed-size or semantic chunking.
    
    Args:
        text: The raw text document content to index.
        metadata: Optional dictionary of custom metadata fields to store alongside the chunks.
        chunk_strategy: The strategy used to segment text: 'fixed' or 'semantic'.
    """
    res = controller.upsert_document(text, metadata, chunk_strategy)
    return json.dumps(res, indent=2)

@mcp.tool()
def dense_search(query: str, limit: int = 5) -> str:
    """
    Performs conceptual/semantic similarity search using dense GTE vectors.
    
    Args:
        query: The search query.
        limit: The maximum number of matches to return.
    """
    res = controller.dense_search(query, limit)
    return json.dumps(res, indent=2)

@mcp.tool()
def sparse_search(query: str, limit: int = 5) -> str:
    """
    Performs strict lexical keyword search using sparse BM25 vectors.
    
    Args:
        query: The search query containing terms to match exactly.
        limit: The maximum number of matches to return.
    """
    res = controller.sparse_search(query, limit)
    return json.dumps(res, indent=2)

@mcp.tool()
def hybrid_search(query: str, limit: int = 5) -> str:
    """
    Performs unified hybrid search (dense semantic + sparse keyword) fused using native Reciprocal Rank Fusion (RRF). Highly recommended for general retrieval.
    
    Args:
        query: The search query.
        limit: The maximum number of matches to return.
    """
    res = controller.hybrid_search(query, limit)
    return json.dumps(res, indent=2)

import sys

class OpenWebUICompatMiddleware:
    def __init__(self, starlette_app):
        self.starlette_app = starlette_app
        # Retrieve the SseServerTransport instance from the Starlette mount
        self.transport = starlette_app.routes[1].app.__self__

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "POST" and (scope["path"] == "/sse" or scope["path"].startswith("/messages")):
            # 1. Intercept the first ASGI message to inspect the payload body
            first_message = await receive()
            body = first_message.get("body", b"")
            
            # 2. If the payload body is completely empty, it represents a client-side
            # dry-run verification check. We return a friendly 200 OK immediately and
            # prevent Pydantic from ever parsing an empty value and crashing!
            if not body or body.strip() == b"":
                sys.stderr.write("[CORS/OpenWebUI Handler] Intercepted empty POST connection-check. Replying with HTTP 200 OK.\n")
                sys.stderr.flush()
                
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"access-control-allow-origin", b"*"),
                    ]
                })
                await send({
                    "type": "http.response.body",
                    "body": b'{"status": "ok", "message": "Qdrant MCP Server Ready"}',
                    "more_body": False
                })
                return
                
            # 3. Rebuild the receive channel with a mock closure so Starlette can read the body normally
            async def mock_receive():
                return first_message
                
            # 4. If the body carries content, rewrite the path and inject the active session_id
            active_sessions = list(self.transport._read_stream_writers.keys())
            if active_sessions:
                session_id = active_sessions[-1]
                scope["path"] = "/messages/"
                scope["raw_path"] = b"/messages/"
                scope["query_string"] = f"session_id={session_id}".encode("utf-8")
                sys.stderr.write(f"[CORS/OpenWebUI-Compat] Rewrote POST -> /messages/ with session_id: {session_id}\n")
                sys.stderr.flush()
            else:
                sys.stderr.write("[CORS/OpenWebUI Handler] POST received but no active session found. Passing forward.\n")
                sys.stderr.flush()
                
            return await self.starlette_app(scope, mock_receive, send)
            
        return await self.starlette_app(scope, receive, send)

# Expose the wrapped app as the main ASGI application
from starlette.middleware.cors import CORSMiddleware

starlette_app = mcp.sse_app()
starlette_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app = OpenWebUICompatMiddleware(starlette_app)

if __name__ == "__main__":
    mcp.run()
