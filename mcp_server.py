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

if __name__ == "__main__":
    mcp.run()
