import sys
import json
import traceback
from typing import Dict, Any, List
from qdrant_controller import QdrantController

# IMPORTANT: All general printing/logging MUST go to sys.stderr. 
# sys.stdout is reserved strictly for JSON-RPC message exchange.
def log(message: str):
    sys.stderr.write(f"[MCP Server] {message}\n")
    sys.stderr.flush()

class MCPServer:
    def __init__(self):
        log("Initializing Qdrant Hybrid Search MCP Server...")
        self.controller = QdrantController()
        log("Qdrant controller successfully initialized.")

    def get_tools_list(self) -> List[Dict[str, Any]]:
        """
        Returns the official list of available MCP tools and their JSON schemas.
        """
        return [
            {
                "name": "init_index",
                "description": "Deletes the existing database collection and creates a fresh, empty one with dual-vector indexing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "upsert_document",
                "description": "Preprocesses, chunks, embeds, and indexes a text document into Qdrant using either fixed-size or semantic chunking.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The raw text document content to index."
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional dictionary of custom metadata fields to store alongside the chunks.",
                            "additionalProperties": True
                        },
                        "chunk_strategy": {
                            "type": "string",
                            "enum": ["fixed", "semantic"],
                            "default": "fixed",
                            "description": "The strategy used to segment text: 'fixed' (standard paragraphs/sentences) or 'semantic' (topic-based)."
                        }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "dense_search",
                "description": "Performs conceptual/semantic similarity search using dense GTE-Qwen2-1.5B-instruct vectors.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query."
                        },
                        "limit": {
                            "type": "integer",
                            "default": 5,
                            "description": "The maximum number of matches to return."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "sparse_search",
                "description": "Performs strict lexical keyword search using sparse BM25 vectors.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query containing terms to match exactly."
                        },
                        "limit": {
                            "type": "integer",
                            "default": 5,
                            "description": "The maximum number of matches to return."
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "hybrid_search",
                "description": "Performs unified hybrid search (dense semantic + sparse keyword) fused using native Reciprocal Rank Fusion (RRF). Highly recommended for general retrieval.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query."
                        },
                        "limit": {
                            "type": "integer",
                            "default": 5,
                            "description": "The maximum number of matches to return."
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Routes the tool execution request to the Qdrant controller.
        """
        log(f"Executing tool: {name} with args: {arguments}")
        
        if name == "init_index":
            msg = self.controller.init_index()
            return [{"type": "text", "text": msg}]
            
        elif name == "upsert_document":
            text = arguments.get("text", "")
            metadata = arguments.get("metadata", None)
            chunk_strategy = arguments.get("chunk_strategy", "fixed")
            
            res = self.controller.upsert_document(text, metadata, chunk_strategy)
            return [{"type": "text", "text": json.dumps(res, indent=2)}]
            
        elif name == "dense_search":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 5)
            
            res = self.controller.dense_search(query, limit)
            return [{"type": "text", "text": json.dumps(res, indent=2)}]
            
        elif name == "sparse_search":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 5)
            
            res = self.controller.sparse_search(query, limit)
            return [{"type": "text", "text": json.dumps(res, indent=2)}]
            
        elif name == "hybrid_search":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 5)
            
            res = self.controller.hybrid_search(query, limit)
            return [{"type": "text", "text": json.dumps(res, indent=2)}]
            
        else:
            raise ValueError(f"Tool not found: {name}")

    def handle_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes standard MCP JSON-RPC 2.0 requests.
        """
        method = req.get("method")
        req_id = req.get("id")
        
        # Build base response skeleton
        response = {
            "jsonrpc": "2.0",
            "id": req_id
        }
        
        try:
            if method == "initialize":
                response["result"] = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "qdrant-hybrid-search",
                        "version": "1.0.0"
                    }
                }
            elif method == "tools/list":
                response["result"] = {
                    "tools": self.get_tools_list()
                }
            elif method == "tools/call":
                params = req.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                content = self.execute_tool(tool_name, arguments)
                response["result"] = {
                    "content": content
                }
            else:
                response["error"] = {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
        except Exception as e:
            log(f"Error handling request: {traceback.format_exc()}")
            response["error"] = {
                "code": -32603,
                "message": str(e),
                "data": traceback.format_exc()
            }
            
        return response

    def run(self):
        """
        Starts the stdio JSON-RPC service loop.
        """
        log("Server stdio loop started. Waiting for host requests...")
        try:
            for line in sys.stdin:
                if not line.strip():
                    continue
                
                try:
                    request = json.loads(line)
                    
                    # Handle JSON-RPC notifications (messages without an ID)
                    if "id" not in request:
                        # e.g., notifications/initialized
                        log(f"Notification received: {request.get('method')}")
                        continue
                        
                    response = self.handle_request(request)
                    
                    # Write response to stdout followed by newline and flush immediately
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
                except Exception as ex:
                    log(f"Error parsing line: {str(ex)}")
        except KeyboardInterrupt:
            log("Stdio loop interrupted. Shutting down gracefully.")

if __name__ == "__main__":
    server = MCPServer()
    server.run()
