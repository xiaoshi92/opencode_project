import uuid
import time
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client import models
from embeddings import EmbeddingHelper

class QdrantController:
    def __init__(self, collection_name: str = "hybrid_knowledge", url: str = "http://localhost:6333"):
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        self.embedder = EmbeddingHelper()
        self.vector_size = 768  # gte-base vector size
        
        # Ensure collection exists and is configured correctly
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Creates the Qdrant collection with both dense and sparse configurations if it doesn't exist.
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                print(f"Creating dual-vector collection '{self.collection_name}' in Qdrant...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self.vector_size,
                            distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(
                                on_disk=True
                            )
                        )
                    }
                )
                print(f"Collection '{self.collection_name}' created successfully!")
            else:
                print(f"Collection '{self.collection_name}' already exists in Qdrant.")
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")
            raise e

    def init_index(self) -> str:
        """
        Public method to force re-initialize the collection.
        This deletes the existing collection and creates a fresh one.
        """
        try:
            # Delete if exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            if self.collection_name in collection_names:
                print(f"Deleting existing collection '{self.collection_name}'...")
                self.client.delete_collection(collection_name=self.collection_name)
                
            self._ensure_collection_exists()
            return f"Index '{self.collection_name}' successfully initialized (cleared and recreated)."
        except Exception as e:
            return f"Error initializing index: {str(e)}"

    def upsert_document(self, text: str, metadata: Optional[Dict[str, Any]] = None, chunk_strategy: str = "fixed") -> Dict[str, Any]:
        """
        Preprocesses, chunks, embeds, and uploads a document to Qdrant.
        """
        if not text.strip():
            return {"status": "ignored", "reason": "Empty text document"}
            
        # 1. Preprocess & Clean text
        cleaned_text = self.embedder.clean_text(text)
        
        # 2. Chunk text
        if chunk_strategy.lower() == "semantic":
            chunks = self.embedder.chunk_text_semantic(cleaned_text)
        else:
            chunks = self.embedder.chunk_text_fixed(cleaned_text)
            
        if not chunks:
            return {"status": "ignored", "reason": "No valid chunks extracted"}
            
        print(f"Uploading {len(chunks)} chunks to Qdrant using '{chunk_strategy}' chunking strategy...")
        
        points = []
        for idx, chunk in enumerate(chunks):
            # Generate vectors
            dense_vec = self.embedder.get_dense_embedding(chunk)
            sparse_vec = self.embedder.get_sparse_embedding(chunk)
            
            # Generate deterministic UUID based on chunk text to prevent duplicate points
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk))
            
            # Combine user metadata with system properties
            payload = {
                "text": chunk,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "timestamp": time.time(),
                "chunk_strategy": chunk_strategy,
                **(metadata or {})
            }
            
            # Construct standard Qdrant PointStruct
            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": models.SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"]
                    )
                },
                payload=payload
            )
            points.append(point)
            
        # 3. Batch upload to Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            return {
                "status": "success",
                "chunks_count": len(chunks),
                "strategy": chunk_strategy,
                "points_uploaded": [p.id for p in points]
            }
        except Exception as e:
            print(f"Error uploading points to Qdrant: {e}")
            return {"status": "error", "message": str(e)}

    def dense_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Executes semantic search using dense Qwen embeddings.
        """
        query_dense = self.embedder.get_dense_embedding(query)
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_dense,
            using="dense",
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in results.points
        ]

    def sparse_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Executes lexical keyword search using sparse BM25 vectors.
        """
        query_sparse = self.embedder.get_sparse_embedding(query)
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(
                indices=query_sparse["indices"],
                values=query_sparse["values"]
            ),
            using="sparse",
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in results.points
        ]

    def hybrid_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Executes hybrid search, retrieving from both dense and sparse indices,
        fusing the results using native Reciprocal Rank Fusion (RRF).
        """
        query_dense = self.embedder.get_dense_embedding(query)
        query_sparse = self.embedder.get_sparse_embedding(query)
        
        # We prefetch candidate pools from both indices (retrieving twice the limit to ensure robust overlap)
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=limit * 2
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse["indices"],
                        values=query_sparse["values"]
                    ),
                    using="sparse",
                    limit=limit * 2
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score, # RRF fusion score
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in results.points
        ]
