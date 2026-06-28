import re
import unicodedata
from typing import List, Dict, Any
import numpy as np
from fastembed import TextEmbedding, SparseTextEmbedding

class EmbeddingHelper:
    def __init__(self):
        # We initialize both models. FastEmbed loads them on demand and caches them.
        print("Initializing dense embedding model (thenlper/gte-base)...")
        self.dense_model = TextEmbedding(model_name="thenlper/gte-base")
        
        print("Initializing sparse embedding model (bm25)...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def clean_text(self, text: str) -> str:
        """
        Normalizes unicode, removes non-printable control characters, 
        and cleans up whitespaces while preserving code symbols.
        """
        if not text:
            return ""
        # 1. Normalize Unicode (standardizes curly quotes, hyphens, ligatures etc.)
        text = unicodedata.normalize("NFKC", text)
        
        # 2. Strip non-printable control characters (ASCII 0-31 and 127-159)
        # We keep newline '\n' and tab '\t' as they carry formatting/code meaning.
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # 3. Clean up excessive whitespaces while preserving line integrity
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # 4. Remove excessive empty lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def chunk_text_fixed(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
        """
        Recursively splits text into fixed-size chunks with overlap.
        Maintains paragraph, sentence, and word integrity.
        """
        if not text:
            return []
            
        separators = ["\n\n", "\n", " ", ""]
        chunks = []
        
        def recurse(subtext: str, current_seps: List[str]):
            if len(subtext) <= chunk_size:
                chunks.append(subtext)
                return
                
            if not current_seps:
                # Fallback hard slice if we have no separators left (rare)
                chunks.append(subtext[:chunk_size])
                recurse(subtext[chunk_size:], current_seps)
                return
                
            sep = current_seps[0]
            parts = subtext.split(sep) if sep else list(subtext)
            
            current_chunk = ""
            for part in parts:
                joiner = sep if current_chunk else ""
                potential_chunk = current_chunk + joiner + part
                
                if len(potential_chunk) <= chunk_size:
                    current_chunk = potential_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        # Carry over overlap
                        overlap_start = max(0, len(current_chunk) - overlap)
                        overlap_text = current_chunk[overlap_start:]
                        current_chunk = overlap_text + joiner + part
                    else:
                        # Single part exceeds size, recurse on it with next separator
                        recurse(part, current_seps[1:])
                        
            if current_chunk:
                chunks.append(current_chunk)
                
        recurse(text, separators)
        
        # Post-filter out empty or tiny chunks under 30 characters
        return [c.strip() for c in chunks if len(c.strip()) >= 30]

    def chunk_text_semantic(self, text: str, max_chunk_size: int = 2000, min_chunk_size: int = 150) -> List[str]:
        """
        Splits text into semantic chunks based on topic shifts.
        Uses sentence-level cosine similarities of dense Qwen embeddings.
        """
        if not text:
            return []
            
        # 1. Split text into individual sentences
        # Regex matches sentence endings (.?!), followed by whitespace and capital letter
        sentence_regex = r'(?<=[.!?])\s+(?=[A-Z0-9])'
        sentences = [s.strip() for s in re.split(sentence_regex, text) if s.strip()]
        
        if len(sentences) <= 1:
            return [text] if text else []
            
        # 2. Embed sentences
        print(f"Generating embeddings for {len(sentences)} sentences...")
        sentence_embeddings = list(self.dense_model.embed(sentences))
        
        # 3. Calculate cosine similarities between adjacent sentences
        similarities = []
        for i in range(len(sentences) - 1):
            v1 = sentence_embeddings[i]
            v2 = sentence_embeddings[i+1]
            
            dot = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            sim = float(dot / (norm_v1 * norm_v2)) if norm_v1 > 0 and norm_v2 > 0 else 0.0
            similarities.append(sim)
            
        # 4. Set threshold at 20th percentile
        # (This adapts to the cohesive style of the current document)
        if len(similarities) > 0:
            threshold = float(np.percentile(similarities, 20))
        else:
            threshold = 0.5
            
        # 5. Assemble chunks based on topic shifts
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        for i, sim in enumerate(similarities):
            sentence = sentences[i + 1]
            current_chunk_length = sum(len(s) for s in current_chunk_sentences)
            
            # Decide if we split
            if sim < threshold and current_chunk_length >= min_chunk_size:
                # Topic shift and chunk is big enough -> Split
                chunks.append(" ".join(current_chunk_sentences))
                # Sentence overlap: carry over 1 sentence for a semantic bridge
                current_chunk_sentences = [current_chunk_sentences[-1], sentence]
            else:
                # Check if adding the next sentence exceeds max_chunk_size
                if current_chunk_length + len(sentence) > max_chunk_size:
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = [sentence]
                else:
                    current_chunk_sentences.append(sentence)
                    
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        return [c.strip() for c in chunks if len(c.strip()) >= 30]

    def get_dense_embedding(self, text: str) -> List[float]:
        """
        Generates a dense vector for a single text chunk using GTE-Qwen.
        """
        # Embed returns a generator of numpy arrays. We extract the first one.
        embeddings = list(self.dense_model.embed([text]))
        return [float(x) for x in embeddings[0]]

    def get_sparse_embedding(self, text: str) -> Dict[str, Any]:
        """
        Generates a sparse vector (indices and values) using FastEmbed's BM25 model.
        Returns format compatible with Qdrant: {"indices": List[int], "values": List[float]}
        """
        # Embed returns a generator of SparseEmbedding objects.
        embeddings = list(self.sparse_model.embed([text]))
        sparse_vec = embeddings[0]
        # sparse_vec has .indices (List[int]) and .values (List[float])
        return {
            "indices": [int(idx) for idx in sparse_vec.indices],
            "values": [float(val) for val in sparse_vec.values]
        }
