from embeddings import EmbeddingHelper
import sys

try:
    print("Initializing EmbeddingHelper...")
    helper = EmbeddingHelper()
    
    print("Testing clean_text...")
    cleaned = helper.clean_text("Hello  World! \n\n Testing control characters: \x03 and Unicode: \u201cquotes\u201d.")
    print("Cleaned text:", repr(cleaned))
    
    print("Testing fixed-size chunking...")
    chunks = helper.chunk_text_fixed("This is a simple sentence that we are using to test chunking.", chunk_size=30, overlap=5)
    print("Chunks:", chunks)
    
    print("Testing dense embedding generation...")
    dense_vec = helper.get_dense_embedding("Test sentence")
    print("Dense dimension:", len(dense_vec))
    
    print("Testing sparse embedding generation...")
    sparse_vec = helper.get_sparse_embedding("Test sentence")
    print("Sparse keys:", list(sparse_vec.keys()))
    print("Sparse indices length:", len(sparse_vec["indices"]))
    print("Sparse values length:", len(sparse_vec["values"]))
    
    print("ALL TESTS PASSED SUCCESSFULLY!")
    sys.exit(0)
except Exception as e:
    import traceback
    print("TEST FAILED!")
    traceback.print_exc()
    sys.exit(1)
