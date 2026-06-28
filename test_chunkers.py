import sys
from embeddings import EmbeddingHelper
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

console = Console()

# Multi-topic text with extremely sharp subject transitions
MULTI_TOPIC_DOCUMENT = (
    "Qdrant is a high-performance, open-source vector search engine written in Rust. "
    "By default, it listens on port 6333 for REST communication and port 6334 for gRPC. "
    "It is highly optimized for fast similarity search across massive collections of vectors. "
    "\n\n"
    "Meanwhile, in marine biology, the study of deep-sea ecosystems has revealed incredible species. "
    "Whales, dolphins, and bioluminescent jellyfish play critical roles in maintaining oceanic cycles. "
    "These creatures survive under immense hydrostatic pressure and complete darkness. "
    "\n\n"
    "On the cybersecurity front, modern web applications require robust security configurations. "
    "Configuring strict Cross-Origin Resource Sharing (CORS) headers and SSL/TLS certificates is mandatory. "
    "Running services on HTTPS port 443 prevents man-in-the-middle attacks and snooping."
)

def main():
    console.print(Panel(
        "[bold cyan]Walkthrough: Understanding Fixed-Size vs. Semantic Chunking[/bold cyan]\n"
        "[dim]This test script showcases how a document's meaning is segmented under different strategies.[/dim]",
        border_style="cyan"
    ))
    
    with console.status("[bold green]Initializing Embedding Models..."):
        helper = EmbeddingHelper()
    
    console.print("\n[bold yellow]Original Multi-Topic Document:[/bold yellow]")
    console.print(Panel(MULTI_TOPIC_DOCUMENT, style="italic dim white"))
    
    # 1. Run Fixed-Size Chunking
    console.print("\n[bold magenta]1. Executing Fixed-Size Chunking (Target: 300 chars, Overlap: 50)...[/bold magenta]")
    fixed_chunks = helper.chunk_text_fixed(MULTI_TOPIC_DOCUMENT, chunk_size=300, overlap=50)
    
    fixed_panels = []
    for idx, chunk in enumerate(fixed_chunks):
        p = Panel(
            chunk, 
            title=f"[bold magenta]Chunk #{idx+1}[/bold magenta]", 
            subtitle=f"[dim]{len(chunk)} chars[/dim]",
            border_style="magenta",
            width=50
        )
        fixed_panels.append(p)
    console.print(Columns(fixed_panels))
    
    # 2. Run Semantic Chunking
    console.print("\n[bold cyan]2. Executing Semantic Chunking (Splitting on topic shifts)...[/bold cyan]")
    semantic_chunks = helper.chunk_text_semantic(MULTI_TOPIC_DOCUMENT)
    
    semantic_panels = []
    for idx, chunk in enumerate(semantic_chunks):
        p = Panel(
            chunk, 
            title=f"[bold cyan]Semantic Chunk #{idx+1}[/bold cyan]", 
            subtitle=f"[dim]{len(chunk)} chars[/dim]",
            border_style="cyan",
            width=50
        )
        semantic_panels.append(p)
    console.print(Columns(semantic_panels))
    
    console.print("\n[bold green]Educational Conclusion:[/bold green]")
    console.print(
        "- [bold magenta]Fixed-Size Chunking[/bold magenta] cuts text blindly at hard character boundaries, which often slices paragraphs or merges irrelevant topics (like Database specs and Marine Life) into the same chunk.\n"
        "- [bold cyan]Semantic Chunking[/bold cyan] analyzes sentence embeddings and splits text *only* when cosine similarity drops (topic shifts). This guarantees high semantic coherence for each chunk, drastically increasing retrieval accuracy!"
    )

if __name__ == "__main__":
    main()
