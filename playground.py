import os
import json
import time
from qdrant_controller import QdrantController
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

def print_welcome_banner():
    banner_text = (
        "[bold cyan]Qdrant Hybrid Search & Semantic Chunking Playground[/bold cyan]\n"
        "[dim]Powered by thenlper/gte-base (Dense) & BM25 (Sparse)[/dim]\n\n"
        "This terminal playground helps you compare different search methods in real-time."
    )
    console.print(Panel(banner_text, border_style="cyan"))

def load_documents_from_jsonl(file_path: str = "sample_data.jsonl"):
    """
    Loads documents line-by-line from a JSONL file.
    """
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error: JSONL dataset file not found at '{file_path}'[/bold red]")
        return []
    
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except Exception as e:
                console.print(f"[bold red]Warning: Skip malformed JSONL line {line_num}: {e}[/bold red]")
                
    return docs

def main():
    print_welcome_banner()
    
    # Initialize the controller
    with console.status("[bold green]Loading Embedding Models & Connecting to Qdrant..."):
        controller = QdrantController()
    
    console.print("[bold green]✓ Ready![/bold green] Connected to local Qdrant on http://localhost:6333\n")
    
    # Check if we should seed sample data
    seed_choice = console.input("[bold yellow]Do you want to clear and seed the database with documents from sample_data.jsonl? (y/N): [/bold yellow]").strip().lower()
    
    if seed_choice == 'y':
        with console.status("[bold magenta]Loading documents from sample_data.jsonl..."):
            documents = load_documents_from_jsonl("sample_data.jsonl")
            
        if not documents:
            console.print("[bold red]No documents loaded. Skipping seeding.[/bold red]\n")
        else:
            with console.status(f"[bold magenta]Resetting index and uploading {len(documents)} documents..."):
                controller.init_index()
                for idx, doc in enumerate(documents, 1):
                    # We upload each document using both fixed-size and semantic chunking for comparison
                    controller.upsert_document(
                        text=doc["text"],
                        metadata=doc.get("metadata", {}),
                        chunk_strategy="fixed"
                    )
            console.print(f"[bold green]✓ Database successfully seeded with {len(documents)} documents from sample_data.jsonl![/bold green]\n")
        
    # Interactive query loop
    while True:
        try:
            query = console.input("\n[bold yellow]Enter a search query (or 'exit' to quit): [/bold yellow]").strip()
            if not query:
                continue
            if query.lower() == 'exit':
                console.print("[bold cyan]Goodbye![/bold cyan]")
                break
                
            limit = 3
            
            # 1. Execute Searches
            with console.status("[bold green]Retrieving matching documents..."):
                dense_results = controller.dense_search(query, limit=limit)
                sparse_results = controller.sparse_search(query, limit=limit)
                hybrid_results = controller.hybrid_search(query, limit=limit)
                
            # 2. Display Dense Search (Semantic)
            table_dense = Table(title=f"Dense Semantic Search (thenlper/gte-base) - Matches for '{query}'", border_style="magenta", show_lines=True)
            table_dense.add_column("Rank/Score", justify="center", style="bold magenta", width=12)
            table_dense.add_column("Content Chunk", style="white")
            table_dense.add_column("Metadata", style="dim cyan", width=25)
            
            for i, res in enumerate(dense_results):
                table_dense.add_row(
                    f"#{i+1}\nScore: {res['score']:.4f}",
                    res["text"],
                    f"ID: {res['metadata'].get('doc_id')}\nCategory: {res['metadata'].get('category')}"
                )
                
            # 3. Display Sparse Search (BM25 Keyword)
            table_sparse = Table(title=f"Sparse Keyword Search (BM25) - Matches for '{query}'", border_style="yellow", show_lines=True)
            table_sparse.add_column("Rank/Score", justify="center", style="bold yellow", width=12)
            table_sparse.add_column("Content Chunk", style="white")
            table_sparse.add_column("Metadata", style="dim cyan", width=25)
            
            for i, res in enumerate(sparse_results):
                table_sparse.add_row(
                    f"#{i+1}\nScore: {res['score']:.4f}",
                    res["text"],
                    f"ID: {res['metadata'].get('doc_id')}\nCategory: {res['metadata'].get('category')}"
                )
                
            # 4. Display Hybrid Search (RRF Combined)
            table_hybrid = Table(title=f"Hybrid Search (RRF Fusion) - Matches for '{query}'", border_style="cyan", show_lines=True)
            table_hybrid.add_column("Rank/Score", justify="center", style="bold cyan", width=12)
            table_hybrid.add_column("Content Chunk", style="white")
            table_hybrid.add_column("Metadata", style="dim cyan", width=25)
            
            for i, res in enumerate(hybrid_results):
                table_hybrid.add_row(
                    f"#{i+1}\nScore: {res['score']:.4f}",
                    res["text"],
                    f"ID: {res['metadata'].get('doc_id')}\nCategory: {res['metadata'].get('category')}"
                )
                
            # Print the comparison tables
            console.print(table_dense)
            console.print("")
            console.print(table_sparse)
            console.print("")
            console.print(table_hybrid)
            
        except KeyboardInterrupt:
            console.print("\n[bold cyan]Goodbye![/bold cyan]")
            break
        except Exception as e:
            console.print(f"[bold red]Error executing search: {str(e)}[/bold red]")

if __name__ == "__main__":
    main()
