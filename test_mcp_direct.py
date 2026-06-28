# Importing the tools directly from our mcp_server module
from mcp_server import hybrid_search, upsert_document, init_index
from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    console.print(Panel(
        "[bold green]Testing MCP Methods Directly in Python[/bold green]\n"
        "[dim]Since FastMCP uses standard Python functions, we can call them directly![/dim]",
        border_style="green"
    ))

    # 1. Initialize the index
    console.print("\n[bold yellow]Calling MCP Tool: 'init_index'[/bold yellow]")
    init_msg = init_index()
    console.print(f"Result: [green]{init_msg}[/green]")

    # 2. Upsert a document using the MCP tool
    console.print("\n[bold yellow]Calling MCP Tool: 'upsert_document'[/bold yellow]")
    upsert_msg = upsert_document(
        text="FastMCP is the official framework to build Model Context Protocol servers in Python.",
        metadata={"source": "mcp-guide"},
        chunk_strategy="fixed"
    )
    console.print(f"Result JSON:\n[cyan]{upsert_msg}[/cyan]")

    # 3. Perform a search using the MCP tool
    console.print("\n[bold yellow]Calling MCP Tool: 'hybrid_search'[/bold yellow]")
    search_msg = hybrid_search(query="mcp framework", limit=1)
    console.print(f"Result JSON:\n[magenta]{search_msg}[/magenta]")

if __name__ == "__main__":
    main()
