"""Command-line interface for the Cognitive Twin."""

import click
from pathlib import Path
from .core import CognitiveTwin
from .api import create_app

@click.group()
def cli():
    """Cognitive Twin - Your personal knowledge assistant."""
    pass

@cli.command()
@click.argument('notes_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
def process(notes_dir: Path):
    """Process notes in the specified directory."""
    twin = CognitiveTwin()
    num_notes = twin.process_notes(notes_dir)
    click.echo(f"Processed {num_notes} notes from {notes_dir}")

@cli.command()
@click.argument('notes_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option('--k', default=5, show_default=True, help='Number of neighbors to consider per note')
@click.option('--threshold', default=0.6, show_default=True, help='Similarity threshold for reporting connections')
def connections(notes_dir: Path, k: int, threshold: float):
    """Process NOTES_DIR and print the Daily Connections Digest."""
    twin = CognitiveTwin()
    num_notes = twin.process_notes(notes_dir)
    if num_notes == 0:
        click.echo("No notes found. Make sure NOTES_DIR contains .md files.")
        return

    summaries = twin.find_connections(k=k, threshold=threshold)

    click.echo("\nDaily Connections Digest:\n")
    if not summaries:
        click.echo("No connections found. Try lowering --threshold or adding more notes.")
        return

    for i, summary in enumerate(summaries, 1):
        click.echo(f"{i}. {summary}\n")

@cli.command()
@click.option('--host', default='127.0.0.1', show_default=True)
@click.option('--port', default=8000, show_default=True, type=int)
def serve(host: str, port: int):
    """Run a local API server for GUI/Obsidian integration."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    cli()