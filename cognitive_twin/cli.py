"""Command-line interface for the Cognitive Twin."""

import click
from pathlib import Path
from .core import CognitiveTwin

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
def connections():
    """Find and summarize connections between notes."""
    twin = CognitiveTwin()
    summaries = twin.find_connections()
    
    click.echo("\nDaily Connections Digest:\n")
    for i, summary in enumerate(summaries, 1):
        click.echo(f"{i}. {summary}\n")

if __name__ == '__main__':
    cli()