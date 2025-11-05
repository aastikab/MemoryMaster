"""FastAPI server for Cognitive Twin GUI/Obsidian integration."""

from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .core import CognitiveTwin

app = FastAPI(title="Cognitive Twin API", version="0.1.0")

# Global twin instance (lazy-loaded)
_twin: Optional[CognitiveTwin] = None


class ProcessNotesRequest(BaseModel):
    notes_dir: str


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class ConnectionRequest(BaseModel):
    k: int = 5
    threshold: float = 0.6


def get_twin() -> CognitiveTwin:
    """Get or create the global CognitiveTwin instance."""
    global _twin
    if _twin is None:
        _twin = CognitiveTwin()
    return _twin


@app.get("/")
def root():
    """API root endpoint."""
    return {"message": "Cognitive Twin API", "version": "0.1.0"}


@app.post("/process")
def process_notes(request: ProcessNotesRequest):
    """Process notes from a directory."""
    try:
        notes_dir = Path(request.notes_dir)
        if not notes_dir.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {notes_dir}")
        
        twin = get_twin()
        num_notes = twin.process_notes(notes_dir)
        return {"status": "success", "num_notes": num_notes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search_notes(request: SearchRequest):
    """Search for similar notes given a query."""
    try:
        twin = get_twin()
        if not twin.notes:
            raise HTTPException(status_code=400, detail="No notes processed. Call /process first.")
        
        # Embed query
        query_embedding = twin.embedder.embed_single(request.query)
        
        # Search FAISS
        D, I = twin.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            min(request.k, len(twin.notes))
        )
        
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(twin.notes):
                similarity = 1 / (1 + dist)  # Convert distance to similarity
                results.append({
                    "note_index": int(idx),
                    "preview": twin.notes[idx][:200],
                    "similarity": float(similarity)
                })
        
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/connections")
def get_connections(request: ConnectionRequest):
    """Get daily connections digest."""
    try:
        twin = get_twin()
        if not twin.notes:
            raise HTTPException(status_code=400, detail="No notes processed. Call /process first.")
        
        summaries = twin.find_connections(k=request.k, threshold=request.threshold)
        return {
            "status": "success",
            "num_connections": len(summaries),
            "connections": summaries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/digest")
def get_daily_digest(k: int = 5, threshold: float = 0.6):
    """Get daily connections digest (convenience endpoint)."""
    try:
        twin = get_twin()
        if not twin.notes:
            raise HTTPException(status_code=400, detail="No notes processed. Call /process first.")
        
        summaries = twin.find_connections(k=k, threshold=threshold)
        
        # Format as markdown for easy display
        markdown = "# Daily Connections Digest\n\n"
        for i, summary in enumerate(summaries, 1):
            markdown += f"## Connection {i}\n\n{summary}\n\n---\n\n"
        
        return JSONResponse(content={"markdown": markdown, "connections": summaries})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app

