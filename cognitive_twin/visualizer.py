"""Visualization module for note clusters and relationships."""

from typing import List, Dict, Tuple, Optional
import numpy as np
import networkx as nx

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

class NoteVisualizer:
    """Visualize note clusters and relationships."""
    
    def __init__(self):
        self.plotly_available = PLOTLY_AVAILABLE
        self.streamlit_available = STREAMLIT_AVAILABLE
    
    def build_note_graph(self, notes: List[str], connections: List[Dict], 
                        similarity_threshold: float = 0.5) -> nx.Graph:
        """Build a NetworkX graph from notes and connections.
        
        Args:
            notes: List of note texts
            connections: List of connection dictionaries with 'index', 'similarity', etc.
            similarity_threshold: Minimum similarity to include edge
        
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Add nodes
        for i, note in enumerate(notes):
            preview = note[:50] + "..." if len(note) > 50 else note
            G.add_node(i, label=f"Note {i}", preview=preview, full_text=note)
        
        # Add edges from connections
        for conn in connections:
            if 'index' in conn and 'similarity' in conn:
                node_idx = conn['index']
                similarity = conn['similarity']
                
                # Find connected nodes (for now, connect to query node if available)
                # In a full implementation, you'd track which notes are connected
                if similarity >= similarity_threshold:
                    # This is a simplified version - you'd need to track actual pairs
                    pass
        
        return G
    
    def create_network_graph(self, notes: List[str], embeddings: np.ndarray, 
                           top_k: int = 3, similarity_threshold: float = 0.5) -> Optional[go.Figure]:
        """Create an interactive network graph visualization.
        
        Args:
            notes: List of note texts
            embeddings: Embedding matrix for notes
            top_k: Number of top connections per note
            similarity_threshold: Minimum similarity to show edge
        
        Returns:
            Plotly figure or None if plotly not available
        """
        if not self.plotly_available:
            return None
        
        # Compute similarity matrix using numpy
        # Normalize embeddings first
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Build graph
        G = nx.Graph()
        
        # Add nodes
        for i, note in enumerate(notes):
            preview = note[:50] + "..." if len(note) > 50 else note
            G.add_node(i, label=f"Note {i}", preview=preview)
        
        # Add edges based on similarity
        edges = []
        edge_weights = []
        for i in range(len(notes)):
            # Get top k similar notes (excluding self)
            similarities = similarity_matrix[i]
            top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Exclude self
            
            for j in top_indices:
                if similarities[j] >= similarity_threshold:
                    G.add_edge(i, j, weight=similarities[j])
                    edges.append((i, j))
                    edge_weights.append(similarities[j])
        
        if len(G.nodes()) == 0:
            return None
        
        # Use spring layout for positioning
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract node positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=G[edge[0]][edge[1]].get('weight', 0.5) * 3, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[G.nodes[node].get('label', f'Note {node}') for node in G.nodes()],
            textposition="middle center",
            marker=dict(
                size=20,
                color='#1f77b4',
                line=dict(width=2, color='white')
            ),
            customdata=[G.nodes[node].get('preview', '') for node in G.nodes()],
            hovertemplate='<b>%{text}</b><br>%{customdata}<extra></extra>'
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title='Note Clusters and Relationships',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           annotations=[dict(
                               text="Hover over nodes to see note previews",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def create_similarity_heatmap(self, notes: List[str], embeddings: np.ndarray) -> Optional[go.Figure]:
        """Create a heatmap of note similarities.
        
        Args:
            notes: List of note texts
            embeddings: Embedding matrix
        
        Returns:
            Plotly figure or None
        """
        if not self.plotly_available:
            return None
        
        # Compute similarity matrix using numpy
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=[f"Note {i}" for i in range(len(notes))],
            y=[f"Note {i}" for i in range(len(notes))],
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Note Similarity Matrix',
            xaxis_title='Notes',
            yaxis_title='Notes',
            width=800,
            height=800
        )
        
        return fig

