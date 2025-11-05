"""
Streamlit app for Cognitive Twin - Interactive Note Analysis
Upload or write notes to find connections with your existing knowledge base
"""

import streamlit as st
from pathlib import Path
import tempfile
import os
from datetime import datetime
from cognitive_twin.core import CognitiveTwin
from cognitive_twin.embedder import NoteEmbedder
from cognitive_twin.summarizer import NoteSummarizer
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Cognitive Twin - Note Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'twin' not in st.session_state:
    st.session_state.twin = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'notes_history' not in st.session_state:
    st.session_state.notes_history = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .connection-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .grounded {
        color: #28a745;
        font-weight: bold;
    }
    .possibly-ungrounded {
        color: #ffc107;
        font-weight: bold;
    }
    .similarity-score {
        background-color: #1f77b4;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.9rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 Cognitive Twin - Note Analyzer</div>', unsafe_allow_html=True)
st.markdown("**Discover connections between your notes using AI-powered semantic analysis**")
st.markdown("---")

# Sidebar - Knowledge Base Setup
with st.sidebar:
    st.header("📚 Knowledge Base Setup")
    
    # Option 1: Load existing notes
    notes_dir = st.text_input(
        "Notes Directory Path",
        value="sample_notes",
        help="Path to your existing notes folder (markdown files)"
    )
    
    if st.button("🔄 Load Notes", use_container_width=True):
        with st.spinner("Loading your knowledge base..."):
            try:
                notes_path = Path(notes_dir)
                if not notes_path.exists():
                    st.error(f"Directory not found: {notes_dir}")
                else:
                    # Initialize twin
                    st.session_state.twin = CognitiveTwin()
                    num_notes = st.session_state.twin.process_notes(notes_path)
                    st.session_state.processed = True
                    st.success(f"✅ Loaded {num_notes} notes!")
                    st.session_state.notes_history = []
            except Exception as e:
                st.error(f"Error loading notes: {e}")
    
    # Display status
    if st.session_state.processed:
        st.info(f"📊 **{len(st.session_state.twin.notes)}** notes in knowledge base")
        
        # Option to reset
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            st.session_state.twin = None
            st.session_state.processed = False
            st.session_state.notes_history = []
            st.rerun()
    else:
        st.warning("⚠️ No knowledge base loaded. Load notes to begin.")
    
    st.markdown("---")
    
    # Settings
    st.header("⚙️ Settings")
    k_neighbors = st.slider("Number of similar notes to find", 1, 10, 3)
    similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.1)
    verify_groundedness = st.checkbox("Verify summary groundedness", value=True)

# Main content area
if not st.session_state.processed:
    # Welcome message when no knowledge base is loaded
    st.info("👈 **Get started:** Load your notes from the sidebar to begin finding connections!")
    
    st.markdown("""
    ### How it works:
    
    1. **Load your knowledge base** - Point to a directory containing your markdown notes
    2. **Write or upload a new note** - Add new content you want to analyze
    3. **Find connections** - AI discovers similar notes from your knowledge base
    4. **Get summaries** - Receive grounded summaries with citations
    
    ### Features:
    - 🔍 **Semantic search** - Finds connections based on meaning, not just keywords
    - 📝 **Citation enforcement** - Every summary includes source quotes
    - ✅ **Groundedness verification** - NLI model checks summary faithfulness
    - ⚡ **Cached embeddings** - Fast repeated operations
    """)
    
else:
    # Main interface when knowledge base is loaded
    tab1, tab2, tab3 = st.tabs(["✍️ Write Note", "📤 Upload Note", "📊 History"])
    
    # Tab 1: Write a new note
    with tab1:
        st.subheader("✍️ Write a New Note")
        
        note_title = st.text_input("Note Title (optional)", placeholder="e.g., My thoughts on learning...")
        note_content = st.text_area(
            "Note Content",
            height=200,
            placeholder="Write your note here...\n\nExample: Today I learned that exercise improves memory formation through increased BDNF levels...",
            help="Write your note content. The system will find similar notes from your knowledge base."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("🔍 Find Connections", use_container_width=True)
        
        if analyze_button and note_content.strip():
            with st.spinner("Analyzing note and finding connections..."):
                try:
                    # Get embeddings for the new note
                    embedder = st.session_state.twin.embedder
                    query_embedding = embedder.embed_single(note_content)
                    
                    # Search for similar notes
                    D, I = st.session_state.twin.index.search(
                        query_embedding.reshape(1, -1).astype('float32'),
                        min(k_neighbors, len(st.session_state.twin.notes))
                    )
                    
                    # Process results
                    connections = []
                    for idx, dist in zip(I[0], D[0]):
                        if idx < len(st.session_state.twin.notes):
                            similarity = 1 / (1 + dist)
                            
                            # Only show if above threshold
                            if similarity >= similarity_threshold:
                                # Generate summary
                                summarizer = st.session_state.twin.summarizer
                                summary = summarizer.summarize_with_citations(
                                    note_content,
                                    st.session_state.twin.notes[idx],
                                    verify=verify_groundedness
                                )
                                
                                connections.append({
                                    'index': int(idx),
                                    'similarity': float(similarity),
                                    'note': st.session_state.twin.notes[idx],
                                    'summary': summary
                                })
                    
                    # Display results
                    st.markdown("---")
                    st.subheader(f"🔗 Found {len(connections)} Connection(s)")
                    
                    if connections:
                        # Save to history
                        st.session_state.notes_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'title': note_title or "Untitled Note",
                            'content': note_content[:200] + "...",
                            'connections': len(connections)
                        })
                        
                        for i, conn in enumerate(connections, 1):
                            with st.expander(f"📄 Connection {i} - Note #{conn['index']} (Similarity: {conn['similarity']:.2%})", expanded=True):
                                # Original note preview
                                st.markdown("**📝 Similar Note:**")
                                st.text(conn['note'][:300] + "..." if len(conn['note']) > 300 else conn['note'])
                                
                                st.markdown("---")
                                
                                # Summary with citations
                                st.markdown("**🤖 AI-Generated Summary:**")
                                st.markdown(conn['summary'])
                    else:
                        st.info(f"No connections found above {similarity_threshold:.0%} similarity threshold. Try lowering the threshold in the sidebar.")
                
                except Exception as e:
                    st.error(f"Error analyzing note: {e}")
        
        elif analyze_button:
            st.warning("⚠️ Please write some content before analyzing.")
    
    # Tab 2: Upload a note file
    with tab2:
        st.subheader("📤 Upload a Note File")
        
        uploaded_file = st.file_uploader(
            "Choose a markdown or text file",
            type=['md', 'txt'],
            help="Upload a markdown or text file to analyze"
        )
        
        if uploaded_file is not None:
            # Read file content
            file_content = uploaded_file.read().decode('utf-8')
            
            st.markdown("**Preview:**")
            with st.container():
                st.text_area("File Content", file_content, height=200, disabled=True)
            
            if st.button("🔍 Find Connections for Uploaded File", use_container_width=True):
                with st.spinner("Analyzing uploaded file..."):
                    try:
                        # Get embeddings for the uploaded content
                        embedder = st.session_state.twin.embedder
                        query_embedding = embedder.embed_single(file_content)
                        
                        # Search for similar notes
                        D, I = st.session_state.twin.index.search(
                            query_embedding.reshape(1, -1).astype('float32'),
                            min(k_neighbors, len(st.session_state.twin.notes))
                        )
                        
                        # Process results
                        connections = []
                        for idx, dist in zip(I[0], D[0]):
                            if idx < len(st.session_state.twin.notes):
                                similarity = 1 / (1 + dist)
                                
                                if similarity >= similarity_threshold:
                                    summarizer = st.session_state.twin.summarizer
                                    summary = summarizer.summarize_with_citations(
                                        file_content,
                                        st.session_state.twin.notes[idx],
                                        verify=verify_groundedness
                                    )
                                    
                                    connections.append({
                                        'index': int(idx),
                                        'similarity': float(similarity),
                                        'note': st.session_state.twin.notes[idx],
                                        'summary': summary
                                    })
                        
                        # Display results
                        st.markdown("---")
                        st.subheader(f"🔗 Found {len(connections)} Connection(s)")
                        
                        if connections:
                            # Save to history
                            st.session_state.notes_history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'title': uploaded_file.name,
                                'content': file_content[:200] + "...",
                                'connections': len(connections)
                            })
                            
                            for i, conn in enumerate(connections, 1):
                                with st.expander(f"📄 Connection {i} - Note #{conn['index']} (Similarity: {conn['similarity']:.2%})", expanded=True):
                                    st.markdown("**📝 Similar Note:**")
                                    st.text(conn['note'][:300] + "..." if len(conn['note']) > 300 else conn['note'])
                                    
                                    st.markdown("---")
                                    
                                    st.markdown("**🤖 AI-Generated Summary:**")
                                    st.markdown(conn['summary'])
                        else:
                            st.info(f"No connections found above {similarity_threshold:.0%} similarity threshold.")
                    
                    except Exception as e:
                        st.error(f"Error analyzing file: {e}")
    
    # Tab 3: History
    with tab3:
        st.subheader("📊 Analysis History")
        
        if st.session_state.notes_history:
            st.info(f"Total analyses: {len(st.session_state.notes_history)}")
            
            for i, entry in enumerate(reversed(st.session_state.notes_history), 1):
                with st.expander(f"🕒 {entry['timestamp']} - {entry['title']}"):
                    st.markdown(f"**Connections found:** {entry['connections']}")
                    st.markdown(f"**Preview:** {entry['content']}")
        else:
            st.info("No analysis history yet. Analyze some notes to see them here!")
        
        if st.session_state.notes_history and st.button("🗑️ Clear History"):
            st.session_state.notes_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🧠 <strong>Cognitive Twin</strong> - Your Personal Knowledge Assistant</p>
    <p>Powered by SentenceTransformers, FAISS, and BART</p>
</div>
""", unsafe_allow_html=True)

