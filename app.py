"""
Streamlit app for Cognitive Twin - Interactive Note Analysis
Upload or write notes to find connections with your existing knowledge base
"""

import streamlit as st
from pathlib import Path
import tempfile
import os
from datetime import datetime
from dotenv import load_dotenv
from cognitive_twin.core import CognitiveTwin
from cognitive_twin.embedder import NoteEmbedder
from cognitive_twin.summarizer import NoteSummarizer, SummaryWithCitations
from cognitive_twin.exporter import SummaryExporter
from cognitive_twin.visualizer import NoteVisualizer
from cognitive_twin.chatbot import ChatbotManager
import numpy as np

# Load environment variables from .env file
load_dotenv()

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
if 'current_connections' not in st.session_state:
    st.session_state.current_connections = []
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'citation_preview' not in st.session_state:
    st.session_state.citation_preview = None

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
    .citation-link {
        color: #1f77b4;
        text-decoration: underline;
        cursor: pointer;
        font-weight: 500;
    }
    .citation-link:hover {
        color: #0d5a9b;
    }
    .note-preview {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def render_clickable_citation(citation_text: str, full_text: str, citation_id: str):
    """Render a clickable citation that shows note preview on click."""
    button_key = f"citation_{citation_id}"
    if st.button(f"📎 {citation_text[:50]}...", key=button_key, use_container_width=False):
        st.session_state.citation_preview = {
            'id': citation_id,
            'text': citation_text,
            'full_text': full_text
        }
    return button_key

def render_summary_with_citations(summary_data, note_index: int, query_note: str = ""):
    """Render summary with clickable citations."""
    if isinstance(summary_data, SummaryWithCitations):
        # Render structured summary
        st.markdown(summary_data.summary)
        
        # Groundedness badge
        if summary_data.grounded_score is not None:
            badge = "✅ Grounded" if summary_data.is_grounded else "⚠ Possibly ungrounded"
            st.markdown(f"**[{badge} ({summary_data.grounded_score:.0%} entailed)]**")
        
        st.markdown("**Citations:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Citation 1:**")
            citation1_text = summary_data.citation1.text[:40] + "..." if len(summary_data.citation1.text) > 40 else summary_data.citation1.text
            
            # Show web link if URL available, otherwise show preview button
            if summary_data.citation1.url:
                st.markdown(f'<a href="{summary_data.citation1.url}" target="_blank" style="text-decoration: none; color: #1f77b4; font-weight: 500;">🔗 {citation1_text}</a>', 
                           unsafe_allow_html=True)
                st.caption(f"Source: {summary_data.citation1.url[:50]}...")
            else:
                if st.button(f"📎 {citation1_text}", 
                            key=f"cite1_{note_index}", use_container_width=True):
                    st.session_state.citation_preview = {
                        'id': f'note_{note_index}_cite1',
                        'text': summary_data.citation1.text,
                        'full_text': summary_data.citation1.full_text,
                        'url': summary_data.citation1.url
                    }
        
        with col2:
            st.markdown("**Citation 2:**")
            citation2_text = summary_data.citation2.text[:40] + "..." if len(summary_data.citation2.text) > 40 else summary_data.citation2.text
            
            # Show web link if URL available, otherwise show preview button
            if summary_data.citation2.url:
                st.markdown(f'<a href="{summary_data.citation2.url}" target="_blank" style="text-decoration: none; color: #1f77b4; font-weight: 500;">🔗 {citation2_text}</a>', 
                           unsafe_allow_html=True)
                st.caption(f"Source: {summary_data.citation2.url[:50]}...")
            else:
                if st.button(f"📎 {citation2_text}", 
                            key=f"cite2_{note_index}", use_container_width=True):
                    st.session_state.citation_preview = {
                        'id': f'note_{note_index}_cite2',
                        'text': summary_data.citation2.text,
                        'full_text': summary_data.citation2.full_text,
                        'url': summary_data.citation2.url
                    }
    else:
        # Fallback to plain text
        st.markdown(summary_data)

# Header
st.markdown('<div class="main-header">🧠 Cognitive Twin - Note Analyzer</div>', unsafe_allow_html=True)
st.markdown("**Discover connections between your notes using AI-powered semantic analysis**")
st.markdown("---")

# Citation preview modal
if st.session_state.citation_preview:
    with st.expander("📄 Citation Preview", expanded=True):
        st.markdown(f"**Citation:** {st.session_state.citation_preview['text']}")
        
        # Show web link if available
        if st.session_state.citation_preview.get('url'):
            st.markdown("---")
            st.markdown("**🔗 Source Article:**")
            st.markdown(f'<a href="{st.session_state.citation_preview["url"]}" target="_blank" style="text-decoration: none; color: #1f77b4; font-weight: 500;">{st.session_state.citation_preview["url"]}</a>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Full Note:**")
        st.markdown(f'<div class="note-preview">{st.session_state.citation_preview["full_text"]}</div>', 
                   unsafe_allow_html=True)
        if st.button("❌ Close Preview"):
            st.session_state.citation_preview = None
            st.rerun()

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
    
    st.markdown("---")
    
    # Chatbot Settings
    st.header("🤖 Chatbot")
    chatbot_provider = st.selectbox("Provider", ["openai", "claude"], index=0)
    use_local_summaries = st.checkbox("Use local summaries only (privacy)", value=True)
    
    if st.button("🔧 Initialize Chatbot", use_container_width=True):
        try:
            st.session_state.chatbot = ChatbotManager(
                provider=chatbot_provider,
                use_local_summaries=use_local_summaries
            )
            if st.session_state.chatbot.is_available():
                st.success("✅ Chatbot initialized!")
            else:
                st.warning("⚠️ Chatbot not available. Check API keys.")
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
    
    if st.session_state.chatbot and st.session_state.chatbot.is_available():
        st.success("✅ Chatbot ready")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chatbot.clear_history()
            st.rerun()

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "✍️ Write Note", "📤 Upload Note", "📦 Batch Process", 
        "📊 Visualizations", "💬 Chatbot", "📊 History"
    ])
    
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
                                # Generate summary with structured citations
                                summarizer = st.session_state.twin.summarizer
                                summary = summarizer.summarize_with_citations(
                                    note_content,
                                    st.session_state.twin.notes[idx],
                                    verify=verify_groundedness,
                                    note1_index=None,  # Query note (not in KB)
                                    note2_index=int(idx),
                                    return_structured=True
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
                        # Save to history and current connections
                        st.session_state.current_connections = connections
                        st.session_state.notes_history.append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'title': note_title or "Untitled Note",
                            'content': note_content[:200] + "...",
                            'connections': len(connections)
                        })
                        
                        # Export buttons
                        col_exp1, col_exp2 = st.columns(2)
                        exporter = SummaryExporter()
                        
                        with col_exp1:
                            md_content = exporter.export_to_markdown(connections)
                            st.download_button(
                                label="📥 Export to Markdown",
                                data=md_content,
                                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                        
                        with col_exp2:
                            try:
                                pdf_buffer = exporter.export_to_pdf(connections)
                                st.download_button(
                                    label="📥 Export to PDF",
                                    data=pdf_buffer.getvalue(),
                                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                            except Exception as e:
                                st.warning(f"PDF export unavailable: {e}")
                        
                        for i, conn in enumerate(connections, 1):
                            with st.expander(f"📄 Connection {i} - Note #{conn['index']} (Similarity: {conn['similarity']:.2%})", expanded=True):
                                # Original note preview
                                st.markdown("**📝 Similar Note:**")
                                st.text(conn['note'][:300] + "..." if len(conn['note']) > 300 else conn['note'])
                                
                                st.markdown("---")
                                
                                # Summary with clickable citations
                                st.markdown("**🤖 AI-Generated Summary:**")
                                render_summary_with_citations(conn['summary'], conn['index'], note_content)
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
                                        verify=verify_groundedness,
                                        note1_index=None,
                                        note2_index=int(idx),
                                        return_structured=True
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
                            st.session_state.current_connections = connections
                            # Save to history
                            st.session_state.notes_history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'title': uploaded_file.name,
                                'content': file_content[:200] + "...",
                                'connections': len(connections)
                            })
                            
                            # Export buttons
                            col_exp1, col_exp2 = st.columns(2)
                            exporter = SummaryExporter()
                            
                            with col_exp1:
                                md_content = exporter.export_to_markdown(connections)
                                st.download_button(
                                    label="📥 Export to Markdown",
                                    data=md_content,
                                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown"
                                )
                            
                            with col_exp2:
                                try:
                                    pdf_buffer = exporter.export_to_pdf(connections)
                                    st.download_button(
                                        label="📥 Export to PDF",
                                        data=pdf_buffer.getvalue(),
                                        file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        mime="application/pdf"
                                    )
                                except Exception as e:
                                    st.warning(f"PDF export unavailable: {e}")
                            
                            for i, conn in enumerate(connections, 1):
                                with st.expander(f"📄 Connection {i} - Note #{conn['index']} (Similarity: {conn['similarity']:.2%})", expanded=True):
                                    st.markdown("**📝 Similar Note:**")
                                    st.text(conn['note'][:300] + "..." if len(conn['note']) > 300 else conn['note'])
                                    
                                    st.markdown("---")
                                    
                                    st.markdown("**🤖 AI-Generated Summary:**")
                                    render_summary_with_citations(conn['summary'], conn['index'], file_content)
                        else:
                            st.info(f"No connections found above {similarity_threshold:.0%} similarity threshold.")
                    
                    except Exception as e:
                        st.error(f"Error analyzing file: {e}")
    
    # Tab 3: Batch Processing
    with tab3:
        st.subheader("📦 Batch Note Processing")
        st.markdown("Process multiple notes at once for efficient analysis.")
        
        uploaded_files = st.file_uploader(
            "Choose multiple markdown or text files",
            type=['md', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple files to process in batch"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
            
            if st.button("🔄 Process Batch", use_container_width=True):
                with st.spinner("Processing batch..."):
                    try:
                        # Read all files
                        batch_notes = []
                        for uploaded_file in uploaded_files:
                            content = uploaded_file.read().decode('utf-8')
                            if content.strip():
                                batch_notes.append(content)
                        
                        if batch_notes:
                            # Process batch
                            num_processed = st.session_state.twin.process_notes_batch(batch_notes)
                            st.success(f"✅ Processed {num_processed} notes!")
                            
                            # Analyze batch
                            st.markdown("---")
                            st.subheader("🔍 Batch Analysis Results")
                            
                            batch_results = st.session_state.twin.analyze_batch(
                                batch_notes,
                                k=k_neighbors,
                                similarity_threshold=similarity_threshold
                            )
                            
                            for i, (file, results) in enumerate(zip(uploaded_files, batch_results)):
                                with st.expander(f"📄 {file.name} - {len(results)} connections"):
                                    if results:
                                        for j, conn in enumerate(results, 1):
                                            st.markdown(f"**Connection {j}:** Note #{conn['index']} (Similarity: {conn['similarity']:.2%})")
                                            st.text(conn['note'][:200] + "...")
                                    else:
                                        st.info("No connections found above threshold.")
                        else:
                            st.warning("No valid content found in uploaded files.")
                    
                    except Exception as e:
                        st.error(f"Error processing batch: {e}")
    
    # Tab 4: Visualizations
    with tab4:
        st.subheader("📊 Note Clusters and Relationships")
        
        if len(st.session_state.twin.notes) > 0:
            visualizer = NoteVisualizer()
            
            # Get embeddings for visualization
            embeddings = st.session_state.twin.embedder.embed_texts(st.session_state.twin.notes)
            
            # Network graph
            st.markdown("### Network Graph of Note Relationships")
            fig = visualizer.create_network_graph(
                st.session_state.twin.notes,
                embeddings,
                top_k=3,
                similarity_threshold=similarity_threshold
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Plotly not available. Install with: pip install plotly")
            
            # Similarity heatmap
            st.markdown("### Similarity Heatmap")
            heatmap_fig = visualizer.create_similarity_heatmap(
                st.session_state.twin.notes,
                embeddings
            )
            
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            else:
                st.warning("Plotly not available for heatmap visualization.")
        else:
            st.info("Load notes first to see visualizations.")
    
    # Tab 5: Chatbot
    with tab5:
        st.subheader("💬 Chat with AI Assistant")
        
        if st.session_state.chatbot and st.session_state.chatbot.is_available():
            # Inject context from current session
            context = None
            if st.session_state.current_connections:
                notes_list = [conn['note'] for conn in st.session_state.current_connections[:5]]
                summaries_list = []
                for conn in st.session_state.current_connections[:3]:
                    if isinstance(conn['summary'], SummaryWithCitations):
                        summaries_list.append(conn['summary'].summary)
                    else:
                        summaries_list.append(str(conn['summary']))
                
                context = st.session_state.chatbot.inject_context(
                    notes_list,
                    st.session_state.current_connections,
                    summaries_list
                )
            
            # Chat interface
            st.markdown("**Ask questions about your notes and connections:**")
            
            # Display chat history
            if st.session_state.chatbot.chat_history:
                for msg in st.session_state.chatbot.chat_history:
                    role_icon = "👤" if msg['role'] == 'user' else "🤖"
                    st.markdown(f"**{role_icon} {msg['role'].title()}:**")
                    st.markdown(msg['content'])
                    st.markdown("---")
            
            # User input
            user_input = st.text_input("Your message:", key="chat_input")
            
            col_chat1, col_chat2 = st.columns([1, 4])
            with col_chat1:
                send_button = st.button("💬 Send", use_container_width=True)
            
            if send_button and user_input:
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.chatbot.chat(user_input, context=context)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("👈 Initialize chatbot from the sidebar to start chatting.")
            if st.session_state.chatbot:
                st.warning("⚠️ Chatbot not available. Check API keys in environment variables.")
    
    # Tab 6: History
    with tab6:
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

