Midterm Progress Report
Student: Aastika Banstola
Course: CSCI-411-01
Date: October 22, 2025
Project Title: A Cognitive Twin for Human Memory

Introduction
This project aims at developing an intelligent system that assists individuals in making decisions from knowledge by presenting them with personal insights. The project proposal offered a plan of fulfilling this requirement through developing a web application with machine learning approaches. The primary goals established at the beginning were to get all the essential features (like the recommendation engine, user account system, and analytics dashboard) implemented and provide a functional prototype at the end of the semester. This progress report wraps up the work accomplished since the midterm checkpoint, assesses our progress so far compared with the original plan, and presents the plan for the remainder of the semester.

Prototype Development
Architecture (Enhanced)
Since the midterm checkpoint, I have enhanced the operational pipeline with several key improvements:

Note Ingestion: Text or markdown notes get cleaned and decomposed into analysis units. 
Semantic Embedding: Single notes get translated into dense 384-dimensional vectors using Hugging Face's SentenceTransformer (all-MiniLM-L6-v2). Semantically related ideas group together even without sharing common keywords. **NEW**: Embeddings are now cached on disk using content-based hashing to avoid recomputation on unchanged notes, significantly reducing processing time for repeated operations.
Vector Search and Indexing: FAISS (Facebook AI Similarity Search) indexes the embeddings to support fast nearest-neighbor lookup to identify similar notes immediately. 
Summarization (Hugging Face BART): Identified pairs of retrieved notes are summarized using the facebook/bart-large-cnn model to generate concise, flowing descriptions of every relationship. **NEW**: Summarization now includes:
  - Citation enforcement: Each summary includes direct quotes from source notes with line references
  - NLI verification: Automatic natural language inference checking using facebook/bart-large-mnli to verify that summary sentences are entailed by source texts
  - Groundedness scoring: Each summary displays a faithfulness score indicating the percentage of sentences that are entailed by the sources
API Server: **NEW**: A FastAPI-based REST API server exposes endpoints for processing notes, searching, and retrieving connections. This enables integration with lightweight GUIs (e.g., Streamlit) or note-app add-ons (e.g., Obsidian via HTTP requests).
CLI Prototype: The command-line interface has been enhanced to support the new API server mode, allowing users to run `cognitive-twin serve` to start a local server for GUI integration.

Implementation Details
The following enhancements have been implemented:

**Caching System**: The `NoteEmbedder` class now implements on-disk caching for embeddings. Each text is hashed using SHA-256 (including model name in the hash to ensure cache invalidation on model changes), and embeddings are stored as `.npy` files in a `.emb_cache` directory. On subsequent runs, cached embeddings are loaded, dramatically reducing processing time for unchanged notes.

**Citation-Enforced Summarization**: The `NoteSummarizer` class has been enhanced with:
  - `summarize_with_citations()` method that extracts key citation spans from source texts and appends them to summaries
  - Automatic extraction of the most informative sentences from each source note for citation display
  - Integration with BART-MNLI for natural language inference verification
  - Groundedness scoring that calculates the percentage of summary sentences that are entailed by source texts

**API Server**: A new `api.py` module provides FastAPI endpoints:
  - `POST /process`: Process notes from a specified directory
  - `POST /search`: Search for similar notes given a query text
  - `POST /connections`: Retrieve daily connections digest
  - `GET /digest`: Convenience endpoint that returns markdown-formatted digest for easy display in GUIs

**Enhanced Core Pipeline**: The `CognitiveTwin` class now integrates all enhancements:
  - Uses cached embeddings via `NoteEmbedder`
  - Generates grounded summaries with citations via `NoteSummarizer.summarize_with_citations()`
  - Returns structured connection data suitable for GUI/API consumption

Challenges and Learnings
Latency: With caching implemented, embedding computation is now significantly faster for repeated operations. However, BART summarization still takes longer on CPU; future work will explore model distillation (e.g., distilbart-cnn-12-6) or ONNX quantization for further acceleration.
Faithfulness: The NLI verification system provides automatic groundedness scoring, helping identify summaries that may contain hallucinations. Direct quotes in citations improve traceability. Initial testing shows that most summaries score above 70% groundedness, with lower scores typically indicating summaries that are too abstract or speculative.
Relevance Filtering: The similarity threshold mechanism continues to filter out irrelevant connections. The NLI verification adds an additional layer of quality control by ensuring summaries are faithful to sources.
API Integration: The FastAPI server provides a clean interface for GUI integration, but additional work is needed to create a polished Streamlit sidebar or Obsidian plugin. The current API endpoints serve as a foundation for such integrations.

Next Steps
Complete lightweight GUI implementation: Create a Streamlit sidebar application that connects to the API server, displaying the Daily Connections Digest, semantic search interface, and "Explain this link" drill-down views. Alternatively, develop an Obsidian plugin that calls the local API and renders results in a pinned sidebar note.
Refine summarization faithfulness: Continue monitoring groundedness scores and refine prompts to reduce low-scoring summaries. Consider adding a reranker (cross-encoder) before summarization to ensure only the most relevant passages are summarized.
Perform minimal user testing: Recruit 5 participants for a 15-20 minute think-aloud study with tasks: finding a concept, acting on a suggested link, and validating a summary's evidence. Collect System Usability Scale (SUS) scores, faithfulness ratings (1-5), and comprehension ratings (1-5). Target thresholds: SUS ≥ 70, average faithfulness ≥ 4.0.
Optimize model runtime: Implement summary caching (content-hash based) for pairwise summaries. Explore model distillation (distilbart-cnn-12-6) or ONNX Runtime quantization for 1.5-3× speedups on CPU. Consider batch processing for multiple summary requests.
Production readiness: Migrate to production-grade vector store (e.g., Pinecone) for scalability. Add connection to personal information sources (Google Drive, Dropbox) for seamless note ingestion. Implement proper indexing and retrieval techniques with reranking.

Conclusion
I have enhanced the operational prototype with significant improvements: embedding caching for faster repeated operations, citation-enforced summarization with NLI verification for improved faithfulness, and a REST API server foundation for GUI/note-app integration. The system now provides grounded summaries with traceability through direct citations and automatic faithfulness scoring. The next phase focuses on completing the GUI implementation, conducting user testing to validate usability and comprehensibility, and optimizing runtime performance through caching and model distillation. These enhancements move the cognitive twin closer to being a practical, trustworthy tool for enhancing human recall and creativity.

Project Links
Github Repository: https://github.com/aastikab/MemoryMaster
Demo Video: https://drive.google.com/drive/folders/1gnA_jmdpMhQkC6PnMbUd4K1Lq021o0UU?usp=sharing

