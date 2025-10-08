# Cognitive Twin - A Personal Knowledge Assistant

This project implements an intelligent system that helps individuals discover connections in their personal notes using the Hugging Face API for advanced natural language processing. It processes markdown notes, finds semantic relationships between them, and generates insightful summaries of these connections.

## Features

- **Note Processing**: Cleanly processes markdown notes
- **Semantic Understanding**: Uses Hugging Face's state-of-the-art language models
- **Smart Connections**: Discovers meaningful relationships between notes using vector similarity
- **Intelligent Summaries**: Generates human-readable summaries of note relationships

## Technical Stack

- **Embedding**: Hugging Face API (all-MiniLM-L6-v2)
- **Summarization**: Hugging Face API (facebook/bart-large-cnn)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Interface**: Command-line interface using Click

## Prerequisites

1. **Hugging Face API Token**:
   - Create an account at [Hugging Face](https://huggingface.co)
   - Get your API token from [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Create a `.env` file in the project root and add your token:
     ```
     HF_API_TOKEN=your_token_here
     ```

## Installation

1. Clone the repository:
   ```bash
   git clone [your-repo-url]
   cd memory
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment:
   ```bash
   cp .env.example .env
   # Edit .env and add your Hugging Face API token
   ```

## Usage

1. **Process Notes**:
   ```bash
   python -m cognitive_twin.cli process /path/to/notes
   ```
   This command processes all markdown files in the specified directory.

2. **Find Connections**:
   ```bash
   python -m cognitive_twin.cli connections
   ```
   This command analyzes processed notes and generates a digest of connections found.

## Note Format

- Notes should be in markdown format (*.md)
- Each note should be a separate file
- Notes can contain any text content

## Example Output

```
Daily Connections Digest:

1. Connection between "Sleep and Memory" and "Cognitive Performance":
   "Research shows that adequate sleep is crucial for memory consolidation and cognitive function, with studies indicating 7-9 hours as optimal for learning and retention."

2. Connection between "Learning Techniques" and "Study Habits":
   "Active recall and spaced repetition demonstrate significant benefits for long-term memory formation and academic performance."
```

## API Usage

The project uses the following Hugging Face models via API:
1. sentence-transformers/all-MiniLM-L6-v2 for embeddings
2. facebook/bart-large-cnn for summarization

Make sure you have sufficient API quota for your usage needs.

## Future Enhancements

- Web interface for easier interaction
- Real-time note processing
- Enhanced summarization quality
- User feedback integration
- Performance optimizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.