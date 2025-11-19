# Cognitive Twin - Demo Guide

This guide will help you run and test all the new features of the Cognitive Twin application.

## Prerequisites

1. **Python 3.8+** installed
2. **Virtual environment** (recommended)

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

**Note:** Some dependencies are optional:
- **PDF Export**: Requires `reportlab` (included)
- **Visualizations**: Requires `plotly` and `networkx` (included)
- **Chatbot**: Requires `openai` or `anthropic` (included, but needs API keys)

## Step 2: Set Up API Keys (Optional - for Chatbot)

The chatbot features require API keys. You can skip this if you only want to test other features.

### For OpenAI:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### For Claude (Anthropic):
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
```

**Or create a `.env` file:**
```
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

## Step 3: Run the Application

```bash
# Make sure you're in the project root directory
streamlit run app.py
```

The app will open in your browser at `http://localhost:8500`

## Step 4: Testing Each Feature

### 1. Load Knowledge Base
1. In the sidebar, you'll see "Notes Directory Path"
2. Enter `sample_notes` (or path to your notes folder)
3. Click "🔄 Load Notes"
4. You should see: "✅ Loaded X notes!"

### 2. Test Clickable Citations
1. Go to **"✍️ Write Note"** tab
2. Enter some text related to your notes (e.g., "memory and learning techniques")
3. Click "🔍 Find Connections"
4. When results appear, look for citation buttons (📎)
5. Click a citation button to see the note preview

**What to check:**
- ✅ Citations appear as clickable buttons
- ✅ Clicking shows full note preview
- ✅ Preview can be closed

### 3. Test Export Functionality
1. After finding connections, you'll see two download buttons:
   - **📥 Export to Markdown**
   - **📥 Export to PDF**
2. Click each button to download
3. Open the downloaded files to verify

**What to check:**
- ✅ Markdown file downloads and opens correctly
- ✅ PDF file downloads (if reportlab is installed)
- ✅ Both contain summary information

### 4. Test Batch Processing
1. Go to **"📦 Batch Process"** tab
2. Click "Choose multiple markdown or text files"
3. Select multiple `.md` or `.txt` files
4. Click "🔄 Process Batch"
5. Review the batch analysis results

**What to check:**
- ✅ Multiple files can be uploaded
- ✅ Batch processing completes successfully
- ✅ Results show connections for each file

### 5. Test Visualizations
1. Go to **"📊 Visualizations"** tab
2. You should see:
   - **Network Graph**: Interactive graph showing note relationships
   - **Similarity Heatmap**: Color-coded similarity matrix

**What to check:**
- ✅ Network graph displays with nodes and edges
- ✅ Hovering over nodes shows note previews
- ✅ Heatmap shows similarity scores
- ✅ Both visualizations are interactive

### 6. Test Chatbot (Optional)
1. In the sidebar, under "🤖 Chatbot":
   - Select provider (OpenAI or Claude)
   - Check "Use local summaries only" for privacy
   - Click "🔧 Initialize Chatbot"
2. Go to **"💬 Chatbot"** tab
3. Type a question like: "What are the main themes in my notes?"
4. Click "💬 Send"

**What to check:**
- ✅ Chatbot initializes (if API keys are set)
- ✅ Chat history displays
- ✅ Responses are contextual (uses current session data)
- ✅ Privacy mode works (local summaries only)

### 7. Test Upload Note
1. Go to **"📤 Upload Note"** tab
2. Upload a single markdown or text file
3. Click "🔍 Find Connections for Uploaded File"
4. Verify all features work (citations, export, etc.)

## Quick Test Checklist

- [ ] App starts without errors
- [ ] Notes load successfully
- [ ] Connections are found
- [ ] Citations are clickable
- [ ] Citation preview works
- [ ] Markdown export works
- [ ] PDF export works (or shows warning if reportlab unavailable)
- [ ] Batch processing works
- [ ] Network graph displays
- [ ] Similarity heatmap displays
- [ ] Chatbot initializes (if API keys set)
- [ ] Chatbot responds with context

## Troubleshooting

### Issue: "Module not found" errors
**Solution:** Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: PDF export shows warning
**Solution:** Install reportlab:
```bash
pip install reportlab
```

### Issue: Visualizations don't show
**Solution:** Install plotly:
```bash
pip install plotly networkx
```

### Issue: Chatbot not available
**Solution:** 
- Set API keys in environment variables
- Or skip chatbot features (they're optional)

### Issue: "No notes found"
**Solution:**
- Check that `sample_notes` directory exists
- Or provide correct path to your notes directory
- Ensure notes are `.md` files

### Issue: App is slow
**Solution:**
- First run downloads models (this is normal, takes a few minutes)
- Subsequent runs use cached models
- Processing many notes may take time

## Sample Test Data

The `sample_notes/` directory contains sample markdown files you can use for testing:
- `cognitive_enhancement.md`
- `learning_techniques.md`
- `memory_research.md`
- `sleep_patterns.md`

## Expected Behavior

1. **First Run**: 
   - Models download (SentenceTransformer, BART, etc.)
   - This may take 5-10 minutes
   - Progress bars will show

2. **Subsequent Runs**:
   - Much faster (uses cached models)
   - Embeddings are cached in `.emb_cache/` directory

3. **Finding Connections**:
   - Takes a few seconds per query
   - Shows similarity scores
   - Generates summaries with citations

## Next Steps

Once everything works:
1. Add your own notes to `sample_notes/` or create a new directory
2. Experiment with different similarity thresholds
3. Try batch processing with multiple files
4. Explore the visualizations
5. Chat with the AI about your notes (if API keys are set)

## Support

If you encounter issues:
1. Check the terminal/console for error messages
2. Verify all dependencies are installed
3. Check that API keys are set correctly (for chatbot)
4. Ensure note files are valid markdown/text files

