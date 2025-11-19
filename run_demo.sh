#!/bin/bash

# Cognitive Twin - Quick Start Script

echo "🧠 Cognitive Twin - Starting Demo..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check for API keys (optional)
echo ""
echo "🔑 Checking API keys..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  No API keys found. Chatbot features will be disabled."
    echo "   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable chatbot."
else
    echo "✅ API keys found. Chatbot features enabled."
fi

echo ""
echo "🚀 Starting Streamlit app..."
echo "   The app will open in your browser at http://localhost:8501"
echo ""
echo "📝 Quick Start:"
echo "   1. Load notes from 'sample_notes' directory"
echo "   2. Write or upload a note to find connections"
echo "   3. Explore all the new features!"
echo ""

# Run Streamlit
streamlit run app.py

