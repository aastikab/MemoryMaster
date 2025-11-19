#!/bin/bash
echo "🔄 Restarting Streamlit app..."
echo ""

# Kill any existing Streamlit processes
pkill -f "streamlit run" 2>/dev/null
sleep 1

# Check if port 8501 is still in use
if lsof -ti:8501 > /dev/null 2>&1; then
    echo "⚠️  Port 8501 is still in use. Killing processes on that port..."
    lsof -ti:8501 | xargs kill -9 2>/dev/null
    sleep 1
fi

echo "✅ Old processes stopped"
echo ""
echo "🚀 Starting Streamlit app..."
echo "   The app will open at: http://localhost:8501"
echo ""

# Start Streamlit
streamlit run app.py
