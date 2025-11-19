#!/bin/bash
echo "🔑 OpenAI API Key Updater"
echo ""
echo "Please enter your OpenAI API key:"
read -s API_KEY
echo ""
echo "Updating .env file..."
cat > .env << ENVFILE
# OpenAI API Key
OPENAI_API_KEY=$OPENAI_API_KEY

# Anthropic API Key (optional - for Claude)
# ANTHROPIC_API_KEY=your-anthropic-key-here
ENVFILE
echo "✅ .env file updated!"
echo ""
echo "Verifying key format..."
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); key = os.getenv('OPENAI_API_KEY'); print(f'Key loaded: {bool(key)}'); print(f'Key starts with sk-: {key.startswith(\"sk-\") if key else False}'); print(f'Key length: {len(key) if key else 0}')"
