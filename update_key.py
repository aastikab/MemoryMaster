#!/usr/bin/env python3
import os

print("=" * 60)
print("OpenAI API Key Updater")
print("=" * 60)
print()
print("Please paste your NEW OpenAI API key below.")
print("(Get it from: https://platform.openai.com/account/api-keys)")
print()
new_key = input("Enter API key: ").strip()

# Remove any quotes if user added them
new_key = new_key.strip('"').strip("'")

# Validate format
if not new_key.startswith('sk-'):
    print("⚠️  Warning: Key should start with 'sk-' or 'sk-proj-'")
    confirm = input("Continue anyway? (y/n): ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        exit(1)

# Update .env file
env_content = f"""# OpenAI API Key
OPENAI_API_KEY={new_key}

# Anthropic API Key (optional - for Claude)
# ANTHROPIC_API_KEY=your-anthropic-key-here
"""

with open('.env', 'w') as f:
    f.write(env_content)

print()
print("✅ .env file updated successfully!")
print(f"✅ Key length: {len(new_key)} characters")
print(f"✅ Key starts with: {new_key[:10]}...")
print()
print("Now restart your Streamlit app to use the new key.")
