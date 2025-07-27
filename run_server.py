#!/usr/bin/env python3
"""
Local development server startup script
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY not found in environment variables")
    print("💡 Please set your OpenAI API key in the .env file")
    sys.exit(1)

print("✅ Environment variables loaded")
print("🚀 Starting server...")

# Import and run the app
if __name__ == "__main__":
    import uvicorn
    from main import app
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
