# fix_chromadb.py
import os
import sys

# 1. Disable .env loading BEFORE importing chromadb
os.environ["IGNORE_DOT_ENV"] = "true"
os.environ["CHROMA_IGNORE_DOT_ENV"] = "true"
os.environ["PYDANTIC_DONT_READ_DOT_ENV"] = "true"

# 2. Monkey patch dotenv to prevent reading files
import dotenv.main as dotenv_module
original_dotenv_values = dotenv_module.dotenv_values

def safe_dotenv_values(*args, **kwargs):
    # Return empty dict instead of reading files
    return {}

dotenv_module.dotenv_values = safe_dotenv_values

# 3. Now import and run your app
try:
    from app import app
    import uvicorn
    
    print("✅ Successfully patched ChromaDB!")
    print("Starting server...")
    
    if __name__ == "__main__":
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()