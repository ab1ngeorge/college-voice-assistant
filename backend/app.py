# app.py - College Voice Assistant Backend
# Add these fixes at the VERY TOP before any imports

import os
import sys

# =====================================================================
# FIX 1: Disable .env loading before ANY imports to prevent ChromaDB errors
# =====================================================================
os.environ["IGNORE_DOT_ENV"] = "true"
os.environ["CHROMA_IGNORE_DOT_ENV"] = "true"
os.environ["PYDANTIC_DONT_READ_DOT_ENV"] = "true"

# =====================================================================
# FIX 2: Monkey patch dotenv to prevent reading corrupted .env files
# =====================================================================
import dotenv.main as dotenv_module

# Save original function
original_dotenv_values = dotenv_module.dotenv_values

def safe_dotenv_values(file_path, encoding="utf-8", **kwargs):
    """Safe version that returns empty dict instead of reading files"""
    try:
        # Try to read with proper encoding, but if fails, return empty
        return original_dotenv_values(file_path, encoding=encoding, **kwargs)
    except UnicodeDecodeError:
        # Return empty dict if encoding error occurs
        return {}
    except FileNotFoundError:
        # Return empty dict if file doesn't exist
        return {}
    except Exception:
        # Catch any other error
        return {}

# Apply the patch
dotenv_module.dotenv_values = safe_dotenv_values

# =====================================================================
# Now import all other packages
# =====================================================================
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import tempfile
from typing import Optional
import json
import asyncio

# Import local modules
from knowledge_base import KnowledgeBase
from rag_engine import GeminiRAGEngine
from voice_utils import VoiceProcessor
from language_detector import LanguageDetector

# =====================================================================
# Initialize FastAPI app
# =====================================================================
app = FastAPI(
    title="College Voice Assistant API",
    version="1.0.0",
    description="Bilingual (Malayalam/English/Manglish) College Voice Assistant using RAG & Google Gemini AI"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# Global instances (will be initialized in startup event)
# =====================================================================
knowledge_base = None
rag_engine = None
voice_processor = None
language_detector = None

# =====================================================================
# Pydantic models for request/response
# =====================================================================
class QueryRequest(BaseModel):
    text: str
    language: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    language: str
    audio_url: Optional[str] = None

class AddDocumentRequest(BaseModel):
    english: str
    malayalam: str
    manglish: Optional[str] = ""

class VoiceQueryResponse(BaseModel):
    text: str
    answer: str
    language: str
    audio_url: Optional[str] = None

# =====================================================================
# Startup event - initialize components
# =====================================================================
@app.on_event("startup")
async def startup_event():
    """Initialize all components when the app starts"""
    global knowledge_base, rag_engine, voice_processor, language_detector
    
    print("=" * 60)
    print("🚀 Starting College Voice Assistant Backend")
    print("=" * 60)
    
    try:
        # Initialize Knowledge Base
        print("📚 Initializing Knowledge Base...")
        knowledge_base = KnowledgeBase()
        
        # Load documents if exists
        if os.path.exists("documents.jsonl"):
            doc_count = knowledge_base.load_documents()
            print(f"   Loaded {doc_count} documents into knowledge base")
        else:
            print("   Warning: documents.jsonl not found, creating empty knowledge base")
            # Create sample documents if none exist
            sample_docs = [
                {
                    "text": "College library opens at 9 AM and closes at 6 PM. | കോളേജ് ലൈബ്രറി രാവിലെ 9 മണിക്ക് തുറക്കുകയും വൈകുന്നേരം 6 മണിക്ക് അടയ്ക്കുകയും ചെയ്യുന്നു. | college library 9 AM thurakkum 6 PM adakkum."
                },
                {
                    "text": "Hostel fee is ₹12,000 per semester. | ഹോസ്റ്റൽ ഫീസ് സെമസ്റ്ററിന് ₹12,000 ആണ്. | hostel fees semesterinu 12,000 rupees anu."
                }
            ]
            
            with open("documents.jsonl", "w", encoding="utf-8") as f:
                for doc in sample_docs:
                    f.write(json.dumps(doc) + "\n")
            
            knowledge_base.load_documents()
            print("   Created sample documents.jsonl file")
        
        # Initialize Language Detector
        print("🌍 Initializing Language Detector...")
        language_detector = LanguageDetector()
        
        # Initialize Voice Processor
        print("🎤 Initializing Voice Processor...")
        voice_processor = VoiceProcessor()
        
        # Initialize RAG Engine with Gemini AI
        print("🧠 Initializing Gemini RAG Engine...")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not gemini_api_key:
            print("   ⚠️  Warning: GEMINI_API_KEY environment variable not set")
            print("   Please set it with: $env:GEMINI_API_KEY='your-api-key'")
            # Create a dummy RAG engine for testing
            class DummyRAGEngine:
                def generate_response(self, query, contexts, language):
                    if language == "ml":
                        return "ജിമിനി എഐ സജീവമാക്കിയിട്ടില്ല. ദയവായി GEMINI_API_KEY സജ്ജമാക്കുക."
                    elif language == "manglish":
                        return "Gemini AI activate cheythittilla. Dayavaayi GEMINI_API_KEY set cheyyu."
                    else:
                        return "Gemini AI not activated. Please set GEMINI_API_KEY environment variable."
            
            rag_engine = DummyRAGEngine()
        else:
            rag_engine = GeminiRAGEngine(gemini_api_key)
            print("   ✅ Gemini RAG Engine initialized successfully")
        
        print("=" * 60)
        print("✅ College Voice Assistant initialized successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        import traceback
        traceback.print_exc()
        raise

# =====================================================================
# API Routes
# =====================================================================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "College Voice Assistant API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "API information",
            "POST /api/query": "Process text query",
            "POST /api/voice-query": "Process voice query from audio",
            "GET /api/audio/{filename}": "Get audio file",
            "POST /api/add-document": "Add new document to knowledge base",
            "GET /api/documents": "Get all documents",
            "GET /api/health": "Health check",
            "GET /api/config": "Configuration info"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "components": {
            "knowledge_base": knowledge_base is not None,
            "rag_engine": rag_engine is not None,
            "voice_processor": voice_processor is not None,
            "language_detector": language_detector is not None
        }
    }

@app.get("/api/config")
async def config_info():
    """Configuration information"""
    return {
        "gemini_api_key_set": bool(os.getenv("GEMINI_API_KEY")),
        "knowledge_base_path": "./chroma_db",
        "documents_file": "documents.jsonl",
        "supported_languages": ["en", "ml", "manglish"]
    }

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process text query from user"""
    try:
        # Detect language if not provided
        if not request.language:
            language = language_detector.detect_language(request.text)
        else:
            language = request.language
        
        print(f"🔍 Query: '{request.text}' (Language: {language})")
        
        # Search knowledge base
        contexts = knowledge_base.search(request.text)
        print(f"   Found {len(contexts)} relevant contexts")
        
        # Generate response
        answer = rag_engine.generate_response(request.text, contexts, language)
        print(f"   Generated answer: '{answer[:100]}...'")
        
        # Generate audio if needed (in production, you'd save this)
        audio_url = None
        
        return QueryResponse(
            answer=answer,
            language=language,
            audio_url=audio_url
        )
    
    except Exception as e:
        print(f"❌ Error in process_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice-query", response_model=VoiceQueryResponse)
async def process_voice_query(audio: UploadFile = File(...)):
    """Process voice query from audio file"""
    try:
        print("🎤 Processing voice query...")
        
        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(await audio.read())
            temp_path = temp_audio.name
        
        print(f"   Audio saved to: {temp_path}")
        
        # Process audio
        text, language = voice_processor.process_audio_file(temp_path)
        
        if not text:
            print("   ❌ Could not recognize speech")
            os.unlink(temp_path)
            raise HTTPException(status_code=400, detail="Could not recognize speech")
        
        print(f"   Recognized text: '{text}' (Language: {language})")
        
        # Search knowledge base
        contexts = knowledge_base.search(text)
        print(f"   Found {len(contexts)} relevant contexts")
        
        # Generate response
        answer = rag_engine.generate_response(text, contexts, language)
        print(f"   Generated answer: '{answer[:100]}...'")
        
        # Generate speech response
        audio_path = voice_processor.text_to_speech(answer, language)
        audio_filename = os.path.basename(audio_path)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return VoiceQueryResponse(
            text=text,
            answer=answer,
            language=language,
            audio_url=f"/api/audio/{audio_filename}"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in process_voice_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    """Serve audio files"""
    try:
        audio_path = os.path.join(tempfile.gettempdir(), filename)
        if os.path.exists(audio_path):
            return FileResponse(audio_path, media_type="audio/mpeg")
        else:
            raise HTTPException(status_code=404, detail="Audio file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/add-document")
async def add_document(request: AddDocumentRequest):
    """Add new document to knowledge base"""
    try:
        print(f"📄 Adding document: {request.english[:50]}...")
        
        doc_id = knowledge_base.add_document(
            request.english,
            request.malayalam,
            request.manglish
        )
        
        # Also append to JSONL file
        with open("documents.jsonl", "a", encoding="utf-8") as f:
            json_line = {
                "text": f"{request.english} | {request.malayalam} | {request.manglish}"
            }
            f.write(json.dumps(json_line) + "\n")
        
        print(f"   ✅ Document added with ID: {doc_id}")
        
        return {
            "message": "Document added successfully",
            "id": doc_id,
            "status": "success"
        }
    
    except Exception as e:
        print(f"❌ Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents(limit: int = 50, skip: int = 0):
    """Get all documents from knowledge base with pagination"""
    try:
        documents = []
        with open("documents.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines[skip:skip+limit]):
                if line.strip():
                    try:
                        doc = json.loads(line.strip())
                        documents.append({
                            "id": f"doc_{i+skip}",
                            "data": doc
                        })
                    except json.JSONDecodeError:
                        continue
        
        return {
            "documents": documents,
            "total": len(lines),
            "limit": limit,
            "skip": skip
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# WebSocket for real-time voice communication
# =====================================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """WebSocket for real-time voice processing"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(data)
                temp_path = temp_audio.name
            
            # Process audio
            text, language = voice_processor.process_audio_file(temp_path)
            
            if text:
                # Process query
                contexts = knowledge_base.search(text)
                answer = rag_engine.generate_response(text, contexts, language)
                
                # Generate audio response
                audio_path = voice_processor.text_to_speech(answer, language)
                
                # Read audio file and send
                with open(audio_path, 'rb') as audio_file:
                    audio_data = audio_file.read()
                
                # Send response
                await websocket.send_json({
                    "text": text,
                    "answer": answer,
                    "language": language,
                    "has_audio": True
                })
                
                # Send audio data
                await websocket.send_bytes(audio_data)
            
            # Clean up
            os.unlink(temp_path)
    
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

# =====================================================================
# Static files for frontend (optional)
# =====================================================================
# Uncomment if you want to serve frontend from backend
# app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

# =====================================================================
# Main entry point
# =====================================================================
if __name__ == "__main__":
    # Print startup banner
    print("\n" + "="*60)
    print("COLLEGE VOICE ASSISTANT - BACKEND SERVER")
    print("="*60)
    print("Starting server on http://localhost:8000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Check for required environment variables
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  WARNING: GEMINI_API_KEY environment variable not set!")
        print("To set it temporarily, run:")
        print('   $env:GEMINI_API_KEY="your-api-key-here"')
        print("\nThe app will run in demo mode without Gemini AI.\n")
    
    # Start the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )