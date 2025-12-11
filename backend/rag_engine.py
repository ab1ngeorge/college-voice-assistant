"""
rag_engine.py - Gemini RAG Engine for College Voice Assistant
Handles retrieval-augmented generation using Google Gemini AI
"""

import os
import sys
from typing import List, Dict, Optional
import google.generativeai as genai
from pydantic import BaseModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiRAGEngine:
    """
    RAG Engine using Google Gemini AI for generating responses
    based on retrieved context from knowledge base.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Gemini RAG Engine.
        
        Args:
            api_key (str, optional): Google Gemini API key. If not provided,
                                     will try to get from GEMINI_API_KEY environment variable.
        
        Raises:
            ValueError: If no API key is found.
        """
        # Get API key from environment or use provided
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            error_msg = "Gemini API key not found. Set GEMINI_API_KEY environment variable."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Configure the API
            genai.configure(api_key=self.api_key)
            logger.info("✅ Google Gemini AI configured successfully")
            
            # List available models to see what's supported
            available_models = list(genai.list_models())
            model_names = [model.name for model in available_models]
            logger.info(f"Available Gemini models: {len(available_models)} models")
            logger.debug(f"Model names: {model_names}")
            
            # Choose the appropriate model
            # Try different model names based on availability
            preferred_models = [
                "gemini-1.5-flash",  # Newer model
                "gemini-1.5-flash-001",  # Specific version
                "gemini-1.0-pro",  # Alternative
                "gemini-pro",  # Older naming
                "models/gemini-pro"  # Full path
            ]
            
            self.model_name = None
            for model in preferred_models:
                # Check if model is in available models
                for available_model in available_models:
                    if model in available_model.name:
                        self.model_name = model
                        logger.info(f"Selected model: {self.model_name}")
                        break
                if self.model_name:
                    break
            
            # If no preferred model found, use the first available one
            if not self.model_name and available_models:
                self.model_name = available_models[0].name.split('/')[-1]  # Get base name
                logger.warning(f"Using fallback model: {self.model_name}")
            
            if not self.model_name:
                logger.error("No Gemini models available")
                raise ValueError("No Gemini models available for your API key")
            
            # Try to import GenerationConfig, fall back to dict if not available
            try:
                from google.generativeai.types import GenerationConfig
                self.generation_config = GenerationConfig(
                    temperature=0.3,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=1024,
                )
                logger.info(f"Using GenerationConfig for {self.model_name}")
            except ImportError:
                # Fall back to dictionary config
                self.generation_config = {
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                }
                logger.info(f"Using dictionary config for {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini RAG Engine: {e}")
            raise
    
    def build_rag_prompt(self, query: str, contexts: List[Dict], language: str) -> str:
        """
        Build RAG prompt based on language and retrieved contexts.
        
        Args:
            query (str): User's query
            contexts (List[Dict]): Retrieved contexts from knowledge base
            language (str): Language code ('en', 'ml', or 'manglish')
        
        Returns:
            str: Formatted prompt for Gemini
        """
        
        # System prompts for different languages
        system_prompts = {
            "ml": {
                "instruction": """നിങ്ങൾ ഒരു കോളേജ് അസിസ്റ്റന്റ് ആണ്. ചുവടെ നൽകിയിരിക്കുന്ന വിവരങ്ങൾ (contexts) മാത്രം ഉപയോഗിച്ച് ഉപയോക്താവിന്റെ ചോദ്യത്തിന് ഉത്തരം നൽകുക.
നിങ്ങൾക്ക് പുറത്തുനിന്നുള്ള വിവരങ്ങൾ ഉപയോഗിക്കാൻ പാടില്ല. വിവരങ്ങൾ കണ്ടെത്താനായില്ലെങ്കിൽ, "ക്ഷമിക്കണം, ഈ വിവരം എന്റെ നോളജ് ബേസിൽ ഇല്ല" എന്ന് പറയുക.
എല്ലായ്പ്പോഴും മലയാളത്തിൽ മറുപടി നൽകുക.""",
                "context_label": "വിവരങ്ങൾ:",
                "user_label": "ഉപയോക്താവ്:"
            },
            "manglish": {
                "instruction": """You are a college assistant. Answer the user's question using ONLY the information provided in the contexts below.
Do not use any external knowledge. If information is not found in contexts, say "Sorry, this information is not in my knowledge base."
Always reply in Manglish (Malayalam written in English letters).""",
                "context_label": "Contexts:",
                "user_label": "User:"
            },
            "en": {
                "instruction": """You are a college assistant. Answer the user's question using ONLY the information provided in the contexts below.
Do not use any external knowledge. If information is not found in contexts, say "Sorry, this information is not in my knowledge base."
Always reply in English.""",
                "context_label": "Contexts:",
                "user_label": "User:"
            }
        }
        
        # Get the appropriate prompt based on language
        prompt_config = system_prompts.get(language, system_prompts["en"])
        
        # Build contexts string
        if contexts:
            contexts_text = "\n".join([
                f"{i+1}. {ctx.get('content', 'No content')}" 
                for i, ctx in enumerate(contexts[:3])  # Use top 3 contexts
            ])
        else:
            contexts_text = "No relevant information found."
        
        # Construct the full prompt
        prompt = f"""{prompt_config['instruction']}

{prompt_config['context_label']}
{contexts_text}

{prompt_config['user_label']} {query}

Assistant:"""
        
        logger.debug(f"Built prompt for language '{language}' with {len(contexts)} contexts")
        return prompt
    
    def generate_response(self, query: str, contexts: List[Dict], language: str) -> str:
        """
        Generate response using Gemini AI based on retrieved contexts.
        
        Args:
            query (str): User's query
            contexts (List[Dict]): Retrieved contexts from knowledge base
            language (str): Language code ('en', 'ml', or 'manglish')
        
        Returns:
            str: Generated response from Gemini
        """
        try:
            logger.info(f"Generating response for query: '{query[:50]}...' in language: {language}")
            
            # Build the RAG prompt
            prompt = self.build_rag_prompt(query, contexts, language)
            
            # Initialize the model
            model = genai.GenerativeModel(self.model_name)
            
            # Generate response
            logger.debug(f"Sending request to Gemini model: {self.model_name}")
            response = model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Extract the response text
            if response and hasattr(response, 'text'):
                generated_text = response.text.strip()
                logger.info(f"Successfully generated response: '{generated_text[:100]}...'")
                return generated_text
            else:
                error_msg = "Gemini returned empty or invalid response"
                logger.warning(error_msg)
                return self._get_fallback_response(language, error_msg)
                
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            
            # Check if it's a model not found error
            if "not found for API version" in str(e) or "is not supported" in str(e):
                logger.error(f"Model '{self.model_name}' is not available. Available models:")
                try:
                    available_models = list(genai.list_models())
                    for model in available_models:
                        logger.error(f"  - {model.name}")
                except:
                    logger.error("  Could not list available models")
            
            return self._get_fallback_response(language, error_msg)
    
    def _get_fallback_response(self, language: str, error_detail: str = "") -> str:
        """
        Get a fallback response when Gemini fails.
        
        Args:
            language (str): Language code
            error_detail (str): Error details for logging
        
        Returns:
            str: Fallback response in appropriate language
        """
        fallback_responses = {
            "ml": "ക്ഷമിക്കണം, പ്രതികരണം സൃഷ്ടിക്കുന്നതിൽ പിശക് സംഭവിച്ചു. വീണ്ടും ശ്രമിക്കുക.",
            "manglish": "Sorry, response create cheyyunathil error sambhavichu. Veendum sramikkuka.",
            "en": "Sorry, there was an error generating the response. Please try again."
        }
        
        response = fallback_responses.get(language, fallback_responses["en"])
        
        if error_detail:
            logger.error(f"Fallback triggered. Original error: {error_detail}")
        
        return response
    
    def stream_response(self, query: str, contexts: List[Dict], language: str):
        """
        Stream response from Gemini (for real-time applications).
        
        Args:
            query (str): User's query
            contexts (List[Dict]): Retrieved contexts
            language (str): Language code
        
        Yields:
            str: Chunks of the generated response
        """
        try:
            prompt = self.build_rag_prompt(query, contexts, language)
            model = genai.GenerativeModel(self.model_name)
            
            # Stream the response
            response_stream = model.generate_content(
                prompt,
                generation_config=self.generation_config,
                stream=True
            )
            
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield self._get_fallback_response(language, str(e))


class DummyRAGEngine:
    """
    Dummy RAG Engine for testing when Gemini API is not available.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize dummy engine (no API key needed)."""
        logger.warning("Using DummyRAGEngine - Gemini API not configured")
        self.model_name = "dummy-model"
    
    def build_rag_prompt(self, query: str, contexts: List[Dict], language: str) -> str:
        """Build dummy prompt."""
        return f"Dummy prompt for: {query}"
    
    def generate_response(self, query: str, contexts: List[Dict], language: str) -> str:
        """Generate dummy response."""
        responses = {
            "ml": "ഇത് ഒരു പരീക്ഷണ മോഡ് ആണ്. ദയവായി നിങ്ങളുടെ Gemini API കീ സജ്ജമാക്കുക.",
            "manglish": "Ithu oru test mode aanu. Dayavaayi ningalude Gemini API key set cheyyu.",
            "en": "This is a test mode. Please configure your Gemini API key."
        }
        return responses.get(language, responses["en"])
    
    def stream_response(self, query: str, contexts: List[Dict], language: str):
        """Stream dummy response."""
        yield self.generate_response(query, contexts, language)


def create_rag_engine(api_key: Optional[str] = None) -> GeminiRAGEngine:
    """
    Factory function to create RAG engine with proper error handling.
    
    Args:
        api_key (str, optional): Gemini API key
    
    Returns:
        GeminiRAGEngine or DummyRAGEngine: RAG engine instance
    """
    try:
        # Check if API key is available
        if not api_key and not os.getenv("GEMINI_API_KEY"):
            logger.warning("No Gemini API key found. Creating DummyRAGEngine.")
            return DummyRAGEngine()
        
        # Try to create the real Gemini engine
        return GeminiRAGEngine(api_key)
        
    except Exception as e:
        logger.error(f"Failed to create GeminiRAGEngine: {e}")
        logger.warning("Falling back to DummyRAGEngine")
        return DummyRAGEngine()


# Helper function to list available models
def list_available_models(api_key: str = None):
    """List all available Gemini models."""
    try:
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            print("No API key provided. Set GEMINI_API_KEY environment variable.")
            return
        
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
        
        print(f"📋 Available Gemini Models ({len(models)} total):")
        print("=" * 80)
        
        for i, model in enumerate(models, 1):
            print(f"{i:2}. {model.name}")
            if hasattr(model, 'description'):
                print(f"    Description: {model.description}")
            if hasattr(model, 'supported_generation_methods'):
                methods = model.supported_generation_methods
                if methods:
                    print(f"    Methods: {', '.join(methods)}")
            print()
        
        return models
        
    except Exception as e:
        print(f"Error listing models: {e}")
        return None


# Test function
def test_rag_engine():
    """Test the RAG engine functionality."""
    print("Testing RAG Engine...")
    
    # Test with a sample API key (or dummy if not available)
    test_key = os.getenv("GEMINI_API_KEY", "test-key")
    
    try:
        # Create engine
        engine = create_rag_engine(test_key)
        print(f"Created RAG Engine: {engine.__class__.__name__}")
        
        # If it's a real Gemini engine, print model info
        if isinstance(engine, GeminiRAGEngine):
            print(f"Using model: {engine.model_name}")
        
        # Test contexts
        test_contexts = [
            {"content": "The college library is open from 9 AM to 6 PM."},
            {"content": "Hostel fee is ₹12,000 per semester."}
        ]
        
        # Test query
        test_query = "What are the library timings?"
        
        # Generate response
        response = engine.generate_response(test_query, test_contexts, "en")
        print(f"Test Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    # Check if user wants to list models
    if len(sys.argv) > 1 and sys.argv[1] == "list-models":
        list_available_models()
    else:
        # Run test
        success = test_rag_engine()
        if success:
            print("✅ RAG Engine test completed successfully!")
        else:
            print("❌ RAG Engine test failed.")
        
        # Offer to list models
        print("\nTo see available models, run:")
        print("  python rag_engine.py list-models")