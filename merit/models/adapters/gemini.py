"""
Adapter for Google Gemini models.
"""
import google.generativeai as genai
import os
import time
from typing import Dict, List, Optional, Union

class GeminiAdapter:
    """Adapter for Google Gemini API."""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash-001"):
        """Initialize the Gemini adapter.
        
        Args:
            api_key: Google Gemini API key
            model_name: Name of the model to use (gemini-2.0-flash-001, gemini-2.0-flash-lite-001, etc.)
        """
        self.model_name = model_name
        
        # Set API key and configure
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            genai.configure(api_key=api_key)
        elif os.environ.get("GOOGLE_API_KEY"):
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        else:
            raise ValueError("API key must be provided either directly or via GOOGLE_API_KEY environment variable")
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(self.model_name)
            print(f"Initialized Gemini model: {model_name}")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7) -> str:
        """Generate text from a prompt."""
        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_length,
                temperature=temperature,
            )
            
            # Generate content
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract the text from the response
            return response.text
        except Exception as e:
            print(f"Error generating text with Gemini: {e}")
            return f"Error: {str(e)}"
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text."""
        # Gemini models don't directly provide embeddings
        # This is a placeholder
        return []
