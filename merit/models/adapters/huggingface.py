"""
Enhanced Hugging Face adapter integrated with local models system.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from typing import Dict, List, Optional, Union
from ..local_models import DeviceManager

class HuggingFaceAdapter:
    """Enhanced adapter for Hugging Face models with better device management."""
    
    def __init__(self, api_token: str = None, model_name: str = "microsoft/DialoGPT-medium", 
                 local_path: str = None, cache_dir: str = None):
        """Initialize the enhanced Hugging Face adapter."""
        self.api_token = api_token
        self.model_name = model_name
        self.local_path = local_path
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/merit_models")
        self.is_local = False
        self.device = DeviceManager.get_optimal_device()
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        
        print(f"Initializing HuggingFace adapter for {model_name} on {self.device}")
        
        if local_path and os.path.exists(local_path):
            self._load_local_model(local_path)
        else:
            self._load_model_from_hub()
    
    def _load_local_model(self, local_path: str):
        """Load model from local path"""
        print(f"Loading model from local path: {local_path}")
        
        try:
            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_path,
                local_files_only=True,
                cache_dir=self.cache_dir
            )
            
            # Set pad token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Tokenizer loaded successfully")
            
            # Get optimal model configuration
            model_config = DeviceManager.get_model_config("auto", self.device)
            
            # Load model
            print(f"Loading model on {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path,
                local_files_only=True,
                cache_dir=self.cache_dir,
                **model_config
            )
            
            print("✓ Local model loaded successfully")
            self.is_local = True
            
        except Exception as e:
            print(f"✗ Error loading local model: {e}")
            self.is_local = False
            
            # Fallback to hub
            if self.api_token:
                self._fallback_to_api()
    
    def _load_model_from_hub(self):
        """Load model from Hugging Face Hub"""
        try:
            print(f"Loading {self.model_name} from Hugging Face Hub...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Set pad token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Get optimal model configuration
            model_config = DeviceManager.get_model_config("auto", self.device)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                **model_config
            )
            
            print("✓ Model loaded from hub successfully")
            self.is_local = True
            
        except Exception as e:
            print(f"✗ Error loading model from hub: {e}")
            self.is_local = False
            
            if self.api_token:
                self._fallback_to_api()
    
    def _fallback_to_api(self):
        """Fallback to Hugging Face API"""
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_token)
            print("Falling back to Hugging Face API")
        except Exception as api_err:
            print(f"Error connecting to Hugging Face API: {api_err}")
                
    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7) -> str:
        """Generate text from a prompt."""
        try:
            if self.is_local:
                # Format as chat for Llama models
                if "llama" in self.model_name.lower():
                    messages = [{"role": "user", "content": prompt}]
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
                
                # Generate with local model
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # Generate with reduced memory settings
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=max_length,
                        temperature=temperature if temperature > 0 else 0.01,
                        do_sample=temperature > 0,
                    )
                
                # Decode the output
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # For chat models, extract just the assistant's response
                if "llama" in self.model_name.lower() and prompt in generated_text:
                    return generated_text[len(prompt):].strip()
                
                return generated_text
            elif hasattr(self, 'client'):
                # Use API if available as fallback
                if "llama" in self.model_name.lower():
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat_completion(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_length,
                        temperature=temperature
                    )
                    return response.choices[0].message.content
                else:
                    response = self.client.text_generation(
                        prompt,
                        model=self.model_name,
                        max_new_tokens=max_length,
                        temperature=temperature,
                        do_sample=True
                    )
                    return response
            else:
                return "No model available (local model failed to load and no API fallback configured)"
        except Exception as e:
            print(f"Error generating text: {e}")
            return f"Error during generation: {str(e)}"
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text."""
        return []  # Simplified implementation
