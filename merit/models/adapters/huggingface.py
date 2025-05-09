# Update merit/models/adapters/huggingface.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from typing import Dict, List, Optional, Union

class HuggingFaceAdapter:
    """Adapter for Hugging Face models."""
    
    def __init__(self, api_token: str = None, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", local_path: str = None):
        """Initialize the Hugging Face adapter."""
        self.api_token = api_token
        self.model_name = model_name
        self.local_path = local_path
        self.is_local = False
        
        if local_path and os.path.exists(local_path):
            print(f"Loading model from local path: {local_path}")
            
            try:
                # Force offline mode
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                
                # Determine device - use MPS if available on Mac
                if torch.backends.mps.is_available():
                    print("Using MPS (Metal) device for Apple Silicon")
                    device = "mps"
                else:
                    print("Using CPU device (MPS not available)")
                    device = "cpu"
                    
                # Load tokenizer
                print("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_path,
                    local_files_only=True
                )
                print("Tokenizer loaded successfully")
                
                # Load model with appropriate settings for Mac
                print(f"Loading model on {device}...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    torch_dtype=torch.float16,
                    device_map=device,
                    local_files_only=True,
                    low_cpu_mem_usage=True
                )
                print("Model loaded successfully")
                self.is_local = True
                self.device = device
            except Exception as e:
                print(f"Error loading local model: {e}")
                import traceback
                traceback.print_exc()
                self.is_local = False
                
                # Fallback to API if desired
                if api_token:
                    try:
                        from huggingface_hub import InferenceClient
                        self.client = InferenceClient(token=api_token)
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
