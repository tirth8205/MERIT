"""
Local model adapters for MERIT that work on MacBook Pro M4 (CPU and MPS).
Supports various 7B and smaller models that can run locally.
"""
import os
import torch
from typing import Dict, List, Optional, Union, Any
import json
from pathlib import Path
import time
import gc

try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        BitsAndBytesConfig,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("Warning: accelerate not available. Install with: pip install accelerate")

try:
    import requests
    import subprocess
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: requests not available for Ollama integration")

try:
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Info: bitsandbytes not available. 4-bit quantization disabled.")


class DeviceManager:
    """Enhanced device management for optimal performance on different hardware"""
    
    @staticmethod
    def get_optimal_device():
        """Get the best available device for model inference"""
        if torch.backends.mps.is_available():
            print("Using MPS (Metal Performance Shaders) for Apple Silicon")
            return "mps"
        elif torch.cuda.is_available():
            print("Using CUDA GPU")
            return "cuda"
        else:
            print("Using CPU (no GPU acceleration available)")
            return "cpu"
    
    @staticmethod
    def get_memory_info():
        """Get available memory information with actual system detection"""
        try:
            import psutil
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024**3)
            available_gb = vm.available / (1024**3)
            used_percent = vm.percent
        except ImportError:
            # Fallback if psutil not available
            total_gb = 8.0
            available_gb = 4.0
            used_percent = 50.0

        if torch.backends.mps.is_available():
            # MPS uses unified memory - get actual system RAM
            mps_allocated = 0
            try:
                # Try to get MPS allocated memory (PyTorch 2.0+)
                mps_allocated = torch.mps.current_allocated_memory() / (1024**3)
            except (AttributeError, RuntimeError):
                pass

            return {
                "device": "mps",
                "total_memory_gb": total_gb,
                "available_memory_gb": available_gb,
                "mps_allocated_gb": mps_allocated,
                "memory_used_percent": used_percent,
                "unified_memory": True,
                "warning": "high_memory" if used_percent > 80 else None
            }
        elif torch.cuda.is_available():
            return {
                "device": "cuda",
                "total_memory": torch.cuda.get_device_properties(0).total_memory,
                "allocated_memory": torch.cuda.memory_allocated(),
                "cached_memory": torch.cuda.memory_reserved(),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "system_total_gb": total_gb
            }
        else:
            return {
                "device": "cpu",
                "available_memory_gb": available_gb,
                "total_memory_gb": total_gb,
                "memory_used_percent": used_percent
            }

    @staticmethod
    def check_memory_pressure() -> bool:
        """Check if system is under memory pressure (>80% used). Returns True if OK."""
        info = DeviceManager.get_memory_info()
        used_percent = info.get("memory_used_percent", 50)
        if used_percent > 80:
            print(f"WARNING: High memory usage ({used_percent:.1f}%). Consider unloading models.")
            return False
        return True
    
    @staticmethod
    def get_model_config(model_size: str, device: str):
        """Get optimal model configuration for device"""
        base_config = {
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True
        }
        
        if device == "mps":
            # MPS-specific optimizations
            base_config.update({
                "device_map": "mps",
                "max_memory": {0: "6GiB"},  # Conservative for unified memory
            })
        elif device == "cuda":
            base_config.update({
                "device_map": "auto",
                "max_memory": {0: "6GiB"}
            })
        else:
            # CPU optimizations
            base_config.update({
                "torch_dtype": torch.float32,
                "device_map": {"": "cpu"}
            })
        
        return base_config


class LocalModelAdapter:
    """Base class for local model adapters"""
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/merit_models")
        self.device = DeviceManager.get_optimal_device()
        self.model = None
        self.tokenizer = None
        self.generation_config = {}
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for local models")
    
    def load_model(self):
        """Load the model and tokenizer"""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate text from prompt"""
        raise NotImplementedError("Subclasses must implement generate")
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Force garbage collection first
        gc.collect()

        # Clear GPU/MPS memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            # Clear MPS cache (PyTorch 2.0+)
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass  # Older PyTorch version

        print(f"Model {self.model_name} unloaded from memory")


class Llama2ChatAdapter(LocalModelAdapter):
    """Adapter for Llama-2-7B-Chat model"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            cache_dir=cache_dir
        )
        
        # Llama-2 specific configuration
        self.chat_template = True
        self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
    
    def load_model(self):
        """Load Llama-2-7B-Chat model"""
        print(f"Loading {self.model_name} on {self.device}...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Get model configuration
            model_config = DeviceManager.get_model_config("7b", self.device)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                **model_config
            )
            
            print(f"✓ {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading {self.model_name}: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate response using Llama-2 chat format"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        try:
            # Format prompt for Llama-2 chat
            formatted_prompt = self._format_chat_prompt(prompt)
            
            # Encode input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                max_length=2048,
                truncation=True
            )
            
            # Move to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    **kwargs
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            response = self._extract_assistant_response(generated_text, formatted_prompt)
            
            return response
            
        except Exception as e:
            print(f"Error generating with {self.model_name}: {e}")
            return f"Error: {str(e)}"
    
    def _format_chat_prompt(self, prompt: str) -> str:
        """Format prompt for Llama-2 chat template"""
        formatted = f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{prompt} [/INST]"
        return formatted
    
    def _extract_assistant_response(self, full_text: str, prompt: str) -> str:
        """Extract assistant response from full generated text"""
        # Remove the prompt part
        if prompt in full_text:
            response = full_text[len(prompt):].strip()
        else:
            response = full_text.strip()
        
        # Clean up any special tokens that might remain
        response = response.replace("</s>", "").strip()
        
        return response


class MistralInstructAdapter(LocalModelAdapter):  
    """Adapter for Mistral-7B-Instruct model"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            model_name="mistralai/Mistral-7B-Instruct-v0.1",
            cache_dir=cache_dir
        )
    
    def load_model(self):
        """Load Mistral-7B-Instruct model"""
        print(f"Loading {self.model_name} on {self.device}...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Load model with appropriate configuration
            model_config = DeviceManager.get_model_config("7b", self.device)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                **model_config
            )
            
            print(f"✓ {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading {self.model_name}: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate response using Mistral instruct format"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        try:
            # Format prompt for Mistral instruct
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Encode and generate
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    **kwargs
                )
            
            # Decode and clean response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(formatted_prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating with {self.model_name}: {e}")
            return f"Error: {str(e)}"


class TinyLlamaAdapter(LocalModelAdapter):
    """Adapter for TinyLlama-1.1B model (very lightweight)"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            cache_dir=cache_dir
        )
    
    def load_model(self):
        """Load TinyLlama model"""
        print(f"Loading {self.model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Lighter configuration for smaller model
            model_config = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            if self.device != "cpu":
                model_config["device_map"] = self.device
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                **model_config
            )
            
            print(f"✓ {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading {self.model_name}: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate response using TinyLlama"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        try:
            # Simple prompt format
            formatted_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    **kwargs
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(formatted_prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating with {self.model_name}: {e}")
            return f"Error: {str(e)}"


class Phi2Adapter(LocalModelAdapter):
    """Adapter for Microsoft Phi-2 model (small but capable instruction model)"""

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            model_name="microsoft/phi-2",
            cache_dir=cache_dir
        )

    def load_model(self):
        """Load Phi-2 model"""
        print(f"Loading {self.model_name} on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model_config = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }

            if self.device != "cpu":
                model_config["device_map"] = self.device

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                **model_config
            )

            print(f"✓ {self.model_name} loaded successfully")

        except Exception as e:
            print(f"✗ Error loading {self.model_name}: {e}")
            raise

    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate response using Phi-2"""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        try:
            # Phi-2 instruction format
            formatted_prompt = f"Instruct: {prompt}\nOutput:"

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)

            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    **kwargs
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(formatted_prompt):].strip()

            return response

        except Exception as e:
            print(f"Error generating with {self.model_name}: {e}")
            return f"Error: {str(e)}"


class Qwen2Adapter(LocalModelAdapter):
    """Adapter for Qwen2-0.5B-Instruct model (very small instruction model)"""

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            model_name="Qwen/Qwen2-0.5B-Instruct",
            cache_dir=cache_dir
        )

    def load_model(self):
        """Load Qwen2 model"""
        print(f"Loading {self.model_name} on {self.device}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )

            model_config = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }

            if self.device != "cpu":
                model_config["device_map"] = self.device

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                **model_config
            )

            print(f"✓ {self.model_name} loaded successfully")

        except Exception as e:
            print(f"✗ Error loading {self.model_name}: {e}")
            raise

    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate response using Qwen2"""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        try:
            # Qwen2 chat format
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)

            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    **kwargs
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the assistant response
            if "assistant" in generated_text.lower():
                response = generated_text.split("assistant")[-1].strip()
            else:
                response = generated_text[len(formatted_prompt):].strip()

            return response

        except Exception as e:
            print(f"Error generating with {self.model_name}: {e}")
            return f"Error: {str(e)}"


class ModelManager:
    """Manages multiple local models for comparative evaluation"""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.available_models = {
            # All models are instruction-tuned and suitable for reasoning
            "llama2-7b-chat": Llama2ChatAdapter,
            "mistral-7b-instruct": MistralInstructAdapter,
            "tinyllama-1b": TinyLlamaAdapter,
            "phi-2": Phi2Adapter,
            "qwen2-0.5b": Qwen2Adapter,
        }
        self.loaded_models = {}

        # Model metadata
        self.model_info = {
            "llama2-7b-chat": {
                "parameters": "7B",
                "type": "Chat-tuned",
                "memory_requirement": "~14GB",
                "license": "Custom (Meta)",
                "description": "Meta's Llama-2 7B chat model"
            },
            "mistral-7b-instruct": {
                "parameters": "7B",
                "type": "Instruction-tuned",
                "memory_requirement": "~14GB",
                "license": "Apache 2.0",
                "description": "Mistral AI's 7B instruction model"
            },
            "tinyllama-1b": {
                "parameters": "1.1B",
                "type": "Chat-tuned",
                "memory_requirement": "~2GB",
                "license": "Apache 2.0",
                "description": "Lightweight TinyLlama chat model - good for quick tests"
            },
            "phi-2": {
                "parameters": "2.7B",
                "type": "Instruction-tuned",
                "memory_requirement": "~6GB",
                "license": "MIT",
                "description": "Microsoft Phi-2 - excellent reasoning for its size"
            },
            "qwen2-0.5b": {
                "parameters": "0.5B",
                "type": "Instruction-tuned",
                "memory_requirement": "~1GB",
                "license": "Apache 2.0",
                "description": "Qwen2-0.5B-Instruct - smallest instruction model"
            }
        }

    def get_recommended_models(self, memory_constraint_gb: int = 8) -> List[str]:
        """Get recommended models based on memory constraints"""
        recommended = []

        if memory_constraint_gb >= 16:
            recommended.extend(["llama2-7b-chat", "mistral-7b-instruct"])

        if memory_constraint_gb >= 6:
            recommended.append("phi-2")

        if memory_constraint_gb >= 2:
            recommended.append("tinyllama-1b")

        if memory_constraint_gb >= 1:
            recommended.append("qwen2-0.5b")

        return recommended

    def load_model(self, model_name: str) -> LocalModelAdapter:
        """Load a specific model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Options: {list(self.available_models.keys())}")

        if model_name in self.loaded_models:
            print(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]

        print(f"Loading model: {model_name}")
        adapter_class = self.available_models[model_name]
        adapter = adapter_class(self.cache_dir)
        adapter.load_model()

        self.loaded_models[model_name] = adapter
        return adapter
    
    def unload_model(self, model_name: str):
        """Unload a specific model"""
        if model_name in self.loaded_models:
            self.loaded_models[model_name].unload_model()
            del self.loaded_models[model_name]
            print(f"Unloaded model: {model_name}")
    
    def unload_all_models(self):
        """Unload all models to free memory"""
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model"""
        return self.model_info.get(model_name, {})
    
    def list_available_models(self) -> Dict[str, Dict]:
        """List all available models with their info"""
        return self.model_info
    
    def benchmark_models(self, models: List[str], test_prompts: List[str], 
                        max_length: int = 500, temperature: float = 0.7) -> Dict:
        """Benchmark multiple models on test prompts"""
        
        results = {
            "models_tested": models,
            "test_prompts": test_prompts,
            "configuration": {
                "max_length": max_length,
                "temperature": temperature
            },
            "results": {},
            "performance_metrics": {}
        }
        
        for model_name in models:
            print(f"\nBenchmarking {model_name}...")
            
            try:
                # Load model
                adapter = self.load_model(model_name)
                
                model_results = []
                total_time = 0
                
                for i, prompt in enumerate(test_prompts):
                    print(f"  Prompt {i+1}/{len(test_prompts)}")
                    
                    start_time = time.time()
                    response = adapter.generate(
                        prompt, 
                        max_length=max_length,
                        temperature=temperature
                    )
                    end_time = time.time()
                    
                    generation_time = end_time - start_time
                    total_time += generation_time
                    
                    model_results.append({
                        "prompt": prompt,
                        "response": response,
                        "generation_time": generation_time,
                        "tokens_per_second": len(response.split()) / generation_time if generation_time > 0 else 0
                    })
                
                # Calculate performance metrics
                avg_time = total_time / len(test_prompts) if test_prompts else 0
                avg_tokens_per_sec = np.mean([r["tokens_per_second"] for r in model_results])
                
                results["results"][model_name] = model_results
                results["performance_metrics"][model_name] = {
                    "average_generation_time": avg_time,
                    "total_time": total_time,
                    "average_tokens_per_second": avg_tokens_per_sec,
                    "model_info": self.get_model_info(model_name)
                }
                
                # Unload model to free memory for next one
                self.unload_model(model_name)
                
            except Exception as e:
                print(f"Error benchmarking {model_name}: {e}")
                results["results"][model_name] = {"error": str(e)}
        
        return results
    
    def create_benchmark_report(self, benchmark_results: Dict, output_file: str):
        """Create a detailed benchmark report"""
        
        report = []
        report.append("LOCAL MODELS BENCHMARK REPORT")
        report.append("=" * 35)
        report.append(f"Models tested: {len(benchmark_results['models_tested'])}")
        report.append(f"Test prompts: {len(benchmark_results['test_prompts'])}")
        report.append(f"Configuration: max_length={benchmark_results['configuration']['max_length']}, "
                     f"temperature={benchmark_results['configuration']['temperature']}")
        report.append("")
        
        # Performance summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 20)
        
        perf_metrics = benchmark_results.get("performance_metrics", {})
        
        # Sort by average generation time
        sorted_models = sorted(
            perf_metrics.items(),
            key=lambda x: x[1].get("average_generation_time", float('inf'))
        )
        
        for model_name, metrics in sorted_models:
            if "error" not in benchmark_results["results"][model_name]:
                info = metrics.get("model_info", {})
                report.append(f"{model_name.upper()} ({info.get('parameters', 'Unknown')} params):")
                report.append(f"  Average time: {metrics.get('average_generation_time', 0):.2f}s")
                report.append(f"  Tokens/second: {metrics.get('average_tokens_per_second', 0):.1f}")
                report.append(f"  Memory req: {info.get('memory_requirement', 'Unknown')}")
                report.append("")
        
        # Model details
        report.append("DETAILED RESULTS")
        report.append("-" * 16)
        
        for model_name, model_results in benchmark_results["results"].items():
            if isinstance(model_results, list):
                report.append(f"{model_name.upper()}:")
                
                for i, result in enumerate(model_results[:2]):  # Show first 2 examples
                    report.append(f"  Example {i+1}:")
                    report.append(f"    Prompt: {result['prompt'][:100]}...")
                    report.append(f"    Response: {result['response'][:150]}...")
                    report.append(f"    Time: {result['generation_time']:.2f}s")
                    report.append("")
            else:
                report.append(f"{model_name.upper()}: {model_results}")
                report.append("")
        
        # Save report  
        with open(output_file, 'w') as f:
            f.write("\n".join(report))
        
        print(f"Benchmark report saved to: {output_file}")
        
        return results


# Utility functions
def get_system_recommendations() -> Dict:
    """Get system-specific model recommendations"""
    device_info = DeviceManager.get_memory_info()
    
    recommendations = {
        "device_info": device_info,
        "recommended_models": [],
        "performance_tips": []
    }
    
    if device_info["device"] == "mps":
        # Apple Silicon recommendations
        recommendations["recommended_models"] = [
            "tinyllama-1b",  # Fastest, lowest memory
            "gpt2-medium",   # Good balance
            "mistral-7b-instruct"  # Best quality if memory allows
        ]
        recommendations["performance_tips"] = [
            "Use smaller batch sizes for better MPS performance",
            "Consider using float16 precision for memory efficiency",
            "Monitor memory usage with Activity Monitor"
        ]
    
    elif device_info["device"] == "cuda":
        # CUDA GPU recommendations
        total_memory_gb = device_info["total_memory"] / (1024**3)
        
        if total_memory_gb >= 12:
            recommendations["recommended_models"] = ["llama2-7b-chat", "mistral-7b-instruct"]
        elif total_memory_gb >= 6:
            recommendations["recommended_models"] = ["tinyllama-1b", "gpt2-large"]
        else:
            recommendations["recommended_models"] = ["gpt2-medium"]
        
        recommendations["performance_tips"] = [
            "Use CUDA for maximum performance",
            "Enable gradient checkpointing for memory efficiency",
            "Consider using quantization for larger models"
        ]
    
    else:
        # CPU recommendations
        recommendations["recommended_models"] = ["gpt2-medium", "tinyllama-1b"]
        recommendations["performance_tips"] = [
            "Use CPU-optimized models",
            "Consider using quantization",
            "Reduce sequence lengths for faster inference"
        ]
    
    return recommendations


class OllamaAdapter:
    """Adapter for Ollama models (runs via Ollama server)"""
    
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.available = self._check_ollama_availability()
        
        if not OLLAMA_AVAILABLE:
            raise ImportError("requests library required for Ollama integration")
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _check_model_availability(self) -> bool:
        """Check if the specific model is available in Ollama"""
        if not self.available:
            return False
        
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model["name"].startswith(self.model_name) for model in models)
        except:
            pass
        return False
    
    def pull_model(self) -> bool:
        """Pull model from Ollama registry"""
        if not self.available:
            print("Ollama server not available. Please start Ollama first.")
            return False
        
        print(f"Pulling {self.model_name} from Ollama...")
        
        try:
            response = requests.post(
                f"{self.host}/api/pull",
                json={"name": self.model_name},
                stream=True,
                timeout=300
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "status" in data:
                            print(f"  {data['status']}")
                        if data.get("status") == "success":
                            print(f"✓ Successfully pulled {self.model_name}")
                            return True
            else:
                print(f"✗ Failed to pull {self.model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Error pulling {self.model_name}: {e}")
            return False
    
    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using Ollama API"""
        if not self.available:
            return "Error: Ollama server not available"
        
        if not self._check_model_availability():
            print(f"Model {self.model_name} not found. Attempting to pull...")
            if not self.pull_model():
                return f"Error: Could not pull model {self.model_name}"
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_length,
                },
                "stream": False
            }
            
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def unload_model(self):
        """Unload model from Ollama (if supported)"""
        # Ollama handles model loading/unloading automatically
        pass


class CodeLlamaAdapter(LocalModelAdapter):
    """Adapter for CodeLlama models via Hugging Face"""
    
    def __init__(self, model_size: str = "7b", cache_dir: Optional[str] = None):
        model_name = f"codellama/CodeLlama-{model_size}-Instruct-hf"
        super().__init__(model_name=model_name, cache_dir=cache_dir)
        self.model_size = model_size
    
    def load_model(self):
        """Load CodeLlama model"""
        print(f"Loading {self.model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_config = DeviceManager.get_model_config(self.model_size, self.device)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                **model_config
            )
            
            print(f"✓ {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading {self.model_name}: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate code using CodeLlama"""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        try:
            # Format prompt for code instruction
            formatted_prompt = f"[INST] {prompt} [/INST]"
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(formatted_prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating with {self.model_name}: {e}")
            return f"Error: {str(e)}"


class Phi3Adapter(LocalModelAdapter):
    """Adapter for Microsoft Phi-3 models"""
    
    def __init__(self, model_size: str = "mini", cache_dir: Optional[str] = None):
        if model_size == "mini":
            model_name = "microsoft/Phi-3-mini-4k-instruct"
        elif model_size == "small":
            model_name = "microsoft/Phi-3-small-8k-instruct"
        else:
            model_name = "microsoft/Phi-3-medium-4k-instruct"
        
        super().__init__(model_name=model_name, cache_dir=cache_dir)
        self.model_size = model_size
    
    def load_model(self):
        """Load Phi-3 model"""
        print(f"Loading {self.model_name} on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            model_config = DeviceManager.get_model_config(
                "3b" if self.model_size == "mini" else "7b", 
                self.device
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                **model_config
            )
            
            print(f"✓ {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading {self.model_name}: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate response using Phi-3"""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        try:
            # Phi-3 chat format
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")

            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # NOTE: use_cache=False to avoid DynamicCache bug in transformers 4.52+
                # This is a workaround for Phi-3 models on MPS
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,  # Fix for DynamicCache bug
                    **kwargs
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(formatted_prompt):].strip()

            return response

        except Exception as e:
            print(f"Error generating with {self.model_name}: {e}")
            return f"Error: {str(e)}"


class EnhancedModelManager(ModelManager):
    """Enhanced model manager with additional models and Ollama support"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        
        # Add additional models
        self.available_models.update({
            # Hugging Face Models
            "codellama-7b": lambda cache_dir: CodeLlamaAdapter("7b", cache_dir),
            "codellama-13b": lambda cache_dir: CodeLlamaAdapter("13b", cache_dir),
            "phi3-mini": lambda cache_dir: Phi3Adapter("mini", cache_dir),
            "phi3-small": lambda cache_dir: Phi3Adapter("small", cache_dir),
            
            # Ollama Models (popular ones)
            "ollama-llama2": lambda cache_dir: OllamaAdapter("llama2"),
            "ollama-llama2-13b": lambda cache_dir: OllamaAdapter("llama2:13b"),
            "ollama-mistral": lambda cache_dir: OllamaAdapter("mistral"),
            "ollama-codellama": lambda cache_dir: OllamaAdapter("codellama"),
            "ollama-phi3": lambda cache_dir: OllamaAdapter("phi3"),
            "ollama-gemma": lambda cache_dir: OllamaAdapter("gemma"),
            "ollama-qwen": lambda cache_dir: OllamaAdapter("qwen"),
            "ollama-deepseek-coder": lambda cache_dir: OllamaAdapter("deepseek-coder"),
            "ollama-wizardcoder": lambda cache_dir: OllamaAdapter("wizardcoder"),
        })
        
        # Update model info
        self.model_info.update({
            "codellama-7b": {
                "parameters": "7B",
                "type": "Code-specialized",
                "memory_requirement": "~14GB",
                "license": "Custom (Meta)",
                "description": "Meta's CodeLlama 7B for code generation"
            },
            "codellama-13b": {
                "parameters": "13B",
                "type": "Code-specialized", 
                "memory_requirement": "~26GB",
                "license": "Custom (Meta)",
                "description": "Meta's CodeLlama 13B for complex coding tasks"
            },
            "phi3-mini": {
                "parameters": "3.8B",
                "type": "Instruction-tuned",
                "memory_requirement": "~8GB",
                "license": "MIT",
                "description": "Microsoft's efficient Phi-3 Mini model"
            },
            "phi3-small": {
                "parameters": "7B",
                "type": "Instruction-tuned",
                "memory_requirement": "~14GB", 
                "license": "MIT",
                "description": "Microsoft's Phi-3 Small model"
            },
            
            # Ollama models
            "ollama-llama2": {
                "parameters": "7B",
                "type": "Chat-tuned (Ollama)",
                "memory_requirement": "~4GB",
                "license": "Custom (Meta)",
                "description": "Llama2 via Ollama (quantized)"
            },
            "ollama-llama2-13b": {
                "parameters": "13B",
                "type": "Chat-tuned (Ollama)",
                "memory_requirement": "~8GB",
                "license": "Custom (Meta)", 
                "description": "Llama2 13B via Ollama (quantized)"
            },
            "ollama-mistral": {
                "parameters": "7B",
                "type": "Instruction-tuned (Ollama)",
                "memory_requirement": "~4GB",
                "license": "Apache 2.0",
                "description": "Mistral 7B via Ollama (quantized)"
            },
            "ollama-codellama": {
                "parameters": "7B",
                "type": "Code-specialized (Ollama)",
                "memory_requirement": "~4GB",
                "license": "Custom (Meta)",
                "description": "CodeLlama via Ollama (quantized)"
            },
            "ollama-phi3": {
                "parameters": "3.8B",
                "type": "Instruction-tuned (Ollama)",
                "memory_requirement": "~2GB",
                "license": "MIT",
                "description": "Phi-3 via Ollama (quantized)"
            },
            "ollama-gemma": {
                "parameters": "7B",
                "type": "Instruction-tuned (Ollama)",
                "memory_requirement": "~4GB", 
                "license": "Apache 2.0",
                "description": "Google's Gemma via Ollama (quantized)"
            },
            "ollama-qwen": {
                "parameters": "7B",
                "type": "Chat-tuned (Ollama)",
                "memory_requirement": "~4GB",
                "license": "Custom (Alibaba)",
                "description": "Qwen via Ollama (quantized)"
            },
            "ollama-deepseek-coder": {
                "parameters": "6.7B",
                "type": "Code-specialized (Ollama)",
                "memory_requirement": "~4GB",
                "license": "Custom",
                "description": "DeepSeek Coder via Ollama (quantized)"
            },
            "ollama-wizardcoder": {
                "parameters": "15B",
                "type": "Code-specialized (Ollama)",
                "memory_requirement": "~8GB",
                "license": "Custom",
                "description": "WizardCoder via Ollama (quantized)"
            }
        })
    
    def get_ollama_models(self) -> List[str]:
        """Get list of available Ollama models"""
        return [name for name in self.available_models.keys() if name.startswith("ollama-")]
    
    def get_huggingface_models(self) -> List[str]:
        """Get list of Hugging Face models"""
        return [name for name in self.available_models.keys() if not name.startswith("ollama-")]
    
    def check_ollama_status(self) -> Dict:
        """Check Ollama server status and available models"""
        if not OLLAMA_AVAILABLE:
            return {"available": False, "reason": "requests library not installed"}
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return {
                    "available": True,
                    "installed_models": [model["name"] for model in models],
                    "server_version": response.headers.get("server", "unknown")
                }
            else:
                return {"available": False, "reason": f"Server error: {response.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"available": False, "reason": "Ollama server not running"}
        except Exception as e:
            return {"available": False, "reason": f"Error: {str(e)}"}
    
    def install_ollama_model(self, model_name: str) -> bool:
        """Install a model via Ollama"""
        if not model_name.startswith("ollama-"):
            print("Error: Only Ollama models can be installed via this method")
            return False
        
        # Extract actual model name (remove ollama- prefix)
        actual_model = model_name.replace("ollama-", "").replace("-", ":")
        if ":" not in actual_model and actual_model not in ["mistral", "phi3", "gemma", "qwen"]:
            actual_model = actual_model  # Keep as is for simple names
        
        adapter = OllamaAdapter(actual_model)
        return adapter.pull_model()


# Helper function to create the default model manager
def create_model_manager(cache_dir: Optional[str] = None, enhanced: bool = True) -> ModelManager:
    """Create model manager (enhanced by default)"""
    if enhanced:
        return EnhancedModelManager(cache_dir)
    else:
        return ModelManager(cache_dir)