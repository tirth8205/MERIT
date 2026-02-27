"""HuggingFace model adapters for MERIT."""
import torch
from typing import Optional
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
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("Info: bitsandbytes not available. 4-bit quantization disabled.")

from .device import DeviceManager


class LocalModelAdapter:
    """Base class for local model adapters"""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir  # None = use HuggingFace default cache
        self.device = DeviceManager.get_optimal_device()
        self.model = None
        self.tokenizer = None
        self.generation_config = {}

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


class Llama3Adapter(LocalModelAdapter):
    """Adapter for Llama-3.1-8B-Instruct model (best for academic benchmarking)"""

    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            cache_dir=cache_dir
        )

    def load_model(self):
        """Load Llama-3.1-8B-Instruct model"""
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

            # Get model configuration (8B model needs more memory)
            model_config = DeviceManager.get_model_config("7b", self.device)

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                **model_config
            )

            print(f"\u2713 {self.model_name} loaded successfully")

        except Exception as e:
            print(f"\u2717 Error loading {self.model_name}: {e}")
            raise

    def generate(self, prompt: str, max_length: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate response using Llama-3 chat format"""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        try:
            # Use Llama-3's native chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

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

            # Extract assistant response (after the prompt)
            if "assistant" in generated_text.lower():
                response = generated_text.split("assistant")[-1].strip()
            else:
                response = generated_text[len(formatted_prompt):].strip()

            return response

        except Exception as e:
            print(f"Error generating with {self.model_name}: {e}")
            return f"Error: {str(e)}"


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

            print(f"\u2713 {self.model_name} loaded successfully")

        except Exception as e:
            print(f"\u2717 Error loading {self.model_name}: {e}")
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

            print(f"\u2713 {self.model_name} loaded successfully")

        except Exception as e:
            print(f"\u2717 Error loading {self.model_name}: {e}")
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

            # Use float32 on MPS to avoid NaN issues with phi-2
            model_config = {
                "torch_dtype": torch.float32 if self.device == "mps" else (torch.float16 if self.device != "cpu" else torch.float32),
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

            print(f"\u2713 {self.model_name} loaded successfully")

        except Exception as e:
            print(f"\u2717 Error loading {self.model_name}: {e}")
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
                # Use greedy decoding on MPS to avoid NaN issues
                use_sampling = temperature > 0 and self.device != "mps"
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature if use_sampling else None,
                    do_sample=use_sampling,
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

            print(f"\u2713 {self.model_name} loaded successfully")

        except Exception as e:
            print(f"\u2717 Error loading {self.model_name}: {e}")
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

            print(f"\u2713 {self.model_name} loaded successfully")

        except Exception as e:
            print(f"\u2717 Error loading {self.model_name}: {e}")
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

            print(f"\u2713 {self.model_name} loaded successfully")

        except Exception as e:
            print(f"\u2717 Error loading {self.model_name}: {e}")
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
