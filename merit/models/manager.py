"""Model manager for MERIT — coordinates loading, unloading, and benchmarking."""
import time
from typing import Dict, List, Optional

import numpy as np

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from .device import DeviceManager
from .huggingface import (
    LocalModelAdapter,
    Llama3Adapter,
    MistralInstructAdapter,
    TinyLlamaAdapter,
    Phi2Adapter,
    Qwen2Adapter,
    CodeLlamaAdapter,
    Phi3Adapter,
)
from .ollama import OllamaAdapter


class ModelManager:
    """Manages multiple local models for comparative evaluation"""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.available_models = {
            # All models are instruction-tuned and suitable for reasoning
            "llama3-8b": Llama3Adapter,
            "mistral-7b-instruct": MistralInstructAdapter,
            "tinyllama-1b": TinyLlamaAdapter,
            "phi-2": Phi2Adapter,
            "qwen2-0.5b": Qwen2Adapter,
        }
        self.loaded_models = {}

        # Model metadata
        self.model_info = {
            "llama3-8b": {
                "parameters": "8B",
                "type": "Instruction-tuned",
                "memory_requirement": "~16GB",
                "license": "Llama 3.1 Community",
                "description": "Meta's Llama-3.1 8B instruction model - best benchmarked"
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
            recommended.extend(["llama3-3b", "mistral-7b-instruct"])

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
