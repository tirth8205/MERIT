"""Ollama model adapter for MERIT."""
import json
from typing import Optional

try:
    import requests
    import subprocess
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: requests not available for Ollama integration")


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
                            print(f"\u2713 Successfully pulled {self.model_name}")
                            return True
            else:
                print(f"\u2717 Failed to pull {self.model_name}: {response.status_code}")
                return False

        except Exception as e:
            print(f"\u2717 Error pulling {self.model_name}: {e}")
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
