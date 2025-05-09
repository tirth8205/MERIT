"""
Basic example of using the MERIT framework.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from merit.models.adapters.huggingface import HuggingFaceAdapter
from merit.core.evaluation import ReasoningEvaluator
from merit.core.metrics import get_default_metric_registry
from merit.datasets.loaders import LogicalReasoningDataset
from merit.core.visualization import ReasoningVisualizer

def main():
    # Get API token
    api_token = os.environ.get("HF_API_TOKEN")
    if not api_token:
        print("Please set the HF_API_TOKEN environment variable.")
        print("You can get a token from https://huggingface.co/settings/tokens")
        return
    
    # Initialize model adapter
    model_adapter = HuggingFaceAdapter(
        api_token=api_token,
        model_name="meta-llama/Meta-Llama-3-8B-Instruct"  # Free to use model
    )
    
    # Initialize evaluator
    evaluator = ReasoningEvaluator(
        metric_registry=get_default_metric_registry(),
        model_adapter=model_adapter
    )
    
    # Load dataset
    dataset = LogicalReasoningDataset()
    dataset.load()
    
    # Run evaluation on a single example
    example = dataset[0]
    result = evaluator.evaluate_prompt(
        prompt=example["prompt"],
        reference=example["reference"],
        temperature=0.3
    )
    
    # Print results
    print("Prompt:", example["prompt"])
    print("-" * 50)
    print("Response:", result["prediction"])
    print("-" * 50)
    print("Metrics:")
    for metric_name, metric_result in result["metrics"].items():
        if isinstance(metric_result, dict) and "score" in metric_result:
            print(f"  {metric_name}: {metric_result['score']:.4f}")
    
    # Create visualizer
    visualizer = ReasoningVisualizer()
    
    # Visualize reasoning steps (this will only work in a Jupyter notebook)
    try:
        html = visualizer.highlight_reasoning_steps(result["prediction"])
        print("Visualization created (will only display properly in Jupyter)")
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
