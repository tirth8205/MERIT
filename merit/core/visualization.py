"""
Visualization tools for reasoning evaluation.
"""
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display, HTML

class ReasoningVisualizer:
    """Visualizer for reasoning evaluation results."""
    
    def __init__(self, theme: str = "dark_background"):
        """Initialize the visualizer.
        
        Args:
            theme: Matplotlib theme to use
        """
        self.theme = theme
        plt.style.use(theme)
    
    def plot_metric_comparison(self, results: List[Dict], metric: str, fig_size: tuple = (10, 6)):
        """Plot comparison of a metric across multiple evaluations.
        
        Args:
            results: List of evaluation results
            metric: Name of the metric to compare
            fig_size: Figure size (width, height)
        """
        # Extract metric scores
        labels = []
        scores = []
        
        for i, result in enumerate(results):
            metrics = result.get("metrics", {})
            metric_result = metrics.get(metric, {})
            
            if isinstance(metric_result, dict) and "score" in metric_result:
                # Use prompt as label, truncated for readability
                prompt = result.get("prompt", f"Example {i}")
                label = prompt[:30] + "..." if len(prompt) > 30 else prompt
                labels.append(label)
                scores.append(metric_result["score"])
        
        # Plot
        plt.figure(figsize=fig_size)
        plt.bar(labels, scores, color='skyblue')
        plt.title(f"{metric.replace('_', ' ').title()} Comparison")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    
    def plot_metric_radar(self, result: Dict, fig_size: tuple = (8, 8)):
        """Plot radar chart of all metrics for a single evaluation.
        
        Args:
            result: Evaluation result
            fig_size: Figure size (width, height)
        """
        # Extract metrics
        metrics = result.get("metrics", {})
        labels = []
        scores = []
        
        for metric_name, metric_result in metrics.items():
            if isinstance(metric_result, dict) and "score" in metric_result:
                labels.append(metric_name.replace('_', ' ').title())
                scores.append(metric_result["score"])
        
        if not scores:
            print("No metric scores found in result")
            return
        
        # Plot radar chart
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        
        # Close the polygon
        scores.append(scores[0])
        angles.append(angles[0])
        labels.append(labels[0])
        
        # Plot
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))
        ax.plot(angles, scores, 'o-', linewidth=2)
        ax.fill(angles, scores, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles), labels)
        ax.set_ylim(0, 1)
        ax.grid(True)
        plt.title("Metric Radar Chart")
        plt.tight_layout()
        plt.show()
    
    def highlight_reasoning_steps(self, prediction: str):
        """Highlight reasoning steps in a model prediction.
        
        Args:
            prediction: Model prediction text
        """
        # Split by lines
        lines = prediction.split('\n')
        
        # Define patterns for reasoning steps
        step_patterns = [
            (r'^\s*(\d+)[.)]\s*(.*)', r'<span style="color:lightgreen"><b>\1.</b> \2</span>'),  # Numbered
            (r'^\s*[•*-]\s*(.*)', r'<span style="color:lightgreen"><b>•</b> \1</span>'),  # Bulleted
            (r'(Step\s+\d+[:.]\s*)(.*)', r'<span style="color:lightgreen"><b>\1</b>\2</span>'),  # "Step 1:"
            (r'(First,\s+)(.*)', r'<span style="color:lightgreen"><b>\1</b>\2</span>'),  # Sequential
            (r'(Second,\s+)(.*)', r'<span style="color:lightgreen"><b>\1</b>\2</span>'),
            (r'(Third,\s+)(.*)', r'<span style="color:lightgreen"><b>\1</b>\2</span>'),
            (r'(Finally,\s+)(.*)', r'<span style="color:lightgreen"><b>\1</b>\2</span>'),
            (r'(Lastly,\s+)(.*)', r'<span style="color:lightgreen"><b>\1</b>\2</span>')
        ]
        
        # Highlight logical connectors
        connectors = ["therefore", "thus", "hence", "so", "because", "since", "consequently"]
        
        highlighted_lines = []
        for line in lines:
            highlighted = line
            
            # Apply step patterns
            for pattern, replacement in step_patterns:
                import re
                highlighted = re.sub(pattern, replacement, highlighted)
            
            # Highlight logical connectors
            for connector in connectors:
                highlighted = highlighted.replace(
                    f" {connector} ", 
                    f' <span style="color:yellow">{connector}</span> '
                )
            
            highlighted_lines.append(highlighted)
        
        # Display as HTML
        html = "<div style='background-color:#1e1e1e; padding:10px; border-radius:5px'>"
        html += "<br>".join(highlighted_lines)
        html += "</div>"
        
        return HTML(html)
    
    def plot_alignment_matrix(self, results: List[Dict], principles: Optional[List[str]] = None):
        """Plot alignment matrix for multiple evaluations.
        
        Args:
            results: List of evaluation results
            principles: List of principles to include
        """
        # Collect alignment data
        alignment_data = []
        
        for result in results:
            metrics = result.get("metrics", {})
            alignment = metrics.get("alignment", {})
            
            if not alignment or "principle_analysis" not in alignment:
                continue
                
            principles_found = alignment["principle_analysis"]
            row = {}
            
            # Use truncated prompt as identifier
            prompt = result.get("prompt", "Unknown")
            identifier = prompt[:20] + "..." if len(prompt) > 20 else prompt
            row["Prompt"] = identifier
            
            # Add principles
            for principle, analysis in principles_found.items():
                if principles and principle not in principles:
                    continue
                row[principle] = 1 if analysis.get("adhered", False) else 0
            
            alignment_data.append(row)
        
        if not alignment_data:
            print("No alignment data found in results")
            return
        
        # Create dataframe
        df = pd.DataFrame(alignment_data)
        df = df.set_index("Prompt")
        
        # Plot heatmap
        plt.figure(figsize=(10, len(df) * 0.5 + 2))
        sns.heatmap(df, cmap="YlGnBu", linewidths=.5, cbar_kws={"label": "Adhered"})
        plt.title("Alignment Matrix")
        plt.tight_layout()
        plt.show()
        
    def plot_reasoning_flow(self, prediction: str, fig_size: tuple = (12, 8)):
        """Plot reasoning flow as a directed graph.
        
        This is a placeholder - in a real implementation, this would extract
        the reasoning structure and visualize it as a graph.
        
        Args:
            prediction: Model prediction text
            fig_size: Figure size (width, height)
        """
        print("Reasoning flow visualization requires NetworkX and additional implementation.")
        print("This is a placeholder for future implementation.")
        
        # In a complete implementation, this would:
        # 1. Extract reasoning steps and their relationships
        # 2. Build a directed graph
        # 3. Visualize the graph with appropriate styling
