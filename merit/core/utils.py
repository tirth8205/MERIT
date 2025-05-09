"""
Utility functions for the MERIT framework.
"""
from typing import Dict, List, Optional, Union
import re
import json
import time
from datetime import datetime

class ReasoningAnnotator:
    """Utility for annotating reasoning patterns in text."""
    
    @staticmethod
    def identify_reasoning_steps(text: str) -> List[Dict]:
        """Identify reasoning steps in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of identified reasoning steps
        """
        # Extract reasoning steps (looking for numbered or bulleted patterns)
        step_patterns = [
            r'^\s*(\d+)[.)]\s*(.*)',  # Numbered: 1. Step one
            r'^\s*[•*-]\s*(.*)',      # Bulleted: • Step one
            r'Step\s+(\d+)[:.]\s*(.*)', # "Step 1: First do this"
            r'First,\s+(.*)',          # Sequential: First, ...
            r'Second,\s+(.*)',         # Sequential: Second, ...  
            r'Third,\s+(.*)',          # Sequential: Third, ...
            r'Finally,\s+(.*)',        # Sequential: Finally, ...
            r'Lastly,\s+(.*)',         # Sequential: Lastly, ...
        ]
        
        # Extract steps using the patterns
        steps = []
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            for pattern in step_patterns:
                match = re.match(pattern, line)
                if match:
                    if len(match.groups()) == 2:  # Numbered pattern
                        step_num, step_text = match.groups()
                        steps.append({
                            "number": step_num, 
                            "text": step_text,
                            "line_number": i
                        })
                    else:  # Bulleted or sequential pattern
                        step_text = match.group(1)
                        steps.append({
                            "number": len(steps)+1, 
                            "text": step_text,
                            "line_number": i
                        })
                    break
        
        return steps
    
    @staticmethod
    def identify_logical_connectors(text: str) -> List[Dict]:
        """Identify logical connectors in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of identified logical connectors
        """
        connectors = [
            "therefore", "thus", "hence", "so", "because", "since", 
            "consequently", "as a result", "it follows that", 
            "this implies", "this means", "which means"
        ]
        
        results = []
        for connector in connectors:
            for match in re.finditer(r'\b' + re.escape(connector) + r'\b', text, re.IGNORECASE):
                start, end = match.span()
                # Get surrounding context
                context_start = max(0, start - 40)
                context_end = min(len(text), end + 40)
                results.append({
                    "connector": match.group(),
                    "position": (start, end),
                    "context": text[context_start:context_end]
                })
        
        return results
    
    @staticmethod
    def identify_claim_evidence_pairs(text: str) -> List[Dict]:
        """Identify claim-evidence pairs in text.
        
        This is a simplified implementation. A production version would
        use more sophisticated NLP techniques.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of identified claim-evidence pairs
        """
        # Simple heuristic patterns for claims
        claim_patterns = [
            r'(?:I claim that|I argue that|It is clear that|We can see that|This shows that|This demonstrates that|This proves that|This indicates that|This suggests that)([^.!?]*[.!?])',
            r'([^.!?]*) (?:because|since|as) ([^.!?]*[.!?])',
            r'([^.!?]*[.!?]) (?:This is because|This is due to|The reason is|That is because) ([^.!?]*[.!?])'
        ]
        
        results = []
        for pattern in claim_patterns:
            for match in re.finditer(pattern, text):
                if len(match.groups()) == 1:
                    claim = match.group(1).strip()
                    results.append({
                        "claim": claim,
                        "evidence": None
                    })
                elif len(match.groups()) == 2:
                    claim = match.group(1).strip()
                    evidence = match.group(2).strip()
                    results.append({
                        "claim": claim,
                        "evidence": evidence
                    })
        
        return results


class BenchmarkRunner:
    """Utility for running and timing evaluations."""
    
    def __init__(self):
        self.results = []
    
    def run(self, evaluator, dataset, name: str = None, **kwargs):
        """Run an evaluation benchmark.
        
        Args:
            evaluator: ReasoningEvaluator instance
            dataset: Dataset to evaluate
            name: Name of the benchmark
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Benchmark results
        """
        name = name or f"Benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Time the evaluation
        start_time = time.time()
        result = evaluator.evaluate_dataset(dataset, **kwargs)
        end_time = time.time()
        
        benchmark_result = {
            "name": name,
            "dataset_size": len(dataset),
            "execution_time": end_time - start_time,
            "result": result
        }
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def save_results(self, path: str):
        """Save benchmark results to file.
        
        Args:
            path: Path to save results
        """
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, path: str):
        """Load benchmark results from file.
        
        Args:
            path: Path to load results from
        """
        with open(path, 'r') as f:
            self.results = json.load(f)
    
    def get_summary(self):
        """Get a summary of benchmark results.
        
        Returns:
            Dictionary with benchmark summaries
        """
        summary = []
        for result in self.results:
            summary.append({
                "name": result["name"],
                "dataset_size": result["dataset_size"],
                "execution_time": result["execution_time"],
                "execution_time_per_item": result["execution_time"] / max(1, result["dataset_size"])
            })
        return summary
