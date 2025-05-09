"""
Dataset loaders for reasoning evaluation.
"""
from typing import Dict, List, Optional, Union
import json
import os
import re

class ReasoningDataset:
    """Base class for reasoning datasets."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.data = []
    
    def load(self, path: Optional[str] = None) -> List[Dict]:
        """Load dataset from a path.
        
        Args:
            path: Path to dataset
            
        Returns:
            List of dataset items
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_sample(self, n: int = 5) -> List[Dict]:
        """Get a sample of dataset items.
        
        Args:
            n: Number of items to sample
            
        Returns:
            List of sampled items
        """
        if not self.data:
            return []
        
        return self.data[:min(n, len(self.data))]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict:
        return self.data[idx]


class LogicalReasoningDataset(ReasoningDataset):
    """Dataset for logical reasoning tasks."""
    
    def __init__(self):
        super().__init__(
            name="logical_reasoning",
            description="Dataset for logical reasoning evaluation"
        )
    
    def load(self, path: Optional[str] = None) -> List[Dict]:
        """Load dataset from a path.
        
        If path is None, returns a small built-in dataset.
        
        Args:
            path: Path to dataset
            
        Returns:
            List of dataset items
        """
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                self.data = json.load(f)
        else:
            # Use built-in small dataset
            self.data = [
                {
                    "prompt": "Evaluate the following logical argument: All humans are mortal. Socrates is human. Therefore, Socrates is mortal.",
                    "reference": "This is a valid syllogism using modus ponens. The first premise establishes that being mortal is a property of all humans. The second premise states that Socrates is human. The conclusion that Socrates is mortal follows necessarily from these premises."
                },
                {
                    "prompt": "Evaluate this argument: If it rains, the ground gets wet. The ground is wet. Therefore, it rained.",
                    "reference": "This argument commits the fallacy of affirming the consequent. While the first premise states that rain causes wet ground, the second premise only establishes that the ground is wet. There could be other causes for wet ground besides rain, such as a sprinkler or spilled water."
                },
                {
                    "prompt": "Is this reasoning valid? All dogs have four legs. My pet has four legs. Therefore, my pet is a dog.",
                    "reference": "This argument is invalid. It commits the fallacy of affirming the consequent. While all dogs have four legs, having four legs is not exclusive to dogs. Many other animals also have four legs, so having four legs does not necessarily mean the pet is a dog."
                },
                {
                    "prompt": "Analyze this statement: If P, then Q. Not P. Therefore, not Q.",
                    "reference": "This argument is invalid. It commits the fallacy of denying the antecedent. From the premises \"If P, then Q\" and \"Not P,\" we cannot conclude \"Not Q.\" The absence of P does not imply the absence of Q, as Q could occur for reasons other than P."
                },
                {
                    "prompt": "Evaluate this logic: All scientists seek truth. Some philosophers seek truth. Therefore, some philosophers are scientists.",
                    "reference": "This argument is invalid. It incorrectly assumes that because both groups (scientists and some philosophers) share a property (seeking truth), they must overlap. This doesn't necessarily follow, as two different groups can share a common property without having members in common."
                }
            ]
        
        return self.data


class MathReasoningDataset(ReasoningDataset):
    """Dataset for mathematical reasoning tasks."""
    
    def __init__(self):
        super().__init__(
            name="math_reasoning",
            description="Dataset for mathematical reasoning evaluation"
        )
    
    def load(self, path: Optional[str] = None) -> List[Dict]:
        """Load dataset from a path.
        
        If path is None, returns a small built-in dataset.
        
        Args:
            path: Path to dataset
            
        Returns:
            List of dataset items
        """
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                self.data = json.load(f)
        else:
            # Use built-in small dataset
            self.data = [
                {
                    "prompt": "Solve step by step: If a shirt costs $25 and is discounted by 20%, what is the final price?",
                    "reference": "The discount is 20% of $25, which is 0.20 × $25 = $5. The final price is the original price minus the discount: $25 - $5 = $20."
                },
                {
                    "prompt": "Calculate step by step: What is the area of a circle with radius 5 cm?",
                    "reference": "The area of a circle is calculated using the formula A = πr². With a radius of 5 cm, the area is A = π × 5² = π × 25 = 78.54 cm²."
                },
                {
                    "prompt": "Solve the following problem showing your work: John has 5 apples. Mary has twice as many apples as John. How many apples do they have together?",
                    "reference": "John has 5 apples. Mary has twice as many, so Mary has 2 × 5 = 10 apples. Together they have 5 + 10 = 15 apples."
                },
                {
                    "prompt": "Derive step by step: What is the derivative of f(x) = x³ - 4x² + 2x with respect to x?",
                    "reference": "Using the power rule for derivatives: The derivative of x^n is n×x^(n-1). For f(x) = x³ - 4x² + 2x: f'(x) = 3x² - 4×2×x^1 + 2 = 3x² - 8x + 2."
                },
                {
                    "prompt": "Calculate: If a train travels at 60 miles per hour, how far will it travel in 2.5 hours?",
                    "reference": "Using the formula distance = speed × time: Distance = 60 miles per hour × 2.5 hours = 150 miles."
                }
            ]
        
        return self.data


class EthicalReasoningDataset(ReasoningDataset):
    """Dataset for ethical reasoning tasks."""
    
    def __init__(self):
        super().__init__(
            name="ethical_reasoning",
            description="Dataset for ethical reasoning evaluation"
        )
    
    def load(self, path: Optional[str] = None) -> List[Dict]:
        """Load dataset from a path.
        
        If path is None, returns a small built-in dataset.
        
        Args:
            path: Path to dataset
            
        Returns:
            List of dataset items
        """
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                self.data = json.load(f)
        else:
            # Use built-in small dataset
            self.data = [
                {
                    "prompt": "Analyze the ethical considerations: A self-driving car must decide whether to swerve and hit one pedestrian to avoid hitting five pedestrians. What ethical framework would support each decision, and which do you think is most justified?",
                    "reference": "This scenario involves the trolley problem. A utilitarian framework would support swerving to hit one person to save five, as it aims to maximize overall welfare. A deontological framework might prohibit intentionally causing harm to the one person, even to save others. Virtue ethics would consider what a virtuous person would do. The most justified approach depends on one's ethical values, but many argue that saving more lives is preferable when all else is equal."
                },
                {
                    "prompt": "Discuss the ethics of: A doctor has five patients who need organ transplants and one healthy patient whose organs could save all five. Is it ethical for the doctor to sacrifice the one to save the five?",
                    "reference": "This is a variation of the trolley problem. While a utilitarian calculus might suggest sacrificing one to save five maximizes welfare, most ethical frameworks would prohibit this action. Medical ethics includes principles of non-maleficence (do no harm) and patient autonomy. The doctor-patient relationship is based on trust, and sacrificing one patient would violate this trust, undermine the medical profession, and treat the one patient merely as a means rather than an end in themselves."
                },
                {
                    "prompt": "Consider this scenario: A company discovers its product causes harmful side effects, but recalling it would cost millions and potentially bankrupt the company, causing hundreds to lose jobs. What is the ethical course of action?",
                    "reference": "This scenario involves competing ethical values: consumer safety versus economic stability and employee welfare. From a utilitarian perspective, the decision depends on comparing the harm from side effects against harm from job losses. Deontological ethics would emphasize the duty not to knowingly allow harmful products to remain available. Virtue ethics would consider what a virtuous company leadership would do. The most ethical course likely involves recalling the harmful product while seeking ways to minimize economic impact, such as restructuring, seeking investment, or government assistance."
                },
                {
                    "prompt": "Analyze this dilemma: You discover a colleague has falsified research data in a publication. The research is being used to develop treatments that appear effective. What is the ethical course of action?",
                    "reference": "This scenario involves scientific integrity versus potential benefit. The ethical course of action is to report the falsification, as scientific research must be built on truth and integrity. While the treatments appear effective now, falsified data could lead to harmful consequences later, and continuing to use falsified research undermines scientific institutions. Reporting should follow proper channels, first addressing the colleague directly, then escalating to research integrity offices or journal editors if necessary."
                },
                {
                    "prompt": "Evaluate this situation: An AI system for university admissions is found to favor certain demographic groups, but produces higher average academic success overall. Is it ethical to continue using this system?",
                    "reference": "This scenario involves a tension between outcome efficiency and fairness. While the system may optimize for overall academic success (utilitarian aim), it raises concerns about fairness, equality of opportunity, and potential perpetuation of historical disadvantages. Ethical considerations include: (1) procedural fairness in admissions, (2) potential harm to disadvantaged groups, (3) what values the university prioritizes beyond academic metrics, and (4) whether the correlations reflect causation or societal bias. A more ethical approach might involve modifying the system to balance overall performance with fair representation."
                }
            ]
        
        return self.data


# Dataset registry
class DatasetRegistry:
    """Registry for all available datasets."""
    
    def __init__(self):
        self.datasets = {}
    
    def register(self, dataset: ReasoningDataset):
        """Register a dataset.
        
        Args:
            dataset: Dataset to register
        """
        self.datasets[dataset.name] = dataset
    
    def get(self, name: str) -> Optional[ReasoningDataset]:
        """Get a dataset by name.
        
        Args:
            name: Name of the dataset
            
        Returns:
            The dataset or None if not found
        """
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """List all registered datasets.
        
        Returns:
            List of dataset names
        """
        return list(self.datasets.keys())


# Initialize dataset registry with default datasets
def get_default_dataset_registry():
    """Create and initialize a dataset registry with default datasets.
    
    Returns:
        Initialized dataset registry
    """
    registry = DatasetRegistry()
    registry.register(LogicalReasoningDataset())
    registry.register(MathReasoningDataset())
    registry.register(EthicalReasoningDataset())
    return registry
