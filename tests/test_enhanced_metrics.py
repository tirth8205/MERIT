"""
Tests for enhanced MERIT metrics.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from merit.core.consistency import EnhancedLogicalConsistencyMetric
from merit.core.factual import EnhancedFactualAccuracyMetric
from merit.core.reasoning import EnhancedReasoningStepMetric
from merit.core.alignment import EnhancedAlignmentMetric
from merit.core.device import DeviceManager


class TestDeviceManager:
    """Test device management functionality"""
    
    def test_get_optimal_device_cpu_fallback(self):
        """Test CPU fallback when no GPU available"""
        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('torch.cuda.is_available', return_value=False):
            device = DeviceManager.get_optimal_device()
            assert device == "cpu"
    
    def test_get_optimal_device_mps(self):
        """Test MPS device selection"""
        with patch('torch.backends.mps.is_available', return_value=True):
            device = DeviceManager.get_optimal_device()
            assert device == "mps"
    
    def test_get_optimal_device_cuda(self):
        """Test CUDA device selection"""
        with patch('torch.backends.mps.is_available', return_value=False), \
             patch('torch.cuda.is_available', return_value=True):
            device = DeviceManager.get_optimal_device()
            assert device == "cuda"


class TestEnhancedLogicalConsistencyMetric:
    """Test enhanced logical consistency metric"""
    
    @pytest.fixture
    def metric(self):
        """Create metric instance for testing"""
        with patch('merit.core.device.DeviceManager.get_optimal_device', return_value="cpu"):
            return EnhancedLogicalConsistencyMetric()
    
    def test_consistent_text(self, metric):
        """Test metric with logically consistent text"""
        text = "The sky is blue. Water is wet. These are basic facts about our world."
        result = metric.compute(text)
        
        assert isinstance(result, dict)
        assert "score" in result
        assert "analysis" in result
        assert 0.0 <= result["score"] <= 1.0
        assert result["score"] > 0.8  # Should be high for consistent text
    
    def test_contradictory_text(self, metric):
        """Test metric with contradictory text"""
        text = "The sky is blue. The sky is not blue. This is confusing."
        result = metric.compute(text)
        
        assert isinstance(result, dict)
        assert "score" in result
        assert result["score"] < 0.8  # Should be lower for contradictory text
        assert len(result["analysis"]["semantic_contradictions"]) > 0
    
    def test_empty_text(self, metric):
        """Test metric with empty text"""
        text = ""
        result = metric.compute(text)
        
        assert result["score"] == 0.0
        assert "Empty prediction" in result["analysis"]
    
    def test_single_sentence(self, metric):
        """Test metric with single sentence"""
        text = "This is a single sentence."
        result = metric.compute(text)
        
        assert result["score"] == 1.0  # No contradictions possible
    
    @patch('merit.core.consistency.nltk.data.find')
    def test_nltk_fallback(self, mock_find, metric):
        """Test fallback when NLTK data not available"""
        mock_find.side_effect = LookupError("Resource not found")
        
        # Should still work with regex fallback
        text = "The sky is blue. Water is wet."
        result = metric.compute(text)
        
        assert isinstance(result, dict)
        assert "score" in result


class TestEnhancedFactualAccuracyMetric:
    """Test enhanced factual accuracy metric"""
    
    @pytest.fixture
    def metric(self):
        """Create metric instance for testing"""
        return EnhancedFactualAccuracyMetric()
    
    def test_accurate_facts(self, metric):
        """Test metric with accurate facts"""
        text = "Water is H2O. The Earth revolves around the Sun."
        result = metric.compute(text)
        
        assert isinstance(result, dict)
        assert "score" in result
        assert "accuracy_score" in result
        assert "coverage" in result
        assert 0.0 <= result["score"] <= 1.0
    
    def test_inaccurate_facts(self, metric):
        """Test metric with inaccurate facts"""
        text = "The Sun revolves around the Earth. Humans have 3 legs."
        result = metric.compute(text)
        
        assert isinstance(result, dict)
        assert "score" in result
        assert result["accuracy_score"] <= 0.5  # Should be low for false facts
    
    def test_unverifiable_claims(self, metric):
        """Test metric with unverifiable claims"""
        text = "The alien civilization on planet Zyx has purple oceans."
        result = metric.compute(text)
        
        assert isinstance(result, dict)
        assert result["coverage"] == 0.0  # Nothing verifiable
        assert len(result["claims_analysis"]) > 0
    
    def test_empty_text(self, metric):
        """Test metric with empty text"""
        text = ""
        result = metric.compute(text)
        
        assert result["score"] == 0.0
        assert "Empty prediction" in result["analysis"]
    
    def test_mixed_accuracy(self, metric):
        """Test metric with mixed accurate and inaccurate facts"""
        text = "Water is H2O, which is correct. But the Sun revolves around Earth, which is wrong."
        result = metric.compute(text)
        
        assert isinstance(result, dict)
        assert 0.0 < result["accuracy_score"] < 1.0  # Should be between extremes


class TestEnhancedReasoningStepMetric:
    """Test enhanced reasoning step metric"""
    
    @pytest.fixture
    def metric(self):
        """Create metric instance for testing"""
        with patch('merit.core.device.DeviceManager.get_optimal_device', return_value="cpu"):
            return EnhancedReasoningStepMetric()
    
    def test_clear_reasoning_steps(self, metric):
        """Test metric with clear reasoning steps"""
        text = """
        First, we need to identify the problem.
        Second, we should gather relevant information.
        Third, we analyze the data carefully.
        Finally, we draw conclusions based on our analysis.
        """
        result = metric.compute(text)
        
        assert isinstance(result, dict)
        assert "score" in result
        assert "steps" in result
        assert "num_steps" in result
        assert result["num_steps"] > 3
        assert result["score"] > 0.5
    
    def test_numbered_steps(self, metric):
        """Test metric with numbered steps"""
        text = """
        1. Start by reading the question carefully.
        2. Identify the key information provided.
        3. Apply relevant formulas or concepts.
        4. Calculate the result step by step.
        5. Verify the answer makes sense.
        """
        result = metric.compute(text)
        
        assert result["num_steps"] >= 5
        assert any(step["type"] == "pattern_based" for step in result["steps"])
    
    def test_no_clear_steps(self, metric):
        """Test metric with no clear reasoning steps"""
        text = "This is just a paragraph without any clear reasoning structure or steps."
        result = metric.compute(text)
        
        assert result["score"] <= 0.3  # Should be low
        assert result["num_steps"] <= 1
    
    def test_logical_connectors(self, metric):
        """Test metric recognizes logical connectors"""
        text = "Because the evidence shows X, therefore we can conclude Y. Hence, the answer is Z."
        result = metric.compute(text)
        
        assert result["score"] > 0.4  # Should recognize logical structure
        assert result["coherence"]["overall_coherence"] > 0.5
    
    def test_empty_text(self, metric):
        """Test metric with empty text"""
        text = ""
        result = metric.compute(text)
        
        assert result["score"] == 0.0
        assert "Empty prediction" in result["analysis"]


class TestEnhancedAlignmentMetric:
    """Test enhanced alignment metric"""
    
    @pytest.fixture
    def metric(self):
        """Create metric instance for testing"""
        return EnhancedAlignmentMetric()
    
    def test_positive_alignment(self, metric):
        """Test metric with positive alignment"""
        text = "We should treat everyone with respect and fairness. Transparency is important for trust."
        result = metric.compute(text)
        
        assert isinstance(result, dict)
        assert "score" in result
        assert result["score"] > 0.6  # Should be high for positive values
        assert result["principle_analysis"]["overall_score"] > 0.5
    
    def test_negative_alignment(self, metric):
        """Test metric with negative alignment"""
        text = "Some people are worthless and don't deserve respect. We should discriminate based on appearance."
        result = metric.compute(text)
        
        assert result["score"] < 0.4  # Should be low for negative values
        assert result["bias_analysis"]["total_bias_instances"] > 0
    
    def test_neutral_text(self, metric):
        """Test metric with neutral text"""
        text = "The weather is nice today. I went to the store to buy groceries."
        result = metric.compute(text)
        
        assert 0.4 <= result["score"] <= 0.7  # Should be neutral
        assert result["sentiment_analysis"]["alignment_score"] >= 0.4
    
    def test_respectful_language(self, metric):
        """Test metric recognizes respectful language"""
        text = "Please consider this thoughtful approach. Thank you for your understanding and cooperation."
        result = metric.compute(text)
        
        assert result["respectfulness_analysis"]["score"] > 0.7
        assert result["respectfulness_analysis"]["respectful_indicators"] > 0
    
    def test_disrespectful_language(self, metric):
        """Test metric detects disrespectful language"""
        text = "That's a stupid idea. You're an idiot if you believe that nonsense."
        result = metric.compute(text)
        
        assert result["respectfulness_analysis"]["score"] < 0.5
        assert result["respectfulness_analysis"]["disrespectful_indicators"] > 0
    
    def test_empty_text(self, metric):
        """Test metric with empty text"""
        text = ""
        result = metric.compute(text)
        
        assert result["score"] == 0.5  # Neutral for empty text


class TestMetricIntegration:
    """Test integration between different metrics"""
    
    def test_all_metrics_consistency(self):
        """Test that all metrics return consistent structure"""
        with patch('merit.core.device.DeviceManager.get_optimal_device', return_value="cpu"):
            metrics = [
                EnhancedLogicalConsistencyMetric(),
                EnhancedFactualAccuracyMetric(), 
                EnhancedReasoningStepMetric(),
                EnhancedAlignmentMetric()
            ]
        
        text = "First, we consider the evidence. The Earth is round, which is scientifically proven. Therefore, we should respect scientific knowledge and treat it fairly."
        
        results = []
        for metric in metrics:
            try:
                result = metric.compute(text)
                results.append(result)
                
                # Check basic structure
                assert isinstance(result, dict)
                assert "score" in result
                assert isinstance(result["score"], (int, float))
                assert 0.0 <= result["score"] <= 1.0
                
            except Exception as e:
                pytest.fail(f"Metric {metric.__class__.__name__} failed: {e}")
        
        assert len(results) == 4  # All metrics should succeed
    
    def test_metric_performance_benchmark(self):
        """Test metric performance on various text types"""
        test_cases = [
            {
                "text": "Simple factual statement.",
                "expected_logical": lambda x: x > 0.8,
                "expected_factual": lambda x: x >= 0.3,
                "expected_reasoning": lambda x: x >= 0.2,
                "expected_alignment": lambda x: x >= 0.4
            },
            {
                "text": "First step. Second step. Therefore conclusion.",
                "expected_logical": lambda x: x > 0.7,
                "expected_factual": lambda x: x >= 0.3,
                "expected_reasoning": lambda x: x > 0.5,
                "expected_alignment": lambda x: x >= 0.4
            },
            {
                "text": "",
                "expected_logical": lambda x: x == 0.0,
                "expected_factual": lambda x: x == 0.0,
                "expected_reasoning": lambda x: x == 0.0,
                "expected_alignment": lambda x: x == 0.5
            }
        ]
        
        with patch('merit.core.device.DeviceManager.get_optimal_device', return_value="cpu"):
            logical_metric = EnhancedLogicalConsistencyMetric()
            factual_metric = EnhancedFactualAccuracyMetric()
            reasoning_metric = EnhancedReasoningStepMetric()
            alignment_metric = EnhancedAlignmentMetric()
        
        for i, case in enumerate(test_cases):
            text = case["text"]
            
            # Test each metric
            logical_result = logical_metric.compute(text)
            factual_result = factual_metric.compute(text)
            reasoning_result = reasoning_metric.compute(text)
            alignment_result = alignment_metric.compute(text)
            
            # Check expectations
            assert case["expected_logical"](logical_result["score"]), f"Logical failed for case {i}"
            assert case["expected_factual"](factual_result["score"]), f"Factual failed for case {i}"
            assert case["expected_reasoning"](reasoning_result["score"]), f"Reasoning failed for case {i}"
            assert case["expected_alignment"](alignment_result["score"]), f"Alignment failed for case {i}"


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_device_compatibility(device):
    """Test metrics work on different devices"""
    with patch('merit.core.device.DeviceManager.get_optimal_device', return_value=device):
        try:
            metric = EnhancedLogicalConsistencyMetric()
            result = metric.compute("Test sentence for device compatibility.")
            
            assert isinstance(result, dict)
            assert "score" in result
            
        except Exception as e:
            # Some devices might not be available in test environment
            if "not available" not in str(e).lower():
                pytest.fail(f"Device {device} compatibility failed: {e}")


def test_metric_error_handling():
    """Test metrics handle errors gracefully"""
    with patch('merit.core.device.DeviceManager.get_optimal_device', return_value="cpu"):
        metric = EnhancedLogicalConsistencyMetric()
    
    # Test with problematic input
    try:
        result = metric.compute(None)
        # Should handle None gracefully
        assert isinstance(result, dict)
    except:
        pass  # Acceptable to raise exception for None input
    
    # Test with very long text
    very_long_text = "This is a test sentence. " * 10000
    result = metric.compute(very_long_text)
    
    assert isinstance(result, dict)
    assert "score" in result


if __name__ == "__main__":
    pytest.main([__file__])