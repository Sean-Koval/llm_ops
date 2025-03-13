"""
Integration tests for the Gemini MLflow integration with real MLflow instance.
Tests the complete workflow with the evaluation framework.
"""

import os
import sys
import json
import time
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import mlflow
import numpy as np
import pandas as pd

# Import modules to test
from llm_ops_pipeline.utils.gemini_mlflow import (
    GeminiMLflowLogger, 
    setup_gemini_autologging,
    GEMINI_AVAILABLE
)
from llm_ops_pipeline.evaluation.llm_evaluation_framework import (
    LLMEvaluationFramework,
    EvaluationResult,
    calculate_classification_metrics
)

# Skip tests if Gemini is not available
pytestmark = pytest.mark.skipif(not GEMINI_AVAILABLE, reason="Google Generative AI package not available")

class TestGeminiMLflowEvaluation:
    """Integration tests for Gemini MLflow integration with the evaluation framework."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mlflow_tracking(self, temp_dir):
        """Set up a local MLflow tracking server."""
        tracking_uri = f"file://{temp_dir}/mlruns"
        previous_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        yield tracking_uri
        mlflow.set_tracking_uri(previous_uri)
    
    @pytest.fixture
    def test_dataset(self):
        """Create a simple test dataset for evaluation."""
        return [
            {"id": 1, "text": "This is a positive comment", "label": "positive"},
            {"id": 2, "text": "This is a negative comment", "label": "negative"},
            {"id": 3, "text": "This is a neutral comment", "label": "neutral"},
            {"id": 4, "text": "I am so happy today", "label": "positive"},
            {"id": 5, "text": "I am very disappointed", "label": "negative"}
        ]
    
    @pytest.fixture
    def test_prompts(self):
        """Create test prompts for evaluation."""
        return {
            "basic": "Classify the sentiment of the following text as positive, negative, or neutral: ",
            "detailed": "Analyze the sentiment in the following text and classify it as either 'positive', 'negative', or 'neutral'. Consider the emotional tone, word choice, and context. Text: "
        }
    
    @pytest.fixture
    def test_config(self, temp_dir, mlflow_tracking):
        """Create a test configuration."""
        return {
            "experiment_name": "gemini_test_evaluation",
            "mlflow": {
                "tracking_uri": mlflow_tracking
            },
            "gemini": {
                "log_inputs": True,
                "log_outputs": True,
                "log_metrics": True,
                "log_parameters": True
            },
            "workspace_dir": f"{temp_dir}/workspace"
        }
    
    @pytest.fixture
    def mock_gemini_model(self):
        """Create a mock for Gemini model to avoid actual API calls."""
        if not GEMINI_AVAILABLE:
            pytest.skip("Google Generative AI package not available")
        
        from google.generativeai import GenerativeModel
        
        # Save original method
        original_generate_content = GenerativeModel.generate_content
        
        # Create mock response class
        class MockResponse:
            def __init__(self, text):
                self.text = text
                self.usage_metadata = type('UsageMetadata', (), {
                    'prompt_token_count': 20, 
                    'candidates_token_count': 10,
                    'total_token_count': 30
                })
        
        # Create mock model
        def mock_generate_content(self, content, **kwargs):
            text = content
            if isinstance(content, list):
                text = " ".join([str(c) for c in content])
            
            # Simple sentiment classification logic
            text = text.lower()
            if "positive" in text or "happy" in text or "good" in text:
                return MockResponse("positive")
            elif "negative" in text or "disappointed" in text or "bad" in text:
                return MockResponse("negative")
            else:
                return MockResponse("neutral")
        
        # Apply mock
        GenerativeModel.generate_content = mock_generate_content
        
        yield
        
        # Restore original
        GenerativeModel.generate_content = original_generate_content
    
    def test_gemini_evaluation_with_autologging(self, temp_dir, mlflow_tracking, test_dataset, 
                                              test_prompts, test_config, mock_gemini_model):
        """Test running a complete evaluation with Gemini autologging."""
        if not GEMINI_AVAILABLE:
            pytest.skip("Google Generative AI package not available")
        
        from google.generativeai import GenerativeModel
        
        # Save dataset to file
        dataset_path = os.path.join(temp_dir, "test_dataset.json")
        with open(dataset_path, "w") as f:
            json.dump(test_dataset, f)
        
        # Save prompts to file
        prompts_path = os.path.join(temp_dir, "test_prompts.json")
        with open(prompts_path, "w") as f:
            json.dump(test_prompts, f)
        
        # Initialize Gemini autologging
        gemini_logger = setup_gemini_autologging(
            tracking_uri=mlflow_tracking,
            experiment_name=test_config["experiment_name"],
            log_inputs=True,
            log_outputs=True,
            log_metrics=True,
            log_parameters=True,
            auto_end_run=False
        )
        
        # Initialize evaluation framework
        eval_framework = LLMEvaluationFramework(
            config=test_config,
            experiment_name=test_config["experiment_name"],
            workspace_dir=test_config["workspace_dir"],
            use_mlflow=True,
            use_langfuse=False,
            use_dvc=False,
            enable_gemini_autologging=True
        )
        
        # Define inference function
        def inference_fn(prompt, example, model_id):
            model = GenerativeModel(model_name=model_id)
            response = model.generate_content([prompt, example["text"]])
            
            # Extract token usage
            token_usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0)
            }
            
            return {
                "prediction": response.text.strip(),
                "token_usage": token_usage
            }
        
        # Define metrics function
        def metrics_fn(examples, predictions):
            true_values = [ex["label"] for ex in examples]
            return calculate_classification_metrics(
                true_values, predictions, 
                labels=["positive", "negative", "neutral"]
            )
        
        # Create experiment
        experiment_id = eval_framework.create_experiment(
            name="gemini_test_evaluation",
            description="Test evaluation with Gemini and MLflow autologging"
        )
        
        # Run evaluation
        results = eval_framework.evaluate_prompts(
            task_id="sentiment_classification",
            prompts=test_prompts,
            dataset=test_dataset,
            inference_fn=inference_fn,
            metrics_fn=metrics_fn,
            model_id="gemini-1.5-flash",
            experiment_id=experiment_id
        )
        
        # Generate report
        report_dir = os.path.join(temp_dir, "reports")
        report = eval_framework.generate_metrics_report(
            results,
            output_dir=report_dir,
            include_plots=True
        )
        
        # Find best result
        best_prompt_id, best_result = eval_framework.find_best_result(
            results,
            metric="accuracy",
            higher_is_better=True
        )
        
        # Verify results were created
        assert len(results) == 2  # Two prompts
        assert all(isinstance(r, EvaluationResult) for r in results.values())
        
        # Verify metrics were calculated
        for result in results.values():
            assert "accuracy" in result.metrics
            assert "f1_macro" in result.metrics
            assert "confusion_matrix" in result.metrics
            assert result.model_id == "gemini-1.5-flash"
            
            # Check token usage tracking
            assert "prompt_tokens" in result.token_usage
            assert "completion_tokens" in result.token_usage
            assert "total_tokens" in result.token_usage
        
        # Verify report was generated
        assert os.path.exists(os.path.join(report_dir, "report.json"))
        assert os.path.exists(os.path.join(report_dir, "report.md"))
        assert os.path.exists(os.path.join(report_dir, "plots"))
        
        # Check the MLflow experiment contains runs
        experiment = mlflow.get_experiment_by_name(test_config["experiment_name"])
        assert experiment is not None
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0
        
        # Disable autologging
        gemini_logger.disable_autologging()

    def test_model_comparison_with_autologging(self, temp_dir, mlflow_tracking, test_dataset, 
                                             test_prompts, test_config, mock_gemini_model):
        """Test comparing different Gemini models with autologging."""
        if not GEMINI_AVAILABLE:
            pytest.skip("Google Generative AI package not available")
        
        from google.generativeai import GenerativeModel
        
        # Initialize Gemini autologging
        gemini_logger = setup_gemini_autologging(
            tracking_uri=mlflow_tracking,
            experiment_name=test_config["experiment_name"] + "_comparison",
            log_inputs=True,
            log_outputs=True,
            log_metrics=True,
            log_parameters=True,
            auto_end_run=False
        )
        
        # Initialize evaluation framework
        eval_framework = LLMEvaluationFramework(
            config=test_config,
            experiment_name=test_config["experiment_name"] + "_comparison",
            workspace_dir=test_config["workspace_dir"],
            use_mlflow=True,
            use_langfuse=False,
            use_dvc=False,
            enable_gemini_autologging=True
        )
        
        # Define models to compare
        models = {
            "gemini-1.5-flash": {
                "temperature": 0.0,
                "top_p": 0.95
            },
            "gemini-1.5-pro": {
                "temperature": 0.0,
                "top_p": 0.95
            }
        }
        
        # Define inference function
        def inference_fn(prompt, example, model_id):
            model = GenerativeModel(model_name=model_id)
            response = model.generate_content([prompt, example["text"]])
            
            # Extract token usage
            token_usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0)
            }
            
            return {
                "prediction": response.text.strip(),
                "token_usage": token_usage
            }
        
        # Define metrics function
        def metrics_fn(examples, predictions):
            true_values = [ex["label"] for ex in examples]
            return calculate_classification_metrics(
                true_values, predictions, 
                labels=["positive", "negative", "neutral"]
            )
        
        # Create experiment
        experiment_id = eval_framework.create_experiment(
            name="gemini_model_comparison",
            description="Test comparing different Gemini models"
        )
        
        # Run evaluation
        results = eval_framework.evaluate_models(
            task_id="model_comparison",
            prompt_id=test_prompts["detailed"],
            models=models,
            dataset=test_dataset,
            inference_fn=inference_fn,
            metrics_fn=metrics_fn,
            experiment_id=experiment_id
        )
        
        # Compare results
        if len(results) >= 2:
            models_list = list(results.keys())
            comparison = eval_framework.compare_results(
                baseline_result=results[models_list[0]],
                new_result=results[models_list[1]],
                output_dir=os.path.join(temp_dir, "comparison")
            )
        
        # Verify results were created
        assert len(results) == 2  # Two models
        assert all(isinstance(r, EvaluationResult) for r in results.values())
        
        # Verify metrics were calculated
        for model_id, result in results.items():
            assert "accuracy" in result.metrics
            assert "f1_macro" in result.metrics
            assert "confusion_matrix" in result.metrics
            assert result.model_id == model_id
            
            # Check token usage tracking
            assert "prompt_tokens" in result.token_usage
            assert "completion_tokens" in result.token_usage
            assert "total_tokens" in result.token_usage
        
        # Check the MLflow experiment contains runs
        experiment = mlflow.get_experiment_by_name(test_config["experiment_name"] + "_comparison")
        assert experiment is not None
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0
        
        # Disable autologging
        gemini_logger.disable_autologging()