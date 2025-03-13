"""
Unit tests for the Gemini MLflow integration.
Tests the autologging functionality, token tracking, and cost estimation.
"""

import os
import pytest
import json
from unittest.mock import patch, MagicMock, call

import mlflow

# Import module to test
from llm_ops_pipeline.utils.gemini_mlflow import (
    GeminiMLflowLogger, 
    setup_gemini_autologging,
    GEMINI_AVAILABLE
)

# Skip tests if Gemini is not available
pytestmark = pytest.mark.skipif(not GEMINI_AVAILABLE, reason="Google Generative AI package not available")

class TestGeminiMLflowIntegration:
    """Test suite for Gemini MLflow integration."""
    
    def test_gemini_logger_initialization(self):
        """Test initialization of the GeminiMLflowLogger class."""
        logger = GeminiMLflowLogger(
            tracking_uri=None,
            experiment_name="test_experiment",
            log_inputs=True,
            log_outputs=True
        )
        
        assert logger.log_inputs is True
        assert logger.log_outputs is True
        assert logger.log_metrics is True
        assert logger.log_parameters is True
        assert logger.auto_end_run is False
    
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_initialize_with_tracking_uri(self, mock_set_experiment, mock_set_tracking_uri):
        """Test initialization with a tracking URI."""
        tracking_uri = "http://localhost:5000"
        experiment_name = "test_experiment"
        
        logger = GeminiMLflowLogger(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name
        )
        
        mock_set_tracking_uri.assert_called_once_with(tracking_uri)
        mock_set_experiment.assert_called_once_with(experiment_name)
    
    @patch('google.generativeai.configure')
    def test_initialize_gemini_with_api_key(self, mock_configure):
        """Test initialization of Gemini with an API key."""
        api_key = "test_api_key"
        logger = GeminiMLflowLogger()
        
        logger.initialize_gemini(api_key=api_key)
        
        mock_configure.assert_called_once_with(api_key=api_key)
    
    @patch('google.generativeai.configure')
    @patch('os.environ.get')
    def test_initialize_gemini_with_env_var(self, mock_env_get, mock_configure):
        """Test initialization of Gemini with an environment variable."""
        mock_env_get.return_value = "env_api_key"
        
        logger = GeminiMLflowLogger()
        logger.initialize_gemini()
        
        mock_configure.assert_called_once_with(api_key="env_api_key")
    
    @patch('llm_ops_pipeline.utils.gemini_mlflow.GenerativeModel.generate_content')
    def test_monkey_patching(self, mock_generate_content):
        """Test monkey patching of Gemini methods."""
        if not GEMINI_AVAILABLE:
            pytest.skip("Google Generative AI package not available")
        
        from google.generativeai import GenerativeModel
        
        logger = GeminiMLflowLogger()
        
        # Check original methods before patching
        assert not hasattr(GenerativeModel, '_extract_generation_config')
        
        # Enable autologging
        logger.enable_autologging()
        
        # Check that methods were patched
        assert hasattr(GenerativeModel, '_extract_generation_config')
        assert hasattr(GenerativeModel, 'log_inputs')
        assert hasattr(GenerativeModel, 'max_input_length')
        
        # Disable autologging
        logger.disable_autologging()
    
    def test_extract_generation_config(self):
        """Test extraction of generation config parameters."""
        logger = GeminiMLflowLogger()
        
        # Test with dictionary input
        config_dict = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024
        }
        
        result = logger._extract_generation_config(config_dict)
        assert result == config_dict
        
        # Test with object input
        class GenerationConfig:
            def __init__(self):
                self.temperature = 0.7
                self.top_p = 0.95
                self.top_k = 40
                self.max_output_tokens = 1024
                self.stop_sequences = None
        
        config_obj = GenerationConfig()
        result = logger._extract_generation_config(config_obj)
        
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.95
        assert result["top_k"] == 40
        assert result["max_output_tokens"] == 1024
        assert "stop_sequences" not in result
    
    def test_truncate_content(self):
        """Test content truncation functionality."""
        logger = GeminiMLflowLogger()
        
        # Test string truncation
        long_string = "a" * 1000
        truncated = logger._truncate_content(long_string)
        assert len(truncated) < 1000
        assert truncated.endswith("...")
        
        # Test short string (no truncation)
        short_string = "a" * 100
        truncated = logger._truncate_content(short_string)
        assert truncated == short_string
        assert not truncated.endswith("...")
    
    def test_estimate_cost(self):
        """Test the cost estimation functionality."""
        logger = GeminiMLflowLogger()
        
        # Test cost estimation for different models
        token_usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500
        }
        
        # Test with gemini-1.0-pro
        cost_pro = logger._estimate_cost(token_usage, "gemini-1.0-pro")
        expected_pro = (1000 / 1000 * 0.00125) + (500 / 1000 * 0.00375)
        assert cost_pro == expected_pro
        
        # Test with gemini-1.5-flash
        cost_flash = logger._estimate_cost(token_usage, "gemini-1.5-flash")
        expected_flash = (1000 / 1000 * 0.00025) + (500 / 1000 * 0.0007)
        assert cost_flash == expected_flash
        
        # Test with default (unknown model)
        cost_default = logger._estimate_cost(token_usage, "unknown-model")
        expected_default = (1000 / 1000 * 0.00125) + (500 / 1000 * 0.00375)
        assert cost_default == expected_default

    @patch('mlflow.start_run')
    @patch('mlflow.end_run')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_param')
    @patch('mlflow.set_tag')
    @patch('mlflow.log_text')
    @patch('mlflow.active_run')
    def test_autologging_functionality(self, mock_active_run, mock_log_text, mock_set_tag, 
                                      mock_log_param, mock_log_metric, mock_end_run, 
                                      mock_start_run):
        """Test the complete autologging functionality."""
        if not GEMINI_AVAILABLE:
            pytest.skip("Google Generative AI package not available")
        
        from google.generativeai import GenerativeModel
        
        # Mock MLflow active run
        mock_active_run.return_value = None
        
        # Set up logger
        logger = GeminiMLflowLogger()
        logger.enable_autologging()
        
        # Create a mock response
        class MockResponse:
            def __init__(self):
                self.text = "This is a test response"
                self.usage_metadata = type('UsageMetadata', (), {
                    'prompt_token_count': 10, 
                    'candidates_token_count': 5,
                    'total_token_count': 15
                })
                self.candidates = [
                    type('Candidate', (), {
                        'safety_ratings': [
                            type('SafetyRating', (), {
                                'category': type('Category', (), {'name': 'HARM_CATEGORY'}),
                                'probability': 0.1
                            })
                        ]
                    })
                ]
        
        # Patch the original generate_content method
        original_generate_content = GenerativeModel.generate_content
        
        try:
            # Create a mock for the original method
            mock_response = MockResponse()
            mock_original = MagicMock(return_value=mock_response)
            GenerativeModel.generate_content = mock_original
            
            # Create model and call method
            model = GenerativeModel("gemini-1.5-flash")
            model._model_name = "gemini-1.5-flash"  # Set model name
            
            # Call the patched method
            result = model.generate_content("Test prompt", generation_config={"temperature": 0.7})
            
            # Verify MLflow tracking calls
            mock_start_run.assert_called_once()
            mock_log_param.assert_any_call("inputs", "Test prompt")
            mock_log_text.assert_called_once_with("This is a test response", "output.txt")
            
            # Verify token metrics were logged
            mock_log_metric.assert_any_call("prompt_tokens", 10)
            mock_log_metric.assert_any_call("completion_tokens", 5)
            mock_log_metric.assert_any_call("total_tokens", 15)
            
            # Verify cost estimation was logged
            expected_cost = (10 / 1000 * 0.00025) + (5 / 1000 * 0.0007)
            mock_log_metric.assert_any_call("cost_estimate_usd", expected_cost)
            
            # Verify safety metrics were logged
            mock_log_metric.assert_any_call("safety_HARM_CATEGORY", 0.1)
            
            # Verify result is returned correctly
            assert result == mock_response
            
        finally:
            # Restore original method
            GenerativeModel.generate_content = original_generate_content
            
            # Disable autologging
            logger.disable_autologging()
    
    @patch('llm_ops_pipeline.utils.gemini_mlflow.GeminiMLflowLogger')
    def test_setup_gemini_autologging(self, mock_logger_class):
        """Test the setup_gemini_autologging utility function."""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        logger = setup_gemini_autologging(
            api_key="test_api_key",
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment"
        )
        
        # Verify GeminiMLflowLogger was created with correct args
        mock_logger_class.assert_called_once_with(
            tracking_uri="http://localhost:5000",
            experiment_name="test_experiment",
            log_inputs=True,
            log_outputs=True,
            log_metrics=True,
            log_parameters=True,
            auto_end_run=False
        )
        
        # Verify methods were called
        mock_logger.initialize_gemini.assert_called_once_with(
            api_key="test_api_key",
            project_id=None,
            location=None
        )
        mock_logger.enable_autologging.assert_called_once()
        
        # Verify logger was returned
        assert logger == mock_logger