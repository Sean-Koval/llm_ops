"""
Unit tests for the PromptManager class in llm_ops_pipeline/utils/prompt_management.py
"""

import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest
import mlflow

from llm_ops_pipeline.utils.prompt_management import PromptManager


class TestPromptManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        
        # Create prompts directory structure
        self.prompts_dir = self.repo_root / "prompts"
        self.prompts_dir.mkdir(exist_ok=True)
        
        self.registry_dir = self.prompts_dir / "registry"
        self.registry_dir.mkdir(exist_ok=True)
        
        # Initialize PromptManager with the test directory
        self.prompt_manager = PromptManager(
            repo_root=str(self.repo_root),
            environment="test"
        )
    
    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that PromptManager initializes correctly."""
        # Test directory structure
        assert self.prompt_manager.repo_root == self.repo_root
        assert self.prompt_manager.prompts_dir == self.prompts_dir
        assert self.prompt_manager.registry_dir == self.registry_dir
        assert self.prompt_manager.environment == "test"
        
        # Test default Langfuse status
        assert self.prompt_manager.langfuse_available is False
        assert self.prompt_manager.langfuse is None
    
    def test_create_and_get_prompt(self):
        """Test creating and retrieving a prompt."""
        prompt_id = "test_prompt"
        prompt_content = "This is a test prompt with {{variable}}."
        
        # Create prompt
        result = self.prompt_manager.create_prompt(
            prompt_id=prompt_id,
            content=prompt_content,
            description="Test description",
            tags=["test", "unit"]
        )
        
        # Verify return value
        assert result["content"] == prompt_content
        assert result["metadata"]["description"] == "Test description"
        assert result["metadata"]["tags"] == ["test", "unit"]
        assert result["metadata"]["version"] == "1.0.0"
        
        # Check that files were created
        prompt_path = self.prompts_dir / prompt_id / "prompt.json"
        metadata_path = self.prompts_dir / prompt_id / "metadata.json"
        registry_path = self.registry_dir / "test.json"
        
        assert prompt_path.exists()
        assert metadata_path.exists()
        assert registry_path.exists()
        
        # Test get_prompt retrieves the same prompt
        retrieved_prompt = self.prompt_manager.get_prompt(prompt_id)
        assert retrieved_prompt["content"] == prompt_content
        
        # Test registry contains correct version
        with open(registry_path, "r") as f:
            registry = json.load(f)
            assert registry[prompt_id] == "1.0.0"
    
    def test_update_prompt(self):
        """Test updating an existing prompt."""
        prompt_id = "update_test_prompt"
        initial_content = "Initial prompt content."
        updated_content = "Updated prompt content."
        
        # Create initial prompt
        self.prompt_manager.create_prompt(
            prompt_id=prompt_id,
            content=initial_content
        )
        
        # Update prompt
        result = self.prompt_manager.update_prompt(
            prompt_id=prompt_id,
            content=updated_content,
            description="Updated description"
        )
        
        # Verify return value
        assert result["content"] == updated_content
        assert result["metadata"]["description"] == "Updated description"
        assert result["metadata"]["version"] == "1.0.1"  # Auto-incremented patch version
        
        # Test get_prompt retrieves updated prompt
        retrieved_prompt = self.prompt_manager.get_prompt(prompt_id)
        assert retrieved_prompt["content"] == updated_content
    
    def test_delete_prompt(self):
        """Test deleting a prompt."""
        prompt_id = "delete_test_prompt"
        
        # Create prompt
        self.prompt_manager.create_prompt(
            prompt_id=prompt_id,
            content="Prompt to be deleted."
        )
        
        # Verify prompt exists
        prompt_dir = self.prompts_dir / prompt_id
        assert prompt_dir.exists()
        
        # Delete prompt
        self.prompt_manager.delete_prompt(prompt_id)
        
        # Verify prompt directory was removed
        assert not prompt_dir.exists()
        
        # Verify prompt was removed from registry
        registry_path = self.registry_dir / "test.json"
        with open(registry_path, "r") as f:
            registry = json.load(f)
            assert prompt_id not in registry
        
        # Verify get_prompt raises ValueError
        with pytest.raises(ValueError):
            self.prompt_manager.get_prompt(prompt_id)
    
    def test_list_prompts(self):
        """Test listing all prompts."""
        # Create multiple prompts
        self.prompt_manager.create_prompt(
            prompt_id="prompt1",
            content="Content 1"
        )
        self.prompt_manager.create_prompt(
            prompt_id="prompt2",
            content="Content 2",
            description="Description 2"
        )
        
        # Test list_prompts
        prompts = self.prompt_manager.list_prompts()
        
        # Verify list contains both prompts
        assert len(prompts) == 2
        assert any(p["id"] == "prompt1" for p in prompts)
        assert any(p["id"] == "prompt2" for p in prompts)
        
        # Verify prompt details
        prompt2 = next(p for p in prompts if p["id"] == "prompt2")
        assert prompt2["description"] == "Description 2"
        assert prompt2["version"] == "1.0.0"
        assert prompt2["source"] == "git"
    
    @patch("llm_ops_pipeline.utils.prompt_management.Langfuse")
    def test_langfuse_integration(self, mock_langfuse_class):
        """Test integration with Langfuse."""
        # Setup mock Langfuse client
        mock_langfuse = MagicMock()
        mock_langfuse_class.return_value = mock_langfuse
        
        # Mock prompt version
        mock_prompt_version = MagicMock()
        mock_prompt_version.prompt = "Langfuse prompt content"
        mock_prompt_version.version = "2.0.0"
        mock_prompt_version.id = "langfuse-id"
        mock_prompt_version.created_at = "2023-01-01T00:00:00Z"
        mock_prompt_version.updated_at = "2023-01-02T00:00:00Z"
        
        # Configure mock
        mock_langfuse.prompts.get_prompt_version_for_environment.return_value = mock_prompt_version
        
        # Initialize PromptManager with Langfuse
        prompt_manager = PromptManager(
            repo_root=str(self.repo_root),
            environment="test",
            langfuse_api_key="test-key",
            langfuse_secret_key="test-secret"
        )
        
        # Verify Langfuse was initialized
        assert prompt_manager.langfuse_available is True
        
        # Test get_prompt using Langfuse
        prompt_id = "langfuse_prompt"
        prompt = prompt_manager.get_prompt(prompt_id)
        
        # Verify prompt content comes from Langfuse
        assert prompt["content"] == "Langfuse prompt content"
        assert prompt["version"] == "2.0.0"
        
        # Verify Langfuse was called correctly
        mock_langfuse.prompts.get_prompt_version_for_environment.assert_called_once_with(
            prompt_name=prompt_id,
            environment="test"
        )
        
        # Verify prompt was synced to disk
        prompt_path = self.prompts_dir / prompt_id / "prompt.json"
        assert prompt_path.exists()
    
    @patch("mlflow.active_run")
    @patch("mlflow.log_metric")
    @patch("mlflow.log_param")
    @patch("mlflow.log_artifact")
    def test_mlflow_logging(self, mock_log_artifact, mock_log_param, mock_log_metric, mock_active_run):
        """Test MLflow integration for prompt usage logging."""
        # Setup MLflow mocks
        mock_active_run.return_value = True
        
        # Create prompt
        prompt_id = "mlflow_test_prompt"
        self.prompt_manager.create_prompt(
            prompt_id=prompt_id,
            content="MLflow test prompt"
        )
        
        # Log prompt usage
        self.prompt_manager.log_prompt_usage(
            prompt_id=prompt_id,
            inputs={"variable": "test"},
            completion="Test completion",
            metadata={
                "metrics": {
                    "accuracy": 0.95,
                    "latency": 0.2
                }
            }
        )
        
        # Verify MLflow was called correctly
        mock_log_param.assert_any_call("prompt_id", prompt_id)
        mock_log_param.assert_any_call("prompt_version", "1.0.0")
        mock_log_metric.assert_any_call("accuracy", 0.95)
        mock_log_metric.assert_any_call("latency", 0.2)
        mock_log_artifact.assert_called_once()
    
    def test_with_prompt_decorator(self):
        """Test the with_prompt decorator."""
        # Create prompt
        prompt_id = "decorator_test"
        prompt_content = "This is a {{variable}} prompt."
        
        self.prompt_manager.create_prompt(
            prompt_id=prompt_id,
            content=prompt_content
        )
        
        # Define test function with decorator
        @self.prompt_manager.with_prompt(prompt_id)
        def test_function(prompt, variable):
            # Emulate template substitution
            return prompt["content"].replace("{{variable}}", variable)
        
        # Test function with decorator
        result = test_function("test")
        
        # Verify result
        assert result == "This is a test prompt."


if __name__ == "__main__":
    unittest.main()