"""
Integration tests for the PromptManager class with MLflow and DVC.
"""

import os
import json
import shutil
import tempfile
from pathlib import Path

import pytest
import mlflow

from llm_ops_pipeline.utils.prompt_management import PromptManager, initialize_dvc_prompt_tracking


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    temp_dir = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    # Initialize git repository
    os.system("git init")
    os.system("git config user.email 'test@example.com'")
    os.system("git config user.name 'Test User'")
    
    try:
        yield temp_dir
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(temp_dir)


@pytest.mark.integration
class TestPromptManagementIntegration:
    def test_mlflow_integration(self, temp_repo):
        """Test integration with MLflow for experiment tracking."""
        # Set up MLflow tracking
        mlflow_dir = os.path.join(temp_repo, "mlruns")
        os.environ["MLFLOW_TRACKING_URI"] = f"file://{mlflow_dir}"
        
        # Initialize prompt manager
        prompt_manager = PromptManager(
            repo_root=temp_repo,
            environment="test"
        )
        
        # Create test prompt
        prompt_id = "integration_test_prompt"
        prompt_manager.create_prompt(
            prompt_id=prompt_id,
            content="This is an integration test prompt with {{variable}}.",
            tags=["integration", "test"]
        )
        
        # Start MLflow run
        with mlflow.start_run(run_name="prompt_integration_test"):
            # Log initial parameters
            mlflow.log_param("test_type", "integration")
            
            # Use prompt and log usage
            prompt_manager.log_prompt_usage(
                prompt_id=prompt_id,
                inputs={"variable": "integration"},
                completion="Test completion for integration",
                metadata={
                    "metrics": {
                        "accuracy": 0.92,
                        "latency": 0.15
                    },
                    "test_info": {
                        "type": "integration"
                    }
                }
            )
            
            run_id = mlflow.active_run().info.run_id
        
        # Verify MLflow artifacts
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        
        assert any(artifact.path == "prompt_usage.json" for artifact in artifacts)
        
        # Verify MLflow parameters
        run = client.get_run(run_id)
        assert run.data.params["prompt_id"] == prompt_id
        assert run.data.params["prompt_version"] == "1.0.0"
        
        # Verify MLflow metrics
        assert run.data.metrics["accuracy"] == 0.92
        assert run.data.metrics["latency"] == 0.15
    
    @pytest.mark.skipif(shutil.which("dvc") is None, reason="DVC not installed")
    def test_dvc_integration(self, temp_repo):
        """Test integration with DVC for prompt dataset versioning."""
        # Skip if DVC is not installed
        if shutil.which("dvc") is None:
            pytest.skip("DVC not installed")
        
        # Initialize DVC
        initialize_dvc_prompt_tracking(temp_repo)
        
        # Verify DVC initialization
        assert os.path.exists(os.path.join(temp_repo, ".dvc"))
        assert os.path.exists(os.path.join(temp_repo, "dvc.yaml"))
        
        # Create dataset directory
        datasets_dir = os.path.join(temp_repo, "prompts", "datasets")
        assert os.path.exists(datasets_dir)
        
        # Add a test dataset
        test_dataset_dir = os.path.join(datasets_dir, "test_examples")
        os.makedirs(test_dataset_dir, exist_ok=True)
        
        # Create a sample dataset file
        with open(os.path.join(test_dataset_dir, "examples.json"), "w") as f:
            json.dump([
                {
                    "input": "This is a test input",
                    "output": "This is the expected output"
                },
                {
                    "input": "Another test input",
                    "output": "Another expected output"
                }
            ], f, indent=2)
        
        # Track with DVC
        os.system("dvc add prompts/datasets")
        
        # Verify DVC tracking
        assert os.path.exists(os.path.join(datasets_dir, ".gitignore"))
        assert os.path.exists(os.path.join(datasets_dir, "test_examples.dvc"))
        
        # Initialize prompt manager
        prompt_manager = PromptManager(
            repo_root=temp_repo,
            environment="test"
        )
        
        # Create a prompt that references the dataset
        prompt_id = "few_shot_example"
        prompt_content = """
        Here are some examples:
        {{#each examples}}
        Input: {{this.input}}
        Output: {{this.output}}
        {{/each}}
        
        Now, given the new input: {{new_input}}
        Output:
        """
        
        prompt_manager.create_prompt(
            prompt_id=prompt_id,
            content=prompt_content,
            description="Few-shot learning prompt",
            tags=["few-shot", "test"]
        )
        
        # Verify prompt creation
        prompt = prompt_manager.get_prompt(prompt_id)
        assert "examples" in prompt_content
        
        # Update dataset
        with open(os.path.join(test_dataset_dir, "examples.json"), "w") as f:
            json.dump([
                {
                    "input": "This is a test input",
                    "output": "This is the expected output"
                },
                {
                    "input": "Another test input",
                    "output": "Another expected output"
                },
                {
                    "input": "A third example",
                    "output": "Third example output"
                }
            ], f, indent=2)
        
        # Update DVC tracking
        os.system("dvc add prompts/datasets")
        
        # Verify we can use the dataset in different versions
        # This would typically be handled by DVC checkout commands
        # which would switch the dataset versions
        assert os.path.exists(os.path.join(test_dataset_dir, "examples.json"))