"""
Prompt Management System - Integrates Langfuse with MLflow and DVC for resilient 
prompt versioning and management.
"""

import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import functools

import mlflow
from langfuse import Langfuse
from langfuse.api.resources.prompts import Prompt, PromptVersion

logger = logging.getLogger(__name__)

class PromptManager:
    """
    Manages prompts across environments with Langfuse integration.
    Provides resilience through git-based backup and MLflow tracking.
    """
    
    def __init__(
        self,
        langfuse_api_key: Optional[str] = None,
        langfuse_secret_key: Optional[str] = None,
        langfuse_host: Optional[str] = None,
        repo_root: Optional[str] = None,
        environment: str = "development",
        cache_ttl: int = 3600  # 1 hour cache TTL by default
    ):
        # Initialize Langfuse client if credentials are provided
        self.langfuse_available = False
        if langfuse_api_key and langfuse_secret_key:
            try:
                self.langfuse = Langfuse(
                    api_key=langfuse_api_key,
                    secret_key=langfuse_secret_key,
                    host=langfuse_host
                )
                self.langfuse_available = True
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse: {e}")
                self.langfuse = None
        else:
            self.langfuse = None
            
        # Set up repository paths
        if repo_root:
            self.repo_root = Path(repo_root)
        else:
            # Try to find git root
            curr_dir = Path.cwd()
            while curr_dir != curr_dir.parent:
                if (curr_dir / ".git").exists():
                    self.repo_root = curr_dir
                    break
                curr_dir = curr_dir.parent
            else:
                self.repo_root = Path.cwd()
                
        self.prompts_dir = self.repo_root / "prompts"
        self.prompts_dir.mkdir(exist_ok=True)
        
        # Set up registry directory
        self.registry_dir = self.prompts_dir / "registry"
        self.registry_dir.mkdir(exist_ok=True)
        
        # Environment settings
        self.environment = environment
        
        # In-memory cache
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = cache_ttl
        
    def _get_prompt_path(self, prompt_id: str) -> Path:
        """Get the file path for a prompt based on its ID."""
        return self.prompts_dir / prompt_id / "prompt.json"
    
    def _get_prompt_metadata_path(self, prompt_id: str) -> Path:
        """Get the metadata file path for a prompt based on its ID."""
        return self.prompts_dir / prompt_id / "metadata.json"
    
    def _get_registry_path(self) -> Path:
        """Get the registry file path for the current environment."""
        return self.registry_dir / f"{self.environment}.json"
    
    def _compute_hash(self, content: Dict[str, Any]) -> str:
        """Compute a hash for the prompt content."""
        serialized = json.dumps(content, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def _load_registry(self) -> Dict[str, str]:
        """Load the environment-specific registry."""
        registry_path = self._get_registry_path()
        if registry_path.exists():
            with open(registry_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_registry(self, registry: Dict[str, str]) -> None:
        """Save the environment-specific registry."""
        registry_path = self._get_registry_path()
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
    
    def _load_prompt_from_disk(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Load a prompt from disk."""
        prompt_path = self._get_prompt_path(prompt_id)
        if not prompt_path.exists():
            return None
        
        with open(prompt_path, "r") as f:
            prompt_data = json.load(f)
        
        metadata_path = self._get_prompt_metadata_path(prompt_id)
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            prompt_data["metadata"] = metadata
        
        return prompt_data
    
    def _save_prompt_to_disk(
        self, 
        prompt_id: str, 
        content: Dict[str, Any], 
        metadata: Dict[str, Any]
    ) -> None:
        """Save a prompt to disk."""
        prompt_dir = self.prompts_dir / prompt_id
        prompt_dir.mkdir(exist_ok=True, parents=True)
        
        prompt_path = prompt_dir / "prompt.json"
        with open(prompt_path, "w") as f:
            json.dump(content, f, indent=2)
        
        metadata_path = prompt_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def get_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get a prompt by ID, with fallback mechanisms.
        
        1. Try to get from cache
        2. Try to get from Langfuse
        3. Fall back to disk
        """
        # Check cache first
        if prompt_id in self.cache:
            import time
            if time.time() - self.cache_timestamps[prompt_id] < self.cache_ttl:
                return self.cache[prompt_id]
        
        # Try Langfuse first if available
        if self.langfuse_available:
            try:
                # Get the prompt version that's active for the current environment
                prompt_version = self.langfuse.prompts.get_prompt_version_for_environment(
                    prompt_name=prompt_id,
                    environment=self.environment
                )
                
                if prompt_version:
                    prompt_content = {
                        "content": prompt_version.prompt,
                        "version": prompt_version.version,
                    }
                    
                    # Cache the prompt
                    import time
                    self.cache[prompt_id] = prompt_content
                    self.cache_timestamps[prompt_id] = time.time()
                    
                    # Sync to disk for resilience
                    self._save_prompt_to_disk(
                        prompt_id=prompt_id,
                        content={"content": prompt_version.prompt},
                        metadata={
                            "version": prompt_version.version,
                            "langfuse_id": prompt_version.id,
                            "created_at": prompt_version.created_at,
                            "updated_at": prompt_version.updated_at,
                            "hash": self._compute_hash({"content": prompt_version.prompt})
                        }
                    )
                    
                    # Update registry
                    registry = self._load_registry()
                    registry[prompt_id] = prompt_version.version
                    self._save_registry(registry)
                    
                    return prompt_content
            except Exception as e:
                logger.warning(f"Failed to get prompt from Langfuse: {e}")
        
        # Fall back to disk
        prompt_data = self._load_prompt_from_disk(prompt_id)
        if prompt_data:
            # Cache the prompt
            import time
            self.cache[prompt_id] = prompt_data
            self.cache_timestamps[prompt_id] = time.time()
            return prompt_data
        
        # If we get here, prompt not found
        raise ValueError(f"Prompt '{prompt_id}' not found in any source")
    
    def create_prompt(
        self, 
        prompt_id: str, 
        content: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new prompt.
        
        Saves to Langfuse if available, always saves to disk.
        """
        if not name:
            name = prompt_id
            
        prompt_content = {"content": content}
        metadata = {
            "name": name,
            "description": description,
            "tags": tags or [],
            "created_at": self._get_iso_datetime(),
            "updated_at": self._get_iso_datetime(),
            "version": "1.0.0",
            "hash": self._compute_hash(prompt_content)
        }
        
        # Save to Langfuse if available
        if self.langfuse_available:
            try:
                # Check if prompt exists
                try:
                    prompt = self.langfuse.prompts.get(prompt_id)
                except:
                    # Create new prompt
                    prompt = self.langfuse.prompts.create(
                        name=prompt_id,
                        description=description or ""
                    )
                
                # Create new version
                prompt_version = self.langfuse.prompts.create_version(
                    prompt_name=prompt_id,
                    prompt=content,
                    name=f"v1.0.0",
                    is_active=True
                )
                
                metadata["langfuse_id"] = prompt_version.id
                metadata["version"] = "1.0.0"
            except Exception as e:
                logger.warning(f"Failed to create prompt in Langfuse: {e}")
        
        # Save to disk
        self._save_prompt_to_disk(
            prompt_id=prompt_id,
            content=prompt_content,
            metadata=metadata
        )
        
        # Update registry
        registry = self._load_registry()
        registry[prompt_id] = "1.0.0"
        self._save_registry(registry)
        
        # Add to cache
        import time
        prompt_data = {**prompt_content, "metadata": metadata}
        self.cache[prompt_id] = prompt_data
        self.cache_timestamps[prompt_id] = time.time()
        
        return prompt_data
    
    def update_prompt(
        self,
        prompt_id: str,
        content: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Update an existing prompt.
        
        Updates in Langfuse if available, always updates on disk.
        """
        # Check if prompt exists
        try:
            existing_prompt = self.get_prompt(prompt_id)
        except ValueError:
            # Create if it doesn't exist
            return self.create_prompt(
                prompt_id=prompt_id,
                content=content,
                description=description,
                tags=tags
            )
        
        # Get existing metadata
        metadata_path = self._get_prompt_metadata_path(prompt_id)
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {
                "name": prompt_id,
                "description": "",
                "tags": [],
                "created_at": self._get_iso_datetime(),
                "version": "1.0.0"
            }
        
        # Update version
        if version:
            new_version = version
        else:
            # Auto-increment version
            current_version = metadata.get("version", "1.0.0")
            version_parts = current_version.split(".")
            patch = int(version_parts[2]) + 1
            new_version = f"{version_parts[0]}.{version_parts[1]}.{patch}"
        
        # Update metadata
        prompt_content = {"content": content}
        metadata.update({
            "description": description or metadata.get("description", ""),
            "tags": tags or metadata.get("tags", []),
            "updated_at": self._get_iso_datetime(),
            "version": new_version,
            "hash": self._compute_hash(prompt_content)
        })
        
        # Update in Langfuse if available
        if self.langfuse_available:
            try:
                prompt_version = self.langfuse.prompts.create_version(
                    prompt_name=prompt_id,
                    prompt=content,
                    name=f"v{new_version}",
                    is_active=True
                )
                
                metadata["langfuse_id"] = prompt_version.id
            except Exception as e:
                logger.warning(f"Failed to update prompt in Langfuse: {e}")
        
        # Save to disk
        self._save_prompt_to_disk(
            prompt_id=prompt_id,
            content=prompt_content,
            metadata=metadata
        )
        
        # Update registry
        registry = self._load_registry()
        registry[prompt_id] = new_version
        self._save_registry(registry)
        
        # Update cache
        import time
        prompt_data = {**prompt_content, "metadata": metadata}
        self.cache[prompt_id] = prompt_data
        self.cache_timestamps[prompt_id] = time.time()
        
        return prompt_data
    
    def _get_iso_datetime(self) -> str:
        """Get current datetime in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"
    
    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all prompts in the system."""
        prompts = []
        
        # Get prompts from disk
        for prompt_dir in self.prompts_dir.iterdir():
            if prompt_dir.is_dir() and prompt_dir.name != "registry":
                prompt_id = prompt_dir.name
                metadata_path = prompt_dir / "metadata.json"
                
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    prompts.append({
                        "id": prompt_id,
                        "name": metadata.get("name", prompt_id),
                        "version": metadata.get("version", "unknown"),
                        "description": metadata.get("description", ""),
                        "updated_at": metadata.get("updated_at", ""),
                        "source": "git"
                    })
        
        # Add prompts from Langfuse if available
        if self.langfuse_available:
            try:
                langfuse_prompts = self.langfuse.prompts.list()
                
                # Filter out prompts we already know about
                existing_ids = {p["id"] for p in prompts}
                
                for prompt in langfuse_prompts:
                    if prompt.name not in existing_ids:
                        prompts.append({
                            "id": prompt.name,
                            "name": prompt.name,
                            "version": "latest",  # We'd need to query each prompt for its version
                            "description": prompt.description or "",
                            "updated_at": prompt.updated_at or "",
                            "source": "langfuse"
                        })
            except Exception as e:
                logger.warning(f"Failed to list prompts from Langfuse: {e}")
        
        return prompts
    
    def delete_prompt(self, prompt_id: str) -> None:
        """Delete a prompt from the system."""
        # Delete from Langfuse if available
        if self.langfuse_available:
            try:
                self.langfuse.prompts.delete(prompt_id)
            except Exception as e:
                logger.warning(f"Failed to delete prompt from Langfuse: {e}")
        
        # Delete from disk
        prompt_dir = self.prompts_dir / prompt_id
        if prompt_dir.exists():
            import shutil
            shutil.rmtree(prompt_dir)
        
        # Remove from registry
        registry = self._load_registry()
        if prompt_id in registry:
            del registry[prompt_id]
            self._save_registry(registry)
        
        # Remove from cache
        if prompt_id in self.cache:
            del self.cache[prompt_id]
            del self.cache_timestamps[prompt_id]
    
    def sync_from_langfuse(self) -> Dict[str, Any]:
        """
        Sync all prompts from Langfuse to the local repository.
        
        Returns a report of the sync operation.
        """
        if not self.langfuse_available:
            raise ValueError("Langfuse is not available")
        
        report = {
            "synced": 0,
            "errors": 0,
            "details": []
        }
        
        try:
            # Get all prompts from Langfuse
            langfuse_prompts = self.langfuse.prompts.list()
            
            for prompt in langfuse_prompts:
                try:
                    # Get active version for current environment
                    prompt_version = self.langfuse.prompts.get_prompt_version_for_environment(
                        prompt_name=prompt.name,
                        environment=self.environment
                    )
                    
                    if prompt_version:
                        # Save to disk
                        prompt_content = {"content": prompt_version.prompt}
                        metadata = {
                            "name": prompt.name,
                            "description": prompt.description or "",
                            "created_at": prompt.created_at or self._get_iso_datetime(),
                            "updated_at": prompt_version.updated_at or self._get_iso_datetime(),
                            "version": prompt_version.version,
                            "langfuse_id": prompt_version.id,
                            "hash": self._compute_hash(prompt_content)
                        }
                        
                        self._save_prompt_to_disk(
                            prompt_id=prompt.name,
                            content=prompt_content,
                            metadata=metadata
                        )
                        
                        report["synced"] += 1
                        report["details"].append({
                            "prompt_id": prompt.name,
                            "version": prompt_version.version,
                            "status": "synced"
                        })
                except Exception as e:
                    report["errors"] += 1
                    report["details"].append({
                        "prompt_id": prompt.name,
                        "error": str(e),
                        "status": "error"
                    })
            
            # Update registry
            registry = self._load_registry()
            for detail in report["details"]:
                if detail["status"] == "synced":
                    registry[detail["prompt_id"]] = detail["version"]
            
            self._save_registry(registry)
        
        except Exception as e:
            logger.error(f"Failed to sync prompts from Langfuse: {e}")
            raise
        
        return report
    
    def log_prompt_usage(
        self, 
        prompt_id: str, 
        inputs: Dict[str, Any],
        completion: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log prompt usage to both Langfuse and MLflow.
        
        This is useful for tracking prompt performance over time.
        """
        # Get prompt
        try:
            prompt_data = self.get_prompt(prompt_id)
            prompt_version = prompt_data.get("metadata", {}).get("version", "unknown")
        except ValueError:
            logger.warning(f"Prompt '{prompt_id}' not found, logging usage without version info")
            prompt_data = None
            prompt_version = "unknown"
        
        # Prepare logging data
        usage_data = {
            "prompt_id": prompt_id,
            "prompt_version": prompt_version,
            "inputs": inputs,
            "completion": completion,
            "metadata": metadata or {},
            "timestamp": self._get_iso_datetime()
        }
        
        # Log to Langfuse if available
        if self.langfuse_available:
            try:
                trace = self.langfuse.trace(
                    name=f"prompt:{prompt_id}",
                    metadata={
                        "prompt_version": prompt_version,
                        **usage_data["metadata"]
                    }
                )
                
                generation = trace.generation(
                    name=f"generate:{prompt_id}",
                    model="custom",
                    prompt=str(inputs),
                    completion=completion
                )
            except Exception as e:
                logger.warning(f"Failed to log prompt usage to Langfuse: {e}")
        
        # Log to MLflow if active
        try:
            if mlflow.active_run():
                # Log metrics if provided
                if "metrics" in usage_data["metadata"]:
                    for key, value in usage_data["metadata"]["metrics"].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
                
                # Log prompt as artifact
                with open("prompt_usage.json", "w") as f:
                    json.dump(usage_data, f, indent=2)
                
                mlflow.log_artifact("prompt_usage.json")
                
                # Log parameters
                mlflow.log_param("prompt_id", prompt_id)
                mlflow.log_param("prompt_version", prompt_version)
        except Exception as e:
            logger.warning(f"Failed to log prompt usage to MLflow: {e}")

    def with_prompt(self, prompt_id: str):
        """
        Decorator for using a prompt in a function.
        
        Example:
            @prompt_manager.with_prompt("classification/sentiment")
            def classify_sentiment(prompt, text):
                # prompt contains the loaded prompt
                inputs = {"text": text}
                # Use the prompt...
                return result
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                prompt = self.get_prompt(prompt_id)
                return func(prompt, *args, **kwargs)
            return wrapper
        return decorator


def initialize_dvc_prompt_tracking(repo_root: str) -> None:
    """
    Initialize DVC for tracking prompt datasets.
    
    This is useful for versioning large datasets used for few-shot learning.
    """
    import subprocess
    import os
    
    # Initialize DVC if not already
    prompts_dir = os.path.join(repo_root, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    
    # Check if DVC is initialized
    if not os.path.exists(os.path.join(repo_root, ".dvc")):
        subprocess.run(["dvc", "init"], cwd=repo_root, check=True)
    
    # Add prompts directory to DVC
    dvc_yaml_path = os.path.join(repo_root, "dvc.yaml")
    if not os.path.exists(dvc_yaml_path):
        with open(dvc_yaml_path, "w") as f:
            f.write("""
stages:
  prompt_datasets:
    cmd: echo "Prompt datasets tracked by DVC"
    deps:
      - prompts/datasets
    outs:
      - prompts/datasets
""")
    
    # Create datasets directory if it doesn't exist
    datasets_dir = os.path.join(prompts_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Add to DVC
    subprocess.run(["dvc", "add", "prompts/datasets"], cwd=repo_root, check=True)
    
    print("DVC prompt tracking initialized")