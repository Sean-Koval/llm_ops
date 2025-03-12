from setuptools import setup, find_packages

setup(
    name="llm_ops_pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.13.0",
        "scikit-learn>=1.2.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pytest>=7.3.1",
        "black>=23.3.0",
        "flake8>=6.0.0",
        "mypy>=1.3.0",
        "wandb>=0.15.0",
        "mlflow>=2.4.0",
    ],
    python_requires=">=3.9",
    author="LLM Ops Team",
    author_email="llmops@example.com",
    description="An end-to-end MLOps pipeline for Large Language Models",
    keywords="mlops, llm, machine learning, pipeline",
)