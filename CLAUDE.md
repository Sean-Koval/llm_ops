# CLAUDE.md - Guide for LLM Ops Pipeline

## Commands
- **Build:** `python -m pip install -e .` 
- **Test all:** `pytest`
- **Test single:** `pytest path/to/test_file.py::test_function_name`
- **Lint:** `flake8` and `black .`
- **Type check:** `mypy .`

## Code Style
- **Imports:** Standard library first, then third-party, then local modules
- **Formatting:** Follow PEP 8, use Black for auto-formatting
- **Types:** Use type annotations for functions and method signatures
- **Naming:** snake_case for variables/functions, PascalCase for classes
- **Documentation:** Docstrings using Google style format
- **Error handling:** Use specific exceptions, prefer explicit error handling
- **Logging:** Use structured logging with proper levels (debug, info, error)

## Project Structure
- This repository implements an ML/LLM operations pipeline
- Follows a modular design for flexible deployment options
- Emphasis on best practices: testing, documentation, error handling