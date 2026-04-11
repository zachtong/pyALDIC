# Contributing to pyALDIC

Thank you for your interest in contributing to pyALDIC! This document provides guidelines for contributing to the project.

## Reporting Bugs

Please open an issue on [GitHub Issues](https://github.com/zachtong/pyALDIC/issues) with:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected vs actual behavior
- Python version, OS, and package versions (`pip list`)
- Sample images if relevant (or a minimal synthetic example)

## Suggesting Features

Open an issue with the label `enhancement` and describe:

- The problem your feature would solve
- How you envision it working
- Whether you're willing to help implement it

## Development Setup

```bash
# Clone the repository
git clone https://github.com/zachtong/pyALDIC.git
cd pyALDIC

# Install in development mode
pip install -e ".[dev]"

# Run the test suite
pytest
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Write tests for any new functionality
3. Ensure all tests pass: `pytest`
4. Keep your changes focused — one feature or fix per PR
5. Write a clear PR description explaining what and why

## Code Style

- Follow PEP 8 conventions
- Use descriptive variable names
- Add docstrings to public functions and classes
- Keep functions focused and under 50 lines where practical

## Testing

- Write tests for all new features and bug fixes
- Place tests in the appropriate `tests/` subdirectory
- Run the full suite before submitting: `pytest`
- Run parallel execution for speed: `pytest -n auto`

## Questions?

Open an issue or reach out to the maintainers. We're happy to help!
