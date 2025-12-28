# Contributing to AugmentLens

Thank you for your interest in contributing to AugmentLens! We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code contributions.

---

## Getting Started

### 1. Fork the Repository

Click the "Fork" button on GitHub to create your own copy of the repository.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/augmentlens.git
cd augmentlens
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

---

## Installation (Development Mode)

Install the package in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

This allows you to make changes to the source code and see them reflected immediately without reinstalling.

---

## Running Tests

We use `pytest` for testing. Run the full test suite with:

```bash
pytest tests/ -v
```

All tests should pass before submitting a pull request. If you're adding a new feature, please include tests for it.

---

## Code Style

- Follow [PEP 8](https://pep8.org/) for Python code style
- Use meaningful variable names (avoid generic names like `data`, `x`, `temp`)
- Write docstrings for all public functions and classes
- Keep comments focused on *why*, not *what*

---

## Submitting a Pull Request

1. **Ensure all tests pass** by running `pytest tests/ -v`
2. **Update documentation** if your changes affect the public API
3. **Write a clear PR description** explaining what your change does and why
4. **Reference any related issues** (e.g., "Fixes #123")

We aim to review PRs within a few days. Don't hesitate to ping us if you haven't heard back!

---

## Reporting Bugs

If you find a bug, please [open an issue](https://github.com/Mohamedhendawy312/augmentlens/issues/new) with:

- A clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your environment (Python version, OS, library versions)

---

## Feature Requests

Have an idea for a new feature? We'd love to hear it! Open an issue with the `enhancement` label and describe:

- What problem it solves
- Your proposed solution
- Any alternatives you've considered

---

## Thank You!

Every contribution helps make AugmentLens better. Whether it's fixing a typo, improving docs, or adding a major feature — we appreciate your time and effort! 🙏
