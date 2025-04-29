# [CS 180 Project] when the ooga is booga

## Deliverable Instructions

Include code for developing/training models and evaluating them (e.g., on the development set), as well as demo code, i.e., runnable code and models for generating predictions for a given input data file, in the form of a Python notebook. It is expected that your submission is organized or structured in such a way that:

- Any code that was written to train models should be separated from the demo code; make sure that your models have been saved and can be readily loaded by the demo code.
- Any code for evaluation (on the development set) is included; this should be separated from your code for training models.
- Someone else outside of your team can follow the code; in other words, provide some documentation, i.e., in-line comments and a README explaining the code structure and how it is run.

Importantly, your README should provide:

1. Attribution to any data sources you used, or code bases you reused.
2. Links to any models that you yourself trained and stored on the cloud.

> [!NOTE]
> If any of your resourcee (e.g., models) are more than 10MB, please do not include them in your submission and instead store them on the cloud.

## Development

### Dependency Installation

This project's dependencies are managed using [`pip`](https://pypi.org/project/pip/). To install the dependencies, run the following command with `pip`:

```bash
pip install -r requirements.txt
```

### Code Linting & Formatting

To ensure consistency in code style and formatting, this project uses [`ruff`](https://github.com/astral-sh/ruff) for code linting and formatting. The following commands may be used to run the `ruff` formatter and linter:

```bash
# Check linting
ruff check

# Fix simple linting errors
uv run ruff check --fix

# Check formatting
uv run ruff format --check

# Fix formatting errors
uv run ruff format
```
