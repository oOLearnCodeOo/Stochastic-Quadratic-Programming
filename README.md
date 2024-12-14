# Stochastic SQP Optimizer

A Python implementation of Stochastic Sequential Quadratic Programming (SQP) for solving constrained and unconstrained optimization problems.

## Features

- Supports both constrained and unconstrained optimization.
- Automatically checks LICQ (Linear Independence Constraint Qualification).
- Includes examples for solving optimization problems, such as the Rosenbrock function and constrained quadratic problems.

## Reference

This project is inspired by research and methodologies related to the SQP algorithm. For more details on SQP algorithms and research, visit **[Bao Yu Zhou's homepage](https://baoyuzhou18.github.io/)**.

## Installation

This project does **not require permanent changes to your `PYTHONPATH`**. You can run the code by temporarily adding the project directory to your `PYTHONPATH` during execution or directly modifying `sys.path` in your scripts.

### Temporary Path Configuration

If you encounter module import issues (e.g., `ModuleNotFoundError`), you can temporarily add the project root directory to the Python path in your script. Add the following code at the beginning of your script:

```python
import sys
import os

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# Add the project root to sys.path
sys.path.insert(0, project_root)

# Now you can import project modules
from optimizer.optimizer import Optimizer
from optimizer.objective import Objective
from optimizer.constraints import Constraint
