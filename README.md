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


Usage

Example: Solving the Rosenbrock Function
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
```


## Usage

### Example: Solving the Rosenbrock Function
```python
import torch
from optimizer.optimizer import Optimizer
from optimizer.objective import Objective
from optimizer.constraints import Constraint

# Define the Rosenbrock function
def objective_func(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Initialize
objective = Objective(objective_func)
constraints = Constraint([])  # No constraints
x0 = torch.tensor([0.0, 0.0], requires_grad=True)

# Create and run the optimizer
optimizer = Optimizer(objective, constraints, x0, max_iter=100, tolerance=1e-6)
optimal_x = optimizer.stochastic_sqp()
print("Optimal solution:", optimal_x.detach().numpy())
```

## Project Structure
stochastic-sqp-optimizer/
├── optimizer/               # Core modules for constraints, objective, and optimizer
│   ├── __init__.py
│   ├── constraints.py
│   ├── objective.py
│   ├── optimizer.py
│   ├── parameters.py
├── tests/                   # Unit tests
│   ├── __init__.py
│   ├── test_constraints.py
│   ├── test_objective.py
├── examples/                # Usage examples
│   ├── example_rosenbrock.py
│   ├── example_constrained.py
├── README.md                # Project documentation
├── LICENSE                  # License file
├── requirements.txt         # Python dependencies
└── setup.py                 # Installation configuration

## License
License

This project is licensed under the MIT License - see the [https://opensource.org/license/mit][LICENSE].
