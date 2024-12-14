import sys
import os
import torch
# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# 将项目根目录添加到 sys.path
sys.path.insert(0, project_root)
from optimizer.optimizer import Optimizer
from optimizer.objective import Objective
from optimizer.constraints import Constraint

# 定义目标函数和约束
def objective_func(x):
    A = torch.tensor([[3.0, 1.0], [1.0, 2.0]], requires_grad=False)
    b_obj = torch.tensor([1.0, 1.0], requires_grad=False)
    residual = A @ x - b_obj
    return 0.5 * torch.sum(residual**2)

def constraint_func_1(x):
    Q = torch.tensor([[1.0, 0.5], [0.5, 1.0]], requires_grad=False)
    b = torch.tensor([1.0, -1.0], requires_grad=False)
    c = -0.5
    return 0.5 * x.T @ Q @ x + b.T @ x + c

def constraint_func_2(x):
    return torch.sum(x) - 1.0
# 初始化
objective = Objective(objective_func)
constraints = Constraint([constraint_func_1, constraint_func_2])
x0 = torch.tensor([0.5, 0.5], requires_grad=True)

# 创建优化器
optimizer = Optimizer(objective, constraints, x0, max_iter=100, tolerance=1e-3)

# 执行优化
optimal_x = optimizer.stochastic_sqp()
print("Optimal solution:", optimal_x.detach().numpy())