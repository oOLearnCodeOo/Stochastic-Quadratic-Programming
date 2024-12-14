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
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def constraint_func_1(x):
    return x[0] + x[1] - 1.0  # x + y = 1


# 初始化
objective = Objective(objective_func)
constraints = Constraint([constraint_func_1])
x0 = torch.tensor([0.0, 0.0], requires_grad=True)

# 创建优化器
optimizer = Optimizer(objective, constraints, x0, max_iter=100, tolerance=1e-6)

# 执行优化
optimal_x = optimizer.stochastic_sqp()
print("Optimal solution:", optimal_x.detach().numpy())