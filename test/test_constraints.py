import sys
import os
import torch

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# 将项目根目录添加到 sys.path
sys.path.insert(0, project_root)
from optimizer.constraints import Constraint

#### 测试LICQ
# 定义两个约束函数
def constraint_func_1(x):
    Q = torch.tensor([[1.0, 0.5], [0.5, 1.0]], requires_grad=False)
    b = torch.tensor([1.0, -1.0], requires_grad=False)
    c = -0.5
    return 0.5 * x.T @ Q @ x + b.T @ x + c

def constraint_func_2(x):
    return torch.sum(x) - 1.0

# 创建约束
constraints = Constraint([constraint_func_1, constraint_func_2])

# 测试点
x0 = torch.tensor([0.5, 0.5], requires_grad=True)

# 检查是否满足 LICQ
is_licq = constraints.check_licq(x0)
print(f"Does LICQ hold at x0? {is_licq}")