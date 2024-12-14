import sys
import os
import torch

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# 将项目根目录添加到 sys.path
sys.path.insert(0, project_root)
import  optimizer
from optimizer.objective import Objective

def test_objective_value():
    """
    测试目标函数值计算。
    """
    # 定义目标函数
    def objective_func(x):
        A = torch.tensor([[3.0, 1.0], [1.0, 2.0]], requires_grad=False)
        b_obj = torch.tensor([1.0, 1.0], requires_grad=False)
        residual = A @ x - b_obj
        return 0.5 * torch.sum(residual**2)

    # 初始化 Objective
    objective = Objective(objective_func)

    # 测试点
    x = torch.tensor([0.0, 0.0], requires_grad=True)

    # 计算目标函数值
    f_val = objective.evaluate(x)
    expected_f_val = 0.5 * torch.sum(torch.tensor([1.0, 1.0])**2)  # 手工计算的目标函数值
    assert torch.isclose(f_val, expected_f_val), f"Expected {expected_f_val}, got {f_val}"

def test_gradient():
    """
    测试目标函数梯度计算。
    """
    # 定义目标函数
    def objective_func(x):
        A = torch.tensor([[3.0, 1.0], [1.0, 2.0]], requires_grad=False)
        b_obj = torch.tensor([1.0, 1.0], requires_grad=False)
        residual = A @ x - b_obj
        return 0.5 * torch.sum(residual**2)

    # 初始化 Objective
    objective = Objective(objective_func)

    # 测试点
    x = torch.tensor([0.0, 0.0], requires_grad=True)

    # 计算梯度
    grad, _ = objective.gradient_and_hessian(x)
    expected_grad = torch.tensor([-3.0, -3.0])  # 手工计算的梯度
    assert torch.allclose(grad, expected_grad, atol=1e-6), f"Expected {expected_grad}, got {grad}"

def test_hessian():
    """
    测试目标函数 Hessian 矩阵计算。
    """
    # 定义目标函数
    def objective_func(x):
        A = torch.tensor([[3.0, 1.0], [1.0, 2.0]], requires_grad=False)
        b_obj = torch.tensor([1.0, 1.0], requires_grad=False)
        residual = A @ x - b_obj
        return 0.5 * torch.sum(residual**2)

    # 初始化 Objective
    objective = Objective(objective_func)

    # 测试点
    x = torch.tensor([0.0, 0.0], requires_grad=True)

    # 计算 Hessian
    _, hessian = objective.gradient_and_hessian(x)
    expected_hessian = torch.tensor([[10.0, 4.0], [4.0, 8.0]])  # 手工计算的 Hessian 矩阵
    assert torch.allclose(hessian, expected_hessian, atol=1e-6), f"Expected {expected_hessian}, got {hessian}"