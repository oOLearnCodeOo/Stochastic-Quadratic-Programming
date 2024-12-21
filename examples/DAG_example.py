import torch
import torch.nn as nn
from torch.nn import Parameter

import os
import sys
# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# 将项目根目录添加到 sys.path
sys.path.insert(0, project_root)
from optimizer.Optimizer_DAG import Optimizer
from optimizer.objective_DAG import Objective
from optimizer.constraints import Constraint

# class G_DAG_Vectorized:
#     def __init__(self, d, seed = 42, device='cpu'):
#         """
#         初始化矩阵 W，仅优化非对角线元素。

#         Parameters:
#         - d: 矩阵维度 (int)
#         - device: 设备 (str)
#         """
#         if seed is not None:
#             torch.manual_seed(seed)
            
#         self.d = d
#         self.device = device

#         # 创建完整的矩阵
#         W_full = torch.randn(d, d, device=device)
#         W_full.fill_diagonal_(0)  # 将对角线元素置为 0
#         diag_indices = torch.arange(0, d * d, step=d + 1)

#         # 创建掩码
#         self.mask = torch.ones(d * d, dtype=torch.bool, device=device)
#         self.mask[diag_indices] = False
#         self.W_vec = torch.nn.Parameter(W_full.view(-1)[self.mask])

#     def reconstruct_matrix(self):
#         """
#         从优化变量重构完整矩阵 W，确保对角线为 0。
#         """
#         W_full = torch.zeros(self.d * self.d, device=self.device)
#         W_full[self.mask] = self.W_vec
#         return W_full.view(self.d, self.d)

def objective_with_mask(W_vec, X, mask, d):
    """
    目标函数 ||X - XW||_F^2，动态生成完整矩阵并排除对角线元素。

    Parameters:
    - W_vec: 优化变量，仅包含非对角线元素 (torch.Tensor)
    - X: 数据矩阵 (torch.Tensor)
    - mask: 非对角线掩码 (torch.Tensor)
    - d: 矩阵维度 (int)

    Returns:
    - loss: 目标函数值
    """
    # 动态生成完整矩阵
    W_full = torch.zeros(d * d, device=W_vec.device)
    W_full[mask] = W_vec
    W = W_full.view(d, d)

    # 计算残差和损失
    residual = X - X @ W
    return torch.norm(residual, p='fro')**2

def dag_constraint(W_vec, d, mask, max_power=3):
    """
    DAG 约束：Tr(W) + Tr(W^2) + ... + Tr(W^max_power) = 0。
    """
    # 动态生成完整矩阵
    W_full = torch.zeros(d * d, device=W_vec.device)
    W_full[mask] = W_vec
    W = W_full.view(d, d)
    W_abs = torch.abs(W)  # 取绝对值

    # 计算约束
    constraint = 0.0
    W_power = W_abs.clone()
    for k in range(1, max_power + 1):
        constraint += k * torch.trace(W_power)
        W_power = torch.matmul(W_power, W)
    return constraint



def main():
    # 数据初始化
    X = torch.randn(100, 10, device='cpu')

    # 优化变量
    d = 10
    device = 'cpu'
    W_full = torch.randn(d, d, device=device)
    W_full.fill_diagonal_(0)
    diag_indices = torch.arange(0, d * d, step=d + 1)
    mask = torch.ones(d * d, dtype=torch.bool, device=device)
    mask[diag_indices] = False
    W_vec = torch.nn.Parameter(W_full.view(-1)[mask])

    # 创建目标和约束
    objective = Objective(lambda W_vec: objective_with_mask(W_vec, X, mask, d))
    constraints = Constraint([lambda W_vec: dag_constraint(W_vec, d, mask, max_power=10)])

    # 创建优化器
    optimizer = Optimizer(objective, constraints, W_vec, mask, d=d, max_iter=100, tolerance=1e-6)

    # 执行优化
    optimal_W_vec = optimizer.stochastic_sqp()

    # 重构完整矩阵
    W_final = torch.zeros(d * d, device=W_vec.device)
    W_final[mask] = optimal_W_vec
    W = W_final.view(d, d)

    print("Final optimized W matrix:\n", W.detach().numpy())


if __name__ == "__main__":
    main()