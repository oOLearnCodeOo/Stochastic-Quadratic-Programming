import torch
from torch.nn import Parameter
from .parameters import Parameters
import os
import sys
# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# 将项目根目录添加到 sys.path
sys.path.insert(0, project_root)

class Optimizer(Parameters):
    def __init__(self, objective, constraints, x0, **kwargs):
        """
        初始化优化器。

        Parameters:
        - objective: `Objective` 类的实例
        - constraints: `Constraint` 类的实例
        - x0: 初始点 (torch.Tensor)
        - kwargs: 优化器参数
        """
        super().__init__(**kwargs)  # 调用父类 Parameters 的初始化方法
        self.objective = objective
        self.constraints = constraints  # 必须是 Constraint 实例
        self.x = x0.clone().detach().requires_grad_(True)

    def evaluate_constraints(self, x):
        """
        计算所有约束值，直接使用 Constraint 类的 evaluate_all。
        """
        return self.constraints.evaluate_all(x)

    def compute_jacobian(self, x):
        """
        计算约束函数的 Jacobian 矩阵，直接使用 Constraint 类的 compute_jacobian。
        """
        return self.constraints.compute_jacobian(x)

    def stochastic_sqp(self):
        """
        执行 Stochastic SQP 优化。
        """
        x = self.x
        for k in range(self.max_iter):
            # 计算目标函数梯度和 Hessian
            grad_f, H_k = self.objective.gradient_and_hessian(x)
            c_k = self.evaluate_constraints(x)  # 计算所有约束值
            J_k = self.compute_jacobian(x)  # 计算 Jacobian 矩阵

            # 更新 Lipschitz 常数和 Gamma（根据采样）
            L = self.update_L(x, grad_f, self.objective.gradient_and_hessian)
            Gamma = self.update_Gamma(x, self.constraints)

            # 构造 KKT 系统并求解
            KKT_matrix = torch.zeros(len(x) + len(c_k), len(x) + len(c_k))
            KKT_matrix[:len(x), :len(x)] = H_k
            KKT_matrix[:len(x), len(x):] = J_k.T
            KKT_matrix[len(x):, :len(x)] = J_k
            rhs = -torch.cat([grad_f, c_k])
            # d_k_y_k = torch.linalg.solve(KKT_matrix, rhs)
            d_k_y_k = torch.linalg.pinv(KKT_matrix) @ rhs
            # epsilon = 1e-6
            # KKT_matrix_reg = KKT_matrix + epsilon * torch.eye(KKT_matrix.size(0))
            # d_k_y_k = torch.linalg.solve(KKT_matrix_reg, rhs)
            d_k = d_k_y_k[:len(x)]

            # 更新参数
            tau_k_prev, xi_k_prev = self.tau_k, self.xi_k
            Delta_l = -1.0  # 示例值，实际需要通过模型缩减计算
            self.tau_k, self.xi_k, self.beta_k = self.update_parameters(
                grad_f, d_k, H_k, c_k, tau_k_prev, xi_k_prev, L, Gamma, Delta_l
            )

            # 更新步长
            alpha_min, alpha_max = self.update_step_size(
                self.beta_k, self.xi_k, self.tau_k, L, Gamma, self.eta, self.theta, Delta_l, c_k, d_k
            )
            alpha_k = max(alpha_min, alpha_max)

            # 更新解
            x = x + alpha_k * d_k
            x = x.clone().detach().requires_grad_(True)

            # 检查收敛
            # print(f"Current Iteration: {k+1}")
            # if torch.norm(grad_f) < self.tolerance and torch.norm(c_k) < self.tolerance:
            #     print(f"Converged at iteration {k+1}")
            #     break
            if torch.norm(d_k) < self.tolerance:
                print(f"Converged at iteration {k+1}")
                break

        # 最后检查 LICQ 条件
        if self.constraints.check_licq(x):
            print("Final solution satisfies LICQ condition.")
        else:
            print("Warning: Final solution does NOT satisfy LICQ condition.")


        return x