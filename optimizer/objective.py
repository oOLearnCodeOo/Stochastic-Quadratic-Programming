import torch

class Objective:
    def __init__(self, func):
        """
        初始化目标函数。

        Parameters:
        - func: 目标函数，接受 x 返回标量
        """
        self.func = func

    def evaluate(self, x):
        """
        计算目标函数值。

        Returns:
        - f: 目标函数值 (scalar)
        """
        return self.func(x)

    def gradient_and_hessian(self, x):
        """
        计算目标函数的梯度和 Hessian 矩阵。

        Returns:
        - grad_f: 梯度向量 (torch.Tensor)
        - H_k: Hessian 矩阵 (torch.Tensor)
        """
        f = self.func(x)
        grad_f = torch.autograd.grad(f, x, create_graph=True)[0]
        hessian_rows = []
        for g in grad_f:
            hessian_row = torch.autograd.grad(g, x, retain_graph=True)[0]
            hessian_rows.append(hessian_row)
        H_k = torch.stack(hessian_rows)
        return grad_f, H_k