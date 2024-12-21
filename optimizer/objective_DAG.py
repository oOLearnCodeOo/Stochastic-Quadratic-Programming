import torch

class Objective:
    def __init__(self, func, data=None):
        """
        初始化目标函数。

        Parameters:
        - func: 目标函数，可接受一个变量或多个变量
        - data: 额外数据参数 (如数据矩阵 X)
        """
        self.func = func
        self.data = data

    def evaluate(self, x):
        """
        计算目标函数值。
        """
        if self.data is not None:
            return self.func(x, self.data)  # 将数据传递给目标函数
        return self.func(x)

    def gradient(self, x):
        """
        计算目标函数的梯度。
        """
        f = self.evaluate(x)
        grad_f = torch.autograd.grad(f, x, create_graph=True)[0]
        return grad_f

    def hessian(self, x):
        """
        计算目标函数的 Hessian 矩阵。
        """
        grad_f = self.gradient(x)
        hessian_rows = []
        for g in grad_f:
            hessian_row = torch.autograd.grad(g, x, retain_graph=True)[0]
            hessian_rows.append(hessian_row)
        H_k = torch.stack(hessian_rows)
        return H_k

    def gradient_and_hessian(self, x):
        """
        计算目标函数的梯度和 Hessian 矩阵。
        """
        grad_f = self.gradient(x)
        H_k = self.hessian(x)
        return grad_f, H_k