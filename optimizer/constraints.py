import torch

class Constraint:
    def __init__(self, func=None):
        """
        初始化约束函数或约束函数列表。
        """
        if func is None:
            self.constraints = []  # 支持动态添加约束
        elif isinstance(func, list):
            self.constraints = func  # 多约束函数
        else:
            self.constraints = [func]  # 单约束函数封装为列表

    def __iter__(self):
        """
        使 Constraint 类支持迭代。
        """
        return iter(self.constraints)

    def evaluate(self, x, index=0):
        """
        计算单个约束值。
        """
        return self.constraints[index](x)

    def evaluate_all(self, x):
        """
        计算所有约束值。
        """
        return torch.stack([constraint(x) for constraint in self.constraints])

    def gradient(self, x, index=0):
        """
        计算单个约束的梯度。
        """
        c = self.constraints[index](x)
        grad_c = torch.autograd.grad(c, x, retain_graph=True)[0]
        return grad_c

    def compute_jacobian(self, x):
        """
        计算所有约束的 Jacobian 矩阵。
        """
        jacobian_rows = []
        for constraint in self.constraints:
            c = constraint(x)
            grad_c = torch.autograd.grad(c, x, retain_graph=True)[0]
            jacobian_rows.append(grad_c)
        return torch.stack(jacobian_rows)

    def add_constraint(self, new_constraint_func):
        """
        动态添加新的约束函数。
        """
        self.constraints.append(new_constraint_func)

    def check_licq(self, x):
        """
        判别是否满足线性独立约束资格条件（LICQ）。

        Parameters:
        - x: 当前点 (torch.Tensor)

        Returns:
        - is_licq: 是否满足 LICQ (bool)
        """
        # 计算约束的 Jacobian 矩阵
        J_c = self.compute_jacobian(x)

        # 获取 Jacobian 矩阵的秩
        rank = torch.linalg.matrix_rank(J_c)

        # 如果秩等于约束数量，说明满足 LICQ
        return rank == len(self.constraints)
    
    def check_licq(self, x, tolerance=None):
        """
        判别是否满足线性独立约束资格条件（LICQ）。

        Parameters:
        - x: 当前点 (torch.Tensor)
        - tolerance: 数值误差容差 (float, 默认根据奇异值动态计算)

        Returns:
        - is_licq: 是否满足 LICQ (bool)
        """
        # 计算约束的 Jacobian 矩阵
        J_c = self.compute_jacobian(x)
        # print("Jacobian matrix is: ", J_c)

        # 计算 Jacobian 矩阵的奇异值
        singular_values = torch.linalg.svdvals(J_c)

        # 动态设置容差
        if tolerance is None:
            tolerance = max(J_c.shape) * torch.finfo(J_c.dtype).eps * singular_values.max()

        # 判断奇异值大于容差的数量是否等于约束数量
        rank = torch.sum(singular_values > tolerance).item()
        return rank == len(self.constraints)
    
#### 测试LICQ
# 定义两个约束函数
# def constraint_func_1(x):
#     Q = torch.tensor([[1.0, 0.5], [0.5, 1.0]], requires_grad=False)
#     b = torch.tensor([1.0, -1.0], requires_grad=False)
#     c = -0.5
#     return 0.5 * x.T @ Q @ x + b.T @ x + c

# def constraint_func_2(x):
#     return torch.sum(x) - 1.0

# # 创建约束
# constraints = Constraint([constraint_func_1, constraint_func_2])

# # 测试点
# x0 = torch.tensor([0.5, 0.5], requires_grad=True)

# # 检查是否满足 LICQ
# is_licq = constraints.check_licq(x0)
# print(f"Does LICQ hold at x0? {is_licq}")