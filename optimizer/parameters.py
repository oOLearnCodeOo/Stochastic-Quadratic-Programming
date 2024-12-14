import torch

class Parameters:
    def __init__(self, **kwargs):
        """
        管理优化器的参数。

        Parameters:
        - kwargs: 参数字典，可以覆盖默认参数
        """
        # 默认参数
        self.tau_k = 1.0  # Merit 参数
        self.xi_k = 0.8  # 收缩参数
        self.beta_k = 0.9  # 步长因子
        self.sigma = 0.1  # 缩减系数
        self.epsilon_tau = 0.05  # Merit 参数收敛阈值
        self.eta = 0.3  # 步长下界参数
        self.theta = 5.0  # 步长调整因子
        self.max_iter = 50  # 最大迭代次数
        self.tolerance = 1e-6  # 收敛阈值
        self.epsilon = 1e-5  # Lipschitz 常数扰动
        self.num_samples = 50  # Lipschitz 常数采样数
        self.zeta = 0.1  # 模型缩减阈值

        # 更新参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def update_tau(self, g_k, d_k, H_k, c_k, tau_k_prev):
        """
        更新 tau 参数。
        """
        directional_derivative = g_k @ d_k + 0.5 * (d_k @ H_k @ d_k)
        if directional_derivative <= 0:
            tau_trial = float('inf')
        else:
            tau_trial = (1 - self.sigma) * torch.norm(c_k, p=1).item() / directional_derivative.item()
        tau_k = min(tau_trial, (1 - self.epsilon_tau) * tau_k_prev) if tau_k_prev > tau_trial else tau_k_prev
        return tau_k, tau_trial

    def update_xi_k(self, xi_k_prev, tau_k, Delta_l, d_k):
        """
        更新 xi_k 参数。
        """
        norm_d_k = torch.norm(d_k).item()
        if norm_d_k > 0:
            xi_k = min(1.0, xi_k_prev + tau_k * norm_d_k / Delta_l)
        else:
            xi_k = xi_k_prev
        return xi_k

    def update_beta_k(self, eta, xi_k, tau_k, L, Gamma):
        """
        更新 beta_k 参数。
        """
        factor = 2 * (1 - eta) * xi_k * tau_k / (tau_k * L + Gamma)
        beta_k = min(1.0, max(factor, self.beta_k))  # 保证 beta_k 在 [factor, 1] 内
        return beta_k

    def update_step_size(self, beta_k, xi_k, tau_k, L, Gamma, eta, theta, Delta_l, c_k, d_k):
        """
        更新步长范围 [alpha_min, alpha_max]。
        """
        alpha_min = (2 * (1 - eta) * beta_k * xi_k * tau_k) / (tau_k * L + Gamma)
        alpha_max = min(alpha_min + theta * beta_k, 1.0)  # 限制在 [0, 1] 内
        return alpha_min, alpha_max

    def update_parameters(self, g_k, d_k, H_k, c_k, tau_k_prev, xi_k_prev, L, Gamma, Delta_l):
        """
        统一更新所有关键参数。
        """
        # 更新 tau_k
        tau_k, _ = self.update_tau(g_k, d_k, H_k, c_k, tau_k_prev)

        # 更新 xi_k
        xi_k = self.update_xi_k(xi_k_prev, tau_k, Delta_l, d_k)

        # 更新 beta_k
        beta_k = self.update_beta_k(self.eta, xi_k, tau_k, L, Gamma)

        return tau_k, xi_k, beta_k
    
    def update_L(self, x, grad_f, compute_gradient_and_hessian):
        """
        更新 Lipschitz 常数 L.

        Parameters:
        - x: 当前点 (torch.Tensor)
        - grad_f: 当前梯度向量 (torch.Tensor)
        - compute_gradient_and_hessian: 计算梯度和 Hessian 的函数

        Returns:
        - L: 目标函数的 Lipschitz 常数
        """
        L_values = []
        for _ in range(self.num_samples):
            # 添加随机扰动
            d = torch.randn_like(x) * self.epsilon
            x_perturbed = x + d

            # 计算扰动点的梯度
            grad_f_perturbed, _ = compute_gradient_and_hessian(x_perturbed)

            # 计算当前扰动下的 Lipschitz 常数
            L = torch.norm(grad_f_perturbed - grad_f) / torch.norm(d)
            L_values.append(L)

        return max(L_values)

    def update_Gamma(self, x, constraints, epsilon=1e-6, num_samples=10):
        """
        更新约束函数的 Lipschitz 常数 Gamma.

        Parameters:
        - x: 当前点 (torch.Tensor)
        - constraints: 约束函数的 Constraint 实例
        - epsilon: 微小扰动 (float)
        - num_samples: 采样次数 (int)

        Returns:
        - gamma_sum: 所有约束的 Lipschitz 常数之和 (float)
        """
        gamma_list = []
        for constraint in constraints:  # 现在 constraints 可直接迭代
            gamma_i_samples = []
            for _ in range(num_samples):
                d = torch.randn_like(x) * epsilon  # 随机扰动
                x_perturbed = x + d

                # 计算当前点和扰动点的 Jacobian
                grad_c = torch.autograd.grad(constraint(x), x, retain_graph=True)[0]
                grad_c_perturbed = torch.autograd.grad(constraint(x_perturbed), x_perturbed, retain_graph=True)[0]

                # 计算当前扰动下的 Lipschitz 常数
                gamma_i = torch.norm(grad_c_perturbed - grad_c) / torch.norm(d)
                gamma_i_samples.append(gamma_i)

            gamma_list.append(max(gamma_i_samples))

        return sum(gamma_list)  # 返回 Gamma 的总和
    
