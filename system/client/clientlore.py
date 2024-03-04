import torch
import numpy as np
from system.client.clientbase import Client


class clientLoRE(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs) \

        self.rank = args.rank
        self.old_model_dict = {}
        self.lowrank_gradient_dict = {}
        self.param = {}

    def train(self):
        train_loader = self.load_train_data()
        self.model.train()

        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(train_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        # low-rank estimation
        self.low_rank_estimation()


    def set_parameters(self, model):
        self.model.load_state_dict(model)
        for key, value in model.items():
            self.old_model_dict[key] = value

    def set_parameters_lora(self, model):
        for key, value in self.model.state_dict().items():
            self.old_model_dict[key] = model[key] + (value - self.lowrank_gradient_dict[key])

        self.model.load_state_dict(self.old_model_dict)

    #########################################################################################################
    ########################################## low_rank_estimation ##########################################
    #########################################################################################################

    def low_rank_estimation(self):
        for (k1, v1), (k2, v2) in zip(self.model.state_dict().items(), self.old_model_dict.items()):
            if k1 != k2:
                raise ValueError("Model state_dict keys are not the same.")
            if v1.shape != v2.shape:
                raise ValueError(f"Tensor shapes for {k1} are not the same.")
            self.low_rank_estimation_layter(k1, v1 - v2)

    # Online calculate lowrank and sparse for each layer
    def low_rank_estimation_layter(self, key, param):
        """
        Online Robust Low-rank Estimation via Stochastic Optimization

        The loss function is
            min_{L,S} { 1/2||M-L-S||_F^2 + lambda1||L||_* + lambda2*||S(:)||_1}
        """
        param_shape = param.shape
        d = param.reshape(-1)

        m, n = len(d), self.rank
        lambda1 = (1.0 / torch.sqrt(torch.tensor(max(m, n), dtype=torch.float32))).to(self.device)
        lambda2 = (1.0 / torch.sqrt(torch.tensor(max(m, n), dtype=torch.float32))).to(self.device)

        # Initialization
        if key not in self.param:
            self.param[key] = {}
            self.param[key]["U"] = torch.rand(m, self.rank).to(self.device)
            self.param[key]["A"] = torch.zeros((self.rank, self.rank)).to(self.device)
            self.param[key]["B"] = torch.zeros((m, self.rank)).to(self.device)

        U = self.param[key]["U"]
        A = self.param[key]["A"]
        B = self.param[key]["B"]

        v, s = self.solve_proj2(d, U, lambda1, lambda2)
        A = A + torch.outer(v, v)
        B = B + torch.outer(torch.sub(d, s), v)
        U = self.update_col(U, A, B, lambda1)

        self.lowrank_gradient_dict[key] = torch.matmul(U, v).reshape(-1).reshape(param_shape)

        self.param[key]["U"] = U
        self.param[key]["A"] = A
        self.param[key]["B"] = B

    # Update U
    def update_col(self, U, A, B, lambda1):
        m, r = U.shape
        A = A + lambda1 * torch.eye(r).to(self.device)
        for j in range(r):
            bj = B[:, j]
            uj = U[:, j]
            aj = A[:, j]
            temp = (bj - torch.matmul(U, aj)) / A[j, j] + uj
            U[:, j] = temp / torch.maximum(torch.linalg.norm(temp), torch.tensor(1))

        return U

    # Calculate v and s by U
    def solve_proj2(self, m, U, lambda1, lambda2):
        """
        solve the problem:
        min_{v, s} 0.5*|m-Uv-s|_2^2 + 0.5*lambda1*|v|^2 + lambda2*|s|_1
        solve the projection by APG
        """
        # Initialization
        n, p = U.shape
        v = torch.zeros(p).to(self.device)
        s = torch.zeros(n).to(self.device)
        I = torch.eye(p).to(self.device)
        converged = False
        maxIter = np.inf
        k = 0
        # Alternatively update
        UUt = torch.matmul(torch.inverse(torch.matmul(U.t(), U) + lambda1 * I), U.t())
        while not converged:
            k += 1
            v_temp = v
            # v = (U'*U + lambda1*I)\(U'*(m-s))
            v = torch.matmul(UUt, m - s)
            s_temp = s
            s = self.soft_threshold(m - torch.matmul(U, v), lambda2)
            stop_error = torch.maximum(torch.linalg.norm(v - v_temp), torch.linalg.norm(s - s_temp)).cpu().numpy() / n
            if stop_error < 10 ** (-7) or k > maxIter:
                converged = True

        return v, s

    # Calculate soft threshold value
    def soft_threshold(self, x, mu):
        """
        y = sgn(x)max(|x| - mu, 0)
        """
        y = torch.maximum(x - mu, torch.tensor(0))
        y = y + torch.minimum(x + mu, torch.tensor(0))
        return y