import torch


class EROptimizer:
    def __init__(self, model, er_method='Spectral', h=1.0, init_damping=0.05,
                 step_clip=1.0, use_ema=True, ema_beta=0.9, k_lanczos=10,
                 use_compression=False, compression_threshold=1e-4):
        self.model = model
        self.er_method = er_method
        self.h = h
        self.step_clip = step_clip
        self.use_ema = use_ema
        self.ema_beta = ema_beta
        self.k_lanczos = k_lanczos

        self.use_compression = use_compression
        self.compression_threshold = compression_threshold

        self.state = {
            'hessian_ema': {},
            'damping': init_damping
        }

    def step(self, loss_fn, X, y):
        loss = loss_fn(self.model(X), y)
        self.model.zero_grad()
        loss.backward(create_graph=True)

        updates = {}
        condition_numbers = {}
        expected_decrease = 0.0
        current_damping = self.state['damping']

        for name, p in self.model.named_parameters():
            if p.grad is None: continue

            grad_1d = p.grad.view(-1)
            n = grad_1d.size(0)
            g_val = grad_1d.detach()

            if self.er_method == 'Lanczos':
                def calc_hv(v):
                    gv = torch.dot(grad_1d, v)
                    hv = torch.autograd.grad(gv, p, retain_graph=True)[0].view(-1)
                    return hv.detach()

                k = min(self.k_lanczos, n)
                Q = torch.zeros((n, k), device=p.device)
                alphas = torch.zeros(k, device=p.device)
                betas = torch.zeros(k, device=p.device)

                g_norm = torch.norm(g_val)
                if g_norm < 1e-8:
                    updates[name] = torch.zeros_like(p)
                    continue

                v = g_val / g_norm
                Q[:, 0] = v
                v_prev = torch.zeros_like(v)
                beta = 0.0

                for j in range(k):
                    w = calc_hv(Q[:, j])
                    w = w - beta * v_prev
                    alpha = torch.dot(w, Q[:, j])
                    alphas[j] = alpha
                    w = w - alpha * Q[:, j]

                    for i in range(j + 1):
                        w = w - torch.dot(w, Q[:, i]) * Q[:, i]

                    beta = torch.norm(w)
                    if j < k - 1:
                        betas[j] = beta
                        if beta > 1e-8:
                            Q[:, j + 1] = w / beta
                            v_prev = Q[:, j]
                        else:
                            k = j + 1
                            break

                T = torch.diag(alphas[:k]) + torch.diag(betas[:k - 1], 1) + torch.diag(betas[:k - 1], -1)
                L, V = torch.linalg.eigh(T)
                lam_max = torch.max(torch.abs(L)).item()
                lam_min = torch.min(torch.abs(L)).item()
                condition_numbers[name] = lam_max / (lam_min + 1e-8)

                diag = torch.zeros_like(L)
                for i, lam in enumerate(L):
                    abs_lam = torch.abs(lam) + current_damping
                    arg = -self.h * abs_lam
                    diag[i] = (1.0 - torch.exp(arg)) / abs_lam

                f_T = V @ torch.diag(diag) @ V.T
                e1 = torch.zeros(k, device=p.device)
                e1[0] = g_norm
                y_step = f_T @ e1
                step = Q[:, :k] @ y_step
                H_eff = None

            else:
                if n > 400:
                    updates[name] = -0.01 * g_val
                    condition_numbers[name] = 1.0
                    expected_decrease += (0.01 * torch.sum(g_val ** 2)).item()
                    continue

                H = []
                for i in range(n):
                    g2 = torch.autograd.grad(grad_1d[i], p, retain_graph=True)[0]
                    H.append(g2.view(-1))

                H = torch.stack(H).detach()
                H = 0.5 * (H + H.T)

                if self.use_ema:
                    if name not in self.state['hessian_ema']:
                        H_new = H
                    else:
                        H_old = self.state['hessian_ema'][name]
                        if H_old.layout == torch.sparse_csr:
                            H_old = H_old.to_dense()
                        H_new = self.ema_beta * H_old + (1 - self.ema_beta) * H

                    if self.use_compression:
                        mask = torch.abs(H_new) > self.compression_threshold
                        H_pruned = H_new * mask
                        self.state['hessian_ema'][name] = H_pruned.to_sparse_csr()
                    else:
                        self.state['hessian_ema'][name] = H_new

                    H_eff = self.state['hessian_ema'][name]
                    if H_eff.layout == torch.sparse_csr:
                        H_eff = H_eff.to_dense()
                else:
                    H_eff = H

                if self.er_method == 'Spectral':
                    L, V = torch.linalg.eigh(H_eff)
                    lam_max = torch.max(torch.abs(L)).item()
                    lam_min = torch.min(torch.abs(L)).item()
                    condition_numbers[name] = lam_max / (lam_min + 1e-8)

                    diag = torch.zeros_like(L)
                    for i, lam in enumerate(L):
                        abs_lam = torch.abs(lam) + current_damping
                        arg = -self.h * abs_lam
                        diag[i] = (1.0 - torch.exp(arg)) / abs_lam

                    H_inv_er = V @ torch.diag(diag) @ V.T
                    step = H_inv_er @ g_val

                elif self.er_method == 'Recursive':
                    G = H_eff.detach()
                    G = G + (current_damping + 1e-4) * torch.eye(n, device=p.device)
                    norm_G = torch.linalg.matrix_norm(G, ord='fro')
                    m = int(torch.ceil(torch.log2(norm_G * self.h / 0.5)).item())
                    m = max(0, m)

                    A = (G * self.h) / (2 ** m)
                    identity = torch.eye(n, device=p.device)
                    Phi = identity - 0.5 * A + (1 / 6.0) * torch.matmul(A, A)
                    ExpA = identity - torch.matmul(A, Phi)

                    for _ in range(m):
                        Phi = 0.5 * torch.matmul(Phi, (identity + ExpA))
                        ExpA = torch.matmul(ExpA, ExpA)

                    step = self.h * (Phi @ g_val)
                    condition_numbers[name] = norm_G.item() / (current_damping + 1e-8)

            step_norm = torch.norm(step)
            if step_norm > self.step_clip:
                step = step * (self.step_clip / step_norm)

            dw = -step
            updates[name] = dw.view(p.size())

            if self.er_method != 'Lanczos' and H_eff is not None:
                dec = -(torch.dot(g_val, dw) + 0.5 * torch.dot(dw, H_eff @ dw))
                expected_decrease += dec.item()
            else:
                expected_decrease += (torch.dot(g_val, -dw)).item()

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in updates:
                    p.add_(updates[name])

        with torch.no_grad():
            new_loss = loss_fn(self.model(X), y).item()

        actual_decrease = loss.item() - new_loss

        if expected_decrease > 1e-8:
            rho = actual_decrease / expected_decrease
        else:
            rho = 0.0

        if rho > 0.75:
            self.state['damping'] = max(1e-4, self.state['damping'] * 0.5)
        elif rho < 0.25:
            self.state['damping'] = min(10.0, self.state['damping'] * 2.0)

        return new_loss, condition_numbers