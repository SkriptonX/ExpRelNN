import torch


class EROptimizer:
    def __init__(self, model, er_method='Spectral', h=1.0, init_damping=0.05,
                 step_clip=1.0, use_ema=True, ema_beta=0.9, k_lanczos=10,
                 use_compression=False, compression_threshold=1e-4, chebyshev_k=15):
        self.model = model
        self.er_method = er_method
        self.h = h
        self.step_clip = step_clip
        self.use_ema = use_ema
        self.ema_beta = ema_beta
        self.k_lanczos = k_lanczos
        self.chebyshev_k = chebyshev_k
        self.use_compression = use_compression
        self.compression_threshold = compression_threshold

        self.state = {
            'damping': {},
            'momentum_buffer': {},
            'step': 0
        }
        self.init_damping = init_damping

    def step(self, loss_fn, X, y):
        if not X.requires_grad:
            X.requires_grad_(True)

        loss = loss_fn(self.model(X), y)
        self.model.zero_grad()

        grads_tuple = torch.autograd.grad(loss, self.model.parameters(), create_graph=True, allow_unused=True)
        global_grads = {}
        for (name, p), g in zip(self.model.named_parameters(), grads_tuple):
            if g is None:
                global_grads[name] = torch.zeros_like(p)
            else:
                global_grads[name] = g

        kaczmarz_grads = {}
        if self.er_method == 'Kaczmarz':
            batch_size_full = X.size(0)
            subset_size = min(16, batch_size_full)
            indices = torch.randperm(batch_size_full, device=X.device)[:subset_size]
            x_sub = X[indices]
            y_sub = y[indices]

            if hasattr(loss_fn, 'X_batch'):
                original_X = loss_fn.X_batch
                loss_fn.X_batch = x_sub

            loss_sub = loss_fn(self.model(x_sub), y_sub)
            k_tuple = torch.autograd.grad(loss_sub, self.model.parameters(), create_graph=True, allow_unused=True)
            for (name, p), g in zip(self.model.named_parameters(), k_tuple):
                kaczmarz_grads[name] = g if g is not None else torch.zeros_like(p)

            if hasattr(loss_fn, 'X_batch'):
                loss_fn.X_batch = original_X

        updates = {}
        condition_numbers = {}
        expected_decrease = 0.0
        current_damping = self.state['damping'].get(name, self.init_damping)

        if 'l_max_cache' not in self.state:
            self.state['l_max_cache'] = {}
            self.state['step_count'] = 0

        self.state['step_count'] += 1

        for name, p in self.model.named_parameters():
            grad_1d = global_grads[name].view(-1)
            n = grad_1d.size(0)
            g_val = grad_1d.detach()

            if not grad_1d.requires_grad:
                updates[name] = (-self.h * g_val).view(p.size())
                condition_numbers[name] = 1.0
                expected_decrease += (self.h * torch.sum(g_val ** 2)).item()
                continue

            def calc_hv(v):
                gv = torch.dot(grad_1d, v)
                hv = torch.autograd.grad(gv, p, retain_graph=True, allow_unused=True)[0]
                if hv is None: return torch.zeros_like(v)
                return hv.view(-1).detach()

            def calc_hv_damped(v):
                return calc_hv(v) + current_damping * v

            if self.er_method == 'Chebyshev':
                if name not in self.state['l_max_cache'] or self.state['step_count'] % 15 == 1:
                    v_iter = g_val / (torch.norm(g_val) + 1e-8)
                    for _ in range(4):
                        Hv = calc_hv_damped(v_iter)
                        l_max_est = torch.abs(torch.dot(v_iter, Hv)).item()
                        v_iter = Hv / (torch.norm(Hv) + 1e-8)

                    self.state['l_max_cache'][name] = l_max_est * 1.2 + 1e-4

                l_max = self.state['l_max_cache'][name]
                condition_numbers[name] = l_max / current_damping

                K = min(self.chebyshev_k, n)
                nodes = torch.cos(torch.pi * (torch.arange(K, dtype=p.dtype, device=p.device) + 0.5) / K)

                x_nodes = l_max * nodes
                lam_orig = x_nodes - current_damping
                abs_lam = torch.abs(lam_orig) + current_damping

                f_nodes = (1.0 - torch.exp(-self.h * abs_lam)) / abs_lam

                c = torch.zeros(K, dtype=p.dtype, device=p.device)
                for j in range(K):
                    cos_terms = torch.cos(j * torch.pi * (torch.arange(K, device=p.device) + 0.5) / K)
                    c[j] = (2.0 / K) * torch.sum(f_nodes * cos_terms)
                c[0] /= 2.0

                v_0 = g_val
                step = c[0] * v_0

                if K > 1:
                    v_1 = (1.0 / l_max) * calc_hv_damped(v_0)
                    step += c[1] * v_1

                    v_k_minus_1 = v_0
                    v_k = v_1
                    for j in range(2, K):
                        v_next = (2.0 / l_max) * calc_hv_damped(v_k) - v_k_minus_1
                        step += c[j] * v_next
                        v_k_minus_1 = v_k
                        v_k = v_next
                force_factor = 2.0
                step = step * force_factor

                H_eff = None

            elif self.er_method == 'Kaczmarz':
                grad_sub = kaczmarz_grads[name].view(-1)

                if not grad_sub.requires_grad:
                    updates[name] = (-self.h * g_val).view(p.size())
                    continue

                H_sub = []
                for i in range(n):
                    g2 = torch.autograd.grad(grad_sub[i], p, retain_graph=True, allow_unused=True)[0]
                    if g2 is None: g2 = torch.zeros_like(p)
                    H_sub.append(g2.view(-1))
                H_sub = torch.stack(H_sub).detach()
                H_sub = 0.5 * (H_sub + H_sub.T)

                L, V = torch.linalg.eigh(H_sub)
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
                H_eff = H_sub

            elif self.er_method == 'Lanczos':
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
                if n > 5000:
                    updates[name] = (-0.01 * g_val).view(p.size())
                    condition_numbers[name] = 1.0
                    expected_decrease += (0.01 * torch.sum(g_val ** 2)).item()
                    continue

                H = []
                for i in range(n):
                    g2 = torch.autograd.grad(grad_1d[i], p, retain_graph=True, allow_unused=True)[0]
                    if g2 is None: g2 = torch.zeros_like(p)
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

                        if H_old.shape != H.shape:
                            H_new = H
                        else:
                            H_new = self.ema_beta * H_old + (1 - self.ema_beta) * H

                    if self.use_compression:
                        max_val = torch.max(torch.abs(H_new)).item()
                        adaptive_threshold = max(self.compression_threshold, max_val * 0.01)
                        mask = torch.abs(H_new) > adaptive_threshold
                        nnz = torch.sum(mask).item()
                        total_elements = H_new.numel()
                        sparsity = 1.0 - (nnz / total_elements)

                        if sparsity > 0.6:
                            H_pruned = H_new * mask
                            self.state['hessian_ema'][name] = H_pruned.to_sparse_csr()
                        else:
                            self.state['hessian_ema'][name] = H_new
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

            if name not in self.state['damping']:
                self.state['damping'][name] = self.init_damping
            step_norm = torch.norm(step)

            if step_norm > self.step_clip:
                step = step * (self.step_clip / step_norm)

            dw = -step
            updates[name] = dw.view(p.size())

            if self.er_method not in ['Lanczos', 'Chebyshev'] and H_eff is not None:
                dec = -(torch.dot(g_val, dw) + 0.5 * torch.dot(dw, H_eff @ dw))
                expected_decrease += dec.item()
            else:
                expected_decrease += (torch.dot(g_val, -dw)).item()

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in updates:
                    if name not in self.state['momentum_buffer']:
                        self.state['momentum_buffer'][name] = torch.zeros_like(p)

                    buf = self.state['momentum_buffer'][name]
                    buf.mul_(0.9).add_(updates[name])

                    p.add_(buf)

        with torch.enable_grad():
            if not X.requires_grad:
                X.requires_grad_(True)
            new_loss_tensor = loss_fn(self.model(X), y)
            new_loss = new_loss_tensor.item()

        actual_decrease = loss.item() - new_loss

        if actual_decrease < 0:
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in updates:
                        p.sub_(self.state['momentum_buffer'][name])
                        self.state['momentum_buffer'][name].zero_()
                        self.state['damping'][name] = min(100000.0, self.state['damping'][name] * 4.0)

            return loss.item(), condition_numbers

        if expected_decrease > 1e-8:
            rho = actual_decrease / expected_decrease
        else:
            rho = 0.0

        for name in updates.keys():
            if rho > 0.75:
                self.state['damping'][name] = max(1e-5, self.state['damping'][name] * 0.5)
            elif rho < 0.25:
                self.state['damping'][name] = min(100000.0, self.state['damping'][name] * 2.0)

        return new_loss, condition_numbers