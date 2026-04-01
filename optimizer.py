import torch


class EROptimizer:
    def __init__(self, model, h=1.0, init_damping=0.05, step_clip=0.5, use_ema=True, ema_beta=0.9):
        self.model = model
        self.h = h
        self.step_clip = step_clip
        self.use_ema = use_ema
        self.ema_beta = ema_beta

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

        for name, p in self.model.named_parameters():
            if p.grad is None: continue

            grad_1d = p.grad.view(-1)
            n = grad_1d.size(0)

            if n > 400:
                updates[name] = -0.01 * p.grad.detach()
                condition_numbers[name] = 1.0
                expected_decrease += (0.01 * torch.sum(p.grad ** 2)).item()
                continue

            H = []
            for i in range(n):
                g2 = torch.autograd.grad(grad_1d[i], p, retain_graph=True)[0]
                H.append(g2.view(-1))


            H = torch.stack(H).detach()
            H = 0.5 * (H + H.T)

            if self.use_ema:
                if name not in self.state['hessian_ema']:
                    self.state['hessian_ema'][name] = H
                else:
                    self.state['hessian_ema'][name] = (
                            self.ema_beta * self.state['hessian_ema'][name] +
                            (1 - self.ema_beta) * H
                    )
                H_eff = self.state['hessian_ema'][name]
            else:
                H_eff = H

            L, V = torch.linalg.eigh(H_eff)

            lam_max = torch.max(torch.abs(L)).item()
            lam_min = torch.min(torch.abs(L)).item()
            condition_numbers[name] = lam_max / (lam_min + 1e-8)

            diag = torch.zeros_like(L)
            current_damping = self.state['damping']

            for i, lam in enumerate(L):
                abs_lam = torch.abs(lam) + current_damping
                arg = -self.h * abs_lam
                diag[i] = (1.0 - torch.exp(arg)) / abs_lam

            H_inv_er = V @ torch.diag(diag) @ V.T

            g_val = grad_1d.detach()
            step = H_inv_er @ g_val

            step_norm = torch.norm(step)
            if step_norm > self.step_clip:
                step = step * (self.step_clip / step_norm)

            dw = -step
            updates[name] = dw.view(p.size())

            dec = -(torch.dot(g_val, dw) + 0.5 * torch.dot(dw, H_eff @ dw))
            expected_decrease += dec.item()

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