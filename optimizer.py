import torch


class EROptimizer:
    def __init__(self, model, h=1.0, damping=0.05, step_clip=0.5):
        """
        h: Параметр релаксации (аналог learning rate для плоских участков).
        damping: Параметр регуляризации для стабилизации деления.
        step_clip: Максимальная норма итогового шага (Trust Region).
        """
        self.model = model
        self.h = h
        self.damping = damping
        self.step_clip = step_clip

    def step(self, loss_fn, X, y):
        loss = loss_fn(self.model(X), y)
        self.model.zero_grad()

        loss.backward(create_graph=True)

        updates = {}
        condition_numbers = {}

        for name, p in self.model.named_parameters():
            if p.grad is None: continue

            grad_1d = p.grad.view(-1)
            n = grad_1d.size(0)

            if n > 400:
                updates[name] = -0.01 * p.grad
                condition_numbers[name] = 1.0
                continue

            H = []
            for i in range(n):
                g2 = torch.autograd.grad(grad_1d[i], p, retain_graph=True)[0]
                H.append(g2.view(-1))
            H = torch.stack(H)

            H = 0.5 * (H + H.T)

            L, V = torch.linalg.eigh(H)

            lam_max = torch.max(torch.abs(L)).item()
            lam_min = torch.min(torch.abs(L)).item()
            condition_numbers[name] = lam_max / (lam_min + 1e-8)

            diag = torch.zeros_like(L)
            for i, lam in enumerate(L):
                # 1. Демпфирование (Levenberg-Marquardt)
                # Добавление damping защищает от взрыва шага при малых лямбда
                abs_lam = torch.abs(lam) + self.damping

                arg = -self.h * abs_lam
                diag[i] = (1.0 - torch.exp(arg)) / abs_lam

            H_inv_er = V @ torch.diag(diag) @ V.T

            # Итоговый шаг оптимизатора
            step = H_inv_er @ grad_1d

            # 2. Ограничение итогового шага (Trust Region)
            # Если шаг слишком большой, пропорционально уменьшаем его длину
            step_norm = torch.norm(step)
            if step_norm > self.step_clip:
                step = step * (self.step_clip / step_norm)

            updates[name] = -step.view(p.size())

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in updates:
                    p.add_(updates[name])

        return loss.item(), condition_numbers