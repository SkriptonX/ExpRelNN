import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from model import FlexibleNN, get_dataset
from optimizer import EROptimizer


def get_memory_footprint_mb(model, optimizer):
    mem_bytes = 0
    for p in model.parameters():
        mem_bytes += p.nelement() * p.element_size()
        if p.grad is not None:
            mem_bytes += p.grad.nelement() * p.grad.element_size()

    if hasattr(optimizer, 'state'):
        for state_dict in optimizer.state.values():
            if isinstance(state_dict, dict):
                for v in state_dict.values():
                    if torch.is_tensor(v):
                        mem_bytes += v.nelement() * v.element_size()

        if 'hessian_ema' in optimizer.state:
            for v in optimizer.state['hessian_ema'].values():
                if v.layout == torch.sparse_csr:
                    mem_bytes += v.values().nelement() * v.values().element_size()
                    mem_bytes += v.crow_indices().nelement() * v.crow_indices().element_size()
                    mem_bytes += v.col_indices().nelement() * v.col_indices().element_size()
                else:
                    mem_bytes += v.nelement() * v.element_size()

    return mem_bytes / (1024 * 1024)


class PINN_AllenCahn_Loss:
    def __init__(self, model, X_batch):
        self.model = model
        self.X_batch = X_batch
        if not self.X_batch.requires_grad:
            self.X_batch.requires_grad_(True)

    def __call__(self, pred, y_dummy):
        du_dX_tuple = torch.autograd.grad(
            pred, self.X_batch,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )

        if du_dX_tuple[0] is None:
            du_dX = torch.zeros_like(self.X_batch)
        else:
            du_dX = du_dX_tuple[0]

        du_dx = du_dX[:, 0:1]
        du_dt = du_dX[:, 1:2]

        if du_dx.requires_grad:
            d2u_dx2_tuple = torch.autograd.grad(
                du_dx, self.X_batch,
                grad_outputs=torch.ones_like(du_dx),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
            if d2u_dx2_tuple[0] is None:
                d2u_dx2 = torch.zeros_like(du_dx)
            else:
                d2u_dx2 = d2u_dx2_tuple[0][:, 0:1]
        else:
            d2u_dx2 = torch.zeros_like(du_dx)

        residual = du_dt - 0.0001 * d2u_dx2 + 5 * (pred ** 3) - 5 * pred

        return torch.mean(residual ** 2)

def check_spectral_cost_benefit(model, loss_fn, X, y):
    loss = loss_fn(model(X), y)
    model.zero_grad()
    loss.backward(create_graph=True)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target_layer = None
    max_size = 0
    for name, p in model.named_parameters():
        if p.grad is not None and p.numel() > max_size and p.numel() <= 400:
            max_size = p.numel()
            target_layer = p

    if target_layer is None:
        return True

    grad_1d = target_layer.grad.view(-1)
    n = grad_1d.size(0)
    H = []
    for i in range(n):
        g2 = torch.autograd.grad(grad_1d[i], target_layer, retain_graph=True)[0]
        H.append(g2.view(-1))
    H = torch.stack(H).detach()
    H = 0.5 * (H + H.T)

    L = torch.linalg.eigvalsh(H)
    lam_max = torch.max(torch.abs(L)).item()
    lam_min = torch.min(torch.abs(L)).item()
    kappa = lam_max / (lam_min + 1e-12)

    cost_adam = kappa
    cost_er = 15 * total_params
    return cost_adam > cost_er * 2.0


def train_network(dataset_name, optim_name, layer_configs, epochs, batch_size,
                  use_ema, use_batching, loss_name, er_method='Spectral', k_lanczos=10,
                  switch_method='Стагнация', switch_epoch=10, seed=42, use_compression=False,
                  chebyshev_k=15, hybrid_base='Adam'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_samples = 4000 if 'SIREN' in dataset_name or 'PINN' in dataset_name else 1000
    X, y = get_dataset(dataset_name, n_samples=n_samples)

    if 'SIREN' in dataset_name:
        output_dim = 1
        loss_fn = nn.MSELoss()
        y_target = y.float()
    elif 'PINN' in dataset_name:
        output_dim = 1
        y_target = y.float()
    elif loss_name == 'Cross-Entropy':
        output_dim = 2
        loss_fn = nn.CrossEntropyLoss()
        y_target = y
    elif loss_name == 'MSE':
        output_dim = 1
        loss_fn = nn.MSELoss()
        y_target = y.float().view(-1, 1)
    else:
        output_dim = 1
        loss_fn = nn.BCEWithLogitsLoss()
        y_target = y.float().view(-1, 1)

    model = FlexibleNN(input_dim=X.shape[1], output_dim=output_dim, layer_configs=layer_configs)

    if use_batching:
        g = torch.Generator()
        g.manual_seed(seed)
        dataset = TensorDataset(X, y_target)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    else:
        loader = [(X, y_target)]

    is_hybrid = (optim_name == 'Hybrid')
    active_optim_name = hybrid_base if is_hybrid else optim_name
    switched_to_er = False
    switch_epoch_record = -1

    def init_optimizer(name):
        if name == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=0.005)
        elif name == 'SGD':
            return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        elif name == 'RMSProp':
            return torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
        elif name == 'L-BFGS':
            return torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                     history_size=50, line_search_fn="strong_wolfe")
        else:
            start_damping = 0.05 if use_batching else 1e-4
            return EROptimizer(model, er_method=er_method, h=1.0,
                               init_damping=start_damping, step_clip=1.0,
                               use_ema=use_ema, ema_beta=0.9, k_lanczos=k_lanczos,
                               use_compression=use_compression, chebyshev_k=chebyshev_k)

    optimizer = init_optimizer(active_optim_name)

    history = {'loss': [], 'time': [], 'cond': {}, 'weight_cond': {}}
    start_time = time.time()

    for epoch in range(epochs):
        if is_hybrid and not switched_to_er:
            do_switch = False
            if switch_method == 'Фиксированная эпоха' and epoch >= switch_epoch:
                do_switch = True
            elif switch_method == 'Стагнация' and epoch > 10:
                recent_losses = history['loss'][-5:]
                if max(recent_losses) - min(recent_losses) < 1e-5:
                    do_switch = True

            if do_switch:
                active_optim_name = 'ER'
                optimizer = init_optimizer('ER')
                switched_to_er = True
                switch_epoch_record = epoch
                model.zero_grad()

        epoch_loss = 0.0
        epoch_conds = {}
        epoch_weight_conds = {}
        batches = 0

        for bx, by in loader:
            if 'PINN' in dataset_name:
                current_loss_fn = PINN_AllenCahn_Loss(model, bx)
            else:
                current_loss_fn = loss_fn

            if active_optim_name in ['Adam', 'SGD', 'RMSProp']:
                model.zero_grad()
                out = model(bx)
                loss = current_loss_fn(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                loss_val = loss.item()
                conds = {}

            elif active_optim_name == 'L-BFGS':
                def closure():
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = current_loss_fn(out, by)
                    loss.backward()
                    return loss

                # Делаем шаг и получаем финальный loss после итераций L-BFGS
                loss_val = optimizer.step(closure).item()
                conds = {}

            else:
                loss_val, conds = optimizer.step(current_loss_fn, bx, by)

            epoch_loss += loss_val
            batches += 1

            for k, v in conds.items():
                if k not in epoch_conds: epoch_conds[k] = []
                epoch_conds[k].append(v)

            with torch.no_grad():
                for name, p in model.named_parameters():
                    if 'weight' in name and p.dim() == 2:
                        s = torch.linalg.svdvals(p)
                        s_max = torch.max(s).item()
                        s_min = torch.min(s).item()
                        kappa_w = s_max / (s_min + 1e-12)

                        if name not in epoch_weight_conds:
                            epoch_weight_conds[name] = []
                        epoch_weight_conds[name].append(kappa_w)

        history['loss'].append(epoch_loss / batches)
        history['time'].append(time.time() - start_time)

        for k, v in epoch_conds.items():
            if k not in history['cond']: history['cond'][k] = []
            history['cond'][k].append(sum(v) / len(v))

        for k, v in epoch_weight_conds.items():
            if k not in history['weight_cond']: history['weight_cond'][k] = []
            history['weight_cond'][k].append(sum(v) / len(v))

    history['final_loss'] = history['loss'][-1]
    history['total_time'] = history['time'][-1]
    history['switch_epoch'] = switch_epoch_record
    history['seed'] = seed

    history['memory_mb'] = get_memory_footprint_mb(model, optimizer)

    return history