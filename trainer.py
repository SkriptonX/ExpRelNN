import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import FlexibleNN, get_dataset
from optimizer import EROptimizer


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
                  switch_method='Стагнация', switch_epoch=10):
    X, y = get_dataset(dataset_name)

    if loss_name == 'Cross-Entropy':
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
        loader = DataLoader(TensorDataset(X, y_target), batch_size=batch_size, shuffle=True)
    else:
        loader = [(X, y_target)]

    is_hybrid = (optim_name == 'Hybrid')
    active_optim_name = 'Adam' if is_hybrid else optim_name
    switched_to_er = False
    switch_epoch_record = -1

    if active_optim_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    elif active_optim_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        start_damping = 0.05 if use_batching else 1e-4
        optimizer = EROptimizer(model, er_method=er_method, h=1.0,
                                init_damping=start_damping, step_clip=1.0,
                                use_ema=use_ema, ema_beta=0.9, k_lanczos=k_lanczos)

    history = {'loss': [], 'time': [], 'cond': {}}
    start_time = time.time()

    for epoch in range(epochs):
        if is_hybrid and not switched_to_er and epoch >= 5:
            do_switch = False

            if switch_method == 'Фиксированная эпоха':
                if epoch >= switch_epoch:
                    do_switch = True
            else:
                loss_diff = history['loss'][-5] - history['loss'][-1]
                if loss_diff < 1e-3:
                    if switch_method == 'Стагнация (Stagnation)':
                        do_switch = True
                    elif switch_method == 'Спектральный (Cost-Benefit)':
                        bx, by = next(iter(loader))
                        do_switch = check_spectral_cost_benefit(model, loss_fn, bx, by)

            if do_switch:
                print(f"Переключение на ER на эпохе {epoch} (Метод: {switch_method})")
                start_damping = 0.05 if use_batching else 1e-4
                optimizer = EROptimizer(model, er_method=er_method, h=1.0,
                                        init_damping=start_damping, step_clip=1.0,
                                        use_ema=use_ema, ema_beta=0.9, k_lanczos=k_lanczos)
                active_optim_name = 'ER'
                switched_to_er = True
                switch_epoch_record = epoch

        epoch_loss = 0.0
        epoch_conds = {}
        batches = 0

        for bx, by in loader:
            if active_optim_name in ['Adam', 'SGD']:
                model.zero_grad()
                out = model(bx)
                loss = loss_fn(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                loss_val = loss.item()
                conds = {}
            else:
                loss_val, conds = optimizer.step(loss_fn, bx, by)

            epoch_loss += loss_val
            batches += 1

            for k, v in conds.items():
                if k not in epoch_conds:
                    epoch_conds[k] = []
                epoch_conds[k].append(v)

        history['loss'].append(epoch_loss / batches)
        history['time'].append(time.time() - start_time)

        for k, v in epoch_conds.items():
            if k not in history['cond']: history['cond'][k] = []
            history['cond'][k].append(sum(v) / len(v))

    history['final_loss'] = history['loss'][-1]
    history['total_time'] = history['time'][-1]
    history['switch_epoch'] = switch_epoch_record

    return history