import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import FlexibleNN, get_dataset
from optimizer import EROptimizer


def train_network(dataset_name, optim_name, layer_configs, epochs, batch_size, use_ema, use_batching, loss_name):
    X, y = get_dataset(dataset_name)

    if loss_name == 'Cross-Entropy':
        output_dim = 2
        loss_fn = nn.CrossEntropyLoss()
        y_target = y
    elif loss_name == 'MSE':
        output_dim = 1
        loss_fn = nn.MSELoss()
        y_target = y.float().view(-1, 1)
    elif loss_name == 'Log Loss':
        output_dim = 1
        loss_fn = nn.BCEWithLogitsLoss()
        y_target = y.float().view(-1, 1)
    else:
        output_dim = 2
        loss_fn = nn.CrossEntropyLoss()
        y_target = y

    model = FlexibleNN(input_dim=X.shape[1], output_dim=output_dim, layer_configs=layer_configs)

    if use_batching:
        dataset = TensorDataset(X, y_target)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        loader = [(X, y_target)]

    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    elif optim_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        start_damping = 0.05 if use_batching else 1e-4
        optimizer = EROptimizer(model, h=1.0, init_damping=start_damping,
                                step_clip=0.5, use_ema=use_ema, ema_beta=0.9)

    history = {'loss': [], 'time': [], 'cond': {}}
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_conds = {}
        batches = 0

        for bx, by in loader:
            if optim_name in ['Adam', 'SGD']:
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
            if k not in history['cond']:
                history['cond'][k] = []
            history['cond'][k].append(sum(v) / len(v))

    history['final_loss'] = history['loss'][-1]
    history['total_time'] = history['time'][-1]

    return history