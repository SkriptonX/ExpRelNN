import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import FlexibleNN, get_dataset
from optimizer import EROptimizer


def train_network(dataset_name, optim_name, layer_configs, epochs, batch_size, use_ema, use_batching):
    X, y = get_dataset(dataset_name)

    model = FlexibleNN(input_dim=X.shape[1], output_dim=2, layer_configs=layer_configs)
    loss_fn = nn.CrossEntropyLoss()

    if use_batching:
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        loader = [(X, y)]

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

    return history