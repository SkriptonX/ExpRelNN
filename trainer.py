import time
import torch
import torch.nn as nn
from model import FlexibleNN, get_dataset
from optimizer import EROptimizer


def train_network(dataset_name, optim_name, layer_configs, epochs):
    X, y = get_dataset(dataset_name)
    model = FlexibleNN(input_dim=X.shape[1], output_dim=2, layer_configs=layer_configs)
    loss_fn = nn.CrossEntropyLoss()

    if optim_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    elif optim_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        # h=1.0, damping=0.05 для стабильности, step_clip=0.5 для предотвращения осцилляций
        optimizer = EROptimizer(model, h=1.0, damping=0.05, step_clip=0.5)

    history = {'loss': [], 'time': [], 'cond': {}}
    start_time = time.time()

    for epoch in range(epochs):
        if optim_name in ['Adam', 'SGD']:
            model.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            optimizer.step()
            loss_val = loss.item()
            conds = {}
        else:
            # ER оптимизатор теперь возвращает loss и числа обусловленности
            loss_val, conds = optimizer.step(loss_fn, X, y)

        history['loss'].append(loss_val)
        history['time'].append(time.time() - start_time)

        for k, v in conds.items():
            if k not in history['cond']:
                history['cond'][k] = []
            history['cond'][k].append(v)

    return history