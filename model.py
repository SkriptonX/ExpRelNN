import torch
import torch.nn as nn
from sklearn.datasets import make_moons, make_classification
import numpy as np


class FlexibleNN(nn.Module):
    def __init__(self, input_dim, output_dim, layer_configs):
        super().__init__()
        layers = []
        prev_dim = input_dim

        # Динамическое создание слоев на основе переданных настроек
        for config in layer_configs:
            units = config['units']
            act_name = config['activation']

            layers.append(nn.Linear(prev_dim, units))

            if act_name == 'Tanh':
                act_fn = nn.Tanh()
            elif act_name == 'ReLU':
                act_fn = nn.ReLU()
            else:
                act_fn = nn.Sigmoid()

            layers.append(act_fn)
            prev_dim = units

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def get_dataset(name):
    if name == 'Moons':
        X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
    elif name == 'Synthetic Ravine':
        np.random.seed(42)
        X_raw = np.random.randn(300, 2)
        U, _, Vt = np.linalg.svd(np.random.randn(2, 2))
        S = np.diag([1.0, 1.0 / 2000.0])
        X = X_raw.dot(U.dot(S).dot(Vt))
        logits = X.dot(np.array([2.0, -1.5])) + 0.5
        y = (logits > 0).astype(int)
    else:
        X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, random_state=42)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)