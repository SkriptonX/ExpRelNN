import torch
import torch.nn as nn
from sklearn.datasets import make_moons, make_classification
import numpy as np


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class FlexibleNN(nn.Module):
    def __init__(self, input_dim, output_dim, layer_configs):
        super().__init__()
        layers = []
        current_dim = input_dim

        activation_dict = {
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid(),
            'Sine': Sine()
        }

        for config in layer_configs:
            layers.append(nn.Linear(current_dim, config['units']))
            layers.append(activation_dict[config['activation']])
            current_dim = config['units']

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                if isinstance(self.network[1], Sine):
                    num_input = m.weight.size(-1)
                    if m == self.network[0]:
                        nn.init.uniform_(m.weight, -1 / num_input, 1 / num_input)
                    else:
                        nn.init.uniform_(m.weight, -np.sqrt(6 / num_input) / 30.0, np.sqrt(6 / num_input) / 30.0)

    def forward(self, x):
        return self.network(x)


def get_dataset(name, n_samples=1000):
    if name == 'SIREN (Image Fitting)':
        x = np.linspace(-1, 1, int(np.sqrt(n_samples)))
        y = np.linspace(-1, 1, int(np.sqrt(n_samples)))
        X_grid, Y_grid = np.meshgrid(x, y)
        X = np.hstack([X_grid.reshape(-1, 1), Y_grid.reshape(-1, 1)])
        Z = np.sin(10 * X[:, 0]) * np.cos(10 * X[:, 1]) + np.exp(-X[:, 0] ** 2 - X[:, 1] ** 2)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Z, dtype=torch.float32).view(-1, 1)

    elif name == 'PINN (Allen-Cahn)':
        X = np.random.uniform(-1, 1, (n_samples, 2))  # [x, t]
        X[:, 1] = (X[:, 1] + 1) / 2.0  # t in [0, 1]
        y = np.zeros((n_samples, 1))
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    elif name == 'Moons':
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