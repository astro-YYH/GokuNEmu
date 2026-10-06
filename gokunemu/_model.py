import torch.nn as nn

_DEFAULT_ACTIVATION = object()


class SimpleNN(nn.Module):
    """Build the dense neural-network architecture shared by training and inference."""

    def __init__(
        self,
        num_layers,
        hidden_size,
        dim_x=1,
        dim_y=1,
        activation=_DEFAULT_ACTIVATION,
    ):
        super().__init__()
        if activation is _DEFAULT_ACTIVATION:
            activation = nn.SiLU()

        layers = [nn.Linear(dim_x, hidden_size)]
        if activation is not None:
            layers.append(activation)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if activation is not None:
                layers.append(activation)
        layers.append(nn.Linear(hidden_size, dim_y))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Evaluate the dense network."""
        return self.network(x)
