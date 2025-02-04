import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden, bias=True))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        shape = x.shape[:-1]
        # x = self.layers(x.view(-1, x.shape[-1]))
        x = self.layers(x.reshape(-1, x.shape[-1]))
        # x = F.sigmoid(x)
        return x.reshape(*shape, -1).permute(0, 3, 1, 2)
