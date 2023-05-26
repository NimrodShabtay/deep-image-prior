import torch.nn as nn
import torch.nn.functional as F
from models.common import GaussianActivation


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, act='relu'):
        super().__init__()
        layers = []
        lastv = in_dim
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'gauss':
            self.act = GaussianActivation(a=1., trainable=False)
        last_layer = len(hidden_list) - 1
        for layer_idx, hidden in enumerate(hidden_list):
            layers.append(nn.Linear(lastv, hidden, bias=True))
            layers.append(self.act)
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        # layers.append(self.act)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        shape = x.shape[:-1]
        # x = self.layers(x.view(-1, x.shape[-1]))
        x = self.layers(x.reshape(-1, x.shape[-1]))
        x = F.sigmoid(x)
        return x.reshape(*shape, -1).permute(0, 3, 1, 2)


# class MLP(nn.Module):
#     def __init__(self, n_in,
#                  n_layers=4, n_hidden_units=256,
#                  act='relu', act_trainable=False,
#                  **kwargs):
#         super().__init__()
#
#         layers = []
#         for i in range(n_layers):
#
#             if i == 0:
#                 l = nn.Linear(n_in, n_hidden_units)
#             elif 0 < i < n_layers-1:
#                 l = nn.Linear(n_hidden_units, n_hidden_units)
#
#             if act == 'relu':
#                 act_ = nn.ReLU(True)
#             elif act == 'gauss':
#                 act_ = GaussianActivation(a=1.0, trainable=act_trainable)
#
#             if i < n_layers-1:
#                 layers += [l, act_]
#             else:
#                 layers += [nn.Linear(n_hidden_units, 3), nn.Sigmoid()]
#
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, x):
#         """
#         x: (B, 2) # pixel uv (normalized)
#         """
#         return self.net(x) # (B, 3) rgb