import torch
import torch.nn as nn


class FCN(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, ksize):
        super().__init__()
        layers = nn.ModuleList()
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Sequential(nn.Conv2d(lastv, hidden, kernel_size=(ksize, ksize), padding='same'),
                                        nn.ReLU(), nn.BatchNorm2d(hidden)))
            lastv = hidden

        layers.append(nn.Conv2d(lastv, out_dim, kernel_size=(ksize, ksize), padding='same'))
        layers.append(nn.Sigmoid())
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FCN_skip(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, ksize):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        lastv = in_dim
        skip_dim = 4
        for hidden_ind, hidden in enumerate(hidden_list):
            if hidden_ind < len(hidden_list) // 2:
                self.encoder.append(nn.Sequential(nn.Conv2d(lastv, hidden, kernel_size=(ksize, ksize), padding='same'),
                                            nn.ReLU(), nn.BatchNorm2d(hidden)))
            elif hidden_ind > len(hidden_list) / 2:
                self.decoder.append(nn.Sequential(nn.Conv2d(lastv + skip_dim, hidden,
                                                            kernel_size=(ksize, ksize), padding='same'),
                                                  nn.ReLU(), nn.BatchNorm2d(hidden)))
            else:
                self.bottleneck = nn.Sequential(nn.Conv2d(lastv, hidden, kernel_size=(ksize, ksize), padding='same'),
                                                  nn.ReLU(), nn.BatchNorm2d(hidden))
            lastv = hidden

        self.skip_con = nn.Sequential(nn.Conv2d(hidden, skip_dim, kernel_size=(1, 1),
                                                padding='same'), nn.ReLU(), nn.BatchNorm2d(skip_dim))

        self.reminder = nn.Sequential(nn.Conv2d(lastv, out_dim, kernel_size=(ksize, ksize), padding='same'),
                                      nn.Sigmoid())

    def forward(self, x):
        out = x
        acts = []
        for layer in self.encoder:
            out = layer(out)
            acts.append(out)

        out = self.bottleneck(out)
        for a, layer in zip(acts[::-1], self.decoder):
            out = layer(torch.cat([out, self.skip_con(a)], dim=1))

        out = self.reminder(out)
        return out
