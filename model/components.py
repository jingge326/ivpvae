import torch
import torch.nn as nn

import utils


class Embedding_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim * 2, 200),
            nn.ReLU(),
            nn.Linear(200, out_dim)
        )
        utils.init_network_weights(self.layers, method='kaiming_uniform_')

    def forward(self, truth, mask):
        x = torch.cat((truth, mask), -1)
        assert (not torch.isnan(x).any())
        out = self.layers(x)
        return out


class Reconst_Mapper_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 200),
            nn.ReLU(),
            nn.Linear(200, out_dim)
        )
        utils.init_network_weights(self.layers, method='kaiming_uniform_')

    def forward(self, data):
        truth = self.layers(data)
        return truth


class Z_to_mu_ReLU(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),)
        utils.init_network_weights(self.net, method='kaiming_uniform_')

    def forward(self, data):
        return self.net(data)


class Z_to_std_ReLU(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim),
            nn.Softplus(),)
        utils.init_network_weights(self.net)

    def forward(self, data):
        return self.net(data)


class BinaryClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )
        utils.init_network_weights(self.layers, method='kaiming_uniform_')

    def forward(self, x):
        return self.layers(x)
