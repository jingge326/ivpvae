import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MLPFlow(nn.Module):
    
    def __init__(
        self,
        z_dims=20,
        device='cuda',
        **kwargs
    ):
        super().__init__()

        self.z_dims = z_dims
        
        self.ivp1 = nn.Linear(self.z_dims + 1, 128)
        
        self.ivp2 = nn.Linear(128, self.z_dims)

        self.fc_z_to_128 = nn.Linear(self.z_dims, 128)
        
        self.fc_128_to_64 = nn.Linear(128, 64)
        
        self.fc_64_to_z = nn.Linear(64, self.z_dims)

        
    def forward(self, x, t):
        # t > [batch_size, num_times, 1]
        # x > [batch_size, 1, z_dims]
        # y > [batch_size, num_times, z_dims]
        
        x = x.repeat_interleave(t.shape[-2], dim=-2)
        
        batch_size, num_times, _ = x.size()
        
        x = x.view((batch_size * num_times, self.z_dims))
        
        t = t.view((batch_size * num_times, 1))
        
        x = torch.cat([x, t], -1)

        x = self.ivp1(x)

        x = F.relu(x)
        
        x = self.ivp2(x)
        
        x = self.fc_z_to_128(x)
        
        x = F.relu(x)
        
        x = self.fc_128_to_64(x)
        
        x = F.relu(x)
        
        x = self.fc_64_to_z(x)
        
        x = x.view((batch_size, num_times, self.z_dims))
        
        return x
    