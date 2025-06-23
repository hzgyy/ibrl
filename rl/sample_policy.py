from dataclasses import dataclass
import torch
import torch.nn as nn

from common_utils import ibrl_utils as utils

def build_fc(in_dim, hidden_dim, action_dim, num_layer, layer_norm, dropout):
    dims = [in_dim]
    dims.extend([hidden_dim for _ in range(num_layer)])

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if layer_norm == 2 and (i == num_layer - 1):
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(dims[-1], action_dim))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)

@dataclass
class DGNConfig:
    num_layer: int = 2
    hidden_dim: int = 128
    dropout: float = 0.5
    layer_norm: int = 0
    lr: int = 0.001
    weight_decay: int = 3e-2
    update_epochs: int = 2

class DGN():
    def __init__(self,obs_dim:int, action_dim:int,cfg:DGNConfig):
        self.cfg = cfg
        self.policy = sample_policy(obs_dim,action_dim,cfg).to("cuda")
        self.optimizer = torch.optim.AdamW(self.policy.parameters(),lr = cfg.lr)
    
    def get_exploration(self,obs: dict[str, torch.Tensor],mu):
        cov = self.policy.forward(obs)
        dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
        return dist.sample()
    
    def update(self,obs,action,mu):
        if mu.requires_grad:
            mu = mu.detach()
        total_loss = 0
        for i in range(self.cfg.update_epochs):
            cov = self.policy.forward(obs)
            dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
            self.optimizer.zero_grad()
            loss = -dist.log_prob(action).mean()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach().item()
        return total_loss/5


class sample_policy(nn.Module):
    def __init__(self,obs_dim:int, action_dim:int,cfg: DGNConfig):
        super().__init__()
        self.cfg = cfg
        self.net = build_fc(obs_dim, cfg.hidden_dim, action_dim, cfg.num_layer, cfg.layer_norm, cfg.dropout)
        
    def forward(self,obs: dict[str, torch.Tensor]):
        vec = self.net(obs["state"])
        return torch.mul(vec.T,vec)