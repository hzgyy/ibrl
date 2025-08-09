import numpy as np
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import common_utils
from common_utils import ibrl_utils as utils
from rl.q_agent import QAgent,QAgentConfig
from rl.critic import Critic, CriticConfig

# class of rft algorithm
class dgnAgent(QAgent):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera: list[str], cfg: QAgentConfig):
        super().__init__(use_state, obs_shape, prop_shape, action_dim, rl_camera, cfg)
        self.action_dim = action_dim
        covnet_hidden_size = 128
        # out_dim = int(action_dim*(action_dim+1)/2)
        out_dim = action_dim
        self.covnet = nn.Sequential(
            nn.Linear(obs_shape[0],covnet_hidden_size),
            nn.ReLU(),
            nn.Linear(covnet_hidden_size,covnet_hidden_size),
            nn.ReLU(),
            nn.Linear(covnet_hidden_size,out_dim),
            nn.Softplus()
        ).to("cuda")
        self.covnet_opt = torch.optim.AdamW(self.covnet.parameters(),lr=3e-4)

    def _get_dgn_dist(self,obs:torch.Tensor,mu:torch.Tensor,eval_mode:bool):
        if obs.dim == 1:
            obs = obs.unsqueeze(0)
        if mu.dim == 1:
            mu = mu.unsqueeze(0)
        assert len(obs.shape) == 2,f'obs dim wrong:{obs.shape}'
        assert len(mu.shape) == 2,f'mu dim wrong:{obs.shape}'
        batch_size = obs.shape[0]
        if eval_mode:
            with torch.no_grad():
                cov_diag = self.covnet(obs)
        else:
            cov_diag = self.covnet(obs)     #shape (batch_size, action_dim)
        #assert cov_diag.dim == 1, "cov net output is not 1 dim"
        covariance_matrix = torch.diag_embed(cov_diag)
        # if cov mat not diag
        # if eval_mode:
        #     with torch.no_grad():
        #         covnet_out = self.covnet(obs)
        # else:
        #     covnet_out = self.covnet(obs)     #shape (batch_size, action_dim)
        # L = torch.zeros(batch_size,self.action_dim,self.action_dim).to("cuda")
        # tril_indices = torch.tril_indices(row=self.action_dim, col=self.action_dim, offset=0)
        # L[:, tril_indices[0], tril_indices[1]] = covnet_out
        # diag_idx = (tril_indices[0] == tril_indices[1])
        # L[:, tril_indices[0][diag_idx], tril_indices[1][diag_idx]] = \
        #     nn.functional.softplus(L[:, tril_indices[0][diag_idx], tril_indices[1][diag_idx]])
        # covariance_matrix = L @ L.transpose(-1,-2)  #shape (B,n,n)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu,covariance_matrix)
        return dist

    def act(
        self, obs: dict[str, torch.Tensor], *, eval_mode=False, stddev=0.0, cpu=True
    ) -> torch.Tensor:
        """This function takes tensor and returns actions in tensor"""
        assert not self.training
        assert not self.actor.training
        unsqueezed = self._maybe_unsqueeze_(obs)

        if not self.use_state:
            assert "feat" not in obs
            obs["feat"] = self._encode(obs, augment=False)
        action = self.get_dgn_action(
            obs=obs,
            eval_mode=eval_mode,
            stddev=stddev,
            clip=None,
            use_target=False,
            eps_greedy=self.cfg.ibrl_eps_greedy
        )

        if unsqueezed:
            action = action.squeeze(0)

        action = action.detach()
        if cpu:
            action = action.cpu()
        return action

    def get_dgn_action(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: Optional[float],
        use_target: bool,
        eps_greedy = 1.0
    ) -> torch.Tensor:
        actor = self.actor_target if use_target else self.actor
        rl_dist = actor.forward(obs, stddev)
        rl_act = rl_dist.mean.detach()
        assert rl_act.requires_grad == False,"rl_act needs grad"
        if eval_mode:
            assert not self.training
            return rl_act
        dgn_dist = self._get_dgn_dist(obs["state"],rl_act,eval_mode=True)
        act = dgn_dist.sample()     #shape (batch_size,act_dim)
        if act.shape[0] == 1:
            act = act[0]
        return act

    def update_dgn_net(self,off_batch):
        obs: dict[str, torch.Tensor] = off_batch.obs
        action: torch.Tensor = off_batch.action["action"]
        if isinstance(self.critic_target, Critic):
            pass
        else:
            with torch.no_grad():
                rl_dist = self.actor(obs,0.01)
                rl_act = rl_dist.mean   #shape (batch_size,action_dim)
                assert rl_act.shape == action.shape,f'rl_act shape wrong'
            dgn_dist = self._get_dgn_dist(obs["state"],rl_act,eval_mode=False)   #shape (B,dist)
        loss = -dgn_dist.log_prob(action).sum(-1).mean()
        self.covnet_opt.zero_grad()
        loss.backward()
        self.covnet_opt.step()
        return loss.detach().item()
            

