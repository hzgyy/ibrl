import numpy as np
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import common_utils
from common_utils import ibrl_utils as utils
from rl.q_agent import QAgent,QAgentConfig
from rl.critic import Critic, CriticConfig

class testAgent(QAgent):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera: str, cfg: QAgentConfig):
        super().__init__(use_state, obs_shape, prop_shape, action_dim, rl_camera, cfg)
    
    def compute_critic_loss(self, batch, stddev: float) -> Tuple[torch.Tensor, dict]:
        obs: dict[str, torch.Tensor] = batch.obs
        reply: dict[str, torch.Tensor] = batch.action
        reward: torch.Tensor = batch.reward
        discount: torch.Tensor = batch.bootstrap
        next_obs: dict[str, torch.Tensor] = batch.next_obs
        with torch.no_grad():
            next_action = self.get_action(
                    obs=next_obs,
                    eval_mode=False,
                    stddev=stddev,
                    clip=self.cfg.stddev_clip,
                    use_target=True,
                )
        if isinstance(self.critic_target, Critic):
            target_q1, target_q2 = self.critic_target.forward(
                next_obs["feat"], next_obs["prop"], next_action
            )
            target_q = torch.min(target_q1, target_q2)
        else:
            target_qs = self.critic_target.forward(next_obs["state"], next_action)
            target_q = target_qs.min(-1)[0]
        next_qs_var = torch.sqrt(target_qs.var(dim=-1,unbiased=False))
        target_q = (reward + (discount * target_q)).detach()

        action = reply["action"]
        loss_fn = nn.functional.mse_loss
        if isinstance(self.critic, Critic):
            q1, q2 = self.critic.forward(obs["feat"], obs["prop"], action)
            critic_loss = loss_fn(q1, target_q) + loss_fn(q2, target_q)
        else:
            qs: torch.Tensor = self.critic(obs["state"], action)
            critic_loss = nn.functional.mse_loss(
                qs, target_q.unsqueeze(1).repeat(1, qs.size(1)), reduction="none"
            )
            #weights = torch.nn.functional.softmax(-next_qs_var,dim=0).unsqueeze(-1)
            weights = (torch.sigmoid(-next_qs_var*0.1)+0.5).unsqueeze(-1).detach()
            critic_loss = (critic_loss*weights).sum(1).mean(0)
        
        metrics = {}
        metrics["train/critic_qt"] = target_q.mean().item()
        metrics["train/critic_loss"] = critic_loss.item()

        return critic_loss,metrics