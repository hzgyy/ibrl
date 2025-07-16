import numpy as np
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import common_utils
from common_utils import ibrl_utils as utils
from rl.q_agent import QAgent,QAgentConfig
from rl.critic import Critic, CriticConfig

class sacAgent(QAgent):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera: str, cfg: QAgentConfig):
        super().__init__(use_state, obs_shape, prop_shape, action_dim, rl_camera, cfg)
    
    def compute_critic_loss(self, batch, stddev: float) -> Tuple[torch.Tensor, dict]:
        obs: dict[str, torch.Tensor] = batch.obs
        reply: dict[str, torch.Tensor] = batch.action
        reward: torch.Tensor = batch.reward
        discount: torch.Tensor = batch.bootstrap
        next_obs: dict[str, torch.Tensor] = batch.next_obs
        with torch.no_grad():
            next_act_dist = self.actor.forward(next_obs,stddev)
            next_action = next_act_dist.sample(0.999)
            next_logprob = next_act_dist.log_prob(next_action).sum(-1)
        if isinstance(self.critic_target, Critic):
            target_q1, target_q2 = self.critic_target.forward(
                next_obs["feat"], next_obs["prop"], next_action
            )
            target_q = torch.min(target_q1, target_q2)
        else:
            target_q = self.critic_target.forward(next_obs["state"], next_action).min(-1)[0]
        
        # add compute log_prob
        metrics = {}
        metrics["train/next_q"] = target_q.detach().mean().item()
        metrics["train/next_logprob"] = next_logprob.detach().mean().item()
        target_q = (reward + (discount * (target_q-self.cfg.sac_alpha*next_logprob))).detach()

        action = reply["action"]
        loss_fn = nn.functional.mse_loss
        if isinstance(self.critic, Critic):
            q1, q2 = self.critic.forward(obs["feat"], obs["prop"], action)
            critic_loss = loss_fn(q1, target_q) + loss_fn(q2, target_q)
        else:
            # change to forward k， in order to update q nets on different batch
            #qs: torch.Tensor = self.critic.forward(obs["state"], action)
            qs: torch.Tensor = self.critic.forward(obs["state"], action)
            critic_loss = nn.functional.mse_loss(
                qs, target_q.unsqueeze(1).repeat(1, qs.size(1)), reduction="none"
            )
            critic_loss = critic_loss.sum(1).mean(0)
        
        
        metrics["train/critic_qt"] = target_q.mean().item()
        metrics["train/critic_loss"] = critic_loss.item()

        return critic_loss,metrics
    
    def compute_actor_loss(self, batch, stddev: float):
        obs = batch.obs
        if not self.use_state:
            assert "feat" in obs, "safety check"

        # action: torch.Tensor = self._act_default(
        #     obs=obs,
        #     eval_mode=False,
        #     stddev=stddev,
        #     clip=self.cfg.stddev_clip,
        #     use_target=False,
        # )
        action_dist = self.actor.forward(obs,stddev)
        action = action_dist.sample(0.999)
        action_logprob = action_dist.log_prob(action).sum(-1)
        if isinstance(self.critic, Critic):
            q = torch.min(*self.critic.forward(obs["feat"], obs["prop"], action))
        else:
            q: torch.Tensor = self.critic(obs["state"], action).min(-1)[0]
        actor_loss = -(q-self.cfg.sac_alpha*action_logprob).mean()
        metrics = {}
        metrics["train/actor_loss"] = actor_loss.detach().item()
        metrics["train/action_logprob"] = action_logprob.detach().mean().item()
        return actor_loss,metrics
    
    def update_actor(
        self, 
        #obs: dict[str, torch.Tensor], 
        batch,
        stddev: float,
        bc_batch,
        ref_agent: "QAgent",
    ):
        # metrics = {}
        actor_loss,metrics = self.compute_actor_loss(batch, stddev)
        #metrics["train/actor_loss"] = actor_loss.detach().item()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        return metrics