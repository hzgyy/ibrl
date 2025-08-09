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
class rftAgent(QAgent):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera: list[str], cfg: QAgentConfig):
        super().__init__(use_state, obs_shape, prop_shape, action_dim, rl_camera, cfg)
    
    def update_actor(
        self,
        obs: dict[str, torch.Tensor],
        stddev: float,
        bc_batch,
        ref_agent: "QAgent",
    ):
        metrics = {}
        actor_loss = self.compute_actor_loss(obs, stddev)
        metrics["train/actor_loss"] = actor_loss.item()
        bc_loss = self.compute_actor_bc_loss(bc_batch, backprop_encoder=False)
        assert actor_loss.size() == bc_loss.size()

        ratio = 1
        if self.cfg.bc_loss_dynamic:
            with torch.no_grad(), utils.eval_mode(self, ref_agent):
                assert ref_agent.cfg.act_method == "rl"

                # temporarily change to rl since we want to regularize actor not hybrid
                act_method = self.cfg.act_method
                self.cfg.act_method = "rl"

                ref_bc_obs = bc_batch.obs.copy()  # shallow copy
                ref_action = ref_agent.act(ref_bc_obs, eval_mode=True, cpu=False)

                # we first get the ref_action and then pop the feature
                # then we get the curr_action so that the obs["feat"] is the current feature
                # which can be used for computing q-values
                bc_obs = bc_batch.obs
                curr_action = self.act(bc_obs, eval_mode=True, cpu=False)

                if isinstance(self.critic, Critic):
                    curr_q = torch.min(*self.critic(bc_obs["feat"], bc_obs["prop"], curr_action))
                    ref_q = torch.min(*self.critic(bc_obs["feat"], bc_obs["prop"], ref_action))
                else:
                    curr_q = self.critic.forward_k(bc_obs["state"], curr_action).min(-1)[0]
                    ref_q = self.critic.forward_k(bc_obs["state"], ref_action).min(-1)[0]

                ratio = (ref_q > curr_q).float().mean().item()

                # recover to original act_method
                self.cfg.act_method = act_method

        loss = actor_loss + (self.cfg.bc_loss_coef * ratio * bc_loss).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()

        metrics["rft/bc_loss"] = bc_loss.mean().item()
        metrics["rft/ratio"] = ratio
        return metrics