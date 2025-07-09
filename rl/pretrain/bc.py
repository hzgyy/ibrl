import numpy as np
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import common_utils
from common_utils import ibrl_utils as utils
from rl.q_agent import QAgent,QAgentConfig
from rl.critic import Critic, CriticConfig

class bcAgent(QAgent):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera: str, cfg: QAgentConfig):
        self.backprop_encoder = True
        super().__init__(use_state, obs_shape, prop_shape, action_dim, rl_camera, cfg)
    
    def update_critic(self, batch, stddev: float):
        return {}

    def compute_actor_loss(self, batch, stddev: float):
        obs: dict[str, torch.Tensor] = batch.obs

        if not self.use_state:
            assert "feat" not in obs, "safety check"
            obs["feat"] = self._encode(obs, augment=True)

        if not self.backprop_encoder and not self.use_state:
            obs["feat"] = obs["feat"].detach()

        pred_action = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=0,
            clip=None,
            use_target=False,
        )
        action: torch.Tensor = batch.action["action"]
        loss = nn.functional.mse_loss(pred_action, action, reduction="none")
        loss = loss.sum(1).mean(0)
        return loss
    
    def update_actor(self, batch, stddev: float, bc_batch, ref_agent: QAgent):
        """pretrain actor and encoder with bc"""
        loss = self.compute_actor_loss(batch, stddev)

        if not self.use_state:
            self.encoder_opt.zero_grad(set_to_none=True)

        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()

        if not self.use_state:
            self.encoder_opt.step()
        self.actor_opt.step()

        return {"pretrain/loss": loss.item()}