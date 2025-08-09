import numpy as np
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import common_utils
from common_utils import ibrl_utils as utils
from rl.q_agent import QAgent,QAgentConfig
from rl.critic import Critic, CriticConfig

class awacAgent(QAgent):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera: list[str], cfg: QAgentConfig):
        super().__init__(use_state, obs_shape, prop_shape, action_dim, rl_camera, cfg)

    def compute_actor_loss(self, batch, stddev: float):
        obs = batch.obs
        old_actions = batch.action["action"]
        if not self.use_state:
            assert "feat" in obs, "safety check"
        #old_actions = action["action"]
        action: torch.Tensor = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=stddev,
            clip=self.cfg.stddev_clip,
            use_target=False,
        )
        action = action.detach()
        if isinstance(self.critic, Critic):
            v_pi = torch.min(*self.critic.forward(obs["feat"], obs["prop"], action))
        else:
            v_pi = self.critic.forward(obs["state"], action).min(-1)[0]
        if isinstance(self.critic, Critic):
            q_old_actions = torch.min(*self.critic.forward(obs["feat"], obs["prop"], old_actions))
        else:
            q_old_actions = self.critic.forward(obs["state"], old_actions).min(-1)[0]
        adv_pi = q_old_actions - v_pi
        beta = self.cfg.awac_beta
        weights = torch.nn.functional.softmax(adv_pi/beta,dim=0)
        log_p = self.actor.logp(obs,old_actions,stddev)
        actor_loss = (-log_p * len(weights)*weights.detach()).mean()
        return actor_loss

