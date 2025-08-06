import numpy as np
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import common_utils
from common_utils import ibrl_utils as utils
from rl.q_agent import QAgent,QAgentConfig
from rl.critic import Critic, CriticConfig, Discriminator,MultiFcQ

class dacAgent(QAgent):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera: str, cfg: QAgentConfig):
        super().__init__(use_state, obs_shape, prop_shape, action_dim, rl_camera, cfg)
        # get preperation data
        repr_dim = self.encoder.repr_dim
        patch_repr_dim = self.encoder.patch_repr_dim
        assert len(prop_shape) == 1
        prop_dim = prop_shape[0] if cfg.use_prop else 0

        # create discriminator
        if use_state:
            self.discriminator = MultiFcQ(obs_shape, action_dim, cfg.state_critic)
        else:
            self.discriminator = Discriminator(
                repr_dim=repr_dim,
                patch_repr_dim=patch_repr_dim,
                prop_dim=prop_dim,
                action_dim=action_dim,
                cfg=self.cfg.critic,
            )
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.cfg.lr)

    def update_discriminator(self,batch,expert_batch):
        # decomposite batch data
        obs: dict[str, torch.Tensor] = batch.obs
        action: dict[str, torch.Tensor] = batch.action
        reward: torch.Tensor = batch.reward
        discount: torch.Tensor = batch.bootstrap
        next_obs: dict[str, torch.Tensor] = batch.next_obs

        expert_obs: dict[str, torch.Tensor] = expert_batch.obs
        expert_action: dict[str, torch.Tensor] = expert_batch.action
        expert_reward: torch.Tensor = expert_batch.reward
        expert_discount: torch.Tensor = expert_batch.bootstrap
        expert_next_obs: dict[str, torch.Tensor] = expert_batch.next_obs
        
        #encode obs
        if not self.use_state:
            obs["feat"] = self._encode(obs, augment=True)
            expert_obs["feat"] = self._encode(expert_obs, augment=True)
            with torch.no_grad():
                next_obs["feat"] = self._encode(next_obs, augment=True)
                expert_next_obs["feat"] = self._encode(expert_next_obs, augment=True)

        #cal batch size
        batch_size = reward.shape[0]
        if isinstance(self.discriminator, Discriminator):
            output = self.discriminator.forward(obs["feat"],obs["prop"],action["action"])
            expert_output = self.discriminator.forward(expert_obs["feat"],expert_obs["prop"],expert_action["action"])
        else:
            output = self.discriminator.forward(obs["state"],action["action"])
            expert_output = self.discriminator.forward(expert_obs["state"],expert_action["action"])

        expert_labels = torch.ones_like(expert_output)
        policy_labels = torch.zeros_like(output)

        # Binary cross-entropy with logits
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_output, expert_labels, reduction='mean'
        )
        policy_loss = F.binary_cross_entropy_with_logits(
            output, policy_labels, reduction='mean'
        )
        gan_loss = expert_loss+policy_loss

        # calculate gradient penalty
        alpha = np.random.uniform(size=(batch_size,1))
        if isinstance(self.discriminator, Discriminator):
            inter_feat = alpha*obs["feat"] + (1-alpha)*expert_obs["feat"]
            inter_prop = alpha*obs["prop"] + (1-alpha)*expert_obs["prop"]
            inter_action = alpha*action["action"] + (1-alpha)*expert_action["action"]
            inter_feat.requires_grad_(True)
            inter_prop.requires_grad_(True)
            inter_action.requires_grad_(True)
            inter_output = self.discriminator(inter_feat,inter_prop,inter_action)
            gradients = torch.autograd.grad(
                outputs=inter_output,
                inputs=[inter_feat,inter_prop,inter_action],
                grad_outputs=torch.ones_like(output),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )
        else:
            inter_state = alpha*obs["state"]+(1-alpha)*expert_obs["state"]
            inter_action = alpha*action["action"] + (1-alpha)*expert_action["action"]
            inter_state.requires_grad_(True)
            inter_action.requires_grad_(True)
            inter_output = self.discriminator(inter_state,inter_action)
            gradients = torch.autograd.grad(
                outputs=inter_output,
                inputs=[inter_state,inter_action],
                grad_outputs=torch.ones_like(output),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )
        grad_norm = torch.cat([g.view(g.size(0), -1) for g in gradients], dim=1)  # shape: [B, D_total]
        grad_norm = grad_norm.norm(2, dim=1)  # L2 norm per sample
        gp_loss = ((grad_norm - 1) ** 2).mean()
        loss = gan_loss + 10.0 * gp_loss
        self.discriminator_opt.zero_grad()
        loss.backward()
        self.discriminator.step()
        return loss.detach().cpu().item()
    
    def get_reward(self,obs,action):
        with torch.no_grad():
            d = self.discriminator(obs,action)
        return -torch.log(1-torch.sigmoid(d)+1e-8)


def wrap_for_absorbing_state():
